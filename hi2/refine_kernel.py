# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Fused local-refinement sweep for one block of columns (HIP, gfx10/gfx11).

:func:`hi2.refine.refine_expanded` visits the columns of a block one after another and, for
every row, tries to move the column's code one level up or down. The torch version spends about
20 small launches per column, so the time of a sweep follows the number of columns, not the
amount of arithmetic. Here one launch handles a whole block of up to 128 columns for every row:
one thread per row keeps its ``g`` segment in LDS, decides each column with the arithmetic of the
torch loop (every product and sum rounded separately, ``-ffp-contract=off``, ``__f*_rn``
intrinsics, ``__float2half_rn``) and pushes an accepted move into the later columns of the
block, so codes and moves are bit-identical to the torch loop. The update of the columns outside
the block stays a torch GEMM with the shapes of the torch loop.

Built on first use like :mod:`hi2.gptq_kernel` (into ``$HI2_KERNEL_DIR/refine_v1``), for gfx10/gfx11 HIP
devices only; anything else, or a failed build, uses the torch loop.
``HI2_REFINE_BACKEND=torch`` forces the torch loop, ``=kernel`` makes a failed build an error.
"""

import os
import threading

import torch

from .gptq_kernel import kernel_dir, supported

BLOCK_COLUMNS = 128

_CPP = r"""
#include <torch/extension.h>
void refine_block(torch::Tensor gb, torch::Tensor c, torch::Tensor lo, torch::Tensor st,
                  torch::Tensor top, torch::Tensor hs, int64_t b0, int64_t L,
                  torch::Tensor dwb);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("refine_block", &refine_block); }
"""

_HIP = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <math.h>

#define ROWS 64
#define LMAX 128
#define LDS_LD (LMAX + 1)  // odd row stride: thread t, column q -> bank (t + q) % 32

// One thread per row. gb [N, L] fp32: the block's slice of g = e Hs (copied to LDS and
// updated there), c [N, K] fp32 codes (in/out), lo/st [N, K] fp32 per-column ladders, top [N]
// fp32 highest code of the row, hs [K, K] fp32 Hessian, dwb [N, L] fp32 out: the change of
// the decoded value of each column (0 where the code did not move).
__global__ void __launch_bounds__(ROWS)
refine_block_kernel(const float* __restrict__ gb, int L, float* __restrict__ c,
                    const float* __restrict__ lo, const float* __restrict__ st,
                    const float* __restrict__ top, const float* __restrict__ hs, int K,
                    int b0, float* __restrict__ dwb, int N) {
  __shared__ float sg[ROWS * LDS_LD];
  const int t = threadIdx.x;
  const long long r = (long long)blockIdx.x * ROWS + t;
  if (r >= N) return;  // no barriers below: each thread only touches its own LDS row
  float* g = sg + t * LDS_LD;
  const float* gr = gb + r * (long long)L;
  for (int q = 0; q < L; ++q) g[q] = gr[q];
  const float tp = top[r];
  const long long base = r * (long long)K;
  for (int jj = 0; jj < L; ++jj) {
    const int j = b0 + jj;
    const float lo_j = lo[base + j], st_j = st[base + j], cj = c[base + j];
    const float cur = __half2float(__float2half_rn(__fadd_rn(lo_j, __fmul_rn(cj, st_j))));
    const float hd = hs[(long long)j * K + j];
    const float gj = g[jj];
    // candidate +1
    const float cp = __fadd_rn(cj, 1.0f);
    const bool okp = (cp >= 0.0f) && (cp <= tp) && (st_j > 0.0f);
    const float dwp = __fsub_rn(
        __half2float(__float2half_rn(__fadd_rn(lo_j, __fmul_rn(cp, st_j)))), cur);
    const float gp = okp ? -__fadd_rn(__fmul_rn(__fmul_rn(2.0f, dwp), gj),
                                      __fmul_rn(__fmul_rn(dwp, dwp), hd))
                         : -INFINITY;
    // candidate -1
    const float cm = __fsub_rn(cj, 1.0f);
    const bool okm = (cm >= 0.0f) && (cm <= tp) && (st_j > 0.0f);
    const float dwm = __fsub_rn(
        __half2float(__float2half_rn(__fadd_rn(lo_j, __fmul_rn(cm, st_j)))), cur);
    const float gm = okm ? -__fadd_rn(__fmul_rn(__fmul_rn(2.0f, dwm), gj),
                                      __fmul_rn(__fmul_rn(dwm, dwm), hd))
                         : -INFINITY;
    const bool pick_m = gm > gp;  // ties go to +1, like torch's max over the stacked pair
    const float best = pick_m ? gm : gp;
    if (best > 0.0f) {  // NaN is false
      const float dw = pick_m ? dwm : dwp;
      c[base + j] = pick_m ? cm : cp;
      dwb[r * (long long)L + jj] = dw;
      const float* hr = hs + (long long)j * K + b0;
      for (int q = jj + 1; q < L; ++q) g[q] = __fadd_rn(g[q], __fmul_rn(dw, hr[q]));
    }
  }
}

void refine_block(torch::Tensor gb, torch::Tensor c, torch::Tensor lo, torch::Tensor st,
                  torch::Tensor top, torch::Tensor hs, int64_t b0, int64_t L,
                  torch::Tensor dwb) {
  TORCH_CHECK(gb.is_contiguous() && c.is_contiguous() && lo.is_contiguous());
  TORCH_CHECK(st.is_contiguous() && top.is_contiguous() && hs.is_contiguous());
  TORCH_CHECK(dwb.is_contiguous());
  TORCH_CHECK(gb.scalar_type() == at::kFloat && c.scalar_type() == at::kFloat);
  TORCH_CHECK(lo.scalar_type() == at::kFloat && st.scalar_type() == at::kFloat);
  TORCH_CHECK(top.scalar_type() == at::kFloat && hs.scalar_type() == at::kFloat);
  TORCH_CHECK(dwb.scalar_type() == at::kFloat);
  TORCH_CHECK(L > 0 && L <= LMAX && gb.size(1) == L && dwb.size(1) == L);
  const int N = c.size(0), K = c.size(1);
  TORCH_CHECK(b0 >= 0 && b0 + L <= K && hs.size(0) == K && hs.size(1) == K);
  if (N == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((N + ROWS - 1) / ROWS);
  refine_block_kernel<<<grid, ROWS, 0, stream>>>(
      gb.data_ptr<float>(), (int)L, c.data_ptr<float>(), lo.data_ptr<float>(),
      st.data_ptr<float>(), top.data_ptr<float>(), hs.data_ptr<float>(), K, (int)b0,
      dwb.data_ptr<float>(), N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_lock = threading.Lock()
_mod = None
_failed: str | None = None


def load():
    """The compiled extension, or None (the caller falls back to the torch loop)."""
    global _mod, _failed
    if _mod is not None or _failed is not None:
        return _mod
    with _lock:
        if _mod is not None or _failed is not None:
            return _mod
        try:
            from torch.utils.cpp_extension import load_inline

            d = (
                kernel_dir() / "refine_v1"
            )  # its own directory: load_inline writes cuda.cu there
            d.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx1030")
            _mod = load_inline(
                name="hi2_refine_v1",
                cpp_sources=[_CPP],
                cuda_sources=[_HIP],
                extra_cuda_cflags=["-O3", "-ffp-contract=off"],
                build_directory=str(d),
                verbose=False,
            )
        except Exception as e:  # build or import failure: use the torch loop
            _failed = f"{type(e).__name__}: {e}"
            if os.environ.get("HI2_REFINE_BACKEND") == "kernel":
                raise
        return _mod


def available(dev: torch.device) -> bool:
    backend = os.environ.get("HI2_REFINE_BACKEND", "auto")
    if backend == "torch":
        return False
    ok = supported(dev) and load() is not None
    if backend == "kernel" and not ok:
        raise RuntimeError(f"refine kernel unavailable on {dev}: {_failed}")
    return ok


def failure() -> str | None:
    return _failed


def refine_block(gb, c, lo_x, st_x, top, hs, b0, length, dw_block) -> None:
    """Launch the fused loop for columns ``[b0, b0 + length)`` (see the module doc): ``c``
    is updated in place and ``dw_block`` [N, length] receives the change of each decoded value."""
    load().refine_block(gb, c, lo_x, st_x, top, hs, int(b0), int(length), dw_block)
