# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Fused GPTQ column loop for one lazy-update block (HIP, gfx10/gfx11).

The reference (:func:`hi2.hessian.gptq_block_reference`) rounds one column at a time
with about 20 small torch ops per column, so a K=9216 tensor costs about 200k kernel
launches per candidate group. Here one launch handles a whole block of up to 128
columns for every row of a batch: one thread per row keeps its row segment in LDS,
rounds column ``i`` with the fixed ladder of its block and pushes the error into the
remaining columns of the block. Rows may come from different tensors (experts): each
row carries a group index that selects its ``u`` factor, its column-to-block map and
its code count.

The arithmetic is the reference's, operation by operation, with every product and sum
rounded separately (``-ffp-contract=off`` plus the ``__f*_rn`` intrinsics, correctly
rounded division, ``rintf`` = round half to even, ``__float2half_rn``), so codes, errors
and the fp16 rounding of ``wq`` are bit-identical to the torch loop. The cross-block
update stays a torch GEMM issued with the reference's shapes (see
:func:`hi2.hessian.encode_jobs_gptq`).

The extension is compiled on first use with ``torch.utils.cpp_extension`` (hipcc) into
``$HI2_KERNEL_DIR`` (default ``~/.cache/hi2/kernels``) and only for gfx10/gfx11 HIP
devices; anything else (CPU, CUDA, a failed build) uses the torch reference.
``HI2_GPTQ_BACKEND=torch`` forces the reference, ``=kernel`` makes a failed build an error.
"""

import os
import threading
from pathlib import Path

import torch

BLOCK_COLUMNS = 128
THREADS = 64  # rows per workgroup (two wave32)

_CPP = r"""
#include <torch/extension.h>
void gptq_block(torch::Tensor wc, int64_t i1, int64_t L, torch::Tensor err1,
                torch::Tensor codes, torch::Tensor lo, torch::Tensor st,
                torch::Tensor row_grp, torch::Tensor u_ptrs, torch::Tensor blk,
                torch::Tensor nmax, torch::Tensor comp);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gptq_block", &gptq_block); }
"""

_HIP = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

#define ROWS 64
#define LMAX 128
#define LDS_LD (LMAX + 1)  // odd row stride: thread t, column c -> bank (t + c) % 32

// One thread per row. wc [N, K] fp32 (read only: the block's row segment is copied to
// LDS and updated there, like the reference's clone w1), err1 [N, L] fp32 out,
// codes [N, K] uint8 out (column positions in the rounding order), lo/st [N, ldp] fp32
// decoded block parameters, row_grp [N] int32, per group: u_ptrs [G] int64 (address of
// a contiguous [K, K] fp32 factor), blk [G, K] int32 (block of each column), nmax [G]
// fp32 (number of codes - 1).
__global__ void __launch_bounds__(ROWS)
gptq_block_kernel(const float* __restrict__ wc, int K, int i1, int L,
                  float* __restrict__ err1, unsigned char* __restrict__ codes,
                  const float* __restrict__ lo, const float* __restrict__ st, int ldp,
                  const int* __restrict__ row_grp, const long long* __restrict__ u_ptrs,
                  const int* __restrict__ blk, const float* __restrict__ nmax, int N,
                  float* __restrict__ comp) {
  __shared__ float sw[ROWS * LDS_LD];
  const int t = threadIdx.x;
  const long long r = (long long)blockIdx.x * ROWS + t;
  if (r >= N) return;  // no barriers below: each thread only touches its own LDS row
  float* w = sw + t * LDS_LD;
  const float* wr = wc + r * (long long)K + i1;
  for (int c = 0; c < L; ++c) w[c] = wr[c];
  const int g = row_grp[r];
  const float* u = reinterpret_cast<const float*>(u_ptrs[g]);
  const int* bl = blk + (long long)g * K;
  const float nm = nmax[g];
  const float* lor = lo + r * (long long)ldp;
  const float* str = st + r * (long long)ldp;
  float* er = err1 + r * (long long)L;
  unsigned char* cr = codes + r * (long long)K;
  float* pr = comp ? comp + r * (long long)K : nullptr;
  for (int i = 0; i < L; ++i) {
    const int j = i1 + i;
    const int b = bl[j];
    const float lo_j = lor[b];
    const float st_j = str[b];
    const float col = w[i];
    const float safe = (st_j == 0.0f) ? 1.0f : st_j;
    float q = rintf(__fdiv_rn(__fsub_rn(col, lo_j), safe));
    q = (q < 0.0f) ? 0.0f : q;    // torch.clamp(0, n - 1), NaN propagates
    q = (q > nm) ? nm : q;
    if (st_j == 0.0f) q = 0.0f;
    const float wq = __half2float(__float2half_rn(__fadd_rn(lo_j, __fmul_rn(q, st_j))));
    cr[j] = (unsigned char)q;
    if (pr) pr[j] = col;  // the value this code was rounded from (QAD init)
    const float* ui = u + (long long)j * K + i1;
    const float e = __fdiv_rn(__fsub_rn(col, wq), ui[i]);
    for (int c = i + 1; c < L; ++c) w[c] = __fsub_rn(w[c], __fmul_rn(e, ui[c]));
    er[i] = e;
  }
}

void gptq_block(torch::Tensor wc, int64_t i1, int64_t L, torch::Tensor err1,
                torch::Tensor codes, torch::Tensor lo, torch::Tensor st,
                torch::Tensor row_grp, torch::Tensor u_ptrs, torch::Tensor blk,
                torch::Tensor nmax, torch::Tensor comp) {
  TORCH_CHECK(wc.is_contiguous() && err1.is_contiguous() && codes.is_contiguous());
  TORCH_CHECK(lo.is_contiguous() && st.is_contiguous() && blk.is_contiguous());
  TORCH_CHECK(wc.scalar_type() == at::kFloat && err1.scalar_type() == at::kFloat);
  TORCH_CHECK(codes.scalar_type() == at::kByte && blk.scalar_type() == at::kInt);
  TORCH_CHECK(row_grp.scalar_type() == at::kInt && u_ptrs.scalar_type() == at::kLong);
  TORCH_CHECK(L > 0 && L <= LMAX && err1.size(1) == L);
  TORCH_CHECK(comp.numel() == 0 || (comp.is_contiguous() && comp.scalar_type() == at::kFloat
              && comp.sizes() == wc.sizes()));
  const int N = wc.size(0), K = wc.size(1);
  if (N == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((N + ROWS - 1) / ROWS);
  gptq_block_kernel<<<grid, ROWS, 0, stream>>>(
      wc.data_ptr<float>(), K, (int)i1, (int)L, err1.data_ptr<float>(),
      codes.data_ptr<unsigned char>(), lo.data_ptr<float>(), st.data_ptr<float>(),
      (int)lo.size(1), row_grp.data_ptr<int>(),
      reinterpret_cast<const long long*>(u_ptrs.data_ptr<int64_t>()),
      blk.data_ptr<int>(), nmax.data_ptr<float>(), N,
      comp.numel() ? comp.data_ptr<float>() : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_lock = threading.Lock()
_mod = None
_failed: str | None = None


def kernel_dir() -> Path:
    d = os.environ.get("HI2_KERNEL_DIR")
    return Path(d) if d else Path.home() / ".cache" / "hi2" / "kernels"


def supported(dev: torch.device) -> bool:
    """A HIP device of the RDNA families the kernel was written for."""
    if dev.type != "cuda" or not torch.version.hip:
        return False
    arch = torch.cuda.get_device_properties(dev).gcnArchName
    return arch.startswith(("gfx10", "gfx11"))


def load():
    """The compiled extension, or None (the caller falls back to the reference)."""
    global _mod, _failed
    if _mod is not None or _failed is not None:
        return _mod
    with _lock:
        if _mod is not None or _failed is not None:
            return _mod
        try:
            from torch.utils.cpp_extension import load_inline

            d = kernel_dir()
            d.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx1030")
            _mod = load_inline(
                name="hi2_gptq_v2",
                cpp_sources=[_CPP],
                cuda_sources=[_HIP],
                extra_cuda_cflags=["-O3", "-ffp-contract=off"],
                build_directory=str(d),
                verbose=False,
            )
        except Exception as e:  # build or import failure: use the torch reference
            _failed = f"{type(e).__name__}: {e}"
            if os.environ.get("HI2_GPTQ_BACKEND") == "kernel":
                raise
        return _mod


def available(dev: torch.device) -> bool:
    if os.environ.get("HI2_GPTQ_BACKEND", "auto") == "torch":
        return False
    return supported(dev) and load() is not None


def failure() -> str | None:
    return _failed


def gptq_block(
    wc, i1, length, err1, codes, lo, st, row_grp, u_ptrs, blk, nmax, comp=None
):
    """Launch the fused column loop for block ``[i1, i1 + length)`` (see the module
    doc); ``comp`` ([N, K] fp32, optional) receives the value each code was rounded from."""
    if comp is None:
        comp = torch.empty(0, dtype=torch.float32, device=wc.device)
    load().gptq_block(
        wc, int(i1), int(length), err1, codes, lo, st, row_grp, u_ptrs, blk, nmax, comp
    )
