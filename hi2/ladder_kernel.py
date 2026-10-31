# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Fused ladder search of :func:`hi2.encode.fit_ladder` (HIP, gfx10/gfx11).

The reference walks a 6x6 grid of ``(a, b)`` range shrinks and refits, and every grid
point costs a handful of full-tile torch kernels (round, clamp, multiply, subtract,
square, weighted sum). On a 512-expert layer that is 1551 Linears x 2 chunks x 6
candidates x ~10 launches over a 1638x640x16 fp32 tile - minutes per layer, with the
cards at 12 %.

Here one thread owns one ``(row, block)``: it reads the block once, keeps the min/max in
registers and evaluates every grid point in place, so the tile is read once per
candidate instead of once per grid point. The arithmetic is the reference's: ``rintf``
is the half-to-even rounding of ``torch.round``, the codes are clamped to
``[0, n-1]``, and the weighted squared error is summed in grid order with a strict
``<``, so ``lo``/``step`` are the same ladder the reference picks (up to float32
summation order within a block).

Built on first use like :mod:`hi2.gptq_kernel`; anything else, or a failed build, uses
the torch loop. ``HI2_LADDER_BACKEND=torch`` forces the reference.
"""

import os
import threading

import torch

from .gptq_kernel import kernel_dir, supported

_CPP = r"""
#include <torch/extension.h>
void ladder_grid(torch::Tensor wb, torch::Tensor db, torch::Tensor grid, int64_t n,
                 torch::Tensor lo, torch::Tensor st, torch::Tensor err);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("ladder_grid", &ladder_grid); }
"""

_HIP = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <hip/hip_runtime.h>
#include <math.h>

// One thread per (row, block): the block is [B] consecutive floats.
__global__ void ladder_grid_k(const float* __restrict__ wb, const float* __restrict__ db,
                              const float* __restrict__ grid, int G, int n, int nb, int B,
                              long total, float* __restrict__ out_lo,
                              float* __restrict__ out_st, float* __restrict__ out_err) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  int blk = (int)(i % nb);
  const float* w = wb + (long)i * B;
  const float* d = db + (long)blk * B;

  float mn = INFINITY, mx = -INFINITY;
  for (int t = 0; t < B; ++t) {
    float v = w[t];
    mn = fminf(mn, v);
    mx = fmaxf(mx, v);
  }
  float span = mx - mn;
  float nm1 = (float)(n - 1);
  float nmax = (float)(n - 1);
  float best_err = INFINITY, best_lo = 0.0f, best_st = 0.0f;
  for (int gi = 0; gi < G; ++gi) {
    float lo = mn + grid[gi] * span;
    for (int gj = 0; gj < G; ++gj) {
      float step = (mx - grid[gj] * span - lo) / nm1;
      float safe = (step == 0.0f) ? 1.0f : step;
      float err = 0.0f;
      for (int t = 0; t < B; ++t) {
        float q = rintf((w[t] - lo) / safe);
        q = fminf(fmaxf(q, 0.0f), nmax);
        float diff = q * step + lo - w[t];
        err += d[t] * diff * diff;
      }
      if (err < best_err) {
        best_err = err;
        best_lo = lo;
        best_st = step;
      }
    }
  }
  out_lo[i] = best_lo;
  out_st[i] = best_st;
  out_err[i] = best_err;
}

void ladder_grid(torch::Tensor wb, torch::Tensor db, torch::Tensor grid, int64_t n,
                 torch::Tensor lo, torch::Tensor st, torch::Tensor err) {
  int N = (int)wb.size(0);
  int nb = (int)wb.size(1);
  int B = (int)wb.size(2);
  int G = (int)grid.size(0);
  long total = (long)N * nb;
  int threads = 256;
  long blocks = (total + threads - 1) / threads;
  ladder_grid_k<<<(int)blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      wb.data_ptr<float>(), db.data_ptr<float>(), grid.data_ptr<float>(), G, (int)n,
      nb, B, total, lo.data_ptr<float>(), st.data_ptr<float>(), err.data_ptr<float>());
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

            d = kernel_dir()
            d.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx1030")
            _mod = load_inline(
                name="hi2_ladder_v1",
                cpp_sources=[_CPP],
                cuda_sources=[_HIP],
                extra_cuda_cflags=["-O3", "-ffp-contract=off"],
                build_directory=str(d),
                verbose=False,
            )
        except Exception as e:  # build or import failure: use the torch reference
            _failed = f"{type(e).__name__}: {e}"
            if os.environ.get("HI2_LADDER_BACKEND") == "kernel":
                raise
        return _mod


def available(dev: torch.device) -> bool:
    if os.environ.get("HI2_LADDER_BACKEND", "auto") == "torch":
        return False
    return supported(dev) and load() is not None


def failure() -> str | None:
    return _failed


def ladder_grid(wb: torch.Tensor, db: torch.Tensor, grid: tuple, n: int):
    """``(lo, step, err)`` per ``(row, block)``, all ``[N, nb]`` float32."""
    g = torch.tensor(grid, dtype=torch.float32, device=wb.device)
    rows, nb = wb.shape[0], wb.shape[1]
    lo = torch.empty(rows, nb, dtype=torch.float32, device=wb.device)
    st = torch.empty(rows, nb, dtype=torch.float32, device=wb.device)
    err = torch.empty(rows, nb, dtype=torch.float32, device=wb.device)
    load().ladder_grid(wb.contiguous(), db.contiguous(), g, int(n), lo, st, err)
    return lo, st, err
