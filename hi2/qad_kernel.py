# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Fused ladder decode of the fake-quantised Linears of QAD, forward and backward (HIP,
gfx10/gfx11).

:meth:`hi2.qad.FakeQuantLinear._group` builds the decode of a tensor from several hundred small
torch ops per group (byte rounding, the float64 parameter decode, ``z``, rounding, clamping, the
float64 ladder value and its STE graph), and autograd runs the same number of ops backwards.
On a Qwen3.5-4B layer that was 61 ms forward and 78 ms backward of a 556 ms optimizer step.
Here one launch per group computes the stored-basis values of all its rows, and one launch
computes the gradients of the latent weights and of the two ladder bytes of every block.

One thread handles one weight (consecutive threads take consecutive columns, so every access is
coalesced); the threads of a block decode its two bytes again and the per-block gradient sums are
tree reductions in LDS. The forward is exact: the value of each weight is ``fp16(lo + q * step)`` with the
sum formed in float64 (as :func:`hi2.format.ladder_dequantize`) and the parameters decoded as
:func:`hi2.format.decode_params`, so it equals the torch path and the exported tensor bit for
bit. The backward is the gradient the torch graph computes (the STE of
:class:`hi2.qad.FakeQuantLinear`), written in closed form; the per-block sums run in a fixed
order, so it agrees with autograd up to float32 summation rounding.

Built on first use like :mod:`hi2.gptq_kernel` (into ``$HI2_KERNEL_DIR/qad_v3``), for gfx10/gfx11
HIP devices only; anything else, or a failed build, uses the torch graph.
``HI2_QAD_BACKEND=torch`` forces the torch graph, ``=kernel`` makes a failed build an error.
"""

import os
import threading

import torch

from .gptq_kernel import kernel_dir, supported

_CPP = r"""
#include <torch/extension.h>
void decode_fwd(torch::Tensor latent, torch::Tensor unit, torch::Tensor lo_u,
                torch::Tensor st_u, torch::Tensor consts, torch::Tensor rows, int64_t block,
                double nmax, torch::Tensor out);
void decode_bwd(torch::Tensor grad, torch::Tensor latent, torch::Tensor unit,
                torch::Tensor lo_u, torch::Tensor st_u, torch::Tensor consts,
                torch::Tensor rows, int64_t block, double nmax, torch::Tensor g_latent,
                torch::Tensor g_lo, torch::Tensor g_st);
void wht_scale(torch::Tensor x, torch::Tensor flags, torch::Tensor scale, bool pre_div,
               torch::Tensor out);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode_fwd", &decode_fwd);
  m.def("decode_bwd", &decode_bwd);
  m.def("wht_scale", &wht_scale);
}
"""

_HIP = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <math.h>

#define TPB 128

static void check(const torch::Tensor& t, at::ScalarType type, const char* name);

// The two bytes of a block and the row constants -> fp16 lo and step (as float):
// bytes = clamp(round(u), 0, 255); lo = fp16(float(double(byte) * c1 + c0)),
// step = fp16(float(byte) * c2) (hi2.format.decode_params).
__device__ __forceinline__ void block_ladder(float lo_u, float st_u, const __half* c,
                                             float& lo_q, float& st_q) {
  float lo_b = fminf(fmaxf(rintf(lo_u), 0.0f), 255.0f);
  float st_b = fminf(fmaxf(rintf(st_u), 0.0f), 255.0f);
  const double lo_d = (double)lo_b * (double)__half2float(c[1]) + (double)__half2float(c[0]);
  lo_q = __half2float(__float2half_rn(__double2float_rn(lo_d)));
  st_q = __half2float(__float2half_rn(__fmul_rn(st_b, __half2float(c[2]))));
}

// z = (v - lo) / safe and the code q = clamp(round(z), 0, nmax), 0 for a zero step.
__device__ __forceinline__ float code_of(float z, float nmax, bool zero) {
  float q = rintf(z);
  q = (q < 0.0f) ? 0.0f : q;  // torch.clamp: NaN propagates
  q = (q > nmax) ? nmax : q;
  return zero ? 0.0f : q;
}

// One thread per weight: consecutive threads take consecutive columns of a row, so every load
// and store is coalesced; the 8 + 8 byte block parameters are decoded again by each thread of the
// block (a few double operations). A workgroup is one row and TPB columns; TPB is a multiple of
// every block size (16, 32, 64), so a block never straddles two workgroups.

// out[rows[r], k] for every row r of the group and every column k.
__global__ void __launch_bounds__(TPB)
decode_fwd_kernel(const float* __restrict__ latent, const float* __restrict__ unit,
                  const float* __restrict__ lo_u, const float* __restrict__ st_u,
                  const __half* __restrict__ consts, const long long* __restrict__ rows,
                  int K, int block_shift, float nmax, float* __restrict__ out) {
  const int k = blockIdx.x * TPB + threadIdx.x;
  if (k >= K) return;
  const int r = blockIdx.y, nblk = K >> block_shift, b = k >> block_shift;
  const long long off = rows[r] * (long long)K + k;
  float lo_q, st_q;
  block_ladder(lo_u[(long long)r * nblk + b], st_u[(long long)r * nblk + b],
               consts + (long long)r * 3, lo_q, st_q);
  const bool zero = (st_q == 0.0f);
  const float safe = zero ? 1.0f : st_q;
  float v = latent[off];
  if (unit) v = __fmul_rn(v, unit[off]);
  const float z = __fdiv_rn(__fsub_rn(v, lo_q), safe);
  const float q = code_of(z, nmax, zero);
  const double val = (double)lo_q + (double)q * (double)st_q;  // exact
  out[off] = (float)static_cast<_Float16>(val);                // one rounding to fp16
}

// Gradients of the torch STE graph (hi2.qad.FakeQuantLinear._group):
//   y = lo_h + q_ste * st_h, z = (v - lo_h) / safe, zc = clamp(z), safe = where(zero, 1, st_h)
// with G the gradient of the decoded value. gz = G * st inside the clamp range, else 0;
// d v = gz / safe; d lo_h = sum(G - gz / safe); d st_h = sum(G * q - [not zero] gz * (z / safe)).
// The per-block sums are tree reductions in LDS over the block's consecutive threads.
__global__ void __launch_bounds__(TPB)
decode_bwd_kernel(const float* __restrict__ grad, const float* __restrict__ latent,
                  const float* __restrict__ unit, const float* __restrict__ lo_u,
                  const float* __restrict__ st_u, const __half* __restrict__ consts,
                  const long long* __restrict__ rows, int K, int block_shift, float nmax,
                  float* __restrict__ g_latent, float* __restrict__ g_lo,
                  float* __restrict__ g_st) {
  __shared__ float s_lo[TPB], s_st[TPB];
  const int t = threadIdx.x, k = blockIdx.x * TPB + t;
  const int r = blockIdx.y, nblk = K >> block_shift, b = k >> block_shift;
  const __half* c = consts + (long long)r * 3;
  float a_lo = 0.0f, a_st = 0.0f;
  if (k < K) {
    const long long off = rows[r] * (long long)K + k;
    float lo_q, st_q;
    block_ladder(lo_u[(long long)r * nblk + b], st_u[(long long)r * nblk + b], c, lo_q, st_q);
    const bool zero = (st_q == 0.0f);
    const float safe = zero ? 1.0f : st_q;
    float v = latent[off];
    const float un = unit ? unit[off] : 1.0f;
    if (unit) v = __fmul_rn(v, un);
    const float z = __fdiv_rn(__fsub_rn(v, lo_q), safe);
    const float q = code_of(z, nmax, zero);
    const float G = grad[off];
    const bool inside = (z >= 0.0f) && (z <= nmax);
    const float gz = inside ? __fmul_rn(G, st_q) : 0.0f;
    const float gv = __fdiv_rn(gz, safe);
    g_latent[off] = unit ? __fmul_rn(gv, un) : gv;
    a_lo = __fsub_rn(G, gv);
    a_st = __fmul_rn(G, q);
    if (!zero) a_st = __fadd_rn(a_st, __fmul_rn(-gz, __fdiv_rn(z, safe)));
  }
  s_lo[t] = a_lo;
  s_st[t] = a_st;
  __syncthreads();
  const int block = 1 << block_shift;
  for (int step = block >> 1; step > 0; step >>= 1) {
    if ((t & (block - 1)) < step) {
      s_lo[t] = __fadd_rn(s_lo[t], s_lo[t + step]);
      s_st[t] = __fadd_rn(s_st[t], s_st[t + step]);
    }
    __syncthreads();
  }
  if ((t & (block - 1)) == 0 && k < K) {
    g_lo[(long long)r * nblk + b] = __fmul_rn(s_lo[t], __half2float(c[1]));
    g_st[(long long)r * nblk + b] = __fmul_rn(s_st[t], __half2float(c[2]));
  }
}

// out = (flagged 16-column groups transformed by the normalised Walsh-Hadamard matrix) / scale,
// or with pre_div: the division first, then the transform. The forward of QAD is "transform,
// then divide by the smoothing vector"; its backward is "divide, then transform" (H is
// symmetric). One workgroup is one row and TPB columns, whole 16-column groups.
__global__ void __launch_bounds__(TPB)
wht_scale_kernel(const float* __restrict__ x, const unsigned char* __restrict__ flags,
                 const __half* __restrict__ scale, int K, int pre_div,
                 float* __restrict__ out) {
  __shared__ float sh[TPB];
  const int t = threadIdx.x, k = blockIdx.x * TPB + t;
  const long long off = (long long)blockIdx.y * K + k;
  const bool valid = k < K;
  float v = valid ? x[off] : 0.0f;
  const float s = (valid && scale) ? __half2float(scale[k]) : 1.0f;
  if (pre_div && scale) v = __fdiv_rn(v, s);
  const float orig = v;
  sh[t] = v;
  __syncthreads();
  for (int h = 1; h < 16; h <<= 1) {  // butterfly inside each 16-lane group
    const float a = sh[t], b = sh[t ^ h];
    __syncthreads();
    sh[t] = (t & h) ? __fsub_rn(b, a) : __fadd_rn(a, b);
    __syncthreads();
  }
  const bool flagged = valid && flags[k >> 4] != 0;
  v = flagged ? __fmul_rn(sh[t], 0.25f) : orig;
  if (!pre_div && scale) v = __fdiv_rn(v, s);
  if (valid) out[off] = v;
}

void wht_scale(torch::Tensor x, torch::Tensor flags, torch::Tensor scale, bool pre_div,
               torch::Tensor out) {
  check(x, at::kFloat, "x");
  check(out, at::kFloat, "out");
  check(flags, at::kByte, "flags");
  TORCH_CHECK(scale.numel() == 0 || (scale.is_contiguous() && scale.scalar_type() == at::kHalf));
  const int K = x.size(1), n_rows = x.size(0);
  TORCH_CHECK(K % 16 == 0 && flags.numel() == K / 16);
  TORCH_CHECK(scale.numel() == 0 || scale.numel() == K);
  if (n_rows == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream();
  wht_scale_kernel<<<dim3((K + TPB - 1) / TPB, n_rows), TPB, 0, stream>>>(
      x.data_ptr<float>(), flags.data_ptr<unsigned char>(),
      scale.numel() ? reinterpret_cast<const __half*>(scale.data_ptr<at::Half>()) : nullptr, K,
      pre_div ? 1 : 0, out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void check(const torch::Tensor& t, at::ScalarType type, const char* name) {
  TORCH_CHECK(t.is_contiguous() && t.scalar_type() == type, name, " must be contiguous ", type);
}

void decode_fwd(torch::Tensor latent, torch::Tensor unit, torch::Tensor lo_u,
                torch::Tensor st_u, torch::Tensor consts, torch::Tensor rows, int64_t block,
                double nmax, torch::Tensor out) {
  check(latent, at::kFloat, "latent");
  check(lo_u, at::kFloat, "lo_u");
  check(st_u, at::kFloat, "st_u");
  check(consts, at::kHalf, "consts");
  check(rows, at::kLong, "rows");
  check(out, at::kFloat, "out");
  TORCH_CHECK(unit.numel() == 0 || (unit.is_contiguous() && unit.scalar_type() == at::kFloat));
  const int K = latent.size(1), n_rows = rows.size(0);
  TORCH_CHECK(block == 16 || block == 32 || block == 64, "block must be 16, 32 or 64");
  TORCH_CHECK(K % block == 0 && lo_u.size(0) == n_rows && lo_u.size(1) == K / block);
  if (n_rows == 0) return;
  const int shift = block == 16 ? 4 : (block == 32 ? 5 : 6);
  auto stream = at::cuda::getCurrentCUDAStream();
  decode_fwd_kernel<<<dim3((K + TPB - 1) / TPB, n_rows), TPB, 0, stream>>>(
      latent.data_ptr<float>(), unit.numel() ? unit.data_ptr<float>() : nullptr,
      lo_u.data_ptr<float>(), st_u.data_ptr<float>(),
      reinterpret_cast<const __half*>(consts.data_ptr<at::Half>()),
      reinterpret_cast<const long long*>(rows.data_ptr<int64_t>()), K, shift, (float)nmax,
      out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void decode_bwd(torch::Tensor grad, torch::Tensor latent, torch::Tensor unit,
                torch::Tensor lo_u, torch::Tensor st_u, torch::Tensor consts,
                torch::Tensor rows, int64_t block, double nmax, torch::Tensor g_latent,
                torch::Tensor g_lo, torch::Tensor g_st) {
  check(grad, at::kFloat, "grad");
  check(latent, at::kFloat, "latent");
  check(lo_u, at::kFloat, "lo_u");
  check(st_u, at::kFloat, "st_u");
  check(consts, at::kHalf, "consts");
  check(rows, at::kLong, "rows");
  check(g_latent, at::kFloat, "g_latent");
  check(g_lo, at::kFloat, "g_lo");
  check(g_st, at::kFloat, "g_st");
  TORCH_CHECK(unit.numel() == 0 || (unit.is_contiguous() && unit.scalar_type() == at::kFloat));
  const int K = latent.size(1), n_rows = rows.size(0);
  TORCH_CHECK(block == 16 || block == 32 || block == 64, "block must be 16, 32 or 64");
  TORCH_CHECK(K % block == 0 && lo_u.size(0) == n_rows && lo_u.size(1) == K / block);
  if (n_rows == 0) return;
  const int shift = block == 16 ? 4 : (block == 32 ? 5 : 6);
  auto stream = at::cuda::getCurrentCUDAStream();
  decode_bwd_kernel<<<dim3((K + TPB - 1) / TPB, n_rows), TPB, 0, stream>>>(
      grad.data_ptr<float>(), latent.data_ptr<float>(),
      unit.numel() ? unit.data_ptr<float>() : nullptr, lo_u.data_ptr<float>(),
      st_u.data_ptr<float>(), reinterpret_cast<const __half*>(consts.data_ptr<at::Half>()),
      reinterpret_cast<const long long*>(rows.data_ptr<int64_t>()), K, shift, (float)nmax,
      g_latent.data_ptr<float>(), g_lo.data_ptr<float>(), g_st.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_lock = threading.Lock()
_mod = None
_failed: str | None = None


def load():
    """The compiled extension, or None (the caller falls back to the torch graph)."""
    global _mod, _failed
    if _mod is not None or _failed is not None:
        return _mod
    with _lock:
        if _mod is not None or _failed is not None:
            return _mod
        try:
            from torch.utils.cpp_extension import load_inline

            d = (
                kernel_dir() / "qad_v3"
            )  # its own directory: load_inline writes cuda.cu there
            d.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx1030")
            _mod = load_inline(
                name="hi2_qad_v3",
                cpp_sources=[_CPP],
                cuda_sources=[_HIP],
                extra_cuda_cflags=["-O3", "-ffp-contract=off"],
                build_directory=str(d),
                verbose=False,
            )
        except Exception as e:  # build or import failure: use the torch graph
            _failed = f"{type(e).__name__}: {e}"
            if os.environ.get("HI2_QAD_BACKEND") == "kernel":
                raise
        return _mod


def available(dev: torch.device) -> bool:
    backend = os.environ.get("HI2_QAD_BACKEND", "auto")
    if backend == "torch":
        return False
    ok = supported(dev) and load() is not None
    if backend == "kernel" and not ok:
        raise RuntimeError(f"qad kernel unavailable on {dev}: {_failed}")
    return ok


def failure() -> str | None:
    return _failed


class _StoredDecode(torch.autograd.Function):
    """``stored = decode(latent, unit, ladder bytes)`` for all groups of one tensor."""

    @staticmethod
    def forward(ctx, latent, unit, meta, *params):
        out = torch.empty_like(latent)
        mod = load()
        for gi, (rows, consts, nbits, block) in enumerate(meta):
            mod.decode_fwd(
                latent, unit, params[2 * gi], params[2 * gi + 1], consts, rows, block,
                float(2**nbits - 1), out,
            )  # fmt: skip
        ctx.meta = meta
        ctx.save_for_backward(latent, unit, *params)
        return out

    @staticmethod
    def backward(ctx, grad):
        latent, unit, *params = ctx.saved_tensors
        grad = grad.contiguous()
        g_latent = torch.empty_like(latent)
        grads = []
        mod = load()
        for gi, (rows, consts, nbits, block) in enumerate(ctx.meta):
            g_lo, g_st = (
                torch.empty_like(params[2 * gi]),
                torch.empty_like(params[2 * gi]),
            )
            mod.decode_bwd(
                grad, latent, unit, params[2 * gi], params[2 * gi + 1], consts, rows, block,
                float(2**nbits - 1), g_latent, g_lo, g_st,
            )  # fmt: skip
            grads += [g_lo, g_st]
        return (g_latent, None, None, *grads)


class _Transform(torch.autograd.Function):
    """``H(x) / scale`` over the flagged 16-column groups; the backward is ``H(grad / scale)``."""

    @staticmethod
    def forward(ctx, x, flags, scale):
        out = torch.empty_like(x)
        load().wht_scale(x.contiguous(), flags, scale, False, out)
        ctx.flags, ctx.scale = flags, scale
        return out

    @staticmethod
    def backward(ctx, grad):
        out = torch.empty_like(grad)
        load().wht_scale(grad.contiguous(), ctx.flags, ctx.scale, True, out)
        return out, None, None


def original(fq) -> torch.Tensor:
    """The weights of a :class:`hi2.qad.FakeQuantLinear` in the model's own basis, with the STE
    graph: the stored values, then the Hadamard transform of the flagged groups and the division
    by the smoothing vector, both fused."""
    w = stored(fq, True)
    if not fq.has_had and not fq.has_scale:
        return w
    return _Transform.apply(
        w, fq.had, fq.col_scale if fq.has_scale else fq.col_scale[:0]
    )


def stored(fq, grad: bool) -> torch.Tensor:
    """The stored-basis weights of a :class:`hi2.qad.FakeQuantLinear` (rows in the tensor's
    own order), with the STE graph when ``grad``."""
    meta = [(g.rows, g.consts, g.nbits, g.block) for g in fq.groups]
    params = [p for g in fq.groups for p in (g.lo_u, g.step_u)]
    latent = fq.latent
    if grad and torch.is_grad_enabled():
        return _StoredDecode.apply(latent, fq.unit, meta, *params)
    with torch.no_grad():
        out = torch.empty_like(latent)
        mod = load()
        for (rows, consts, nbits, block), lo_u, st_u in zip(
            meta, params[::2], params[1::2]
        ):
            mod.decode_fwd(
                latent.detach(), fq.unit, lo_u.detach(), st_u.detach(), consts, rows, block,
                float(2**nbits - 1), out,
            )  # fmt: skip
        return out
