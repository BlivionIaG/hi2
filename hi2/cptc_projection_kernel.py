# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Fused block-local CPTC projection (Triton, gfx10/gfx11, Wave32).

:func:`hi2.cptc.project_codes` updates the ``ceil(L/K)`` K-bit symbols that read one
code position in ``t mod period`` phases and ``max(1, sweeps)`` sweeps, and spends a
handful of small torch launches per phase and per candidate symbol. This module keeps
the same deterministic greedy coordinate projection, but one Triton program owns one
row: it loads the row's ``B`` codes once, and runs the fixed phase/sweep/candidate
loops in registers. One launch performs every sweep of the whole batch; there is no
host synchronization between phases, no ``[N, B]`` int64 state, and only the ``B``
codes of the current row are held in registers.

The arithmetic is the reference's, operation by operation:

* the window ``q`` is rebuilt from the current cyclic code stream and masked to
  ``2**depth - 1``; candidate ``sym`` replaces the K bits at shift ``j*K`` of the
  ``ceil(L/K)`` windows that read the position;
* ``index = (((a0*q) & 0xFFFF) >> 10) + (((a1*q) & 0xFFFF) >> 10)`` (Q6 mirror
  codec, the serialized format's fixed tap width);
* ``lut`` is the already mirror-expanded fp16 table, so the stored value is
  ``fp16(scale_fp32 * lut_fp32)`` (both operands are exact fp16 values in fp32);
* the loss is ``d[t] * (value - w[t])**2``; each candidate accumulates its
  affected-window terms in the reference's offset order ``0 .. span-1``, so the
  fp32 rounding matches the reference's reduction. A tree reduction can round two
  distinct candidate errors to one value and flip a near-tie (the 2048-row
  K2/B64 counterexample did exactly that). The lowest symbol wins exact ties.
  ``d`` is optional and, like the reference, is not part of the state.

The phase loop of the reference is preserved exactly: every position of one phase
is decided against the codes at the start of that phase, and the chosen symbols are
written before the next phase. Positions of a phase never share a window, so the
simultaneous update is well defined. Inputs are read-only; the result is a fresh
uint8 ``[N, B]`` tensor. ``N == 0`` returns the empty result without touching a
device or the Triton runtime.

Only the serialized CPTC geometry is accepted: ``nbits`` 2 or 4, ``depth`` 4..16
(the projection itself also works below ``2*nbits``; policy validation stays with
the codec), and ``block`` 16/32/64. The phase period is a power of two that divides
all three block sizes for every accepted depth; unsupported metadata raises instead
of silently changing the depth or the schedule. The ``L16, K2/K4, B16/32/64``
default fastpath is covered by the offline gfx1030 compile proof in the parent's
study log.
"""

import torch

try:  # keep the module importable in CPU-only dev/tests environments
    import triton
    import triton.language as tl
except Exception as exc:  # pragma: no cover - exercised only without Triton
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = exc
else:
    _TRITON_IMPORT_ERROR = None

_TAP_Q = 6  # the serialized format's fixed Q6 mirror table
_BLOCKS = (16, 32, 64)
_NBITS = (2, 4)
_MIN_DEPTH = 4
_MAX_DEPTH = 16
_TABLE_ENTRIES = 2 ** (_TAP_Q + 1) - 1


def _geometry(nbits: int, depth: int, block: int) -> tuple[int, int, int]:
    """``(span, period, positions_per_phase)`` for one serialized CPTC geometry."""
    if nbits not in _NBITS:
        raise ValueError(f"CPTC projection needs 2 or 4 bits per value, got {nbits}")
    if not _MIN_DEPTH <= depth <= _MAX_DEPTH:
        raise ValueError(f"CPTC projection depth {depth} outside 4..16")
    if block not in _BLOCKS:
        raise ValueError(f"CPTC projection block {block} outside {_BLOCKS}")
    span = (depth + nbits - 1) // nbits
    period = 1 << (span - 1).bit_length()
    if block % period:
        raise ValueError(f"CPTC projection period {period} does not divide {block}")
    return span, period, block // period


if triton is not None:

    @triton.jit
    def _f32_to_f16_rn(x):
        """Round fp32 to fp16 with IEEE round-to-nearest-even, including subnormals.

        Triton's implicit fp32->fp16 cast on gfx1030 flushes subnormal results to
        zero.  The reference stores ``fp16(scale_fp32 * value_fp32)`` with the
        host conversion's denormal support; the explicit projection test
        ``test_gpu_projection_zero_subnormal_scales_and_ties`` exposes the
        difference.  This integer path is the same round-to-nearest-even
        conversion for normals, subnormals and overflow.
        """
        bits = tl.cast(x, tl.int32, bitcast=True)
        sign = (bits >> 16) & 0x8000
        exp = (bits >> 23) & 0xFF
        mant = bits & 0x7FFFFF
        unbiased = exp - 127
        special = (exp == 0) | (exp == 0xFF)

        # Normal fp16 results: 23 -> 10 explicit mantissa bits, ties to even.
        he = exp - 112
        keep = mant >> 13
        rem = mant & 0x1FFF
        round_up = (rem > 0x1000) | ((rem == 0x1000) & ((keep & 1) == 1))
        keep = keep + round_up.to(tl.int32)
        carry = keep >= 0x400
        keep = tl.where(carry, 0, keep)
        he = he + carry.to(tl.int32)
        normal_bits = sign | (he << 10) | keep
        normal_bits = tl.where(he >= 31, sign | 0x7C00, normal_bits)
        normal_f32 = tl.cast(
            normal_bits.to(tl.uint16), tl.float16, bitcast=True
        ).to(tl.float32)

        # Subnormal fp16 results: round in units of 2**-24 with float arithmetic.
        # The gfx1030 bitcast/denormal path was not reliable here; this is exact
        # for the projection's product range and preserves the sign.
        scaled = tl.where(
            special, 0.0, tl.minimum(tl.abs(x) * 16777216.0, 1024.0)
        )
        q0 = tl.floor(scaled)
        frac = scaled - q0
        q = q0.to(tl.int32)
        round_up_s = (frac > 0.5) | ((frac == 0.5) & ((q & 1) == 1))
        q = q + round_up_s.to(tl.int32)
        sub_f32 = q.to(tl.float32) * 5.960464477539063e-08
        sub_f32 = tl.where(sign != 0, -sub_f32, sub_f32)

        chosen = tl.where(unbiased < -14, sub_f32, normal_f32)
        return tl.where(special, x, chosen)

    @triton.jit
    def _project_kernel(
        w_ptr,  # *fp32 [N, B], row stride may exceed B
        init_ptr,  # *u8 [N, B], row stride may exceed B
        scales_ptr,  # *fp16 [N]
        lut_ptr,  # *fp16 [127], mirror-expanded
        d_ptr,  # *fp32 [B], or w_ptr when HAS_D is 0
        out_ptr,  # *u8 [N, B], contiguous
        w_row,
        w_col,
        c_row,
        c_col,
        s_step,
        l_step,
        d_step,
        a0,
        a1,
        sweep_count,
        phase_count,
        NBITS: tl.constexpr,
        DEPTH: tl.constexpr,
        BLOCK: tl.constexpr,
        SPAN: tl.constexpr,
        PERIOD: tl.constexpr,
        NPOS: tl.constexpr,
        SYMS: tl.constexpr,
        HAS_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        row64 = row.to(tl.int64)
        positions = tl.arange(0, BLOCK)
        # The whole row lives in registers: one int32 per code.
        codes = tl.load(init_ptr + row64 * c_row + positions * c_col).to(tl.int32)
        scale = tl.load(scales_ptr + row64 * s_step).to(tl.float32)

        tap = tl.arange(0, PERIOD)
        shift = tap * NBITS
        depth_mask = (1 << DEPTH) - 1
        syms = tl.arange(0, SYMS)

        for _sweep in range(0, sweep_count):
            for phase in range(0, phase_count):
                pos = phase + tl.arange(0, NPOS) * PERIOD
                # One accumulator per candidate. The reference sums the affected
                # windows left to right (offset 0 .. span-1); accumulating each
                # window term into the same register in that order reproduces the
                # reference's fp32 reduction exactly, including near-ties. The old
                # tree reduction could round two distinct errors to one value and
                # keep the lower symbol where the reference kept the other one.
                cand_err = tl.zeros([SYMS, NPOS], tl.float32)
                for j in tl.static_range(SPAN):
                    affected = (pos + BLOCK - j) % BLOCK
                    # q[t] for the window that reads each position at shift j*K.
                    # The window axis is padded to PERIOD; the depth mask discards
                    # the padding bits for depths that are not a multiple of K.
                    window = (affected[:, None] + tap[None, :]) % BLOCK
                    gathered = tl.gather(
                        codes, tl.reshape(window, [NPOS * PERIOD]), 0
                    )
                    q = tl.sum(
                        tl.reshape(gathered, [NPOS, PERIOD]) << shift[None, :],
                        axis=1,
                    ) & depth_mask
                    keep_j = ~(((1 << NBITS) - 1) << (j * NBITS))
                    qc = (
                        (q[None, :] & keep_j) | (syms[:, None] << (j * NBITS))
                    ) & depth_mask
                    idx = ((a0 * qc) & 0xFFFF) >> 10  # 16 - Q6
                    idx += ((a1 * qc) & 0xFFFF) >> 10
                    value = tl.load(lut_ptr + idx * l_step).to(tl.float32)
                    val = _f32_to_f16_rn(scale * value).to(tl.float32)
                    wv = tl.load(w_ptr + row64 * w_row + affected * w_col)
                    diff = val - wv[None, :]
                    if HAS_D:
                        dv = tl.load(d_ptr + affected * d_step)
                        cand_err += dv[None, :] * (diff * diff)
                    else:
                        cand_err += diff * diff

                best_err = tl.min(cand_err, axis=0)
                best_sym = tl.min(
                    tl.where(cand_err == best_err[None, :], syms[:, None], SYMS),
                    axis=0,
                )

                picked = tl.sum(
                    tl.where(positions[None, :] == pos[:, None], best_sym[:, None], 0),
                    axis=0,
                )
                touched = tl.sum(
                    (positions[None, :] == pos[:, None]).to(tl.int32), axis=0
                )
                codes = tl.where(touched > 0, picked, codes)

        tl.store(out_ptr + row64 * BLOCK + positions, codes.to(tl.uint8))


def project_codes_cuda(
    w: torch.Tensor,
    init_codes: torch.Tensor,
    scales: torch.Tensor,
    lut: torch.Tensor,
    a0: int,
    a1: int,
    nbits: int,
    depth: int,
    block: int,
    sweeps: int = 1,
    d: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused :func:`hi2.cptc.project_codes` for GPU tensors.

    ``w`` is fp32 ``[N, block]`` (row stride arbitrary), ``init_codes`` uint8
    ``[N, block]`` (row stride arbitrary), ``scales`` fp16 ``[N]``, ``lut`` the
    mirror-expanded fp16 table ``[127]``, and ``d`` an optional fp32 ``[block]``
    weight. ``(a0, a1)`` are the serialized table-6 coefficients. Returns a fresh
    uint8 ``[N, block]``; inputs are never modified. ``N == 0`` returns an empty
    result without a launch.
    """
    nbits = int(nbits)
    depth = int(depth)
    block = int(block)
    a0 = int(a0)
    a1 = int(a1)
    sweeps = int(sweeps)

    if not isinstance(w, torch.Tensor) or w.dim() != 2:
        raise ValueError("project_codes_cuda expects w as a [N, block] tensor")
    n, cols = w.shape
    if cols != block:
        raise ValueError(
            f"project_codes_cuda expects [N, {block}] weights, got {tuple(w.shape)}"
        )
    if w.dtype != torch.float32:
        raise ValueError(f"project_codes_cuda expects fp32 weights, got {w.dtype}")
    if not isinstance(init_codes, torch.Tensor) or init_codes.shape != (n, block):
        raise ValueError("project_codes_cuda expects init_codes as [N, block]")
    if init_codes.dtype != torch.uint8:
        raise ValueError(
            f"project_codes_cuda expects uint8 init codes, got {init_codes.dtype}"
        )
    if not isinstance(scales, torch.Tensor) or scales.dim() != 1 or scales.numel() != n:
        raise ValueError("project_codes_cuda expects scales as [N]")
    if scales.dtype != torch.float16:
        raise ValueError(f"project_codes_cuda expects fp16 scales, got {scales.dtype}")
    if not isinstance(lut, torch.Tensor) or lut.dim() != 1:
        raise ValueError("project_codes_cuda expects lut as [127] fp16")
    if lut.numel() != _TABLE_ENTRIES:
        raise ValueError(
            f"project_codes_cuda expects {_TABLE_ENTRIES} lut entries, "
            f"got {lut.numel()}"
        )
    if lut.dtype != torch.float16:
        raise ValueError(f"project_codes_cuda expects an fp16 lut, got {lut.dtype}")
    if not 0 <= a0 <= 0xFFFF or not 0 <= a1 <= 0xFFFF:
        raise ValueError("project_codes_cuda needs 16-bit coefficients a0/a1")
    if d is not None:
        if not isinstance(d, torch.Tensor) or d.dim() != 1 or d.numel() != block:
            raise ValueError("project_codes_cuda expects an optional d as [block]")
        if d.dtype != torch.float32:
            raise ValueError(f"project_codes_cuda expects fp32 d, got {d.dtype}")
    span, period, npos = _geometry(nbits, depth, block)

    tensors = (init_codes, scales, lut) if d is None else (init_codes, scales, lut, d)
    for tensor in tensors:
        if tensor.device != w.device:
            raise ValueError("project_codes_cuda inputs must share one device")

    if n == 0:
        return torch.empty((0, block), dtype=torch.uint8, device=w.device)
    if tl is None:
        raise RuntimeError(
            f"CPTC projection kernel needs Triton ({_TRITON_IMPORT_ERROR})"
        )
    if not w.is_cuda:
        raise ValueError("project_codes_cuda needs CUDA/HIP tensors")

    out = torch.empty((n, block), dtype=torch.uint8, device=w.device)
    num_warps = 1 if block <= 32 else 2
    _project_kernel[(n,)](
        w,
        init_codes,
        scales,
        lut,
        w if d is None else d,
        out,
        w.stride(0),
        w.stride(1),
        init_codes.stride(0),
        init_codes.stride(1),
        scales.stride(0),
        lut.stride(0),
        0 if d is None else d.stride(0),
        a0,
        a1,
        max(1, sweeps),
        period,
        NBITS=nbits,
        DEPTH=depth,
        BLOCK=block,
        SPAN=span,
        PERIOD=period,
        NPOS=npos,
        SYMS=1 << nbits,
        HAS_D=d is not None,
        num_warps=num_warps,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return out
