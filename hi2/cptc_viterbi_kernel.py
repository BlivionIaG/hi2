# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Fused tail-biting Viterbi encoder for CPTC L16 blocks (HIP/Triton, gfx10/gfx11).

This is the GPU implementation of :func:`hi2.cptc.viterbi_codes` for the default
CPTC format (``depth=16``, ``nbits`` 2 or 4, ``block`` 16/32/64). It is a separate
module on purpose: :mod:`hi2.cptc` stays the portable reference and nothing here
imports it.

Algorithm (paper arXiv 2611.22019, Section 3.2) is reproduced exactly, not
approximated: the free pass over the rotated walk ``t = (B//2 - 1 - i) % B``
starts from ``a = 0`` and its traceback supplies the three ordered candidate
pins ``(q_free[0] >> K, q_free[B//2] >> K, best_free_end)``. Each pin is then
searched from ``a = +inf`` except ``a[pin] = 0`` to a *closed* path whose end
statistic equals the pin; the earliest pin whose closed cost is strictly lowest
wins. The returned SSE is measured on the actual closed code stream, through
the fp16 state values and the column weights ``d``; it is not the DP cost.

Layout. The reference materialises an ``[N, H, S]`` fp32 cost tensor per
trellis step (``H = 2**K``, ``S = 2**(L-K)``; 65536 values per row for L16).
That is both a large temporary and a host/device round trip per step. Here one
program owns one row and walks the S prefix statistics in 1D ``[RT]`` tiles;
the H high-symbol choices stream through registers and only the running min and
one uint8 backpointer per state survive. A double-buffered fp32 ``[2, S]`` row
scratch carries the statistic between trellis steps: the next step's
predecessor read touches states written by other waves of the workgroup, so the
program barriers between the store and the following load. The backpointer
scratch is one uint8 ``[B, S]`` slab per row and is reused by each of the four
passes; rows are launched in bounded chunks so scratch stays at a fixed few
tens of MB. There is no host synchronisation inside a pass and no
cross-program barrier: the whole DP and its traceback run in one launch.

The computed entry :func:`viterbi_scaled_codes_cuda` takes the serialized
mirror-expanded fp16 table ``[127]``, per-row fp16 block scales and the codec's
``(a0, a1)``. A tiny prologue kernel rounds each row's 127 decoder products
with the projection module's exact fp16 round-to-nearest-even helper (gfx1030's
implicit fp32->fp16 cast flushes subnormal results), and the fused DP and SSE
then read that per-row table. The ``[N, 2**L]`` expanded table is never
materialized. The generic values entry stays in place as the independent oracle
for the tests.

Ties follow the reference: the minimum over the high symbol uses the lowest
``h`` (strict ``<`` while scanning ``h`` upward) and the free end state is the
lowest prefix statistic with the minimal cost. Floating-point fusion is
disabled for these launches so every cost is rounded in the reference order
(subtract, square, multiply by ``d[t]``, add the predecessor); without that,
``v_fmac`` changes near-tie decisions relative to PyTorch's elementwise
reference.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .cptc_projection_kernel import _f32_to_f16_rn

_DEPTH = 16
_NBITS = (2, 4)
_BLOCKS = (16, 32, 64)
_LUT_ENTRIES = 127  # mirror-expanded Q6 table, Algorithm 1's full index range

# States per register tile. 2048 keeps the running min and backpointer tiles
# small at num_warps=8, so the 16K-prefix (K=2) laps never hold a whole-state
# tile and do not spill to scratch.
_TILE_VALUES = 2048
# Backpointer scratch budget per launch. The uint8 [B, S] slab dominates.
_SCRATCH_BYTES = 64 << 20
_NUM_WARPS = 8

# Launch options shared by every kernel here. ``enable_fp_fusion=False`` is the
# HIP backend's fp-contract switch; it makes ``base + sq * d`` round sq * d
# before the add, exactly like the elementwise reference.
_LAUNCH_OPTIONS = {"num_warps": _NUM_WARPS, "enable_fp_fusion": False}


@triton.jit
def _scale_lut_kernel(
    lut_ptr,
    scales_ptr,
    out_ptr,
    s_stride,
    LUT: tl.constexpr,
):
    """Round one row's fp16 decoder table once for the DP and the SSE.

    ``out[row, i] = fp32(fp16(scale[row] * lut[i]))`` for ``i < LUT``; that is
    exactly the value the expanded oracle table holds at every state whose
    decode index is ``i``, so the hot loops load a 127-entry row table instead
    of rounding one product per trellis state.
    """
    row = tl.program_id(0)
    row64 = row.to(tl.int64)
    r = tl.arange(0, 128)
    mask = r < LUT
    lutv = tl.load(lut_ptr + r, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scales_ptr + row64 * s_stride).to(tl.float32)
    v = _f32_to_f16_rn(scale * lutv).to(tl.float32)
    tl.store(out_ptr + row64 * LUT + r, v, mask=mask)


@triton.jit
def _viterbi_lap_kernel(
    w_ptr,
    d_ptr,
    values_ptr,
    pins_ptr,
    a_ptr,
    bp_ptr,
    codes_ptr,
    cost_ptr,
    w_stride,
    v_stride,
    scaled_ptr,
    t_stride,
    B: tl.constexpr,
    K: tl.constexpr,
    DEPTH: tl.constexpr,
    S: tl.constexpr,
    RT: tl.constexpr,
    LAP_COL: tl.constexpr,
    PIN_COL: tl.constexpr,
    FREE: tl.constexpr,
    SCALED: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
):
    """One trellis lap (DP + traceback) for one row.

    ``FREE`` runs the open pass with ``a = 0`` and writes the three pins to
    ``pins_ptr[row, 0:3]``; otherwise the lap starts from ``pins_ptr[row,
    PIN_COL]`` and writes the closed cost to ``cost_ptr[row, LAP_COL]`` and the
    closed code stream to ``codes_ptr[row, LAP_COL, :]``. Outer buffers are
    sized per launch chunk; all offsets are row-local.

    ``SCALED`` selects the value source: a load from the expanded fp32 oracle
    table, or the row's precomputed ``[127]`` scaled decoder table produced by
    :func:`_scale_lut_kernel`. ``A0``/``A1`` reconstruct the table index for
    the latter.
    """
    H: tl.constexpr = 1 << K
    J: tl.constexpr = S // H
    SPAN: tl.constexpr = (DEPTH + K - 1) // K
    QMASK: tl.constexpr = (1 << DEPTH) - 1
    CUT: tl.constexpr = B // 2

    row = tl.program_id(0)
    row64 = row.to(tl.int64)
    one = tl.arange(0, 1)
    off = tl.arange(0, RT)

    a_row = row64 * (2 * S)
    bp_row = row64 * (B * S)
    codes_row = row64 * (3 * B) + LAP_COL * B
    w_row = row64 * w_stride
    v_row = row64 * v_stride
    t_row = row64 * t_stride

    # Start statistic: all-zero for the free pass, a delta at the pin otherwise.
    if FREE:
        pin = tl.zeros([1], tl.int32)
        for j0 in tl.static_range(0, S, RT):
            tl.store(
                a_ptr + a_row + j0 + off, tl.zeros([RT], tl.float32)
            )
    else:
        pin = tl.load(pins_ptr + row64 * 3 + PIN_COL + one).to(tl.int32)
        for j0 in tl.static_range(0, S, RT):
            init = tl.where(j0 + off == pin, 0.0, float("inf"))
            tl.store(a_ptr + a_row + j0 + off, init)
    tl.debug_barrier()

    # Forward over the reference walk order. The trellis transition at step i
    # reads the predecessor statistic a[h*J + (r >> K)] and adds the exact
    # per-edge cost of window q = h*S + r at position t = walk(i).
    for i in range(0, B):
        t = (CUT - 1 + B - i) % B
        src = i & 1
        dst = 1 - src
        wt = tl.load(w_ptr + w_row + t)
        dt = tl.load(d_ptr + t)
        for j0 in tl.static_range(0, S, RT):
            best = tl.full([RT], float("inf"), tl.float32)
            bph = tl.zeros([RT], tl.int32)
            for h in tl.range(0, H, loop_unroll_factor=1):
                base = tl.load(
                    a_ptr
                    + a_row
                    + src * S
                    + h * J
                    + (j0 >> K)
                    + (off >> K)
                )
                qv = h * S + j0 + off
                if SCALED:
                    idx = ((A0 * qv) & 0xFFFF) >> 10
                    idx += ((A1 * qv) & 0xFFFF) >> 10
                    v = tl.load(scaled_ptr + t_row + idx)
                else:
                    v = tl.load(values_ptr + v_row + qv).to(tl.float32)
                diff = v - wt
                cand = base + diff * diff * dt
                take = cand < best
                best = tl.where(take, cand, best)
                bph = tl.where(take, h, bph)
            tl.store(a_ptr + a_row + dst * S + j0 + off, best)
            tl.store(
                bp_ptr + bp_row + i * S + j0 + off, bph.to(tl.uint8)
            )
        tl.debug_barrier()

    # B is even for every supported block, so the last step wrote buffer 0.
    final = B & 1

    if FREE:
        # Lowest prefix statistic with the minimal free end cost.
        best_m = tl.full([1], float("inf"), tl.float32)
        # ``torch.argmin`` returns the first index when every cost is equal, so
        # an all-inf tile (non-finite inputs) must end at index 0, not S.
        best_i = tl.zeros([1], tl.int32)
        for j0 in tl.static_range(0, S, RT):
            ridx = j0 + off
            av = tl.load(a_ptr + a_row + final * S + ridx)
            m = tl.min(av)
            idx = tl.min(tl.where(av == m, ridx, S))
            take = m < best_m
            best_m = tl.where(take, m, best_m)
            best_i = tl.where(take, idx, best_i)
        end_r = best_i
    else:
        closed = tl.load(a_ptr + a_row + final * S + pin + one)
        tl.store(cost_ptr + row64 * 3 + PIN_COL + one, closed)

    # Backtrace. ``bp[i]`` is the high-symbol choice at step i, so the
    # predecessor statistic is h*J + (r >> K); L < 2K is irrelevant here
    # because the shift is applied to the statistic, not to the window.
    if FREE:
        r_cur = end_r
    else:
        r_cur = pin
    for ii in range(0, B - 1):
        i = B - 1 - ii
        t = (CUT - 1 + B - i) % B
        tl.store(codes_ptr + codes_row + t + one, (r_cur & (H - 1)).to(tl.uint8))
        bpv = tl.load(bp_ptr + bp_row + i * S + r_cur).to(tl.int32)
        r_cur = bpv * J + (r_cur >> K)
    t0 = CUT - 1
    tl.store(codes_ptr + codes_row + t0 + one, (r_cur & (H - 1)).to(tl.uint8))

    if FREE:
        # Candidate pins, in the reference's evaluation order: window 0,
        # mid-block window, best free end statistic.
        tl.debug_barrier()
        q0 = tl.zeros([1], tl.int32)
        for j in tl.static_range(0, SPAN):
            c = tl.load(codes_ptr + codes_row + j + one).to(tl.int32)
            q0 = q0 | (c << (j * K))
        q0 = q0 & QMASK
        tl.store(pins_ptr + row64 * 3 + 0 + one, q0 >> K)

        q1 = tl.zeros([1], tl.int32)
        for j in tl.static_range(0, SPAN):
            c = tl.load(codes_ptr + codes_row + CUT + j + one).to(tl.int32)
            q1 = q1 | (c << (j * K))
        q1 = q1 & QMASK
        tl.store(pins_ptr + row64 * 3 + 1 + one, q1 >> K)
        tl.store(pins_ptr + row64 * 3 + 2 + one, end_r)


@triton.jit
def _select_sse_kernel(
    w_ptr,
    d_ptr,
    values_ptr,
    cand_codes_ptr,
    cand_costs_ptr,
    out_codes_ptr,
    sse_ptr,
    w_stride,
    v_stride,
    scaled_ptr,
    t_stride,
    B: tl.constexpr,
    K: tl.constexpr,
    DEPTH: tl.constexpr,
    SCALED: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
):
    """Pick the earliest lowest closed candidate and measure its closed SSE.

    The selection is the reference's strict ``closed < best`` in pin order, so
    ties keep the earlier candidate. The SSE is the reconstructed cyclic
    stream's exact fp16 decode error, weighted by ``d``.
    """
    SPAN: tl.constexpr = (DEPTH + K - 1) // K
    QMASK: tl.constexpr = (1 << DEPTH) - 1

    row = tl.program_id(0)
    row64 = row.to(tl.int64)
    one = tl.arange(0, 1)
    tt = tl.arange(0, B)
    base = row64 * (3 * B)
    t_row = row64 * t_stride

    best = tl.load(cand_costs_ptr + row64 * 3 + 0 + one)
    win = tl.zeros([1], tl.int32)
    c1 = tl.load(cand_costs_ptr + row64 * 3 + 1 + one)
    take = c1 < best
    best = tl.where(take, c1, best)
    win = tl.where(take, 1, win)
    c2 = tl.load(cand_costs_ptr + row64 * 3 + 2 + one)
    take = c2 < best
    win = tl.where(take, 2, win)

    codes = tl.load(cand_codes_ptr + base + win * B + tt)
    tl.store(out_codes_ptr + row64 * B + tt, codes)

    q = tl.zeros([B], tl.int32)
    for j in tl.static_range(0, SPAN):
        cj = tl.load(
            cand_codes_ptr + base + win * B + ((tt + j) % B)
        ).to(tl.int32)
        q = q | (cj << (j * K))
    q = q & QMASK

    if SCALED:
        idx = ((A0 * q) & 0xFFFF) >> 10
        idx += ((A1 * q) & 0xFFFF) >> 10
        v = tl.load(scaled_ptr + t_row + idx)
    else:
        # The oracle values are contractually fp16-exact; round explicitly so
        # subnormal decoder outputs survive (the implicit gfx1030 cast flushes
        # them), matching the reference's host-side ``to(fp16)``.
        v = tl.load(values_ptr + row64 * v_stride + q).to(tl.float32)
        v = _f32_to_f16_rn(v).to(tl.float32)
    wt = tl.load(w_ptr + row64 * w_stride + tt)
    dt = tl.load(d_ptr + tt)
    diff = v - wt
    tl.store(sse_ptr + row64, tl.sum(dt * diff * diff))


def _check_geometry(nbits, depth, block):
    if depth != _DEPTH:
        raise NotImplementedError(
            f"cptc_viterbi_kernel implements L={_DEPTH} only, got L={depth}; "
            "use hi2.cptc.viterbi_codes for other depths"
        )
    if nbits not in _NBITS:
        raise ValueError(
            f"cptc_viterbi_kernel supports K in {_NBITS}, got {nbits}"
        )
    if block not in _BLOCKS:
        raise ValueError(
            f"cptc_viterbi_kernel supports block in {_BLOCKS}, got {block}"
        )


def _check_w_d(w, d, block):
    if not isinstance(w, torch.Tensor):
        raise TypeError("w must be a torch.Tensor")
    if not isinstance(d, torch.Tensor):
        raise TypeError("d must be a torch.Tensor")
    if w.dim() != 2 or w.shape[1] != block:
        raise ValueError(f"w must be [N, {block}], got {tuple(w.shape)}")
    if w.dtype != torch.float32:
        raise TypeError(f"w must be float32, got {w.dtype}")
    if w.stride(1) != 1:
        raise ValueError("w must have stride(1) == 1")
    if d.dim() != 1 or d.shape[0] != block:
        raise ValueError(f"d must be [{block}], got {tuple(d.shape)}")
    if d.dtype != torch.float32:
        raise TypeError(f"d must be float32, got {d.dtype}")


def _check_cuda(w, d, extra):
    if not w.is_cuda:
        raise ValueError(
            "cptc_viterbi_kernel requires CUDA/HIP tensors; "
            "use hi2.cptc.viterbi_codes on CPU"
        )
    for tensor in (d,) + tuple(extra):
        if tensor.device != w.device:
            raise ValueError("w, d and value inputs must live on the same device")


def _validate(w, d, values, nbits, depth, block):
    _check_geometry(nbits, depth, block)
    _check_w_d(w, d, block)
    if not isinstance(values, torch.Tensor):
        raise TypeError("values must be a torch.Tensor")
    if values.dim() not in (1, 2):
        raise ValueError(
            f"values must be [2**L] or [N, 2**L], got {tuple(values.shape)}"
        )
    if values.shape[-1] != 1 << depth:
        raise ValueError(
            f"values must have {1 << depth} entries, got {tuple(values.shape)}"
        )
    if values.dtype not in (torch.float32, torch.float16):
        raise TypeError(
            f"values must be float32 or float16, got {values.dtype}"
        )
    if values.dim() == 2 and values.shape[0] != w.shape[0]:
        raise ValueError(
            f"values rows {values.shape[0]} do not match w rows {w.shape[0]}"
        )
    _check_cuda(w, d, (values,))


def _validate_scaled(w, d, lut, scales, a0, a1, nbits, depth, block):
    _check_geometry(nbits, depth, block)
    _check_w_d(w, d, block)
    if not isinstance(lut, torch.Tensor):
        raise TypeError("lut must be a torch.Tensor")
    if lut.dim() != 1 or lut.numel() != _LUT_ENTRIES:
        raise ValueError(
            f"lut must be the mirror-expanded fp16 [{_LUT_ENTRIES}] table, "
            f"got {tuple(lut.shape)}"
        )
    if lut.dtype != torch.float16:
        raise TypeError(f"lut must be float16, got {lut.dtype}")
    if not isinstance(scales, torch.Tensor):
        raise TypeError("scales must be a torch.Tensor")
    if scales.dim() != 1 or scales.shape[0] != w.shape[0]:
        raise ValueError(
            f"scales must be [{w.shape[0]}] fp16 row scales, "
            f"got {tuple(scales.shape)}"
        )
    if scales.dtype != torch.float16:
        raise TypeError(f"scales must be float16, got {scales.dtype}")
    if not 0 <= a0 <= 0xFFFF or not 0 <= a1 <= 0xFFFF:
        raise ValueError("a0/a1 must be 16-bit CPTC coefficients")
    if not (a0 | a1) & 1:
        raise ValueError("a0/a1 do not encode the L16 depth")
    _check_cuda(w, d, (lut, scales))


@torch.no_grad()
def viterbi_codes_cuda(w, d, values, nbits, depth, block):
    """Closed tail-biting Viterbi codes for one CPTC block, on the GPU.

    ``w`` is float32 ``[N, block]`` with ``stride(1) == 1`` (the row stride may
    exceed ``block``); ``d`` is float32 ``[block]``; ``values`` is the exact
    block-scaled CPTC state table as float32 or float16, either shared
    ``[2**L]`` or per row ``[N, 2**L]``. Returns ``(codes uint8 [N, block],
    sse float32 [N])``. Only L16 with K in {2, 4} and block in {16, 32, 64} is
    implemented; other depths raise ``NotImplementedError`` so the caller can
    use the reference. ``N == 0`` returns empty tensors without a launch.
    """
    nbits = int(nbits)
    depth = int(depth)
    block = int(block)
    _validate(w, d, values, nbits, depth, block)
    return _codes_impl(
        w, d, values, None, None, 0, 0, nbits, depth, block, False
    )


@torch.no_grad()
def viterbi_scaled_codes_cuda(w, d, lut, scales, a0, a1, nbits, depth, block):
    """Closed tail-biting Viterbi codes from the serialized decoder table.

    ``w`` is float32 ``[N, block]`` with ``stride(1) == 1`` (the row stride may
    exceed ``block``); ``d`` is float32 ``[block]``; ``lut`` is the serialized
    mirror-expanded fp16 table ``[127]`` (``index_table_value(arange(127),
    codec.table, codec.table_scale)``); ``scales`` is fp16 ``[N]`` and
    ``(a0, a1)`` are the codec's Table-6 coefficients. Every 16-bit window
    ``q`` decodes to ``fp16(scale * lut[((a0*q)&0xFFFF)>>10 +
    ((a1*q)&0xFFFF)>>10])``, rounded once per row into a bounded ``[N, 127]``
    fp32 scratch with the projection module's exact fp16 converter; the fused
    DP and SSE read that scratch, which holds the same values as the generic
    oracle. Returns ``(codes uint8 [N, block], sse float32 [N])``; the
    ``[N, 2**L]`` expanded table is never allocated. Only L16 with K in {2, 4},
    block in {16, 32, 64}; ``N == 0`` returns empty tensors without a launch.
    """
    nbits = int(nbits)
    depth = int(depth)
    block = int(block)
    a0 = int(a0)
    a1 = int(a1)
    _validate_scaled(w, d, lut, scales, a0, a1, nbits, depth, block)
    return _codes_impl(
        w, d, None, lut, scales, a0, a1, nbits, depth, block, True
    )


def _codes_impl(w, d, values, lut, scales, a0, a1, nbits, depth, block, scaled):
    n = int(w.shape[0])
    codes = torch.empty((n, block), dtype=torch.uint8, device=w.device)
    sse = torch.empty((n,), dtype=torch.float32, device=w.device)
    if n == 0:
        return codes, sse

    if d.stride(0) != 1:
        d = d.contiguous()
    if scaled:
        if lut.stride(0) != 1:
            lut = lut.contiguous()
        values_ptr = lut
        v_stride = 0
        s_stride = int(scales.stride(0))
    else:
        if values.stride(-1) != 1:
            values = values.contiguous()
        values_2d = values.dim() == 2
        v_stride = int(values.stride(0)) if values_2d else 0
        values_ptr = values
        s_stride = 0

    s_len = 1 << (depth - nbits)
    rt = min(s_len, _TILE_VALUES)

    # Bounded scratch: rows per launch are capped by the uint8 [B, S] slab.
    chunk = max(1, _SCRATCH_BYTES // (block * s_len))
    chunk = min(chunk, n)

    a_buf = torch.empty((chunk, 2, s_len), dtype=torch.float32, device=w.device)
    bp_buf = torch.empty(
        (chunk, block, s_len), dtype=torch.uint8, device=w.device
    )
    pins = torch.empty((chunk, 3), dtype=torch.int32, device=w.device)
    cand_codes = torch.empty(
        (chunk, 3, block), dtype=torch.uint8, device=w.device
    )
    cand_costs = torch.empty((chunk, 3), dtype=torch.float32, device=w.device)
    if scaled:
        # 127 fp32 values per row: the whole per-row decoder table.
        scaled_buf = torch.empty(
            (chunk, _LUT_ENTRIES), dtype=torch.float32, device=w.device
        )

    for start in range(0, n, chunk):
        rows = min(chunk, n - start)
        w_c = w[start : start + rows]
        if scaled:
            v_c = values_ptr
            scaled_c = scaled_buf[:rows]
            _scale_lut_kernel[(rows,)](
                lut,
                scales[start : start + rows],
                scaled_c,
                s_stride,
                LUT=_LUT_ENTRIES,
                **_LAUNCH_OPTIONS,
            )
            t_stride = _LUT_ENTRIES
        else:
            v_c = values[start : start + rows] if values_2d else values
            # The unused computed specialization still receives a pointer; the
            # shared strides make both value sources use one kernel signature.
            scaled_c = v_c
            t_stride = 0
        codes_c = codes[start : start + rows]
        sse_c = sse[start : start + rows]
        a_c = a_buf[:rows]
        bp_c = bp_buf[:rows]
        pins_c = pins[:rows]
        cand_c = cand_codes[:rows]
        costs_c = cand_costs[:rows]
        w_stride = w_c.stride(0)

        args = (
            w_c,
            d,
            v_c,
            pins_c,
            a_c,
            bp_c,
            cand_c,
            costs_c,
            w_stride,
            v_stride,
            scaled_c,
            t_stride,
        )
        _viterbi_lap_kernel[(rows,)](
            *args,
            B=block,
            K=nbits,
            DEPTH=depth,
            S=s_len,
            RT=rt,
            LAP_COL=0,
            PIN_COL=0,
            FREE=True,
            SCALED=scaled,
            A0=a0,
            A1=a1,
            **_LAUNCH_OPTIONS,
        )
        for lap in range(3):
            _viterbi_lap_kernel[(rows,)](
                *args,
                B=block,
                K=nbits,
                DEPTH=depth,
                S=s_len,
                RT=rt,
                LAP_COL=lap,
                PIN_COL=lap,
                FREE=False,
                SCALED=scaled,
                A0=a0,
                A1=a1,
                **_LAUNCH_OPTIONS,
            )
        _select_sse_kernel[(rows,)](
            w_c,
            d,
            v_c,
            cand_c,
            costs_c,
            codes_c,
            sse_c,
            w_stride,
            v_stride,
            scaled_c,
            t_stride,
            B=block,
            K=nbits,
            DEPTH=depth,
            SCALED=scaled,
            A0=a0,
            A1=a1,
            **_LAUNCH_OPTIONS,
        )
    return codes, sse
