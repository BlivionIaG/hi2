# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Computed Permutation Trellis Coding (CPTC), arXiv:2611.22019 implementation.

This is a clean-room implementation of the paper's decoder (Algorithm 1/2) and
encoder (Section 3.2, Appendix D.3); no public reference implementation exists.
Every stored block of ``B`` weights is an independent tail-biting bitshift trellis
of depth ``L`` with ``K`` bits per value. The state is the 16-bit window
``q_t = ((q_{t+1} << K) | c_t) & (2**L - 1)`` over ``D = ceil(L/K)`` consecutive
codes; ``c_t`` is the ``t``-th little-endian ``K``-bit code of the block, wrapping
at the block boundary (Algorithm 2).

Decoder (Algorithm 1, ``Q = 6``, ``s = 0``)::

    i  = ((a0 * q) & 0xFFFF) >> (16 - Q) + ((a1 * q) & 0xFFFF) >> (16 - Q)
    v  = table_scale * (table[i] if i < 2**Q else -table[2**(Q+1) - 2 - i])

with a mirror-symmetric 64-entry INT8 table (the paper's non-E2M5 variant, Table 16:
within 0.001 effective bits of E2M5 up to K=4). Exact arithmetic rule, which the
reference decoder, the encoder and the consumer must share:

* ``table_value`` is ``fp16(int8(table[i]) * table_scale)`` (one rounding; the
  product is exact in fp32 because ``table_scale`` is fp16);
* the reconstructed weight is ``fp16(block_scale * table_value)`` (one rounding).

Coefficients are the paper's Table 6 pairs per ``(L, K)``; the depth is metadata and
the table is serialized, so an independent decoder needs no coefficient search. The
coefficient pair's index histogram over the uniform state must be mirror-symmetric
for the folded table to be exact; :func:`validate_codec` enforces it.

The encoder runs a two-pass tail-biting Viterbi search (cut at ``B/2``: a free-start
pass fixes the seam state, a pinned second pass backtraces) in ``O(B * 2**L)`` time
and ``O(B * 2**(L-K))`` backtrace bytes per row. :func:`refine_codes` is a bounded,
deterministic coordinate-descent projector used by QAD (and optionally by PTQ):
changing one code changes exactly ``D`` decoded windows, so codes of one phase
``t mod D`` have disjoint update regions.
"""

import base64
import hashlib
import json
import math
import struct
from dataclasses import dataclass, field
from functools import lru_cache

import torch

from .format import decode_params, fwht_groups, pack_codes, unpack_codes

TAP_Q = 6
CPTC_ROW_OVERHEAD_BITS = 32 + 32 + 8 + 8
STARTS = 4  # closed tail-biting Viterbi candidate wrap-seam statistics
TABLE_ENTRIES = 2**TAP_Q  # stored (lower) half, one byte each
TABLE_RANGE = 2 ** (TAP_Q + 1) - 1  # full mirrored table, Algorithm 1's index range
STATE_BITS = 16
BLOCK_VALUES = (16, 32, 64)

# Paper Table 6, rows L >= 7: (a0, a1) per (L, K), in hex. Rows 1..6 use one pair
# for every K (the paper says so explicitly); the K=4 column holds it.
_COEFF_TABLE6 = {
    1: {4: (0x0000, 0x8000)},
    2: {4: (0x0000, 0xC000)},
    3: {4: (0x0000, 0xA000)},
    4: {4: (0x0000, 0x5000)},
    5: {4: (0x0000, 0x2800)},
    6: {4: (0x0000, 0xA400)},
    7: {
        1: (0x7A00, 0x7A00),
        2: (0x3A00, 0x3A00),
        3: (0xA200, 0xA200),
        4: (0x6200, 0xC400),
        5: (0x2E00, 0x5C00),
        6: (0x0200, 0x0400),
    },
    8: {
        1: (0x8700, 0x8700),
        2: (0x7A00, 0xBD00),
        3: (0x6100, 0x6100),
        4: (0x1B00, 0x3600),
        5: (0x7700, 0xDC00),
        6: (0x0400, 0x8100),
    },
    9: {
        1: (0x8580, 0x8580),
        2: (0x4280, 0x4280),
        3: (0xE180, 0xE180),
        4: (0x6100, 0xB080),
        5: (0x6500, 0x9280),
        6: (0xCC80, 0xD900),
    },
    10: {
        1: (0x8940, 0x8940),
        2: (0x41C0, 0x41C0),
        3: (0x8A80, 0x9540),
        4: (0x39C0, 0x4E00),
        5: (0x1A40, 0xA400),
        6: (0x1380, 0x99C0),
    },
    11: {
        1: (0x7920, 0x7920),
        2: (0x41E0, 0x41E0),
        3: (0x7480, 0xAD20),
        4: (0x5D00, 0xB3A0),
        5: (0x6040, 0xA820),
        6: (0x1400, 0xC0A0),
    },
    12: {
        1: (0x5F60, 0x6BB0),
        2: (0x77D0, 0xC7A0),
        3: (0x7DA0, 0xE2D0),
        4: (0x81D0, 0xEE80),
        5: (0x6810, 0xA020),
        6: (0x7020, 0x9410),
    },
    13: {
        1: (0x61C8, 0x6F20),
        2: (0x3490, 0x8C48),
        3: (0x8320, 0xDCC8),
        4: (0x51C0, 0xDE38),
        5: (0xD820, 0xF008),
        6: (0x3008, 0x7C10),
    },
    14: {
        1: (0x6BBC, 0x7178),
        2: (0x7708, 0xC884),
        3: (0x74B0, 0xAB2C),
        4: (0x6D20, 0x82A4),
        5: (0x07F0, 0xCFFC),
        6: (0x5008, 0xE404),
    },
    15: {
        1: (0x6B6E, 0x73DC),
        2: (0x792E, 0xC75C),
        3: (0x1D48, 0x8252),
        4: (0x11CA, 0x5E50),
        5: (0x9808, 0xA002),
        6: (0x5A84, 0x9DC2),
    },
    16: {
        1: (0x9097, 0x97AE),
        2: (0x7AE6, 0xC8B3),
        3: (0x7474, 0xAB1D),
        4: (0x6A78, 0x85CF),
        5: (0x1C04, 0x3C01),
        6: (0x0AEE, 0xD337),
    },
}


def coefficients(depth: int, nbits: int) -> tuple[int, int]:
    """The paper's ``(a0, a1)`` for a trellis depth and bit width (Table 6)."""
    if depth not in _COEFF_TABLE6:
        raise ValueError(f"CPTC depth {depth} outside 1..16")
    row = _COEFF_TABLE6[depth]
    if depth <= 6:
        return row[4]
    if nbits not in row:
        raise ValueError(f"CPTC supports 1..6 bits per value, got {nbits}")
    return row[nbits]


def depth_of(a0: int, a1: int) -> int:
    """``L = 16 - ctz(a0 | a1)`` (paper Section 3.4)."""
    orv = a0 | a1
    if orv == 0:
        raise ValueError("CPTC coefficients are both zero")
    return STATE_BITS - ((orv & -orv).bit_length() - 1)


# --------------------------------------------------------------- decoder core


def decode_index(q: torch.Tensor, a0: int, a1: int, tap_q: int = TAP_Q):
    """Algorithm 1 line 2-4: the 7-bit table index for one 16-bit window."""
    x0 = ((a0 * q) & 0xFFFF) >> (STATE_BITS - tap_q)
    x1 = ((a1 * q) & 0xFFFF) >> (STATE_BITS - tap_q)
    return x0 + x1


def index_table_value(index: torch.Tensor, table: torch.Tensor, scale):
    """Signed INT8 table lookup through the mirror fold, as fp16.

    ``table`` holds the lower ``2**Q`` entries; index ``i >= 2**Q`` reads
    ``-table[2**(Q+1) - 2 - i]`` (Algorithm 1 lines 5-9).
    """
    half = table.numel()
    lower = index < half
    pos = index.clamp(max=half - 1)
    neg = (2 * half - 2 - index).clamp(min=0, max=half - 1)
    i8 = torch.where(lower, table[pos], -table[neg])
    return (i8.float() * float(scale)).to(torch.float16)


@lru_cache(maxsize=128)
def _index_histogram(depth: int, a0: int, a1: int, tap_q: int) -> torch.Tensor:
    q = torch.arange(1 << depth, dtype=torch.int64, device="cpu")
    idx = decode_index(q, a0, a1, tap_q)
    return torch.bincount(idx, minlength=2 ** (tap_q + 1) - 1)


def index_histogram(depth: int, nbits: int, tap_q: int = TAP_Q) -> torch.Tensor:
    """Exact index counts for the published coefficient pair over uniform states."""
    a0, a1 = coefficients(depth, nbits)
    return _index_histogram(depth, a0, a1, tap_q).clone()


def validate_codec(codec: "CptcCodec") -> None:
    """Fail closed on metadata that would make an independent decoder disagree."""
    if not 1 <= codec.depth <= STATE_BITS:
        raise ValueError(f"CPTC depth {codec.depth} outside 1..16")
    if codec.tap_q != TAP_Q:
        raise ValueError(f"CPTC tap width must be {TAP_Q}, got {codec.tap_q}")
    if codec.shift != 0:
        raise ValueError("CPTC shift must be 0 for the serialized Q6 mirror format")
    if depth_of(codec.a0, codec.a1) != codec.depth:
        raise ValueError("CPTC coefficients do not encode the declared depth")
    if codec.depth < 2 * codec.nbits:
        raise ValueError(
            f"CPTC depth {codec.depth} < 2*K ({2 * codec.nbits}); the compressed "
            "Viterbi recurrence needs two full codes in the state"
        )
    if codec.table.numel() != TABLE_ENTRIES or codec.table.dtype != torch.int8:
        raise ValueError("CPTC table must be 64 int8 entries (Q=6)")
    if bool((codec.table == -128).any()):
        raise ValueError("CPTC mirrored int8 table cannot contain -128")
    if codec.table_scale.numel() != 1 or codec.table_scale.dtype != torch.float16:
        raise ValueError("CPTC table scale must be one fp16 value")
    if not bool(torch.isfinite(codec.table_scale)) or float(codec.table_scale) <= 0:
        raise ValueError("CPTC table scale must be finite and positive")
    values = (codec.table.float() * float(codec.table_scale)).half()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("CPTC decoded table must be finite in fp16")
    hist = _index_histogram(codec.depth, codec.a0, codec.a1, codec.tap_q)
    if bool((hist < 0).any()):
        raise ValueError("CPTC index histogram is negative")
    if bool((hist == 0).all()):
        raise ValueError("CPTC coefficient pair never reaches the table")
    # The fold reads the upper half only for i >= 2**Q; it is exact when the pair's
    # index histogram is mirror-symmetric. A pair whose indices never reach the upper
    # half (the paper's L <= 6 scalar embeddings) uses the lower half directly.
    half = 2**codec.tap_q
    if bool((hist[half:] != 0).any()) and not bool(
        (hist == hist.flip(0).clone()).all()
    ):
        raise ValueError(
            "CPTC coefficient pair has a non-mirror-symmetric index histogram"
        )


# --------------------------------------------------------------------- codec


@dataclass
class CptcCodec:
    """Per-bit-width CPTC parameters: depth, coefficients, mirrored INT8 table."""

    nbits: int
    depth: int
    a0: int
    a1: int
    table: torch.Tensor  # int8 [2**Q]
    table_scale: torch.Tensor  # fp16 scalar
    tap_q: int = TAP_Q
    shift: int = 0

    def __post_init__(self):
        self.table = self.table.to(torch.int8).reshape(-1)
        self.table_scale = self.table_scale.to(torch.float16).reshape(())
        validate_codec(self)

    def state_values(self, device=None) -> torch.Tensor:
        """fp16 output of every ``2**L`` state (exact decoder table, one rounding)."""
        q = torch.arange(1 << self.depth, dtype=torch.int64, device=device)
        idx = decode_index(q, self.a0, self.a1, self.tap_q)
        return index_table_value(
            idx, self.table.to(device), float(self.table_scale)
        )


_CODEC_CACHE: dict[tuple, CptcCodec] = {}


def analytic_table(depth: int, nbits: int, tap_q: int = TAP_Q) -> torch.Tensor:
    """Equation (2) starting table for a standard-normal target, as float64.

    Cell ``i`` owns the probability interval of the exact uniform-state index law
    and holds the conditional mean of ``N(0, 1)`` over it.
    """
    hist = index_histogram(depth, nbits, tap_q).double()
    cdf = torch.cumsum(hist, 0) / hist.sum()
    lo = torch.cat([torch.zeros(1, dtype=cdf.dtype), cdf[:-1]])
    # Normal CDF is not available for tensor quantiles without scipy; erfinv is.
    def inv(p):
        p = p.clamp(1e-6, 1 - 1e-6)
        return math.sqrt(2.0) * torch.erfinv(2 * p - 1)

    z_lo, z_hi = inv(lo), inv(cdf)
    phi = lambda z: torch.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)  # noqa: E731
    mass = cdf - lo
    mean = torch.where(mass > 0, (phi(z_lo) - phi(z_hi)) / mass.clamp_min(1e-30), 0.0)
    # Empty cells (possible for small depths) take the interval midpoint quantile.
    mid = inv((lo + cdf) / 2)
    return torch.where(mass > 0, mean, mid).clamp(-8.0, 8.0)


def _fit_int8(full: torch.Tensor, sample_x=None, sample_idx=None) -> tuple:
    """Snap a symmetric FP64 table to a mirror-folded INT8 table + fp16 scale.

    With samples, the storage scale is chosen from a small deterministic grid by the
    reconstruction MSE of the samples assigned to the table cells; without samples,
    the max-abs rule is used.
    """
    half = full[:TABLE_ENTRIES].clone()
    half[TABLE_ENTRIES - 1] = full[TABLE_ENTRIES - 1]
    maxabs = half.abs().max().clamp_min(1e-30)
    best = None
    for factor in (126.0, 127.0, 128.0):
        scale = (maxabs / factor).to(torch.float16)
        s = float(scale)
        if s <= 0 or not math.isfinite(s):
            continue
        i8 = torch.round(half / s).clamp(-127, 127).to(torch.int8)
        decoded = index_table_value(
            torch.arange(TABLE_ENTRIES), i8, s
        ).double()
        if sample_x is None:
            err = ((decoded - half) ** 2).mean()
        else:
            vals = index_table_value(sample_idx.reshape(-1), i8, s)
            err = ((sample_x.reshape(-1) - vals.double()) ** 2).mean()
        key = (float(err), factor)
        if best is None or key < best[0]:
            best = (key, i8, scale)
    if best is None:
        raise ValueError("CPTC table fit failed")
    return best[1], best[2]


def _cptc_train_codes(x, codec, block, batch=64):
    """Scale-1 tail-biting Viterbi codes for training rows ``[N, block]``."""
    d = torch.ones(block)
    values = codec.state_values().float()
    out = torch.empty_like(x, dtype=torch.uint8)
    for s in range(0, x.shape[0], batch):
        e = min(x.shape[0], s + batch)
        codes, _ = viterbi_codes(
            x[s:e], d, values, codec.nbits, codec.depth, block
        )
        out[s:e] = codes
    return out


def generate_table(
    depth: int,
    nbits: int,
    tap_q: int = TAP_Q,
    rounds: int = 0,
    samples: int = 1 << 18,
    block: int = 64,
    seed: int = 0,
) -> CptcCodec:
    """A standard-normal CPTC codec (paper Section 3.3 / Appendix A.2).

    ``rounds=0`` uses Equation (2) only. Training follows the paper's protocol at
    unit scale: tail-biting Viterbi assignment, mirrored-cell pooled means, damped
    rounds, and held-out selection. ``samples`` values are drawn per round.
    """
    if depth < 2 * nbits:
        raise ValueError("CPTC table generation needs depth >= 2*K")
    a0, a1 = coefficients(depth, nbits)
    full = analytic_table(depth, nbits, tap_q)
    # Appendix A.2 starting width (absorbed by the block scale after training).
    n = depth - nbits + 1
    full = full * math.sqrt((1.0 + 2.0 / n) * (1.0 - 4.0 ** (-nbits)))
    codec = CptcCodec(nbits, depth, a0, a1, *_fit_int8(full))
    if rounds <= 0:
        return codec
    gen = torch.Generator().manual_seed(int(seed) + 1_000_003 * depth + nbits)
    n_rows = max(1, int(samples) // block)
    hold_gen = torch.Generator().manual_seed(int(seed) + 7_919)
    x_hold = torch.randn(min(n_rows, 1024), block, generator=hold_gen)
    d = torch.ones(block)
    best = (float("inf"), None)
    for r in range(int(rounds)):
        x = torch.randn(n_rows, block, generator=gen)
        codes = _cptc_train_codes(x, codec, block)
        idx = window_indices(codes, nbits, depth, block)
        cnt = torch.bincount(idx.reshape(-1), minlength=TABLE_RANGE).double()
        tot = torch.bincount(
            idx.reshape(-1), weights=x.reshape(-1).double(), minlength=TABLE_RANGE
        )
        pooled = full.clone()
        for i in range(TABLE_ENTRIES):
            j = TABLE_RANGE - 1 - i
            ci, cj = float(cnt[i]), float(cnt[j])
            if i != j and ci + cj > 0:
                pooled[i] = (float(tot[i]) - float(tot[j])) / (ci + cj)
            elif ci > 0:
                pooled[i] = float(tot[i]) / ci
            elif cj > 0:
                pooled[i] = -float(tot[j]) / cj
        # 30 plain Lloyd rounds then 20 damped ones; identical update equations.
        damp = 1.0 if r < 30 else 0.5
        full = (1.0 - damp) * full + damp * pooled
        table, scale = _fit_int8(full)
        codec = CptcCodec(nbits, depth, a0, a1, table, scale)
        hc = _cptc_train_codes(x_hold, codec, block)
        err, _ = _block_sse(
            x_hold, d, codec.state_values(), hc, codec, block, 1.0
        )
        mean = float(err.mean()) / block
        if mean < best[0]:
            best = (mean, codec)
    return best[1]


def get_codec(
    depth: int,
    nbits: int,
    tap_q: int = TAP_Q,
    rounds: int = 0,
    samples: int = 1 << 16,
    seed: int = 0,
) -> CptcCodec:
    """Process-wide deterministic codec cache (tables are small; generation is not)."""
    key = (depth, nbits, tap_q, rounds, samples, seed)
    codec = _CODEC_CACHE.get(key)
    if codec is None:
        codec = generate_table(depth, nbits, tap_q, rounds, samples, seed=seed)
        _CODEC_CACHE[key] = codec
    return codec


# ------------------------------------------------------------ stream helpers


def check_block(nbits: int, block: int) -> None:
    if block < 2:
        raise ValueError(f"CPTC block must be at least 2 values, got {block}")
    if nbits not in (1, 2, 3, 4, 5, 6):
        raise ValueError(f"CPTC supports 1..6 bits per value, got {nbits}")
def check_format_block(nbits: int, block: int) -> None:
    """Serialized CPTC blocks are the ladder's B: 16, 32 or 64 (TP64-sliceable)."""
    if block not in BLOCK_VALUES:
        raise ValueError(f"CPTC checkpoint block must be one of {BLOCK_VALUES}")
    if (block * nbits) % 32:
        raise ValueError(
            f"CPTC block {block} x {nbits} bits is not a whole number of 32-bit words"
        )


def window_indices(codes: torch.Tensor, nbits: int, depth: int, block: int):
    """Little-endian tail-biting windows ``q_t`` of a block ``[N, block]`` of codes."""
    check_block(nbits, block)
    d = (depth + nbits - 1) // nbits
    if d > block:
        raise ValueError(f"trellis depth {depth} does not fit block {block}")
    q = torch.zeros(codes.shape, dtype=torch.int64, device=codes.device)
    c = codes.to(torch.int64)
    for i in range(d):
        q |= torch.roll(c, -i, dims=-1) << (i * nbits)
    return q & ((1 << depth) - 1)


def window_indices_blocks(codes: torch.Tensor, nbits: int, depth: int, block: int):
    """``window_indices`` for every ``block`` chunk of ``[N, K]`` without cross-block roll."""
    check_block(nbits, block)
    n, k = codes.shape
    if k % block:
        raise ValueError(f"K={k} is not a multiple of CPTC block {block}")
    d = (depth + nbits - 1) // nbits
    if d > block:
        raise ValueError(f"trellis depth {depth} does not fit block {block}")
    c = codes.reshape(n, k // block, block).to(torch.int64)
    q = torch.zeros_like(c)
    for i in range(d):
        q |= torch.roll(c, -i, dims=-1) << (i * nbits)
    return (q & ((1 << depth) - 1)).reshape(n, k)


def block_values(
    codes: torch.Tensor,
    codec: CptcCodec,
    scales: torch.Tensor,
    block: int,
    device=None,
) -> torch.Tensor:
    """Exact reconstructed stored-basis values of a block ``[N, block]``.

    ``codes`` [N, block] uint8, ``scales`` [N] fp16. Returns fp16 [N, block] using
    the two documented roundings (table lookup, then block scale).
    """
    dev = codes.device if device is None else device
    q = window_indices(codes.to(dev), codec.nbits, codec.depth, block)
    idx = decode_index(q, codec.a0, codec.a1, codec.tap_q)
    v = index_table_value(idx, codec.table.to(dev), float(codec.table_scale))
    s = scales.to(dev, torch.float16).reshape(-1, 1)
    return (s * v).to(torch.float16)


# ------------------------------------------------------- QAD code projection


def projection_period(depth: int, nbits: int) -> int:
    """Symbols of one code window rounded up to a power of two.

    A K-bit symbol is read by ``ceil(depth/K)`` consecutive windows; positions in the
    same phase ``t mod period`` therefore never touch a common window when
    ``period >= ceil(depth/K)``. The power-of-two rounding makes the phase count divide
    the CPTC block sizes (16/32/64) for every supported depth (4..16).
    """
    span = (depth + nbits - 1) // nbits
    return 1 << (span - 1).bit_length()


def check_projection(depth: int, nbits: int, block: int) -> int:
    period = projection_period(depth, nbits)
    if block % period:
        raise ValueError(
            f"CPTC projection period {period} does not divide block {block}"
        )
    return period


def _codec_lut(codec: CptcCodec, device) -> torch.Tensor:
    """The mirror-expanded fp16 table as float32, once per codec/device."""
    return index_table_value(
        torch.arange(1 << (codec.tap_q + 1), device=device),
        codec.table.to(device),
        codec.table_scale,
    ).float()


@torch.no_grad()
def project_codes(
    w: torch.Tensor,
    codec: CptcCodec,
    block: int,
    init_codes: torch.Tensor,
    scales: torch.Tensor,
    sweeps: int = 1,
    d: torch.Tensor | None = None,
    lut: torch.Tensor | None = None,
) -> torch.Tensor:
    """Deterministic block-local coordinate projection from an immutable code stream.

    Each position ``t`` is assigned the symbol ``0..2**K-1`` that minimises the exact
    fp16 decode error over the ``ceil(L/K)`` windows that read it, with the other
    symbols fixed. Positions are updated in ``t mod period`` phases (the period is a
    power of two that divides the block), so a phase's windows never intersect and the
    simultaneous choices cannot interfere. Starting from the caller's ``init_codes``
    every call makes the result a pure function of ``(w, scales, init_codes)``.

    ``w``/``init_codes`` are ``[N, block]``; ``d`` optionally weights columns;
    ``lut`` may supply the cached mirror-expanded fp16 table as float32.
    """
    nbits = codec.nbits
    period = check_projection(codec.depth, nbits, block)
    n, b = w.shape
    if b != block or init_codes.shape != (n, block):
        raise ValueError("project_codes expects [N, block] weights and codes")
    span = (codec.depth + nbits - 1) // nbits
    dev = w.device
    codes = init_codes.to(dev, torch.int64).clone()
    wf = w.float()
    df = (
        torch.ones(block, device=dev, dtype=torch.float32)
        if d is None
        else d.to(dev, torch.float32).reshape(-1)
    )
    s = scales.to(dev, torch.float16).float().reshape(-1, 1)
    if lut is None:
        lut = _codec_lut(codec, dev)
    into = torch.arange(span, device=dev)
    shift = into * nbits
    keep = ~(((1 << nbits) - 1) << shift)
    for _ in range(max(1, sweeps)):
        for phase in range(period):
            pos = torch.arange(phase, block, period, device=dev)
            affected = (pos[:, None] - into[None, :]) % block
            q = window_indices(codes, nbits, codec.depth, block)
            qa = q[:, affected]
            wa = wf[:, affected]
            da = df[affected]
            best_err = None
            best_sym = None
            for sym in range(1 << nbits):
                qc = ((qa & keep) | (sym << shift)) & ((1 << codec.depth) - 1)
                idx = decode_index(qc, codec.a0, codec.a1, codec.tap_q)
                vals = (s[:, None] * lut[idx]).to(torch.float16).float()
                err = (da * (vals - wa) ** 2).sum(-1)
                if best_err is None:
                    best_err, best_sym = err, torch.full_like(
                        err, sym, dtype=torch.long
                    )
                else:
                    m = err < best_err
                    best_err = torch.where(m, err, best_err)
                    best_sym = torch.where(
                        m, torch.full_like(best_sym, sym), best_sym
                    )
            codes[:, pos] = best_sym
    return codes.to(torch.uint8)


@torch.no_grad()
def project_codes_blocks(
    w: torch.Tensor,
    codec: CptcCodec,
    block: int,
    init_codes: torch.Tensor,
    scales: torch.Tensor,
    sweeps: int = 1,
    d: torch.Tensor | None = None,
    lut: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply :func:`project_codes` to every ``block`` column chunk at once.

    ``w``/``init_codes`` are ``[N, K]`` with ``K % block == 0`` and ``scales`` is
    ``[N, K/block]``.  The phase schedule and per-block objective are exactly the
    single-block reference; batching only removes the Python/per-block dispatch and
    the repeated LUT construction.  ``d`` is ``[K]`` or ``[block]`` and ``lut`` may
    be the cached mirror-expanded table from :func:`_codec_lut`.
    """
    nbits = codec.nbits
    n, k = w.shape
    if k % block:
        raise ValueError(f"K={k} is not a multiple of CPTC block {block}")
    nblk = k // block
    period = check_projection(codec.depth, nbits, block)
    span = (codec.depth + nbits - 1) // nbits
    dev = w.device
    if init_codes.shape != (n, k):
        raise ValueError("project_codes_blocks expects [N, K] init codes")
    codes = init_codes.to(dev, torch.int64).clone()
    wf = w.float()
    s = scales.to(dev, torch.float16).float().reshape(n, nblk)
    if d is None:
        da_all = torch.ones(nblk * block, device=dev, dtype=torch.float32)
    else:
        df = d.to(dev, torch.float32).reshape(-1)
        if df.numel() == block:
            da_all = df.repeat(nblk)
        elif df.numel() == k:
            da_all = df
        else:
            raise ValueError("projection d must have [block] or [K] entries")
    if lut is None:
        lut = _codec_lut(codec, dev)
    into = torch.arange(span, device=dev)
    shift = into * nbits
    keep = ~(((1 << nbits) - 1) << shift)
    block_ids = torch.arange(nblk, device=dev)
    for _ in range(max(1, sweeps)):
        for phase in range(period):
            pos = torch.arange(phase, block, period, device=dev)
            local = (pos[:, None] - into[None, :]) % block
            affected = (
                block_ids[:, None, None] * block + local[None, :, :]
            ).reshape(-1)
            pos_global = (block_ids[:, None] * block + pos[None, :]).reshape(-1)
            block_of = (
                block_ids[:, None, None]
                .expand(nblk, pos.numel(), span)
                .reshape(-1)
            )
            q = window_indices_blocks(codes, nbits, codec.depth, block)
            npos = nblk * pos.numel()
            qa = q[:, affected].reshape(n, npos, span)
            wa = wf[:, affected].reshape(n, npos, span)
            da = da_all[affected].reshape(npos, span)
            sa = s[:, block_of].reshape(n, npos, span)
            best_err = None
            best_sym = None
            for sym in range(1 << nbits):
                qc = ((qa & keep) | (sym << shift)) & ((1 << codec.depth) - 1)
                idx = decode_index(qc, codec.a0, codec.a1, codec.tap_q)
                vals = (sa * lut[idx]).to(torch.float16).float()
                err = (da * (vals - wa) ** 2).sum(-1)
                if best_err is None:
                    best_err, best_sym = err, torch.full_like(
                        err, sym, dtype=torch.long
                    )
                else:
                    m = err < best_err
                    best_err = torch.where(m, err, best_err)
                    best_sym = torch.where(
                        m, torch.full_like(best_sym, sym), best_sym
                    )
            codes[:, pos_global] = best_sym
    return codes.to(torch.uint8)


# ----------------------------------------------------------------- encoders


_STATE_CACHE = {}
_SCALED_LUT_CACHE = {}


def _state_values(codec: CptcCodec, device):
    """Per-codec state table (fp16, fp32, device copy) without repeated CPU copies."""
    key = (id(codec), str(torch.device(device)))
    entry = _STATE_CACHE.get(key)
    if entry is not None and entry[0] is codec:
        return entry[1], entry[2], entry[3]
    cpu = codec.state_values()
    entry = (codec, cpu, cpu.float(), cpu.to(device))
    _STATE_CACHE[key] = entry
    return entry[1], entry[2], entry[3]


def _scaled_table(codec: CptcCodec, device):
    """The serialized 127-entry fp16 LUT used by the fused scaled kernel."""
    key = (id(codec), str(torch.device(device)))
    entry = _SCALED_LUT_CACHE.get(key)
    if entry is not None and entry[0] is codec:
        return entry[1]
    lut = index_table_value(
        torch.arange(TABLE_RANGE, device=device),
        codec.table.to(device),
        codec.table_scale,
    )
    _SCALED_LUT_CACHE[key] = (codec, lut)
    return lut


def _initial_scale(w, d, codec: CptcCodec):
    """Weighted std ratio between the block and the unit-scale state table."""
    sd = d.sum().clamp_min(1e-30)
    wr = ((d * w * w).sum(-1) / sd).clamp_min(1e-30).sqrt()
    vals = _state_values(codec, torch.device("cpu"))[1]
    vr = ((vals * vals).mean()).clamp_min(1e-30).sqrt()
    # Zero and subnormal fp16 scales are valid CPTC outputs.  Only the
    # normalisation denominator above needs a finite floor; clamping the scale
    # here would make a zero weight block decode as a nonzero lattice value.
    return (wr / vr).to(torch.float16).clamp(min=0.0, max=65504)


def _refit_scale(w, d, values, codes, codec, block):
    """Weighted least-squares block scale against the exact table values."""
    v = values[
        window_indices(codes, codec.nbits, codec.depth, block)
    ].float()
    num = (d * w * v).sum(-1)
    den = (d * v * v).sum(-1).clamp_min(1e-30)
    s = (num / den).to(torch.float16).clamp(min=0.0, max=65504)
    return s, v


def _gather_state(values, q):
    """Gather decoded values for states ``q``; ``values`` is [2**L] or [N, 2**L]."""
    if values.dim() == 1:
        return values[q]
    flat = q.reshape(values.shape[0], -1)
    return values.gather(1, flat).reshape(q.shape)


def _block_sse(w, d, values, codes, codec, block, scale):
    v = _gather_state(values, window_indices(codes, codec.nbits, codec.depth, block)).float()
    s = scale.to(torch.float16).reshape(-1, 1) if isinstance(scale, torch.Tensor) else (
        torch.full((w.shape[0], 1), float(scale), dtype=torch.float16, device=w.device)
    )
    wq = (s * v.to(torch.float16)).to(torch.float16)
    diff = wq.float() - w
    return (d * diff * diff).sum(-1), wq


def greedy_codes(w, values, nbits, depth, block):
    """One-pass greedy encoder (valid stream; used as the CD/QAD initialisation)."""
    n = w.shape[0]
    mask = (1 << depth) - 1
    q = torch.zeros(n, dtype=torch.int64, device=w.device)
    codes = torch.zeros(n, block, dtype=torch.uint8, device=w.device)
    for t in range(block - 1, -1, -1):
        best_cost = best_c = best_q = None
        for c in range(1 << nbits):
            nq = ((q << nbits) | c) & mask
            cost = (_gather_state(values, nq) - w[:, t]).abs()
            if best_cost is None:
                best_cost, best_c, best_q = cost, torch.full_like(q, c), nq
            else:
                take = cost < best_cost
                best_cost = torch.where(take, cost, best_cost)
                best_c = torch.where(take, torch.full_like(q, c), best_c)
                best_q = torch.where(take, nq, best_q)
        codes[:, t] = best_c.to(torch.uint8)
        q = best_q
    return codes


def viterbi_codes(w, d, values, nbits, depth, block):
    """Closed tail-biting Viterbi encoder for one block (paper Section 3.2).

    A free pass supplies candidate wrap-seam statistics from its own traceback (the
    window at position 0, the mid-block window, and the best end state); each is
    pinned and searched to a *closed* path whose end statistic equals the pin. The
    free path's start statistic is the strongest cheap estimator of the cycle state
    and the pinned closure is what makes the emitted stream a valid block. Returns
    ``(codes, sse)``. Cost is ``1 + len(pins)`` trellis laps.
    """
    check_block(nbits, block)
    if w.is_cuda and depth == 16 and nbits in (2, 4) and block in (16, 32, 64):
        # The fused kernel is the real L16 implementation; a launch/compile failure
        # propagates instead of silently falling back to the slow reference.
        from .cptc_viterbi_kernel import viterbi_codes_cuda

        return viterbi_codes_cuda(w, d, values, nbits, depth, block)
    n = w.shape[0]
    s_len = 1 << (depth - nbits)  # compressed statistic r = low L-K bits of q
    step = 1 << nbits
    r = torch.arange(s_len, device=w.device, dtype=torch.int64)
    p_high = torch.arange(step, device=w.device, dtype=torch.int64)
    # q = p_high * S + r; its high K bits equal the predecessor statistic's low
    # K bits, so ``a_idx`` is exactly p = q >> K.
    q_all = (p_high[:, None] * s_len + r[None, :]).reshape(-1)
    values = values.to(w.device)
    if values.dim() == 1:
        v_all = values[q_all].reshape(step, s_len)
    else:
        v_all = values[:, q_all].reshape(values.shape[0], step, s_len)
    # ``q >> K`` is the predecessor prefix for every depth.  The old
    # ``p * (S // step) + (r >> K)`` form is only equivalent when S is a multiple
    # of step (L >= 2K); at short depths it drops the carry from p * S.
    a_idx = (q_all >> nbits).reshape(step, s_len)

    # The paper rotates the block by half a block so the wrap seam is interior.
    cut = block // 2
    walk = [(cut - 1 - i) % block for i in range(block)]
    rows = torch.arange(n, device=w.device)

    def forward(v0, record):
        a = v0
        bps = []
        for t in walk:
            base = a[:, a_idx].reshape(n, step, s_len)
            vv = v_all if v_all.dim() == 3 else v_all.unsqueeze(0)
            cost = base + (vv - w[:, t, None, None]) ** 2 * d[t]
            a, bp = cost.min(dim=1)
            if record:
                bps.append(bp.to(torch.int16))
        return a, bps

    def backtrace(start, bps):
        r_cur = start
        codes = torch.zeros(n, block, dtype=torch.uint8, device=w.device)
        for i in range(block - 1, -1, -1):
            t = walk[i]
            q = bps[i][rows, r_cur].to(torch.int64) * s_len + r_cur
            codes[:, t] = (q & (step - 1)).to(torch.uint8)
            if i:
                r_cur = q >> nbits
        return codes

    def closed_sse(codes):
        return _block_sse(
            w, d, values, codes, _CodecView(nbits, depth), block,
            torch.ones(n, device=w.device, dtype=torch.float16),
        )[0]

    a0, bps0 = forward(
        torch.zeros(n, s_len, device=w.device, dtype=torch.float32), True
    )
    free = backtrace(a0.argmin(-1), bps0)
    q_free = window_indices(free, nbits, depth, block)
    pins = [
        (q_free[:, 0] >> nbits),
        (q_free[:, block // 2] >> nbits),
        a0.argmin(-1),
    ]
    best_cost = None
    best_codes = None
    for pin in pins:
        pinned = torch.full((n, s_len), float("inf"), device=w.device)
        pinned[rows, pin] = 0.0
        af, bps = forward(pinned, True)
        closed = af[rows, pin]
        take = (
            torch.ones(n, dtype=torch.bool, device=w.device)
            if best_cost is None
            else closed < best_cost
        )
        if not bool(take.any()):
            continue
        codes = backtrace(pin, bps)
        if best_codes is None:
            best_codes = codes
            best_cost = closed
        else:
            best_codes = torch.where(take[:, None], codes, best_codes)
            best_cost = torch.where(take, closed, best_cost)
    return best_codes, closed_sse(best_codes)


@dataclass
class _CodecView:
    """The three fields :func:`window_indices` needs, for cost helpers."""

    nbits: int
    depth: int


def refine_codes(codes, w, d, values, nbits, depth, block, sweeps: int = 1):
    """Bounded deterministic coordinate descent on the packed stream.

    Replaces one code at a time by the best of its ``2**K`` values against the
    current latent/scale targets. Changing code ``t`` changes exactly
    ``D = ceil(L/K)`` windows; codes of one phase ``t mod D`` therefore have
    disjoint update regions and are swept in fixed phase order.
    """
    check_block(nbits, block)
    dlen = (depth + nbits - 1) // nbits
    if dlen * 2 > block:
        raise ValueError(
            f"coordinate projection needs block {block} >= 2*ceil(L/K)={2 * dlen}"
        )
    n = codes.shape[0]
    values = values.to(w.device)
    mask = (1 << depth) - 1
    cmask = (1 << nbits) - 1
    rows = torch.arange(n, device=w.device)
    for _ in range(int(sweeps)):
        for phase in range(dlen):
            q = window_indices(codes, nbits, depth, block)
            for t in range(phase, block, dlen):
                affected = [(t - i) % block for i in range(dlen)]
                base = []
                for i, j in enumerate(affected):
                    keep = q[:, j] & ~(cmask << (i * nbits))
                    base.append(keep)
                best_cost = None
                best_c = None
                best_q = None
                for c in range(1 << nbits):
                    cost = torch.zeros(n, device=w.device)
                    qs = []
                    for i, j in enumerate(affected):
                        qc = (base[i] | (c << (i * nbits))) & mask
                        qs.append(qc)
                        diff = _gather_state(values, qc).float() - w[:, j]
                        cost = cost + d[j] * diff * diff
                    if best_cost is None:
                        best_cost, best_c, best_q = cost, c, qs
                    else:
                        take = cost < best_cost
                        best_cost = torch.where(take, cost, best_cost)
                        best_c = torch.where(
                            take, torch.full_like(rows, c), best_c
                        )
                        best_q = [
                            torch.where(take, qc, old)
                            for qc, old in zip(qs, best_q)
                        ]
                codes[:, t] = best_c.to(torch.uint8)
                for i, j in enumerate(affected):
                    q[:, j] = best_q[i]
    return codes


def encode_block(
    w,
    d,
    codec: CptcCodec,
    block: int,
    method: str = "viterbi",
    effort: str = "normal",
    sweeps: int = 0,
):
    """Quantise one block ``[N, block]``; returns ``(codes, scale, sse)``.

    All cost decisions use the exact fp16 decoder outputs for the current fp16
    block scale. CPTC's optimal per-block scale is much wider than the variance
    ratio (the index law concentrates on the centre), so a small deterministic
    grid is searched; each candidate is re-fitted by weighted least squares and
    kept only when the exact decoded error falls.
    """
    grid = {
        "fast": (1.0, 1.5),
        "normal": (0.85, 1.0, 1.3, 1.6),
        "max": (0.8, 1.0, 1.2, 1.45, 1.7),
    }.get(effort, (1.0, 1.5))
    base = _initial_scale(w, d, codec)
    state = _state_values(codec, w.device)[2]
    use_scaled = (
        method == "viterbi"
        and w.is_cuda
        and codec.depth == 16
        and codec.nbits in (2, 4)
        and block in (16, 32, 64)
    )
    best_codes = best_scale = best_sse = None
    for factor in grid:
        scale = (base * factor).to(torch.float16)
        values = None
        if use_scaled:
            from .cptc_viterbi_kernel import viterbi_scaled_codes_cuda

            codes, _ = viterbi_scaled_codes_cuda(
                w,
                d,
                _scaled_table(codec, w.device),
                scale,
                codec.a0,
                codec.a1,
                codec.nbits,
                codec.depth,
                block,
            )
        else:
            values = (scale.reshape(-1, 1) * state[None, :]).to(torch.float16).float()
            if method == "viterbi":
                codes, _ = viterbi_codes(w, d, values, codec.nbits, codec.depth, block)
            else:
                codes = greedy_codes(w, values, codec.nbits, codec.depth, block)
        if sweeps:
            if values is None:
                values = (scale.reshape(-1, 1) * state[None, :]).to(torch.float16).float()
            refine_codes(codes, w, d, values, codec.nbits, codec.depth, block, sweeps)
        for cand_scale in (scale, None):
            if cand_scale is None:
                v = state[
                    window_indices(codes, codec.nbits, codec.depth, block)
                ].float()
                num = (d * w * v).sum(-1)
                den = (d * v * v).sum(-1).clamp_min(1e-30)
                cand_scale = (num / den).to(torch.float16).clamp(
                    min=0.0, max=65504
                )
            sse, _ = _block_sse(w, d, state, codes, codec, block, cand_scale)
            if best_sse is None:
                best_codes, best_scale, best_sse = (
                    codes.clone(),
                    cand_scale.clone(),
                    sse.clone(),
                )
            else:
                take = sse < best_sse
                best_codes = torch.where(take.reshape(-1, 1), codes, best_codes)
                best_scale = torch.where(take, cand_scale, best_scale)
                best_sse = torch.where(take, sse, best_sse)
    return best_codes, best_scale, best_sse


def encode_rows_cptc(
    w,
    d,
    nbits: int,
    block: int,
    codec: CptcCodec,
    effort: str = "normal",
    method: str = "viterbi",
    sweeps: int = 0,
):
    """Encode ``w[N, K]`` (stored basis) in independent tail-biting blocks.

    Returns ``(codes uint8 [N, K], scales fp16 [N, K/B], sse float32 [N])``.
    """
    check_block(nbits, block)
    n, k = w.shape
    if k % block:
        raise ValueError(f"K={k} is not a multiple of CPTC block {block}")
    nb = k // block
    out_codes = torch.empty(n, k, dtype=torch.uint8, device=w.device)
    out_scales = torch.empty(n, nb, dtype=torch.float16, device=w.device)
    sse = torch.zeros(n, dtype=torch.float32, device=w.device)
    d = d.to(w.device, torch.float32)
    rows = max(1, (64 << 20) // (4 * (1 << codec.depth)))
    for s in range(0, n, rows):
        e = min(n, s + rows)
        for b in range(nb):
            lo = b * block
            wb = w[s:e, lo : lo + block].float()
            db = d[lo : lo + block]
            codes, scale, err = encode_block(
                wb, db, codec, block, method=method, effort=effort, sweeps=sweeps
            )
            out_codes[s:e, lo : lo + block] = codes
            out_scales[s:e, b] = scale
            sse[s:e] += err
    return out_codes, out_scales, sse


@torch.no_grad()
def encode_rows_cptc_gptq(
    w,
    d,
    u,
    nbits: int,
    block: int,
    codec: CptcCodec,
    effort: str = "normal",
    sweeps: int = 0,
    comp: bool = False,
    return_state: bool = False,
):
    """Error-compensated (GPTQ-style) CPTC rounding at block granularity.

    A block's codes are chosen jointly (tail-biting trellis).  With ``D = W_B - Q_B``
    fixed, the column errors satisfy ``E U_BB = D`` for the upper inverse factor
    ``U``; one triangular solve gives ``E`` and one GEMM pushes
    ``E @ U_Btail`` into every later block.  This is the exact joint-block analogue
    of the ladder recurrence (including non-diagonal ``U``); a per-column loop over
    the pre-compensation block would silently drop the within-block updates.
    """
    check_block(nbits, block)
    n, k = w.shape
    if k % block:
        raise ValueError(f"K={k} is not a multiple of CPTC block {block}")
    wc = w.float().clone()
    codes = torch.zeros(n, k, dtype=torch.uint8, device=w.device)
    scales = torch.empty(n, k // block, dtype=torch.float16, device=w.device)
    comps = torch.zeros(n, k, dtype=torch.float32, device=w.device) if comp else None
    d = d.to(w.device, torch.float32)
    states = []
    for b in range(k // block):
        j0, j1 = b * block, (b + 1) * block
        wb = wc[:, j0:j1].clone()
        cb, sb, _ = encode_block(
            wb,
            d[j0:j1],
            codec,
            block,
            method="viterbi",
            effort=effort,
            sweeps=sweeps,
        )
        codes[:, j0:j1] = cb
        scales[:, b] = sb
        wq = block_values(cb, codec, sb, block).float()
        if comps is not None:
            comps[:, j0:j1] = wb
        if j1 < k:
            ub = torch.triu(u[j0:j1, j0:j1].float())
            if block == 1:
                err = (wb - wq) / ub[0, 0].clamp_min(1e-30)
            else:
                # D = E U with U upper; transposed it is a lower-triangular solve.
                err = torch.linalg.solve_triangular(
                    ub.T.contiguous(), (wb - wq).T.contiguous(), upper=False
                ).T
            wc[:, j1:] -= err @ u[j0:j1, j1:]
        if return_state:
            states.append(wc.clone())
    rec = torch.empty(n, k, dtype=torch.float16, device=w.device)
    for b in range(k // block):
        rec[:, b * block : (b + 1) * block] = block_values(
            codes[:, b * block : (b + 1) * block], codec, scales[:, b], block
        )
    diff = rec.float() - w
    sse = (d * diff * diff).sum(-1)
    if return_state:
        return codes, scales, sse, comps, wc, states
    return codes, scales, sse, comps


# -------------------------------------------------------------- container


@dataclass
class CptcTensor:
    """Serialized CPTC tensor: same code buffers as HI2, a per-block scale instead
    of the ladder parameters, and one codec per bit width (K) used by its rows."""

    codes: torch.Tensor  # int32 [sum words]
    code_off: torch.Tensor  # int32 [R+1]
    scale: torch.Tensor  # fp16 [sum blocks]
    param_off: torch.Tensor  # int32 [R+1], offsets in blocks
    row_nbits: torch.Tensor  # uint8 [R]
    row_block: torch.Tensor  # uint8 [R]
    had: torch.Tensor  # uint8 [K/16]
    shape: tuple[int, int]
    codecs: dict[int, CptcCodec] = field(default_factory=dict)
    col_scale: torch.Tensor | None = None


CPTC_FIELDS = (
    "codes",
    "code_off",
    "scale",
    "param_off",
    "row_nbits",
    "row_block",
    "had",
)


def assemble_cptc(
    pieces,
    row_nbits,
    row_block,
    had,
    k,
    codecs,
    col_scale=None,
    stager=None,
):
    """Build the flat CPTC buffers from ``(rows, codes[N,K], scales[N,K/B])`` groups."""
    dev = pieces[0][1].device if pieces else torch.device("cpu")
    r = row_nbits.numel()
    for n_bits, b in set(zip(row_nbits.tolist(), row_block.tolist())):
        check_format_block(int(n_bits), int(b))
    nb = row_nbits.to(torch.int64)
    words = (k * nb) // 32
    blocks = k // row_block.to(torch.int64)
    code_off = torch.zeros(r + 1, dtype=torch.int64)
    code_off[1:] = torch.cumsum(words, 0)
    param_off = torch.zeros(r + 1, dtype=torch.int64)
    param_off[1:] = torch.cumsum(blocks, 0)
    code_off_d, param_off_d = code_off.to(dev), param_off.to(dev)
    codes = torch.zeros(int(code_off[-1]), dtype=torch.int32, device=dev)
    scales = torch.zeros(int(param_off[-1]), dtype=torch.float16, device=dev)
    for rows, c, s in pieces:
        rows_c = rows.cpu()
        rows_d = rows.to(dev)
        nbits = int(nb[rows_c[0]])
        wr = k * nbits // 32
        widx = code_off_d[rows_d].unsqueeze(1) + torch.arange(wr, device=dev)
        codes[widx.reshape(-1)] = pack_codes(c.to(dev), nbits).reshape(-1)
        nblk = int(blocks[rows_c[0]])
        pidx = (
            param_off_d[rows_d].unsqueeze(1) + torch.arange(nblk, device=dev)
        ).reshape(-1)
        scales[pidx] = s.to(dev).reshape(-1)
    to_host = stager.to_host if stager is not None else (lambda t: t.cpu())
    return CptcTensor(
        codes=to_host(codes),
        code_off=code_off.to(torch.int32),
        scale=to_host(scales),
        param_off=param_off.to(torch.int32),
        row_nbits=row_nbits.to(torch.uint8),
        row_block=row_block.to(torch.uint8),
        had=had.to(torch.uint8).reshape(-1),
        shape=(r, k),
        codecs=dict(codecs),
        col_scale=col_scale,
    )


def _on(t: CptcTensor, device):
    if device is None:
        return t
    return CptcTensor(
        **{f: getattr(t, f).to(device) for f in CPTC_FIELDS},
        shape=t.shape,
        codecs=t.codecs,
        col_scale=None if t.col_scale is None else t.col_scale.to(device),
    )


def dequantize_stored_cptc(t: CptcTensor, device=None) -> torch.Tensor:
    """fp16 reconstruction in the stored basis as float64 ``[R, K]`` (exact)."""
    t = _on(t, device)
    dev = t.codes.device
    r, k = t.shape
    out = torch.zeros(r, k, dtype=torch.float64, device=dev)
    nb = t.row_nbits.to(torch.int64)
    bl = t.row_block.to(torch.int64)
    for nbits in sorted(set(nb.tolist())):
        for bs in sorted(set(bl.tolist())):
            rows = ((nb == nbits) & (bl == bs)).nonzero().flatten()
            if rows.numel() == 0:
                continue
            codec = t.codecs.get(int(nbits))
            if codec is None:
                raise ValueError(f"CPTC tensor has no codec for K={nbits}")
            wr = k * nbits // 32
            widx = t.code_off[rows].long().unsqueeze(1) + torch.arange(wr, device=dev)
            codes = unpack_codes(t.codes[widx], nbits).reshape(rows.numel(), -1, bs)
            nblk = k // bs
            pidx = (
                t.param_off[rows].long().unsqueeze(1) + torch.arange(nblk, device=dev)
            ).reshape(rows.numel(), nblk)
            scales = t.scale[pidx]
            vals = block_values(
                codes.reshape(-1, bs), codec, scales.reshape(-1), bs, dev
            ).double()
            out[rows] = vals.reshape(rows.numel(), k)
    return out


def dequantize_cptc(t: CptcTensor, device=None) -> torch.Tensor:
    """Reconstruction in the original basis (exact, device-independent)."""
    w = fwht_groups(dequantize_stored_cptc(t, device), t.had)
    if t.col_scale is not None:
        w = w / t.col_scale.to(w.device).double()
    return w


def reference_linear_cptc(x: torch.Tensor, t: CptcTensor) -> torch.Tensor:
    """``x @ W.T`` as a CPTC kernel computes it (reference semantics)."""
    xd = x.double()
    if t.col_scale is not None:
        xd = xd / t.col_scale.double()
    return fwht_groups(xd, t.had) @ dequantize_stored_cptc(t).T


def tensor_bpw_cptc(t: CptcTensor) -> float:
    """Code bits + fp16 block scales + per-row index + tables/coefficients per K."""
    r, k = t.shape
    nb = t.row_nbits.to(torch.float64)
    bl = t.row_block.to(torch.float64)
    row_bits = (nb * k + 16 * k / bl).sum().item()
    table_bits = 0
    for nbits in set(t.row_nbits.tolist()):
        codec = t.codecs.get(int(nbits))
        if codec is None:
            raise ValueError(f"CPTC tensor has no codec for K={nbits}")
        table_bits += 8 * codec.table.numel() + 16 + 32
    extra = 16 * k if t.col_scale is not None else 0
    return (row_bits + CPTC_ROW_OVERHEAD_BITS * r + table_bits + extra) / (r * k)


# ---------------------------------------------------------- serialization


def _codec_canonical(codec: CptcCodec) -> dict:
    return {
        "k": int(codec.nbits),
        "depth": int(codec.depth),
        "tap_q": int(codec.tap_q),
        "shift": int(codec.shift),
        "a0": f"0x{codec.a0:04x}",
        "a1": f"0x{codec.a1:04x}",
        "table_scale_bits": "0x"
        + struct.pack(">e", float(codec.table_scale)).hex(),
        "table_i8_b64": base64.b64encode(
            (codec.table.to(torch.int16) + 256).to(torch.uint8).numpy().tobytes()
        ).decode("ascii"),
    }


def checkpoint_config(codecs: dict[int, CptcCodec]) -> dict:
    """Global ``cptc`` config for a v3.3 checkpoint (runtime contract, revision 1).

    Exactly the keys the vLLM HI2 reader accepts: one depth and one coefficient
    pair per K, one mirrored INT8 table per K, no per-tensor fields.
    """
    if set(codecs) != {2, 4}:
        raise ValueError("CPTC checkpoint config needs codecs for K=2 and K=4")
    depths = {c.depth for c in codecs.values()}
    taps = {c.tap_q for c in codecs.values()}
    if len(depths) != 1 or len(taps) != 1:
        raise ValueError("CPTC checkpoint codecs must share depth and tap_q")
    depth = depths.pop()
    tap_q = taps.pop()
    if tap_q != TAP_Q:
        raise ValueError(f"CPTC checkpoint requires tap_q={TAP_Q}, got {tap_q}")
    tables = {}
    for k in (2, 4):
        c = codecs[k]
        if c.nbits != k or c.depth != depth:
            raise ValueError("CPTC checkpoint codec does not match its K/depth")
        validate_codec(c)
        hist = _index_histogram(depth, c.a0, c.a1, tap_q)
        if not torch.equal(hist, hist.flip(0)):
            raise ValueError(f"CPTC K={k} index histogram is not symmetric")
        if c.table.numel() != TABLE_ENTRIES or int(c.table[-1]) != 0:
            raise ValueError("CPTC checkpoint table must be 64 entries with zero center")
        tables[str(k)] = {
            "a0": int(c.a0),
            "a1": int(c.a1),
            "table_scale": float(c.table_scale),
            "tbl_b64": base64.b64encode(
                c.table.to(torch.int8).numpy().tobytes()
            ).decode("ascii"),
        }
    return {
        "name": "cptc",
        "revision": 1,
        "depth": int(depth),
        "tap_q": int(tap_q),
        "layout": "mirror_int8",
        "shift": 0,
        "block": "row_block",
        "tables": tables,
    }


def codecs_from_checkpoint_config(cfg: dict) -> dict[int, CptcCodec]:
    """Parse the runtime ``cptc`` config back into validated codecs (fail closed)."""
    if not isinstance(cfg, dict):
        raise ValueError("CPTC config must be an object")
    expected = {
        "name": "cptc",
        "revision": 1,
        "tap_q": 6,
        "layout": "mirror_int8",
        "shift": 0,
        "block": "row_block",
    }
    if set(cfg) != set(expected) | {"depth", "tables"}:
        raise ValueError("CPTC config: invalid config keys")
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"CPTC config: unsupported {key}={cfg.get(key)!r}")
    depth = cfg.get("depth")
    if type(depth) is not int or not 8 <= depth <= 16:
        raise ValueError("CPTC config: shared 2/4-bit depth must be an integer in [8, 16]")
    raw = cfg.get("tables")
    if not isinstance(raw, dict) or set(raw) != {"2", "4"}:
        raise ValueError("CPTC config: tables for exactly 2 and 4 bits required")
    out = {}
    for k in (2, 4):
        entry = raw[str(k)]
        if not isinstance(entry, dict) or set(entry) != {
            "a0", "a1", "table_scale", "tbl_b64"
        }:
            raise ValueError("CPTC config: invalid table entry keys")
        a0, a1 = entry["a0"], entry["a1"]
        if type(a0) is not int or type(a1) is not int or not (0 <= a0 <= 0xFFFF):
            raise ValueError("CPTC config: coefficients must be uint16")
        if not (0 <= a1 <= 0xFFFF) or depth_of(a0, a1) != depth:
            raise ValueError("CPTC config: coefficients do not match depth")
        scale = entry["table_scale"]
        if isinstance(scale, bool) or not isinstance(scale, (int, float)):
            raise ValueError("CPTC config: table_scale must be numeric")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("CPTC config: table_scale must be finite and positive")
        try:
            blob = base64.b64decode(entry["tbl_b64"], validate=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("CPTC config: invalid base64 table") from exc
        if len(blob) != TABLE_ENTRIES:
            raise ValueError("CPTC config: table must contain 64 int8 entries")
        table = (torch.tensor(list(blob), dtype=torch.int16) - 256).to(torch.int8)
        if int(table[-1]) != 0:
            raise ValueError("CPTC config: table center is not zero")
        codec = CptcCodec(k, depth, a0, a1, table, torch.tensor(float(scale), dtype=torch.float16))
        validate_codec(codec)
        hist = _index_histogram(depth, a0, a1, TAP_Q)
        if not torch.equal(hist, hist.flip(0)):
            raise ValueError("CPTC config: index histogram is not mirror symmetric")
        out[k] = codec
    return out


def codec_config(codec: CptcCodec) -> dict:
    """JSON-safe, canonical codec metadata (hex coefficients, exact fp16 bits)."""
    d = _codec_canonical(codec)
    d["digest"] = hashlib.sha256(
        json.dumps(d, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return d


def codec_from_config(cfg: dict) -> CptcCodec:
    """Parse and validate one serialized codec (independent-runtime reference)."""
    if not isinstance(cfg, dict):
        raise ValueError("CPTC codec metadata must be an object")
    need = {
        "k",
        "depth",
        "tap_q",
        "shift",
        "a0",
        "a1",
        "table_scale_bits",
        "table_i8_b64",
        "digest",
    }
    if set(cfg) != need:
        raise ValueError(f"CPTC codec metadata keys must be exactly {sorted(need)}")
    body = {k: cfg[k] for k in need - {"digest"}}
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != cfg["digest"]:
        raise ValueError("CPTC codec metadata digest mismatch")
    for name in ("a0", "a1", "table_scale_bits"):
        if (
            not isinstance(cfg[name], str)
            or not cfg[name].startswith("0x")
            or len(cfg[name]) > 10
        ):
            raise ValueError(f"CPTC codec {name} must be a 0x-prefixed hex string")
    try:
        a0, a1 = int(cfg["a0"], 16), int(cfg["a1"], 16)
        raw = bytes.fromhex(cfg["table_scale_bits"][2:])
        scale = torch.tensor(
            struct.unpack(">e", raw)[0], dtype=torch.float16
        )
        blob = base64.b64decode(cfg["table_i8_b64"], validate=True)
    except (ValueError, struct.error) as exc:
        raise ValueError(f"CPTC codec: {exc}") from exc
    if len(raw) != 2:
        raise ValueError("CPTC table_scale_bits must be one fp16")
    if len(blob) != TABLE_ENTRIES:
        raise ValueError(f"CPTC table must be {TABLE_ENTRIES} bytes")
    table = (torch.tensor(list(blob), dtype=torch.int16) - 256).to(torch.int8)
    return CptcCodec(
        int(cfg["k"]),
        int(cfg["depth"]),
        a0,
        a1,
        table,
        scale,
        int(cfg["tap_q"]),
        int(cfg["shift"]),
    )


def codecs_config(codecs: dict[int, CptcCodec]) -> dict[str, dict]:
    return {str(k): codec_config(c) for k, c in sorted(codecs.items())}


def codecs_from_config(cfg: dict) -> dict[int, CptcCodec]:
    if not isinstance(cfg, dict) or not cfg:
        raise ValueError("CPTC checkpoint needs a non-empty tables mapping")
    out = {}
    for key, value in cfg.items():
        nbits = int(key)
        if str(nbits) != str(key):
            raise ValueError(f"CPTC table key {key!r} is not an integer bit width")
        codec = codec_from_config(value)
        if codec.nbits != nbits:
            raise ValueError(f"CPTC table key {key} disagrees with K={codec.nbits}")
        out[nbits] = codec
    return out


def validate_codec_set(codecs: dict[int, CptcCodec], expected_k) -> None:
    for k in sorted(set(int(x) for x in expected_k)):
        if k not in codecs:
            raise ValueError(f"CPTC checkpoint has no codec for K={k}")
