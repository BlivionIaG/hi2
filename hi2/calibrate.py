# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Per-tensor calibration for flex (spec 8).

Two steps bracket the global allocation (``allocate.py``):

1. :func:`build_table` picks the rotation flags and computes, for every row and
   every candidate ``(nbits, block)``, the activation-weighted squared error.
2. :func:`encode_with_picks` encodes the rows with the configurations the
   allocator chose.
"""

from dataclasses import dataclass, field

import torch

from .encode import encode_rows
from .executor import Executor
from .format import (
    GROUP,
    HI2Tensor,
    assemble,
    fwht_groups,
)
from .stats import ColumnStats

ALL_CANDIDATES = [(nb, b) for nb in (2, 4) for b in (16, 32, 64)]
# INT8 rows (v_dot4_i32_i8 in the I8 mode) for very sensitive projections; opt-in via --bits
INT8_CANDIDATES = [(8, 32), (8, 64)]
SELECTABLE = ALL_CANDIDATES + INT8_CANDIDATES
ROTATION_ROWS = 1024  # rows sampled to decide rotation per 64-column chunk
CHUNK = 64


@dataclass
class Config:
    candidates: list[tuple[int, int]] = field(
        default_factory=lambda: list(ALL_CANDIDATES)
    )
    # on | auto | off. Rotation alone hurts once an activation-aware encoder is used
    # (it mixes the channel scales the encoder exploits), but rotation after smoothing
    # helps at every bit width, and rotation helps a data-free encoder a lot
    # (Qwen2.5-0.5B, see the v3.2 spec).
    rotation: str = "on"
    # Smoothing exponent alpha: s = (E[x^2] / geomean) ** (alpha / 2), W -> W * s,
    # x -> x / s before the rotation. 0 disables it; needs calibration statistics.
    smooth: float = 0.5
    # gptq | nearest. gptq (spec 8.3) is Hessian-aware error-compensated rounding:
    # on Qwen3.5-0.8B at 3.0 bpw it takes the perplexity ratio from 1.92 to 1.32. It
    # needs calibration data; data-free runs use nearest rounding.
    rounding: str = "gptq"
    # Sequential propagation (spec 8.4): encode layer by layer on the quantized prefix.
    sequential: bool = False
    damp: float = 0.01  # Hessian damping, relative to the mean diagonal
    effort: str = "normal"  # fast | normal | max


@dataclass
class TensorTable:
    name: str
    shape: tuple[int, int]
    had: torch.Tensor  # uint8 [K/16]
    sse: torch.Tensor  # float32 [C, R], activation-weighted squared error
    candidates: list[tuple[int, int]]
    energy: float  # weighted energy sum(d * w^2) in the stored basis
    n_tokens: int  # calibration tokens behind the weights (0 = data-free)
    col_scale: torch.Tensor | None = None  # fp16 [K] smoothing vector, or None


def candidate_row_bits(nbits: int, block: int, k: int) -> float:
    return nbits * k + 16 * k / block


def _weights(
    stats: ColumnStats | None, k: int, flags, device, inv_scale=None
) -> torch.Tensor:
    if stats is None:
        return torch.ones(k, device=device)
    return stats.diag(flags, inv_scale).to(device)


def smoothing_scale(stats: ColumnStats | None, alpha: float) -> torch.Tensor | None:
    """fp16 smoothing vector ``s`` (or None): ``s_j = (d_j / geomean(d)) ** (alpha/2)``
    with ``d_j = E[x_j^2]``; rounded to fp16 so offline and device agree."""
    if alpha <= 0 or stats is None or stats.n == 0:
        return None
    d = stats.diag().clamp_min(1e-12)
    s = (d / d.log().mean().exp()).pow(alpha / 2).clamp(0.05, 20.0)
    return s.to(torch.float16).cpu()


def _smoothed(w: torch.Tensor, s: torch.Tensor | None):
    """``(W * s, 1 / s)``; both unchanged/None without smoothing."""
    if s is None:
        return w, None
    sf = s.float()
    return w.float() * sf, 1.0 / sf


def choose_rotation(
    w: torch.Tensor, stats: ColumnStats | None, cfg: Config, device, inv_scale=None
) -> torch.Tensor:
    """Per-64-column-chunk flags ``[K/16]`` uint8 (uniform across rows)."""
    k = w.shape[1]
    ng = k // GROUP
    if cfg.rotation == "off":
        return torch.zeros(ng, dtype=torch.uint8)
    if cfg.rotation == "on":
        return torch.ones(ng, dtype=torch.uint8)
    r = w.shape[0]
    idx = torch.arange(0, r, max(1, r // ROTATION_ROWS))[:ROTATION_ROWS]
    ws = w[idx].to(device, torch.float32)
    err = []
    for rot in (0, 1):
        fl = torch.full((ng,), rot, dtype=torch.uint8)
        wt = fwht_groups(ws, fl) if rot else ws
        d = _weights(stats, k, fl, device, inv_scale)
        enc = encode_rows(wt, d, 2, 64, "fast")
        # encode_rows only returns the total per row; recompute per chunk
        rec_err = _chunk_errors(wt, d, enc, 2, 64)
        err.append(rec_err)
    flags = (err[1] < err[0]).to(torch.uint8).repeat_interleave(CHUNK // GROUP)
    return flags.cpu()


def _chunk_errors(wt, d, enc, nbits, block) -> torch.Tensor:
    """Weighted INT2/B64 error per 64-column chunk (one block per chunk per row)."""
    from .format import decode_params, ladder_dequantize

    r, k = wt.shape
    lo, step = decode_params(
        enc.params[..., 0],
        enc.params[..., 1],
        enc.consts[:, 0:1],
        enc.consts[:, 1:2],
        enc.consts[:, 2:3],
    )
    rec = ladder_dequantize(
        enc.codes.reshape(r, k // block, block), lo.float(), step.float(), torch.float32
    ).float()
    wb = wt.reshape(r, k // block, block)
    db = d.reshape(1, k // block, block)
    return (db * (rec - wb) ** 2).sum((0, 2))


def build_table(
    name: str,
    w: torch.Tensor,
    stats: ColumnStats | None,
    cfg: Config,
    ex: Executor,
) -> TensorTable:
    r, k = w.shape
    s16 = smoothing_scale(stats, cfg.smooth)
    w, inv = _smoothed(w, s16)
    had = choose_rotation(w, stats, cfg, ex.primary, inv)

    def shard(dev, lo, hi):
        wt = w[lo:hi].to(dev, torch.float32)
        wt = fwht_groups(wt, had) if had.any() else wt
        d = _weights(stats, k, had, dev, inv)
        sse = torch.stack(
            [encode_rows(wt, d, nb, b, cfg.effort).sse for nb, b in cfg.candidates]
        )
        return sse.cpu(), float((d * wt**2).sum())

    parts = ex.map_rows(shard, r)
    sse = torch.cat([p[0] for p in parts], dim=1)
    energy = sum(p[1] for p in parts)
    return TensorTable(
        name,
        (r, k),
        had,
        sse,
        list(cfg.candidates),
        energy,
        stats.n if stats else 0,
        s16,
    )


def encode_with_picks(
    w: torch.Tensor,
    stats: ColumnStats | None,
    table: TensorTable,
    picks: torch.Tensor,
    cfg: Config,
    ex: Executor,
    sigma: torch.Tensor | None = None,
) -> HI2Tensor:
    """Encode rows with the candidate index ``picks[r]`` chosen by the allocator."""
    r, k = w.shape
    had = table.had
    w, inv = _smoothed(w, table.col_scale)
    nbits = torch.tensor([table.candidates[i][0] for i in picks.tolist()])
    block = torch.tensor([table.candidates[i][1] for i in picks.tolist()])

    use_gptq = cfg.rounding == "gptq" and sigma is not None

    def shard(dev, lo, hi):
        wt = w[lo:hi].to(dev, torch.float32)
        wt = fwht_groups(wt, had) if had.any() else wt
        if use_gptq:
            from .hessian import encode_rows_gptq, inverse_factor, transform_sigma

            hs = transform_sigma(sigma.to(dev), table.col_scale, had)
            d = hs.diagonal().clone()
            u = inverse_factor(hs, cfg.damp)
            del hs
        else:
            d = _weights(stats, k, had, dev, inv)
        pieces = []
        for ci, (nb, b) in enumerate(table.candidates):
            rows = (picks[lo:hi] == ci).nonzero().flatten()
            if rows.numel() == 0:
                continue
            if use_gptq:
                enc = encode_rows_gptq(wt[rows.to(dev)], d, u, nb, b, cfg.effort)
            else:
                enc = encode_rows(wt[rows.to(dev)], d, nb, b, cfg.effort)
            pieces.append(
                (rows + lo, enc.codes.cpu(), enc.params.cpu(), enc.consts.cpu())
            )
        return pieces

    pieces = [p for part in ex.map_rows(shard, r) for p in part]
    return assemble(pieces, nbits, block, had, k, table.col_scale)
