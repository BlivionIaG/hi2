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
ROTATION_ROWS = 1024  # rows sampled to decide rotation per 64-column chunk
CHUNK = 64


@dataclass
class Config:
    candidates: list[tuple[int, int]] = field(
        default_factory=lambda: list(ALL_CANDIDATES)
    )
    # off | auto | on. Off by default: with calibration data, rotation measured worse
    # end to end (Qwen2.5-0.5B: auto +4% perplexity, on +53%). See the v3.2 spec.
    rotation: str = "off"
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


def candidate_row_bits(nbits: int, block: int, k: int) -> float:
    return nbits * k + 16 * k / block


def _weights(stats: ColumnStats | None, k: int, flags, device) -> torch.Tensor:
    if stats is None:
        return torch.ones(k, device=device)
    return stats.diag(flags).to(device)


def choose_rotation(
    w: torch.Tensor, stats: ColumnStats | None, cfg: Config, device
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
        d = _weights(stats, k, fl, device)
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
    had = choose_rotation(w, stats, cfg, ex.primary)

    def shard(dev, lo, hi):
        wt = w[lo:hi].to(dev, torch.float32)
        wt = fwht_groups(wt, had) if had.any() else wt
        d = _weights(stats, k, had, dev)
        sse = torch.stack(
            [encode_rows(wt, d, nb, b, cfg.effort).sse for nb, b in cfg.candidates]
        )
        return sse.cpu(), float((d * wt**2).sum())

    parts = ex.map_rows(shard, r)
    sse = torch.cat([p[0] for p in parts], dim=1)
    energy = sum(p[1] for p in parts)
    return TensorTable(
        name, (r, k), had, sse, list(cfg.candidates), energy, stats.n if stats else 0
    )


def encode_with_picks(
    w: torch.Tensor,
    stats: ColumnStats | None,
    table: TensorTable,
    picks: torch.Tensor,
    cfg: Config,
    ex: Executor,
) -> HI2Tensor:
    """Encode rows with the candidate index ``picks[r]`` chosen by the allocator."""
    r, k = w.shape
    had = table.had
    nbits = torch.tensor([table.candidates[i][0] for i in picks.tolist()])
    block = torch.tensor([table.candidates[i][1] for i in picks.tolist()])

    def shard(dev, lo, hi):
        wt = w[lo:hi].to(dev, torch.float32)
        wt = fwht_groups(wt, had) if had.any() else wt
        d = _weights(stats, k, had, dev)
        pieces = []
        for ci, (nb, b) in enumerate(table.candidates):
            rows = (picks[lo:hi] == ci).nonzero().flatten()
            if rows.numel() == 0:
                continue
            enc = encode_rows(wt[rows.to(dev)], d, nb, b, cfg.effort)
            pieces.append(
                (rows + lo, enc.codes.cpu(), enc.params.cpu(), enc.consts.cpu())
            )
        return pieces

    pieces = [p for part in ex.map_rows(shard, r) for p in part]
    return assemble(pieces, nbits, block, had, k)
