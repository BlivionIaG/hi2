# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Global allocation: pick every row's ``(nbits, block)`` across the whole model.

Minimise ``sum_t s_t * sse_t / energy_t`` (relative weighted output error per
tensor, scaled by an optional sensitivity ``s_t``) subject to a model-wide
bits-per-weight target. For a multiplier ``lam`` every row independently picks the
candidate minimising ``cost + lam * bits``; ``lam`` is bisected to meet the target.
"""

from dataclasses import dataclass

import torch

from .calibrate import TensorTable, candidate_row_bits

ROW_OVERHEAD_BITS = 48 + 2 + 1  # row constants + block-size flag + INT4 flag, per row


@dataclass
class Allocation:
    picks: dict[str, torch.Tensor]  # name -> candidate index per row
    bpw: float
    lam: float
    total_weights: int
    feasible: bool


def _costs(t: TensorTable, s: float):
    k = t.shape[1]
    cost = (s / max(t.energy, 1e-30)) * t.sse  # [C, R]
    bits = torch.tensor(
        [candidate_row_bits(nb, b, k) for nb, b in t.candidates], dtype=torch.float64
    )
    return cost.double(), bits


def allocate_global(
    tables: list[TensorTable],
    target_bpw: float,
    sensitivity: dict[str, float] | None = None,
    iters: int = 80,
) -> Allocation:
    sens = sensitivity or {}
    data = [(t, *_costs(t, sens.get(t.name, 1.0))) for t in tables]
    total_w = sum(t.shape[0] * t.shape[1] for t in tables)
    overhead = sum(
        ROW_OVERHEAD_BITS * t.shape[0]
        + t.shape[1] // 16
        + (16 * t.shape[1] if t.col_scale is not None else 0)  # smoothing vector
        for t in tables
    )
    budget = target_bpw * total_w - overhead

    def solve(lam: float):
        picks, bits_total = {}, 0.0
        for t, cost, bits in data:
            p = (cost + lam * bits.unsqueeze(1)).argmin(0)
            picks[t.name] = p
            bits_total += bits[p].sum().item()
        return picks, bits_total

    lam_hi = 1.0
    for t, cost, bits in data:
        span_b = (bits.max() - bits.min()).clamp_min(1e-9)
        lam_hi = max(
            lam_hi,
            2 * ((cost.max(0).values - cost.min(0).values) / span_b).max().item(),
        )
    lo, hi = lam_hi * 1e-12, lam_hi  # lam_hi -> fewest bits, lo -> most
    picks, bits_total = solve(hi)
    min_bits = bits_total
    if min_bits > budget:  # even the cheapest assignment is over target
        return Allocation(picks, (bits_total + overhead) / total_w, hi, total_w, False)
    best = solve(lo)
    if best[1] <= budget:  # even the most accurate one fits
        return Allocation(best[0], (best[1] + overhead) / total_w, lo, total_w, True)
    best_lam = hi
    for _ in range(iters):
        mid = (lo * hi) ** 0.5
        p, b = solve(mid)
        if b <= budget:
            hi, picks, bits_total, best_lam = mid, p, b, mid
        else:
            lo = mid
    return Allocation(picks, (bits_total + overhead) / total_w, best_lam, total_w, True)
