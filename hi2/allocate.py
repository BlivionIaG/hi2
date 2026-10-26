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
    error_ratio: float = float(
        "nan"
    )  # weighted error / best achievable with these candidates
    frac_by_bits: dict[int, float] | None = None  # share of rows per code width


def _costs(t: TensorTable, s: float):
    k = t.shape[1]
    cost = (s / max(t.energy, 1e-30)) * t.sse  # [C, R]
    bits = torch.tensor(
        [candidate_row_bits(nb, b, k) for nb, b in t.candidates], dtype=torch.float64
    )
    return cost.double(), bits


def _summary(data, picks):
    """Weighted error relative to the best achievable, and the row share per width."""
    err = floor = 0.0
    counts: dict[int, int] = {}
    rows = 0
    for t, cost, _ in data:
        p = picks[t.name]
        idx = torch.arange(p.numel())
        err += cost[p, idx].sum().item()
        floor += cost.min(0).values.sum().item()
        for ci, (nb, _b) in enumerate(t.candidates):
            counts[nb] = counts.get(nb, 0) + int((p == ci).sum())
        rows += p.numel()
    return err / max(floor, 1e-30), {nb: n / max(rows, 1) for nb, n in counts.items()}


def allocate_by_quality(
    tables: list[TensorTable],
    max_error_ratio: float,
    sensitivity: dict[str, float] | None = None,
    iters: int = 80,
) -> Allocation:
    """Quality-driven allocation: the fewest bits whose sensitivity-weighted error is at
    most ``max_error_ratio`` times the best achievable with the allowed candidates (1.0 =
    every row at its most accurate candidate). Rows that matter less end up on INT2 and
    the rest on INT4 (or INT8 if allowed); the model-wide bpw is a result, not an input."""
    sens = sensitivity or {}
    data = [(t, *_costs(t, sens.get(t.name, 1.0))) for t in tables]
    total_w = sum(t.shape[0] * t.shape[1] for t in tables)
    overhead = sum(
        ROW_OVERHEAD_BITS * t.shape[0]
        + t.shape[1] // 16
        + (16 * t.shape[1] if t.col_scale is not None else 0)
        for t in tables
    )
    floor = sum(cost.min(0).values.sum().item() for _, cost, _ in data)
    budget = max_error_ratio * floor

    def solve(lam: float):
        picks, err = {}, 0.0
        for t, cost, bits in data:
            p = (cost + lam * bits.unsqueeze(1)).argmin(0)
            picks[t.name] = p
            err += cost[p, torch.arange(p.numel())].sum().item()
        return picks, err

    lam_hi = 1.0
    for _t, cost, bits in data:
        span_b = (bits.max() - bits.min()).clamp_min(1e-9)
        lam_hi = max(
            lam_hi,
            2 * ((cost.max(0).values - cost.min(0).values) / span_b).max().item(),
        )
    lo, hi = lam_hi * 1e-12, lam_hi  # larger lam -> fewer bits, more error
    picks, err = solve(lo)
    best_lam, feasible = lo, err <= budget
    if feasible:
        for _ in range(iters):
            mid = (lo * hi) ** 0.5
            p, e = solve(mid)
            if e <= budget:
                lo, picks, best_lam = mid, p, mid
            else:
                hi = mid
    bits_total = sum(bits[picks[t.name]].sum().item() for t, _, bits in data)
    ratio, frac = _summary(data, picks)
    return Allocation(
        picks,
        (bits_total + overhead) / total_w,
        best_lam,
        total_w,
        feasible,
        ratio,
        frac,
    )


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

    def done(p, b, lam, ok):
        ratio, frac = _summary(data, p)
        return Allocation(p, (b + overhead) / total_w, lam, total_w, ok, ratio, frac)

    if min_bits > budget:  # even the cheapest assignment is over target
        return done(picks, bits_total, hi, False)
    best = solve(lo)
    if best[1] <= budget:  # even the most accurate one fits
        return done(best[0], best[1], lo, True)
    best_lam = hi
    for _ in range(iters):
        mid = (lo * hi) ** 0.5
        p, b = solve(mid)
        if b <= budget:
            hi, picks, bits_total, best_lam = mid, p, b, mid
        else:
            lo = mid
    return done(picks, bits_total, best_lam, True)
