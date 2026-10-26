# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""The vectorised allocator gives the same picks, bpw and ratio as the per-tensor one
(the pre-optimisation code is inlined below as the reference)."""

import pytest
import torch
from hi2.allocate import (
    ROW_OVERHEAD_BITS,
    Allocation,
    allocate_by_quality,
    allocate_global,
)
from hi2.calibrate import (
    ALL_CANDIDATES,
    INT8_CANDIDATES,
    TensorTable,
    candidate_row_bits,
)


# ---------------------------------------------------------------- reference (verbatim)
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


def ref_allocate_by_quality(
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


def ref_allocate_global(
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


# ---------------------------------------------------------------- tests


def _tables(seed, n=40):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        cands = (
            list(ALL_CANDIDATES)
            if i % 5
            else list(ALL_CANDIDATES) + list(INT8_CANDIDATES)
        )
        if i % 7 == 3:
            cands = [(2, 32), (4, 64)]
        r = int(torch.randint(8, 300, (1,), generator=g))
        k = 64 * int(torch.randint(1, 40, (1,), generator=g))
        base = torch.rand(r, generator=g) * 10 ** float(torch.randn(1, generator=g))
        sse = torch.stack(
            [
                base
                * (2.0 ** (-nb))
                * (1 + b / 64)
                * (1 + 0.1 * torch.rand(r, generator=g))
                for nb, b in cands
            ]
        ).float()
        cs = torch.ones(k, dtype=torch.float16) if i % 2 else None
        out.append(
            TensorTable(
                f"t{i}",
                (r, k),
                torch.zeros(k // 16, dtype=torch.uint8),
                sse,
                cands,
                float(sse.sum()) * 3,
                100,
                cs,
            )
        )
    return out


def _same(a, b):
    assert a.picks.keys() == b.picks.keys()
    for n in a.picks:
        assert torch.equal(a.picks[n], b.picks[n]), n
    assert (a.bpw, a.lam, a.feasible, a.error_ratio, a.frac_by_bits) == (
        b.bpw,
        b.lam,
        b.feasible,
        b.error_ratio,
        b.frac_by_bits,
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("target", [2.4, 3.0, 3.7, 9.0])
def test_global_allocation_unchanged(seed, target):
    ts = _tables(seed)
    sens = {t.name: 0.5 + (i % 4) for i, t in enumerate(ts)}
    _same(allocate_global(ts, target, sens), ref_allocate_global(ts, target, sens))


@pytest.mark.parametrize("seed", [0, 3])
@pytest.mark.parametrize("ratio", [0.5, 1.5, 4.0, 12.0])
def test_quality_allocation_unchanged(seed, ratio):
    ts = _tables(seed)
    _same(
        allocate_by_quality(ts, ratio, None), ref_allocate_by_quality(ts, ratio, None)
    )
