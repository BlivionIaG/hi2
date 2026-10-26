# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Global allocation: pick every row's ``(nbits, block)`` across the whole model.

Minimise ``sum_t s_t * sse_t / energy_t`` (relative weighted output error per
tensor, scaled by an optional sensitivity ``s_t``) subject to a model-wide
bits-per-weight target. For a multiplier ``lam`` every row independently picks the
candidate minimising ``cost + lam * bits``; ``lam`` is bisected to meet the target.
"""

import math
from dataclasses import dataclass

import torch

from .calibrate import TensorTable, candidate_row_bits

ROW_OVERHEAD_BITS = 48 + 2 + 1  # row constants + block-size flag + INT4 flag, per row


def row_overhead_bits(t: TensorTable) -> int:
    """Per-row overhead of spec 2.5 for the code widths the table allows: the INT4 flag
    is ``ceil(log2(#widths))`` bits (1 for INT2/INT4 as before, 2 with INT8 added)."""
    widths = len({nb for nb, _ in t.candidates})
    return ROW_OVERHEAD_BITS - 1 + max(1, math.ceil(math.log2(max(widths, 1))))


def _overhead(tables) -> int:
    return sum(
        row_overhead_bits(t) * t.shape[0]
        + t.shape[1] // 16
        + (16 * t.shape[1] if t.col_scale is not None else 0)  # smoothing vector
        for t in tables
    )


@dataclass
class Allocation:
    picks: dict[str, torch.Tensor]  # name -> candidate index per row
    bpw: float
    lam: float
    feasible: bool
    error_ratio: float = float(
        "nan"
    )  # weighted error / best achievable with these candidates
    frac_by_bits: dict[int, float] | None = None  # share of rows per code width


def restrict(t: TensorTable, keep) -> TensorTable:
    """A copy of ``t`` with only the candidates ``(nbits, block)`` for which ``keep`` is
    true (picks then index the reduced list; the encoder reads ``t.candidates``)."""
    import dataclasses

    idx = [i for i, c in enumerate(t.candidates) if keep(c)]
    if len(idx) == len(t.candidates) or not idx:
        return t
    return dataclasses.replace(
        t, sse=t.sse[idx], candidates=[t.candidates[i] for i in idx]
    )


def force_int8(
    tables: dict[str, TensorTable], sens: dict[str, float] | None, top: int
) -> tuple[dict[str, TensorTable], list[str]]:
    """Experiment rule: the ``top`` tensors with the highest probe sensitivity (among those
    with INT8 candidates) keep only their INT8 candidates, every other tensor loses them."""
    sens = sens or {}
    cand = [n for n, t in tables.items() if any(nb == 8 for nb, _ in t.candidates)]
    chosen = sorted(cand, key=lambda n: -sens.get(n, 1.0))[:top]
    out = {}
    for n, t in tables.items():
        if n in chosen:
            out[n] = restrict(t, lambda c: c[0] == 8)
        else:
            out[n] = restrict(t, lambda c: c[0] != 8)
    return out, chosen


def _costs(t: TensorTable, s: float):
    k = t.shape[1]
    cost = (s / max(t.energy, 1e-30)) * t.sse  # [C, R]
    bits = torch.tensor(
        [candidate_row_bits(nb, b, k) for nb, b in t.candidates], dtype=torch.float64
    )
    return cost.double(), bits


def _argmin0(v: torch.Tensor) -> torch.Tensor:
    """``v.argmin(0)`` (first index of the minimum, ``v`` without NaN) as C-1 vector
    compares: torch's strided dim-0 argmin is ~30x slower on CPU for ``[6, 10^5]``."""
    best = v[0].clone()
    idx = torch.zeros(v.shape[1], dtype=torch.int64)
    for c in range(1, v.shape[0]):
        m = v[c] < best
        best = torch.where(m, v[c], best)
        idx.masked_fill_(m, c)
    return idx


class _Flat:
    """All tables' costs and bits side by side (``[C, sum R]``, missing candidates of a
    table padded with an infinite cost), so a multiplier is solved with one argmin
    instead of one per tensor. The per-tensor sums are taken over the same values in
    the same order as before, so the bisection, and every pick, is unchanged."""

    def __init__(self, data):
        c = max(cost.shape[0] for _, cost, _ in data)
        self.spans = []
        costs, bits = [], []
        off = 0
        for t, cost, b in data:
            r = cost.shape[1]
            pad = c - cost.shape[0]
            cc = torch.cat([cost, torch.full((pad, r), float("inf"), dtype=cost.dtype)])
            bb = (
                torch.cat([b, torch.zeros(pad, dtype=b.dtype)])
                .unsqueeze(1)
                .expand(c, r)
            )
            costs.append(cc)
            bits.append(bb)
            self.spans.append((t.name, off, off + r))
            off += r
        self.cost = torch.cat(costs, 1)
        self.bits = torch.cat(bits, 1).contiguous()

    def solve(self, lam: float):
        """``(picks, picked bits per tensor, picked cost per tensor)``."""
        p = _argmin0(self.cost + lam * self.bits)
        pb = self.bits.gather(0, p[None])[0]
        pc = self.cost.gather(0, p[None])[0]
        picks, nbits, err = {}, [], []
        for name, a, b in self.spans:
            picks[name] = p[a:b].clone()
            nbits.append(pb[a:b].sum().item())
            err.append(pc[a:b].sum().item())
        return picks, nbits, err


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
    overhead = _overhead(tables)
    floor = sum(cost.min(0).values.sum().item() for _, cost, _ in data)
    budget = max_error_ratio * floor

    flat = _Flat(data)

    def solve(lam: float):
        picks, _, errs = flat.solve(lam)
        err = 0.0
        for e in errs:
            err += e
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
    overhead = _overhead(tables)
    budget = target_bpw * total_w - overhead

    flat = _Flat(data)

    def solve(lam: float):
        picks, nbits, _ = flat.solve(lam)
        bits_total = 0.0
        for b in nbits:
            bits_total += b
        return picks, bits_total

    lam_hi = 1.0
    for _t, cost, bits in data:
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
        return Allocation(p, (b + overhead) / total_w, lam, ok, ratio, frac)

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
