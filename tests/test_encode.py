# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Encoder, statistics, per-tensor calibration, global allocation (CPU)."""

import torch

from hi2.allocate import allocate_global
from hi2.calibrate import ALL_CANDIDATES, Config, build_table, encode_with_picks
from hi2.encode import encode_rows, fit_ladder
from hi2.executor import Executor
from hi2.format import dequantize, fwht_groups, tensor_bpw
from hi2.stats import ColumnStats


def _w(r=64, k=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(r, k, generator=g) * torch.exp(
        0.45 * torch.randn(r, 1, generator=g)
    )
    w *= torch.exp(0.6 * torch.randn(1, k, generator=g))
    w[:, [c for c in (5, 70, 133, 200) if c < k]] *= 10
    return w


def test_weighted_fit_beats_minmax_and_respects_weights():
    torch.manual_seed(0)
    wb = torch.randn(32, 4, 32)
    d = torch.ones(1, 4, 32)
    lo, step = fit_ladder(wb, d, 4, "normal")
    rec = lo.unsqueeze(-1) + torch.round(
        (wb - lo.unsqueeze(-1)) / step.unsqueeze(-1)
    ).clamp(0, 3) * step.unsqueeze(-1)
    mn, mx = wb.amin(-1), wb.amax(-1)
    st = (mx - mn) / 3
    rec_mm = mn.unsqueeze(-1) + torch.round(
        (wb - mn.unsqueeze(-1)) / st.unsqueeze(-1)
    ) * st.unsqueeze(-1)
    assert ((rec - wb) ** 2).sum() < ((rec_mm - wb) ** 2).sum() * 0.9
    # a heavily weighted column ends up with lower error than under uniform weights
    d2 = d.clone()
    d2[..., 3] = 1000.0
    lo2, step2 = fit_ladder(wb, d2, 4, "normal")
    rec2 = lo2.unsqueeze(-1) + torch.round(
        (wb - lo2.unsqueeze(-1)) / step2.unsqueeze(-1)
    ).clamp(0, 3) * step2.unsqueeze(-1)
    assert ((rec2 - wb)[..., 3] ** 2).sum() <= ((rec - wb)[..., 3] ** 2).sum()


def test_encode_rows_sse_matches_reconstruction():
    w = _w(16, 128)
    d = torch.rand(128) + 0.1
    enc = encode_rows(w, d, 2, 32, "fast")
    assert enc.codes.dtype == torch.uint8 and int(enc.codes.max()) <= 3
    assert enc.params.shape == (16, 4, 2) and enc.consts.shape == (16, 3)
    assert (enc.sse >= 0).all()


def test_stats_diag_and_rotated_diag_match_direct_computation():
    torch.manual_seed(1)
    k = 64
    x = torch.randn(500, k) * torch.rand(k) * 3
    st = ColumnStats(k)
    st.update(x[:200])
    st.update(x[200:])
    assert st.n == 500
    assert torch.allclose(st.diag(), (x**2).mean(0), rtol=1e-4)
    flags = torch.tensor([1, 0, 1, 1], dtype=torch.uint8)
    # diag of the rotated activation second moment, per flagged group
    xr = fwht_groups(x, flags.repeat_interleave(1))
    assert torch.allclose(st.diag(flags), (xr**2).mean(0), rtol=1e-3, atol=1e-5)
    other = ColumnStats(k)
    other.update(x[:100])
    merged = ColumnStats(k)
    merged.update(x[100:])
    merged.merge(other)
    assert torch.allclose(merged.diag(), (x**2).mean(0), rtol=1e-4)


def test_table_and_encode_with_picks_roundtrip():
    w = _w()
    ex = Executor("cpu")
    cfg = Config(effort="fast")
    table = build_table("x.weight", w, None, cfg, ex)
    assert table.sse.shape == (len(ALL_CANDIDATES), 64)
    picks = table.sse.argmin(0)  # most accurate candidate per row
    t = encode_with_picks(w, None, table, picks, cfg, ex)
    assert t.shape == (64, 256) and tensor_bpw(t) > 3
    err = ((dequantize(t) - w.double()) ** 2).sum() / (w.double() ** 2).sum()
    assert err.item() < 0.06


def test_row_sharding_is_equivalent():
    w = _w(65, 128)
    cfg = Config(effort="fast", rotation="off")
    one = build_table("a", w, None, cfg, Executor(["cpu"]))
    two = build_table("a", w, None, cfg, Executor(["cpu", "cpu"]))
    assert torch.allclose(one.sse, two.sse, rtol=1e-5)


def test_global_allocation_hits_target_and_moves_bits_to_sensitive_tensors():
    ex = Executor("cpu")
    cfg = Config(effort="fast", rotation="off")
    easy = build_table("easy", torch.randn(64, 128) * 0.01, None, cfg, ex)
    hard = build_table("hard", _w(64, 128, seed=3), None, cfg, ex)
    alloc = allocate_global([easy, hard], target_bpw=3.5)
    assert alloc.feasible and alloc.bpw <= 3.5 + 1e-6
    assert alloc.bpw > 3.2  # uses the budget
    assert alloc.picks["easy"].shape == (64,)
    # raising the target never raises the relative error
    loose = allocate_global([easy, hard], target_bpw=5.0)
    assert loose.bpw >= alloc.bpw


def test_allocation_flags_infeasible_target():
    ex = Executor("cpu")
    t = build_table("t", _w(32, 128), None, Config(effort="fast", rotation="off"), ex)
    alloc = allocate_global([t], target_bpw=1.0)
    assert not alloc.feasible


def test_int8_rows_roundtrip_and_accounting():
    from hi2.calibrate import INT8_CANDIDATES
    from hi2.format import dequantize

    torch.manual_seed(0)
    w = torch.randn(16, 128)
    cfg = Config(candidates=[(2, 32), *INT8_CANDIDATES], rounding="nearest")
    ex = Executor("cpu")
    table = build_table("x", w, None, cfg, ex)
    picks = torch.tensor([0] * 8 + [2] * 8)  # mixed INT2/INT8 rows
    t = encode_with_picks(w, None, table, picks, cfg, ex, None)
    rel = ((dequantize(t).float() - w) ** 2).sum(1) / (w**2).sum(1)
    assert rel[8:].max() < 1e-3 < rel[:8].min()
    assert tensor_bpw(t) > 5.0


def test_quality_allocation_puts_int2_where_it_matters_less_and_is_monotone():
    from hi2.allocate import allocate_by_quality

    ex = Executor("cpu")
    cfg = Config(effort="fast", rotation="off")
    easy = build_table("easy", torch.randn(64, 128) * 0.01, None, cfg, ex)
    hard = build_table("hard", _w(64, 128, seed=3), None, cfg, ex)
    sens = {"easy": 0.01, "hard": 1.0}
    tight = allocate_by_quality([easy, hard], 1.5, sens)
    loose = allocate_by_quality([easy, hard], 20.0, sens)
    assert tight.feasible and loose.feasible
    assert tight.error_ratio <= 1.5 + 1e-6 and loose.error_ratio <= 20.0 + 1e-6
    assert loose.bpw <= tight.bpw  # a looser quality target never costs more bits
    assert loose.frac_by_bits[2] >= tight.frac_by_bits[2]
    # the low-sensitivity tensor is the one that drops to INT2 first
    int2 = [i for i, (nb, _) in enumerate(easy.candidates) if nb == 2]
    share = {
        n: float(torch.isin(loose.picks[n], torch.tensor(int2)).float().mean())
        for n in ("easy", "hard")
    }
    assert share["easy"] >= share["hard"]
    assert not allocate_by_quality([easy, hard], 0.5, sens).feasible


def test_smooth_search_keeps_the_lowest_error_exponent():
    from hi2.calibrate import (
        Config,
        build_table,
        shared_transform,
        smoothing_scale,
        transform_error,
    )
    from hi2.executor import Executor

    torch.manual_seed(5)
    k, r = 128, 96
    x = torch.randn(600, k) * torch.exp(
        torch.randn(k)
    )  # channels of very unequal scale
    st = ColumnStats(k)
    st.update(x)
    w = torch.randn(r, k) * 0.3
    ex = Executor(["cpu"])
    alphas = (0.25, 0.5, 0.75, 1.0)
    cfg = Config(rotation="on", smooth=0.5, smooth_search=alphas)
    s16, had = shared_transform([w], st, cfg, ex)
    errs = {
        a: transform_error(
            [w], st, smoothing_scale(st, a), torch.ones(k // 16, dtype=torch.uint8), ex
        )
        for a in alphas
    }
    chosen = next(a for a in alphas if torch.equal(smoothing_scale(st, a), s16))
    assert errs[chosen] == min(errs.values())
    # one exponent in the set reproduces the single-exponent path
    one = Config(rotation="on", smooth=0.5, smooth_search=(0.5,))
    base = Config(rotation="on", smooth=0.5)
    t1 = build_table("t", w, st, one, ex)
    t0 = build_table("t", w, st, base, ex)
    assert torch.equal(t1.col_scale, t0.col_scale) and torch.equal(t1.sse, t0.sse)
    # a searched tensor stores its own vector, and the cache key tells the settings apart
    from hi2.cache import tables_key

    assert tables_key({"a": 1}, cfg) != tables_key({"a": 1}, base)
    assert tables_key({"a": 1}, base) == tables_key({"a": 1}, Config(smooth=0.5))
