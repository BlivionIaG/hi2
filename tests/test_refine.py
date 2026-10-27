# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Local code refinement (CPU): a sweep never raises the Hessian objective, codes stay in
their range, a sweep that finds nothing to move returns the input, and the refined rows
decode like any other rows."""

import pytest
import torch

pytest.importorskip("safetensors")

from hi2.encode import encode_rows  # noqa: E402
from hi2.format import decode_params  # noqa: E402
from hi2.hessian import (  # noqa: E402
    encode_rows_gptq_reference,
    inverse_factor,
)
from hi2.refine import objective, refine_codes, refine_encoded  # noqa: E402


def _setup(nbits, block, rows=24, k=128, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(400, k) * torch.exp(0.5 * torch.randn(k))
    hs = (x.T @ x) / 400
    w = torch.randn(rows, k) * 0.3
    enc = encode_rows_gptq_reference(
        w, hs.diagonal().clone(), inverse_factor(hs), nbits, block
    )
    lo, step = decode_params(
        enc.params[..., 0],
        enc.params[..., 1],
        enc.consts[:, 0:1],
        enc.consts[:, 1:2],
        enc.consts[:, 2:3],
    )
    return w, hs, enc, lo.float(), step.float()


@pytest.mark.parametrize("nbits,block", [(2, 32), (4, 64), (2, 16)])
def test_a_sweep_never_raises_the_objective_and_keeps_codes_in_range(nbits, block):
    w, hs, enc, lo, step = _setup(nbits, block)
    before = objective(w, enc.codes, lo, step, hs, block)
    out = refine_codes(w, enc.codes, lo, step, hs, nbits, block, passes=2)
    assert out.dtype == enc.codes.dtype and out.shape == enc.codes.shape
    assert int(out.max()) <= 2**nbits - 1 and int(out.min()) >= 0
    after = objective(w, out, lo, step, hs, block)
    assert after <= before and after < before * 0.99  # GPTQ leaves a gap to close
    one = refine_codes(w, enc.codes, lo, step, hs, nbits, block, passes=1)
    assert (
        objective(w, one, lo, step, hs, block) >= after - 1e-6
    )  # more sweeps do not hurt
    # moves are single steps: no code changed by more than one level per sweep
    assert int((one.int() - enc.codes.int()).abs().max()) <= 1


def test_nothing_to_move_returns_the_input_and_zero_passes_are_a_no_op():
    w, hs, enc, lo, step = _setup(2, 32)
    zero = refine_codes(w, enc.codes, lo, step, hs, 2, 32, passes=0)
    assert torch.equal(zero, enc.codes)
    once = refine_codes(
        w, enc.codes, lo, step, hs, 2, 32, passes=40
    )  # converges, stops early
    again = refine_codes(w, once, lo, step, hs, 2, 32, passes=1)
    assert torch.equal(again, once)


def test_block_lazy_update_matches_a_plain_column_loop():
    """The blocked sweep makes the decisions of the simple loop (same objective)."""
    w, hs, enc, lo, step = _setup(2, 32, rows=8, k=256, seed=3)
    out = refine_codes(w, enc.codes, lo, step, hs, 2, 32, passes=1)
    c = enc.codes.float().clone()
    idx = torch.arange(256) // 32
    e = (lo[:, idx] + c * step[:, idx]).half().float() - w
    g = e @ hs
    for j in range(256):
        lo_j, st_j = lo[:, j // 32], step[:, j // 32]
        cur = (lo_j + c[:, j] * st_j).half().float()
        best_gain = torch.zeros(8)
        best_dw = torch.zeros(8)
        best_d = torch.zeros(8)
        for d in (1.0, -1.0):
            cn = c[:, j] + d
            ok = (cn >= 0) & (cn <= 3) & (st_j > 0)
            dw = (lo_j + cn * st_j).half().float() - cur
            gain = torch.where(ok, -(2 * dw * g[:, j] + dw * dw * hs[j, j]), -torch.inf)
            better = gain > best_gain
            best_gain = torch.where(better, gain, best_gain)
            best_dw = torch.where(better, dw, best_dw)
            best_d = torch.where(better, torch.full((8,), d), best_d)
        take = best_gain > 0
        c[:, j] += torch.where(take, best_d, torch.zeros(8))
        g += torch.where(take, best_dw, torch.zeros(8))[:, None] * hs[j][None, :]
    ref = objective(w, c, lo, step, hs, 32)
    got = objective(w, out, lo, step, hs, 32)
    assert got == pytest.approx(ref, rel=1e-4)


def test_refine_encoded_keeps_the_ladders_and_updates_sse():
    w, hs, enc, *_ = _setup(4, 64)
    out = refine_encoded(enc, w, hs, 4, 64, passes=2)
    assert torch.equal(out.params, enc.params) and torch.equal(out.consts, enc.consts)
    assert out.codes.dtype == enc.codes.dtype
    assert not torch.equal(out.codes, enc.codes)
    assert out.sse.shape == enc.sse.shape
    # the nearest-rounding encoder's rows are valid input too
    near = encode_rows(w, hs.diagonal().clone(), 4, 64, "fast")
    assert refine_encoded(near, w, hs, 4, 64, passes=1).codes.shape == near.codes.shape


def test_encode_with_picks_refines_only_when_asked_and_keeps_the_ladders():
    from hi2.calibrate import Config, build_table, encode_with_picks
    from hi2.executor import Executor
    from hi2.format import dequantize
    from hi2.stats import ColumnStats

    torch.manual_seed(4)
    r, k = 64, 128
    x = torch.randn(600, k) * torch.exp(0.5 * torch.randn(k))
    st = ColumnStats(k)
    st.update(x)
    sigma = (x.T @ x) / x.shape[0]
    w = torch.randn(r, k) * 0.3
    ex = Executor(["cpu"])

    def encode(passes):
        cfg = Config(effort="fast", refine_passes=passes)
        table = build_table("t", w, st, cfg, ex)
        picks = torch.full((r,), table.candidates.index((2, 32)), dtype=torch.long)
        return encode_with_picks(w, st, table, picks, cfg, ex, sigma)

    base, refined = encode(0), encode(2)
    assert torch.equal(base.blk_params, refined.blk_params)
    assert torch.equal(base.row_consts, refined.row_consts)
    assert not torch.equal(base.codes, refined.codes)

    def output_error(t):
        e = dequantize(t).float() - w
        return float((e @ sigma * e).sum())

    assert output_error(refined) < 0.99 * output_error(base)
    # the default of Config is off, and the same call twice is deterministic
    assert torch.equal(encode(0).codes, base.codes)
    assert torch.equal(encode(2).codes, refined.codes)


def test_groups_of_different_candidates_share_a_sweep_with_the_same_result():
    from hi2.refine import refine_groups

    w2, hs, enc2, *_ = _setup(2, 32, rows=16, k=128, seed=7)
    w4, _, enc4, *_ = _setup(4, 64, rows=16, k=128, seed=7)
    # same Hessian for both groups: _setup draws it from the seed, so the rows differ only
    both = refine_groups([(w2, enc2, 2, 32), (w4, enc4, 4, 64)], hs, passes=2)
    alone2 = refine_groups([(w2, enc2, 2, 32)], hs, passes=2)[0]
    alone4 = refine_groups([(w4, enc4, 4, 64)], hs, passes=2)[0]
    assert torch.equal(both[0], alone2) and torch.equal(both[1], alone4)
    assert both[0].dtype == enc2.codes.dtype
    assert int(both[0].max()) <= 3 and int(both[1].max()) <= 15
