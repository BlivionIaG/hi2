# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Stacked MoE layout (CPU): stack/split is byte-exact, the stack's offsets and shapes
hold the documented invariants, checkpoints convert both ways without re-encoding and
dequantize bit-identically."""

import json

import pytest
import torch

pytest.importorskip("safetensors")

from hi2.encode import encode_rows  # noqa: E402
from hi2.format import (  # noqa: E402
    FIELDS,
    HI2Tensor,
    assemble,
    dequantize,
    dequantize_stored,
)
from hi2.stack import (  # noqa: E402
    MARK,
    Hi2Checkpoint,
    StackError,
    build_stack,
    restack,
    split_stack,
)

K, N, H, E = 128, 64, 128, 5


def _mixed(r, k, seed, had=None, col_scale=None):
    """A tensor with per-row INT2/INT4 and B16/32/64 through the real encoder."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(r, k, generator=g)
    nbits = torch.tensor([2, 4])[torch.randint(0, 2, (r,), generator=g)]
    block = torch.tensor([16, 32, 64])[torch.randint(0, 3, (r,), generator=g)]
    pieces = []
    for nb in (2, 4):
        for b in (16, 32, 64):
            rows = ((nbits == nb) & (block == b)).nonzero().flatten()
            if rows.numel():
                enc = encode_rows(w[rows], torch.ones(k), nb, b, "fast")
                pieces.append((rows, enc.codes, enc.params, enc.consts))
    if had is None:
        had = torch.zeros(k // 16, dtype=torch.uint8)
    return assemble(pieces, nbits, block, had, k, col_scale)


def _expert(e, k=K, rows=(N, N), smooth=True):
    g = torch.Generator().manual_seed(100 + e)
    had = torch.randint(0, 2, (k // 16,), generator=g).to(torch.uint8)
    cs = (torch.rand(k, generator=g) + 0.5).half() if smooth else None
    return [
        _mixed(r, k, 10 * e + i, had.clone(), None if cs is None else cs.clone())
        for i, r in enumerate(rows)
    ]


def _same(a: HI2Tensor, b: HI2Tensor) -> bool:
    for f in FIELDS:
        x, y = getattr(a, f), getattr(b, f)
        if x.dtype != y.dtype or x.shape != y.shape or not torch.equal(x, y):
            return False
    if (a.col_scale is None) != (b.col_scale is None):
        return False
    if a.col_scale is not None and (
        a.col_scale.dtype != b.col_scale.dtype
        or not torch.equal(a.col_scale, b.col_scale)
    ):
        return False
    return a.shape == b.shape


def test_stack_invariants_and_byte_exact_split():
    experts = [_expert(e) for e in range(E)]
    d = build_stack(experts)
    r = 2 * N * E
    assert d["codes"].dtype == torch.int32 and d["codes"].dim() == 1
    assert d["blk_params"].dtype == torch.uint8 and d["blk_params"].dim() == 1
    for off in ("code_off", "param_off", MARK):
        assert d[off].dtype == torch.int32
    assert d["row_consts"].shape == (r, 3) and d["row_consts"].dtype == torch.float16
    assert d["row_nbits"].shape == d["row_block"].shape == (r,)
    assert d["had"].shape == (E, K // 16) and d["had"].dtype == torch.uint8
    assert d["col_scale"].shape == (E, K) and d["col_scale"].dtype == torch.float16
    assert d[MARK].tolist() == list(range(0, r + 1, N))
    # rebased offsets: one valid flat container over all R rows
    nb, bl = d["row_nbits"].long(), d["row_block"].long()
    co, po = d["code_off"].long(), d["param_off"].long()
    assert co[0] == 0 and torch.equal(co[1:] - co[:-1], K * nb // 32)
    assert po[0] == 0 and torch.equal(po[1:] - po[:-1], K // bl)
    assert co[-1] == d["codes"].numel() and 2 * po[-1] == d["blk_params"].numel()
    flat = HI2Tensor(
        **{f: d[f] for f in FIELDS if f != "had"}, had=d["had"][0], shape=(r, K)
    )
    want = torch.cat([dequantize_stored(t) for ex in experts for t in ex])
    assert torch.equal(dequantize_stored(flat), want)
    back = split_stack(d, 2)
    for e in range(E):
        for p in range(2):
            assert _same(back[e][p], experts[e][p]), (e, p)
            assert torch.equal(dequantize(back[e][p]), dequantize(experts[e][p]))


def test_down_stack_and_unsmoothed():
    experts = [_expert(e, k=N, rows=(H,), smooth=False) for e in range(3)]
    d = build_stack(experts)
    assert "col_scale" not in d and d["had"].shape == (3, N // 16)
    assert d[MARK].tolist() == [0, H, 2 * H, 3 * H]
    for e, (t,) in enumerate(split_stack(d, 1)):
        assert _same(t, experts[e][0])


def test_unstackable_groups_are_rejected():
    ex = [_expert(e) for e in range(2)]
    bad = [list(x) for x in ex]
    bad[1][1] = _mixed(
        N, K, 7, torch.ones(K // 16, dtype=torch.uint8), ex[1][0].col_scale
    )
    with pytest.raises(StackError, match="rotation"):
        build_stack(bad)
    mixed = [_expert(0), _expert(1, smooth=False)]
    with pytest.raises(StackError, match="smoothing"):
        build_stack(mixed)
    with pytest.raises(StackError, match="K="):
        build_stack([_expert(0), _expert(1, k=64)])


# ----------------------------------------------------- checkpoint level


def _write_per_expert(d, layers=2, experts=4, unstackable_layer=None):
    """A hand-made per-expert HI2 checkpoint over two shards; layer 1's experts are
    split across both shards."""
    from hi2.checkpoint import to_tensors
    from safetensors.torch import save_file

    d.mkdir(parents=True)
    shards = [{}, {}]
    report = {}
    for li in range(layers):
        pre = f"model.layers.{li}.mlp.experts"
        for e in range(experts):
            smooth = not (li == unstackable_layer and e == 1)
            gate, up = _expert(10 * li + e, smooth=smooth)
            (down,) = _expert(50 + 10 * li + e, k=N, rows=(H,), smooth=True)
            sh = 0 if li == 0 or e < experts // 2 else 1
            for proj, t in (("gate_proj", gate), ("up_proj", up), ("down_proj", down)):
                base = f"{pre}.{e}.{proj}"
                shards[sh].update(to_tensors(t, base))
                report[base + ".weight"] = {"bpw": 0.0}
        q = _mixed(32, H, 900 + li)
        shards[li % 2].update(to_tensors(q, f"model.layers.{li}.self_attn.q_proj"))
        shards[li % 2][f"model.layers.{li}.mlp.gate.weight"] = torch.randn(
            experts, H
        ).half()
    wmap = {}
    for i, s in enumerate(shards):
        name = f"model-0000{i + 1}-of-00002.safetensors"
        save_file(s, str(d / name), metadata={"format": "pt"})
        wmap.update(dict.fromkeys(s, name))
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": wmap})
    )
    (d / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_method": "rdna_hint_flex",
                    "version": "3.2",
                }
            }
        )
    )
    (d / "hi2_report.json").write_text(json.dumps({"summary": {}, "tensors": report}))
    (d / "tokenizer.json").write_text("{}")
    return wmap


def _all_tensors(d):
    from hi2.checkpoint import weight_map
    from safetensors import safe_open

    out = {}
    wmap = weight_map(d)
    for sh in sorted(set(wmap.values())):
        with safe_open(d / sh, "pt") as f:
            for k in f.keys():  # noqa: SIM118
                out[k] = f.get_tensor(k)
    return out


def test_checkpoint_restack_both_ways(tmp_path):
    from hi2.checkpoint import load_dequantized, weight_map

    src = tmp_path / "pe"
    _write_per_expert(src)
    st = tmp_path / "st"
    info = restack(src, st, "stacked", log=lambda *_: None)
    assert info["stacks"] == 4
    assert info["before"]["tensors"] == 2 * 4 * 3 * 9 + 2 * 8 + 2
    assert info["after"]["tensors"] == 2 * 2 * 10 + 2 * 8 + 2
    wmap = weight_map(st)
    for li in range(2):  # the two stacks of a layer share a shard
        pre = f"model.layers.{li}.mlp.experts"
        assert wmap[f"{pre}.gate_up.hi2_codes"] == wmap[f"{pre}.down.hi2_{MARK}"]
        assert not any(k.startswith(f"{pre}.0.") for k in wmap)
    qc = json.loads((st / "config.json").read_text())["quantization_config"]
    assert qc["moe_layout"] == "stacked"
    assert qc["moe_stacks"] == {
        "down": ["down_proj"],
        "gate_up": ["gate_proj", "up_proj"],
    }
    assert (
        json.loads((st / "hi2_report.json").read_text())["summary"]["moe_layout"]
        == "stacked"
    )
    assert (st / "tokenizer.json").exists()
    ck_a, ck_b = Hi2Checkpoint(src), Hi2Checkpoint(st)
    assert set(ck_a.logical()) == set(ck_b.logical())
    for base in ck_a.logical():
        assert _same(ck_a.get(base), ck_b.get(base)), base
    da, db = load_dequantized(src), load_dequantized(st)
    assert da.keys() == db.keys()
    for k in da:
        assert da[k].dtype == db[k].dtype and torch.equal(da[k], db[k]), k
    # and back: every array byte-identical to the original checkpoint
    back = tmp_path / "back"
    restack(st, back, "per-expert", log=lambda *_: None)
    ta, tb = _all_tensors(src), _all_tensors(back)
    assert ta.keys() == tb.keys()
    for k in ta:
        assert ta[k].dtype == tb[k].dtype and ta[k].shape == tb[k].shape, k
        assert torch.equal(ta[k], tb[k]), k
    assert (
        "moe_stacks"
        not in json.loads((back / "config.json").read_text())["quantization_config"]
    )
    with pytest.raises(SystemExit):
        restack(src, st, "stacked", log=lambda *_: None)


def test_unstackable_group_stays_per_expert(tmp_path):
    from hi2.checkpoint import load_dequantized, weight_map

    src = tmp_path / "pe"
    _write_per_expert(src, unstackable_layer=1)
    st = tmp_path / "st"
    msgs = []
    info = restack(src, st, "stacked", log=msgs.append)
    assert info["stacks"] == 3
    assert any("layers.1.mlp.experts.gate_up kept per expert" in m for m in msgs)
    wmap = weight_map(st)
    assert "model.layers.1.mlp.experts.1.gate_proj.hi2_codes" in wmap
    assert "model.layers.1.mlp.experts.down.hi2_part_off" in wmap
    da, db = load_dequantized(src), load_dequantized(st)
    assert all(torch.equal(da[k], db[k]) for k in da)
