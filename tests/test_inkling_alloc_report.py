# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""``experiments/inkling_alloc_report.py``: the shares are weighted by weights (INT2/4/8) and by
rows (blocks), the groups and projections add up to the whole, the stock size follows the expert
format of the layer, and a report without routed experts is refused."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _import():
    name = "hi2_test_inkling_alloc_report"
    if name in sys.modules:
        return sys.modules[name]
    path = (
        Path(__file__).resolve().parents[1] / "experiments" / "inkling_alloc_report.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _tensor(layer, proj, rows, cols, bpw, f4, f8, hist, expert=0):
    """A tensor of ``rows`` x ``cols`` weights; ``hist`` gives the share of the rows per block."""
    name = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
    return name, {
        "shape": [rows, cols],
        "bpw": bpw,
        "frac_int4": f4,
        "frac_int8": f8,
        "block_hist": {b: round(share * rows) for b, share in hist.items()},
    }


@pytest.fixture
def report():
    # layer 0 (NVFP4 in the stock model): 10,000 x 100 weights at 2.5 bpw, all INT2, rows at block 64;
    # 30,000 x 100 at 4.5 bpw, all INT4, a third of the rows at block 16, two thirds at 32
    # layer 20 (MXFP8): 20,000 x 200 at 6.0 bpw, half INT4 and half INT8, rows at block 32
    # (the widths differ so that a share by rows is not a share by weights)
    tensors = dict(
        [
            _tensor(0, "gate_proj", 10_000, 100, 2.5, 0.0, 0.0, {"64": 1.0}),
            _tensor(
                0, "down_proj", 30_000, 100, 4.5, 1.0, 0.0, {"16": 1 / 3, "32": 2 / 3}
            ),
            _tensor(20, "up_proj", 20_000, 200, 6.0, 0.5, 0.5, {"32": 1.0}, expert=7),
            (
                "model.language_model.layers.0.self_attn.q_proj.weight",
                {"shape": [4, 4]},
            ),
            ("lm_head.weight", {"shape": [5, 5], "bpw": 16}),
        ]
    )
    return {"summary": {}, "tensors": tensors}


def test_shares_are_weighted_by_weights_and_blocks_by_rows(report):
    a = _import().allocation_by_group(report)
    whole = a["all"]
    assert whole["tensors"] == 3 and whole["weights_g"] == pytest.approx(0.008)
    # INT2: 1e6 of 8e6 weights; INT4: 3e6 + 2e6; INT8: 2e6
    assert whole["int2"] == pytest.approx(1 / 8)
    assert whole["int4"] == pytest.approx(5 / 8)
    assert whole["int8"] == pytest.approx(2 / 8)
    assert whole["int2"] + whole["int4"] + whole["int8"] == pytest.approx(1.0)
    # blocks count rows: 60,000 rows, 10,000 at 64, 10,000 at 16, 40,000 at 32
    assert (whole["b16"], whole["b32"], whole["b64"]) == pytest.approx(
        (1 / 6, 4 / 6, 1 / 6), abs=1e-4
    )
    # bits per weight: (1e6 x 2.5 + 3e6 x 4.5 + 4e6 x 6.0) / 8e6
    assert whole["bpw"] == pytest.approx((2.5 + 13.5 + 24.0) / 8)
    assert whole["gib"] == pytest.approx(40e6 / 8 / 2**30, abs=1e-3)


def test_groups_projections_and_layers_partition_the_experts(report):
    a = _import().allocation_by_group(report)
    for scopes in (
        ("L0-5", "L20-23"),
        ("gate_proj", "up_proj", "down_proj"),
        ("layer 00", "layer 20"),
    ):
        assert sum(a[s]["weights_g"] for s in scopes) == pytest.approx(
            a["all"]["weights_g"]
        )
        assert sum(a[s]["tensors"] for s in scopes) == a["all"]["tensors"]
    assert "L6-17" not in a and "layer 05" not in a
    assert a["layer 20"]["int8"] == pytest.approx(0.5)
    assert a["gate_proj"]["int2"] == pytest.approx(1.0)


def test_layer_groups_cover_the_24_layers_without_overlap():
    tensors = dict(
        _tensor(layer, "gate_proj", 100, 10, 3.0, 0.0, 0.0, {"16": 1.0})
        for layer in range(24)
    )
    a = _import().allocation_by_group({"tensors": tensors})
    groups = ("L0-5", "L6-17", "L18-19", "L20-23")
    assert [a[g]["tensors"] for g in groups] == [6, 12, 2, 4]
    assert (
        a["all"]["tensors"] == 24
        and len([k for k in a if k.startswith("layer ")]) == 24
    )


def test_the_stock_size_follows_the_format_of_the_layer(report):
    a = _import().allocation_by_group(report)
    gib = 8 * 2**30
    assert a["layer 00"]["stock_gib"] == pytest.approx(4e6 * 4.5 / gib, abs=1e-3)
    assert a["layer 20"]["stock_gib"] == pytest.approx(4e6 * 8.25 / gib, abs=1e-3)
    # the same weights cost 8.25 bits in an MXFP8 layer against 4.5 in an NVFP4 one
    assert a["layer 20"]["stock_gib"] > a["layer 00"]["stock_gib"]


def test_command_line_prints_writes_and_refuses(report, tmp_path, capsys):
    mod = _import()
    run = tmp_path / "run"
    run.mkdir()
    (run / "hi2_report.json").write_text(json.dumps(report))
    out = tmp_path / "alloc.json"
    assert mod.main([str(run), "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "INT2 12.5 % INT4 62.5 % INT8 25.0 %" in text and "L0-5" in text
    assert json.loads(out.read_text())["all"]["tensors"] == 3
    assert mod.main([str(run / "hi2_report.json"), "--markdown"]) == 0
    assert "| layer 20 |" in capsys.readouterr().out
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"tensors": {"lm_head.weight": {"shape": [2, 2]}}}))
    assert mod.main([str(empty)]) == 1
