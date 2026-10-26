# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""``hi2 size`` (CPU): the byte accounting equals the sizes of the tensors on disk, the
weight counts match the tensor shapes, and both expert layouts give the same linear
totals."""

import pytest
import torch

pytest.importorskip("safetensors")

from test_stack import _write_per_expert  # noqa: E402

from hi2.size import GROUPS, format_report, size_report  # noqa: E402
from hi2.stack import restack  # noqa: E402


def _tensor_bytes(d):
    from safetensors import safe_open

    from hi2.checkpoint import weight_map

    wmap = weight_map(d)
    out = {}
    for sh in sorted(set(wmap.values())):
        with safe_open(d / sh, "pt") as f:
            for k in f.keys():  # noqa: SIM118
                t = f.get_tensor(k)
                out[k] = t.numel() * t.element_size()
    return out


def test_bytes_add_up_to_the_tensors_on_disk(tmp_path):
    d = tmp_path / "pe"
    _write_per_expert(d)
    rep = size_report(d)
    on_disk = _tensor_bytes(d)
    assert rep["tensor_bytes"] == sum(on_disk.values())
    hi2 = sum(v for k, v in on_disk.items() if ".hi2_" in k)
    assert rep["linears"]["bytes"] == hi2
    assert sum(rep["linears"]["by_field"].values()) == hi2
    assert set(rep["linears"]["by_field"]) == set(GROUPS)
    assert rep["dense_bytes"] == rep["tensor_bytes"] - hi2
    assert rep["layout"] == "per-expert"


def test_weight_counts_and_precision_shares(tmp_path):
    d = tmp_path / "pe"
    _write_per_expert(d, layers=2, experts=4)
    rep = size_report(d)
    # per layer: 4 experts x (gate 64x128, up 64x128, down 128x64) + q_proj 32x128
    per_layer = 4 * (64 * 128 + 64 * 128 + 128 * 64) + 32 * 128
    assert rep["linears"]["weights"] == 2 * per_layer
    share = rep["linears"]["precision_share"]
    assert set(share) <= {"INT2", "INT4", "INT8"}
    assert sum(share.values()) == pytest.approx(1.0)
    lin = rep["linears"]
    assert lin["bpw"] == pytest.approx(8 * lin["bytes"] / lin["weights"])
    assert sum(lin["bpw_by_field"].values()) == pytest.approx(lin["bpw"])


def test_stacked_layout_counts_the_same_weights(tmp_path):
    d, s = tmp_path / "pe", tmp_path / "st"
    _write_per_expert(d)
    restack(d, s, "stacked", log=lambda *_: None)
    a, b = size_report(d), size_report(s)
    assert b["layout"] == "stacked"
    assert a["linears"]["weights"] == b["linears"]["weights"]
    assert a["linears"]["precision_share"] == b["linears"]["precision_share"]
    for g in ("codes", "block parameters", "row constants"):
        assert a["linears"]["by_field"][g] == b["linears"]["by_field"][g]
    # a stack keeps one smoothing vector for gate and up, and fewer offsets
    for g in ("smoothing vector", "row index"):
        assert b["linears"]["by_field"][g] <= a["linears"]["by_field"][g]
    assert b["linears"]["bytes"] < a["linears"]["bytes"]
    assert b["tensor_bytes"] == sum(_tensor_bytes(s).values())


def test_groupings_and_text(tmp_path):
    d = tmp_path / "pe"
    _write_per_expert(d)
    kinds = size_report(d, by="kind")["rows"]
    assert "mlp.experts.*.gate_proj" in kinds and "self_attn.q_proj" in kinds
    assert set(size_report(d, by="layer")["rows"]) == {"0", "1"}
    assert len(size_report(d, by="tensor")["rows"]) > len(kinds)
    text = format_report(size_report(d), top=3)
    assert "Quantized linears" in text and "more (use --top)" in text


def test_source_ratio(tmp_path):
    from safetensors.torch import save_file

    d = tmp_path / "pe"
    _write_per_expert(d)
    src = tmp_path / "src"
    src.mkdir()
    save_file(
        {"model.layers.0.self_attn.q_proj.weight": torch.zeros(1000, 1000).half()},
        str(src / "model.safetensors"),
        metadata={"format": "pt"},
    )
    rep = size_report(d, src)
    assert rep["source"]["tensor_bytes"] == 2_000_000
    assert rep["source"]["ratio"] == pytest.approx(2_000_000 / rep["tensor_bytes"])
