# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""``experiments/inkling_target_table.py``: the sizes come from the report and from the files on disk,
the groups' bits and predicted errors are weighted by weights, a missing evaluation prints as "-", and the
rank correlation of the groups' KL with their bits, INT2 share, predicted error and weight error is the
textbook Spearman coefficient."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

GIB = 2**30


def _import():
    name = "hi2_test_inkling_target_table"
    if name in sys.modules:
        return sys.modules[name]
    path = (
        Path(__file__).resolve().parents[1] / "experiments" / "inkling_target_table.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


WEIGHTS = 100_000 * 1_000  # one expert tensor per layer


def _report(bpw, int4, err):
    """One expert tensor of 1e8 weights in each of four layers."""
    tensors = {}
    for i in range(4):
        tensors[f"model.language_model.layers.{i}.mlp.experts.0.gate_proj.weight"] = {
            "shape": [100_000, 1_000],
            "bpw": bpw[i],
            "frac_int4": int4[i],
            "frac_int8": 0.0,
            "block_hist": {"64": 100_000},
            "pred_rel_err": err[i],
        }
    tensors["lm_head.weight"] = {"shape": [5, 5], "bpw": 16}
    return {"summary": {}, "tensors": tensors}


def _eval(kl, top1=0.9, ppl=1.01, sem=0.01, groups=None, rms=None):
    ev = {
        "student": {
            "kl_mean": kl,
            "kl_sem": sem,
            "top1_agreement": top1,
            "ppl_ratio": ppl,
        }
    }
    if groups:
        ev["groups"] = {
            k: {"kl_mean": v, "kl_sem": 0.001, "top1_agreement": 0.95, "ppl_ratio": 1.0}
            for k, v in groups.items()
        }
    if rms:
        ev["weight_error_per_layer"] = {
            str(i): {"relative_rms": v} for i, v in enumerate(rms)
        }
    return ev


@pytest.fixture
def tree(tmp_path):
    runs, evals = tmp_path / "runs", tmp_path / "eval"
    evals.mkdir()
    # 3.0: fewer bits everywhere; 3.5: more bits in the later layers
    specs = {
        "3.0": ([2.5, 2.5, 2.5, 2.5], [0.5, 0.5, 0.5, 0.5], [0.04, 0.03, 0.02, 0.01]),
        "3.5": ([3.0, 3.2, 3.4, 3.6], [0.6, 0.7, 0.8, 0.9], [0.04, 0.03, 0.02, 0.01]),
    }
    for t, (bpw, int4, err) in specs.items():
        d = runs / f"bpw{t}"
        d.mkdir(parents=True)
        (d / "hi2_report.json").write_text(json.dumps(_report(bpw, int4, err)))
        for name, size in (
            ("model-00001-of-00001.safetensors", GIB),
            ("model-vision.safetensors", GIB // 2),
        ):
            with open(d / name, "wb") as f:
                f.truncate(size)  # sparse: the size on disk is what is read
    (evals / "bpw3.5_mixed.json").write_text(json.dumps(_eval(0.10, 0.93, 1.0095)))
    (evals / "bpw3.0_mixed.json").write_text(json.dumps(_eval(0.12, 0.915, 1.0243)))
    (evals / "bpw3.5_wiki.json").write_text(json.dumps(_eval(0.047, 0.913, 1.033)))
    # KL falls with the bits and with the later layers; the weight error is not monotone
    (evals / "bpw3.5_fine.json").write_text(
        json.dumps(
            _eval(
                0.10,
                groups={"0": 0.4, "1": 0.3, "2": 0.2, "3": 0.1},
                rms=[0.1, 0.3, 0.2, 0.4],
            )
        )
    )
    return runs, evals


def test_ranks_average_ties_and_spearman_is_the_textbook_coefficient():
    m = _import()
    assert m.ranks([10.0, 30.0, 20.0]) == [1.0, 3.0, 2.0]
    assert m.ranks([1.0, 2.0, 2.0, 3.0]) == [1.0, 2.5, 2.5, 4.0]
    assert m.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert m.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    # rank differences 3, 0, 0, -3: 1 - 6 * 18 / (4 * 15)
    assert m.spearman([4, 3, 2, 1], [1, 3, 2, 4]) == pytest.approx(-0.8)
    assert m.spearman([1, 2], [1, 2]) is None  # too few
    assert m.spearman([1, 2, 3], [5, 5, 5]) is None  # constant


def test_sizes_come_from_the_report_and_from_the_files(tree):
    m = _import()
    runs, evals = tree
    data = m.collect(runs, evals, ["3.0", "3.5"], None)
    e30, e35 = data["targets"]["3.0"], data["targets"]["3.5"]
    # 4 tensors of 1e8 weights: bits per weight 2.5 and (3.0 + 3.2 + 3.4 + 3.6) / 4 = 3.3
    assert e30["experts"]["weights_g"] == pytest.approx(0.4)
    assert e30["experts"]["bpw"] == pytest.approx(2.5)
    assert e35["experts"]["bpw"] == pytest.approx(3.3)
    assert e35["experts"]["gib"] == pytest.approx(4e8 * 3.3 / 8 / GIB, abs=1e-3)
    assert e35["total_gib"] == pytest.approx(1.5)  # the text file and the vision tower
    assert e35["experts"]["int4"] == pytest.approx((0.6 + 0.7 + 0.8 + 0.9) / 4)
    text = m.render(data)
    row = next(r for r in text.splitlines() if r.startswith("| 3.5 |"))
    assert (
        "| 0.154 | 1.346 | 1.500 | 3.300 |" in row
    )  # experts, the rest of the directory, the total, bpw
    assert (
        "0.0470 +- 0.0100 / 0.913 / x1.0330" in row
    )  # the wiki set of 3.5, with its own sem
    wiki30 = next(r for r in text.splitlines() if r.startswith("| 3.0 |")).split("|")
    assert wiki30[-3].strip() == "-"  # no wiki evaluation of 3.0: a dash, not a zero


def test_int8_rows_are_not_counted_as_int2():
    m = _import()
    report = {
        "tensors": {
            "model.language_model.layers.0.mlp.experts.0.up_proj.weight": {
                "shape": [10, 100],
                "bpw": 5.0,
                "frac_int4": 0.5,
                "frac_int8": 0.25,
                "pred_rel_err": 0.02,
            },
            "model.language_model.layers.0.mlp.experts.0.down_proj.bias": {
                "shape": [10, 100],
                "bpw": 16.0,
            },
            "model.language_model.embed_tokens.weight": {
                "shape": [10, 100],
                "bpw": 16.0,
            },
        }
    }
    layers = m.layer_stats(report)
    assert list(layers) == [0]  # only the expert weights
    assert layers[0]["w2"] == pytest.approx(250.0)
    assert layers[0]["weights"] == 1000


def test_the_layers_of_a_group_are_weighted_by_their_weights():
    m = _import()
    layers = {
        0: {
            "weights": 1_000,
            "bits": 1_000 * 2.0,
            "w2": 1_000.0,
            "err": 1_000 * 0.1,
            "err2": 1_000 * 0.01,
        },
        1: {
            "weights": 3_000,
            "bits": 3_000 * 4.0,
            "w2": 0.0,
            "err": 3_000 * 0.02,
            "err2": 3_000 * 0.0004,
        },
    }
    g = m.group_stats(layers, 0, 1)
    assert g["bpw"] == pytest.approx((2.0 + 12.0) / 4)
    assert g["int2"] == pytest.approx(0.25)
    assert g["pred_err"] == pytest.approx((0.1 + 0.06) / 4)
    assert g["pred_err_rms"] == pytest.approx(((0.01 + 0.0012) / 4) ** 0.5)
    assert m.group_stats(layers, 5, 6) == {}
    assert m.parse_group("3-5") == (3, 5) and m.parse_group("7") == (7, 7)


def test_the_group_table_ranks_the_kl_against_bits_int2_predicted_and_measured_error(
    tree,
):
    m = _import()
    runs, evals = tree
    text = m.render(m.collect(runs, evals, ["3.0", "3.5"], None))
    assert "Groups of layers at 3.5 bpw" in text
    assert "Groups of layers at 3.0" not in text  # no fine evaluation of 3.0
    # KL [0.4 .. 0.1] against bits [3.0 .. 3.6]: -1; against the INT2 share (falls with the bits): +1;
    # against the predicted error [0.04 .. 0.01]: +1; against the weight error [0.1, 0.3, 0.2, 0.4]: -0.8
    line = next(x for x in text.splitlines() if x.startswith("Spearman"))
    assert "bits per weight -1.00" in line and "INT2 share +1.00" in line
    assert "predicted relative error +1.00" in line and "weight RMS error -0.80" in line
    assert "(4 groups)" in line
    # the first group: 3.0 bits at 3.5, 2.5 at 3.0, KL 0.4, weight error 0.1
    row = next(x for x in text.splitlines() if x.startswith("| 0 |"))
    assert "0.4000" in row and "3.000" in row and "2.500" in row and "0.100" in row
    # the KL per squared error: 0.4 / 0.04^2 = 250 and 0.4 / 0.1^2 = 40 in the first group
    assert row.endswith("| 0.100 | 250.0 | 40.0 |")
    # over the four groups: 250, 333.3, 500, 1000 (x4.0) and 40, 3.33, 5, 0.625 (x64.0)
    assert "250.0 to 1000.0 (x4.0) per squared predicted error" in text
    assert "0.6 to 40.0 (x64.0) per squared weight error" in text


def test_kl_per_squared_error_uses_the_rms_of_the_predicted_error(tmp_path):
    m = _import()
    run = tmp_path / "runs" / "bpw3.5"
    run.mkdir(parents=True)
    # one layer, two tensors of equal weights: errors 0.02 and 0.06 (mean 0.04, rms^2 0.002)
    tensors = {}
    for name, err in (("0.up_proj", 0.02), ("1.up_proj", 0.06)):
        tensors[f"model.language_model.layers.0.mlp.experts.{name}.weight"] = {
            "shape": [10, 100],
            "bpw": 3.5,
            "frac_int4": 0.5,
            "frac_int8": 0.0,
            "pred_rel_err": err,
        }
    (run / "hi2_report.json").write_text(json.dumps({"tensors": tensors}))
    evals = tmp_path / "eval"
    evals.mkdir()
    (evals / "bpw3.5_fine.json").write_text(
        json.dumps(_eval(0.4, groups={"0": 0.4}, rms=[0.2]))
    )
    text = m.render(m.collect(tmp_path / "runs", evals, ["3.5"], None))
    row = next(x for x in text.splitlines() if x.startswith("| 0 |"))
    assert "| 0.0400 / 0.0447 |" in row  # mean and rms of the predicted error
    assert row.endswith("| 0.200 | 200.0 | 10.0 |")  # 0.4 / 0.002 and 0.4 / 0.2^2


def _load_report(shift):
    """Four layers of eight experts: the load gets more skewed with the layer, the bits follow the
    tokens inside a layer and fall with the layer (``shift`` moves them all)."""
    tensors = {}
    for layer in range(4):
        for i in range(8):
            base = f"model.language_model.layers.{layer}.mlp.experts.{i}."
            tensors[base + "gate_proj.weight"] = {
                "shape": [10, 100],
                "bpw": 4.0 - 0.4 * layer + 0.1 * i + shift,
                "frac_int4": 0.5,
                "n_tokens": (i + 1) ** (1 + layer),
            }
            # the up projection sees the same tokens: its count must not be read
            tensors[base + "up_proj.weight"] = {
                "shape": [10, 100],
                "bpw": 1.0,
                "frac_int4": 0.5,
                "n_tokens": 1,
            }
    return {"tensors": tensors}


def test_expert_load_is_read_from_the_gate_and_ranks_tokens_against_bits():
    m = _import()
    report = _load_report(0.0)
    # shuffle one layer's tensors: the order of the report must not matter
    items = list(report["tensors"].items())
    report["tensors"] = dict(reversed(items))
    load = m.expert_load(report)
    a = load[0]  # tokens 1..8
    assert a["experts"] == 8 and a["mean_tokens"] == pytest.approx(4.5)
    assert a["cv"] == pytest.approx((5.25**0.5) / 4.5)
    assert a["hot"] == pytest.approx(
        8 / 36
    )  # the hottest eighth of 8 experts is one expert
    assert a["bpw"] == pytest.approx(4.35)
    assert a["bpw_cold"] == pytest.approx(
        (4.0 + 4.1) / 2
    )  # the two least routed experts
    assert a["bpw_hot"] == pytest.approx((4.6 + 4.7) / 2)
    assert a["rho"] == pytest.approx(1.0)
    assert load[3]["cv"] > load[0]["cv"] and load[3]["hot"] > load[0]["hot"]
    # a layer whose bits fall with the tokens
    for i in range(8):
        report["tensors"][
            f"model.language_model.layers.0.mlp.experts.{i}.gate_proj.weight"
        ]["bpw"] = 5.0 - 0.1 * i
    assert m.expert_load(report)[0]["rho"] == pytest.approx(-1.0)
    assert m.expert_load({"tensors": {}}) == {}


def test_bits_of_the_cold_and_hot_quarters_are_weighted_by_weights():
    m = _import()
    tensors = {}
    for i in range(8):
        tensors[f"model.language_model.layers.0.mlp.experts.{i}.gate_proj.weight"] = {
            # the least routed expert is three times as large as the others
            "shape": [30 if i == 0 else 10, 100],
            "bpw": 4.0 + 0.1 * i,
            "n_tokens": 1 + i,
        }
    a = m.expert_load({"tensors": tensors})[0]
    assert a["bpw_cold"] == pytest.approx((3 * 4.0 + 4.1) / 4)  # experts 0 and 1
    assert a["bpw_hot"] == pytest.approx((4.6 + 4.7) / 2)
    assert a["bpw"] == pytest.approx(
        (3 * 4.0 + sum(4.0 + 0.1 * i for i in range(1, 8))) / 10
    )


def test_the_two_layer_correlations_are_told_apart_in_the_summary_line():
    m = _import()
    load = {
        i: {
            "experts": 8,
            "mean_tokens": 100.0,
            "cv": 0.1 * (i + 1),  # grows with the layer, the bits fall: -1
            "hot": 0.4 - 0.1 * i,  # falls with the layer like the bits: +1
            "bpw": 4.0 - i,
            "bpw_cold": 1.0,
            "bpw_hot": 2.0,
            "rho": rho,
        }
        for i, rho in enumerate((0.9, 0.8, None, 1.0))
    }
    text = m.render({"stock": None, "targets": {"3.0": {"evals": {}, "load": load}}})
    assert (
        "against the CV of its load -1.00 and against the share of its hottest eighth +1.00"
        in text
    )
    assert (
        "within a layer: mean +0.90." in text
    )  # the layer without a correlation is not counted


def test_the_load_table_and_its_correlations(tmp_path):
    m = _import()
    runs, evals = tmp_path / "runs", tmp_path / "eval"
    evals.mkdir()
    for t, shift in (("3.0", -0.5), ("3.5", 0.0)):
        (runs / f"bpw{t}").mkdir(parents=True)
        (runs / f"bpw{t}" / "hi2_report.json").write_text(
            json.dumps(_load_report(shift))
        )
    text = m.render(m.collect(runs, evals, ["3.0", "3.5"], None))
    assert "#### Expert load against bits" in text
    # layer 0: 4.5 tokens per expert on average (printed as an integer), CV 0.51, the hottest eighth
    # (one expert of eight) takes 8 / 36 = 22 % of the tokens; the bits of its two least and two most
    # routed experts at 3.0 (shift -0.5) and at 3.5
    assert "| 00 | 4 | 0.51 | 22 | 3.55 / 4.15 (+1.00) | 4.05 / 4.65 (+1.00) |" in text
    assert (
        "| 03 | 1096 | 1.25 | 47 | 2.35 / 2.95 (+1.00) | 2.85 / 3.45 (+1.00) |" in text
    )
    # the layer bits fall as the skew grows: -1 against both measures, and every expert's bits follow its tokens
    assert text.count("a layer's bits per weight against the CV of its load -1.00") == 2
    assert text.count("against the share of its hottest eighth -1.00") == 2
    assert text.count("within a layer: mean +1.00") == 2


def test_bits_per_layer_are_listed_for_every_target(tree):
    m = _import()
    runs, evals = tree
    text = m.render(m.collect(runs, evals, ["3.0", "3.5"], None))
    rows = [x for x in text.splitlines() if x[:2] == "| " and x[2:4].isdigit()]
    assert rows == [
        "| 00 | 2.500 | 3.000 |",
        "| 01 | 2.500 | 3.200 |",
        "| 02 | 2.500 | 3.400 |",
        "| 03 | 2.500 | 3.600 |",
    ]


def test_the_stock_directory_sizes_are_reported(tree, tmp_path):
    m = _import()
    runs, evals = tree
    stock = tmp_path / "stock"
    stock.mkdir()
    for name, size in (
        ("model-00001-of-00002.safetensors", GIB),
        ("mtp.safetensors", GIB // 4),
    ):
        with open(stock / name, "wb") as f:
            f.truncate(size)
    text = m.render(m.collect(runs, evals, ["3.5"], stock))
    assert "stock files 1.250 GiB (MTP 0.250)" in text


def test_command_line_prints_writes_and_refuses(tree, tmp_path, capsys):
    m = _import()
    runs, evals = tree
    out = tmp_path / "table.json"
    assert (
        m.main(
            [
                "--runs",
                str(runs),
                "--evals",
                str(evals),
                "--targets",
                "3.0",
                "3.5",
                "--json",
                str(out),
            ]
        )
        == 0
    )
    assert "#### Targets" in capsys.readouterr().out
    data = json.loads(out.read_text())
    assert data["targets"]["3.5"]["layers"]["3"]["bpw"] == pytest.approx(3.6)
    assert data["targets"]["3.5"]["evals"]["fine"]["groups"]["0"] == pytest.approx(0.4)
    assert m.main(["--runs", str(tmp_path / "none"), "--evals", str(evals)]) == 1


# ------------------------------------------- the windows, the routing and the tagged targets


def _eval_windows(kls, routing=None, sem=0.01):
    ev = _eval(sum(kls) / len(kls), sem=sem)
    ev["student"]["kl_per_window"] = list(kls)
    if routing is not None:
        ev["routing"] = routing
    return ev


def _routing_block(experts, assignments, bits, per_window):
    rest = 1.0 - assignments
    return {
        "experts": {"int2": experts, "int4": 1.0 - experts, "int8": 0.0},
        "assignments": {"int2": assignments, "int4": rest, "int8": 0.0},
        "mean_bits": bits,
        "per_window": {"int2": per_window},
    }


@pytest.fixture
def sweep(tmp_path):
    """Three encodes of one size (3.5, then 3.5 with the load flattened by powers 0.5 and 0) and the
    mixed evaluations of 32 windows, the first as ``routing`` only."""
    runs, evals = tmp_path / "runs", tmp_path / "eval"
    evals.mkdir()
    for t, bpw in (("3.5", 3.3), ("3.5-p0.5", 3.4), ("3.5-p0.0", 3.5)):
        d = runs / f"bpw{t}"
        d.mkdir(parents=True)
        (d / "hi2_report.json").write_text(
            json.dumps(_report([bpw] * 4, [0.5] * 4, [0.03] * 4))
        )
        with open(d / "model-00001-of-00001.safetensors", "wb") as f:
            f.truncate(GIB)
    base = [0.05 + 0.001 * w for w in range(32)]
    base[17] = 0.5
    base[30] = 0.3
    flat = [x * 0.9 for x in base]
    flat[17] = 0.4
    flat[30] = 0.35
    route = {
        "groups": {
            "0-5": _routing_block(
                0.30, 0.20, 3.1, [0.2 + 0.001 * w for w in range(32)]
            ),
            "6-17": _routing_block(0.40, 0.25, 3.3, [0.3] * 32),
        },
        "all_layers": _routing_block(0.35, 0.22, 3.2, [0.25] * 32),
    }
    (evals / "bpw3.5_routing.json").write_text(json.dumps(_eval_windows(base, route)))
    (evals / "bpw3.5-p0.5_mixed.json").write_text(json.dumps(_eval_windows(flat)))
    (evals / "bpw3.5-p0.0_routing.json").write_text(
        json.dumps(_eval_windows(flat, route))
    )
    return runs, evals, base, flat


def test_a_paired_difference_is_worked_by_hand():
    m = _import()
    d = m.paired([0.1, 0.2, 0.3], [0.2, 0.2, 0.2])  # differences -0.1, 0, 0.1
    assert d["mean"] == pytest.approx(0.0, abs=1e-12)
    assert d["sem"] == pytest.approx((0.02 / 2 / 3) ** 0.5)
    assert d["better"] == 1 and d["windows"] == 3
    assert m.paired([0.5], [0.4])["sem"] is None  # one window has no spread


def test_the_routing_evaluation_stands_for_the_mixed_one_and_tags_name_targets(sweep):
    m = _import()
    runs, evals, base, flat = sweep
    data = m.collect(runs, evals, ["3.5", "3.5-p0.5", "3.5-p0.0"], None)
    table = m.render(data)
    rows: dict[str, str] = {}
    for ln in table.splitlines():
        if ln.startswith("| 3.5"):
            rows.setdefault(
                ln.split("|")[1].strip(), ln
            )  # the first: the targets table
    mean = f"{sum(base) / 32:.4f}"
    assert mean in rows["3.5"] and f"{sum(flat) / 32:.4f}" in rows["3.5-p0.5"]
    assert f"{sum(flat) / 32:.4f}" in rows["3.5-p0.0"]  # a routing eval too
    assert "3.300" in rows["3.5"] and "3.500" in rows["3.5-p0.0"]  # their own reports


def test_the_windows_table_lists_the_windows_the_clean_mean_and_the_paired_difference(
    sweep,
):
    m = _import()
    runs, evals, base, flat = sweep
    data = m.collect(runs, evals, ["3.5", "3.5-p0.5", "3.5-p0.0"], None)
    lines = m.render(data, windows=(17, 30), clean=24).splitlines()
    start = lines.index("#### Windows of the mixed set (KL per window)")
    head, _, *rows = lines[start + 2 : start + 7]
    assert "window 17" in head and "window 30" in head and "windows 0-23 mean" in head
    first, second = rows[0], rows[1]
    cells = [c.strip() for c in first.split("|")[1:-1]]
    assert cells[0] == "3.5" and cells[2] == "0.5000" and cells[3] == "0.3000"
    assert cells[4] == f"{sum(base[:24]) / 24:.4f}" and cells[5:] == ["-", "-"]
    cells = [c.strip() for c in second.split("|")[1:-1]]
    assert cells[0] == "3.5-p0.5" and cells[2] == "0.4000" and cells[3] == "0.3500"
    want = m.paired(flat, base)
    assert cells[5].startswith(f"{want['mean']:+.4f} +- ")
    assert cells[6] == f"{want['better']} of 32"


def test_the_routing_table_gives_the_class_shares_by_group_and_the_windows(sweep):
    m = _import()
    runs, evals, *_ = sweep
    data = m.collect(runs, evals, ["3.5", "3.5-p0.5", "3.5-p0.0"], None)
    text = m.render(data, windows=(17,))
    assert "#### Routing of the student" in text
    rows = [ln for ln in text.splitlines() if ln.startswith("| 3.5") and " 0-5 " in ln]
    assert len(rows) == 2  # 3.5 and 3.5-p0.0; 3.5-p0.5 has no routing log
    cells = [c.strip() for c in rows[0].split("|")[1:-1]]
    assert cells == ["3.5", "0-5", "30.0", "20.0", "80.0", "0.0", "3.100", "21.7"]
    assert not any(ln.startswith("| 3.5-p0.5 | 0-5") for ln in text.splitlines())
    whole = [ln for ln in text.splitlines() if ln.startswith("| 3.5 | all")]
    assert whole and "35.0" in whole[0] and "22.0" in whole[0]


def test_nothing_is_added_without_windows_or_routing(tree):
    m = _import()
    runs, evals = tree
    text = m.render(m.collect(runs, evals, ["3.0", "3.5"], None))
    assert "#### Windows" not in text and "#### Routing" not in text


def test_the_command_line_takes_windows_and_the_clean_count(sweep, capsys, tmp_path):
    m = _import()
    runs, evals, *_ = sweep
    out = tmp_path / "t.json"
    argv = ["--runs", str(runs), "--evals", str(evals), "--windows", "17", "30"]
    argv += ["--clean", "24", "--targets", "3.5", "3.5-p0.5", "3.5-p0.0"]
    assert m.main([*argv, "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert (
        "window 17" in text and "windows 0-23 mean" in text and "#### Routing" in text
    )
    saved = json.loads(out.read_text())
    assert saved["targets"]["3.5"]["evals"]["routing"]["routing"]["groups"]["0-5"]
