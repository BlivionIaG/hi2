# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""bench paired statistics against known numbers, and compare.py end to end."""

import hashlib
import json
import math
import random

import pytest

from bench import compare as cmp


def test_mcnemar_exact_known_values():
    # b=10, c=2: 2 * P(X <= 2 | n=12, 1/2) = 2 * (1 + 12 + 66) / 4096
    assert cmp.mcnemar_exact(10, 2) == pytest.approx(158 / 4096)
    assert cmp.mcnemar_exact(2, 10) == pytest.approx(158 / 4096)
    assert cmp.mcnemar_exact(0, 0) == 1.0
    assert cmp.mcnemar_exact(5, 5) == 1.0
    assert cmp.mcnemar_exact(0, 6) == pytest.approx(2 / 64)


def test_bootstrap_matches_item_resampling():
    b, c, n = 30, 10, 400  # diff = -0.05, se = sqrt((b + c)/n - d^2)/sqrt(n)
    lo, hi = cmp.bootstrap_diff_ci(b, c, n, reps=20000)
    assert lo < -0.05 < hi and hi < 0
    se = math.sqrt((b + c) / n - 0.05**2) / math.sqrt(n)
    assert lo == pytest.approx(-0.05 - 1.96 * se, abs=0.004)
    assert hi == pytest.approx(-0.05 + 1.96 * se, abs=0.004)
    # brute-force item bootstrap gives the same interval
    rng = random.Random(1)
    items = [-1] * b + [1] * c + [0] * (n - b - c)
    d = sorted(sum(rng.choices(items, k=n)) / n for _ in range(4000))
    assert lo == pytest.approx(d[100], abs=0.004) and hi == pytest.approx(
        d[3899], abs=0.004
    )
    assert cmp.bootstrap_diff_ci(b, c, n) == cmp.bootstrap_diff_ci(b, c, n)
    assert cmp.bootstrap_diff_ci(0, 0, 50) == (0.0, 0.0)


def test_sample_size_for_one_point():
    # Connor (1987): pd = 0.10, delta = 0.01 -> ((1.96*0.3162 + 0.8416*0.3161)/0.01)^2
    assert cmp.n_needed(0.10) == math.ceil(
        (1.959964 * math.sqrt(0.10) + 0.841621 * math.sqrt(0.10 - 1e-4)) ** 2 / 1e-4
    )
    assert 7800 < cmp.n_needed(0.10) < 7900
    assert cmp.n_needed(0.02) < cmp.n_needed(0.10)


def test_wilson():
    lo, hi = cmp.wilson(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(
        0.5962, abs=1e-3
    )


def _run(oks, agree=None, error=()):
    rows = {}
    for i, ok in enumerate(oks):
        r = {
            "id": f"x/{i}",
            "group": "even" if i % 2 == 0 else "odd",
            "ok": ok,
            "details": {"tool": agree[i] if agree else "t", "frac": float(ok)},
        }
        if i in error:
            r = {"id": f"x/{i}", "error": "boom", "ok": False}
        rows[f"x/{i}"] = r
    cfg = {"version": "1", "seed": 0, "agree": [["tool"]], "metrics": ["frac"]}
    return {"config": {"tasks": {"t": cfg}}, "results": {"t": rows}}


def test_paired_counts_and_report(tmp_path):
    ref = [True] * 8 + [False] * 2  # 80 %
    cand = [True] * 6 + [False] * 2 + [True, False]  # 70 %: 2 R->W, 1 W->R
    res = cmp.compare(_run(ref), _run(cand, agree=list("tttttttttu"), error=(9,)))
    t = res["tasks"]["t"]
    assert t["n"] == 9 and t["skipped"] == 1  # item 9 failed in the candidate
    assert (t["b"], t["c"]) == (2, 1)
    assert t["ref_acc"] == pytest.approx(8 / 9) and t["cand_acc"] == pytest.approx(
        7 / 9
    )
    assert t["flip_right_to_wrong"] == pytest.approx(2 / 9)
    assert t["recovery"] == pytest.approx(7 / 8)
    assert t["mcnemar_p"] == pytest.approx(1.0)
    assert t["agree"][0]["rate"] == pytest.approx(1.0)  # the 'u' item errored
    assert set(t["groups"]) == {"even", "odd"}
    assert t["metrics"]["cand"]["frac"] == pytest.approx(7 / 9)
    a, b = tmp_path / "ref.json", tmp_path / "cand.json"
    a.write_text(json.dumps(_run(ref)))
    b.write_text(json.dumps(_run(cand)))
    md = tmp_path / "out.md"
    assert (
        cmp.main(
            [
                str(a),
                str(b),
                "--md",
                str(md),
                "--json",
                str(tmp_path / "o.json"),
                "--groups",
            ]
        )
        == 0
    )
    text = md.read_text()
    assert "| t | 10 | 80.0 | 70.0 | -10.0" in text and "n for 1 pt" in text
    assert "agree on tool" in text and "| t | odd |" in text


def test_config_mismatch_warns():
    a, b = _run([True]), _run([True])
    b["config"]["tasks"]["t"]["version"] = "2"
    assert any("version" in w for w in cmp.compare(a, b)["warnings"])


def _complete(oks):
    report = _run(oks)
    report["status"] = "complete"
    report["config"]["requested_tasks"] = ["t"]
    report["config"]["tasks"]["t"]["n"] = len(oks)
    report["timing"] = {"_wall": 1.25}
    report["attempts"] = [{"command": ["bench/run.py", "--tasks", "t"]}]
    return report


def test_missing_and_failed_tasks_remain_visible_in_comparison(tmp_path):
    ref, cand = _complete([True, False]), _complete([False, False])
    ref["config"]["requested_tasks"] += ["missing_data", "omitted"]
    cand["config"]["requested_tasks"].append("missing_data")
    for name in ("missing_data", "omitted"):
        ref["config"]["tasks"][name] = {"version": "1", "n": 1}
        ref["results"][name] = {"a": {"ok": True}}
    cand["status"] = "incomplete"
    cand["completion"] = {
        "tasks": {
            "missing_data": {
                "status": "load_error",
                "expected": None,
                "error": "dataset absent",
            },
        }
    }
    result = cmp.compare(ref, cand)
    assert result["status"] == "incomplete"
    assert set(result["tasks"]) == {"t", "missing_data", "omitted"}
    assert result["tasks"]["t"]["cand_acc"] == 0
    missing = result["tasks"]["missing_data"]
    assert missing["measurement_status"] == "missing"
    assert missing["completion"]["candidate"]["error"] == "dataset absent"
    assert result["tasks"]["omitted"]["completion"]["candidate"]["status"] == "missing"
    a, b, out = (
        tmp_path / "ref.json",
        tmp_path / "cand.json",
        tmp_path / "comparison.json",
    )
    a.write_text(json.dumps(ref))
    b.write_text(json.dumps(cand))
    assert cmp.main([str(a), str(b), "--json", str(out)]) == 1
    text = out.read_text()
    saved = json.loads(text)
    assert "NaN" not in text
    assert saved["candidates"][str(b)]["tasks"]["missing_data"]["cand_acc"] is None
    assert saved["inputs"][str(b)]["sources"]["benchmark"]["data"] == cand


def test_score_errors_are_not_low_accuracy_pairs():
    ref, cand = _complete([True, True]), _complete([False, False])
    cand["results"]["t"]["x/1"]["details"]["score_error"] = "fixture scorer crashed"
    task = cmp.compare(ref, cand)["tasks"]["t"]
    assert task["n"] == 1 and task["cand_acc"] == 0
    assert task["skipped"] == 1
    assert task["status"] == "incomplete"
    assert task["completion"]["candidate"]["failed"] == 1


def test_missing_agreement_details_are_not_agreement():
    ref, cand = _complete([True]), _complete([True])
    ref["results"]["t"]["x/0"]["details"] = {}
    cand["results"]["t"]["x/0"]["details"] = {}
    result = cmp.compare(ref, cand)["tasks"]["t"]["agree"][0]
    assert result["n"] == 0 and math.isnan(result["rate"])


def test_unknown_legacy_completeness_is_not_claimed_complete():
    result = cmp.compare(_run([True]), _run([True]))
    assert result["status"] == "unverified"
    assert result["tasks"]["t"]["completion"]["reference"]["expected"] is None


def test_dataset_fingerprint_mismatch_is_explicit():
    ref, cand = _complete([True]), _complete([True])
    ref["config"]["tasks"]["t"]["items_sha256"] = "first-dataset"
    cand["config"]["tasks"]["t"]["items_sha256"] = "second-dataset"
    result = cmp.compare(ref, cand)
    assert result["status"] == "incomplete"
    assert any("items_sha256 differs" in warning for warning in result["warnings"])


def test_report_combines_real_input_values_without_inventing_measurements(tmp_path):
    a, b = tmp_path / "ref.json", tmp_path / "candidate.json"
    a.write_text(json.dumps(_complete([True, False])))
    b.write_text(json.dumps(_complete([False, False])))
    manifest = {
        "schema_version": 1,
        "run_id": "fixture-run",
        "state": "succeeded",
        "source": {"path": "/models/source", "fingerprint": "fixture-source"},
        "config": {"target_bpw": 3.0, "cfg": {"rounding": "gptq"}},
        "calibration": {
            "paths": [{"path": "calibration.jsonl", "sha256": "fixture-calibration"}]
        },
        "environment": {"python": "fixture-python", "code_sha256": "fixture-code"},
        "outputs": ["/models/quantized"],
        "result": {"allocation_bpw": 2.9},
    }
    size = {
        "path": "/models/quantized",
        "linears": {"bpw": 3.125},
        "tensor_bytes": 1000,
        "file_bytes": 1100,
    }
    quality = {
        "schema_version": 1,
        "status": "target_not_met",
        "metrics": {"kld_nats_per_token": 0.2, "ppl_quantized": 4.0},
        "criteria": {"max_kld": 0.1},
        "dataset": {"path": "heldout.jsonl", "sha256": "fixture-heldout"},
        "artifact": {"path": "/models/quantized", "run_id": "fixture-run"},
        "context": "dense_numerical",
    }
    for name, data in (("manifest", manifest), ("size", size), ("quality", quality)):
        (tmp_path / f"{name}.json").write_text(json.dumps(data))
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "ref.json": {"runtime_context": {"kind": "dense_numerical"}},
                "candidate.json": {
                    "manifest": "manifest.json",
                    "size": "size.json",
                    "quality": "quality.json",
                    "runtime_context": {"kind": "native", "backend": "fixture-only"},
                },
            }
        )
    )
    out, md = tmp_path / "comparison.json", tmp_path / "comparison.md"
    args = [
        str(a),
        str(b),
        "--metadata",
        str(metadata),
        "--json",
        str(out),
        "--md",
        str(md),
    ]
    assert cmp.main(args) == 0  # low quality is not incomplete benchmark execution
    saved = json.loads(out.read_text())
    assert saved["schema_version"] == 1 and saved["status"] == "complete"
    assert saved["command"] == ["bench/compare.py", *args]
    r, c = saved["inputs"][str(a)], saved["inputs"][str(b)]
    assert c["source"] == manifest["source"]
    assert c["effective_recipe"] == manifest["config"]
    assert c["quality_dataset"] == quality["dataset"]
    assert c["sources"]["manifest"]["data"] == manifest
    assert c["sources"]["quality"]["data"] == quality
    assert (
        c["sources"]["benchmark"]["sha256"]
        == hashlib.sha256(b.read_bytes()).hexdigest()
    )
    assert c["commands"] == _complete([False, False])["attempts"]
    measurements = c["measurements"]
    assert measurements["allocation_bpw"]["status"] == "predicted"
    assert measurements["allocation_bpw"]["value"] == 2.9
    assert measurements["stored_quantized_bpw"]["status"] == "measured"
    assert measurements["stored_quantized_bpw"]["value"] == 3.125
    assert measurements["whole_model_tensor_bytes"]["value"] == 1000
    assert measurements["whole_model_file_bytes"]["value"] == 1100
    assert measurements["benchmark_wall_seconds"]["value"] == 1.25
    assert measurements["quality"]["scope"] == "dense_numerical"
    assert measurements["quality"]["value"] == quality["metrics"]
    for name in ("peak_memory_bytes", "native_kernel_seconds"):
        assert (
            measurements[name]["status"] == "missing"
            and measurements[name]["value"] is None
        )
    assert r["measurements"]["allocation_bpw"]["status"] == "missing"
    assert r["measurements"]["quality"]["status"] == "missing"
    result = saved["candidates"][str(b)]
    assert result["runtime"]["performance_compared"] is False
    assert any("dense_numerical vs native" in warning for warning in result["warnings"])
    text = md.read_text()
    assert "fixture-source" in text and "| allocation_bpw | predicted | 2.9" in text
    assert "| peak_memory_bytes | missing | -" in text
