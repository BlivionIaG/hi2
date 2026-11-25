# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""CPU tests for the v2 paired comparison and gate."""

import json
from pathlib import Path

import numpy as np

from experiments.inkling_v2_compare import compare, update_run_json


def _doc(rows):
    sets = {}
    for key, (cat, lang, old_kl) in rows.items():
        sets[key] = {
            "category": cat,
            "language": lang,
            "length": 2048,
            "windows": len(old_kl),
            "kl_per_window": old_kl,
            "top1_agreement": 0.9,
            "ppl_ratio": 1.01,
        }
    return {"eval_set": {"sets": sets}}


def test_compare_pairs_windows_and_adopts_a_pooled_gain():
    old = _doc({"a": ("code", "py", [1.0, 2.0, 3.0])})
    new = _doc({"a": ("code", "py", [0.8, 1.8, 2.8])})
    r = compare(old, new)
    assert r["pooled"]["windows"] == 3
    assert abs(r["pooled"]["delta"] + 0.2) < 1e-12
    assert r["adopt"] is True


def test_compare_keeps_old_when_a_category_worsens():
    old = _doc(
        {
            "a": ("code", "py", [1.0, 2.0, 3.0]),
            "b": ("math", "en", [1.0, 2.0, 3.0]),
        }
    )
    new = _doc(
        {
            "a": ("code", "py", [0.7, 1.7, 2.7]),
            "b": ("math", "en", [1.1, 2.1, 3.1]),
        }
    )
    r = compare(old, new)
    assert r["pooled_improves_beyond_sem"] is True
    assert r["category_gate"]["math"]["worsens"] is True
    assert r["adopt"] is False


def test_compare_requires_a_pooled_gain_beyond_sem():
    old = _doc({"a": ("code", "py", [1.0, 2.0, 3.0])})
    new = _doc({"a": ("code", "py", [1.1, 1.9, 3.1])})
    r = compare(old, new)
    assert r["pooled"]["delta"] >= 0
    assert r["adopt"] is False


def test_ood_single_file_shape_is_reported_and_not_in_the_gate():
    old = _doc({"a": ("code", "py", [1.0, 2.0])})
    new = _doc({"a": ("code", "py", [0.9, 1.9])})
    ood_old = {"student": {"windows": 2, "kl_per_window": [1.0, 2.0]}}
    ood_new = {"student": {"windows": 2, "kl_per_window": [1.0, 2.0]}}
    r = compare(old, new, ood_old, ood_new)
    assert r["adopt"] is True
    assert r["ood"]["ood"]["delta"] == 0.0
    assert "ood" not in r["category_gate"]


def test_compare_limit_keeps_the_first_windows_in_sorted_set_order():
    old = _doc(
        {
            "b": ("math", "en", [1.0, 2.0]),
            "a": ("code", "py", [1.0, 2.0]),
        }
    )
    new = _doc(
        {
            "b": ("math", "en", [0.5, 1.5]),
            "a": ("code", "py", [0.5, 1.5]),
        }
    )
    r = compare(old, new, windows=3)
    assert r["pooled"]["windows"] == 3
    assert set(r["sets"]) == {"a", "b"}
    assert r["sets"]["a"]["windows"] == 2
    assert r["sets"]["b"]["windows"] == 1


def test_update_run_json_adds_the_calibration_meta(tmp_path: Path):
    p = tmp_path / "hi2_run.json"
    p.write_text(json.dumps({"config": {"calib": ["x"]}}))
    update_run_json(p, {"manifest_sha256": "abc", "excluded_license_classes": ["gpl2"]})
    d = json.loads(p.read_text())
    assert d["config"]["calib"] == ["x"]
    assert d["calib_v2"]["manifest_sha256"] == "abc"
