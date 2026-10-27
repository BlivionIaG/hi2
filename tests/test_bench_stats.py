# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""bench paired statistics against known numbers, and compare.py end to end."""

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
