# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Paired comparison of ``bench/run.py`` results against a reference run.

Per task, over the items both runs completed: the accuracy of each run, the difference
(candidate - reference) with a paired bootstrap 95% CI, the exact McNemar p, the flip
rates (reference right -> candidate wrong, and the reverse, as shares of the items), and
the recovery (candidate / reference accuracy). Tasks with ``AGREE`` keys also get the
agreement with the reference on those details (Wilson 95% CI). A power note gives the
paired sample size that detects a 1-point difference at the observed discordance.

    python bench/compare.py results/bf16.json results/hi2.json [...] --md out.md --json out.json
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path
from statistics import NormalDist

Z = NormalDist().inv_cdf


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p over the discordant pairs (binomial, p = 1/2)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def _binom(rng: random.Random, n: int, p: float) -> int:
    if p <= 0 or n == 0:
        return 0
    if p >= 1:
        return n
    if hasattr(rng, "binomialvariate"):  # Python 3.12+
        return rng.binomialvariate(n, p)
    return sum(rng.random() < p for _ in range(n))


def bootstrap_diff_ci(
    b: int, c: int, n: int, reps: int = 10000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """Percentile CI of the paired accuracy difference (c - b) / n.

    Resampling n items with replacement only changes how many fall in the two discordant
    cells, so each replicate draws those counts from Multinomial(n; b/n, c/n, rest):
    the same distribution as the item-level bootstrap, at no per-item cost."""
    if n == 0:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    diffs = []
    for _ in range(reps):
        bb = _binom(rng, n, b / n)
        cc = _binom(rng, n - bb, c / (n - b)) if n > b else 0
        diffs.append((cc - bb) / n)
    diffs.sort()
    lo = diffs[int((1 - level) / 2 * reps)]
    hi = diffs[min(reps - 1, int((1 + level) / 2 * reps))]
    return lo, hi


def n_needed(
    discordance: float, delta: float = 0.01, alpha: float = 0.05, power: float = 0.8
) -> int:
    """Paired items for McNemar to detect an accuracy difference ``delta`` (Connor 1987)."""
    pd = max(discordance, delta + 1e-9)
    za, zb = Z(1 - alpha / 2), Z(power)
    return math.ceil(
        (za * math.sqrt(pd) + zb * math.sqrt(pd - delta**2)) ** 2 / delta**2
    )


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return c - h, c + h


def paired(ref: dict, cand: dict) -> dict:
    """Paired statistics over the items both runs completed without a request error."""
    ids = sorted(
        i
        for i in ref.keys() & cand.keys()
        if "error" not in ref[i] and "error" not in cand[i]
    )
    n = len(ids)
    r = [bool(ref[i]["ok"]) for i in ids]
    c = [bool(cand[i]["ok"]) for i in ids]
    b = sum(x and not y for x, y in zip(r, c))  # reference right, candidate wrong
    w = sum(y and not x for x, y in zip(r, c))  # reference wrong, candidate right
    acc_r, acc_c = (sum(r) / n, sum(c) / n) if n else (math.nan, math.nan)
    lo, hi = bootstrap_diff_ci(b, w, n)
    return {
        "n": n,
        "ref_acc": acc_r,
        "cand_acc": acc_c,
        "diff": acc_c - acc_r,
        "ci95": [lo, hi],
        "mcnemar_p": mcnemar_exact(b, w),
        "b": b,
        "c": w,
        "flip_right_to_wrong": b / n if n else math.nan,
        "flip_wrong_to_right": w / n if n else math.nan,
        "recovery": acc_c / acc_r if n and acc_r else math.nan,
        "n_for_1pt": n_needed((b + w) / n) if b + w else None,
        "skipped": len(ref.keys() | cand.keys()) - n,
    }


def agreement(ref: dict, cand: dict, keys: list[str]) -> dict:
    ids = [
        i
        for i in ref.keys() & cand.keys()
        if "error" not in ref[i] and "error" not in cand[i]
    ]
    k = sum(
        all(ref[i]["details"].get(x) == cand[i]["details"].get(x) for x in keys)
        for i in ids
    )
    lo, hi = wilson(k, len(ids))
    return {
        "keys": keys,
        "n": len(ids),
        "rate": k / len(ids) if ids else math.nan,
        "ci95": [lo, hi],
    }


def groups(ref: dict, cand: dict) -> dict:
    out = {}
    for g in sorted({r.get("group") for r in ref.values() if r.get("group")}):
        ids = {i for i, r in ref.items() if r.get("group") == g}
        out[g] = paired(
            {i: ref[i] for i in ids}, {i: cand[i] for i in ids & cand.keys()}
        )
    return out


def metric_means(rows: dict, keys) -> dict:
    out = {}
    for k in keys:
        v = [
            float(r["details"][k])
            for r in rows.values()
            if "error" not in r and k in r.get("details", {})
        ]
        out[k] = sum(v) / len(v) if v else math.nan
    return out


def compare(ref_run: dict, cand_run: dict) -> dict:
    out = {"tasks": {}, "warnings": []}
    rc, cc = ref_run["config"]["tasks"], cand_run["config"]["tasks"]
    tot = {"b": 0, "c": 0, "n": 0, "ref": 0, "cand": 0}
    for name in [t for t in rc if t in cc]:
        for k in ("version", "seed", "fingerprint"):
            if rc[name].get(k) != cc[name].get(k):
                out["warnings"].append(
                    f"{name}: {k} differs ({rc[name].get(k)} vs {cc[name].get(k)})"
                )
        ref, cand = ref_run["results"][name], cand_run["results"][name]
        res = paired(ref, cand)
        res["groups"] = groups(ref, cand)
        res["agree"] = [
            agreement(ref, cand, list(k)) for k in rc[name].get("agree", [])
        ]
        mk = rc[name].get("metrics", [])
        res["metrics"] = {"ref": metric_means(ref, mk), "cand": metric_means(cand, mk)}
        out["tasks"][name] = res
        tot["b"] += res["b"]
        tot["c"] += res["c"]
        tot["n"] += res["n"]
        tot["ref"] += round(res["ref_acc"] * res["n"]) if res["n"] else 0
        tot["cand"] += round(res["cand_acc"] * res["n"]) if res["n"] else 0
    t = out["tasks"].values()
    n = tot["n"]
    out["overall"] = {
        "tasks": len(out["tasks"]),
        "macro_ref_acc": _mean([x["ref_acc"] for x in t]),
        "macro_cand_acc": _mean([x["cand_acc"] for x in t]),
        "macro_recovery": _mean([x["recovery"] for x in t]),
        "pooled_n": n,
        "pooled_diff": (tot["cand"] - tot["ref"]) / n if n else math.nan,
        "pooled_ci95": list(bootstrap_diff_ci(tot["b"], tot["c"], n)),
        "pooled_mcnemar_p": mcnemar_exact(tot["b"], tot["c"]),
    }
    return out


def _mean(v):
    v = [x for x in v if not math.isnan(x)]
    return sum(v) / len(v) if v else math.nan


def _pct(x, signed=False):
    return "-" if math.isnan(x) else (f"{100 * x:+.1f}" if signed else f"{100 * x:.1f}")


def _num(x):
    return "-" if math.isnan(x) else f"{x:.3f}"


def markdown(ref_path: str, cand_path: str, res: dict, show_groups: bool) -> str:
    lines = [f"## {Path(cand_path).stem} vs {Path(ref_path).stem}", ""]
    lines += [f"> warning: {w}" for w in res["warnings"]]
    lines += [
        "| task | n | ref % | cand % | diff [95% CI] | McNemar p | R->W % | W->R % "
        "| recovery | n for 1 pt |",
        "|" + "---|" * 10,
    ]
    for name, r in res["tasks"].items():
        lo, hi = r["ci95"]
        lines.append(
            f"| {name} | {r['n']} | {_pct(r['ref_acc'])} | {_pct(r['cand_acc'])} | "
            f"{_pct(r['diff'], True)} [{_pct(lo, True)}, {_pct(hi, True)}] | "
            f"{r['mcnemar_p']:.3g} | {_pct(r['flip_right_to_wrong'])} | "
            f"{_pct(r['flip_wrong_to_right'])} | {_num(r['recovery'])} | "
            f"{r['n_for_1pt'] or '-'} |"
        )
    o = res["overall"]
    lo, hi = o["pooled_ci95"]
    lines.append(
        f"| **all (pooled)** | {o['pooled_n']} | {_pct(o['macro_ref_acc'])} | "
        f"{_pct(o['macro_cand_acc'])} | {_pct(o['pooled_diff'], True)} "
        f"[{_pct(lo, True)}, {_pct(hi, True)}] | {o['pooled_mcnemar_p']:.3g} | | | "
        f"{_num(o['macro_recovery'])} | |"
    )
    lines += [
        "",
        "ref/cand % in the last row are macro means over tasks; diff and p are "
        "pooled over all items. R->W: reference right, candidate wrong.",
        "n for 1 pt: paired items for 80% power to detect a 1-point difference "
        "(alpha 0.05, McNemar) at this pair's discordance rate (- when no item "
        "flipped); about 78500 x the discordance.",
    ]
    extra = [
        (n, r) for n, r in res["tasks"].items() if r["agree"] or r["metrics"]["ref"]
    ]
    if extra:
        lines += ["", "| task | measure | ref | cand |", "|---|---|---|---|"]
        for name, r in extra:
            for a in r["agree"]:
                lo, hi = a["ci95"]
                lines.append(
                    f"| {name} | agree on {'+'.join(a['keys'])} | 100 | "
                    f"{_pct(a['rate'])} [{_pct(lo)}, {_pct(hi)}] |"
                )
            for k, v in r["metrics"]["ref"].items():
                lines.append(
                    f"| {name} | mean {k} | {v:.4f} | {r['metrics']['cand'][k]:.4f} |"
                )
    if show_groups:
        lines += [
            "",
            "| task | group | n | ref % | cand % | diff |",
            "|---|---|---|---|---|---|",
        ]
        for name, r in res["tasks"].items():
            for g, x in r["groups"].items():
                lines.append(
                    f"| {name} | {g} | {x['n']} | {_pct(x['ref_acc'])} | "
                    f"{_pct(x['cand_acc'])} | {_pct(x['diff'], True)} |"
                )
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="reference first, then the candidates")
    ap.add_argument("--md", type=Path, help="also write the markdown here")
    ap.add_argument("--json", type=Path, help="write the full statistics here")
    ap.add_argument("--groups", action="store_true", help="per-group tables")
    a = ap.parse_args(argv)
    if len(a.runs) < 2:
        ap.error("need a reference and at least one candidate")
    ref = json.loads(Path(a.runs[0]).read_text())
    md, js = [], {"reference": a.runs[0], "candidates": {}}
    for path in a.runs[1:]:
        res = compare(ref, json.loads(Path(path).read_text()))
        js["candidates"][path] = res
        md.append(markdown(a.runs[0], path, res, a.groups))
    text = "\n".join(md)
    print(text)
    if a.md:
        a.md.write_text(text)
    if a.json:
        a.json.write_text(json.dumps(js, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
