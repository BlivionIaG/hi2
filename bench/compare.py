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
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from statistics import NormalDist

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.run import completed  # noqa: E402

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
    """Paired statistics over responses with a completed score in both runs."""
    ids = sorted(
        i for i in ref.keys() & cand.keys() if completed(ref[i]) and completed(cand[i])
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
        "mcnemar_p": mcnemar_exact(b, w) if n else math.nan,
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
        if completed(ref[i])
        and completed(cand[i])
        and all(
            x in ref[i].get("details", {}) and x in cand[i].get("details", {})
            for x in keys
        )
    ]
    k = sum(
        all(ref[i]["details"][x] == cand[i]["details"][x] for x in keys) for i in ids
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
            if completed(r) and k in r.get("details", {})
        ]
        out[k] = sum(v) / len(v) if v else math.nan
    return out


def task_names(run: dict) -> list[str]:
    requested = run.get("config", {}).get("requested_tasks")
    if requested is not None:
        return list(requested)
    return list(
        dict.fromkeys(
            [
                *run.get("config", {}).get("tasks", {}),
                *run.get("results", {}),
                *run.get("completion", {}).get("tasks", {}),
            ]
        )
    )


def task_completion(run: dict, name: str) -> dict:
    if name not in task_names(run):
        return {"status": "missing", "expected": None, "completed": 0, "missing": None}
    cfg = run.get("config", {}).get("tasks", {}).get(name, {})
    declared = run.get("completion", {}).get("tasks", {}).get(name, {})
    rows = (
        run.get("results", {}).get(name, {})
        if declared.get("status") != "load_error"
        else {}
    )
    expected = declared.get("expected", cfg.get("n"))
    done = sum(completed(row) for row in rows.values())
    failed = len(rows) - done
    status = declared.get("status")
    if status is None:
        status = (
            "incomplete"
            if failed or not rows or (expected is not None and done != expected)
            else "complete"
            if expected is not None
            else "unverified"
        )
    elif status == "complete" and (failed or not expected or done != expected):
        status = "incomplete"
    return {
        "status": status,
        "expected": expected,
        "completed": done,
        "failed": failed,
        "missing": max(0, expected - len(rows)) if expected is not None else None,
        "error": declared.get("error"),
        "dataset": {
            key: cfg.get(key)
            for key in (
                "datasets",
                "items_sha256",
                "fingerprint",
                "version",
                "seed",
                "limit",
            )
        },
    }


def read_source(path: Path) -> dict:
    raw = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "data": json.loads(raw),
    }


def measurement(
    value, unit: str, source: str | None, *, status="measured", scope=None
) -> dict:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        value, status = None, "missing"
    return {
        "status": status,
        "value": value,
        "unit": unit,
        "source": source,
        "scope": scope,
    }


def evidence(run: dict, sources: dict, runtime_context: dict | None = None) -> dict:
    """Adapt the real size/eval/manifest outputs; absent inputs remain absent."""
    manifest = sources.get("manifest", {}).get("data", {})
    size = sources.get("size", {}).get("data", {})
    quality = sources.get("quality", {}).get("data", {})
    result = manifest.get("result") or {}
    metrics = quality.get("metrics", quality)
    quality_measured = bool(metrics) and quality.get("status") != "not_measured"
    context = runtime_context or {"kind": "unknown"}
    if context.get("kind") not in ("native", "dense_numerical", "unknown"):
        raise ValueError(
            "runtime_context.kind must be native, dense_numerical or unknown"
        )
    measurements = {
        "allocation_bpw": measurement(
            result.get("allocation_bpw"),
            "bits/weight",
            "manifest#/result/allocation_bpw",
            status="predicted",
            scope="allocator estimate, not stored bytes",
        ),
        "stored_quantized_bpw": measurement(
            size.get("linears", {}).get("bpw"),
            "bits/weight",
            "size#/linears/bpw",
            scope="quantized linears including their storage metadata",
        ),
        "whole_model_tensor_bytes": measurement(
            size.get("tensor_bytes"),
            "bytes",
            "size#/tensor_bytes",
            scope="all tensors",
        ),
        "whole_model_file_bytes": measurement(
            size.get("file_bytes"),
            "bytes",
            "size#/file_bytes",
            scope="weight shard files, not ancillary model files",
        ),
        "quality": measurement(
            metrics if quality_measured else None,
            "per-metric",
            "quality",
            scope=quality.get("context", "dense_numerical"),
        ),
        "benchmark_wall_seconds": measurement(
            run.get("timing", {}).get("_wall"),
            "seconds",
            "benchmark#/timing/_wall",
            scope="client harness including requests/scoring; cumulative over resumes",
        ),
        "peak_memory_bytes": measurement(None, "bytes", None, scope="not collected"),
        "native_kernel_seconds": measurement(
            None, "seconds", None, scope="not collected"
        ),
    }
    return {
        "sources": sources,
        "source": manifest.get("source"),
        "artifact": quality.get("artifact")
        or (
            {"run_id": manifest.get("run_id"), "outputs": manifest.get("outputs")}
            if manifest
            else None
        ),
        "calibration": manifest.get("calibration"),
        "quality_dataset": quality.get("dataset"),
        "effective_recipe": manifest.get("config"),
        "environment": manifest.get("environment"),
        "commands": run.get("attempts", []),
        "config": run.get("config", {}),
        "runtime_context": context,
        "quality_status": quality.get("status"),
        "measurements": measurements,
    }


def json_values(value):
    """Standards-compliant JSON: unavailable statistics are null, never NaN."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_values(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_values(v) for v in value]
    return value


def compare(ref_run: dict, cand_run: dict) -> dict:
    out = {"schema_version": 1, "tasks": {}, "warnings": []}
    rc = ref_run.get("config", {}).get("tasks", {})
    cc = cand_run.get("config", {}).get("tasks", {})
    tot = {"b": 0, "c": 0, "n": 0, "ref": 0, "cand": 0}
    for name in dict.fromkeys([*task_names(ref_run), *task_names(cand_run)]):
        mismatches = []
        for k in ("version", "seed", "fingerprint", "items_sha256"):
            if rc.get(name, {}).get(k) != cc.get(name, {}).get(k):
                mismatches.append(k)
                out["warnings"].append(
                    f"{name}: {k} differs "
                    f"({rc.get(name, {}).get(k)} vs {cc.get(name, {}).get(k)})"
                )
        ref_status, cand_status = (
            task_completion(ref_run, name),
            task_completion(cand_run, name),
        )
        ref = (
            ref_run.get("results", {}).get(name, {})
            if name in task_names(ref_run)
            else {}
        )
        cand = (
            cand_run.get("results", {}).get(name, {})
            if name in task_names(cand_run)
            else {}
        )
        if ref_status["status"] == "load_error":
            ref = {}
        if cand_status["status"] == "load_error":
            cand = {}
        res = paired(ref, cand)
        res["completion"] = {"reference": ref_status, "candidate": cand_status}
        res["status"] = (
            "incomplete"
            if mismatches
            or any(
                s["status"] not in ("complete", "unverified")
                for s in (ref_status, cand_status)
            )
            else "unverified"
            if any(s["status"] == "unverified" for s in (ref_status, cand_status))
            else "complete"
        )
        res["measurement_status"] = "measured" if res["n"] else "missing"
        if res["status"] != "complete":
            out["warnings"].append(
                f"{name}: comparison {res['status']}; "
                f"reference={ref_status['status']}, candidate={cand_status['status']}"
            )
        res["groups"] = groups(ref, cand)
        res["agree"] = [
            agreement(ref, cand, list(k)) for k in rc.get(name, {}).get("agree", [])
        ]
        mk = rc.get(name, {}).get("metrics", [])
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
        "pooled_mcnemar_p": mcnemar_exact(tot["b"], tot["c"]) if n else math.nan,
    }
    states = [x["status"] for x in out["tasks"].values()]
    run_incomplete = any(
        run.get("status", "complete") != "complete"
        or run.get("completion", {}).get("errors")
        for run in (ref_run, cand_run)
    )
    out["status"] = (
        "incomplete"
        if not states or "incomplete" in states or run_incomplete
        else "unverified"
        if "unverified" in states
        else "complete"
    )
    out["runtime"] = {
        "performance_compared": False,
        "reason": "paired task quality and client timing are not native-kernel performance",
    }
    return out


def _mean(v):
    v = [x for x in v if not math.isnan(x)]
    return sum(v) / len(v) if v else math.nan


def _pct(x, signed=False):
    return "-" if math.isnan(x) else (f"{100 * x:+.1f}" if signed else f"{100 * x:.1f}")


def _num(x):
    return "-" if math.isnan(x) else f"{x:.3f}"


def markdown(
    ref_path: str,
    cand_path: str,
    res: dict,
    show_groups: bool,
    reference: dict | None = None,
    candidate: dict | None = None,
) -> str:
    lines = [
        f"## {Path(cand_path).stem} vs {Path(ref_path).stem}",
        "",
        f"Comparison status: **{res['status']}**. Scores below use completed pairs only.",
        "This is a quality comparison, not a native-kernel performance comparison.",
        "",
    ]
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
    lines += [
        "",
        "| task | comparison | reference completed / expected | reference status | "
        "candidate completed / expected | candidate status | unpaired observed items |",
        "|" + "---|" * 7,
    ]
    for name, r in res["tasks"].items():
        rs, cs = r["completion"]["reference"], r["completion"]["candidate"]
        re = rs["expected"] if rs["expected"] is not None else "unknown"
        ce = cs["expected"] if cs["expected"] is not None else "unknown"
        lines.append(
            f"| {name} | {r['status']} | {rs['completed']} / {re} | {rs['status']} | "
            f"{cs['completed']} / {ce} | {cs['status']} | {r['skipped']} |"
        )
    if reference is not None and candidate is not None:
        for label, data in (("reference", reference), ("candidate", candidate)):
            src = data["sources"]["benchmark"]
            lines += [
                "",
                f"### {label} provenance",
                "",
                f"- Benchmark: `{src['path']}`; SHA256 `{src['sha256']}`.",
                f"- Source: `{json.dumps(data['source'], sort_keys=True)}`.",
                f"- Artifact: `{json.dumps(data['artifact'], sort_keys=True)}`.",
                f"- Runtime context (operator-declared): `{json.dumps(data['runtime_context'], sort_keys=True)}`.",
                f"- Quality status: `{data['quality_status'] or 'missing'}`.",
                "- Full raw reports, recipe, datasets and command history are retained in JSON.",
                "",
                "| measurement | status | value | unit | scope |",
                "|---|---|---|---|---|",
            ]
            for key, m in data["measurements"].items():
                value = (
                    "-"
                    if m["value"] is None
                    else json.dumps(m["value"], sort_keys=True)
                )
                lines.append(
                    f"| {key} | {m['status']} | {value} | {m['unit']} | {m['scope']} |"
                )
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
    ap.add_argument(
        "--metadata",
        type=Path,
        help="JSON mapping run paths to manifest/size/quality paths and runtime_context",
    )
    a = ap.parse_args(argv)
    if len(a.runs) < 2:
        ap.error("need a reference and at least one candidate")
    metadata_source = read_source(a.metadata) if a.metadata else None
    root = a.metadata.resolve().parent if a.metadata else Path.cwd()
    metadata = {
        str((root / key).resolve()): value
        for key, value in (metadata_source["data"] if metadata_source else {}).items()
    }
    inputs = {}
    for path in a.runs:
        src = read_source(Path(path))
        entry = metadata.get(src["path"], {})
        sources = {"benchmark": src}
        for kind in ("manifest", "size", "quality"):
            if entry.get(kind):
                sources[kind] = read_source(root / entry[kind])
        inputs[path] = evidence(src["data"], sources, entry.get("runtime_context"))
    ref = inputs[a.runs[0]]["sources"]["benchmark"]["data"]
    md = []
    js = {
        "schema_version": 1,
        "reference": a.runs[0],
        "candidates": {},
        "inputs": inputs,
        "command": ["bench/compare.py", *(sys.argv[1:] if argv is None else argv)],
        "metadata": metadata_source,
    }
    for path in a.runs[1:]:
        cand = inputs[path]["sources"]["benchmark"]["data"]
        res = compare(ref, cand)
        res["source_runs"] = {"reference": a.runs[0], "candidate": path}
        rkind = inputs[a.runs[0]]["runtime_context"]["kind"]
        ckind = inputs[path]["runtime_context"]["kind"]
        if rkind != ckind or "unknown" in (rkind, ckind):
            res["warnings"].append(
                f"runtime contexts are {rkind} vs {ckind}; do not infer native performance"
            )
        js["candidates"][path] = res
        md.append(
            markdown(a.runs[0], path, res, a.groups, inputs[a.runs[0]], inputs[path])
        )
    states = [r["status"] for r in js["candidates"].values()]
    js["status"] = (
        "incomplete"
        if "incomplete" in states
        else "unverified"
        if "unverified" in states
        else "complete"
    )
    text = "\n".join(md)
    print(text)
    if a.md:
        a.md.write_text(text)
    if a.json:
        a.json.write_text(json.dumps(json_values(js), indent=1, allow_nan=False))
    return 1 if js["status"] == "incomplete" else 0


if __name__ == "__main__":
    sys.exit(main())
