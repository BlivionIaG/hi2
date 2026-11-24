# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Pair two runs, per category and per task, and report the deterministic floor.

The reference is the stock endpoint and the candidate is the quantized one (or a
rerun of stock).  Only items completed and scored in both runs enter the paired
statistics; incomplete items are counted and make the report incomplete rather
than being scored as wrong.  ``--floor`` compares the reference with a second
stock run: its deltas and flips are the noise floor a quantized delta must
exceed to be interesting.
"""

import json
from pathlib import Path

from tools.inkling_eval import stats
from tools.inkling_eval.runner import load_records, read_meta

CHECK_KEYS = (
    "plan_sha256",
    "template",
    "seed",
    "reasoning_effort",
    "contexts",
    "haystack_sha256",
    "decoding",
)


def _items(records: dict[str, dict]) -> dict[str, dict]:
    return {item_id: record for item_id, record in records.items() if record.get("category")}


def _paired(ref: dict, cand: dict, key: str) -> dict[str, dict]:
    values: dict[str, dict] = {}
    for item_id in sorted(set(ref) & set(cand)):
        left, right = ref[item_id], cand[item_id]
        if left.get("status") != "complete" or right.get("status") != "complete":
            continue
        left_score, right_score = left.get("score"), right.get("score")
        if not isinstance(left_score, dict) or not isinstance(right_score, dict):
            continue
        if left_score.get("ok") is None or right_score.get("ok") is None:
            continue
        name = left.get(key) or right.get(key) or "unknown"
        values.setdefault(name, {"pairs": [], "ref_only": 0, "cand_only": 0})
        values[name]["pairs"].append((bool(left_score["ok"]), bool(right_score["ok"])))
    for item_id in set(ref) - set(cand):
        name = ref[item_id].get(key) or "unknown"
        values.setdefault(name, {"pairs": [], "ref_only": 0, "cand_only": 0})["ref_only"] += 1
    for item_id in set(cand) - set(ref):
        name = cand[item_id].get(key) or "unknown"
        values.setdefault(name, {"pairs": [], "ref_only": 0, "cand_only": 0})["cand_only"] += 1
    return values


def _summarize(ref: dict, cand: dict, key: str, *, reps: int) -> dict:
    grouped = _paired(ref, cand, key)
    result = {}
    for name in sorted(grouped):
        entry = grouped[name]
        summary = stats.paired_stats(entry["pairs"], reps=reps)
        summary["ref_only"] = entry["ref_only"]
        summary["cand_only"] = entry["cand_only"]
        result[name] = summary
    return result


def _planned(meta: dict | None) -> int | None:
    if not meta or not isinstance(meta.get("plan_counts"), dict):
        return None
    return sum(
        int(entry.get("items") or 0)
        for entry in meta["plan_counts"].values()
        if isinstance(entry, dict)
    )


def load_run(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"run file not found: {path}")
    return {
        "path": str(path),
        "meta": read_meta(path),
        "records": _items(load_records(path)),
    }


def compare(
    ref_path: Path,
    cand_path: Path,
    *,
    floor_path: Path | None = None,
    reps: int = stats.DEFAULT_REPS,
) -> dict:
    ref_run, cand_run = load_run(ref_path), load_run(cand_path)
    ref_meta, cand_meta = ref_run["meta"] or {}, cand_run["meta"] or {}
    ref, cand = ref_run["records"], cand_run["records"]
    checks = {}
    mismatched = []
    for key in CHECK_KEYS:
        left, right = ref_meta.get(key), cand_meta.get(key)
        same = left == right
        checks[key] = {"same": same, "ref": left, "cand": right}
        if not same:
            mismatched.append(key)
    ref_planned, cand_planned = _planned(ref_meta), _planned(cand_meta)
    ref_complete = sum(1 for record in ref.values() if record.get("status") == "complete")
    cand_complete = sum(1 for record in cand.values() if record.get("status") == "complete")
    paired_ids = {
        item_id
        for item_id in set(ref) & set(cand)
        if ref[item_id].get("status") == "complete"
        and cand[item_id].get("status") == "complete"
        and isinstance(ref[item_id].get("score"), dict)
        and isinstance(cand[item_id].get("score"), dict)
    }
    complete = (
        not mismatched
        and ref_planned is not None
        and cand_planned is not None
        and ref_complete == ref_planned
        and cand_complete == cand_planned
    )
    status = "mismatched" if mismatched else ("complete" if complete else "incomplete")
    categories = _summarize(ref, cand, "category", reps=reps)
    tasks = _summarize(ref, cand, "subcategory", reps=reps)
    floor = None
    if floor_path is not None:
        floor_run = load_run(floor_path)
        floor_records = floor_run["records"]
        floor_categories = _summarize(ref, floor_records, "category", reps=reps)
        deltas = [
            entry["delta"]
            for entry in floor_categories.values()
            if entry["delta"] is not None
        ]
        floor = {
            "path": str(floor_path),
            "label": (floor_run["meta"] or {}).get("label"),
            "categories": floor_categories,
            "max_abs_delta": max((abs(delta) for delta in deltas), default=None),
            "total_flips": sum(
                entry["flips_ref_right"] + entry["flips_cand_right"]
                for entry in floor_categories.values()
            ),
            "status": "complete" if floor_records else "empty",
        }
    return {
        "schema": 1,
        "status": status,
        "comparable": not mismatched,
        "ref": {
            "path": ref_run["path"],
            "label": ref_meta.get("label"),
            "url": ref_meta.get("url"),
            "planned": ref_planned,
            "complete": ref_complete,
            "records": len(ref),
        },
        "cand": {
            "path": cand_run["path"],
            "label": cand_meta.get("label"),
            "url": cand_meta.get("url"),
            "planned": cand_planned,
            "complete": cand_complete,
            "records": len(cand),
        },
        "pairing": {
            "paired": len(paired_ids),
            "ref_only": len(set(ref) - set(cand)),
            "cand_only": len(set(cand) - set(ref)),
        },
        "checks": checks,
        "categories": categories,
        "tasks": tasks,
        "floor": floor,
    }


def _pct(value) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}"


def _delta(value) -> str:
    return "n/a" if value is None else f"{100 * value:+.1f}"


def markdown(report: dict) -> str:
    """A short table: categories first, then tasks, then the determinism floor."""
    ref_label = report["ref"]["label"] or "ref"
    cand_label = report["cand"]["label"] or "cand"
    lines = [
        f"# Inkling task evaluation: {cand_label} vs {ref_label}",
        "",
        f"status: **{report['status']}**; paired {report['pairing']['paired']} items; "
        f"{ref_label} {report['ref']['complete']}/{report['ref']['planned']} complete, "
        f"{cand_label} {report['cand']['complete']}/{report['cand']['planned']} complete",
        "",
        "| category | n | ref % | cand % | delta pp | 95% CI (pp) | McNemar p | MDD pp |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for name, entry in report["categories"].items():
        ci = (
            "n/a"
            if entry["ci_low"] is None
            else f"{100 * entry['ci_low']:+.1f} to {100 * entry['ci_high']:+.1f}"
        )
        mdd = "n/a" if entry["mdd"] is None else f"{100 * entry['mdd']:.1f}"
        lines.append(
            f"| {name} | {entry['n']} | {_pct(entry['ref_acc'])} | {_pct(entry['cand_acc'])} | "
            f"{_delta(entry['delta'])} | {ci} | {entry['mcnemar_p']} | {mdd} |"
        )
    lines += ["", "## Tasks", "", "| task | n | ref % | cand % | delta pp | flips (r/c) |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, entry in report["tasks"].items():
        lines.append(
            f"| {name} | {entry['n']} | {_pct(entry['ref_acc'])} | {_pct(entry['cand_acc'])} | "
            f"{_delta(entry['delta'])} | {entry['flips_ref_right']}/{entry['flips_cand_right']} |"
        )
    if report["floor"]:
        floor = report["floor"]
        lines += [
            "",
            f"determinism floor ({floor['label'] or 'second stock run'}): "
            f"max |delta| {_pct(floor['max_abs_delta'])} pp, {floor['total_flips']} flips",
        ]
    else:
        lines += ["", "determinism floor: not requested (pass --floor with a second stock run)"]
    if report["status"] != "complete":
        lines += ["", f"incomplete or mismatched: {json.dumps(report['checks'], sort_keys=True)[:600]}"]
    return "\n".join(lines) + "\n"


def write_report(report: dict, json_path: Path, md_path: Path | None = None) -> None:
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    if md_path is not None:
        Path(md_path).write_text(markdown(report))
