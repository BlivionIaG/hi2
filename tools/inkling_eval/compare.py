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

from tools.inkling_eval import runner, stats
from tools.inkling_eval.runner import load_records, read_meta, truncation_summary

CHECK_KEYS = (
    "plan_sha256",
    "set",
    "template",
    "seed",
    "reasoning_effort",
    "contexts",
    "haystack_sha256",
    "decoding",
)


def _truncated(record: dict) -> bool:
    return record.get("status") == "truncated" or record.get("finish_reason") == "length"


def _items(records: dict[str, dict]) -> dict[str, dict]:
    return {item_id: record for item_id, record in records.items() if record.get("category")}


def _paired(ref: dict, cand: dict, key: str) -> dict[str, dict]:
    values: dict[str, dict] = {}
    for item_id in sorted(set(ref) & set(cand)):
        left, right = ref[item_id], cand[item_id]
        if left.get("status") != "complete" or right.get("status") != "complete":
            continue
        if _truncated(left) or _truncated(right):
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
    # Keep a row for categories whose items were all truncated or missing: the
    # truncation counts are the evidence there, not an absent row.
    for record in list(ref.values()) + list(cand.values()):
        name = record.get(key) or "unknown"
        grouped.setdefault(name, {"pairs": [], "ref_only": 0, "cand_only": 0})
    result = {}
    for name in sorted(grouped):
        entry = grouped[name]
        summary = stats.paired_stats(entry["pairs"], reps=reps)
        summary["ref_only"] = entry["ref_only"]
        summary["cand_only"] = entry["cand_only"]
        result[name] = summary
    return result


def _trunc_counts(records: dict, key: str) -> dict:
    counts: dict[str, dict] = {}
    for record in records.values():
        if not record.get("category"):
            continue
        name = record.get(key) or "unknown"
        entry = counts.setdefault(name, {"items": 0, "truncated": 0})
        entry["items"] += 1
        if _truncated(record):
            entry["truncated"] += 1
    return counts


def _add_trunc_counts(entries: dict, ref: dict, cand: dict) -> None:
    for name, entry in entries.items():
        left = ref.get(name, {})
        right = cand.get(name, {})
        left_items = left.get("items") or 0
        right_items = right.get("items") or 0
        left_trunc = left.get("truncated") or 0
        right_trunc = right.get("truncated") or 0
        entry["truncated_ref"] = left_trunc
        entry["truncated_cand"] = right_trunc
        entry["truncated_rate_ref"] = round(left_trunc / left_items, 4) if left_items else 0.0
        entry["truncated_rate_cand"] = round(right_trunc / right_items, 4) if right_items else 0.0


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
    ref_complete = sum(
        1 for record in ref.values() if record.get("status") == "complete" and not _truncated(record)
    )
    cand_complete = sum(
        1 for record in cand.values() if record.get("status") == "complete" and not _truncated(record)
    )
    ref_truncated = sum(1 for record in ref.values() if _truncated(record))
    cand_truncated = sum(1 for record in cand.values() if _truncated(record))
    paired_ids = {
        item_id
        for item_id in set(ref) & set(cand)
        if ref[item_id].get("status") == "complete"
        and cand[item_id].get("status") == "complete"
        and not _truncated(ref[item_id])
        and not _truncated(cand[item_id])
        and isinstance(ref[item_id].get("score"), dict)
        and isinstance(cand[item_id].get("score"), dict)
    }
    complete = (
        not mismatched
        and ref_planned is not None
        and cand_planned is not None
        and ref_complete + ref_truncated == ref_planned
        and cand_complete + cand_truncated == cand_planned
    )
    categories = _summarize(ref, cand, "category", reps=reps)
    tasks = _summarize(ref, cand, "subcategory", reps=reps)
    _add_trunc_counts(categories, _trunc_counts(ref, "category"), _trunc_counts(cand, "category"))
    _add_trunc_counts(tasks, _trunc_counts(ref, "subcategory"), _trunc_counts(cand, "subcategory"))
    truncation = {
        "gate": runner.TRUNCATION_GATE,
        "ref": truncation_summary(ref, (ref_meta or {}).get("plan_counts")),
        "cand": truncation_summary(cand, (cand_meta or {}).get("plan_counts")),
    }
    truncation["failed"] = [f"ref:{category}" for category in truncation["ref"]["failed"]] + [
        f"cand:{category}" for category in truncation["cand"]["failed"]
    ]
    truncation["unreliable"] = bool(truncation["failed"])
    status = (
        "mismatched"
        if mismatched
        else "truncated"
        if truncation["unreliable"]
        else "complete"
        if complete
        else "incomplete"
    )
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
            "truncation": truncation_summary(
                floor_records, (floor_run["meta"] or {}).get("plan_counts")
            ),
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
            "truncated": ref_truncated,
            "records": len(ref),
        },
        "cand": {
            "path": cand_run["path"],
            "label": cand_meta.get("label"),
            "url": cand_meta.get("url"),
            "planned": cand_planned,
            "complete": cand_complete,
            "truncated": cand_truncated,
            "records": len(cand),
        },
        "pairing": {
            "paired": len(paired_ids),
            "ref_only": len(set(ref) - set(cand)),
            "cand_only": len(set(cand) - set(ref)),
            "ref_truncated": ref_truncated,
            "cand_truncated": cand_truncated,
        },
        "checks": checks,
        "truncation": truncation,
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
    truncation = report.get("truncation") or {}
    ref_trunc = report["ref"].get("truncated", 0)
    cand_trunc = report["cand"].get("truncated", 0)
    lines = [
        f"# Inkling task evaluation: {cand_label} vs {ref_label}",
        "",
        f"status: **{report['status']}**; paired {report['pairing']['paired']} items; "
        f"{ref_label} {report['ref']['complete']}/{report['ref']['planned']} complete, "
        f"{ref_trunc} truncated; {cand_label} {report['cand']['complete']}/{report['cand']['planned']} "
        f"complete, {cand_trunc} truncated",
    ]
    gate = truncation.get("gate", 0.05)
    if truncation.get("unreliable"):
        lines.append(
            f"truncation gate (>{100 * gate:.0f}% per category): **FAIL** "
            f"[{', '.join(truncation.get('failed') or [])}]; **UNRELIABLE, do not interpret the deltas**"
        )
    else:
        lines.append(f"truncation gate (>{100 * gate:.0f}% per category): pass")
    lines += [
        "",
        "| category | n | trunc ref/cand | ref % | cand % | delta pp | 95% CI (pp) | McNemar p | MDD pp |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for name, entry in report["categories"].items():
        ci = (
            "n/a"
            if entry["ci_low"] is None
            else f"{100 * entry['ci_low']:+.1f} to {100 * entry['ci_high']:+.1f}"
        )
        mdd = "n/a" if entry["mdd"] is None else f"{100 * entry['mdd']:.1f}"
        lines.append(
            f"| {name} | {entry['n']} | {entry.get('truncated_ref', 0)}/{entry.get('truncated_cand', 0)} | "
            f"{_pct(entry['ref_acc'])} | {_pct(entry['cand_acc'])} | "
            f"{_delta(entry['delta'])} | {ci} | {entry['mcnemar_p']} | {mdd} |"
        )
    lines += ["", "## Tasks", "", "| task | n | trunc r/c | ref % | cand % | delta pp | flips (r/c) |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, entry in report["tasks"].items():
        lines.append(
            f"| {name} | {entry['n']} | {entry.get('truncated_ref', 0)}/{entry.get('truncated_cand', 0)} | "
            f"{_pct(entry['ref_acc'])} | {_pct(entry['cand_acc'])} | "
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
    if report["status"] == "truncated":
        lines += [
            "",
            "**UNRELIABLE**: the server hit max_tokens; truncated items are excluded from "
            "accuracy and the paired statistics. Raise the task budget and rerun before "
            "interpreting any delta or flip.",
        ]
    elif report["status"] != "complete":
        lines += ["", f"incomplete or mismatched: {json.dumps(report['checks'], sort_keys=True)[:600]}"]
    return "\n".join(lines) + "\n"


def write_report(report: dict, json_path: Path, md_path: Path | None = None) -> None:
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    if md_path is not None:
        Path(md_path).write_text(markdown(report))
