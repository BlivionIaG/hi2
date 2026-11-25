# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Paired comparison of two Inkling KL evaluations on the v2 held-out set.

The v2 evaluator (``inkling_kl_eval.py --ids DIR``) reports the per-window KL in
``eval_set.sets[key]["kl_per_window"]``.  This script pairs those windows between two
models, pools them by category and language, and applies the pre-registered gate:

    adopt the new output iff pooled paired mean(new - old) < -sem(new - old)
    AND for every held-out category mean(new - old) <= sem(new - old).

``sem`` is the sample standard deviation of the paired per-window differences divided by
sqrt(n); the OOD set is reported separately and is not part of the gate.  ``--update-run-json``
records the v2 calibration provenance in an output's ``hi2_run.json``.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _sets(doc: dict) -> dict:
    """``{key: {category, language, length, windows, kl_per_window, ...}}`` for a v2
    directory result, or one synthetic ``ood`` set for a single-file result."""
    if isinstance(doc.get("eval_set"), dict) and "sets" in doc["eval_set"]:
        return doc["eval_set"]["sets"]
    if "student" in doc:
        s = doc["student"]
        return {
            "ood": {
                "category": "ood",
                "language": "en",
                "length": doc.get("seq_len"),
                "windows": s["windows"],
                **s,
            }
        }
    raise SystemExit("evaluation JSON has neither eval_set.sets nor student")


def _stats(old: np.ndarray, new: np.ndarray) -> dict:
    old = np.asarray(old, dtype=np.float64)
    new = np.asarray(new, dtype=np.float64)
    d = new - old
    n = len(d)
    sem = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else None
    return {
        "windows": n,
        "old_mean": float(old.mean()),
        "new_mean": float(new.mean()),
        "delta": float(d.mean()),
        "sem": sem,
        "delta_median": float(np.median(d)),
        "delta_min": float(d.min()),
        "delta_max": float(d.max()),
    }


def _pair_rows(old_doc: dict, new_doc: dict, limit: int = 0) -> dict:
    old, new = _sets(old_doc), _sets(new_doc)
    if old.keys() != new.keys():
        raise SystemExit(f"set keys differ: {old.keys() ^ new.keys()}")
    rows = {}
    remaining = limit if limit and limit > 0 else None
    for key in sorted(old):
        a, b = old[key], new[key]
        if a["windows"] != b["windows"]:
            raise SystemExit(f"{key}: window counts differ")
        old_kl = np.asarray(a["kl_per_window"], dtype=np.float64)
        new_kl = np.asarray(b["kl_per_window"], dtype=np.float64)
        if len(old_kl) != len(new_kl):
            raise SystemExit(f"{key}: KL list lengths differ")
        take = len(old_kl) if remaining is None else min(remaining, len(old_kl))
        if take == 0:
            break
        if remaining is not None:
            remaining -= take
        old_kl = old_kl[:take]
        new_kl = new_kl[:take]
        row = _stats(old_kl, new_kl)
        pairs = sorted(
            enumerate(zip(old_kl, new_kl)),
            key=lambda t: float(t[1][1] - t[1][0]),
            reverse=True,
        )
        row.update(
            category=a["category"],
            language=a["language"],
            length=a["length"],
            old_top1=a.get("top1_agreement"),
            new_top1=b.get("top1_agreement"),
            old_ppl_ratio=a.get("ppl_ratio"),
            new_ppl_ratio=b.get("ppl_ratio"),
            old_position_buckets=a.get("position_buckets"),
            new_position_buckets=b.get("position_buckets"),
            _old=old_kl.tolist(),
            _new=new_kl.tolist(),
            worst=[
                {
                    "window": int(i),
                    "old": float(x),
                    "new": float(y),
                    "delta": float(y - x),
                }
                for i, (x, y) in pairs[:5]
            ],
        )
        rows[key] = row
    return rows


def _pool_rows(rows: dict, field: str) -> dict:
    groups: dict[str, list[dict]] = {}
    for key, row in rows.items():
        groups.setdefault(str(row[field]), []).append(row)
    out = {}
    for name, items in sorted(groups.items()):
        n = sum(i["windows"] for i in items)
        old = sum((i["_old"] for i in items), [])
        new = sum((i["_new"] for i in items), [])
        s = _stats(np.asarray(old), np.asarray(new))
        s.update(
            sets=sorted(k for k, r in rows.items() if r[field] == name),
            old_top1=sum((i["old_top1"] or 0.0) * i["windows"] for i in items) / n,
            new_top1=sum((i["new_top1"] or 0.0) * i["windows"] for i in items) / n,
            old_ppl_ratio=sum(
                (i["old_ppl_ratio"] or 1.0) * i["windows"] for i in items
            )
            / n,
            new_ppl_ratio=sum(
                (i["new_ppl_ratio"] or 1.0) * i["windows"] for i in items
            )
            / n,
            worst=sorted(
                [
                    w | {"set": k}
                    for k, i in rows.items()
                    if i[field] == name
                    for w in i["worst"]
                ],
                key=lambda w: w["delta"],
                reverse=True,
            )[:5],
        )
        out[name] = s
    return out


def compare(
    old_heldout: dict, new_heldout: dict, old_ood=None, new_ood=None, windows: int = 0
) -> dict:
    held = _pair_rows(old_heldout, new_heldout, limit=windows)
    categories = _pool_rows(held, "category")
    languages = _pool_rows(held, "language")
    pooled = _stats(
        np.asarray(sum((r["_old"] for r in held.values()), [])),
        np.asarray(sum((r["_new"] for r in held.values()), [])),
    )
    for r in held.values():
        r.pop("_old", None)
        r.pop("_new", None)
    pooled_ok = pooled["sem"] is not None and pooled["delta"] < -pooled["sem"]
    category_gate = {
        name: {
            "delta": d["delta"],
            "sem": d["sem"],
            "worsens": d["sem"] is None or d["delta"] > d["sem"],
        }
        for name, d in categories.items()
    }
    ood = _pair_rows(old_ood, new_ood) if old_ood is not None and new_ood is not None else None
    return {
        "sets": held,
        "pooled": pooled,
        "pooled_improves_beyond_sem": bool(pooled_ok),
        "categories": categories,
        "languages": languages,
        "category_gate": category_gate,
        "ood": ood,
        "adopt": bool(
            pooled_ok and not any(v["worsens"] for v in category_gate.values())
        ),
        "gate": "paired: adopt iff pooled delta < -sem and every held-out category delta <= sem",
    }


def update_run_json(path: Path, calib_meta: dict) -> None:
    doc = json.loads(path.read_text())
    doc["calib_v2"] = calib_meta
    path.write_text(json.dumps(doc, indent=2) + "\n")


def _md(result: dict) -> str:
    lines = [
        "# Inkling v2 PTQ comparison",
        "",
        f"gate: `{result['gate']}`",
        "",
        f"**decision: {'ADOPT NEW' if result['adopt'] else 'KEEP OLD'}**",
        "",
        "| set | n | old KL | new KL | delta | sem | old top1 | new top1 | old ppl | new ppl |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, r in [
        ("pooled", result["pooled"]),
        *[(f"cat:{k}", v) for k, v in result["categories"].items()],
        *[(f"lang:{k}", v) for k, v in result["languages"].items()],
    ]:
        sem = "n/a" if r["sem"] is None else f"{r['sem']:.4f}"
        lines.append(
            f"| {name} | {r['windows']} | {r['old_mean']:.5f} | {r['new_mean']:.5f} | "
            f"{r['delta']:+.5f} | {sem} | {r.get('old_top1', 0):.4f} | "
            f"{r.get('new_top1', 0):.4f} | {r.get('old_ppl_ratio', 1):.4f} | "
            f"{r.get('new_ppl_ratio', 1):.4f} |"
        )
    lines += ["", "## worst windows (new - old)", ""]
    for name, r in result["sets"].items():
        for w in r["worst"][:2]:
            lines.append(
                f"- {name} window {w['window']}: {w['old']:.5f} -> {w['new']:.5f} "
                f"({w['delta']:+.5f})"
            )
    if result.get("ood"):
        lines += ["", "## OOD (report only)", ""]
        for name, r in result["ood"].items():
            lines.append(
                f"- {name}: old {r['old_mean']:.5f}, new {r['new_mean']:.5f}, "
                f"delta {r['delta']:+.5f}, sem {r['sem']}"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--old-heldout", type=Path, required=True)
    ap.add_argument("--new-heldout", type=Path, required=True)
    ap.add_argument("--old-ood", type=Path)
    ap.add_argument("--new-ood", type=Path)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path)
    ap.add_argument("--run-json", type=Path)
    ap.add_argument("--calib-meta", type=Path)
    ap.add_argument(
        "--windows",
        type=int,
        default=0,
        help="score only the first N held-out windows in sorted set order (0: all)",
    )
    a = ap.parse_args()

    def load(p):
        return json.loads(p.read_text())

    result = compare(
        load(a.old_heldout),
        load(a.new_heldout),
        load(a.old_ood) if a.old_ood else None,
        load(a.new_ood) if a.new_ood else None,
        windows=a.windows,
    )
    result["windows_requested"] = a.windows
    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    a.out_json.write_text(json.dumps(result, indent=2) + "\n")
    if a.out_md:
        a.out_md.write_text(_md(result))
    if a.run_json and a.calib_meta:
        update_run_json(a.run_json, load(a.calib_meta))
    print(_md(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
