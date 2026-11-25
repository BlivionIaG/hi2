# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""A small materialized held-out subset of a v2 calibration directory.

The v2 held-out set has 27 files/820 windows. Stage 8 scores a fixed, balanced
subset in sorted file order, with row indices spread over each file. Quotas:
code 18 (2 per language), multilingual 24 (2 per language), agentic 20,
math 20, general 19, long 19 at n=120. Long context uses 8K and 16K rows when
the 16K preflight passed; 32K is never selected.

    python experiments/inkling_eval_subset.py --src DIR --dst DIR \\
        --max-length 16384 --windows 120 --balanced
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def spaced_rows(n: int, take: int) -> list[int]:
    """``take`` row indices spread over ``[0, n)``."""
    take = min(int(take), int(n))
    if take <= 0:
        return []
    if take == 1:
        return [0]
    idx = sorted({round(i * (n - 1) / (take - 1)) for i in range(take)})
    if len(idx) < take:
        seen = set(idx)
        for i in range(n):
            if i not in seen:
                idx.append(i)
                seen.add(i)
                if len(idx) == take:
                    break
        idx.sort()
    return idx[:take]


def select_windows(counts: dict[str, int], total: int) -> dict[str, int]:
    """Simple selection: first window of every file in sorted order, then
    extra windows in the same order, until ``total``."""
    selected = {name: 0 for name in sorted(counts)}
    remaining = min(int(total), sum(max(0, int(n)) for n in counts.values()))
    if remaining <= 0:
        return selected
    for name in selected:
        if remaining and counts[name] > 0:
            selected[name] = 1
            remaining -= 1
    while remaining:
        progressed = False
        for name in selected:
            if remaining and selected[name] < counts[name]:
                selected[name] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return {k: v for k, v in selected.items() if v}


def category_quotas(total: int) -> dict[str, int]:
    """Fixed quotas with >=8 per category, 2 per code/multilingual language.

    ``total`` must be at least 18 + 24 + 8 * 4 = 74; at 120 the quotas are
    code 18, multilingual 24, agentic 20, math 20, general 19, long 19.
    """
    total = int(total)
    if total < 74:
        raise SystemExit(f"balanced subset needs at least 74 windows, got {total}")
    rest = total - 42
    base, rem = divmod(rest, 4)
    q = {
        "code": 18,
        "multilingual": 24,
        "agentic": base + (1 if rem >= 1 else 0),
        "math_reasoning": base + (1 if rem >= 2 else 0),
        "general_chat": base + (1 if rem >= 3 else 0),
        "long_context": base,
    }
    assert sum(q.values()) == total
    return q


def _take_files(
    names: list[str],
    quota: int,
    files: dict,
    used: dict[str, list[int]],
) -> None:
    """Distribute ``quota`` rows over files in sorted order, spaced per file."""
    names = [n for n in names if int(files[n]["windows"]) > 0]
    if not names:
        return
    base, rem = divmod(int(quota), len(names))
    for i, name in enumerate(names):
        want = base + (1 if i < rem else 0)
        n = int(files[name]["windows"])
        rows = spaced_rows(n, min(want, n))
        if rows:
            used.setdefault(name, []).extend(rows)


def select_balanced(
    files: dict, max_length: int, total: int
) -> dict[str, list[int]]:
    """Fixed category/language quotas, row-spread; returns row indices per file."""
    quotas = category_quotas(total)
    used: dict[str, list[int]] = {}
    by_cat: dict[str, list[str]] = {}
    for name, info in sorted(files.items()):
        if not name.startswith("heldout_"):
            continue
        if int(info["length"]) > int(max_length):
            continue
        by_cat.setdefault(str(info["category"]), []).append(name)

    # code and multilingual: >= 2 rows per language.
    for cat in ("code", "multilingual"):
        names = by_cat.get(cat, [])
        langs: dict[str, list[str]] = {}
        for name in names:
            langs.setdefault(str(files[name]["language"]), []).append(name)
        per_lang = 2
        for lang in sorted(langs):
            for name in langs[lang]:
                n = int(files[name]["windows"])
                used.setdefault(name, []).extend(spaced_rows(n, min(per_lang, n)))

    # Long context: prefer 16K when it is in the allowed set, otherwise 8K.
    long_names = by_cat.get("long_context", [])
    if not long_names:
        # No long window survives max_length: redistribute its quota over the
        # remaining three categories instead of filling in file-sort order.
        quotas = dict(quotas)
        rest = total - quotas["code"] - quotas["multilingual"]
        base, rem = divmod(rest, 3)
        for i, cat in enumerate(("agentic", "math_reasoning", "general_chat")):
            quotas[cat] = base + (1 if i < rem else 0)
        quotas["long_context"] = 0

    # One file per agentic/math/general category.
    for cat in ("agentic", "math_reasoning", "general_chat"):
        _take_files(by_cat.get(cat, []), quotas[cat], files, used)

    by_len: dict[int, list[str]] = {}
    for name in long_names:
        by_len.setdefault(int(files[name]["length"]), []).append(name)
    lengths = sorted(by_len)
    if 16384 in by_len and 8192 in by_len:
        q8 = quotas["long_context"] // 2
        q16 = quotas["long_context"] - q8
        _take_files(by_len[8192], q8, files, used)
        _take_files(by_len[16384], q16, files, used)
    else:
        _take_files(long_names, quotas["long_context"], files, used)

    # Fill any deficit from files with remaining rows (never 32K: max_length).
    have = sum(len(v) for v in used.values())
    if have < total:
        for name in sorted(files):
            if not name.startswith("heldout_"):
                continue
            if int(files[name]["length"]) > int(max_length):
                continue
            if have >= total:
                break
            n = int(files[name]["windows"])
            already = set(used.get(name, []))
            for row in spaced_rows(n, n):
                if row not in already and have < total:
                    used.setdefault(name, []).append(row)
                    already.add(row)
                    have += 1
    # Trim deterministic, largest file first, if quota rounding over-delivers.
    if have > total:
        for name in sorted(used, key=lambda n: (-len(used[n]), n)):
            while have > total and len(used[name]) > 1:
                used[name].pop()
                have -= 1
            if have == total:
                break
    return {k: sorted(v) for k, v in used.items() if v}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_subset(
    src: Path,
    dst: Path,
    max_length: int,
    windows: int = 120,
    balanced: bool = True,
) -> dict:
    """Write selected windows under ``dst`` and return the new manifest."""
    src, dst = Path(src), Path(dst)
    manifest = json.loads((src / "manifest.json").read_text())
    files = manifest["files"]
    if balanced:
        selected = select_balanced(files, max_length, windows)
    else:
        eligible = {
            name: int(info["windows"])
            for name, info in sorted(files.items())
            if name.startswith("heldout_") and int(info["length"]) <= int(max_length)
        }
        counts = select_windows(eligible, windows)
        selected = {name: list(range(take)) for name, take in counts.items()}
    if not selected:
        raise SystemExit("no windows selected")
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    new_files = {}
    for name, rows in sorted(selected.items()):
        ids = np.load(src / name, mmap_mode="r")
        out_ids = ids[sorted(set(int(r) for r in rows))]
        out = dst / name
        np.save(out, out_ids)
        info = dict(files[name])
        info["bytes"] = out.stat().st_size
        info["sha256"] = _sha(out)
        info["windows"] = int(out_ids.shape[0])
        new_files[name] = info
    manifest["files"] = new_files
    manifest["subset"] = {
        "balanced": bool(balanced),
        "windows": sum(i["windows"] for i in new_files.values()),
        "max_length": int(max_length),
        "row_spread": True,
    }
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--max-length", type=int, default=16384)
    ap.add_argument("--windows", type=int, default=120)
    ap.add_argument(
        "--balanced",
        action="store_true",
        help="fixed category/language quotas and row-spread (stage 8)",
    )
    a = ap.parse_args(argv)
    m = build_subset(a.src, a.dst, a.max_length, a.windows, a.balanced)
    print(
        json.dumps(
            {
                "dst": str(a.dst),
                "files": len(m["files"]),
                "windows": m["subset"]["windows"],
                "max_length": a.max_length,
                "categories": sorted({i["category"] for i in m["files"].values()}),
                "languages": sorted({i["language"] for i in m["files"].values()}),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
