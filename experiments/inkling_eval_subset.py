# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""A small materialized held-out subset of a v2 calibration directory.

``inkling_v2_compare`` and the evaluator want a directory with a manifest. The
v2 held-out set has 27 files and 820 windows; stage 8 scores only the first
``--windows`` windows in sorted file order, dropping windows longer than
``--max-length`` (the 32K files are not preflighted). This module writes that
subset with the exact bytes and sha256 the evaluator checks.

    python experiments/inkling_eval_subset.py --src DIR --dst DIR \\
        --max-length 16384 --windows 32
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def select_windows(counts: dict[str, int], total: int) -> dict[str, int]:
    """First window of every file in sorted order, then extra windows in the
    same order, until ``total``."""
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


def build_subset(src: Path, dst: Path, max_length: int, windows: int = 32) -> dict:
    """Write the selected windows under ``dst`` and return the new manifest."""
    manifest = json.loads((Path(src) / "manifest.json").read_text())
    files = manifest["files"]
    eligible = {
        name: int(info["windows"])
        for name, info in sorted(files.items())
        if name.startswith("heldout_") and int(info["length"]) <= int(max_length)
    }
    if not eligible:
        raise SystemExit(f"no heldout files at length <= {max_length}")
    selected = select_windows(eligible, windows)
    if not selected:
        raise SystemExit("no windows selected")
    dst = Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    new_files = {}
    for name, take in selected.items():
        ids = np.load(Path(src) / name, mmap_mode="r")[:take]
        out = dst / name
        np.save(out, ids)
        info = dict(files[name])
        info["bytes"] = out.stat().st_size
        info["sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
        info["windows"] = int(take)
        new_files[name] = info
    manifest["files"] = new_files
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--max-length", type=int, default=16384)
    ap.add_argument("--windows", type=int, default=32)
    a = ap.parse_args(argv)
    m = build_subset(a.src, a.dst, a.max_length, a.windows)
    print(
        json.dumps(
            {
                "dst": str(a.dst),
                "files": len(m["files"]),
                "windows": sum(i["windows"] for i in m["files"].values()),
                "max_length": a.max_length,
                "categories": sorted({i["category"] for i in m["files"].values()}),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
