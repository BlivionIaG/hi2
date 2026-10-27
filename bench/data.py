# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Dataset access for the tasks.

Rows come from ``$HI2_BENCH_DATA/<path>__<name>__<split>.jsonl`` when that file exists (an
export, so a client without ``datasets`` works), else from ``datasets.load_dataset`` with
``cache_dir=$HI2_BENCH_CACHE`` (unset: the ``datasets`` default). ``HF_DATASETS_OFFLINE=1``
makes ``datasets`` use its cache only.

    python -m bench.data export DIR [--tasks gsm8k,mbpp]   # write the jsonl exports
"""

import argparse
import json
import os
import random
from pathlib import Path

ENV_DATA = "HI2_BENCH_DATA"
ENV_CACHE = "HI2_BENCH_CACHE"


class DatasetMissing(RuntimeError):
    """A dataset the task needs is not available; the message says how to get it."""


def export_name(path: str, name: str | None, split: str) -> str:
    return (
        "__".join([path.replace("/", "__"), *([name] if name else []), split])
        + ".jsonl"
    )


def hf_rows(path: str, name: str | None, split: str) -> list[dict]:
    d = os.environ.get(ENV_DATA)
    f = Path(d) / export_name(path, name, split) if d else None
    if f is not None and f.exists():
        return [json.loads(line) for line in f.read_text().splitlines() if line]
    try:
        import datasets
    except ImportError as e:
        raise DatasetMissing(
            f"{path} ({name or '-'}, {split}): the `datasets` package is not installed and "
            f"no export is in ${ENV_DATA}. Install it, or run `python -m bench.data export "
            f"DIR` where it is and point ${ENV_DATA} at DIR."
        ) from e
    try:
        ds = datasets.load_dataset(
            path, name, split=split, cache_dir=os.environ.get(ENV_CACHE)
        )
    except Exception as e:
        raise DatasetMissing(
            f"{path} ({name or '-'}, {split}) could not be loaded: {e}. Online, `datasets` "
            f"downloads it; offline (HF_DATASETS_OFFLINE=1) it must already be in the cache "
            f"(${ENV_CACHE} or the default) or exported to ${ENV_DATA}."
        ) from e
    return [dict(r) for r in ds]


def sample(items: list, limit: int | None, seed: int) -> list:
    """A seeded subset that keeps the dataset order; the whole list when it fits."""
    if limit is None or limit >= len(items):
        return items
    keep = sorted(random.Random(seed).sample(range(len(items)), limit))
    return [items[i] for i in keep]


def main() -> int:
    from bench import tasks

    ap = argparse.ArgumentParser(description="export the task datasets to jsonl")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("dir", type=Path)
    e.add_argument("--tasks", default="all")
    a = ap.parse_args()
    a.dir.mkdir(parents=True, exist_ok=True)
    for t in tasks.resolve(a.tasks).values():
        for spec in getattr(t, "DATASETS", ()):
            rows = hf_rows(*spec)
            out = a.dir / export_name(*spec)
            out.write_text("".join(json.dumps(r) + "\n" for r in rows))
            print(f"{out}: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
