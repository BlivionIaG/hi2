# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Regenerate the vendored benchmark subsets from the Hub datasets.

Run this only when the vendored exports must change; the evaluation itself never
imports ``datasets`` and never touches the network.  The selection is fixed:
GSM8K test rows 0-39, MBPP sanitized test rows 0-9, HumanEval test rows 0-19.
Authored files (``math_authored.jsonl``, ``c_code.jsonl``, ``ts_code.jsonl`` and
``multilingual_data.py``) are not generated here.

    HF_DATASETS_OFFLINE=1 python -m tools.inkling_eval.build_data

With the Hugging Face caches populated the command is offline; without them the
``datasets`` package downloads the three public datasets once.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"


def _write(name: str, rows: list[dict]) -> None:
    path = DATA / name
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"{name}: {len(rows)} rows, {path.stat().st_size} bytes, sha256 {digest}")


def main(argv=None) -> int:
    global DATA
    from datasets import load_dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DATA))
    args = parser.parse_args(argv)
    DATA = Path(args.out)
    DATA.mkdir(parents=True, exist_ok=True)

    gsm = load_dataset("openai/gsm8k", "main", split="test")
    _write(
        "gsm8k.jsonl",
        [
            {
                "source": "openai/gsm8k",
                "split": "test",
                "row": index,
                "question": gsm[index]["question"],
                "answer": gsm[index]["answer"],
            }
            for index in range(40)
        ],
    )

    mbpp = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    mbpp_rows = []
    for index in range(10):
        row = mbpp[index]
        match = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", row["code"])
        mbpp_rows.append(
            {
                "task_id": row["task_id"],
                "prompt": row["prompt"],
                "code": row["code"],
                "test_imports": row["test_imports"],
                "test_list": row["test_list"],
                "entry_point": match.group(1),
            }
        )
    _write("mbpp.jsonl", mbpp_rows)

    humaneval = load_dataset("openai/openai_humaneval", split="test")
    _write(
        "humaneval.jsonl",
        [
            {
                "task_id": humaneval[index]["task_id"],
                "prompt": humaneval[index]["prompt"],
                "canonical_solution": humaneval[index]["canonical_solution"],
                "test": humaneval[index]["test"],
                "entry_point": humaneval[index]["entry_point"],
            }
            for index in range(20)
        ],
    )
    print("update data/SOURCES.md when the hashes change")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
