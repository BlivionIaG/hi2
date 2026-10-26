# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Task-style QAD calibration text from TRAINING splits only (docs/rdna2/hi2-lowbit.md).

Writes one text file of
* ARC-Easy and ARC-Challenge **training** questions (llama.cpp ``arc-*-training.bin`` of
  ikawrakow/validation-datasets-for-llama.cpp), each as ``Question: "..." Answer: <correct
  answer>`` (the scorer's prompt shape, so this is training-domain adaptation), shuffled
  and grouped in blocks, interleaved with
* WikiText-2 **train** paragraphs (``wiki.train.raw.gz`` of the same repository) as
  general encyclopedic prose.

The scored sets (ARC validation, HellaSwag validation, MMLU test) are never read. No
HellaSwag, MMLU or WinoGrande training split is on the box, so none is used.

usage: make_calib_tasks.py <ikawrakow dataset dir> <out.txt> [--wiki-chars N] [--seed S]
"""

import argparse
import gzip
import random
import struct
from pathlib import Path


def _string(buf: bytes, off: int) -> tuple[str, int]:
    (n,) = struct.unpack_from("<I", buf, off)
    off += 4
    return buf[off : off + n].decode("utf-8", "replace"), off + n


def _answers(buf: bytes, off: int):
    (n,) = struct.unpack_from("<i", buf, off)
    off += 4
    ans = []
    for _ in range(n):
        a, off = _string(buf, off)
        ans.append(a)
    labels = list(struct.unpack_from(f"<{n}i", buf, off))
    return ans, labels, off + 4 * n


def read_mc(path: Path) -> list[tuple[str, list[str], list[int]]]:
    """Entries of a llama.cpp multiple-choice ``.bin`` (see its convert.cpp)."""
    buf = path.read_bytes()
    (n,) = struct.unpack_from("<I", buf, 0)
    off = 4 + 4 * n  # skip the position table
    out = []
    for _ in range(n):
        q, off = _string(buf, off)
        ans, labels, off = _answers(buf, off)
        _, _, off = _answers(buf, off)  # multiple-correct (unused for ARC)
        out.append((q, ans, labels))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--wiki-chars", type=int, default=2_400_000)
    ap.add_argument("--block", type=int, default=12, help="questions per block")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    qa = []
    for name in ("arc-easy-training.bin", "arc-challenge-training.bin"):
        for q, ans, labels in read_mc(a.data / name):
            if 1 in labels:
                qa.append(f"{q.strip()} {ans[labels.index(1)].strip()}")
    rnd.shuffle(qa)
    with gzip.open(a.data / "wiki.train.raw.gz", "rt", encoding="utf-8") as f:
        wiki = f.read(a.wiki_chars)
    paras = [p.strip() for p in wiki.split("\n \n") if len(p.strip()) > 200]
    blocks = ["\n".join(qa[i : i + a.block]) for i in range(0, len(qa), a.block)]
    out, pi = [], 0
    per = max(1, len(paras) // max(1, len(blocks)))
    for b in blocks:
        out.append(b)
        out.extend(paras[pi : pi + per])
        pi += per
    out.extend(paras[pi:])
    a.out.write_text("\n\n".join(out))
    print(
        f"{len(qa)} ARC training questions in {len(blocks)} blocks, {len(paras)} wiki "
        f"paragraphs, {sum(len(x) for x in out)} chars -> {a.out}"
    )


if __name__ == "__main__":
    main()
