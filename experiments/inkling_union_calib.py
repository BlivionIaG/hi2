#!/usr/bin/env python3
"""Build the 50/50 calibration union for bpw3.5-union (token-exact, no decode).

`hi2 quantize --calib` concatenates the given text files and takes the FIRST
`calib_samples * seq_len` tokens (no shuffle), so a plain concatenation of the
two corpora would take all 256 windows from the first one.  This script cuts
the first corpus at a TOKEN boundary found from the tokenizer's offset mapping
(no decode/re-encode round trip) and verifies from the ids themselves that the
union's first 128 windows are stage4/calib/calib.txt and windows 128..255 are
calib_v2_balanced.txt.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

SRC = Path(os.environ["SRC"])
CALIB_V1 = Path(os.environ["CALIB_V1"])
CALIB_V2 = Path(os.environ["CALIB_V2"])
OUT_TXT = Path(os.environ["OUT_TXT"])
OUT_META = Path(os.environ["OUT_META"])
SAMPLES = int(os.environ.get("CALIB_SAMPLES", "256"))
SEQ = int(os.environ.get("CALIB_SEQ_LEN", "2048"))
HALF = SAMPLES // 2
NEED = HALF * SEQ


def ids_of(tok, text: str) -> torch.Tensor:
    return tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]


def ids_and_offsets(tok, text: str):
    enc = tok(
        text,
        return_tensors="pt",
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    return enc["input_ids"][0], enc["offset_mapping"][0]


def attempt(tok, v1_text, v2_text, ids_v1, ids_v2, sep: str):
    sep_ids = ids_of(tok, sep)
    s = int(sep_ids.numel())
    keep = NEED - s
    _, offs = ids_and_offsets(tok, v1_text)
    k = int(offs[keep][0])
    prefix = v1_text[:k]
    prefix_ids = ids_of(tok, prefix)
    if int(prefix_ids.numel()) != keep:
        return None
    union = prefix + sep + v2_text
    u = ids_of(tok, union)
    if u.numel() < SAMPLES * SEQ:
        return None
    left_ok = torch.equal(u[:keep], prefix_ids)
    if s:
        left_ok = left_ok and torch.equal(u[keep:keep + s], sep_ids)
    right_ok = torch.equal(u[NEED : NEED + NEED], ids_v2[:NEED])
    if left_ok and right_ok:
        return union, prefix, prefix_ids, sep_ids, s, u
    return None


def main() -> int:
    tok = AutoTokenizer.from_pretrained(SRC, trust_remote_code=False)
    v1_text = CALIB_V1.read_text()
    v2_text = CALIB_V2.read_text()
    ids_v1 = ids_of(tok, v1_text)
    ids_v2 = ids_of(tok, v2_text)
    print(f"corpora tokens: v1 {ids_v1.numel()} v2 {ids_v2.numel()} need {NEED} each")
    if ids_v1.numel() < NEED or ids_v2.numel() < NEED:
        print("FAIL: a corpus has fewer than 128 x 2048 tokens")
        return 1
    built = None
    for sep in ("", "\n"):
        built = attempt(tok, v1_text, v2_text, ids_v1, ids_v2, sep)
        if built is not None:
            break
    if built is None:
        print("FAIL: no exact token-boundary union for sep '' or '\\n'")
        return 1
    union, prefix, prefix_ids, sep_ids, s, u = built
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text(union)
    sha = hashlib.sha256(OUT_TXT.read_bytes()).hexdigest()
    windows = u[: SAMPLES * SEQ].reshape(SAMPLES, SEQ)
    v1_windows = windows[:HALF]
    v2_windows = windows[HALF:]
    v1_source = ids_v1[:NEED].reshape(HALF, SEQ)
    v2_source = ids_v2[:NEED].reshape(HALF, SEQ)
    pure_v1 = int((v1_windows == v1_source).all(dim=1).sum())
    pure_v2 = int((v2_windows == v2_source).all(dim=1).sum())
    meta = {
        "union": str(OUT_TXT),
        "sha256": sha,
        "samples": SAMPLES,
        "seq_len": SEQ,
        "half_samples": HALF,
        "separator": "" if s == 0 else "\n",
        "separator_tokens": s,
        "calib_v1": {
            "path": str(CALIB_V1),
            "tokens": int(ids_v1.numel()),
            "used_tokens": int(prefix_ids.numel()),
            "windows_fully_from_v1": pure_v1,
        },
        "calib_v2": {
            "path": str(CALIB_V2),
            "tokens": int(ids_v2.numel()),
            "used_tokens": NEED,
            "windows_fully_from_v2": pure_v2,
        },
        "windows": {"v1": [0, HALF - 1], "v2": [HALF, SAMPLES - 1]},
        "verified": {
            "first_half_from_v1": pure_v1 == HALF,
            "second_half_from_v2": pure_v2 == HALF,
            "union_tokens": SAMPLES * SEQ,
            "v1_last_window_note": (
                "window 127 is 2048-%d v1 tokens + the separator" % s if s else
                "window 127 is pure v1"
            ),
        },
    }
    OUT_META.write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta, indent=1))
    ok = pure_v1 >= HALF - (1 if s else 0) and pure_v2 == HALF
    print("BUDGET-OK" if ok else "FAIL: window provenance mismatch")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
