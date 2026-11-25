# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Balanced training text from the v2 calibration windows (CPU only).

``hi2 quantize --calib`` reads text and takes its first ``--calib-samples x --seq-len``
tokens.  Concatenating the six ``src/*.jsonl`` files would fill that prefix with agentic
and code rows only.  This script decodes every ``calib_*`` window of a v2 directory
round-robin over the 27 category/language/length files, so a prefix of the output covers
all six categories, all 12 languages and the 8k/16k/32k long windows.  Decoded long
windows contribute several 2048-token windows to a 2048-token calibration pass; the
separate GPU preflight measures the full-length pass.

    python experiments/inkling_calib_v2_text.py --calib-dir stage4/calib_v2_nocopyleft \
        --src ~/hfcache/inkling2-mini --out stage8/calib_v2_balanced.txt \
        --meta stage8/calib_v2_balanced.json
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def interleave(entries):
    """Round-robin the windows of ``[(key, ids)]``; every file's window i precedes any
    file's window i+1, so a prefix covers all files."""
    order = []
    longest = max((len(ids) for _, ids in entries), default=0)
    for i in range(longest):
        for key, ids in entries:
            if i < len(ids):
                order.append((key, i))
    return order


def load_calib(calib_dir: Path):
    mpath = calib_dir / "manifest.json"
    manifest = json.loads(mpath.read_text())
    if manifest.get("schema") != "hi2.inkling-calib-v2/1" or not manifest.get("ok"):
        raise SystemExit(f"{mpath}: not a verified v2 manifest")
    entries = []
    for name in sorted(manifest["files"]):
        if not (name.startswith("calib_") and name.endswith("_ids.npy")):
            continue
        ids = np.load(calib_dir / name)
        if ids.ndim != 2 or ids.dtype.kind != "i":
            raise SystemExit(f"{name}: expected [windows, length] int ids")
        key = name[len("calib_") : -len("_ids.npy")]
        entries.append((key, ids))
    if not entries:
        raise SystemExit(f"{calib_dir}: no calib_*_ids.npy")
    return manifest, entries


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--calib-dir", type=Path, required=True)
    ap.add_argument("--src", type=Path, required=True, help="stock model/tokenizer")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True)
    a = ap.parse_args()

    from transformers import AutoTokenizer

    manifest, entries = load_calib(a.calib_dir)
    by_key = dict(entries)
    tok = AutoTokenizer.from_pretrained(a.src, trust_remote_code=False)
    parts = []
    windows = 0
    for key, i in interleave(entries):
        row = by_key[key][i]
        parts.append(
            tok.decode(
                row.tolist(),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
        windows += 1
    text = "\n\n".join(parts) + "\n"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(text)
    meta = {
        "calib_dir": str(a.calib_dir),
        "manifest_sha256": hashlib.sha256(
            (a.calib_dir / "manifest.json").read_bytes()
        ).hexdigest(),
        "excluded_license_classes": manifest.get("excluded_license_classes", []),
        "excluded_records": manifest.get("excluded_records"),
        "files": [
            {
                "key": k,
                "windows": int(v.shape[0]),
                "length": int(v.shape[1]),
            }
            for k, v in entries
        ],
        "windows": windows,
        "text": str(a.out),
        "text_sha256": hashlib.sha256(a.out.read_bytes()).hexdigest(),
        "text_tokens": len(tok(text, add_special_tokens=False).input_ids),
        "order": "round-robin over calib_* files, then window index",
    }
    a.meta.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
