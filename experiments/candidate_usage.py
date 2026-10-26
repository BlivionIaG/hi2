# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Which (nbits, block) candidates the allocator actually picks, over finished HI2
checkpoints (rows and weights per candidate, from ``hi2_row_nbits`` / ``hi2_row_block``).

usage: python experiments/candidate_usage.py OUT_DIR [OUT_DIR ...] [--json out.json]
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def usage(dst: Path) -> dict:
    idx = json.loads((dst / "model.safetensors.index.json").read_text())["weight_map"]
    rows: dict[str, int] = {}
    weights: dict[str, int] = {}
    for shard in sorted(set(idx.values())):
        with safe_open(dst / shard, "pt") as f:
            for k in f.keys():  # noqa: SIM118
                if not k.endswith(".hi2_row_nbits"):
                    continue
                base = k.removesuffix(".hi2_row_nbits")
                nb = f.get_tensor(k).to(torch.int64)
                bl = f.get_tensor(base + ".hi2_row_block").to(torch.int64)
                kdim = f.get_slice(base + ".hi2_had").get_shape()[-1] * 16
                for a, b in torch.unique(torch.stack([nb, bl], 1), dim=0).tolist():
                    n = int(((nb == a) & (bl == b)).sum())
                    key = f"INT{a}/B{b}"
                    rows[key] = rows.get(key, 0) + n
                    weights[key] = weights.get(key, 0) + n * kdim
    tr, tw = sum(rows.values()), sum(weights.values())
    return {
        "rows_pct": {k: round(100 * v / tr, 2) for k, v in sorted(rows.items())},
        "weights_pct": {k: round(100 * v / tw, 2) for k, v in sorted(weights.items())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    out = {}
    for d in a.dirs:
        if not (d / "model.safetensors.index.json").exists():
            continue
        try:
            out[str(d)] = usage(d)
        except FileNotFoundError:  # shards deleted to save space
            continue
        print(d.name, json.dumps(out[str(d)]["weights_pct"]), flush=True)
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
