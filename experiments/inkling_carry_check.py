# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Does an HI2 output of the stock Inkling-2 directory keep the vision tower bit for bit?

``hi2.inkling_source.carry_vision`` (called by ``hi2.checkpoint``'s output step) copies
the ``model.visual.*`` tensors of the stock directory into the output's
``model-vision.safetensors`` and lists them in the output index. This runs it on the real
directory into a scratch directory and reads every tensor back: name, dtype, shape and bit
pattern against the stock's, the index entries against the stock index, and the bytes
against the file. It also says whether ``mtp.safetensors`` exists (it is deployed beside
the output, not carried) and whether ``hf_quant_config.json`` would be linked (it is not).

Exit code 0: every tensor equal and indexed. 1: a difference or a missing tensor.

    python experiments/inkling_carry_check.py --src DIR --out SCRATCH_DIR
"""

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open

from hi2.inkling_source import STALE_FILES, VISION_PREFIX, VISION_SHARD, carry_vision


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, required=True, help="stock Inkling-2 directory")
    ap.add_argument("--out", type=Path, required=True, help="scratch output directory")
    ap.add_argument("--json", type=Path, default=Path("inkling_carry_check.json"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    new_map: dict[str, str] = {}
    t0 = time.time()
    n = carry_vision(a.src, a.out, new_map)
    seconds = time.time() - t0
    stock_index = json.loads((a.src / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    want = sorted(k for k in stock_index if k.startswith(VISION_PREFIX))
    bad: list[str] = []
    elements = nbytes = 0
    with safe_open(str(a.out / VISION_SHARD), "pt") as got:
        have = sorted(got.keys())
        if have != want:
            bad.append(f"names: {len(have)} written, {len(want)} in the stock index")
        by_shard: dict[str, list[str]] = {}
        for k in want:
            by_shard.setdefault(stock_index[k], []).append(k)
        for shard, names in by_shard.items():
            with safe_open(str(a.src / shard), "pt") as ref:
                for k in names:
                    if k not in have:
                        continue
                    r, g = ref.get_tensor(k), got.get_tensor(k)
                    elements += r.numel()
                    nbytes += r.numel() * r.element_size()
                    same = (
                        r.dtype == g.dtype
                        and r.shape == g.shape
                        and torch.equal(
                            r.contiguous().reshape(-1).view(torch.uint8),
                            g.contiguous().reshape(-1).view(torch.uint8),
                        )
                    )
                    if not same:
                        bad.append(k)
    if new_map != dict.fromkeys(want, VISION_SHARD):
        bad.append("the index entries differ from the stock vision names")
    report = {
        "src": str(a.src),
        "tensors": n,
        "stock_vision_tensors": len(want),
        "elements": elements,
        "bytes": nbytes,
        "file_bytes": (a.out / VISION_SHARD).stat().st_size,
        "seconds": round(seconds, 2),
        "mismatches": bad,
        "mtp_safetensors_in_stock": (a.src / "mtp.safetensors").exists(),
        "stale_files_not_linked": sorted(
            STALE_FILES & {p.name for p in a.src.iterdir()}
        ),
        "ok": not bad and n == len(want) > 0,
    }
    a.json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
