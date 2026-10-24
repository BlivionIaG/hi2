# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Second-level scale scope: per row vs per tensor, linear vs log 8-bit block step.

Same held-out protocol as scale_study.py (calibrate on the first half of the
tokens, evaluate on the second). INT2 and INT4, rotation off, weighted encoder.
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import scale_study as ss  # noqa: E402

MODES = {
    "fp16 per block (v3.1)": 0,
    "row  lin step, lin lo (v3.2 draft)": 1,
    "row  lin step, z lo": "row/lin/z",
    "row  log step, z lo": "row/log/z",
    "tensor lin step, lin lo": "tensor/lin/lin",
    "tensor lin step, z lo": "tensor/lin/z",
    "tensor log step, z lo (nvfp4-like scope)": "tensor/log/z",
}


def main(model_dir, text_file, out_dir):
    weights, acts, _ = ss.capture(Path(model_dir), Path(text_file).read_text())
    g = torch.Generator().manual_seed(0)
    rows = {}
    for key, w in weights.items():
        x = acts[key]
        half = x.shape[0] // 2
        idx = torch.randperm(w.shape[0], generator=g)[: ss.ROWS]
        rows[key] = (w[idx], x[:half], x[half:])
    out = []
    for bits, B in ((2, 16), (2, 32), (2, 64), (4, 32)):
        for label, mode in MODES.items():
            errs = []
            for w, x1, x2 in rows.values():
                k = w.shape[1]
                d = (x1**2).mean(0)
                wb = w.reshape(w.shape[0], k // B, B)
                db = d.reshape(1, k // B, B).expand(w.shape[0], -1, -1)
                rec = ss.encode(wb, db, 2**bits, "affine", "mse_w", mode).reshape(
                    w.shape
                )
                errs.append(ss.rel(x2 @ rec.T, x2 @ w.T))
            r = dict(
                bits=bits,
                B=B,
                mode=label,
                out_err=sum(errs) / len(errs),
                bpw=bits
                + (
                    32 / B
                    if mode == 0
                    else 16 / B + (48 / 896 if str(mode)[:3] != "ten" else 0)
                ),
            )
            out.append(r)
            print(r, flush=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "scope_study.json").write_text(json.dumps(out, indent=1))
    print("DONE")


if __name__ == "__main__":
    main(*sys.argv[1:4])
