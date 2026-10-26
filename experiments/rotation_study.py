# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Where does rotation help? Group size, random signs and AWQ-style smoothing.

For each matrix the weights and activations are transformed by the same
orthogonal-times-diagonal map ``T`` (``W' = W T``, ``X' = X T^-T``) so the exact
product is unchanged: per-column smoothing ``s`` (``W s``, ``X / s``), random signs,
then an orthonormal Hadamard of size ``G`` on groups of ``G`` columns. The weighted
encoder of the v3.2 draft quantises ``W'`` using the second moment of ``X'``
(calibration half), and the error is measured on the other half of the tokens.

    python experiments/rotation_study.py <model_dir> <text_file> <out_dir>
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import scale_study as ss  # noqa: E402


def hadamard(g: int) -> torch.Tensor:
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < g:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / g**0.5).float()


def transform(
    m: torch.Tensor, s: torch.Tensor, sign: torch.Tensor, g: int, inv_s: bool
):
    """Apply smoothing, signs and the group Hadamard to the columns of ``m``."""
    m = m / s if inv_s else m * s
    m = m * sign
    if g > 1:
        k = m.shape[1]
        m = (m.reshape(m.shape[0], k // g, g) @ hadamard(g)).reshape(m.shape[0], k)
    return m


def kind(key):
    return key[1]


def main(model_dir, text_file, out_dir):
    weights, acts, _ = ss.capture(Path(model_dir), Path(text_file).read_text())
    gen = torch.Generator().manual_seed(0)
    rows = {}
    for key, w in weights.items():
        x = acts[key]
        half = x.shape[0] // 2
        idx = torch.randperm(w.shape[0], generator=gen)[: ss.ROWS]
        rows[key] = (w[idx], x[:half], x[half:])
    variants = []
    for alpha in (0.0, 0.5, 1.0):
        for g, signs in ((1, False), (16, False), (16, True), (64, True), (128, True)):
            if g == 1 and signs:
                continue
            variants.append((alpha, g, signs))
    out = []
    for bits, B in ((2, 32), (4, 32)):
        for alpha, g, signs in variants:
            per = {}
            for key, (w, x1, x2) in rows.items():
                k = w.shape[1]
                d = (x1**2).mean(0).clamp_min(1e-8)
                s = (
                    (d / d.log().mean().exp()).pow(alpha / 2)
                    if alpha
                    else torch.ones(k)
                )
                sg = (
                    (torch.randint(0, 2, (k,), generator=gen) * 2 - 1).float()
                    if signs
                    else torch.ones(k)
                )
                wt = transform(w, s, sg, g, False)
                x1t = transform(x1, s, sg, g, True)
                x2t = transform(x2, s, sg, g, True)
                dt = (x1t**2).mean(0)
                n = 2**bits
                wb = wt.reshape(w.shape[0], k // B, B)
                db = dt.reshape(1, k // B, B).expand(w.shape[0], -1, -1)
                rec = ss.encode(wb, db, n, "affine", "mse_w", 1).reshape(w.shape)
                per[key] = ss.rel(x2t @ rec.T, x2 @ w.T)
            by_kind = {}
            for key, e in per.items():
                by_kind.setdefault(kind(key), []).append(e)
            row = dict(
                bits=bits,
                B=B,
                alpha=alpha,
                G=g,
                signs=signs,
                out_err=sum(per.values()) / len(per),
                by_kind={k_: sum(v) / len(v) for k_, v in by_kind.items()},
            )
            out.append(row)
            print(row, flush=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "rotation_study.json").write_text(json.dumps(out, indent=1))
    print("DONE")


if __name__ == "__main__":
    main(*sys.argv[1:4])
