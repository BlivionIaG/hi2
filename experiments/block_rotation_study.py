# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Does a wider input-side rotation let HI2 use larger blocks (fewer block parameters)?

MLP projections of a real model, activation-weighted encoder with nearest rounding, smoothing 0.5
then rotation of group size G, block size B, INT2 and INT4. Output error is measured on held-out
tokens (calibration on the first half). bpw counts the code bits, 8-bit block parameters (16/B)
and row constants (48/K). Rows stay independent: only the input dimension is rotated.

    python experiments/block_rotation_study.py <model_dir> <text_file> <out_dir>
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import rotation_study as rs  # noqa: E402
import scale_study as ss  # noqa: E402

LAYERS = (0, 8, 16, 23)
PROJS = ("mlp.up_proj", "mlp.down_proj", "mlp.gate_proj")


def capture(model_dir: Path, text: str, tokens: int = 4096):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = (
        AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
        .eval()
        .cuda()
    )
    ids = tok(text, return_tensors="pt").input_ids[:, :tokens]
    layers = model.model.layers
    acts, weights, hooks = {}, {}, []
    for li in LAYERS:
        for path in PROJS:
            mod = layers[li].get_submodule(path)
            weights[(li, path)] = mod.weight.detach().cpu().clone()
            hooks.append(
                mod.register_forward_pre_hook(
                    lambda m, a, key=(li, path): acts.__setitem__(
                        key, a[0].detach().reshape(-1, a[0].shape[-1]).clone()
                    )
                )
            )
    with torch.no_grad():
        model(ids.cuda())
    for h in hooks:
        h.remove()
    return weights, {k: v.cpu() for k, v in acts.items()}


def main(model_dir, text_file, out_dir):
    weights, acts = capture(Path(model_dir), Path(text_file).read_text())
    gen = torch.Generator().manual_seed(0)
    rows = {}
    for key, w in weights.items():
        x = acts[key]
        half = x.shape[0] // 2
        idx = torch.randperm(w.shape[0], generator=gen)[:256]
        rows[key] = (w[idx].cpu(), x[:half], x[half:])
    out = []
    for bits in (2, 4):
        for g, b in (
            (16, 16),
            (16, 32),
            (16, 64),
            (128, 64),
            (128, 128),
            (128, 256),
            (256, 256),
        ):
            errs = []
            for key, (w, x1, x2) in rows.items():
                k = w.shape[1]
                if k % b or k % g:
                    continue
                d = (x1**2).mean(0).clamp_min(1e-8)
                s = (d / d.log().mean().exp()).pow(0.25)
                one = torch.ones(k)
                wt = rs.transform(w, s, one, g, False)
                x1t = rs.transform(x1, s, one, g, True)
                x2t = rs.transform(x2, s, one, g, True)
                dt = (x1t**2).mean(0)
                wb = wt.reshape(w.shape[0], k // b, b)
                db = dt.reshape(1, k // b, b).expand(w.shape[0], -1, -1)
                rec = ss.encode(wb, db, 2**bits, "affine", "mse_w", 1).reshape(w.shape)
                errs.append(ss.rel(x2t @ rec.T, x2 @ w.T))
            k0 = next(iter(rows.values()))[0].shape[1]
            row = dict(
                bits=bits,
                G=g,
                B=b,
                n=len(errs),
                out_err=sum(errs) / max(len(errs), 1),
                bpw=bits + 16 / b + 48 / k0,
            )
            out.append(row)
            print(row, flush=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "block_rotation_study.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
