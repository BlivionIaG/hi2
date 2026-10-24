# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scale-system / encoder / int8-activation study on real weights and activations.

Everything is output-space: relative L2 error of ``X @ W_hat.T`` against
``X @ W.T`` with activations ``X`` captured from the model on real text. Rows
of each matrix are subsampled (row errors are independent, so this is
unbiased). CPU only.

    python experiments/scale_study.py <model_dir> <text_file> <out_dir>
"""

import json
import sys
from pathlib import Path

import torch
from hi2.format import fwht_groups

torch.set_grad_enabled(False)
ROWS = 256
LAYERS = (0, 6, 12, 18, 23)
PROJS = {  # (module path suffix, kind)
    "q_proj": "self_attn.q_proj",
    "o_proj": "self_attn.o_proj",
    "up_proj": "mlp.up_proj",
    "down_proj": "mlp.down_proj",
}


def capture(model_dir: Path, text: str, tokens: int = 4096):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    ids = tok(text, return_tensors="pt").input_ids[:, :tokens]
    acts, weights = {}, {}
    hooks = []
    for li in LAYERS:
        layer = model.model.layers[li]
        for name, path in PROJS.items():
            mod = layer.get_submodule(path)
            weights[(li, name)] = mod.weight.detach().clone()
            hooks.append(
                mod.register_forward_pre_hook(
                    lambda m, a, key=(li, name): acts.__setitem__(
                        key, a[0].detach().reshape(-1, a[0].shape[-1]).clone()
                    )
                )
            )
    model(ids)
    for h in hooks:
        h.remove()
    return weights, acts, ids.shape[1]


# ---- ladders -----------------------------------------------------------------
def _lsq_affine(c, w, d):
    """Weighted LS fit of w ~ lo + c*step per block. c,w,d: [N,nb,B]."""
    sd = d.sum(-1)
    sc = (d * c).sum(-1)
    scc = (d * c * c).sum(-1)
    sw = (d * w).sum(-1)
    scw = (d * c * w).sum(-1)
    det = sd * scc - sc * sc
    ok = det.abs() > 1e-20
    step = torch.where(
        ok,
        (sd * scw - sc * sw) / det.where(ok, torch.ones_like(det)),
        torch.zeros_like(det),
    )
    lo = torch.where(ok, (sw - step * sc) / sd.clamp_min(1e-20), w.mean(-1))
    return lo, step


def quant_params(lo, step, n, mode):
    """8-bit block parameters. ``mode``: 1 (row/lin/lin, the v3.2 draft) or a
    string ``scope/stepkind/lokind``: scope row|tensor, stepkind lin|log,
    lokind lin (affine range in scope) | z (lo = z * step, z in 8 bits)."""
    scope, skind, lkind = ("row", "lin", "lin") if mode == 1 else mode.split("/")
    keep = scope == "row"

    def amax(t):
        return t.abs().amax(-1, keepdim=True) if keep else t.abs().amax().reshape(1, 1)

    smax = amax(step).clamp_min(1e-12)
    if skind == "lin":
        qstep = torch.round(step / smax * 255).clamp(0, 255) / 255 * smax
    else:  # 16 steps per octave over 16 octaves below the maximum; 0 stays 0
        u = torch.round(255 + 16 * torch.log2(step.clamp_min(1e-30) / smax)).clamp(
            1, 255
        )
        qstep = torch.where(
            step == 0, torch.zeros_like(step), smax * torch.exp2((u - 255) / 16)
        )
    if lkind == "lin":
        lmn = lo.amin(-1, keepdim=True) if keep else lo.amin().reshape(1, 1)
        lmx = lo.amax(-1, keepdim=True) if keep else lo.amax().reshape(1, 1)
        span = (lmx - lmn).clamp_min(1e-12)
        qlo = lmn + torch.round((lo - lmn) / span * 255).clamp(0, 255) / 255 * span
    else:
        lim = (n - 1) * 1.0
        z = (lo / qstep.clamp_min(1e-30)).clamp(-2 * lim, lim)
        zq = -2 * lim + torch.round((z + 2 * lim) / (3 * lim) * 255).clamp(
            0, 255
        ) / 255 * (3 * lim)
        qlo = torch.where(qstep == 0, lo, zq * qstep)
    return qlo, qstep


def encode(w, d, n, system, encoder, hier):
    """Quantise blocks ``w[N,nb,B]``; returns reconstruction. ``system``:
    'affine' (lo+step) or 'sym' (step only, levels centred); ``hier`` = 8-bit
    block params relative to a per-row fp16 maximum, else fp16 per block."""
    mn, mx = w.amin(-1), w.amax(-1)
    half = (n - 1) / 2

    def finish(lo, step):
        if hier:
            lo, step = quant_params(lo, step, n, hier)
        lo, step = lo.half().float(), step.half().float()
        if system == "sym":
            lo = -half * step
        safe = torch.where(step == 0, torch.ones_like(step), step)
        c = torch.round((w - lo.unsqueeze(-1)) / safe.unsqueeze(-1)).clamp(0, n - 1)
        return lo.unsqueeze(-1) + c * step.unsqueeze(-1)

    if encoder == "minmax":
        if system == "affine":
            return finish(mn, (mx - mn) / (n - 1))
        return finish(mn, w.abs().amax(-1) / half)
    dw = d if encoder == "mse_w" else torch.ones_like(d)
    best, best_err = None, None
    grid = [0.0, 0.04, 0.08, 0.12, 0.16, 0.20]
    if system == "affine":
        cands = [(a, b) for a in grid for b in grid]
    else:
        cands = [(a, a) for a in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4)]
    for a, b in cands:
        if system == "affine":
            span = mx - mn
            lo, hi = mn + a * span, mx - b * span
            rec = finish(lo, (hi - lo) / (n - 1))
        else:
            rec = finish(mn, w.abs().amax(-1) * (1 - a) / half)
        err = (dw * (rec - w) ** 2).sum(-1)
        if best is None:
            best, best_err = rec, err
        else:
            m = err < best_err
            best = torch.where(m.unsqueeze(-1), rec, best)
            best_err = torch.where(m, err, best_err)
    # two least-squares refinements from the best grid point
    rec = best
    for _ in range(2):
        if system == "affine":
            c = torch.zeros_like(w)
            lo0, st0 = _lsq_affine(
                torch.round(
                    (rec - rec.amin(-1, keepdim=True))
                    / (
                        rec.amax(-1, keepdim=True) - rec.amin(-1, keepdim=True)
                    ).clamp_min(1e-12)
                    * (n - 1)
                ),
                rec,
                torch.ones_like(d),
            )
            step0 = (rec.amax(-1) - rec.amin(-1)) / (n - 1)
            safe = torch.where(step0 == 0, torch.ones_like(step0), step0)
            c = torch.round(
                (w - rec.amin(-1, keepdim=True)) / safe.unsqueeze(-1)
            ).clamp(0, n - 1)
            lo, step = _lsq_affine(c, w, dw)
        else:
            step0 = (rec.amax(-1) - rec.amin(-1)) / (n - 1)
            safe = torch.where(step0 == 0, torch.ones_like(step0), step0)
            u = (
                torch.round(
                    (w - rec.amin(-1, keepdim=True)) / safe.unsqueeze(-1)
                ).clamp(0, n - 1)
                - half
            )
            step = (dw * u * w).sum(-1) / (dw * u * u).sum(-1).clamp_min(1e-20)
            lo = mn
        new = finish(lo, step)
        e_new = (dw * (new - w) ** 2).sum(-1)
        e_old = (dw * (rec - w) ** 2).sum(-1)
        rec = torch.where((e_new < e_old).unsqueeze(-1), new, rec)
    return rec


def scale_bits(system, hier, B, k):
    if hier:
        per = 16 if system == "affine" else 8
        scope = "row" if hier == 1 else hier.split("/")[0]
        return per / B + (48 / k if scope == "row" else 0.0)
    return (32 if system == "affine" else 16) / B


# ---- activations -------------------------------------------------------------
def act_int8(x, group):
    """Symmetric int8 with fp16 scale per ``group`` columns (0 = per token)."""
    t, k = x.shape
    g = k if group == 0 else group
    xb = x.reshape(t, k // g, g)
    s = (xb.abs().amax(-1, keepdim=True) / 127).clamp_min(1e-12).half().float()
    return (torch.round(xb / s).clamp(-127, 127) * s).reshape(t, k)


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def main(model_dir, text_file, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights, acts, ntok = capture(Path(model_dir), Path(text_file).read_text())
    print(f"captured {len(weights)} matrices, {ntok} tokens", flush=True)
    g = torch.Generator().manual_seed(0)
    rows = {}
    for key, w in weights.items():
        x = acts[key]
        half = x.shape[0] // 2
        idx = torch.randperm(w.shape[0], generator=g)[:ROWS]
        rows[key] = (
            w[idx],
            x[:half],
            x[half:],
        )  # calibrate on first half, evaluate on second
    keep = []
    for rot in (0, 1, 2):  # 2 = per-64-column-chunk choice by weighted error
        for bits in (2, 4):
            for B in (16, 32, 64):
                for system, hier in (
                    ("affine", 0),
                    ("sym", 0),
                    ("affine", 1),
                    ("sym", 1),
                ):
                    encs = (
                        ("minmax", "mse", "mse_w")
                        if (system == "affine" and not hier)
                        else ("mse_w",)
                    )
                    for encoder in encs:
                        if rot == 2 and not (system == "affine" and encoder == "mse_w"):
                            continue
                        keep.append((rot, bits, B, system, hier, encoder))
    print(f"{len(keep)} configs x {len(rows)} matrices", flush=True)
    results = []
    cache = {}

    def flags_for(rot, w, x1, bits_):
        k = w.shape[1]
        if rot in (0, 1):
            return torch.full((k // 16,), rot)
        key = id(w)
        if key not in cache:
            errs = []
            for r in (0, 1):
                fl = torch.full((k // 16,), r)
                wt, xt = (fwht_groups(w, fl), fwht_groups(x1, fl)) if r else (w, x1)
                d = (xt**2).mean(0)
                wb = wt.reshape(w.shape[0], k // 64, 64)
                db = d.reshape(1, k // 64, 64).expand(w.shape[0], -1, -1)
                rec = encode(wb, db, 4, "affine", "mse_w", 0)
                errs.append((db * (rec - wb) ** 2).sum((0, 2)))
            cache[key] = (errs[1] < errs[0]).to(torch.uint8).repeat_interleave(4)
        return cache[key]

    for ci, (rot, bits, B, system, hier, encoder) in enumerate(keep):
        errs, werrs, rotfrac = [], [], []
        for key, (w, x1, x2) in rows.items():
            k = w.shape[1]
            flags = flags_for(rot, w, x1, bits)
            rotfrac.append(flags.float().mean().item())
            wt = fwht_groups(w, flags)
            x1t, x2t = fwht_groups(x1, flags), fwht_groups(x2, flags)
            d = (x1t**2).mean(0)
            n = 2**bits
            wb = wt.reshape(w.shape[0], k // B, B)
            db = d.reshape(1, k // B, B).expand(w.shape[0], -1, -1)
            rec = encode(wb, db, n, system, encoder, hier).reshape(w.shape)
            errs.append(rel(x2t @ rec.T, x2t @ wt.T))
            werrs.append(rel(rec, wt))
        bpw = bits + scale_bits(system, hier, B, 896)
        row = dict(
            rot=rot,
            bits=bits,
            B=B,
            system=system,
            hier=hier,
            encoder=encoder,
            bpw=bpw,
            out_err=sum(errs) / len(errs),
            w_err=sum(werrs) / len(werrs),
            rot_frac=sum(rotfrac) / len(rotfrac),
        )
        results.append(row)
        print(f"[{ci + 1}/{len(keep)}] {row}", flush=True)
    # int8 activation study
    act_rows = []
    for rot in (0, 1):
        for group in (0, 64, 32, 16):
            e_act, e_out = [], []
            for key, (w, _x1, x) in rows.items():
                k = w.shape[1]
                flags = torch.full((k // 16,), rot)
                wt = fwht_groups(w, flags) if rot else w
                xt = fwht_groups(x, flags) if rot else x
                xq = act_int8(xt, group)
                e_act.append(rel(xq, xt))
                e_out.append(rel(xq @ wt.T, xt @ wt.T))
            act_rows.append(
                dict(
                    rot=rot,
                    group=group,
                    act_err=sum(e_act) / len(e_act),
                    out_err_fp_weights=sum(e_out) / len(e_out),
                )
            )
            print(act_rows[-1], flush=True)
    (out_dir / "scale_study.json").write_text(
        json.dumps({"weights": results, "act_int8": act_rows}, indent=1)
    )
    print("DONE", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
