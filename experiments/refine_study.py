# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Local code refinement (spec 8.9): how much does coordinate descent on the codes lower the
objective that GPTQ already minimises?

For real tensors of a model, with the Hessian from a statistics cache entry, GPTQ rounds the
rows at one candidate (bits, block); then up to ``--passes`` sweeps over the columns try to
move each code one level up or down, keeping a move when it lowers ``e Hs e^T`` (``e`` the
error row in the stored basis; the change of one column costs O(rows) given ``g = e Hs``).
The ladders stay fixed. The script prints the objective after each sweep and the share of
codes that moved.

    python refine_study.py --model Qwen3.5-0.8B --stats ~/.cache/hi2/stats/<entry> \\
        --tensors layers.3.mlp.down_proj layers.12.mlp.gate_proj --device cuda:0
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

from hi2.cache import StatsView
from hi2.calibrate import Config, _smoothed, build_table
from hi2.checkpoint import make_resolver, weight_map
from hi2.executor import Executor
from hi2.format import decode_params, fwht_groups
from hi2.hessian import encode_rows_gptq_reference, inverse_factor, transform_sigma
from hi2.stats import ColumnStats


@torch.no_grad()
def refine(codes, lo, step, w, hs, n_levels, block, passes):
    """Coordinate descent on the codes; returns ``(codes, [objective after each sweep])``."""
    r, k = w.shape
    blk = torch.arange(k, device=w.device) // block
    st, lo_c = step[:, blk], lo[:, blk]
    c = codes.clone()

    def value(cc):
        return (lo_c + cc * st).to(torch.float16).float()

    e = value(c) - w
    g = e @ hs
    hd = hs.diagonal()
    history = [float((e * g).sum())]
    zero = torch.zeros(r, device=w.device)
    for _ in range(passes):
        for j in range(k):
            stj, cj = st[:, j], c[:, j]
            cur = value(c)[:, j]
            best_gain, best_dw, best_d = zero.clone(), zero.clone(), zero.clone()
            for d in (1.0, -1.0):
                cn = cj + d
                ok = (cn >= 0) & (cn <= n_levels - 1) & (stj > 0)
                dw = (lo_c[:, j] + cn * stj).to(torch.float16).float() - cur
                gain = torch.where(
                    ok, -(2 * dw * g[:, j] + dw * dw * hd[j]), -torch.inf
                )
                better = gain > best_gain
                best_gain = torch.where(better, gain, best_gain)
                best_dw = torch.where(better, dw, best_dw)
                best_d = torch.where(better, torch.full_like(zero, d), best_d)
            acc = best_gain > 1e-12
            if acc.any():
                dwj = torch.where(acc, best_dw, zero)
                c[:, j] = cj + torch.where(acc, best_d, zero)
                g += dwj[:, None] * hs[j][None, :]
                e[:, j] += dwj
        history.append(float((e * g).sum()))
    return c, history


def original_basis(codes, lo, step, block, had, inv):
    """Dequantized rows of the stored basis, back in the model's own basis."""
    idx = torch.arange(codes.shape[1], device=codes.device) // block
    q = (lo.float()[:, idx] + codes * step.float()[:, idx]).half().float()
    q = fwht_groups(q, had) if had.any() else q
    return q if inv is None else q * inv


class KlProbe:
    """KL(original || model with one tensor replaced) on a few held-out sequences."""

    def __init__(self, model_dir, text, skip, dev, n_seq=8, seq_len=512):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_dir)
        ids = tok(Path(text).read_text()[skip:], return_tensors="pt").input_ids[0]
        ids = ids[: n_seq * seq_len].reshape(n_seq, seq_len).to(dev)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=torch.float32
        )
        self.model = self.model.to(dev).eval()
        self.ids = ids
        self.params = dict(self.model.named_parameters())
        with torch.no_grad():
            self.ref = [
                torch.log_softmax(self.model(c[None]).logits.float(), -1) for c in ids
            ]

    @torch.no_grad()
    def kl(self, name, weight):
        p = self.params[name]
        keep = p.data.clone()
        p.data.copy_(weight.to(p.dtype))
        total = 0.0
        for c, r in zip(self.ids, self.ref):
            lq = torch.log_softmax(self.model(c[None]).logits.float(), -1)
            total += float((r.exp() * (r - lq)).sum())
        p.data.copy_(keep)
        return total / (self.ids.numel())


def run(a) -> list[dict]:
    dev = torch.device(a.device)
    wmap = weight_map(Path(a.model))
    resolve = make_resolver(wmap)
    work = Path(tempfile.mkdtemp(dir=a.scratch))
    (work / "cache_ref.json").write_text(
        json.dumps({"stats": str(Path(a.stats).expanduser())})
    )
    view = StatsView(work)
    ex = Executor([a.device])
    cfg = Config(effort="fast")
    from safetensors import safe_open

    out = []
    probe = KlProbe(a.model, a.kl_text, a.kl_skip, dev) if a.kl_text else None
    for frag in a.tensors:
        name = next(n for n in wmap if frag in n and n.endswith(".weight"))
        with safe_open(Path(a.model) / wmap[name], "pt") as f:
            w = f.get_tensor(name).float()
        ckpt = resolve(name) or name
        sigma = view.sigma(ckpt)
        st = view.stats_state(ckpt)
        stats = ColumnStats.from_state(st, "cpu") if st is not None else None
        table = build_table(ckpt, w, stats, cfg, ex)
        hs = transform_sigma(sigma.to(dev), table.col_scale, table.had)
        wt, _ = _smoothed(
            w.to(dev), None if table.col_scale is None else table.col_scale.to(dev)
        )
        wt = (
            fwht_groups(wt.float(), table.had.to(dev))
            if table.had.any()
            else wt.float()
        )
        u = inverse_factor(hs, 0.01)
        for nb, blk in a.configs:
            enc = encode_rows_gptq_reference(
                wt, hs.diagonal().clone(), u, nb, blk, "normal"
            )
            lo_u8, st_u8 = enc.params[..., 0], enc.params[..., 1]
            lo, step = decode_params(
                lo_u8, st_u8, enc.consts[:, 0:1], enc.consts[:, 1:2], enc.consts[:, 2:3]
            )
            c, hist = refine(
                enc.codes.float(),
                lo.float(),
                step.float(),
                wt,
                hs,
                2**nb,
                blk,
                a.passes,
            )
            moved = float((c != enc.codes.float()).float().mean())
            kls = None
            if probe is not None:
                inv = (
                    None
                    if table.col_scale is None
                    else 1.0 / table.col_scale.to(dev).float()
                )
                had = table.had.to(dev)
                names = [n for n in probe.params if resolve(n) == ckpt or n == name]
                kls = {
                    "gptq": probe.kl(
                        names[0],
                        original_basis(enc.codes.float(), lo, step, blk, had, inv),
                    ),
                    "refined": probe.kl(
                        names[0], original_basis(c, lo, step, blk, had, inv)
                    ),
                }
            row = {
                "tensor": ckpt,
                "shape": list(w.shape),
                "config": f"INT{nb}/B{blk}",
                "objective": hist,
                "gain_pct": [100 * (1 - h / hist[0]) for h in hist[1:]],
                "moved_share": moved,
                "kl": kls,
            }
            out.append(row)
            print(
                f"{ckpt.split('layers.')[-1]:<34}{row['config']:<10}"
                + " ".join(f"{g:6.2f}%" for g in row["gain_pct"])
                + f"   moved {100 * moved:.1f}% of the codes"
                + (
                    ""
                    if kls is None
                    else f"   KL {kls['gptq']:.3e} -> {kls['refined']:.3e} "
                    f"({100 * (1 - kls['refined'] / kls['gptq']):.1f}% lower)"
                ),
                flush=True,
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--stats", required=True, help="a statistics cache entry directory")
    ap.add_argument("--tensors", nargs="+", required=True, help="name fragments")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--scratch", default=".")
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--kl-text", default=None, help="held-out text: also measure the KL cost"
    )
    ap.add_argument("--kl-skip", type=int, default=800_000, help="characters to skip")
    ap.add_argument(
        "--configs",
        nargs="+",
        default=["2:32", "4:64"],
        help="bits:block candidates to round the rows at",
    )
    a = ap.parse_args()
    a.configs = [tuple(int(x) for x in c.split(":")) for c in a.configs]
    rows = run(a)
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
