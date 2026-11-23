# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Weight-only (data-free) HI2 allocation dry run on sampled real Inkling-2-Mini weights.

HI2 already has a weight-only mode: ``build_table(name, w, stats=None, ...)`` scores every
candidate ``(nbits, block)`` of every row against identity column weights (no activation
statistics, no Hessian), which is what ``hi2 quantize`` does without ``--calib``. This
script runs that mode, unmodified, on real dequantized experts and attention linears, then
the repo's own allocators on the result:

* ``allocate_global``: every row picks the candidate that minimises ``cost + lam * bits``,
  ``lam`` is bisected to a bits-per-weight target (``--targets``);
* ``allocate_by_quality``: the fewest bits within ``--ratios`` times the best achievable
  error (the default flow of ``hi2 quantize``, INT2/INT4 only).

Population. A handful of layers and experts stand for the 24 x 128. Each sampled tensor
is tiled (``sse`` repeated, ``energy`` scaled) by the number of real tensors it stands
for, so the bit budget is that of the whole population, and its sensitivity is
``layers represented x hi2.moe.expert_weights`` (a group of experts weighs as one dense
projection). Each replicated row then has exactly the per-row cost and bits of the real
problem, and the allocators run unmodified. Layers are stratified by expert format
(NVFP4 layers 0-19, MXFP8 layers 20-23); the attention linears of a sampled layer stand
for the layers of its class.

This is a first-order estimate. The costs are weight-space errors in the rotated basis
with identity column weights: no activations, no Hessian, no routing statistics, no KL.
The split that matters needs calibration forward passes through the model.

    python experiments/inkling_alloc_dryrun.py CKPT --out alloc_dryrun.json \\
        --inventory inventory.json
"""

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import torch

from hi2.allocate import (
    allocate_by_quality,
    allocate_global,
    restrict,
    row_overhead_bits,
)
from hi2.calibrate import (
    SELECTABLE,
    Config,
    TensorTable,
    build_table,
    candidate_row_bits,
)
from hi2.executor import Executor
from hi2.inkling_source import PROJECTIONS, InklingSource, expert_name, tensor_role
from hi2.moe import expert_weights

GIB = 2**30


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- sampled tables


def attn_names(src: InklingSource, layer: int) -> dict[str, str]:
    """HF name -> short name (``wq_du`` ...) of the attention linears of ``layer``."""
    serving = {v: k for k, v in src.hf_names().items()}
    out = {}
    for hf, s in serving.items():
        if f".layers.{layer}." in s and tensor_role(s) == "attn_linear":
            out[hf] = s.split(".")[-2]
    return dict(sorted(out.items()))


def build_tables(src, layers, n_experts, with_attn, cfg, ex, cache_dir):
    """``({name: TensorTable}, {name: {layer, kind, proj}}, expert ids)``; tables are
    cached on disk (they are a few KB each), the dequantized weights are not."""
    tag = (
        f"{'-'.join(f'{a}x{b}' for a, b in cfg.candidates)}_{cfg.effort}_{cfg.rotation}"
    )
    cache = cache_dir / tag
    cache.mkdir(parents=True, exist_ok=True)
    ids = src.sample_experts(n_experts)
    tables: dict[str, TensorTable] = {}
    meta: dict[str, dict] = {}

    def cached(name):
        p = cache / f"{name}.pt"
        return torch.load(p, weights_only=False) if p.exists() else None

    def keep(name, t):
        torch.save(t, cache / f"{name}.pt")

    for layer in layers:
        t0 = time.time()
        for a in range(0, len(ids), 4):
            part = ids[a : a + 4]
            want = [(e, p, expert_name(layer, e, p)) for e in part for p in PROJECTIONS]
            have = {name: cached(name) for _, _, name in want}
            if any(t is None for t in have.values()):
                gate, up = src.gate_up(layer, part)
                down = src.down(layer, part)
                ws = {(e, "gate_proj"): gate[j] for j, e in enumerate(part)} | {
                    (e, "up_proj"): up[j] for j, e in enumerate(part)
                }
                ws |= {(e, "down_proj"): down[j] for j, e in enumerate(part)}
            for e, proj, name in want:
                t = have[name]
                if t is None:
                    t = build_table(name, ws[(e, proj)], None, cfg, ex)
                    keep(name, t)
                tables[name] = t
                meta[name] = {"layer": layer, "kind": "expert", "proj": proj}
        if with_attn:
            for hf, short in attn_names(src, layer).items():
                t = cached(hf)
                if t is None:
                    t = build_table(hf, src.load(hf, torch.float32), None, cfg, ex)
                    keep(hf, t)
                tables[hf] = t
                meta[hf] = {"layer": layer, "kind": "attn", "proj": short}
        log(
            f"layer {layer} ({src.expert_format(layer)}): {len(tables)} tables so far "
            f"[{time.time() - t0:.0f}s]"
        )
    return tables, meta, ids


# ---------------------------------------------------------------- population


def multipliers(src: InklingSource, layers, n_sampled: int):
    """``(layers represented per sampled layer, real experts per sampled expert)``."""
    by_fmt: dict[str, list[int]] = {}
    for layer in range(src.num_layers):
        by_fmt.setdefault(src.expert_format(layer), []).append(layer)
    lm = {}
    for layer in layers:
        fmt = src.expert_format(layer)
        sampled = [x for x in layers if src.expert_format(x) == fmt]
        q = len(by_fmt[fmt]) / len(sampled)
        if q != int(q):
            raise SystemExit(
                f"{len(by_fmt[fmt])} {fmt} layers cannot be tiled by {len(sampled)} sampled layers"
            )
        lm[layer] = int(q)
    if src.n_experts % n_sampled:
        raise SystemExit(
            f"{src.n_experts} experts cannot be tiled by {n_sampled} sampled"
        )
    return lm, src.n_experts // n_sampled


def tile(t: TensorTable, m: int) -> TensorTable:
    """``m`` copies of the rows of ``t`` as one table (same per-row cost and bits)."""
    if m == 1:
        return t
    return dataclasses.replace(
        t,
        shape=(t.shape[0] * m, t.shape[1]),
        sse=t.sse.repeat(1, m),
        energy=t.energy * m,
    )


class Agg:
    """Population-weighted totals of a set of tensors."""

    def __init__(self):
        self.weights = 0
        self.bits = 0.0
        self.sse = 0.0
        self.energy = 0.0
        self.by_nbits: dict[int, float] = {}
        self.by_cfg: dict[str, float] = {}

    def add(self, t: TensorTable, picks: torch.Tensor, mult: int) -> None:
        r, k = t.shape
        counts = torch.bincount(picks, minlength=len(t.candidates)).tolist()
        ov = row_overhead_bits(t)
        bits = ov * r + k // 16
        for (nb, b), c in zip(t.candidates, counts):
            bits += c * candidate_row_bits(nb, b, k)
            self.by_nbits[nb] = self.by_nbits.get(nb, 0) + mult * c * k
            self.by_cfg[f"INT{nb}/B{b}"] = (
                self.by_cfg.get(f"INT{nb}/B{b}", 0) + mult * c * k
            )
        self.weights += mult * r * k
        self.bits += mult * bits
        self.sse += mult * float(t.sse.gather(0, picks[None])[0].sum())
        self.energy += mult * t.energy

    def result(self) -> dict:
        w = max(self.weights, 1)
        return {
            "weights": self.weights,
            "bytes": self.bits / 8,
            "gib": self.bits / 8 / GIB,
            "bpw": self.bits / w,
            "nmse": self.sse / max(self.energy, 1e-30),
            "share_by_nbits": {
                f"INT{k}": v / w for k, v in sorted(self.by_nbits.items())
            },
            "share_by_config": {k: v / w for k, v in sorted(self.by_cfg.items())},
        }


def describe(alloc, views, meta, mult, fmt_of) -> dict:
    """Aggregates of one allocation by class, layer and projection."""
    groups: dict[str, Agg] = {}

    def add(key, t, picks, m):
        groups.setdefault(key, Agg()).add(t, picks, m)

    for name, t in views.items():
        md = meta[name]
        picks = alloc.picks[name][: t.shape[0]]  # the first copy of the rows
        fmt = fmt_of[md["layer"]]
        kind = "experts" if md["kind"] == "expert" else "attention"
        for key in (
            "all",
            f"{kind}",
            f"{kind}.{fmt}",
            f"{kind}.{fmt}.{md['proj']}",
            f"L{md['layer']:02d}.{kind}",
        ):
            add(key, t, picks, mult[name])
    return {
        "allocator_bpw": alloc.bpw,
        "lambda": alloc.lam,
        "feasible": alloc.feasible,
        "allocator_error_ratio": alloc.error_ratio,
        "allocator_row_share_by_nbits": {
            f"INT{k}": v for k, v in (alloc.frac_by_bits or {}).items()
        },
        "groups": {k: v.result() for k, v in sorted(groups.items())},
    }


def uniform_baselines(views, meta, mult, fmt_of) -> dict:
    """Every candidate applied to every row: its bpw and weight NMSE per class."""
    out: dict[str, dict] = {}
    for name, t in views.items():
        md = meta[name]
        kind = "experts" if md["kind"] == "expert" else "attention"
        key = f"{kind}.{fmt_of[md['layer']]}"
        a = out.setdefault(key, {})
        r, k = t.shape
        for ci, (nb, b) in enumerate(t.candidates):
            row = a.setdefault(f"INT{nb}/B{b}", [0.0, 0.0, 0.0, 0])
            bits = row_overhead_bits(t) * r + k // 16 + r * candidate_row_bits(nb, b, k)
            row[0] += mult[name] * bits
            row[1] += mult[name] * float(t.sse[ci].sum())
            row[2] += mult[name] * t.energy
            row[3] += mult[name] * r * k
    return {
        key: {c: {"bpw": v[0] / v[3], "nmse": v[1] / v[2]} for c, v in cs.items()}
        for key, cs in out.items()
    }


def implied_sizes(res: dict, inv: dict | None, with_attn: bool) -> dict | None:
    """Totals of the checkpoint if the experts (and the attention linears if allocated)
    take the allocation's bytes and everything else stays as stored (``inventory.json``)."""
    if inv is None:
        return None
    t = inv["totals"]
    attn_stock = next(c["bytes"] for c in inv["classes"] if c["class"] == "attn_linear")
    experts = res["groups"]["experts"]["bytes"]
    attn = res["groups"]["attention"]["bytes"] if with_attn else attn_stock
    main = experts + attn + t["bytes_non_expert_main_file"] - attn_stock
    return {
        "experts_gib": experts / GIB,
        "attention_gib": attn / GIB,
        "main_file_gib": main / GIB,
        "main_file_text_only_gib": (main - t["bytes_vision"]) / GIB,
        "review_definition_gib": (main + t["bytes_mtp_file"] - t["bytes_markov_w1"])
        / GIB,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--out", type=Path, default=Path("inkling_alloc_dryrun.json"))
    ap.add_argument(
        "--inventory", type=Path, help="inventory.json of inkling_inventory.py"
    )
    ap.add_argument("--layers", default="0,6,12,17,20,23")
    ap.add_argument("--experts", type=int, default=8, help="sampled experts per layer")
    ap.add_argument("--targets", default="3.0,3.5,4.0", help="bpw targets")
    ap.add_argument("--ratios", default="4,6", help="max error ratios (INT2/INT4 only)")
    ap.add_argument("--bits", default="2,4,8")
    ap.add_argument("--effort", default="normal", choices=("fast", "normal", "max"))
    ap.add_argument("--rotation", default="on", choices=("on", "auto", "off"))
    ap.add_argument(
        "--variants",
        default="experts,experts+attention",
        help="what is allocated: the experts alone (the attention linears stay bf16) "
        "and/or the experts together with the attention linears",
    )
    ap.add_argument(
        "--build-only", action="store_true", help="tabulate and cache, stop"
    )
    ap.add_argument("--cache", type=Path, default=Path("alloc_tables"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    layers = [int(x) for x in a.layers.split(",")]
    bits = {int(x) for x in a.bits.split(",")}
    cfg = Config(
        candidates=[c for c in SELECTABLE if c[0] in bits],
        rotation=a.rotation,
        effort=a.effort,
        rounding="nearest",
    )
    wanted = a.variants.split(",")
    inv = json.loads(a.inventory.read_text()) if a.inventory else None
    ex = Executor(a.device)
    t_start = time.time()
    with InklingSource(a.ckpt) as src:
        tables, meta, ids = build_tables(
            src, layers, a.experts, "experts+attention" in wanted, cfg, ex, a.cache
        )
        if a.build_only:
            log(f"built {len(tables)} tables in {time.time() - t_start:.0f}s")
            return 0
        lm, e_mult = multipliers(src, layers, len(ids))
        fmt_of = {layer: src.expert_format(layer) for layer in range(src.num_layers)}
    log(f"tables ready: {len(tables)} tensors, {time.time() - t_start:.0f}s")
    mult = {
        n: lm[m["layer"]] * (e_mult if m["kind"] == "expert" else 1)
        for n, m in meta.items()
    }
    result = {
        "checkpoint": str(a.ckpt),
        "caveat": (
            "first-order, weight-only: identity column weights in the rotated basis, no "
            "activations, no Hessian, no routing statistics, no KL"
        ),
        "sampled_layers": layers,
        "sampled_experts": ids,
        "layers_represented": lm,
        "experts_represented_per_sample": e_mult,
        "candidates": cfg.candidates,
        "effort": a.effort,
        "rotation": a.rotation,
        "tensors": len(tables),
        "runs": [],
    }
    variants = [
        (v, sel)
        for v, sel in (
            ("experts", lambda m: m["kind"] == "expert"),
            ("experts+attention", lambda m: True),
        )
        if v in wanted
    ]
    sets = [("INT2/4/8", None)] if 8 in bits else []
    sets.append(("INT2/4", lambda c: c[0] != 8))
    targets = [float(x) for x in a.targets.split(",") if x]
    ratios = [float(x) for x in a.ratios.split(",") if x]
    for vname, vsel in variants:
        names = [n for n in tables if vsel(meta[n])]
        w_exp = expert_weights(
            {n: tables[n] for n in names if meta[n]["kind"] == "expert"}
        )
        sens = {n: lm[meta[n]["layer"]] * w_exp.get(n, 1.0) for n in names}
        for sname, keep in sets:
            views = {
                n: (restrict(tables[n], keep) if keep else tables[n]) for n in names
            }
            tiled = [tile(t, mult[n]) for n, t in views.items()]
            base = uniform_baselines(views, meta, mult, fmt_of)
            runs = [("bpw", x) for x in targets]
            if keep is not None and vname == "experts":  # ratio: vs the best candidate
                runs += [("ratio", x) for x in ratios]
            for mode, x in runs:
                t0 = time.time()
                alloc = (
                    allocate_global(tiled, x, sens)
                    if mode == "bpw"
                    else allocate_by_quality(tiled, x, sens)
                )
                rec = describe(alloc, views, meta, mult, fmt_of)
                rec |= {
                    "variant": vname,
                    "candidate_set": sname,
                    "mode": mode,
                    "value": x,
                    "uniform_baselines": base,
                    "implied_sizes": implied_sizes(rec, inv, vname != "experts"),
                    "seconds": time.time() - t0,
                }
                result["runs"].append(rec)
                g = rec["groups"]
                log(
                    f"{vname:<18} {sname:<9} {mode} {x:g}: allocator bpw {alloc.bpw:.4f}, "
                    f"experts {g['experts']['bpw']:.4f} bpw "
                    f"{g['experts']['share_by_nbits']} "
                    f"{g['experts']['gib']:.3f} GiB [{rec['seconds']:.0f}s]"
                )
                a.out.parent.mkdir(parents=True, exist_ok=True)
                a.out.write_text(json.dumps(result, indent=1) + "\n")
    log(f"wrote {a.out}; total {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
