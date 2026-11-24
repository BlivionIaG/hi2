# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Is a bf16 class of Inkling-2-Mini free to store as INT8? (exploratory, eval only)

The stock model keeps three classes of tensors in bf16 that ``hi2`` leaves alone: the attention
projections (``q k v o`` and the output gate, the relative-position bias; about 645 M weights),
the embedding and the LM head (about 463 M and 461 M). ``apply`` replaces ONE class in the model
that ``experiments/inkling_kl_eval.py --int8-class`` scores by what HI2's INT8 row candidate
makes of it: every row ``(8, B)``, encoded data-free by ``hi2.calibrate`` (nearest rounding,
the format's own 8-bit ladder per block of ``B`` weights, rows rotated like every HI2 tensor)
and decoded by ``hi2.format.dequantize``, the weights a runtime would read back. Nothing is
stored; no runtime is involved. The data-free encode is the upper bound of what the format
loses at 8 bits: an activation-aware encode (GPTQ) only does better.

``report`` compares the scores of such models with the same model with the class in bf16, per
class and held-out set, window by window (the same windows, the same teacher): the paired
difference of the KL with its standard error over windows, the top-1 agreement, the perplexity
ratio and the bytes the class saves. A class passes on a set when the KL increase is within
the paired noise (``--z-pass`` standard errors, default 1) and the top-1 agreement moved by no
more than ``--top1-tol`` (default 1e-4); it passes when it passes on every set. The same models
on top of the stock experts (``--stock-student``) isolate what the INT8 class costs alone.

    python experiments/inkling_int8_probe.py report --eval stage4/eval --student bpw3.5 \\
        --classes attn embed lm_head --block 64 --json int8_classes.json
"""

import argparse
import copy
import json
import math
import re
import sys
from pathlib import Path

import torch
from torch import nn

CLASSES = ("attn", "embed", "lm_head")
SETS = ("mixed", "wiki", "pydocs")
#: the attention projections of every decoder layer, as ``hi2`` would see them: ``q k v``, the
#: output projection and its gate, the relative-position bias. Not the sink projection or the
#: short convolutions (a few thousand weights each, they stay as they are).
ATTENTION = re.compile(
    r"(?:^|\.)layers\.\d+\.self_attn\."
    r"(?:(?:q|k|v)_proj\.linear|o_proj\.(?:linear|gate_proj)|relative_position_bias\.linear)$"
)
CHUNK_ROWS = (
    16384  # rows encoded at a time: the rows of a tensor never depend on each other
)


def class_weights(model: nn.Module, cls: str) -> dict[str, nn.Parameter]:
    """``{name: weight}`` of every tensor of the class (never empty)."""
    if cls == "attn":
        found = {
            n: m.weight
            for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and ATTENTION.search(n)
        }
    elif cls == "embed":
        found = {"embed_tokens": model.get_input_embeddings().weight}
    elif cls == "lm_head":
        found = {"lm_head": model.get_output_embeddings().weight}
    else:
        raise SystemExit(f"unknown class {cls!r}; choose from {CLASSES}")
    if not found:
        raise SystemExit(f"the model has no tensor of the class {cls!r}")
    return found


def int8_bits(rows: int, k: int, block: int) -> float:
    """The bits of ``rows`` rows of ``k`` weights stored as ``(8, block)``: the codes, the
    two 8-bit parameters of every block, 48 bits of constants per row and the one bit of
    metadata of a tensor whose rotation flags all agree (``hi2.format.tensor_bpw``)."""
    return rows * (8 * k + 16 * k / block + 48) + 1


def int8_roundtrip(w: torch.Tensor, block: int, ex) -> torch.Tensor:
    """What HI2's ``(8, block)`` candidate makes of ``w`` ``[R, K]``: its decode in float32
    (same device as ``w``). Rows are encoded in chunks, which changes nothing: no row of an
    encode depends on another."""
    from hi2.calibrate import Config, build_table, encode_with_picks
    from hi2.format import dequantize

    cfg = Config(candidates=[(8, block)], rounding="nearest", smooth=0.0)
    out = torch.zeros(w.shape, dtype=torch.float32)
    for s in range(0, w.shape[0], CHUNK_ROWS):
        part = w[s : s + CHUNK_ROWS].float()
        table = build_table("int8", part, None, cfg, ex)
        picks = torch.zeros(part.shape[0], dtype=torch.int64)
        t = encode_with_picks(part, None, table, picks, cfg, ex)
        out[s : s + CHUNK_ROWS] = dequantize(t).float()
    return out.to(w.device)


@torch.no_grad()
def apply(model: nn.Module, cls: str, block: int, ex, log=print, lm=None):
    """Store ONE class as INT8 and read it back: the weights of ``cls`` in ``model`` become
    their decode (attention, embedding), or, for the LM head, a copy of ``lm`` (the model's
    logits function, ``hi2.finalkl._logits_fn``) whose head is the decode while the model
    keeps its own. Returns ``(info, student logits function or None)``; ``info`` has the
    tensors, weights, bytes in bf16 and as INT8, and how far the decode is from the weights."""
    params = class_weights(model, cls)
    if cls == "lm_head" and lm is None:
        raise SystemExit("the LM head class needs the logits function (lm=...)")
    info = {
        "class": cls,
        "scheme": f"HI2 INT8 row candidate (8, {block}), data-free nearest rounding",
        "block": block,
        "tensors": len(params),
        "elements": 0,
        "bytes_bf16": 0,
        "bytes_int8": 0.0,
    }
    sq_err = sq_w = max_err = 0.0
    lm_student = copy.deepcopy(lm) if cls == "lm_head" else None
    for p in params.values():
        w = p.detach()
        dq = int8_roundtrip(w, block, ex)
        err = (dq - w.float()).double()
        sq_err += float(err.pow(2).sum())
        sq_w += float(w.double().pow(2).sum())
        max_err = max(max_err, float(err.abs().max()))
        r, k = w.shape
        info["elements"] += r * k
        info["bytes_bf16"] += 2 * r * k
        info["bytes_int8"] += int8_bits(r, k, block) / 8
        if cls == "lm_head":
            head = getattr(lm_student, "head", lm_student)
            head.weight.data = dq.to(device=head.weight.device, dtype=head.weight.dtype)
        else:
            p.data = dq.to(p.dtype)
    info["bytes_saved"] = info["bytes_bf16"] - info["bytes_int8"]
    info["bits_per_weight"] = 8 * info["bytes_int8"] / info["elements"]
    info["weight_rms_rel_error"] = math.sqrt(sq_err / sq_w)
    info["weight_max_abs_error"] = max_err
    log(
        f"INT8 class {cls}: {info['tensors']} tensors, {info['elements'] / 1e6:.1f} M weights, "
        f"{info['bytes_bf16'] / 2**30:.3f} -> {info['bytes_int8'] / 2**30:.3f} GiB "
        f"({info['bits_per_weight']:.3f} bits per weight), rms error "
        f"{info['weight_rms_rel_error']:.3g} of the weights"
    )
    return info, lm_student


# ------------------------------------------------------------------------------ the report


def _paired():
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    from inkling_target_table import paired

    return paired


def compare(base: dict, var: dict, z_pass: float = 1.0, top1_tol: float = 1e-4) -> dict:
    """One class on one set against the same model with the class in bf16: both are eval
    results of ``inkling_kl_eval.py`` (the same windows). ``increase`` is the paired
    ``KL(variant) - KL(bf16)`` over windows with its standard error; the class passes when
    the increase is at most ``z_pass`` standard errors and the top-1 agreement moved by no
    more than ``top1_tol``."""
    b, v = base["student"], var["student"]
    if len(b["kl_per_window"]) != len(v["kl_per_window"]):
        raise SystemExit("the two evals scored different numbers of windows")
    d = _paired()(v["kl_per_window"], b["kl_per_window"])
    sem = d["sem"]
    z = None if not sem else d["mean"] / sem
    top1 = v["top1_agreement"] - b["top1_agreement"]
    kl_ok = d["mean"] <= (z_pass * sem if sem is not None else 0.0)
    top1_ok = abs(top1) <= top1_tol
    return {
        "windows": d["windows"],
        "kl_bf16": b["kl_mean"],
        "kl_int8": v["kl_mean"],
        "kl_int8_sem": v["kl_sem"],
        "increase": d["mean"],
        "increase_sem": sem,
        "z": z,
        "windows_worse": sum(
            x > y for x, y in zip(v["kl_per_window"], b["kl_per_window"])
        ),
        "top1_bf16": b["top1_agreement"],
        "top1_int8": v["top1_agreement"],
        "top1_change": top1,
        "ppl_bf16": b["ppl_ratio"],
        "ppl_int8": v["ppl_ratio"],
        "kl_ok": bool(kl_ok),
        "top1_ok": bool(top1_ok),
        "pass": bool(kl_ok and top1_ok),
    }


def alone(var: dict, ptq: dict) -> dict:
    """The class on top of the stock experts (the student is the teacher but for the class):
    its KL, top-1 and perplexity ratio, and the KL as a share of the PTQ student's."""
    v, p = var["student"], ptq["student"]
    return {
        "kl": v["kl_mean"],
        "kl_sem": v["kl_sem"],
        "top1": v["top1_agreement"],
        "ppl": v["ppl_ratio"],
        "share_of_ptq_kl": v["kl_mean"] / p["kl_mean"],
    }


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def build_report(
    eval_dir: Path,
    student: str,
    classes,
    block: int,
    sets=SETS,
    stock: bool = True,
    z_pass: float = 1.0,
    top1_tol: float = 1e-4,
) -> dict:
    out: dict = {
        "student": student,
        "block": block,
        "z_pass": z_pass,
        "top1_tol": top1_tol,
    }
    for cls in classes:
        entry: dict = {"sets": {}}
        for s in sets:
            base = _load(eval_dir / f"{student}_{s}.json")
            var = _load(eval_dir / f"{student}-int8{cls}{block}_{s}.json")
            entry["sets"][s] = compare(base, var, z_pass, top1_tol)
            entry["info"] = var["int8"]
            if stock and (eval_dir / f"stock-int8{cls}{block}_{s}.json").exists():
                st = _load(eval_dir / f"stock-int8{cls}{block}_{s}.json")
                entry["sets"][s]["alone"] = alone(st, base)
        entry["pass"] = all(x["pass"] for x in entry["sets"].values())
        out[cls] = entry
    return out


def render(rep: dict) -> str:
    L = [
        f"INT8 classes on {rep['student']} (HI2 INT8 candidate (8, {rep['block']}), data-free; "
        f"exploratory). Pass: KL increase <= {rep['z_pass']:g} paired standard errors and "
        f"|top-1 change| <= {rep['top1_tol']:g} on every set.",
        "",
        "| class | GiB bf16 -> INT8 (saved) | set | KL bf16 | KL INT8 | increase +- sem (z) "
        "| worse windows | top-1 bf16 -> INT8 | ppl bf16 -> INT8 | alone KL (share of PTQ) | pass |",
        "| --- | --- | --- | ---: | ---: | --- | ---: | --- | --- | --- | --- |",
    ]
    for cls in CLASSES:
        if cls not in rep:
            continue
        e = rep[cls]
        i = e["info"]
        size = (
            f"{i['bytes_bf16'] / 2**30:.3f} -> {i['bytes_int8'] / 2**30:.3f} "
            f"({i['bytes_saved'] / 2**30:.3f})"
        )
        for s, c in e["sets"].items():
            z = "n/a" if c["z"] is None else f"{c['z']:+.2f}"
            sem = "n/a" if c["increase_sem"] is None else f"{c['increase_sem']:.2g}"
            solo = c.get("alone")
            solo_s = (
                "-"
                if solo is None
                else f"{solo['kl']:.3g} ({100 * solo['share_of_ptq_kl']:.1f} %)"
            )
            L.append(
                f"| {cls} | {size} | {s} | {c['kl_bf16']:.5f} | {c['kl_int8']:.5f} "
                f"| {c['increase']:+.2e} +- {sem} ({z}) | {c['windows_worse']}/{c['windows']} "
                f"| {c['top1_bf16']:.4f} -> {c['top1_int8']:.4f} "
                f"| x{c['ppl_bf16']:.4f} -> x{c['ppl_int8']:.4f} | {solo_s} "
                f"| {'yes' if c['pass'] else 'NO'} |"
            )
        L.append(
            f"| **{cls}** | | | | | | | | | | **{'PASS' if e['pass'] else 'stays bf16'}** |"
        )
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="compare the INT8 class evals with the bf16 ones")
    r.add_argument(
        "--eval", type=Path, required=True, help="directory of the eval JSONs"
    )
    r.add_argument(
        "--student", default="bpw3.5", help="name of the PTQ student's evals"
    )
    r.add_argument("--classes", nargs="+", choices=CLASSES, default=list(CLASSES))
    r.add_argument("--block", type=int, choices=(32, 64), default=64)
    r.add_argument("--sets", nargs="+", default=list(SETS))
    r.add_argument("--z-pass", type=float, default=1.0)
    r.add_argument("--top1-tol", type=float, default=1e-4)
    r.add_argument(
        "--no-stock", action="store_true", help="skip the stock-student evals"
    )
    r.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)
    rep = build_report(
        a.eval,
        a.student,
        a.classes,
        a.block,
        a.sets,
        not a.no_stock,
        a.z_pass,
        a.top1_tol,
    )
    print(render(rep))
    if a.json:
        a.json.write_text(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
