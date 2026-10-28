# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Does the QAD still learn the same in float16?

The QAD step of a 27B is float32-compute-bound with the matrix units of the GPU idle
(docs/qad.md, "The 27B at four GPUs"). With ``--qad-dtype float16`` the layer's matrix
products run in float16 (the latents, the ladder bytes and the loss stay float32), so the
matrix path of the GPU can be used. That changes the numerics, so it has to be measured
before a 27B run uses it.

This script takes one real decoder layer of a real checkpoint and runs it through the
production training loop (``hi2.qad.train_layer``) in every dtype asked for, and prints

* the time of a step,
* the held-out nMSE of the layer after every evaluation (the signal),
* the relative difference of the latent gradient against the float32 run (the noise the
  dtype adds to the update).

Note: bfloat16 is not an option on gfx1030 - the matrix path of RDNA2 has no
``fdot2.bf16`` instruction and the compiler stops with "Cannot select: intrinsic
%llvm.amdgcn.fdot2.bf16.bf16".

usage:
    python qad_dtype_check.py --model Qwen3.8-27B --ckpt runs27b/q27b-ptq-b3.0 --layer 10
"""

import argparse
import copy
import time
from pathlib import Path

import torch
from torch.func import functional_call

from hi2.checkpoint import make_resolver, weight_map
from hi2.layerwise import capture_inputs, find_layers, to_device
from hi2.qad import FakeQuantLinear, QadConfig, train_layer
from hi2.stack import Hi2Checkpoint


def say(*a):
    print(*a, flush=True)


def build_fq(model: str, ckpt: str, layer, path: str, index: int, dev):
    """The fake-quantised Linears of one layer, from an HI2 checkpoint.

    The checkpoint names its tensors after the model it was made from, which may nest
    the text model (``model.language_model.layers.N``) where the model here does not
    (``model.layers.N``) or the other way round, so several spellings are tried."""
    ck = Hi2Checkpoint(Path(ckpt))
    known = ck.logical()
    resolve = make_resolver(weight_map(Path(model)))
    fq, names = {}, []
    for name, mod in layer.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        cands = []
        r = resolve(f"{path}.{index}.{name}.weight")
        if r:
            cands.append(r.removesuffix(".weight"))
        cands += [
            f"{path}.{index}.{name}",
            f"model.layers.{index}.{name}",
            f"model.language_model.layers.{index}.{name}",
        ]
        base = next((c for c in cands if c in known), None)
        if base is None:
            continue
        fq[name] = FakeQuantLinear(
            mod.weight.detach().float(), ck.get(base), None, "weights", "step"
        )
        names.append(name)
    return fq, names


def latent_grads(layer, fq, xs, kw, dev, dtype):
    """One forward and backward in ``dtype``; the gradients of the latents."""
    lay = copy.deepcopy(layer).to(dev, dtype=dtype)
    for p in lay.parameters():
        p.requires_grad_(False)
    for m in fq.values():
        m.to(dev)
    w = {f"{n}.weight": m() for n, m in fq.items()}
    wd = {k: v.detach().to(dtype).detach().requires_grad_(True) for k, v in w.items()}
    for x in xs:
        out = functional_call(lay, wd, (x.to(dev, dtype=dtype),), kw)
        out = out[0] if isinstance(out, tuple) else out
        ((out.float() - x.float()) ** 2).mean().backward()
    torch.autograd.backward(
        list(w.values()), [wd[k].grad.float() for k in w if wd[k].grad is not None]
    )
    grads = {
        f"{n}.{i}": (p.grad.detach().clone() if p.grad is not None else None)
        for n, m in fq.items()
        for i, p in enumerate(m.param_groups()[0])
    }
    return grads


def rel_diff(a, b):
    """The relative difference of two gradient dictionaries."""
    num = den = 0.0
    for k, ga in a.items():
        gb = b.get(k)
        if ga is None or gb is None:
            continue
        num += ((ga - gb) ** 2).sum().item()
        den += (ga**2).sum().item()
    return (num / den) ** 0.5 if den else float("nan")


def grad_norm(g):
    """The norm of a gradient dictionary and how many of its entries are not finite."""
    total = 0.0
    bad = 0
    for ga in g.values():
        if ga is None:
            continue
        total += (ga**2).sum().item()
        bad += int((~torch.isfinite(ga)).sum().item())
    return total**0.5, bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=4, help="sequences per optimizer step")
    ap.add_argument("--val", type=int, default=2, help="held-out sequences")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--dtypes", default="float32,float16")
    a = ap.parse_args()
    dev = torch.device(a.device)

    from transformers import AutoModelForCausalLM

    say("loading the model ...")
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    path, layers = find_layers(model)
    n_seq = a.batch + a.val
    ids = torch.randint(100, 5000, (n_seq, a.seq_len))
    hidden, kwargs = capture_inputs(model, layers, ids)
    kw = to_device(kwargs, dev)

    # the student input of the layer: the output of the (unquantized) prefix, and the
    # teacher target: the output of the unquantized layer itself. The prefix runs on the
    # GPU one layer at a time (a 27B layer does not fit many times over).
    xs = [h.to(dev) for h in hidden]
    with torch.no_grad():
        for lay in layers[: a.layer]:
            lay = lay.to(dev)
            for i, x in enumerate(xs):
                out = lay(x, **kw)
                xs[i] = (out[0] if isinstance(out, tuple) else out).detach()
            lay = lay.cpu()
    layer = layers[a.layer].to(dev)
    with torch.no_grad():
        ys = []
        for x in xs:
            out = layer(x, **kw)
            ys.append((out[0] if isinstance(out, tuple) else out).detach().clone())
    layer = layer.cpu()  # the fake-quantised modules are built on the CPU, like the pipeline

    fq, names = build_fq(a.model, a.ckpt, layer, path, a.layer, dev)
    if not fq:
        raise SystemExit(f"no quantized Linear of layer {a.layer} is in {a.ckpt}")
    for m in fq.values():
        m.to(dev)
    n_w = sum(m.shape[0] * m.shape[1] for m in fq.values())
    say(
        f"layer {a.layer}: {len(fq)} quantized Linears, {n_w / 1e6:.0f} M weights, "
        f"{n_seq} x {a.seq_len} tokens"
    )
    say("names: " + ", ".join(names))

    dtypes = [d.strip() for d in a.dtypes.split(",") if d.strip()]
    # the noise: the gradient of the float32 run against the others
    ref = None
    for d in dtypes:
        fq2 = copy.deepcopy(fq)
        g = latent_grads(layer, fq2, xs, kw, dev, getattr(torch, d))
        norm, bad = grad_norm(g)
        if ref is None:
            ref = g
            say(f"  {d}: gradient norm {norm:.4e}, {bad} non-finite entries")
        else:
            say(
                f"  {d}: gradient norm {norm:.4e}, {bad} non-finite entries, "
                f"difference against float32 {rel_diff(g, ref):.3e}"
            )

    # the signal: the same training loop in every dtype
    infos = {}
    for d in dtypes:
        lines = []
        fq3 = copy.deepcopy(fq)
        qc = QadConfig(
            steps=a.steps,
            batch=a.batch,
            evals=4,
            min_gain=0.0,
            eval_every=max(1, a.steps // 4),
            dtype=d,
            seed=0,
        )
        t0 = time.time()
        info = train_layer(
            layer,
            fq3,
            xs,
            ys,
            list(range(a.batch)),
            list(range(a.batch, n_seq)),
            kwargs,
            qc,
            dev,
            log=lines.append,
            tag=f"{d}",
        )
        wall = time.time() - t0
        infos[d] = info
        say(
            f"{d:>9}: {info.get('steps', a.steps)} steps in {wall:.1f} s "
            f"({wall / max(1, info.get('steps', a.steps)) * 1e3:.0f} ms a step), "
            f"held-out nMSE {info.get('val_before'):.4e} -> {info.get('val_after'):.4e}"
        )
        for line in lines:
            say("   ", line)

    say("")
    say("| dtype | held-out nMSE | vs float32 | ms a step |")
    say("| --- | ---: | ---: | ---: |")
    base = None
    for d in dtypes:
        info = infos[d]
        final = info["val_after"]
        if base is None:
            base = final
        rel = f"{100 * (final - base) / base:+.2f} %"
        ms = 1e3 * info["seconds"] / max(1, info["steps"])
        say(f"| {d} | {final:.4e} | {rel} | {ms:.0f} |")


if __name__ == "__main__":
    main()
