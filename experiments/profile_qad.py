# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Where a QAD optimizer step spends its time, on the fake-quantised Linears of one real decoder
layer: decoding the weights (forward), the layer forward and backward, the backward through the
decode, and the optimizer step. Random hidden states (timing only); the quantized tensors come
from an existing HI2 checkpoint of the model.

    python profile_qad.py --model Qwen3.5-4B --ckpt out/q4b --layer 10 --device cuda:0
"""

import argparse
import os
import time
from pathlib import Path

import torch
from torch.func import functional_call

from hi2.checkpoint import make_resolver, weight_map
from hi2.layerwise import capture_inputs, find_layers
from hi2.qad import FakeQuantLinear
from hi2.stack import Hi2Checkpoint


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def timed(fn, dev, repeat):
    fn()  # warm-up (kernel builds, allocator)
    sync(dev)
    t = time.time()
    for _ in range(repeat):
        out = fn()
    sync(dev)
    return (time.time() - t) / repeat, out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=4, help="sequences per optimizer step")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument(
        "--profile", action="store_true", help="also print the top device ops"
    )
    a = ap.parse_args()
    dev = torch.device(a.device)

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    path, layers = find_layers(model)
    ids = torch.randint(100, 5000, (a.batch, a.seq_len))
    hidden, kwargs = capture_inputs(model, layers, ids)
    layer = layers[a.layer]
    ck = Hi2Checkpoint(Path(a.ckpt))
    resolve = make_resolver(weight_map(Path(a.model)))
    known = ck.logical()
    fq, names = {}, []
    for name, mod in layer.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        full = resolve(f"{path}.{a.layer}.{name}.weight")
        base = full.removesuffix(".weight") if full else None
        if base not in known:
            continue
        fq[name] = FakeQuantLinear(
            mod.weight.detach().float(), ck.get(base), None, "weights", "step"
        )
        names.append(name)
    layer = layer.to(dev)
    for p in layer.parameters():
        p.requires_grad_(False)
    for m in fq.values():
        m.to(dev)
    from hi2.layerwise import to_device

    kw = to_device(kwargs, dev)
    xs = [h.to(dev).float() for h in hidden]
    n_w = sum(m.shape[0] * m.shape[1] for m in fq.values())
    print(
        f"layer {a.layer}: {len(fq)} quantized Linears, {n_w / 1e6:.0f} M weights, {a.batch} x {a.seq_len} tokens"
    )

    lat = [p for m in fq.values() for p in m.param_groups()[0]]
    prm = [p for m in fq.values() for p in m.param_groups()[1]]
    fused = os.environ.get("PROFILE_FUSED_ADAMW") == "1"
    opt = torch.optim.AdamW(
        [{"params": lat, "lr": 0.01}, {"params": prm, "lr": 0.05}],
        weight_decay=0.0,
        fused=fused,
    )

    def decode():
        return {f"{n}.weight": m() for n, m in fq.items()}

    def decode_nograd():
        with torch.no_grad():
            return {f"{n}.weight": m() for n, m in fq.items()}

    def layer_fwd_bwd(wd):
        wd = {k: v.detach().requires_grad_(True) for k, v in wd.items()}
        for x in xs:
            out = functional_call(layer, wd, (x,), kw)
            out = out[0] if isinstance(out, tuple) else out
            ((out - x) ** 2).mean().backward()
        return wd

    def back_through_decode(w, wd, keep=False):
        torch.autograd.backward(
            list(w.values()), [wd[k].grad for k in w], retain_graph=keep
        )

    def step_all():
        w = decode()
        wd = layer_fwd_bwd(w)
        back_through_decode(w, wd)
        opt.step()
        opt.zero_grad(set_to_none=True)
        for m in fq.values():
            m.project()

    def stored_only():
        return [m.stored(True) for m in fq.values()]

    def stored_back(parts):
        for part in parts:
            part.backward(torch.ones_like(part), retain_graph=True)

    t_sto, parts = timed(stored_only, dev, a.repeat)
    t_sto_b, _ = timed(lambda: stored_back(parts), dev, a.repeat)
    t_dec, _ = timed(decode_nograd, dev, a.repeat)
    t_dec_g, w = timed(decode, dev, a.repeat)
    t_lf, wd = timed(lambda: layer_fwd_bwd(w), dev, a.repeat)
    t_bd, _ = timed(lambda: back_through_decode(w, wd, True), dev, a.repeat)
    t_opt, _ = timed(lambda: opt.step(), dev, a.repeat)
    t_step, _ = timed(step_all, dev, a.repeat)
    print(f"  ladder decode only (stored basis), forward   {t_sto * 1e3:9.1f} ms")
    print(f"  ladder decode only, backward                 {t_sto_b * 1e3:9.1f} ms")
    print(f"decode, no graph        {t_dec * 1e3:9.1f} ms")
    print(f"decode, with STE graph  {t_dec_g * 1e3:9.1f} ms")
    print(f"layer fwd+bwd ({a.batch} seqs)  {t_lf * 1e3:9.1f} ms")
    print(f"backward through decode {t_bd * 1e3:9.1f} ms")
    print(f"optimizer step          {t_opt * 1e3:9.1f} ms")
    print(f"whole step              {t_step * 1e3:9.1f} ms")
    if a.profile:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            step_all()
            sync(dev)
        print(
            prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=18, max_name_column_width=60
            )
        )


if __name__ == "__main__":
    main()
