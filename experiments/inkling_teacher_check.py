# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Assemble the stock Inkling-2-Mini from ``hi2.inkling_source`` and run it on the CPU.

Builds ``Inkling2LoopForConditionalGeneration`` (the fork's transformers) on the meta
device, allocates it in ``--dtype`` and fills every text-model tensor from the reader: the
dense tensors by ``load``, ``message_input.query`` by ``message_query``, the experts by
``gate_up`` / ``down`` (decoded float32, rounded once to ``--dtype``). The vision tower
stays empty (the prompt is text). Then it runs one prompt and records, per decoder layer
and per call (the loop layers run three passes), the largest absolute activation and the
number of non-finite values, the top-8 tokens, and, with ``--compare`` (the logits file of
an earlier run), the difference to it: largest absolute logit difference and the KL of
the next-token distribution at every position.

Two uses: the stock model as the reader gives it against the fork's CPU oracle (the top-8
of ``ref_checkpoint.py``), and a float16 run against a float32 one (does a half-precision
teacher stay finite, and how far is it). The float32 model is 105 GB of host memory.

    python experiments/inkling_teacher_check.py CKPT --transformers DIR --dtype float32 \\
        --out run32.json --save-logits run32.pt
    python experiments/inkling_teacher_check.py CKPT --transformers DIR --dtype float16 \\
        --out run16.json --compare run32.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from hi2.inkling_source import MESSAGE_QUERY, InklingSource

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build(cfg, cls, dtype):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device("meta"):
            model = cls(cfg)
    finally:
        torch.set_default_dtype(prev)
    return model.to_empty(device="cpu").eval()


def fill(model, src: InklingSource, chunk: int) -> dict:
    named = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    filled: set[str] = set()
    bad: list[str] = []
    with torch.no_grad():
        for hf in [*src.hf_names().values(), MESSAGE_QUERY]:
            t = named[hf]
            t.copy_(src.load(hf).to(t.dtype).view_as(t))
            filled.add(hf)
            if not torch.isfinite(t).all():
                bad.append(hf)  # a cast to a narrower dtype overflowed
        i = src.inter
        for layer in range(src.num_layers):
            ex = model.model.language_model.layers[layer].mlp.experts
            for a in range(0, src.n_experts, chunk):
                part = list(range(a, min(a + chunk, src.n_experts)))
                gate, up = src.gate_up(layer, part)
                ex.gate_up_proj[part[0] : part[-1] + 1, :i] = gate.to(
                    ex.gate_up_proj.dtype
                )
                ex.gate_up_proj[part[0] : part[-1] + 1, i:] = up.to(
                    ex.gate_up_proj.dtype
                )
                ex.down_proj[part[0] : part[-1] + 1] = src.down(layer, part).to(
                    ex.down_proj.dtype
                )
            filled.update(
                f"model.language_model.layers.{layer}.mlp.experts.{n}"
                for n in ("gate_up_proj", "down_proj")
            )
            log(f"layer {layer} experts filled")
    left = [
        n for n in named if n not in filled and not n.startswith("model.vision_tower")
    ]
    return {
        "filled": len(filled),
        "text_tensors_not_filled": left,
        "dense_tensors_with_non_finite_weights": bad,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--transformers", type=Path, required=True)
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="float32")
    ap.add_argument("--prompt", default="What is the capital of France?")
    ap.add_argument("--chat-template-file", type=Path, default=None)
    ap.add_argument(
        "--text-file", type=Path, default=None, help="raw text instead of a chat prompt"
    )
    ap.add_argument(
        "--max-tokens", type=int, default=0, help="truncate the input (0: all)"
    )
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument(
        "--compare", type=Path, default=None, help="logits of an earlier run"
    )
    ap.add_argument("--save-logits", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("inkling_teacher_check.json"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    sys.path.insert(0, str(a.transformers))
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.inkling2_loop.modeling_inkling2_loop import (
        Inkling2LoopForConditionalGeneration,
    )

    dtype = DTYPES[a.dtype]
    cfg = AutoConfig.from_pretrained(a.ckpt)
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    if a.text_file is not None:
        ids = tok(a.text_file.read_text(), return_tensors="pt")["input_ids"]
    else:
        template = a.chat_template_file.read_text() if a.chat_template_file else None
        ids = tok.apply_chat_template(
            [{"role": "user", "content": a.prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            chat_template=template,
        )
        ids = torch.as_tensor(ids["input_ids"] if hasattr(ids, "keys") else ids)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    if a.max_tokens:
        ids = ids[:, : a.max_tokens]
    log(f"input {tuple(ids.shape)} tokens, model dtype {a.dtype}")

    t0 = time.time()
    model = build(cfg, Inkling2LoopForConditionalGeneration, dtype)
    log(
        f"allocated {sum(p.numel() for p in model.parameters()) / 1e9:.2f} B parameters"
    )
    with InklingSource(a.ckpt) as src:
        filled = fill(model, src, a.chunk)
    log(
        f"filled {filled['filled']} tensors in {time.time() - t0:.0f}s; not filled: {len(filled['text_tensors_not_filled'])}"
    )

    stats: dict[str, list] = {}

    def watch(name):
        def hook(_m, _args, out):
            x = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(x):
                f = x.float()
                stats.setdefault(name, []).append(
                    {
                        "max_abs": float(f.abs().max()),
                        "non_finite": int((~torch.isfinite(f)).sum()),
                    }
                )

        return hook

    handles = []
    for n, m in model.named_modules():
        if n.startswith("model.language_model.layers.") and (
            n.count(".") == 3 or n.endswith((".mlp", ".self_attn"))
        ):
            handles.append(m.register_forward_hook(watch(n)))
    t1 = time.time()
    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False).logits.float()
    log(f"forward {time.time() - t1:.0f}s, logits {tuple(logits.shape)}")
    for h in handles:
        h.remove()

    last = logits[0, -1]
    top = torch.topk(last, 8)
    result = {
        "dtype": a.dtype,
        "tokens": int(ids.shape[1]),
        "filled": filled["filled"],
        "text_tensors_not_filled": filled["text_tensors_not_filled"],
        "dense_tensors_with_non_finite_weights": filled[
            "dense_tensors_with_non_finite_weights"
        ],
        "top8": [
            (tok.decode([int(t)]), round(float(v), 3))
            for v, t in zip(top.values, top.indices)
        ],
        "logits_finite": bool(torch.isfinite(logits).all()),
        "logits_max_abs": float(logits.abs().max()),
        "seconds_build_fill": t1 - t0,
        "seconds_forward": time.time() - t1,
    }
    layers = {}
    for n, calls in sorted(stats.items(), key=lambda kv: (len(kv[0]), kv[0])):
        layers[n] = {
            "calls": len(calls),
            "max_abs": max(c["max_abs"] for c in calls),
            "non_finite": sum(c["non_finite"] for c in calls),
        }
    result["modules"] = layers
    dec = {n: v for n, v in layers.items() if n.count(".") == 3}
    result["decoder_layers_max_abs"] = max(v["max_abs"] for v in dec.values())
    result["non_finite_total"] = sum(v["non_finite"] for v in layers.values())
    if a.compare is not None:
        ref = torch.load(a.compare)
        diff = (logits - ref).abs()
        p, q = torch.log_softmax(ref, -1), torch.log_softmax(logits, -1)
        kl = (p.exp() * (p - q)).sum(-1)  # KL(reference || this run), per position
        result["compare"] = {
            "reference": str(a.compare),
            "max_abs_logit_diff": float(diff.max()),
            "mean_abs_logit_diff": float(diff.mean()),
            "kl_mean": float(kl.mean()),
            "kl_max": float(kl.max()),
            "kl_last": float(kl[0, -1]) if kl.dim() == 2 else float(kl[-1]),
            "top1_agree": float((logits.argmax(-1) == ref.argmax(-1)).float().mean()),
        }
    if a.save_logits is not None:
        a.save_logits.parent.mkdir(parents=True, exist_ok=True)
        torch.save(logits.cpu(), a.save_logits)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1) + "\n")
    brief = {
        k: v
        for k, v in result.items()
        if k not in ("modules", "text_tensors_not_filled")
    }
    log(json.dumps(brief, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
