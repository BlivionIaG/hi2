# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""End-to-end quality of an HI2 checkpoint: perplexity of the original and the
quantised model, KL(original || quantised) and top-1 agreement, on held-out text.

The quantised weights are dequantised back to dense tensors and run through the
original architecture (a numerical check of the format, not a kernel test).
"""

import shutil
from pathlib import Path

import torch

from .checkpoint import export_dense
from .pinned import module_to_device

# KL(original || quantised) per token below which the quantised model is treated as
# distribution-lossless (the "KL below about 0.01" threshold of the SLQ paper, Fig. 1b).
LOSSLESS_KLD = 0.01


def _place(model, devices: list[str]):
    """Put ``model`` on one device, or split it layer-wise over several (a model that does
    not fit one GPU in the evaluation dtype). Returns the device of the inputs."""
    devs = [torch.device(d) for d in devices]
    if len(devs) == 1:
        return module_to_device(model, devs[0]), devs[0]
    from .sensitivity import shard_layers

    shard_layers(model, devs)
    return model, devs[0]


@torch.no_grad()
def evaluate(
    src: Path,
    dst: Path,
    text: Path,
    samples: int = 8,
    seq_len: int = 2048,
    device: str = "cpu",
    skip_tokens: int = 0,
    log=print,
    dtype: str = "float32",
) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .pipeline import _attach_mmap_ngram, _shrink_ngram_embedding

    devices = [d for d in device.split(",") if d]
    tok = AutoTokenizer.from_pretrained(src)
    ids = load_calibration_ids_offset(text, tok, samples, seq_len, skip_tokens)
    _shrink_ngram_embedding(log)
    model, first = _place(
        AutoModelForCausalLM.from_pretrained(src, dtype=getattr(torch, dtype)).eval(),
        devices,
    )
    _attach_mmap_ngram(model, log)
    ref_logp = []
    for i in range(ids.shape[0]):
        lg = model(ids[i : i + 1].to(first)).logits[0].float()
        ref_logp.append(torch.log_softmax(lg, -1).cpu())
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Load the quantised model through the same class and loader as the original, from
    # an exported fp16 copy, so checkpoint/module name differences are handled by
    # transformers rather than by us.
    dense = dst.parent / (dst.name + ".eval-dense")
    export_dense(src, dst, dense)
    try:
        model, first = _place(
            AutoModelForCausalLM.from_pretrained(
                dense, dtype=getattr(torch, dtype)
            ).eval(),
            devices,
        )
        _attach_mmap_ngram(model, log)
    finally:
        shutil.rmtree(dense, ignore_errors=True)
    nll_ref = nll_q = kl = agree = acc = 0.0
    n = 0
    for i in range(ids.shape[0]):
        lg = model(ids[i : i + 1].to(first)).logits[0].float()
        lp_q = torch.log_softmax(lg, -1).cpu()
        lp_r = ref_logp[i]
        tgt = ids[i, 1:]
        nll_ref += -lp_r[:-1].gather(1, tgt[:, None]).sum().item()
        nll_q += -lp_q[:-1].gather(1, tgt[:, None]).sum().item()
        kl += (lp_r.exp() * (lp_r - lp_q)).sum(-1)[:-1].sum().item()
        agree += (lp_r.argmax(-1) == lp_q.argmax(-1))[:-1].sum().item()
        # expected acceptance rate of speculative sampling: sum_x min(p, q) per token
        acc += torch.minimum(lp_r.exp(), lp_q.exp()).sum(-1)[:-1].sum().item()
        n += tgt.numel()
        log(f"seq {i + 1}/{ids.shape[0]}")
    return {
        "tokens": n,
        "ppl_original": float(torch.exp(torch.tensor(nll_ref / n))),
        "ppl_quantized": float(torch.exp(torch.tensor(nll_q / n))),
        "kld_nats_per_token": kl / n,
        "top1_agreement": agree / n,
        "ear": acc / n,
        "lossless": kl / n < LOSSLESS_KLD,
        "dtype": dtype,
    }


def load_calibration_ids_offset(text, tok, samples, seq_len, skip_tokens):
    """Like ``load_calibration_ids`` but skipping the first ``skip_tokens`` tokens
    (the calibration region), so evaluation text is held out."""
    import json

    if text.suffix == ".jsonl":
        raw = "\n\n".join(
            json.loads(ln)["text"] for ln in text.read_text().splitlines() if ln
        )
    else:
        raw = text.read_text()
    ids = tok(raw, return_tensors="pt", add_special_tokens=False).input_ids[0][
        skip_tokens:
    ]
    need = samples * seq_len
    if ids.numel() < need:
        raise SystemExit(f"eval text has {ids.numel()} usable tokens, need {need}")
    return ids[:need].reshape(samples, seq_len)
