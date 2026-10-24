# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""End-to-end quality of an HI2 checkpoint: perplexity of the original and the
quantised model, KL(original || quantised) and top-1 agreement, on held-out text.

The quantised weights are dequantised back to dense tensors and run through the
original architecture (a numerical check of the format, not a kernel test).
"""

import shutil
from pathlib import Path

import torch

from .checkpoint import export_dense


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
) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(src)
    ids = load_calibration_ids_offset(text, tok, samples, seq_len, skip_tokens)
    model = (
        AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32).eval().to(device)
    )
    ref_logp = []
    for i in range(ids.shape[0]):
        lg = model(ids[i : i + 1].to(device)).logits[0].float()
        ref_logp.append(torch.log_softmax(lg, -1).cpu())
    del model
    # Load the quantised model through the same class and loader as the original, from
    # an exported fp16 copy, so checkpoint/module name differences are handled by
    # transformers rather than by us.
    dense = dst.parent / (dst.name + ".eval-dense")
    export_dense(src, dst, dense)
    try:
        model = AutoModelForCausalLM.from_pretrained(dense, dtype=torch.float32)
        model = model.eval().to(device)
    finally:
        shutil.rmtree(dense, ignore_errors=True)
    nll_ref = nll_q = kl = agree = 0.0
    n = 0
    for i in range(ids.shape[0]):
        lg = model(ids[i : i + 1].to(device)).logits[0].float()
        lp_q = torch.log_softmax(lg, -1).cpu()
        lp_r = ref_logp[i]
        tgt = ids[i, 1:]
        nll_ref += -lp_r[:-1].gather(1, tgt[:, None]).sum().item()
        nll_q += -lp_q[:-1].gather(1, tgt[:, None]).sum().item()
        kl += (lp_r.exp() * (lp_r - lp_q)).sum(-1)[:-1].sum().item()
        agree += (lp_r.argmax(-1) == lp_q.argmax(-1))[:-1].sum().item()
        n += tgt.numel()
        log(f"seq {i + 1}/{ids.shape[0]}")
    out = {
        "tokens": n,
        "ppl_original": float(torch.exp(torch.tensor(nll_ref / n))),
        "ppl_quantized": float(torch.exp(torch.tensor(nll_q / n))),
        "kld_nats_per_token": kl / n,
        "top1_agreement": agree / n,
    }
    return out


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
