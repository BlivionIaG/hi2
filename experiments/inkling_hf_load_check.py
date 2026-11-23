# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""What transformers itself makes of the stock Inkling-2-Mini directory, against the reader.

``AutoModelForCausalLM.from_pretrained(CKPT)`` is how ``hi2.pipeline._load_model`` builds a
model. With the fork's transformers (``--transformers``) and ``Inkling2LoopForConditionalGeneration``
registered for the causal-LM auto class (``hi2.inkling_source.register_auto_class``, which the
model loaders of hi2 call themselves), this loads the serving checkpoint (the unknown
``modelopt`` quantization tag is skipped with a warning) and its own conversion renames the
tensors, composes ``message_input.query`` and dequantizes the experts. This script loads the
model in ``--dtype`` and compares every text-model tensor with ``hi2.inkling_source``: the
dense tensors and ``message_input.query`` exactly, the experts of every layer elementwise
(count of different elements, largest absolute difference) against the reader's float32
decode rounded once to ``--dtype``.

The loader casts every source tensor of an expert stack to the parameter dtype before
transformers' ``Inkling2LoopDequantizeExperts`` runs, ``amax`` included, which the reader (and the
vLLM fork's decoders) keep in float32. ``op_isolation`` runs that op on four experts of one NVFP4
and one MXFP8 layer with the sources cast and as stored, and compares both with the loaded model
and with the reader; ``amax_cast`` is the relative error the cast puts on every ``amax``.

Needs the fork's transformers and the host memory of the model (52 GB in half precision).

    python experiments/inkling_hf_load_check.py CKPT --transformers DIR --dtype float16
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from hi2.inkling_source import (
    MESSAGE_QUERY,
    SERVING_PREFIX,
    InklingSource,
    register_auto_class,
)

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
SUFFIXES = ("", "_scale", "_amax", "_outer_scale_denominator")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class _Model:
    """The one thing the op asks of ``model``: the dtype of the target parameter."""

    def __init__(self, dtype: torch.dtype):
        self._param = torch.empty(0, dtype=dtype)

    def get_parameter(self, name: str) -> torch.Tensor:
        return self._param


def _diff(got: torch.Tensor, want: torch.Tensor) -> dict:
    d = (got.float() - want.float()).abs()
    return {
        "different": int((d > 0).sum()),
        "elements": d.numel(),
        "max_abs_diff": float(d.max()),
    }


def amax_cast(src: InklingSource, dtype: torch.dtype) -> dict:
    """Relative error of ``amax.to(dtype)`` over every expert of every NVFP4 layer."""
    worst, sq, n = 0.0, 0.0, 0
    for layer in range(src.num_layers):
        if src.expert_format(layer) != "nvfp4":
            continue
        for which in ("w13", "w2"):
            amax = src.tensor(
                f"{SERVING_PREFIX}.layers.{layer}.mlp.experts.{which}_weight_amax"
            ).double()
            rel = (amax.to(dtype).double() - amax) / amax
            worst = max(worst, float(rel.abs().max()))
            sq += float(rel.pow(2).sum())
            n += rel.numel()
    return {"amax_values": n, "max_rel_error": worst, "rms_rel_error": (sq / n) ** 0.5}


def op_isolation(model, src: InklingSource, dtype: torch.dtype) -> list:
    """transformers' dequantization op on four experts, sources cast or as stored."""
    from transformers.core_model_loading import Inkling2LoopDequantizeExperts

    op = Inkling2LoopDequantizeExperts()
    experts = [0, 17, 64, 127]
    inter = src.inter
    out = []
    for layer in (0, 20):
        ex = model.model.language_model.layers[layer].mlp.experts
        for which in ("w13", "w2"):
            base = f"{SERVING_PREFIX}.layers.{layer}.mlp.experts.{which}_weight"
            nvfp4 = src.expert_format(layer) == "nvfp4"
            tensors = {}
            for key, suffix in zip("wsad", SUFFIXES):
                if key in "ad" and not nvfp4:
                    continue  # MXFP8 has neither amax nor denominator
                t = src.tensor(base + suffix)
                tensors[key] = t[experts] if key in "wsa" else t
            ref = src.stack(layer, which, experts, "cpu").to(dtype)
            rec = {"layer": layer, "format": src.expert_format(layer), "stack": which}
            for mode in ("sources_cast", "sources_as_stored"):
                inp = {
                    k: (t.to(dtype) if mode == "sources_cast" else t)
                    for k, t in tensors.items()
                }
                got = op.convert(
                    inp, list("wsad"), ["t"], model=_Model(dtype), full_layer_name="p"
                )["t"]
                if (
                    which == "w13"
                ):  # the loader's Interleave(dim=1) undoes the interleave
                    loaded = ex.gate_up_proj[experts]
                    same = torch.equal(got[:, 0::2], loaded[:, :inter]) and torch.equal(
                        got[:, 1::2], loaded[:, inter:]
                    )
                else:
                    same = torch.equal(got, ex.down_proj[experts])
                rec[mode] = {**_diff(got, ref), "equals_loaded_model": bool(same)}
            out.append(rec)
            log(
                f"op layer {layer} {which}: cast differs from reader in "
                f"{rec['sources_cast']['different']}, as stored in "
                f"{rec['sources_as_stored']['different']} of {rec['sources_cast']['elements']}"
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--transformers", type=Path, required=True)
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="float16")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("inkling_hf_load_check.json"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    sys.path.insert(0, str(a.transformers))
    from transformers import AutoModelForCausalLM

    dtype = DTYPES[a.dtype]
    if not register_auto_class(a.ckpt):
        raise SystemExit(f"{a.ckpt}: not an Inkling-2 directory (model_type)")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        a.ckpt,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        ignore_mismatched_sizes=True,  # as hi2.pipeline._load_model
    ).eval()
    result: dict = {
        "dtype": a.dtype,
        "class": type(model).__name__,
        "load_seconds": time.time() - t0,
    }
    log(f"loaded {result['class']} in {result['load_seconds']:.0f}s")
    named = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    with torch.no_grad(), InklingSource(a.ckpt) as src:
        dense_equal, dense_diff = 0, []
        for hf in src.hf_names().values():
            want = src.load(hf).to(named[hf].dtype)
            same = tuple(want.shape) == tuple(named[hf].shape) and torch.equal(
                want, named[hf]
            )
            dense_equal += same
            if not same:
                dense_diff.append(hf)
        result["dense_tensors"] = len(src.hf_names())
        result["dense_equal"] = dense_equal
        result["dense_different"] = dense_diff[:20]
        mq = src.message_query().to(named[MESSAGE_QUERY].dtype)
        result["message_query_equal"] = bool(torch.equal(mq, named[MESSAGE_QUERY]))
        log(
            f"dense equal {dense_equal}/{result['dense_tensors']}, "
            f"message_input.query equal {result['message_query_equal']}"
        )
        result["amax_cast"] = amax_cast(src, dtype)
        log(f"amax cast: {result['amax_cast']}")
        result["op_isolation"] = op_isolation(model, src, dtype)
        layers = []
        for layer in range(src.num_layers):
            ex = model.model.language_model.layers[layer].mlp.experts
            gate_up, down = src.fused_layer(layer, 16, "cpu", dtype)
            entry = {"layer": layer, "format": src.expert_format(layer)}
            for name, got, want in (
                ("gate_up", ex.gate_up_proj, gate_up),
                ("down", ex.down_proj, down),
            ):
                entry[name] = _diff(got, want)
            layers.append(entry)
            log(
                f"layer {layer:2d} {entry['format']}: gate_up "
                f"{entry['gate_up']['different']} / down {entry['down']['different']} "
                f"elements differ, max abs "
                f"{max(entry['gate_up']['max_abs_diff'], entry['down']['max_abs_diff']):.3g}"
            )
        result["layers"] = layers
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
