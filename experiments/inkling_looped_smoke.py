# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""The layer program of ``hi2.looped`` on the real Inkling-2-Mini: the GPU smoke.

Stages, each timed, with the peak memory of every device:

* ``load``: the model in RAM (float16 by default), experts unfused, ``check_unfused``;
* ``program``: ``looped.teacher_head_inputs`` on ``--seqs`` sequences of ``--seq-len`` tokens,
  one layer at a time on ``--devices``, the model never placed: 48 layer calls;
* ``calibrate`` (``--calib-layers N``, default 0 = skipped): ``looped.calibrate_looped`` over
  the first ``N`` layers, the statistics tables of their Linears;
* ``forward``: ``sensitivity.shard_layers`` over ``--devices`` (the layer-wise split the
  final-KL phase uses: ``final_input`` beside the last layers, ``message_input`` beside the
  embeddings), then the model's own forward of every sequence. The program's head inputs
  are compared with it (largest absolute difference relative to the largest value), and
  the devices of ``final_input`` / ``message_input`` / the head are recorded.

Exit code 0: every stage ran, every value is finite and the head inputs agree within
``--tol`` (the two runs differ only in float summation order: batching, the device).
``--devices cpu`` runs the same stages without a GPU (the tiny model of the fork's
``make_tiny_checkpoint`` for a dry run).

    python experiments/inkling_looped_smoke.py --src DIR --transformers DIR \\
        --devices cuda:0 cuda:1 [--seqs 2] [--seq-len 128] [--calib-layers 2]
"""

import argparse
import gc
import json
import re
import resource
import sys
import time
from pathlib import Path

import torch


def _peaks(devices) -> dict[str, float]:
    return {
        str(d): round(torch.cuda.max_memory_allocated(d) / 2**30, 2)
        for d in devices
        if d.type == "cuda"
    }


def _reset(devices) -> None:
    for d in devices:
        if d.type == "cuda":
            torch.cuda.synchronize(d)
            torch.cuda.reset_peak_memory_stats(d)


def _rss_gib() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, required=True, help="Inkling-2 directory")
    ap.add_argument(
        "--transformers",
        type=Path,
        required=True,
        help="dir holding the fork's transformers",
    )
    ap.add_argument("--devices", nargs="+", default=["cpu"])
    ap.add_argument("--seqs", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--compute-dtype", default="float16")
    ap.add_argument(
        "--text", type=Path, default=None, help="calibration text (default: random ids)"
    )
    ap.add_argument("--calib-layers", type=int, default=0)
    ap.add_argument("--tol", type=float, default=2e-2)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("inkling_looped_smoke.json"))
    a = ap.parse_args()
    sys.path.insert(0, str(a.transformers))

    from hi2 import looped
    from hi2.calibrate import Config
    from hi2.checkpoint import DEFAULT_EXCLUDE, make_resolver, source_map
    from hi2.executor import Executor
    from hi2.finalkl import _head_input
    from hi2.layerwise import calibrate_layerwise
    from hi2.pipeline import RunConfig, _load_model, load_calibration_ids
    from hi2.sensitivity import shard_layers

    a.workdir.mkdir(parents=True, exist_ok=True)
    ex = Executor(a.devices)
    res: dict = {
        "src": str(a.src),
        "devices": a.devices,
        "seqs": a.seqs,
        "seq_len": a.seq_len,
        "dtype": a.compute_dtype,
    }

    def stage(name, fn):
        _reset(ex.devices)
        t0 = time.perf_counter()
        out = fn()
        res[name] = {
            "seconds": round(time.perf_counter() - t0, 1),
            "peak_gib": _peaks(ex.devices),
            "rss_gib": _rss_gib(),
        }
        print(name, json.dumps(res[name]), flush=True)
        return out

    rc = RunConfig(
        src=a.src,
        dst=a.workdir / "out",
        workdir=a.workdir,
        compute_dtype=a.compute_dtype,
        devices=a.devices,
    )
    model = stage("load", lambda: _load_model(rc))
    if a.text is not None:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(a.src)
        ids = load_calibration_ids(a.text, tok, a.seqs, a.seq_len)
    else:
        g = torch.Generator().manual_seed(0)
        vocab = min(model.config.text_config.unpadded_vocab_size, 5000)
        ids = torch.randint(0, vocab, (a.seqs, a.seq_len), generator=g)
    heads = stage(
        "program", lambda: looped.teacher_head_inputs(model, ids, ex, log=print)
    )
    res["program"]["finite"] = bool(torch.isfinite(heads).all())
    res["program"]["head_inputs_shape"] = list(heads.shape)
    if a.calib_layers > 0:
        tables = stage(
            "calibrate",
            lambda: calibrate_layerwise(
                model,
                ids,
                Config(effort="fast"),
                ex,
                a.workdir / "calib",
                re.compile(looped.DEFAULT_INCLUDE),
                re.compile(DEFAULT_EXCLUDE),
                a.calib_layers,
                print,
                make_resolver(source_map(a.src)),
            ),
        )
        res["calibrate"]["tables"] = len(tables)
        gc.collect()

    def forward():
        handles = shard_layers(model, ex.devices)
        text = model.model.language_model
        res["placement"] = {
            "embed_tokens": str(text.embed_tokens.weight.device),
            "message_input": str(text.message_input.query.device),
            "layer_0": str(text.layers[0].mlp.gate.weight.device),
            "layer_last": str(text.layers[-1].mlp.gate.weight.device),
            "final_input": str(next(text.final_input.parameters()).device),
            "norm": str(next(text.norm.parameters()).device),
            "lm_head": str(model.lm_head.weight.device),
        }
        outs = []
        with torch.no_grad():
            for i in range(ids.shape[0]):
                h = _head_input(model, ids[i : i + 1].to(ex.devices[0]))
                outs.append(h.float().cpu())
        for h in handles:
            h.remove()
        return torch.stack(outs)

    ref = stage("forward", forward)
    diff = float((heads - ref).abs().max())
    scale = float(ref.abs().max())
    res["compare"] = {
        "max_abs_diff": diff,
        "max_abs_value": scale,
        "relative": diff / max(scale, 1e-30),
        "tol": a.tol,
        "finite": bool(torch.isfinite(ref).all()),
    }
    res["ok"] = bool(
        res["program"]["finite"]
        and res["compare"]["finite"]
        and res["compare"]["relative"] <= a.tol
    )
    a.out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
