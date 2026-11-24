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

``--student-steps N`` adds a ``student`` stage: a stand-in student (random HI2 tensors of
the size of every routed-expert projection, rows alternating over ``--student-bits``) in
place of the experts, the pipeline's own fit line and the formula of
``hi2.finalkl.student_footprint``, then ``N`` final-KL steps of one window whose peak
memory per device (against the formula's ``need_gib``) and time per step are recorded.
The stand-in values are noise: only the sizes and the speed mean anything.

    python experiments/inkling_looped_smoke.py --src DIR --transformers DIR \\
        --devices cuda:0 cuda:1 [--seqs 2] [--seq-len 128] [--calib-layers 2] \\
        [--student-steps 3]
"""

import argparse
import gc
import json
import math
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


def standin(r: int, k: int, bits=(2, 4), block: int = 32, gen=None, span: float = 0.02):
    """An HI2 tensor of random codes and ladder bytes with the buffers of a real one.

    Rows alternate over ``bits``, half of the 16-column groups carry a Hadamard flag and
    the row constants put the decoded weights within ``span``: the bytes, the decode and
    the gradient path of a quantized expert, and noise for values, so no loss it gives
    means anything."""
    from hi2.format import HI2Tensor

    nb = torch.tensor([bits[i % len(bits)] for i in range(r)], dtype=torch.uint8)
    words = nb.to(torch.int64) * k // 32
    code_off = torch.zeros(r + 1, dtype=torch.int32)
    code_off[1:] = torch.cumsum(words, 0).to(torch.int32)
    nblk = k // block
    param_off = torch.arange(r + 1, dtype=torch.int32) * nblk
    codes = torch.randint(
        -(2**31), 2**31 - 1, (int(code_off[-1]),), dtype=torch.int32, generator=gen
    )
    params = torch.randint(0, 256, (2 * r * nblk,), dtype=torch.uint8, generator=gen)
    consts = torch.empty(r, 3, dtype=torch.float16)
    consts[:, 0] = -span / 2
    consts[:, 1] = span / 255
    consts[:, 2] = span / 255 / (2.0 ** nb.float() - 1)
    had = (torch.rand(k // 16, generator=gen) < 0.5).to(torch.uint8)
    return HI2Tensor(
        codes,
        code_off,
        params,
        param_off,
        consts,
        nb,
        torch.full((r,), block, dtype=torch.uint8),
        had,
        (r, k),
    )


def student(model, ids, heads, ex, a, res) -> None:
    """The stand-in student of the final-KL phase on the model as it stands (after the
    forward stage): every routed-expert Linear a :class:`LadderLinear` of a random HI2
    tensor, the fit line and the formula, then ``--student-steps`` steps of
    :func:`hi2.finalkl.final_kl` on window 0 (window 1 held out), whose peak memory per
    device and time per step are recorded against the formula. The time of a step is the
    gap between two optimizer steps (no held-out pass in between)."""
    from types import SimpleNamespace

    from hi2.checkpoint import make_resolver, source_map
    from hi2.finalkl import LadderLinear, final_kl, student_footprint
    from hi2.qad import make_config

    det = res.setdefault("student_detail", {})

    def flush():
        a.out.write_text(json.dumps(res, indent=2, default=str))

    resolve = make_resolver(source_map(a.src))
    gen = torch.Generator().manual_seed(1)
    bits = tuple(int(x) for x in a.student_bits.split(","))
    t0 = time.perf_counter()
    tensors = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and ".mlp.experts." in name:
            tensors[resolve(f"{name}.weight")] = standin(
                mod.out_features, mod.in_features, bits, a.student_block, gen
            )
    weights = sum(t.shape[0] * t.shape[1] for t in tensors.values())
    code_bits = sum(32 * t.codes.numel() for t in tensors.values()) / weights
    det["standins"] = {
        "tensors": len(tensors),
        "weights": weights,
        "code_bits_per_weight": round(code_bits, 3),
        "bits_per_weight_with_ladder": round(code_bits + 16 / a.student_block, 3),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    print("standins", json.dumps(det["standins"]), flush=True)
    qc = make_config(
        "fast", {"final_kl": a.student_steps, "final_batch": 1, "final_evals": 1}
    )
    foot = student_footprint(model, tensors, resolve, qc, a.seq_len, len(ex.devices))

    def gib(xs):
        return [round(x / 2**30, 2) for x in xs]

    det["formula"] = {
        "static_gib": gib(foot["static"]),
        "saved_inputs_gib": gib(foot["activations"]),
        "transient_gib": round(foot["transient"] / 2**30, 2),
        "logits_gib": round(foot["logits"] / 2**30, 2),
        "need_gib": gib(foot["devices"]),
    }
    print("formula", json.dumps(det["formula"]), flush=True)
    cuda = [d for d in ex.devices if d.type == "cuda"]
    if cuda:
        from hi2.pipeline import _looped_fits

        rc = SimpleNamespace(src=a.src, qad=qc)
        det["fits"] = _looped_fits(rc, ex, model, tensors, a.seq_len, print)
    flush()
    marks: list = []
    grads: list = []  # per step: the trained values that had a gradient, and all of them
    orig = torch.optim.AdamW.step
    orig_zero = torch.optim.AdamW.zero_grad

    def fill(opt):  # a gradient for every trained value: an unrouted expert has none
        for g in opt.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)

    def zeroed(self, *args, **kw):
        out = orig_zero(self, *args, **kw)
        if a.student_full_grads:
            fill(self)
        return out

    def timed(self, *args, **kw):
        params = [p for g in self.param_groups for p in g["params"]]
        grads.append(
            {
                "with_gradient": sum(p.numel() for p in params if p.grad is not None),
                "total": sum(p.numel() for p in params),
            }
        )
        if a.student_full_grads:
            fill(self)
        out = orig(self, *args, **kw)
        for d in cuda:
            torch.cuda.synchronize(d)
        marks.append(
            (time.perf_counter(), [torch.cuda.memory_allocated(d) for d in cuda])
        )
        print(
            f"optimizer step {len(marks)}: {marks[-1][0] - t1:.0f} s after the stage "
            f"began, allocated {[round(x / 2**30, 2) for x in marks[-1][1]]} GiB, "
            f"peak {_peaks(ex.devices)}",
            flush=True,
        )
        det["steps_done"] = len(marks)
        flush()
        return out

    export = LadderLinear.export
    if not a.student_export:
        LadderLinear.export = lambda self, stager=None: None
    torch.optim.AdamW.step = timed
    torch.optim.AdamW.zero_grad = zeroed
    t1 = time.perf_counter()
    try:
        _, _, info = final_kl(
            model,
            ids,
            [0],
            [min(1, ids.shape[0] - 1)],
            tensors,
            {},
            qc,
            ex.devices,
            resolve,
            print,
            head_inputs=heads,
        )
    except BaseException as e:  # what was measured before the failure stays in the JSON
        det["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        det["error_peak_gib"] = _peaks(ex.devices)
        det["steps_done"] = len(marks)
        flush()
        raise
    finally:
        torch.optim.AdamW.step = orig
        torch.optim.AdamW.zero_grad = orig_zero
        LadderLinear.export = export
    losses = [t[1] for t in info["trace"] if t[1] is not None]
    det["final_kl"] = {
        "steps": a.student_steps,
        "seconds_total": round(time.perf_counter() - t1, 1),
        "first_step_done_s": round(marks[0][0] - t1, 1) if marks else None,
        "step_seconds": [round(y[0] - x[0], 2) for x, y in zip(marks, marks[1:])],
        "allocated_gib_after_step": [gib(m[1]) for m in marks],
        "peak_gib": _peaks(ex.devices),
        "peak_reserved_gib": {
            str(d): round(torch.cuda.max_memory_reserved(d) / 2**30, 2) for d in cuda
        },
        "allocator_peak_gib": {  # inactive_split: free bytes inside segments in use
            str(d): {
                k: round(
                    torch.cuda.memory_stats(d).get(f"{k}_bytes.all.peak", 0) / 2**30, 2
                )
                for k in ("allocated", "reserved", "inactive_split")
            }
            for d in cuda
        },
        "gradient_values": grads,
        "val_before": info["val_before"],
        "val_after": info["val_after"],
        "train_losses": losses,
    }
    print("final_kl", json.dumps(det["final_kl"]), flush=True)
    res["student_ok"] = bool(
        len(marks) == a.student_steps
        and all(math.isfinite(x) for x in (info["val_before"], *losses))
    )


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
    ap.add_argument(
        "--compare-tokens",
        type=int,
        default=0,
        help="compare the first N positions against the model's own forward on the host "
        "(0 = every position, the model sharded over --devices: needs the whole model "
        "on them)",
    )
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("inkling_looped_smoke.json"))
    ap.add_argument(
        "--student-steps",
        type=int,
        default=0,
        help="after the forward stage, build a stand-in student (random HI2 tensors for "
        "every routed-expert Linear), print the fit line and the formula of "
        "hi2.finalkl.student_footprint, run this many final-KL steps of one window and "
        "measure the peak memory per device and the time of a step (0 = skip)",
    )
    ap.add_argument(
        "--student-bits",
        default="2,4",
        help="code widths the stand-in rows alternate over (2,4 = 3.0 bits per weight)",
    )
    ap.add_argument("--student-block", type=int, default=32)
    ap.add_argument(
        "--student-export",
        action="store_true",
        help="also time the export of the trained ladders back to HI2 tensors",
    )
    ap.add_argument(
        "--student-full-grads",
        action="store_true",
        help="give every trained value a gradient before each step, as if every expert "
        "had been routed (an unrouted expert has no gradient and no optimizer state: "
        "the worst case of the memory, which random stand-in tensors do not reach)",
    )
    ap.add_argument(
        "--skip-compare",
        action="store_true",
        help="no reference forward and no comparison: the student stage alone",
    )
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
        a.out.write_text(
            json.dumps(res, indent=2, default=str)
        )  # kept if a later stage dies
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

    def run_student():
        gc.collect()
        for d in ex.devices:
            if d.type == "cuda":
                torch.cuda.empty_cache()
        stage("student", lambda: student(model, ids, heads, ex, a, res))
        res["ok"] = bool(res["ok"] and res["student_ok"])

    if a.skip_compare:  # the student stage alone: no reference forward, no comparison
        res["ok"] = res["program"]["finite"]
        if a.student_steps > 0:
            run_student()
        a.out.write_text(json.dumps(res, indent=2, default=str))
        print(json.dumps(res, indent=2, default=str))
        return 0 if res["ok"] else 1

    def forward():
        if a.compare_tokens > 0:
            # the model's own forward on the host, on a prefix: the fp16 model of 49 GiB
            # does not fit two devices of 30 GiB once the allocator has rounded its 5 MiB
            # tensors up (an OOM at 24 GiB on the first device, the stage-3 run)
            with torch.no_grad():
                return torch.stack(
                    [
                        _head_input(model, ids[i : i + 1, : a.compare_tokens]).float()
                        for i in range(ids.shape[0])
                    ]
                )
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
    if a.compare_tokens > 0:  # the program again, on the length the reference ran
        got = stage(
            "program_short",
            lambda: looped.teacher_head_inputs(
                model, ids[:, : a.compare_tokens], ex, log=print
            ),
        )
        res["compare_prefix_relative"] = float(
            (heads[:, : a.compare_tokens] - got).abs().max()
            / max(float(ref.abs().max()), 1e-30)
        )  # the long run's first positions against the short run
    else:
        got = heads
    diff = float((got - ref).abs().max())
    scale = float(ref.abs().max())
    per_position = (got - ref).abs().amax(dim=-1) / max(scale, 1e-30)
    res["compare"] = {
        "positions": int(ref.shape[1]),
        "max_abs_diff": diff,
        "max_abs_value": scale,
        "relative": diff / max(scale, 1e-30),
        "median_position_relative": float(per_position.median()),
        "p99_position_relative": float(torch.quantile(per_position.flatten(), 0.99)),
        "tol": a.tol,
        "finite": bool(torch.isfinite(ref).all()),
    }
    res["ok"] = bool(
        res["program"]["finite"]
        and res["compare"]["finite"]
        and res["compare"]["relative"] <= a.tol
    )
    if a.student_steps > 0:
        run_student()
    a.out.write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
