# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""``hi2 quantize`` on the stock Inkling-2-Mini directory, as far as a CPU goes.

hi2 reads a source checkpoint by tensor name. The stock directory is a serving layout
(``model.llm.*``, experts fused and packed in NVFP4 / MXFP8), so hi2 reads it through the
decoded view of ``hi2.inkling_source`` (``hi2.checkpoint.source_map`` / ``open_source``):
Hugging Face names, per-expert float32 projections, nothing staged on disk. This script
runs the stages of ``hi2 quantize`` that need no GPU pass over the real directory and
records what happened in ``--out/quantize_dryrun.json``:

* ``plan``: ``hi2 quantize --dry-run`` (``hi2.pipeline.plan``) of the experts-only PTQ at
  ``--target-bpw`` with INT2/INT4/INT8, and, with ``--calib``, of the calibrated QAD +
  final-KL run (checkpoint facts, resource estimates, phases);
* ``census``: every tensor of the view by pattern: elements, dtype, and whether the
  default ``--exclude`` (and an experts-only ``--include``) makes it an HI2 candidate;
* ``sample``: a real data-free ``hi2.pipeline.run`` (tables, global allocation, encoding,
  validation) on a few experts and attention linears of two layers, with the view cut to
  those experts (the dense tensors are all there), then ``verify_checkpoint`` of the output
  against the source by name;
* ``model``: ``hi2.pipeline._load_model`` (the fork's transformers class registered by
  ``register_auto_class``, ``unfuse_experts``, ``check_unfused``) in ``--compute-dtype``,
  the parameter count against the view's, and a forward of the chat prompt of
  ``inkling_teacher_check.py`` against its saved float32 logits (``--compare``);
* ``probe``: the calibrated layer-wise driver (``hi2.layerwise.calibrate_layerwise``) on a
  tiny real ``Inkling2LoopForConditionalGeneration`` (the fork's ``make_tiny_checkpoint``):
  it assumes ``layer(hidden, **kwargs) -> hidden``, the layers of Inkling take
  ``(hidden, block_states[, messages], ...)`` and the loop layers run three passes.

    python experiments/inkling_quantize_dryrun.py CKPT --out DIR --transformers DIR \\
        [--tools DIR] [--calib TEXT] [--compare LOGITS.pt] [--compute-dtype float16]

Exit code 0 when every requested stage did what it should (``probe`` is expected to fail:
it passes when the failure is the one described above).
"""

import argparse
import contextlib
import json
import re
import resource
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import torch

from hi2.calibrate import SELECTABLE, Config
from hi2.checkpoint import DEFAULT_EXCLUDE, is_eligible, open_source, source_map
from hi2.inkling_source import PROJECTIONS, expert_name, stock_view
from hi2.moe import expert_parts

EXPERTS = r"\.mlp\.experts\."
STAGES = ("plan", "census", "sample", "model", "probe")
DTYPE_OF = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rss_gib() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1)


def config(a, tag: str, rounding: str = "nearest", **kw):
    from hi2.pipeline import RunConfig

    bits = {int(b) for b in a.bits.split(",")}
    base = dict(
        src=a.ckpt,
        dst=a.out / tag / "dst",
        workdir=a.out / tag / "work",
        devices=["cpu"],
        target_bpw=a.target_bpw,
        include=EXPERTS,
        compute_dtype=a.compute_dtype,
        cfg=Config(
            candidates=[c for c in SELECTABLE if c[0] in bits],
            effort=a.effort,
            rounding=rounding,
        ),
    )
    base.update(kw)
    return RunConfig(**base)


# ---------------------------------------------------------------- plan


def stage_plan(a) -> dict:
    from hi2.pipeline import _plan_resolved, plan
    from hi2.service import resolve_config

    out: dict = {}
    t0 = time.time()
    p = plan(config(a, "plan_ptq"), log=log)  # with the run-ownership fingerprint
    out["ptq_data_free"] = {
        "seconds": round(time.time() - t0, 1),
        "phases": p["phases"],
        "layers": p["layers"],
        "checkpoint": {
            k: p["checkpoint"][k]
            for k in (
                "model_type",
                "layers",
                "hidden_size",
                "tensor_count",
                "shard_bytes",
                "parameters",
                "tensor_bytes",
                "largest_tensor_bytes",
                "largest_hessian_bytes_fp32",
            )
        },
        "resource_estimates": {
            k: v
            for k, v in p["resource_estimates"].items()
            if k not in ("storage", "warning", "scope")
        },
    }
    view = stock_view(a.ckpt)
    out["ptq_data_free"]["view_parameters"] = view.parameters()
    if a.calib is not None:
        from hi2.qad import QadConfig

        rc = config(
            a,
            "plan_qad",
            rounding="gptq",
            calib=a.calib,
            calib_samples=4,
            seq_len=a.plan_seq_len,
            qad=QadConfig(samples=4, val=2, final_kl=100),
        )
        t0 = time.time()
        p = _plan_resolved(resolve_config(rc, log, check_ownership=False), log)
        out["qad_final_kl"] = {
            "seconds": round(time.time() - t0, 1),
            "phases": p["phases"],
            "calib_tokens_available": p.get("calib_tokens_available"),
            "calib_tokens_needed": p["calib_tokens_needed"],
            "qad_trained_layers": p.get("qad_trained_layers"),
            "qad_hidden_state_ram_gib_fp32": p.get("qad_hidden_state_ram_gib_fp32"),
        }
        from hi2.qad import estimate_seconds

        sec, n_layers = estimate_seconds(
            a.ckpt, rc.qad, a.plan_seq_len, rc.include, rc.exclude, True
        )
        out["qad_final_kl"]["qad_estimate_seconds_gpu"] = round(sec)
        out["qad_final_kl"]["qad_estimate_layers"] = n_layers
    return out


# ---------------------------------------------------------------- census


def stage_census(a) -> dict:
    exc = re.compile(DEFAULT_EXCLUDE)
    inc = re.compile(EXPERTS)
    wmap = source_map(a.ckpt)
    rows: dict[str, dict] = {}
    with open_source(a.ckpt, next(iter(wmap.values()))) as f:
        for name in wmap:
            sl = f.get_slice(name)
            shape, dtype = sl.get_shape(), sl.get_dtype()
            tdt = DTYPE_OF.get(dtype, torch.int8)
            pattern = re.sub(r"(?<=\.)\d+(?=\.)", "N", name)
            r = rows.setdefault(
                pattern,
                {
                    "tensors": 0,
                    "elements": 0,
                    "dtype": dtype,
                    "shape": shape,
                    "candidate_default": 0,
                    "candidate_experts_only": 0,
                },
            )
            r["tensors"] += 1
            r["elements"] += torch.Size(shape).numel()
            r["candidate_default"] += is_eligible(name, shape, tdt, None, exc)
            r["candidate_experts_only"] += is_eligible(name, shape, tdt, inc, exc)
    total = sum(r["elements"] for r in rows.values())

    def share(key):
        return sum(r["elements"] for r in rows.values() if r[key])

    expert = sum(r["elements"] for p, r in rows.items() if ".mlp.experts." in p)
    return {
        "tensors": sum(r["tensors"] for r in rows.values()),
        "elements": total,
        "expert_elements": expert,
        "candidate_elements_default_exclude": share("candidate_default"),
        "candidate_elements_experts_only": share("candidate_experts_only"),
        "non_expert_candidates_by_default": {
            p: {k: r[k] for k in ("tensors", "elements", "dtype", "shape")}
            for p, r in sorted(rows.items(), key=lambda kv: -kv[1]["elements"])
            if r["candidate_default"] and ".mlp.experts." not in p
        },
        "patterns": dict(sorted(rows.items(), key=lambda kv: -kv[1]["elements"])),
    }


# ---------------------------------------------------------------- sample


@contextlib.contextmanager
def cut_view(ckpt, layers, experts):
    """The stock view with only the given experts of the given layers (every dense tensor
    stays): a sample of the real directory that ``write_checkpoints`` can pass over."""
    view = stock_view(ckpt)
    keep = {
        expert_name(layer, e, p)
        for layer in layers
        for e in experts
        for p in PROJECTIONS
    }
    full = view._keys
    view._keys = [k for k in full if expert_parts(k) is None or k in keep]
    try:
        yield view
    finally:
        view._keys = full


def stage_sample(a) -> dict:
    from hi2.checkpoint import verify_checkpoint
    from hi2.pipeline import run

    layers = [int(x) for x in a.layers.split(",")]
    experts = [int(x) for x in a.experts.split(",")]
    lay, exp = "|".join(map(str, layers)), "|".join(map(str, experts))
    include = (
        rf"(\.layers\.({lay})\.self_attn\.(q|k|v|o)_proj\.linear\.weight$"
        rf"|\.layers\.({lay})\.mlp\.experts\.({exp})\.)"
    )
    rc = config(a, "sample", include=include)
    t0 = time.time()
    with cut_view(a.ckpt, layers, experts) as view:
        names = view.keys()
        summary = run(rc, log=log)
        verified = verify_checkpoint(rc.src, rc.dst, 8, log=log)
    report = json.loads((rc.dst / "hi2_report.json").read_text())["tensors"]
    groups: dict[str, dict] = {}
    for name, r in report.items():
        layer, rest = re.search(r"layers\.(\d+)\.(.*)$", name).groups()
        key = "L{}:{}".format(layer, re.sub(r"\.\d+\.", ".E.", rest))
        g = groups.setdefault(
            key,
            {
                "tensors": 0,
                "params": 0,
                "bits": 0.0,
                "int4": 0.0,
                "int8": 0.0,
                "blocks": {},
            },
        )
        n = r["shape"][0] * r["shape"][1]
        g["tensors"] += 1
        g["params"] += n
        g["bits"] += r["bpw"] * n
        g["int4"] += r["frac_int4"] * r["shape"][0]
        g["int8"] += r.get("frac_int8", 0.0) * r["shape"][0]
        for b, c in r["block_hist"].items():
            g["blocks"][b] = g["blocks"].get(b, 0) + c
    for g in groups.values():
        g["bpw"] = round(g.pop("bits") / g["params"], 3)
    out_files = sorted(p.name for p in rc.dst.iterdir())
    return {
        "layers": layers,
        "experts": experts,
        "include": include,
        "view_names": len(names),
        "seconds": round(time.time() - t0, 1),
        "tensors": summary["tensors"],
        "avg_bpw": round(summary["avg_bpw"], 4),
        "allocation_bpw": round(summary["allocation_bpw"], 4),
        "frac_rows_by_bits": summary["frac_rows_by_bits"],
        "runtime_s": summary["runtime_s"],
        "peak_memory_gib": summary["peak_memory_gib"],
        "experts_summary": summary.get("experts"),
        "verify": verified,
        "groups": groups,
        "output_files": out_files,
    }


# ---------------------------------------------------------------- model


def stage_model(a) -> dict:
    from transformers import AutoTokenizer

    from hi2.pipeline import _load_model

    rc = config(a, "model")
    out: dict = {"compute_dtype": a.compute_dtype}
    t0 = time.time()
    model = _load_model(rc, log=log)  # register, from_pretrained, unfuse, check_unfused
    out["load_seconds"] = round(time.time() - t0, 1)
    out["peak_rss_gib"] = rss_gib()
    params = sum(p.numel() for p in model.parameters())
    out["parameters"] = params
    out["view_parameters"] = stock_view(a.ckpt).parameters()
    out["class"] = type(model).__name__
    experts = model.model.language_model.layers[0].mlp.experts
    out["experts_class"] = type(experts).__name__
    out["experts_gated_activation"] = bool(getattr(experts, "gated", False))
    tok = AutoTokenizer.from_pretrained(a.ckpt, trust_remote_code=False)
    ids = tok.apply_chat_template(
        [{"role": "user", "content": a.prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
    )
    ids = torch.as_tensor(ids["input_ids"] if hasattr(ids, "keys") else ids)
    ids = ids.unsqueeze(0) if ids.dim() == 1 else ids
    out["prompt_tokens"] = int(ids.shape[1])
    t1 = time.time()
    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False).logits.float()
    out["forward_seconds"] = round(time.time() - t1, 1)
    out["logits_finite"] = bool(torch.isfinite(logits).all())
    top = torch.topk(logits[0, -1], 8)
    out["top8"] = [
        (tok.decode([int(t)]), round(float(v), 3))
        for v, t in zip(top.values, top.indices)
    ]
    if a.compare is not None:
        ref = torch.load(a.compare).float()
        if ref.shape == logits.shape:
            lp_r, lp_s = torch.log_softmax(ref, -1), torch.log_softmax(logits, -1)
            out["versus_saved_logits"] = {
                "file": str(a.compare),
                "max_abs_logit_diff": float((logits - ref).abs().max()),
                "kl_max_over_positions": float(
                    (lp_r.exp() * (lp_r - lp_s)).sum(-1).max()
                ),
                "top1_agree": bool((ref.argmax(-1) == logits.argmax(-1)).all()),
            }
        else:
            out["versus_saved_logits"] = (
                f"shape {tuple(logits.shape)} vs {tuple(ref.shape)}"
            )
    return out


def pipeline_probe(a, work: Path, ids: torch.Tensor) -> dict:
    """``hi2.pipeline.run`` and the sensitivity probe on the tiny directory, which has the
    stock layout and a real Inkling config: what works, and what fails and how."""
    from hi2.executor import Executor
    from hi2.pipeline import RunConfig, run
    from hi2.sensitivity import probe_sensitivity

    out: dict = {}
    quiet = {"log": lambda *_: None}
    scratch = work.parent / (
        work.name + "_out"
    )  # outputs must not sit inside the source

    def rc(tag, rounding="nearest", **kw):
        base = dict(
            src=work,
            dst=scratch / tag / "dst",
            workdir=scratch / tag / "work",
            include=EXPERTS,
            devices=["cpu"],
            compute_dtype="float32",
            target_bpw=3.5,
            cfg=Config(effort="fast", rounding=rounding),
        )
        return RunConfig(**{**base, **kw})

    def attempt(tag, **kw):
        cfg = rc(tag, **kw)
        try:
            summary = run(cfg, **quiet)
            out[tag] = {
                "ok": True,
                "tensors": summary["tensors"],
                "avg_bpw": round(summary["avg_bpw"], 3),
            }
        except (Exception, SystemExit) as exc:
            out[tag] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:240]}"}
        return cfg

    free = attempt("data_free_ptq")
    attempt(
        "calibrated_ptq", rounding="gptq", calib_ids=ids, calib_samples=2, seq_len=12
    )
    try:
        tables = {
            p.name.removesuffix(".pt"): torch.load(p, weights_only=False)
            for p in (free.workdir / "tables").iterdir()
        }
        sids = torch.randint(0, 50, (3, 32))

        def probe(skip_prefix):
            return probe_sensitivity(
                free.src,
                tables,
                free.workdir,
                sids,
                free.cfg,
                Executor("cpu"),
                skip_prefix=skip_prefix,
                **quiet,
            )

        full, fast = probe(False), probe(True)
        worst = max(abs(fast[k] - full[k]) / max(abs(full[k]), 1e-30) for k in full)
        out["sensitivity_probe"] = {
            "ok": True,
            "units": len(full),
            "prefix_replay_equals_full_forward": full == fast,
            "largest_relative_difference": worst,
        }
    except (Exception, SystemExit) as exc:
        out["sensitivity_probe"] = {
            "ok": False,
            "error": f"{type(exc).__name__}: {str(exc)[:240]}",
        }
    shutil.rmtree(scratch, ignore_errors=True)
    return out


# ---------------------------------------------------------------- probe


def stage_probe(a) -> dict:
    """The layer-wise driver on a tiny real Inkling model."""
    sys.path.insert(0, str(a.tools))
    import make_tiny_checkpoint as make
    from transformers import AutoConfig
    from transformers.models.inkling2_loop.modeling_inkling2_loop import (
        Inkling2LoopForConditionalGeneration,
    )

    from hi2.executor import Executor
    from hi2.layerwise import calibrate_layerwise, capture_inputs, find_layers
    from hi2.moe import unfuse_experts

    out: dict = {}
    work = Path(tempfile.mkdtemp(prefix="inkling_probe_", dir=a.out))
    make.write_checkpoint(work, make.TinySpec(num_hidden_layers=6), seed=11)
    torch.manual_seed(0)
    model = Inkling2LoopForConditionalGeneration(
        AutoConfig.from_pretrained(work)
    ).eval()
    unfuse_experts(model)
    ids = torch.randint(0, 50, (2, 12))
    path, layers = find_layers(model)
    out["layers_path"] = path
    out["layer_class"] = type(layers[0]).__name__
    hidden, kwargs = capture_inputs(model, layers, ids)
    out["captured_hidden_shape"] = list(hidden[0].shape)
    out["captured_kwargs"] = sorted(kwargs)
    try:
        with torch.no_grad():
            layers[0](hidden[0], **kwargs)
        out["layer_replay"] = "ok"
    except Exception as exc:
        out["layer_replay"] = f"{type(exc).__name__}: {str(exc)[:240]}"
    try:
        calibrate_layerwise(
            model,
            ids,
            Config(effort="fast", rounding="nearest"),
            Executor(["cpu"]),
            work / "w",
            None,
            re.compile(DEFAULT_EXCLUDE),
            None,
            lambda *_: None,
        )
        out["calibrate_layerwise"] = "ok"
    except Exception as exc:
        out["calibrate_layerwise"] = f"{type(exc).__name__}: {str(exc)[:240]}"
    out["expected_failure"] = "block_states" in out["layer_replay"]
    out["pipeline"] = pipeline_probe(a, work, ids)
    shutil.rmtree(work, ignore_errors=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt", type=Path)
    ap.add_argument(
        "--out", type=Path, required=True, help="absolute work and result dir"
    )
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument(
        "--transformers", type=Path, default=None, help="the fork's transformers tree"
    )
    ap.add_argument(
        "--tools", type=Path, default=None, help="tools/rdna/inkling2 of the fork"
    )
    ap.add_argument(
        "--compute-dtype", default="float16", choices=("float32", "float16", "bfloat16")
    )
    ap.add_argument("--target-bpw", type=float, default=3.5)
    ap.add_argument("--bits", default="2,4,8")
    ap.add_argument("--effort", default="fast", choices=("fast", "normal", "max"))
    ap.add_argument(
        "--layers", default="0,20", help="sample: one NVFP4 and one MXFP8 layer"
    )
    ap.add_argument("--experts", default="0,64", help="sample: experts per layer")
    ap.add_argument(
        "--calib", type=Path, default=None, help="text for the calibrated plan"
    )
    ap.add_argument("--plan-seq-len", type=int, default=512)
    ap.add_argument(
        "--compare", type=Path, default=None, help="saved float32 logits of the prompt"
    )
    ap.add_argument("--prompt", default="What is the capital of France?")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.transformers is not None:
        sys.path.insert(0, str(a.transformers))
    a.out.mkdir(parents=True, exist_ok=True)
    result_path = a.out / "quantize_dryrun.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    stages = {
        "plan": stage_plan,
        "census": stage_census,
        "sample": stage_sample,
        "model": stage_model,
        "probe": stage_probe,
    }
    failed = []
    for name in a.stages.split(","):
        log(f"stage {name}")
        t0 = time.time()
        try:
            res = stages[name](a)
            res["ok"] = res.get("expected_failure", True) if name == "probe" else True
        except SystemExit as exc:
            res = {"ok": False, "error": f"SystemExit: {exc}"}
        except Exception as exc:
            res = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc()[-1500:],
            }
        res["stage_seconds"] = round(time.time() - t0, 1)
        result[name] = res
        result_path.write_text(json.dumps(result, indent=1, default=str) + "\n")
        log(
            f"stage {name}: {'ok' if res['ok'] else 'FAILED'} in {res['stage_seconds']} s"
        )
        if not res["ok"]:
            failed.append(name)
    log(f"peak rss {rss_gib()} GiB; failed stages: {failed or 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
