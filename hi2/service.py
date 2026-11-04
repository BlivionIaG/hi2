# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Shared configuration, inspection and execution contracts for local adapters.

Compute imports are deliberately lazy: discovery and recipe parsing must not
initialize a model or accelerator. Numerical work stays in the existing pipeline.
"""

import dataclasses
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

from .runtime import SCHEMA_VERSION, atomic_json, environment, file_digest, read_run


def config_dict(rc) -> dict:
    """Serialize settings without deepcopying calibration tensors."""
    from .cache import ids_fingerprint

    def encode(value):
        if isinstance(value, Path):
            return str(value.resolve())
        if dataclasses.is_dataclass(value):
            return {
                f.name: encode(getattr(value, f.name))
                for f in dataclasses.fields(value)
            }
        if isinstance(value, (list, tuple)):
            return [encode(v) for v in value]
        if isinstance(value, dict):
            return {str(k): encode(v) for k, v in value.items()}
        return value

    result = {}
    for item in dataclasses.fields(rc):
        value = getattr(rc, item.name)
        if item.name == "calib_ids" and value is not None:
            result[item.name] = {
                "shape": list(value.shape),
                "sha256": ids_fingerprint(value),
            }
        else:
            result[item.name] = encode(value)
    return result


def config_from_dict(document: dict, overrides: dict | None = None):
    """Load a versioned recipe; explicit overrides win over file settings."""
    from .calibrate import Config
    from .pipeline import RunConfig
    from .qad import QadConfig

    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("recipe schema_version must be 1")
    if set(document) - {"schema_version", "config"}:
        raise ValueError("recipe accepts only schema_version and config")
    values = document.get("config")
    if not isinstance(values, dict):
        raise ValueError("recipe config must be an object")
    values = dict(values)
    values.update(overrides or {})
    allowed = {f.name for f in dataclasses.fields(RunConfig)}
    if extra := set(values) - allowed:
        raise ValueError(f"unknown recipe settings: {sorted(extra)}")
    if values.get("calib_ids") is not None:
        raise ValueError(
            "in-memory calibration tensors cannot be replayed; use calibration files"
        )
    for key in ("src", "dst", "workdir", "cache_dir"):
        if values.get(key) is not None:
            values[key] = Path(values[key])
    if values.get("calib") is not None:
        value = values["calib"]
        values["calib"] = (
            [Path(p) for p in value] if isinstance(value, list) else Path(value)
        )
    if "cfg" in values:
        settings = dict(values["cfg"])
        if "candidates" in settings:
            settings["candidates"] = [tuple(v) for v in settings["candidates"]]
        if "smooth_search" in settings:
            settings["smooth_search"] = tuple(settings["smooth_search"])
        values["cfg"] = Config(**settings)
    if values.get("qad") is not None:
        values["qad"] = QadConfig(**values["qad"])
    try:
        return RunConfig(**values)
    except TypeError as exc:
        raise ValueError(f"invalid recipe: {exc}") from exc


def save_recipe(path: Path, rc) -> None:
    config = config_dict(rc)
    if config.get("calib_ids") is not None:
        raise ValueError("use calibration files to save a replayable recipe")
    atomic_json(path, {"schema_version": SCHEMA_VERSION, "config": config})


def output_paths(rc) -> list[Path]:
    from .pipeline import output_dir

    if rc.max_error_ratios and len(rc.max_error_ratios) > 1:
        return [output_dir(rc.dst, ratio) for ratio in rc.max_error_ratios]
    return [rc.dst]


def _positive(name, value, *, zero=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (not zero and value == 0):
        raise ValueError(f"{name} must be {'nonnegative' if zero else 'positive'}")


def resolve_config(rc, log=lambda message: None, *, check_ownership=True):
    """Resolve all fallbacks before either planning or allocating compute work."""
    import regex
    import torch

    from .calibrate import SELECTABLE
    from .paths import check_output_paths
    from .pipeline import calib_paths, check_inputs, default_ratio
    from .qad import make_config

    rc = dataclasses.replace(rc, cfg=dataclasses.replace(rc.cfg))
    for key in ("src", "dst", "workdir", "cache_dir"):
        if getattr(rc, key) is not None:
            setattr(rc, key, Path(getattr(rc, key)).resolve())
    if rc.calib is not None:
        rc.calib = [p.resolve() for p in calib_paths(rc.calib)]
    for key in ("calib_samples", "seq_len", "sens_seqs", "sens_len", "sens_batch"):
        _positive(key, getattr(rc, key))
        if not isinstance(getattr(rc, key), int):
            raise ValueError(f"{key} must be an integer")
    for key in ("cache_devices", "sigma_fp16_above", "int8_force"):
        _positive(key, getattr(rc, key), zero=True)
    for key in ("sens_floor", "expert_share_floor"):
        _positive(key, getattr(rc, key), zero=True)
    for key in ("smooth", "damp", "expert_min_tokens", "refine_passes"):
        _positive(key, getattr(rc.cfg, key), zero=True)
    if rc.limit_layers is not None:
        _positive("limit_layers", rc.limit_layers)
    if not rc.cfg.candidates or any(
        tuple(c) not in SELECTABLE for c in rc.cfg.candidates
    ):
        raise ValueError("every candidate must be a supported bit-width/block pair")
    if rc.cfg.rotation not in ("on", "off", "auto"):
        raise ValueError("rotation must be on, off or auto")
    if rc.cfg.rounding not in ("nearest", "gptq") or rc.cfg.effort not in (
        "fast",
        "normal",
        "max",
    ):
        raise ValueError("unsupported rounding or effort")
    if rc.sensitivity not in ("none", "probe") or rc.moe_layout not in (
        "stacked",
        "per-expert",
    ):
        raise ValueError("unsupported sensitivity or MoE layout")
    if rc.compute_dtype not in ("float32", "float16", "bfloat16"):
        raise ValueError("compute_dtype must be float32, float16 or bfloat16")
    if rc.sens_dtype not in ("auto", "float32", "float16", "bfloat16"):
        raise ValueError("unsupported sensitivity dtype")
    regex.compile(rc.exclude)
    if rc.include:
        regex.compile(rc.include)
    ratios = list(rc.max_error_ratios or [])
    if rc.max_error_ratio is not None:
        if ratios:
            raise ValueError("use max_error_ratio or max_error_ratios, not both")
        ratios = [rc.max_error_ratio]
    if rc.target_bpw is not None:
        _positive("target_bpw", rc.target_bpw)
        if ratios:
            raise ValueError("give either target_bpw or max_error_ratio, not both")
    for ratio in ratios:
        _positive("max_error_ratio", ratio)
        if ratio < 1:
            raise ValueError("max_error_ratio must be at least 1")
    if len(set(ratios)) != len(ratios):
        raise ValueError("duplicate error ratios generate colliding outputs")
    if len(ratios) == 1:
        rc.max_error_ratio, rc.max_error_ratios = ratios[0], None
    if not rc.devices:
        raise ValueError("select at least one compute device")
    devices = []
    for value in rc.devices:
        device = torch.device(value)
        if device.type == "cuda":
            index = device.index if device.index is not None else 0
            if not torch.cuda.is_available() or index >= torch.cuda.device_count():
                raise RuntimeError(f"requested accelerator is unavailable: {value}")
            device = torch.device("cuda", index)
        elif device.type != "cpu":
            raise ValueError(f"unsupported quantizer device {value}; use cpu or cuda:N")
        devices.append(str(device))
    rc.devices = devices
    calibrated = rc.calib is not None or rc.calib_ids is not None
    if rc.calib is not None and rc.calib_ids is not None:
        raise ValueError("provide calibration files or in-memory ids, not both")
    if rc.qad is not None:
        rc.qad = make_config("default", dataclasses.asdict(rc.qad))
    if not calibrated:
        if rc.qad is not None:
            log("warning: no calibration; QAD falls back to PTQ only")
            rc.qad = None
        if rc.cfg.rounding == "gptq":
            log("no calibration data: using nearest rounding")
            rc.cfg.rounding = "nearest"
            rc.cfg.sequential = False
    if rc.sensitivity == "probe" and not calibrated:
        raise ValueError("sensitivity probe needs calibration data")
    if rc.qad is not None and rc.cfg.sequential:
        raise ValueError("QAD already uses the quantized prefix; drop sequential")
    if rc.target_bpw is None and not ratios:
        rc.max_error_ratio = default_ratio(rc)
    if rc.calib_ids is not None:
        ids = rc.calib_ids
        if (
            ids.ndim != 2
            or ids.shape[1] != rc.seq_len
            or ids.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("calib_ids must be integer [samples, seq_len] token IDs")
    outputs = [rc.workdir, *output_paths(rc)]
    if rc.cache_dir is not None:
        outputs.append(rc.cache_dir)
    check_output_paths(inputs=[rc.src, *(rc.calib or [])], outputs=outputs)
    rc._input_facts = check_inputs(rc, log)
    rc._checkpoint_info = inspect_model(rc.src)
    rc._resource_estimates = resource_estimates(rc)
    from .runtime import check_run_ownership

    rc._identity = (
        check_run_ownership(rc.workdir, output_paths(rc), lambda: run_identity(rc))
        if check_ownership
        else None
    )
    return rc


def inspect_model(path: Path) -> dict:
    from safetensors import safe_open

    from .checkpoint import weight_map

    path = Path(path).resolve()
    config = json.loads((path / "config.json").read_text())
    mapping = weight_map(path)
    if not mapping:
        raise ValueError(f"checkpoint has no tensors: {path}")
    size = 0
    count = 0
    tensor_bytes = 0
    largest_tensor_bytes = 0
    largest_hessian_bytes = 0
    parameters = 0
    dtype_bytes = {
        "F64": 8,
        "F32": 4,
        "F16": 2,
        "BF16": 2,
        "I64": 8,
        "I32": 4,
        "I16": 2,
        "I8": 1,
        "U8": 1,
        "BOOL": 1,
    }
    for name in sorted(set(mapping.values())):
        shard = (path / name).resolve()
        if not shard.is_file():
            raise ValueError(f"missing checkpoint shard: {name}")
        with safe_open(shard, framework="pt") as stream:
            present = set(stream.keys())
            missing = {k for k, v in mapping.items() if v == name} - present
            if missing:
                raise ValueError(
                    f"{name}: missing indexed tensors {sorted(missing)[:3]}"
                )
            count += len(present)
            for key in present:
                tensor = stream.get_slice(key)
                shape = tensor.get_shape()
                elements = math.prod(shape)
                parameters += elements
                size_bytes = elements * dtype_bytes.get(tensor.get_dtype(), 1)
                tensor_bytes += size_bytes
                largest_tensor_bytes = max(largest_tensor_bytes, size_bytes)
                if len(shape) == 2 and key.endswith(".weight"):
                    largest_hessian_bytes = max(
                        largest_hessian_bytes, shape[1] ** 2 * 4
                    )
        size += shard.stat().st_size
    text = config.get("text_config", config)
    return {
        "schema_version": SCHEMA_VERSION,
        "path": str(path),
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures", []),
        "layers": text.get("num_hidden_layers"),
        "hidden_size": text.get("hidden_size"),
        "quantization_config": config.get("quantization_config"),
        "tensor_count": count,
        "shard_bytes": size,
        "parameters": parameters,
        "tensor_bytes": tensor_bytes,
        "largest_tensor_bytes": largest_tensor_bytes,
        "largest_hessian_bytes_fp32": largest_hessian_bytes,
        "compatibility": {
            "metadata": "readable",
            "native_runtime": "not_measured",
            "model_execution": "not_measured",
        },
    }


def run_identity(rc) -> dict:
    from .cache import auxiliary_model_inputs, model_fingerprint
    from .pipeline import calib_paths

    config = config_dict(rc)
    data = []
    for path in calib_paths(rc.calib) if rc.calib is not None else []:
        data.append({"path": str(path.resolve()), "sha256": file_digest(path)})
    assets = {}
    for path in sorted(rc.src.iterdir()):
        if (
            path.is_file()
            and path.suffix != ".safetensors"
            and not path.name.startswith(".")
        ):
            assets[path.name] = file_digest(path)
    return {
        "config": config,
        "source": {
            "path": str(rc.src),
            "fingerprint": model_fingerprint(rc.src),
            "assets": assets,
            "auxiliary_inputs": auxiliary_model_inputs(),
        },
        "calibration": {"paths": data, "ids_fingerprint": config.get("calib_ids")},
        "environment": environment(),
    }


def plan_run(rc, log=lambda message: None) -> dict:
    from .pipeline import plan

    return plan(rc, log)


def execute_run(rc, log=lambda message: None) -> dict:
    from .pipeline import run

    return run(rc, log)


def inspect_artifact(path: Path) -> dict:
    from .checkpoint import validate_checkpoint
    from .size import size_report

    result = inspect_model(path)
    if result["quantization_config"]:
        result["validation"] = validate_checkpoint(Path(path), log=lambda _: None)
        result["size"] = size_report(Path(path))
    marker = Path(path) / "hi2_run.json"
    if marker.exists():
        result["run"] = json.loads(marker.read_text())
    return result


def doctor() -> dict:
    info = environment()
    try:
        import torch

        info["torch_backend"] = (
            "rocm" if torch.version.hip else "cuda" if torch.version.cuda else "cpu"
        )
        info["devices"] = [
            {"index": i, "name": torch.cuda.get_device_name(i)}
            for i in range(torch.cuda.device_count())
        ]
        info["cpu_available"] = True
        info["fused_kernel"] = {
            "compiler": shutil.which("hipcc"),
            "fallback": "torch",
            "execution": "not_measured",
        }
    except ImportError as exc:
        info.update(cpu_available=False, error=str(exc))
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ready" if info.get("cpu_available") else "unavailable",
        **info,
    }


def status(workdir: Path) -> dict:
    record = read_run(Path(workdir))
    return {
        "schema_version": SCHEMA_VERSION,
        **record,
        "recovery": "Replay the recorded configuration only with matching input identities; use fresh output/work paths for changes.",
    }


class Progress:
    """A shared event sink, also callable by existing text-log producers."""

    def __init__(self, sink=None, *, quiet=False, jsonl=False):
        self.sink = sink
        self.quiet = quiet
        self.jsonl = jsonl
        self.started = time.monotonic()

    def event(self, phase, status, **details):
        item = {
            "schema_version": SCHEMA_VERSION,
            "phase": phase,
            "status": status,
            "elapsed_s": round(time.monotonic() - self.started, 3),
            **details,
        }
        if self.sink:
            self.sink(item)
        if not self.quiet:
            print(
                json.dumps(item)
                if self.jsonl
                else f"{phase}: {status}" + (f" {details}" if details else ""),
                file=sys.stderr,
            )
        return item

    def __call__(self, message):
        self.event("log", "message", message=str(message))


def resource_estimates(rc) -> dict:
    """Metadata-only bounds, deliberately not a GPU-fit or runtime guarantee."""
    model = rc._checkpoint_info
    storage = []
    for purpose, path in [
        ("work", rc.workdir),
        *[("output", p) for p in output_paths(rc)],
    ]:
        parent = path.resolve()
        while not parent.exists():
            parent = parent.parent
        if not parent.is_dir():
            raise ValueError(f"{purpose} ancestor is not a directory: {parent}")
        if not os.access(parent, os.W_OK | os.X_OK):
            raise ValueError(f"{purpose} parent is not writable: {parent}")
        storage.append(
            {
                "purpose": purpose,
                "path": str(path),
                "free_bytes": shutil.disk_usage(parent).free,
            }
        )
    try:
        memory = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        memory = None
    full_model_stages = []
    if rc.sensitivity == "probe":
        full_model_stages.append("sensitivity_probe")
    if rc.qad is not None and rc.qad.final_kl:
        full_model_stages.append("final_kl")
    return {
        "scope": "metadata-derived component sizes; not simultaneous peak estimates",
        "source_tensor_bytes": model["tensor_bytes"],
        "largest_source_tensor_bytes": model["largest_tensor_bytes"],
        "largest_hessian_bytes_fp32": model["largest_hessian_bytes_fp32"]
        if rc.cfg.rounding == "gptq"
        else 0,
        "dense_export_bytes_fp16": model["parameters"] * 2,
        "full_model_weights_bytes_fp32": model["parameters"] * 4,
        "full_model_stages": full_model_stages,
        "host_physical_memory_bytes": memory,
        "storage": storage,
        "total_runtime_s": None,
        "peak_vram_bytes": None,
        "warning": "These are component sizes, not fit guarantees. Candidate tables, activations, optimizer state, cache copies and evaluation exports require additional space.",
    }


def plan_tuning(
    rc, *, target_kld, text, ratios, device, dtype, skip_tokens, log=lambda _: None
):
    """Preflight measured tuning without pretending the final ratio is known."""
    import torch
    from transformers import AutoTokenizer

    from .evalq import load_calibration_ids_offset
    from .paths import check_output_paths
    from .pipeline import _plan_resolved, calib_paths

    _positive("target_kld", target_kld)
    if rc.calib is None or text is None:
        raise ValueError("target KLD needs calibration and a separate tuning text")
    if (
        rc.target_bpw is not None
        or rc.max_error_ratio is not None
        or rc.max_error_ratios
    ):
        raise ValueError("target KLD replaces target_bpw and error-ratio targets")
    if not ratios or len(set(ratios)) != len(ratios):
        raise ValueError("tuning ratios must be nonempty and unique")
    for ratio in ratios:
        _positive("tuning ratio", ratio)
        if ratio < 1:
            raise ValueError("tuning ratios must be at least 1")
    _positive("eval_skip_tokens", skip_tokens, zero=True)
    if not isinstance(skip_tokens, int) or dtype not in (
        "float32",
        "float16",
        "bfloat16",
    ):
        raise ValueError("invalid tuning evaluation offset or dtype")
    if not device.split(",") or any(not d for d in device.split(",")):
        raise ValueError("select tuning evaluation devices")
    for name in device.split(","):
        dev = torch.device(name)
        if dev.type not in ("cpu", "cuda"):
            raise ValueError("tuning evaluation supports cpu or cuda:N")
        if dev.type == "cuda" and (
            not torch.cuda.is_available()
            or (dev.index or 0) >= torch.cuda.device_count()
        ):
            raise RuntimeError(f"tuning evaluation accelerator unavailable: {name}")
    text = Path(text).resolve()
    if not text.is_file():
        raise ValueError(f"tuning evaluation text is missing: {text}")
    check_output_paths(
        inputs=[rc.src, *calib_paths(rc.calib), text],
        outputs=[rc.dst, rc.workdir, rc.dst.parent / f"{rc.dst.name}.tune.json"],
    )
    if (
        rc.workdir.exists()
        and any(p.name != ".run.lock" for p in rc.workdir.iterdir())
        and not (rc.workdir / "run.json").is_file()
    ):
        raise ValueError(f"unowned work directory {rc.workdir}; choose a fresh workdir")
    if (
        rc.dst.exists()
        and (not rc.dst.is_dir() or any(rc.dst.iterdir()))
        and not (rc.dst / ".hi2-owner.json").is_file()
    ):
        raise ValueError(f"unowned output already exists: {rc.dst}")
    resolved = resolve_config(rc, log, check_ownership=False)
    tokenizer = AutoTokenizer.from_pretrained(resolved.src, trust_remote_code=False)
    load_calibration_ids_offset(text, tokenizer, 8, 2048, skip_tokens)
    result = _plan_resolved(resolved, log)
    result["max_error_ratio"] = None
    result["config"]["max_error_ratio"] = None
    result["tuning"] = {
        "target_kld": target_kld,
        "trial_ratios": ratios,
        "text": str(text),
        "samples": 8,
        "seq_len": 2048,
        "device": device,
        "dtype": dtype,
        "skip_tokens": skip_tokens,
        "maximum_extensions": 4,
        "selected_ratio": None,
        "final_quality": "not_measured",
        "resume_identity": "checked again after the selected ratio is measured",
    }
    result["phases"] = [
        "tuning_ptq_trials",
        "tuning_dense_evaluation",
        *result["phases"],
    ]
    result["resource_estimates"]["full_model_stages"].append("tuning_dense_evaluation")
    return result
