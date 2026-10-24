# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Phase 1 with calibration data: layer-by-layer statistics collection.

The model lives in RAM; one decoder layer at a time is copied to each compute
device, the calibration sequences are run through it (split across devices) while
hooks accumulate per-Linear second-moment statistics, then its Linears get their
candidate tables. Hidden states between layers stay in RAM, so peak GPU memory is
about one layer. Activations are propagated from the unquantised model: the final
configuration is not known until the global allocation.
"""

import contextlib
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from torch import nn

from .calibrate import Config, build_table
from .checkpoint import is_eligible, table_stats_path
from .executor import Executor, activate
from .stats import ColumnStats

LAYER_PATHS = (
    "model.layers",
    "model.model.layers",
    "model.language_model.layers",
    "language_model.model.layers",
    "transformer.h",
    "gpt_neox.layers",
    "model.decoder.layers",
)


class _Stop(Exception):
    pass


def find_layers(model: nn.Module) -> tuple[str, nn.ModuleList]:
    for path in LAYER_PATHS:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, nn.ModuleList):
            return path, obj
    raise ValueError(f"cannot find the decoder layers (tried {LAYER_PATHS})")


def to_device(obj, dev):
    if isinstance(obj, torch.Tensor):
        return obj.to(dev)
    if isinstance(obj, tuple):
        return tuple(to_device(o, dev) for o in obj)
    if isinstance(obj, list):
        return [to_device(o, dev) for o in obj]
    if isinstance(obj, dict):
        return {k: to_device(v, dev) for k, v in obj.items()}
    return obj


class _Catcher(nn.Module):
    """Stands in for layer 0: records its input and keyword arguments, then stops."""

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer
        self.inputs: list[torch.Tensor] = []
        self.kwargs: dict | None = None

    def __getattr__(self, name):  # e.g. attention_type read by the model
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.layer, name)

    def forward(self, hidden_states, *args, **kwargs):
        self.inputs.append(hidden_states.detach().cpu())
        if self.kwargs is None:
            self.kwargs = kwargs
        raise _Stop


def capture_inputs(model: nn.Module, layers: nn.ModuleList, ids: torch.Tensor):
    """Layer-0 hidden states for each calibration sequence (batch size 1)."""
    catcher = _Catcher(layers[0])
    layers[0] = catcher
    try:
        for i in range(ids.shape[0]):
            with contextlib.suppress(_Stop):
                model(ids[i : i + 1], use_cache=False)
    finally:
        layers[0] = catcher.layer
    kwargs = {k: v for k, v in (catcher.kwargs or {}).items() if k != "past_key_values"}
    kwargs["use_cache"] = False
    return catcher.inputs, kwargs


def _run(layer, dev, items, hidden, kwargs, linears, k_of):
    """Run ``items`` (indices) through a device copy of ``layer``, returning the
    outputs and this device's per-Linear statistics."""
    stats = {n: ColumnStats(k_of[n], dev) for n in linears}
    hooks = []
    for n, mod in layer.named_modules():
        if n in stats:
            hooks.append(
                mod.register_forward_pre_hook(
                    lambda m, a, key=n: stats[key].update(a[0].detach())
                )
            )
    kw = to_device(kwargs, dev)
    outs = {}
    with torch.no_grad():
        for i in items:
            out = layer(hidden[i].to(dev), **kw)
            outs[i] = (out[0] if isinstance(out, tuple) else out).detach().cpu()
    for h in hooks:
        h.remove()
    return outs, stats


def _device_pass(layer, dev, items, hidden, kwargs, linears, k_of):
    activate(dev)
    return _run(copy.deepcopy(layer).to(dev), dev, items, hidden, kwargs, linears, k_of)


def _layer_pass(layer, devices, n_seq, hidden, kwargs, linears, k_of):
    """Run all sequences through ``layer``, split round-robin over ``devices``
    (one thread each); returns each device's ``(outputs, statistics)``."""
    splits = [list(range(i, n_seq, len(devices))) for i in range(len(devices))]
    jobs = list(zip(devices, splits))
    if len(jobs) == 1:
        dev, items = jobs[0]
        return [_device_pass(layer, dev, items, hidden, kwargs, linears, k_of)]
    with ThreadPoolExecutor(len(jobs)) as pool:
        futs = [
            pool.submit(_device_pass, layer, dev, items, hidden, kwargs, linears, k_of)
            for dev, items in jobs
        ]
        return [f.result() for f in futs]


def calibrate_layerwise(
    model: nn.Module,
    ids: torch.Tensor,
    cfg: Config,
    ex: Executor,
    workdir: Path,
    include,
    exclude,
    limit_layers: int | None = None,
    log=print,
    resolve=None,
):
    """Return ``{checkpoint tensor name: TensorTable}`` and write each Linear's
    statistics under ``workdir/stats``."""
    path, layers = find_layers(model)
    prefix = path + "."
    hidden, kwargs = capture_inputs(model, layers, ids)
    n_seq = len(hidden)
    log(f"captured {n_seq} calibration sequences of {tuple(hidden[0].shape)}")
    tables = {}
    (workdir / "stats").mkdir(parents=True, exist_ok=True)
    for li, layer in enumerate(layers):
        if limit_layers is not None and li >= limit_layers:
            break
        linears = {}
        for n, mod in layer.named_modules():
            full = f"{prefix}{li}.{n}.weight"
            if isinstance(mod, nn.Linear) and is_eligible(
                full, mod.weight.shape, mod.weight.dtype, include, exclude
            ):
                ckpt = resolve(full) if resolve else full
                if ckpt is None:
                    log(f"warning: {full} has no tensor in the checkpoint; skipped")
                    continue
                linears[n] = ckpt
        k_of = {n: layer.get_submodule(n).in_features for n in linears}
        results = _layer_pass(layer, ex.devices, n_seq, hidden, kwargs, linears, k_of)
        merged = {n: ColumnStats(k_of[n], ex.primary) for n in linears}
        for outs, stats in results:
            for i, o in outs.items():
                hidden[i] = o
            for n, st in stats.items():
                merged[n].merge(st)
        for n, full in linears.items():
            st = merged[n]
            use = st if st.n > 0 else None
            if use is None:
                log(f"warning: {full} saw no calibration tokens; data-free weights")
            else:
                torch.save(st.state(), table_stats_path(workdir, full))
            w = layer.get_submodule(n).weight.detach().float().cpu()
            tables[full] = build_table(full, w, use, cfg, ex)
        log(f"layer {li}: {len(linears)} linears tabulated")
    return tables
