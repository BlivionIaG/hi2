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
from .checkpoint import is_eligible, table_hessian_path, table_stats_path
from .executor import Executor, activate
from .hessian import HessianStats
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


def _run(layer, dev, items, hidden, kwargs, linears, k_of, with_hessian=False):
    """Run ``items`` (indices) through a device copy of ``layer``, returning the
    outputs, this device's per-Linear statistics and (optionally) full second
    moments. Linears that receive the very same tensor (q, k, v; gate, up) share one
    accumulator, found on the first item."""
    stats = {n: ColumnStats(k_of[n], dev) for n in linears}
    hess: dict[str, HessianStats] = {}
    leader_of: dict[str, str] = {}
    groups: dict[tuple, str] = {}
    keep_alive: list[torch.Tensor] = []
    first = [True]

    def make_hook(key):
        def hook(_mod, args):
            x = args[0].detach()
            stats[key].update(x)
            if not with_hessian:
                return
            if first[0]:
                keep_alive.append(x)  # keeps the address from being reused
                gk = (x.data_ptr(), tuple(x.shape))
                leader_of[key] = groups.setdefault(gk, key)
                if leader_of[key] == key:
                    hess[key] = HessianStats(k_of[key], dev)
            if leader_of[key] == key:
                hess[key].update(x)

        return hook

    hooks = []
    for n, mod in layer.named_modules():
        if n in stats:
            hooks.append(mod.register_forward_pre_hook(make_hook(n)))
    kw = to_device(kwargs, dev)
    outs = {}
    with torch.no_grad():
        for i in items:
            out = layer(hidden[i].to(dev), **kw)
            outs[i] = (out[0] if isinstance(out, tuple) else out).detach().cpu()
            first[0] = False
            keep_alive.clear()
    for h in hooks:
        h.remove()
    for key, leader in leader_of.items():
        hess.setdefault(key, hess[leader])
    return outs, stats, hess


def _device_pass(layer, dev, items, hidden, kwargs, linears, k_of, with_hessian):
    activate(dev)
    return _run(
        copy.deepcopy(layer).to(dev),
        dev,
        items,
        hidden,
        kwargs,
        linears,
        k_of,
        with_hessian,
    )


def _layer_pass(
    layer, devices, n_seq, hidden, kwargs, linears, k_of, with_hessian=False
):
    """Run all sequences through ``layer``, split round-robin over ``devices``
    (one thread each); returns each device's ``(outputs, statistics, hessians)``."""
    splits = [list(range(i, n_seq, len(devices))) for i in range(len(devices))]
    jobs = list(zip(devices, splits))
    if len(jobs) == 1:
        dev, items = jobs[0]
        return [
            _device_pass(layer, dev, items, hidden, kwargs, linears, k_of, with_hessian)
        ]
    with ThreadPoolExecutor(len(jobs)) as pool:
        futs = [
            pool.submit(
                _device_pass,
                layer,
                dev,
                items,
                hidden,
                kwargs,
                linears,
                k_of,
                with_hessian,
            )
            for dev, items in jobs
        ]
        return [f.result() for f in futs]


def _merge_hessians(results, linears, primary) -> dict[str, torch.Tensor]:
    """Merge each device's accumulators (shared inputs counted once) and normalise by
    the token count: ``Sigma = E[x x^T]`` per checkpoint tensor name (CPU)."""
    first = results[0][2]
    seen: dict[int, torch.Tensor] = {}
    out: dict[str, torch.Tensor] = {}
    for name, ckpt in linears.items():
        acc = first.get(name)
        if acc is None:
            continue
        gid = id(acc)
        if gid not in seen:
            total = HessianStats(acc.k, primary)
            for _, _, hess in results:
                if name in hess:
                    total.merge(hess[name])
            seen[gid] = total.mean().cpu()
        out[ckpt] = seen[gid]
    return out


def _save_hessians(results, linears, workdir: Path, primary) -> None:
    """Write ``Sigma`` per Linear under ``workdir/hessian``."""
    (workdir / "hessian").mkdir(parents=True, exist_ok=True)
    written: dict[int, Path] = {}
    for ckpt, sigma in _merge_hessians(results, linears, primary).items():
        path = table_hessian_path(workdir, ckpt)
        if (
            id(sigma) in written
        ):  # shared input (q/k/v, gate/up): store once, link the rest
            path.unlink(missing_ok=True)
            path.symlink_to(written[id(sigma)].name)
        else:
            torch.save(sigma, path)
            written[id(sigma)] = path


def encode_layerwise(
    model: nn.Module,
    ids: torch.Tensor,
    cfg: Config,
    ex: Executor,
    tables: dict,
    picks: dict,
    include,
    exclude,
    limit_layers: int | None = None,
    log=print,
    resolve=None,
) -> dict:
    """Spec 8.4: the encode pass with sequential propagation.

    Layers are quantized in order; the Hessian of each Linear is accumulated from the
    activations produced by the *already quantized* earlier layers, the Linear is
    encoded with error compensation, and its dequantized weights replace the original
    ones before the layer's outputs are propagated. Returns ``{checkpoint name:
    HI2Tensor}``.
    """
    from .calibrate import encode_with_picks
    from .format import dequantize

    path, layers = find_layers(model)
    prefix = path + "."
    hidden, kwargs = capture_inputs(model, layers, ids)
    n_seq = len(hidden)
    encoded: dict = {}
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
                if ckpt is not None and ckpt in tables:
                    linears[n] = ckpt
        k_of = {n: layer.get_submodule(n).in_features for n in linears}
        results = _layer_pass(
            layer, ex.devices, n_seq, hidden, kwargs, linears, k_of, True
        )
        sigmas = _merge_hessians(results, linears, ex.primary)
        del results
        for n, ckpt in linears.items():
            mod = layer.get_submodule(n)
            w = mod.weight.detach().float().cpu()
            t = encode_with_picks(
                w, None, tables[ckpt], picks[ckpt], cfg, ex, sigmas.get(ckpt)
            )
            encoded[ckpt] = t
            mod.weight.data = dequantize(t).to(mod.weight.dtype)
        outs = _layer_pass(layer, ex.devices, n_seq, hidden, kwargs, {}, {}, False)
        for o, _, _ in outs:
            for i, v in o.items():
                hidden[i] = v
        log(f"layer {li}: {len(linears)} linears encoded (sequential)")
    return encoded


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
        want_h = cfg.rounding == "gptq" and not cfg.sequential
        results = _layer_pass(
            layer, ex.devices, n_seq, hidden, kwargs, linears, k_of, want_h
        )
        merged = {n: ColumnStats(k_of[n], ex.primary) for n in linears}
        for outs, stats, _ in results:
            for i, o in outs.items():
                hidden[i] = o
            for n, st in stats.items():
                merged[n].merge(st)
        if want_h:
            _save_hessians(results, linears, workdir, ex.primary)
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
