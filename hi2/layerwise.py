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
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from torch import nn

from .calibrate import Config, build_table, shared_transform
from .checkpoint import is_eligible, table_hessian_path, table_stats_path
from .executor import Executor, activate
from .hessian import HessianStats
from .moe import blend, coverage, expert_parts, group_key, sibling_leader
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
    outputs, this device's per-Linear statistics, (optionally) full second moments and
    the leader of each Linear that ran.

    Linears that receive the very same tensor (q, k, v; gate and up of one expert)
    share one accumulator, kept under the first of them (the leader). Sharing is decided
    the first time a Linear runs, which for a routed expert can be any sequence, by the
    input's address among the inputs of the current sequence (kept alive until the
    sequence is done, so an address is never reused within it). An expert Linear only
    runs on the tokens routed to it, so its statistics are those tokens' only."""
    stats: dict[str, ColumnStats] = {}
    hess: dict[str, HessianStats] = {}
    leader_of: dict[str, str] = {}
    groups: dict[tuple, str] = {}
    keep_alive: list[torch.Tensor] = []

    def make_hook(key):
        def hook(_mod, args):
            x = args[0].detach()
            gk = (x.data_ptr(), tuple(x.shape))
            lead = leader_of.get(key)
            if lead is None:
                lead = leader_of[key] = groups.get(gk, key)
                if lead == key:
                    stats[key] = ColumnStats(k_of[key], dev)
                    if with_hessian:
                        hess[key] = HessianStats(k_of[key], dev)
            if gk not in groups:
                groups[gk] = lead
                keep_alive.append(x)
            if lead == key:
                stats[key].update(x)
                if with_hessian:
                    hess[key].update(x)

        return hook

    hooks = []
    for n, mod in layer.named_modules():
        if n in linears:
            hooks.append(mod.register_forward_pre_hook(make_hook(n)))
    kw = to_device(kwargs, dev)
    outs = {}
    with torch.no_grad():
        for i in items:
            out = layer(hidden[i].to(dev), **kw)
            outs[i] = (out[0] if isinstance(out, tuple) else out).detach().cpu()
            groups.clear()
            keep_alive.clear()
    for h in hooks:
        h.remove()
    return outs, stats, hess, leader_of


def _merge(results, linears, k_of, primary, want_h, min_tokens, log=print):
    """Merge the devices' accumulators per leader (shared inputs counted once).

    Returns ``(lead, stats, hess, counts, fallback)``: the leader of every Linear,
    the summed :class:`ColumnStats` (and :class:`HessianStats` when ``want_h``) per
    leader, the tokens each Linear saw, and the Linears whose statistics were topped
    up. An expert that saw fewer than ``min_tokens`` x its widest input tokens borrows
    the layer average of its projection (all experts' tokens pooled) for the missing
    tokens (:func:`hi2.moe.blend`); its own tokens are kept."""
    lead: dict[str, str] = {}
    for *_, lo in results:
        lead.update(lo)
    for n in linears:  # never ran on any device: structural siblings, else itself
        if n not in lead:
            sib = sibling_leader(n)
            lead[n] = lead.get(sib, sib) if sib in linears else n
    leaders = list(dict.fromkeys(lead[n] for n in linears))
    stats, hess = {}, {}
    for ld in leaders:
        stats[ld] = ColumnStats(k_of[ld], primary)
        for r in results:
            if ld in r[1]:
                stats[ld].merge(r[1][ld])
        if want_h:
            hess[ld] = HessianStats(k_of[ld], primary)
            for r in results:
                if ld in r[2]:
                    hess[ld].merge(r[2][ld])
    counts = {n: stats[lead[n]].n for n in linears}
    fallback: set[str] = set()
    if min_tokens <= 0:
        return lead, stats, hess, counts, fallback
    width: dict[tuple, int] = {}
    for n in linears:
        p = expert_parts(n)
        if p is not None:
            width[p[:2]] = max(width.get(p[:2], 0), k_of[n])
    by_group: dict[str, list[str]] = {}
    for ld in leaders:
        if expert_parts(ld) is not None:
            by_group.setdefault(group_key(ld), []).append(ld)
    for key, lds in by_group.items():
        tot_n = sum(stats[ld].n for ld in lds)
        tot_s = sum(stats[ld].S for ld in lds)
        tot_h = sum(hess[ld].S for ld in lds) if want_h else None
        short = []
        for ld in lds:
            need = int(min_tokens * width[expert_parts(ld)[:2]])
            if stats[ld].n >= need or tot_n == 0:
                continue
            short.append(f"{expert_parts(ld)[1]}:{stats[ld].n}")
            st = stats[ld]
            st.S, st.n = blend(st.S, st.n, tot_s, tot_n, need)
            if want_h:
                h = hess[ld]
                h.S, h.n = blend(h.S, h.n, tot_h, tot_n, need)
            fallback |= {n for n in linears if lead[n] == ld}
        if short:
            log(
                f"  {key}: {len(short)}/{len(lds)} experts below the token floor, "
                f"topped up with the layer average (expert:tokens {' '.join(short)})"
            )
    return lead, stats, hess, counts, fallback


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


def _save_hessians(lead, hess, linears, workdir: Path) -> dict[str, torch.Tensor]:
    """Write ``Sigma = E[x x^T]`` once per leader under ``workdir/hessian`` (the other
    Linears of a shared input are symlinks); returns ``{checkpoint name: Sigma}``."""
    (workdir / "hessian").mkdir(parents=True, exist_ok=True)
    out: dict[str, torch.Tensor] = {}
    for ld, h in hess.items():
        sigma = h.mean().cpu()
        path = table_hessian_path(workdir, linears[ld])
        torch.save(sigma, path)
        for n, ckpt in linears.items():
            if lead[n] == ld:
                out[ckpt] = sigma
                if n != ld:
                    link = table_hessian_path(workdir, ckpt)
                    link.unlink(missing_ok=True)
                    link.symlink_to(path.name)
    return out


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
        lead, _, hess, _, _ = _merge(
            results, linears, k_of, ex.primary, True, cfg.expert_min_tokens, log
        )
        del results
        sigmas = {ckpt: hess[lead[n]].mean().cpu() for n, ckpt in linears.items()}
        del hess
        for n, ckpt in linears.items():
            mod = layer.get_submodule(n)
            w = mod.weight.detach().float().cpu()
            t = encode_with_picks(
                w, None, tables[ckpt], picks[ckpt], cfg, ex, sigmas.get(ckpt)
            )
            encoded[ckpt] = t
            mod.weight.data = dequantize(t).to(mod.weight.dtype)
        outs = _layer_pass(layer, ex.devices, n_seq, hidden, kwargs, {}, {}, False)
        for o, *_ in outs:
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
    expert_counts: dict[str, int] = {}
    expert_fallback: set[str] = set()
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
        for outs, *_ in results:
            for i, o in outs.items():
                hidden[i] = o
        lead, merged, hess, counts, fallback = _merge(
            results, linears, k_of, ex.primary, want_h, cfg.expert_min_tokens, log
        )
        del results
        if want_h:
            _save_hessians(lead, hess, linears, workdir)
        del hess
        members: dict[str, list[str]] = {}
        for n in linears:
            members.setdefault(lead[n], []).append(n)
        for ld, names in members.items():
            st = merged[ld] if merged[ld].n > 0 else None
            ws = {
                n: layer.get_submodule(n).weight.detach().float().cpu() for n in names
            }
            transform = shared_transform([ws[n] for n in names], st, cfg, ex)
            for n in names:
                full = linears[n]
                if st is None:
                    log(f"warning: {full} saw no calibration tokens; data-free weights")
                else:
                    torch.save(st.state(), table_stats_path(workdir, full))
                tables[full] = build_table(
                    full, ws[n], st, cfg, ex, transform, counts[n]
                )
                if expert_parts(n) is not None:
                    expert_counts[full] = counts[n]
                    if n in fallback:
                        expert_fallback.add(full)
        log(f"layer {li}: {len(linears)} linears tabulated")
    if expert_counts:
        cov = coverage(expert_counts, expert_fallback)
        for key, c in cov.items():
            log(
                f"  {key}: tokens per expert min {c['min']} median {c['median']:g} "
                f"max {c['max']}, {len(c['fallback'])} topped up"
            )
        (workdir / "expert_tokens.json").write_text(
            json.dumps(
                {
                    "min_tokens_per_input_width": cfg.expert_min_tokens,
                    "counts": expert_counts,
                    "fallback": sorted(expert_fallback),
                    "coverage": cov,
                },
                indent=1,
            )
        )
    return tables
