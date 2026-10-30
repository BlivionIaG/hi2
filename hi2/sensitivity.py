# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Sensitivity-aware allocation weights (spec 8.6).

The allocator minimises ``sum_t s_t * rel_err_t`` where ``rel_err_t`` is the proxy error
of tensor ``t`` (activation-weighted squared error over its weighted energy). With
``s_t = 1`` that treats a unit of proxy error as equally harmful everywhere, which it is
not. Here ``s_t`` is measured: quantize only tensor ``t`` at a fixed reference
configuration (nearest rounding), run a small batch through the whole model, and take

    s_t = KL(original || model with only t quantized) / rel_err_t(reference config)

so ``s_t * rel_err`` is a prediction of the KL the tensor will cost at any configuration.
"""

import copy
import os

import regex as re
import torch
from torch import nn

from .cache import StatsView
from .calibrate import Config, TensorTable, encode_with_picks
from .checkpoint import make_resolver, weight_map
from .executor import Executor
from .pinned import PinnedWindow, module_to_device
from .format import dequantize
from .stats import ColumnStats

REF_CANDIDATE = (2, 32)
REF_ON_DEVICE_BYTES = 4 << 30  # reference log-probs stay on the GPU below this
FIT_FRACTION = (
    0.6  # the weights may take this share of one device (the rest: activations)
)
SHARD_FRACTION = 0.8  # a shard may take this share of a device
_LAYER_IDX = re.compile(r"\blayers\.(\d+)\.")


def _layer_index(name: str) -> int:
    m = _LAYER_IDX.search(name)
    return int(m.group(1)) if m else 0


def _layers(model):
    from .layerwise import find_layers

    try:
        return find_layers(model)[1]
    except ValueError:
        return None


class _Replay(nn.Module):
    """Stands in for a decoder layer before the perturbed one: returns that layer's
    recorded reference output (attributes such as ``attention_type`` come from it)."""

    def __init__(self, layer: nn.Module, out):
        super().__init__()
        self.layer = layer
        self.out = out

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.layer, name)

    def forward(self, *args, **kwargs):
        return self.out


def probe_fits(weight_bytes: int, device_bytes: int, n_devices: int) -> bool:
    """Whether the probe model fits on one device or split over ``n_devices``."""
    return (
        weight_bytes <= FIT_FRACTION * device_bytes
        or weight_bytes / n_devices <= SHARD_FRACTION * device_bytes
    )


def fp32_weight_bytes(src) -> int:
    """The model's weights in float32, estimated from the stored files and the checkpoint
    dtype (bf16/fp16 store half the fp32 bytes). Used to resolve ``--sens-dtype auto``
    without loading the model twice."""
    import json
    from pathlib import Path

    src = Path(src)
    cfg = json.loads((src / "config.json").read_text())
    dt = cfg.get("torch_dtype") or cfg.get("dtype") or "bfloat16"
    scale = 1 if str(dt).lower() in ("float32", "fp32") else 2
    files = list(src.glob("*.safetensors")) or list(src.glob("*.bin"))
    return sum(f.stat().st_size for f in files) * scale


def plan_placement(weight_bytes: int, device_bytes: int, n_devices: int) -> str:
    """Where the probe model goes: ``single`` (one device holds it), ``shard`` (split
    layer-wise over the devices) or an error when even the shards do not fit."""
    if weight_bytes <= FIT_FRACTION * device_bytes:
        return "single"
    if probe_fits(weight_bytes, device_bytes, n_devices):
        return "shard"
    raise SystemExit(
        f"sensitivity probe: the model's {weight_bytes / 2**30:.0f} GiB of weights do not "
        f"fit {n_devices} device(s) of {device_bytes / 2**30:.0f} GiB. Use more devices, "
        "--sens-dtype float16, or --sensitivity none"
    )


def _move_own(module: nn.Module, dev: torch.device) -> None:
    """Move the parameters and buffers a module owns itself (its children stay).

    The copies go through a pinned window: the probe loads tens of gigabytes of weights
    onto the cards, and the copy engine must not address pageable host memory.
    """
    stager = PinnedWindow()
    for p in module.parameters(recurse=False):
        p.data = stager.to_device(p.data, dev)
    for name, b in module.named_buffers(recurse=False):
        module._buffers[name] = stager.to_device(b, dev)


def shard_layers(model: nn.Module, devices: list[torch.device]) -> None:
    """Split the model layer-wise over ``devices``: embeddings and everything before the
    layers on the first, contiguous groups of decoder layers on each, the final norm and
    the output head on the last (tied embeddings are copied). A pre-hook on every layer
    moves its inputs to its device. Returns the hook handles (remove them to undo the
    input moves)."""
    from .layerwise import find_layers, to_device

    path, layers = find_layers(model)
    head = model.get_output_embeddings()
    emb = model.get_input_embeddings()
    if head is not None and head.weight is emb.weight:
        # tied: the head gets its own copy on the last device (the probe never perturbs
        # the embeddings, so the copy cannot drift)
        head.weight = nn.Parameter(emb.weight.detach().clone())
    first, last = devices[0], devices[-1]
    in_layers = {id(m) for m in layers.modules()}
    for m in model.modules():
        if id(m) not in in_layers:  # the layers go to their own devices below
            _move_own(m, first)
    parent = model.get_submodule(path.rsplit(".", 1)[0]) if "." in path else model
    for tail in (head, getattr(parent, "norm", None)):
        if tail is not None:
            module_to_device(tail, last)
    n = len(devices)
    handles = []
    for k, dev in enumerate(devices):
        for layer in layers[len(layers) * k // n : len(layers) * (k + 1) // n]:
            module_to_device(layer, dev)
            handles.append(
                layer.register_forward_pre_hook(
                    lambda _m, args, kwargs, dev=dev: (
                        to_device(args, dev),
                        to_device(kwargs, dev),
                    ),
                    with_kwargs=True,
                )
            )
    return handles


def load_probe_model(src, dtype: torch.dtype, ex: Executor, log=print):
    """``(model, first device, sharded)`` of the probe. The model sits on the primary
    device when its weights fit there, otherwise it is split layer-wise across the
    executor's devices (``HI2_PROBE_SHARD=1`` forces the split, for tests)."""
    from transformers import AutoModelForCausalLM

    from .moe import unfuse_experts

    model = AutoModelForCausalLM.from_pretrained(src, dtype=dtype).eval()
    unfuse_experts(model)
    dev = ex.primary
    if dev.type == "cpu":
        return model, dev, False
    need = sum(p.numel() * p.element_size() for p in model.parameters())
    cap = torch.cuda.get_device_properties(dev).total_memory
    plan = "shard" if os.environ.get("HI2_PROBE_SHARD") == "1" else None
    plan = plan or plan_placement(need, cap, len(ex.devices))
    if plan == "single" or len(ex.devices) == 1:
        return module_to_device(model, dev), dev, False
    shard_layers(model, ex.devices)
    held: dict[str, int] = {}
    for p in model.parameters():
        held[str(p.device)] = held.get(str(p.device), 0) + p.numel() * p.element_size()
    log(
        "sensitivity probe: model split layer-wise, "
        + ", ".join(f"{d} {b / 2**30:.1f} GiB" for d, b in sorted(held.items()))
    )
    return model, dev, True


@torch.no_grad()
def probe_sensitivity(
    src,
    tables: dict[str, TensorTable],
    workdir,
    ids: torch.Tensor,
    cfg: Config,
    ex: Executor,
    ref_candidate: tuple[int, int] = REF_CANDIDATE,
    floor: float = 0.05,
    log=print,
    units: list[list[tuple[str, float]]] | None = None,
    dtype: torch.dtype = torch.float32,
    batch: int = 1,
    skip_prefix: bool = True,
) -> dict[str, float]:
    """Return ``{checkpoint tensor name: s_t}`` (normalised to median 1).

    ``units`` are groups of ``(name, weight)`` quantized together in one probe (MoE:
    all experts of a layer and projection, weighted by routed-token share, see
    :func:`hi2.moe.probe_units`); the default is one unit per tensor with weight 1. A
    unit's sensitivity is ``s_u = dKL / sum_t w_t rel_err_t``; the median and the floor
    apply to ``s_u``, and each member gets ``s_u * w_t``, so the unit's predicted KL is
    ``sum_t s_t rel_err_t``.

    ``skip_prefix`` replays the reference outputs of the decoder layers before a unit's
    first layer instead of recomputing them (they do not change), which gives the same
    logits bit for bit. ``batch`` sequences go through the model per forward (1 = the
    original per-sequence probe; larger is faster but rounds the logits differently)."""
    model, dev, sharded = load_probe_model(src, dtype, ex, log)
    fwd = {"use_cache": False} if sharded else {}  # no per-layer caches across devices
    resolve = make_resolver(weight_map(src))
    params = {}
    for name, p in model.named_parameters():
        ckpt = resolve(name)
        if ckpt in tables and p.ndim == 2:
            params[ckpt] = p
    ids = ids.to(dev)
    chunks = [ids[i : i + batch] for i in range(0, ids.shape[0], batch)]
    layers = _layers(model)
    replay: list[dict[int, object]] = [
        dict() for _ in chunks
    ]  # chunk -> layer -> output
    hooks = []
    if skip_prefix and layers is not None:
        for li, layer in enumerate(layers):

            def keep(_m, _a, out, li=li):
                replay[_current[0]][li] = out

            hooks.append(layer.register_forward_hook(keep))
    _current = [0]
    ref_bytes = (
        sum(c.numel() for c in chunks) * model.config.get_text_config().vocab_size * 2
    )
    on_dev = dev.type != "cpu" and ref_bytes < REF_ON_DEVICE_BYTES
    ref = []
    for ci, c in enumerate(chunks):
        _current[0] = ci
        lg = model(c, **fwd).logits.float()
        if not torch.isfinite(lg).all():
            raise SystemExit(
                f"sensitivity probe: non-finite logits in {dtype}; use float32"
            )
        r = torch.log_softmax(lg, -1).half()
        ref.append(r if on_dev else r.cpu())
        del lg
    for h in hooks:
        h.remove()

    def kl(first_layer: int) -> float:
        total, n = 0.0, 0
        start = first_layer if hooks else 0
        saved = []
        for ci, c in enumerate(chunks):
            for li in range(start):  # unchanged prefix: replay its outputs
                if ci == 0:
                    saved.append(layers[li])
                layers[li] = _Replay(saved[li], replay[ci][li])
            lq = torch.log_softmax(model(c, **fwd).logits.float(), -1)
            lr = ref[ci].to(lq.device).float()
            total += (lr.exp() * (lr - lq)).sum().item()
            n += lr.shape[0] * lr.shape[1]
        for li, layer in enumerate(saved):
            layers[li] = layer
        return total / n

    view = StatsView(workdir)
    near = copy.copy(cfg)
    near.rounding = "nearest"
    near.sequential = False
    units = units or [[(n, 1.0)] for n in tables]
    raw: list[tuple[list[tuple[str, float]], float, float]] = []
    for k, unit in enumerate(units):
        members = [(n, w) for n, w in unit if n in params]
        if not members:
            continue
        rel, saved = 0.0, []
        for name, wt in members:
            table = tables[name]
            cand = (
                ref_candidate
                if ref_candidate in table.candidates
                else table.candidates[0]
            )
            ci = table.candidates.index(cand)
            picks = torch.full((table.shape[0],), ci, dtype=torch.long)
            p = params[name]
            st = view.stats_state(name)
            stats = (
                ColumnStats.from_state(st, "cpu")
                if st is not None and st["n"] > 0
                else None
            )
            t = encode_with_picks(
                p.detach().float().cpu(), stats, table, picks, near, ex
            )
            rel += wt * table.sse[ci].sum().item() / max(table.energy, 1e-30)
            saved.append((p, p.data.clone()))
            p.data.copy_(dequantize(t, p.device).to(p.dtype))
        first = min((_layer_index(n) for n, _ in members), default=0)
        dkl = max(kl(first), 0.0)
        for p, orig in saved:
            p.data.copy_(orig)
        del saved
        raw.append((members, dkl, rel))
        if k % 20 == 0:
            what = (
                members[0][0]
                if len(members) == 1
                else f"{members[0][0]} (+{len(members) - 1})"
            )
            log(
                f"sensitivity {k + 1}/{len(units)}: {what} dKL={dkl:.2e} rel_err={rel:.3f}"
            )
    s_unit = [dkl / max(rel, 1e-12) for _, dkl, rel in raw]
    vals = sorted(s_unit)
    med = vals[len(vals) // 2] if vals else 1.0
    out = {}
    for (members, _, _), su in zip(raw, s_unit):
        for name, wt in members:
            out[name] = max(su / med, floor) * wt
    return out
