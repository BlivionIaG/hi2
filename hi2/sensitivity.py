# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
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

import torch

from .calibrate import Config, TensorTable, encode_with_picks
from .checkpoint import make_resolver, table_stats_path, weight_map
from .executor import Executor
from .format import dequantize
from .stats import ColumnStats

REF_CANDIDATE = (2, 32)


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
) -> dict[str, float]:
    """Return ``{checkpoint tensor name: s_t}`` (normalised to median 1).

    ``units`` are groups of ``(name, weight)`` quantized together in one probe (MoE:
    all experts of a layer and projection, weighted by routed-token share, see
    :func:`hi2.moe.probe_units`); the default is one unit per tensor with weight 1. A
    unit's sensitivity is ``s_u = dKL / sum_t w_t rel_err_t``; the median and the floor
    apply to ``s_u``, and each member gets ``s_u * w_t``, so the unit's predicted KL is
    ``sum_t s_t rel_err_t``."""
    from transformers import AutoModelForCausalLM

    from .moe import unfuse_experts

    dev = ex.primary
    model = AutoModelForCausalLM.from_pretrained(src, dtype=dtype).eval()
    unfuse_experts(model)
    model = model.to(dev)
    resolve = make_resolver(weight_map(src))
    params = {}
    for name, p in model.named_parameters():
        ckpt = resolve(name)
        if ckpt in tables and p.ndim == 2:
            params[ckpt] = p
    ids = ids.to(dev)
    ref = []
    for i in range(ids.shape[0]):
        lg = model(ids[i : i + 1]).logits[0].float()
        if not torch.isfinite(lg).all():
            raise SystemExit(
                f"sensitivity probe: non-finite logits in {dtype}; use float32"
            )
        ref.append(torch.log_softmax(lg, -1).half().cpu())

    def kl() -> float:
        total, n = 0.0, 0
        for i in range(ids.shape[0]):
            lq = torch.log_softmax(model(ids[i : i + 1]).logits[0].float(), -1)
            lr = ref[i].to(dev).float()
            total += (lr.exp() * (lr - lq)).sum().item()
            n += lr.shape[0]
        return total / n

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
            sp = table_stats_path(workdir, name)
            stats = (
                ColumnStats.from_state(torch.load(sp), "cpu") if sp.exists() else None
            )
            t = encode_with_picks(
                p.detach().float().cpu(), stats, table, picks, near, ex
            )
            rel += wt * table.sse[ci].sum().item() / max(table.energy, 1e-30)
            saved.append((p, p.data.clone()))
            p.data.copy_(dequantize(t).to(p.dtype).to(dev))
        dkl = max(kl(), 0.0)
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
