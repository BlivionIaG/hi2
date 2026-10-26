# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Mixture-of-experts support: per-expert Linear views, routed-token statistics,
per-expert allocation weights.

Checkpoints such as OLMoE store every expert as 2-D tensors
(``mlp.experts.E.{gate,up,down}_proj.weight``), which is also what the fork loads, but
recent transformers fuse them into 3-D parameters (``experts.gate_up_proj [E, 2I, H]``,
``experts.down_proj [E, H, I]``) when the model is built. :func:`unfuse_experts`
replaces such a module by a ModuleList of per-expert ``gate_proj`` / ``up_proj`` /
``down_proj`` Linears (views of the same storage) whose module paths are the checkpoint
names, with the same forward. Every expert Linear then gets the existing Linear hooks
and sees only the tokens routed to it; gate and up receive the very same tensor, so
they share one statistics accumulator and one Hessian.
"""

import statistics

import regex as re
import torch
from torch import nn

# ``<prefix>experts.<E>.<proj>`` (module path or checkpoint name, with or without .weight)
EXPERT_RE = re.compile(
    r"^(?P<pre>.*?\bexperts)\.(?P<e>\d+)\.(?P<proj>[^.]+)(?:\.weight)?$"
)
# Siblings of an expert that always receive the same input (gate/up; Mixtral w1/w3).
SHARED_INPUT = {"up_proj": "gate_proj", "w3": "w1"}


def expert_parts(name: str) -> tuple[str, int, str] | None:
    """``(group prefix, expert index, projection)`` or None for a non-expert name."""
    m = EXPERT_RE.match(name)
    if m is None:
        return None
    return m.group("pre"), int(m.group("e")), m.group("proj")


def group_key(name: str) -> str:
    """Name with the expert index replaced by ``*``: one key per layer and projection."""
    p = expert_parts(name)
    return name if p is None else f"{p[0]}.*.{p[2]}"


def sibling_leader(name: str) -> str | None:
    """For an expert ``up_proj`` (``w3``), the ``gate_proj`` (``w1``) it shares its input with."""
    p = expert_parts(name)
    if p is None or p[2] not in SHARED_INPUT:
        return None
    return f"{p[0]}.{p[1]}.{SHARED_INPUT[p[2]]}"


def _view_linear(w: torch.Tensor) -> nn.Linear:
    lin = nn.Linear(w.shape[1], w.shape[0], bias=False, device="meta")
    lin.weight = nn.Parameter(w, requires_grad=False)
    return lin


class ExpertMLP(nn.Module):
    def __init__(self, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor):
        super().__init__()
        self.gate_proj = _view_linear(gate)
        self.up_proj = _view_linear(up)
        self.down_proj = _view_linear(down)


class UnfusedExperts(nn.ModuleList):
    """Per-expert Linear view of fused 3-D experts, same forward as transformers'
    ``*Experts(hidden_states, top_k_index, top_k_weights)``."""

    def __init__(self, fused: nn.Module):
        gu = fused.gate_up_proj.data
        down = fused.down_proj.data
        e, two_i, h = gu.shape
        i = two_i // 2
        if two_i % 2 or tuple(down.shape) != (e, h, i):
            raise NotImplementedError(
                f"unsupported fused expert layout gate_up {tuple(gu.shape)} down {tuple(down.shape)}"
            )
        super().__init__(ExpertMLP(gu[x, :i], gu[x, i:], down[x]) for x in range(e))
        self.act_fn = fused.act_fn
        self.num_experts = e

    def forward(self, hidden_states, top_k_index, top_k_weights):
        out = torch.zeros_like(hidden_states)
        k = top_k_index.shape[-1]
        flat = top_k_index.reshape(-1)
        order = torch.argsort(flat, stable=True)  # token order kept inside an expert
        counts = torch.bincount(flat, minlength=len(self)).tolist()  # one host sync
        start = 0
        for e, mlp in enumerate(self):
            sel = order[start : start + counts[e]]
            start += counts[e]
            if sel.numel() == 0:
                continue
            tok, pos = sel // k, sel % k
            x = hidden_states[tok]  # one tensor for gate and up: shared statistics
            y = mlp.down_proj(self.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x))
            y = y * top_k_weights[tok, pos, None]
            out.index_add_(0, tok, y.to(out.dtype))
        return out


def is_fused_experts(mod: nn.Module) -> bool:
    gu = getattr(mod, "gate_up_proj", None)
    dn = getattr(mod, "down_proj", None)
    return (
        isinstance(gu, torch.Tensor)
        and isinstance(dn, torch.Tensor)
        and gu.ndim == 3
        and dn.ndim == 3
        and hasattr(mod, "act_fn")
        and not getattr(mod, "has_bias", False)
    )


def unfuse_experts(model: nn.Module) -> int:
    """Replace every fused-expert module of ``model`` in place; returns how many."""
    targets = [n for n, m in model.named_modules() if is_fused_experts(m)]
    for n in targets:
        model.set_submodule(n, UnfusedExperts(model.get_submodule(n)))
    return len(targets)


def check_unfused(model: nn.Module, src, resolve, log=print) -> None:
    """Compare the first expert Linear views with the checkpoint tensors they claim to
    be, so a wrong split (e.g. up before gate) fails loudly instead of silently."""
    from safetensors import safe_open

    from .checkpoint import weight_map

    wmap = weight_map(src)
    checked = 0
    for name, p in model.named_parameters():
        if expert_parts(name) is None or expert_parts(name)[1] != 0:
            continue
        ckpt = resolve(name)
        if ckpt is None:
            raise SystemExit(f"{name}: no checkpoint tensor (fused experts on disk?)")
        with safe_open(src / wmap[ckpt], "pt") as f:
            ref = f.get_tensor(ckpt).to(p.dtype)
        if ref.shape != p.shape or not torch.equal(ref, p.data.cpu()):
            raise SystemExit(f"{name}: unfused expert view differs from {ckpt}")
        checked += 1
        if checked >= 3:
            break
    log(f"unfused experts: {checked} views match the checkpoint")


# ---------------------------------------------------------------- statistics


def blend(own_sum, own_n: int, avg_sum, avg_n: int, need: int):
    """Fill an expert's accumulator up to ``need`` tokens with the layer average:
    ``own + (need - n) * avg / avg_n`` (``need`` tokens in total)."""
    fill = (need - own_n) / max(avg_n, 1)
    return own_sum + fill * avg_sum.to(own_sum.device), need


def coverage(counts: dict[str, int], fallback) -> dict:
    """Per layer and projection group: min / median / max routed tokens per expert
    and the experts that borrowed statistics (``fallback``: a set of names)."""
    by: dict[str, list[tuple[int, int, bool]]] = {}
    for name, n in counts.items():
        p = expert_parts(name)
        if p is not None:
            by.setdefault(group_key(name), []).append((p[1], n, name in fallback))
    out = {}

    def natural(key):  # layers.2 before layers.10
        return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", key)]

    for key in sorted(by, key=natural):
        rows = by[key]
        ns = [n for _, n, _ in rows]
        out[key] = {
            "experts": len(rows),
            "min": min(ns),
            "median": statistics.median(ns),
            "max": max(ns),
            "fallback": sorted(e for e, _, fb in rows if fb),
        }
    return out


# ---------------------------------------------------------------- allocation


def expert_weights(tables: dict, share_floor: float = 0.05) -> dict[str, float]:
    """Allocation multiplier per tensor: 1 for dense tensors, ``share_e / E`` for the
    E experts of one layer and projection, with ``share_e = n_e / mean(n)`` (routed
    tokens relative to a uniform router, floored at ``share_floor``).

    All experts of a group together weigh as much as one dense projection (each token
    sees a router-weighted average of its experts' errors), and inside the group an
    expert counts in proportion to the tokens it serves."""
    groups: dict[str, list[str]] = {}
    for name in tables:
        if expert_parts(name) is not None:
            groups.setdefault(group_key(name), []).append(name)
    out: dict[str, float] = {}
    for names in groups.values():
        ns = [tables[n].n_tokens for n in names]
        mean = sum(ns) / len(ns)
        for n, c in zip(names, ns):
            share = c / mean if mean > 0 else 1.0
            out[n] = max(share, share_floor) / len(names)
    return out


def probe_units(
    tables: dict, weights: dict[str, float]
) -> list[list[tuple[str, float]]]:
    """Units of the sensitivity probe: a dense tensor alone (weight 1), or all experts
    of a layer and projection together with their :func:`expert_weights`."""
    units: dict[str, list[tuple[str, float]]] = {}
    for name in tables:
        units.setdefault(group_key(name), []).append((name, weights.get(name, 1.0)))
    return list(units.values())
