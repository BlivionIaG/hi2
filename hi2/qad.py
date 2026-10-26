# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Layer-wise quantization-aware distillation (QAD), spec section 13.

After the PTQ pipeline has allocated every row's ``(nbits, B)`` (and computed the
smoothing vectors, rotation flags and Hessians), the decoder layers are trained one
at a time, like the sequential encode pass of spec 8.4:

* the student input of layer ``L`` is the output of the already quantized prefix,
  the target is the unquantized layer applied to the unquantized prefix's output;
  both sets of hidden states live in RAM, one layer is on the GPU;
* every eligible Linear of the layer is replaced by :class:`FakeQuantLinear`: fp32
  latent weights in the stored basis plus the per-block ladder parameters (in 8-bit
  units), with the row picks, the per-row fp16 constants, ``s`` and the Hadamard
  flags frozen. The forward is the format's exact decode (codes, 8-bit parameters,
  one fp16 rounding) with straight-through gradients, mapped back to the original
  basis so the layer runs unchanged;
* the loss is the normalised MSE of the layer output; AdamW, cosine schedule; the
  layer keeps whichever of the PTQ initialisation and the trained state has the
  lower loss on held-out calibration sequences.

The export takes the codes and the 8-bit parameters exactly as the last forward saw
them, so a QAD checkpoint is a normal v3.2 checkpoint (no format change).
"""

import copy
import gc
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn

from .calibrate import Config, TensorTable
from .format import (
    HI2Tensor,
    assemble,
    decode_params,
    dequantize,
    fwht_groups,
    ladder_dequantize,
)

# ---- configuration -----------------------------------------------------------


@dataclass
class QadConfig:
    steps: int = 300  # optimizer steps per layer (the most any layer gets)
    # Step budget: the run spends at most budget x steps x (trained layers) optimizer steps.
    # Below 1, each layer's allotment follows the gain the previous layers showed (layers
    # that improve little get fewer steps), between min_steps and steps.
    budget: float = 1.0
    # Early stop: a layer stops when one evaluation interval improves its held-out nMSE by
    # less than this fraction (0 = never), once it has run min_steps.
    min_gain: float = 0.0
    min_steps: int = 0
    eval_every: int = 0  # steps between held-out evaluations (0 = steps / evals)
    batch: int = 4  # sequences per optimizer step (gradient accumulation)
    # AdamW lr of the latent weights, in latent_units: with "step" the latent is stored in
    # units of its block's quantization step, so lr_w is a fraction of a step per update
    # and transfers across models; "abs" (lr ~1e-4) over-steps on Qwen3.5-4B early layers.
    lr_w: float = 0.01
    latent_units: str = "step"
    lr_p: float = 0.05  # AdamW lr of lo/step, in 8-bit levels per step
    samples: int = 256  # training sequences of seq_len tokens
    val: int = 8  # extra held-out calibration sequences (loss before/after, keep-best)
    # weights: latents start at the unquantized weights (step 0 = nearest rounding on the PTQ
    # ladders; measured slightly better on Qwen3.5-0.8B); ptq: at the GPTQ-compensated weights
    # (step 0 reproduces the PTQ checkpoint exactly). Keep-best guards both.
    init: str = "weights"
    layers: str | None = None  # e.g. "0-5,9"; None = every layer
    warmup: float = 0.05  # fraction of steps with linear warm-up
    keep_best: bool = True
    buffer_dtype: str = "auto"  # hidden-state buffers in RAM: auto | float32 | bfloat16
    seed: int = 0
    evals: int = 4  # held-out evaluations per layer (logged)
    # random: windows across the whole calibration text; balanced: equal windows per
    # calibration file; head: the first tokens
    sampling: str = "random"
    # ---- low-bit options (docs/lowbit.md); all off by default -------------
    # Train the layer's RMSNorm weights (fp32, AdamW lr_norm) together with its Linears;
    # they are written to the checkpoint in fp16 (the runtime dtype on RDNA).
    train_norms: bool = False
    lr_norm: float = 1e-4
    # For the last kl_layers layers, add kl_weight x KL(teacher || student) of the logits
    # (final norm + LM head applied to the layer output; exact at the last layer, a logit
    # lens before it) on kl_tokens random positions per sequence to the nMSE.
    kl_layers: int = 0
    kl_weight: float = 1.0
    kl_tokens: int = 256
    # Final logit-KL phase after the layer-wise one (hi2.finalkl): final_kl optimizer steps
    # of KL(teacher || student) over the whole network, training only the per-block ladder
    # bytes and the RMSNorm weights (codes frozen); 0 = off.
    final_kl: int = 0
    final_batch: int = 2  # sequences per step
    # ladder bytes, in 8-bit levels per step: 0.02 is the Qwen3.5-4B value (0.05 was best on
    # Qwen3.5-0.8B but raised the 4B held-out KL above its start)
    final_lr_p: float = 0.02
    final_lr_norm: float = 5e-5
    final_norms: bool = True
    final_bytes: bool = True
    final_tokens: int = 0  # positions per sequence in the loss (0 = all)
    final_evals: int = 4


# Fields of the final logit-KL phase: never part of the layer-wise fingerprint (they do
# not change the layer artifacts), fingerprinted separately by hi2.finalkl.
FINAL_FIELDS = {f for f in QadConfig.__dataclass_fields__ if f.startswith("final_")}


# Chosen on Qwen3.5-0.8B at 3.0 bpw (docs/qad.md); everything else default.
# fast = half the steps spent where layers still gain, with early stopping (better than a
# uniform 100 steps at the same time); no reduced budget stayed within noise of default.
PRESETS: dict[str, dict] = {
    "fast": dict(budget=0.5, min_gain=0.01, min_steps=75, eval_every=50),
    "default": dict(),
    "thorough": dict(steps=600, batch=8, samples=512),
}


def make_config(preset: str = "default", overrides: dict | None = None) -> QadConfig:
    if preset not in PRESETS:
        raise SystemExit(f"unknown QAD preset {preset!r} (choose {list(PRESETS)})")
    qc = QadConfig(**PRESETS[preset])
    for k, v in (overrides or {}).items():
        if v is None:
            continue
        if not hasattr(qc, k):
            raise SystemExit(
                f"unknown QAD option {k!r}; valid: {list(asdict(QadConfig()))}"
            )
        cur = getattr(qc, k)
        if isinstance(cur, bool):
            v = (
                v
                if isinstance(v, bool)
                else str(v).lower() in ("1", "true", "yes", "on")
            )
        elif cur is not None:
            v = type(cur)(v)
        setattr(qc, k, v)
    if qc.sampling not in ("random", "balanced", "head"):
        raise SystemExit("QAD sampling must be 'random', 'balanced' or 'head'")
    if qc.latent_units not in ("abs", "step"):
        raise SystemExit("QAD latent_units must be 'abs' or 'step'")
    if qc.init not in ("ptq", "weights"):
        raise SystemExit("QAD init must be 'ptq' or 'weights'")
    if qc.samples < 1 or qc.steps < 0 or qc.batch < 1 or qc.val < 1:
        raise SystemExit("QAD samples/batch/val must be >= 1 and steps >= 0")
    return qc


# Measured QAD step time on one Radeon PRO V620 (4 x 2048 tokens per step): 0.35 s for the
# 21.5M quantized parameters per layer of Qwen3.5-0.8B, about 0.95 s for Qwen3.5-4B.
_STEP_S, _STEP_P, _STEP_EXP, _OVERHEAD = 0.35, 21.5e6, 0.6, 1.15


def estimate_seconds(
    src, qc: "QadConfig", seq_len: int, include=None, exclude=None, gpu: bool = True
) -> tuple[float, int]:
    """Rough QAD wall time (seconds) and the number of decoder layers, from the checkpoint's
    tensor shapes (no weights read). Scales the measured step time by parameters per layer
    (power 0.6) and tokens per step; a CPU is assumed 30x slower."""
    import json as _json
    from pathlib import Path as _Path

    import regex as re
    from safetensors import safe_open

    from .checkpoint import DEFAULT_EXCLUDE, is_eligible, weight_map

    src = _Path(src)
    cfg = _json.loads((src / "config.json").read_text())
    n_layers = int(cfg.get("text_config", cfg).get("num_hidden_layers", 0) or 0)
    inc = re.compile(include) if include else None
    exc = re.compile(exclude or DEFAULT_EXCLUDE)
    total = 0
    wmap = weight_map(src)
    for shard in sorted(set(wmap.values())):
        with safe_open(src / shard, "pt") as f:
            for name in f.keys():  # noqa: SIM118
                if "layers." not in name or name.startswith("mtp"):
                    continue
                sl = f.get_slice(name)
                shape = sl.get_shape()
                dt = (
                    torch.float16
                    if sl.get_dtype() in ("F16", "BF16", "F32")
                    else torch.int8
                )
                if is_eligible(name, shape, dt, inc, exc):
                    total += shape[0] * shape[1]
    if n_layers <= 0 or total == 0:
        return 0.0, n_layers
    per_layer = total / n_layers
    step = _STEP_S * (per_layer / _STEP_P) ** _STEP_EXP * (qc.batch * seq_len / 8192)
    steps = qc.steps * min(qc.budget, 1.0) * len(parse_layers(qc.layers, n_layers))
    sec = step * steps * _OVERHEAD
    return (sec if gpu else 30 * sec), n_layers


def parse_layers(spec: str | None, n_layers: int) -> set[int]:
    """``"0-3,7"`` -> {0, 1, 2, 3, 7}; None -> every layer."""
    if spec is None or spec.strip() in ("", "all"):
        return set(range(n_layers))
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    bad = [i for i in out if i < 0 or i >= n_layers]
    if bad:
        raise SystemExit(
            f"--qad-layers {spec!r}: no layer {bad[0]} (model has {n_layers})"
        )
    return out


# ---- fake-quantised Linear ---------------------------------------------------


def _ste(value: torch.Tensor, surrogate: torch.Tensor) -> torch.Tensor:
    """``value`` exactly in the forward, the gradient of ``surrogate`` in the backward
    (``s - s.detach()`` is exactly zero in floating point)."""
    return value + (surrogate - surrogate.detach())


class _Group(nn.Module):
    """Rows sharing one ``(nbits, B)``: trainable 8-bit-unit ladder parameters."""

    def __init__(self, rows, nbits, block, lo_u8, step_u8, consts):
        super().__init__()
        self.nbits, self.block = int(nbits), int(block)
        self.register_buffer("rows", rows.to(torch.int64))
        self.register_buffer("consts", consts.to(torch.float16))  # [r, 3], frozen
        # continuous byte values; the forward rounds them (STE) to the stored bytes
        self.lo_u = nn.Parameter(lo_u8.float())
        self.step_u = nn.Parameter(step_u8.float())

    def bytes(self):
        return (
            torch.round(self.lo_u.detach()).clamp(0, 255),
            torch.round(self.step_u.detach()).clamp(0, 255),
        )

    @torch.no_grad()
    def project(self):
        self.lo_u.clamp_(0, 255)
        self.step_u.clamp_(0, 255)


class FakeQuantLinear(nn.Module):
    """The exact HI2 decode of one tensor as a differentiable function of fp32 latent
    weights (stored basis ``H (W s)``) and the per-block 8-bit ladder parameters.

    Frozen: per-row ``(nbits, B)``, the per-row fp16 constants, ``s``, the flags.
    ``forward()`` returns the weight in the original basis, ``export()`` the
    :class:`HI2Tensor` whose ``dequantize_stored`` equals the forward's stored values
    bit for bit.
    """

    def __init__(
        self,
        w: torch.Tensor,
        t0: HI2Tensor,
        comp: torch.Tensor | None = None,
        init: str = "ptq",
        units: str = "abs",
    ):
        super().__init__()
        r, k = t0.shape
        self.shape = (r, k)
        self.register_buffer("had", t0.had.clone())
        cs = t0.col_scale
        self.register_buffer(
            "col_scale",
            cs.clone() if cs is not None else torch.ones(0, dtype=torch.float16),
        )
        self.has_scale = cs is not None
        self.register_buffer("row_nbits", t0.row_nbits.clone())
        self.register_buffer("row_block", t0.row_block.clone())
        if init == "weights" or comp is None:
            v = w.float()
            if cs is not None:
                v = v * cs.float()
            v = fwht_groups(v, t0.had) if bool(t0.had.any()) else v
        else:
            v = comp.float()
        groups = []
        nb = t0.row_nbits.to(torch.int64)
        bl = t0.row_block.to(torch.int64)
        for nbits in sorted(set(nb.tolist())):
            for bs in sorted(set(bl.tolist())):
                rows = ((nb == nbits) & (bl == bs)).nonzero().flatten()
                if rows.numel() == 0:
                    continue
                nblk = k // bs
                pidx = (
                    t0.param_off[rows].to(torch.int64).unsqueeze(1) + torch.arange(nblk)
                ).unsqueeze(-1) * 2 + torch.arange(2)
                p = t0.blk_params[pidx]
                groups.append(
                    _Group(rows, nbits, bs, p[..., 0], p[..., 1], t0.row_consts[rows])
                )
        self.groups = nn.ModuleList(groups)
        perm = torch.cat([g.rows for g in groups])
        self.register_buffer("inv_perm", torch.argsort(perm))
        # units="step": the latent is stored in units of its block's initial quantization
        # step, so the latent learning rate means "fraction of a step per update" and
        # transfers across models and bit widths; "abs": plain stored-basis values.
        self.step_units = units == "step"
        unit = torch.ones(r, k)
        if self.step_units:
            for g in groups:
                c = g.consts
                _, st = decode_params(*g.bytes(), c[:, 0:1], c[:, 1:2], c[:, 2:3])
                st = st.float()
                nz = (st > 0).float()
                row_mean = (st * nz).sum(1, keepdim=True) / nz.sum(
                    1, keepdim=True
                ).clamp_min(1)
                st = torch.where(st > 0, st, row_mean)
                st = torch.where(st > 0, st, torch.ones_like(st))
                unit[g.rows] = st.repeat_interleave(g.block, dim=1)
        self.register_buffer("unit", unit if self.step_units else torch.ones(0))
        self.latent = nn.Parameter(
            (v / unit).contiguous() if self.step_units else v.clone().contiguous()
        )

    def param_groups(self):
        return [self.latent], [p for g in self.groups for p in (g.lo_u, g.step_u)]

    def project(self):
        for g in self.groups:
            g.project()

    def _group(self, g: _Group, grad: bool):
        """Stored-basis values of the group's rows ``[r, K]`` (with the STE graph when
        ``grad``) plus the exact codes and bytes."""
        k = self.shape[1]
        nblk, n = k // g.block, 2**g.nbits
        lo_b, st_b = g.bytes()
        c = g.consts
        lo_q, st_q = decode_params(lo_b, st_b, c[:, 0:1], c[:, 1:2], c[:, 2:3])
        lo_qf, st_qf = lo_q.float(), st_q.float()
        v = self.latent[g.rows]
        if self.step_units:
            v = v * self.unit[g.rows]
        v = v.reshape(-1, nblk, g.block)
        zero = st_qf == 0
        if grad:
            lo_c = g.lo_u * c[:, 1:2].float() + c[:, 0:1].float()
            st_c = g.step_u * c[:, 2:3].float()
            lo_h = _ste(lo_qf, lo_c)
            st_h = _ste(st_qf, st_c)
        else:
            v = v.detach()
            lo_h, st_h = lo_qf, st_qf
        safe = torch.where(zero, torch.ones_like(st_h), st_h)
        z = (v - lo_h.unsqueeze(-1)) / safe.unsqueeze(-1)
        with torch.no_grad():
            q = torch.round(z).clamp(0, n - 1)
            q = torch.where(zero.unsqueeze(-1), torch.zeros_like(q), q)
            val = ladder_dequantize(
                q, lo_qf, st_qf
            ).float()  # float64 inside, one fp16 rounding
        if grad:
            zc = z.clamp(0, n - 1)
            y = lo_h.unsqueeze(-1) + _ste(q, zc) * st_h.unsqueeze(-1)
            val = _ste(val, y)
        return val.reshape(-1, k), q, lo_b, st_b

    def stored(self, grad: bool = True) -> torch.Tensor:
        parts = [self._group(g, grad)[0] for g in self.groups]
        return torch.cat(parts)[self.inv_perm]

    def forward(self) -> torch.Tensor:
        w = self.stored(True)
        if bool(self.had.any()):
            w = fwht_groups(w, self.had)
        if self.has_scale:
            w = w / self.col_scale.float()
        return w

    @torch.no_grad()
    def export(self) -> HI2Tensor:
        pieces = []
        k = self.shape[1]
        for g in self.groups:
            _, q, lo_b, st_b = self._group(g, False)
            params = torch.stack([lo_b, st_b], dim=-1).to(torch.uint8)
            pieces.append(
                (
                    g.rows.cpu(),
                    q.reshape(-1, k).to(torch.uint8).cpu(),
                    params.cpu(),
                    g.consts.cpu(),
                )
            )
        return assemble(
            pieces,
            self.row_nbits.cpu(),
            self.row_block.cpu(),
            self.had.cpu(),
            k,
            self.col_scale.cpu() if self.has_scale else None,
        )


# ---- one layer ---------------------------------------------------------------


def _nmse(out: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    return ((out - tgt) ** 2).mean() / (tgt**2).mean().clamp_min(1e-30)


def _out(o):
    return o[0] if isinstance(o, tuple) else o


def is_norm_param(name: str, p: torch.Tensor) -> bool:
    """RMSNorm/LayerNorm weights (``input_layernorm.weight``, ``q_norm.weight``,
    ``norm.weight``, ...): 1-D parameters of a module whose name contains ``norm``."""
    parts = name.split(".")
    return (
        p.ndim == 1 and parts[-1] == "weight" and len(parts) > 1 and "norm" in parts[-2]
    )


def head_kl(head, out: torch.Tensor, tgt: torch.Tensor, pos) -> torch.Tensor:
    """Mean KL(teacher || student) of the logits ``lm_head(norm(h))`` at the positions
    ``pos`` of ``[1, T, H]`` hidden states (teacher ``tgt``, student ``out``)."""
    norm, lm = head
    with torch.no_grad():
        lt = torch.log_softmax(lm(norm(tgt[0, pos])).float(), -1)
    ls = torch.log_softmax(lm(norm(out[0, pos])).float(), -1)
    return (lt.exp() * (lt - ls)).sum(-1).mean()


def train_layer(
    layer: nn.Module,
    fq: dict[str, FakeQuantLinear],
    xs: list[torch.Tensor],
    ys: list[torch.Tensor],
    train_idx: list[int],
    val_idx: list[int],
    kwargs: dict,
    qc: QadConfig,
    dev: torch.device,
    log=print,
    tag: str = "",
    fallback: dict[str, FakeQuantLinear] | None = None,
    steps: int | None = None,
    head: tuple | None = None,
) -> dict:
    """Train the fake-quantised Linears ``fq`` (submodule name -> module) of ``layer``
    so that ``layer(xs[i]) ~ ys[i]``. Returns loss/time info; ``fq`` is left at the
    best state (trained, or the initialisation when ``keep_best`` and it was better).
    ``fallback`` (same structure, e.g. the PTQ encoding) is scored on the held-out
    sequences too; with ``keep_best`` its state is taken when it beats training, and
    ``val_before`` is its loss.

    With ``qc.train_norms`` the layer's norm weights train too (``info["norms"]`` holds
    the kept values, empty when the PTQ fallback wins); ``head`` = (final norm, LM head)
    adds ``qc.kl_weight`` x the logit KL on ``qc.kl_tokens`` positions to the loss."""
    from torch.func import functional_call

    from .layerwise import to_device

    t0 = time.time()
    acc = getattr(torch, "accelerator", None)
    on_gpu = (
        dev.type != "cpu"
        and acc is not None
        and hasattr(acc, "reset_peak_memory_stats")
    )
    if on_gpu:
        acc.reset_peak_memory_stats()
    layer = copy.deepcopy(layer).to(dev)
    for p in layer.parameters():
        p.requires_grad_(False)
    for m in fq.values():
        m.to(dev)
    kw = to_device(kwargs, dev)
    norm_params = {}
    if qc.train_norms:
        for n, p in layer.named_parameters():
            if is_norm_param(n, p):
                p.data = p.data.float()
                p.requires_grad_(True)
                norm_params[n] = p
    hd = None
    if head is not None:
        hd = tuple(copy.deepcopy(m).to(dev).float() for m in head)
        for m in hd:
            for p in m.parameters():
                p.requires_grad_(False)
        kg = torch.Generator().manual_seed(qc.seed + 1)
        n_tok = xs[0].shape[-2]
        k_tok = min(qc.kl_tokens, n_tok) if qc.kl_tokens > 0 else n_tok
        val_pos = {i: torch.randperm(n_tok, generator=kg)[:k_tok] for i in val_idx}

    def loss_fn(out, i, pos=None):
        tgt = ys[i].to(dev).float()
        loss = _nmse(out, tgt)
        if hd is not None:
            if pos is None:
                pos = torch.randperm(n_tok, generator=kg)[:k_tok]
            loss = loss + qc.kl_weight * head_kl(hd, out, tgt, pos.to(dev))
        return loss

    def weights(mods=None):
        return {f"{n}.weight": m() for n, m in (mods or fq).items()}

    def run(wd, i):
        return _out(functional_call(layer, wd, (xs[i].to(dev).float(),), kw))

    @torch.no_grad()
    def val_loss(mods=None) -> float:
        wd = {k: v.detach() for k, v in weights(mods).items()}
        return sum(
            loss_fn(run(wd, i), i, val_pos[i] if hd is not None else None).item()
            for i in val_idx
        ) / max(len(val_idx), 1)

    def snapshot():
        st = {
            n: {k: v.detach().clone() for k, v in m.state_dict().items()}
            for n, m in fq.items()
        }
        st["__norms__"] = {n: p.detach().clone() for n, p in norm_params.items()}
        return st

    def restore(st):
        for n, m in fq.items():
            m.load_state_dict(st[n])
        for n, p in norm_params.items():
            p.data.copy_(st["__norms__"][n])

    loss0 = val_loss()
    loss_fb = None
    if fallback is not None:
        for m in fallback.values():
            m.to(dev)
        loss_fb = val_loss(fallback)
    best, best_state = loss0, snapshot() if qc.keep_best else None
    lat, prm = [], []
    for m in fq.values():
        a, b = m.param_groups()
        lat += a
        prm += b
    groups = [{"params": lat, "lr": qc.lr_w}, {"params": prm, "lr": qc.lr_p}]
    if norm_params:
        groups.append({"params": list(norm_params.values()), "lr": qc.lr_norm})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    n_steps = qc.steps if steps is None else steps  # this layer's allotment
    warm = max(1, int(qc.warmup * n_steps))

    def lr_scale(s):
        if s < warm:
            return (s + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, n_steps - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)
    g = torch.Generator().manual_seed(qc.seed)
    order: list[int] = []
    every = qc.eval_every or max(1, n_steps // max(1, qc.evals))
    trace = [(0, None, loss0)]
    train_loss = float("nan")
    spent, stopped = 0, False
    for step in range(n_steps):
        spent = step + 1
        if len(order) < qc.batch:
            order += [
                train_idx[j]
                for j in torch.randperm(len(train_idx), generator=g).tolist()
            ]
        batch, order = order[: qc.batch], order[qc.batch :]
        w = weights()
        wd = {k: v.detach().requires_grad_(True) for k, v in w.items()}
        tot = 0.0
        for i in batch:
            loss = loss_fn(run(wd, i), i) / len(batch)
            loss.backward()
            tot += loss.item()
        torch.autograd.backward(list(w.values()), [wd[k].grad for k in w])
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        for m in fq.values():
            m.project()
        train_loss = tot
        if (step + 1) % every == 0 or step + 1 == n_steps:
            prev = trace[-1][2]
            vl = val_loss()
            trace.append((step + 1, tot, vl))
            if qc.keep_best and vl < best:
                best, best_state = vl, snapshot()
            log(f"  {tag} step {step + 1}/{n_steps} train {tot:.4e} val {vl:.4e}")
            if (
                qc.min_gain > 0
                and step + 1 >= qc.min_steps
                and step + 1 < n_steps
                and prev - vl < qc.min_gain * prev
            ):
                stopped = True
                log(
                    f"  {tag} early stop at step {step + 1}: gain under {qc.min_gain:.1%}"
                )
                break
    final = trace[-1][2]
    kept = "trained"
    if qc.keep_best and best_state is not None and (best < final or n_steps == 0):
        restore(best_state)
        kept = "init" if best == loss0 else "best"
    after = min(best, final) if qc.keep_best else final
    norms = {n: p.detach().cpu().clone() for n, p in norm_params.items()}
    if qc.keep_best and loss_fb is not None and loss_fb <= after:
        for n in list(fq):  # the caller exports from fq: hand it the fallback modules
            fq[n] = fallback[n]
        kept, after, norms = "ptq", loss_fb, {}
    out = {
        "val_before": loss0 if loss_fb is None else loss_fb,
        "val_init": loss0,
        "val_after": after,
        "val_final": final,
        "train_last": train_loss,
        "kept": kept,
        "steps": spent,
        "allotted": n_steps,
        "early_stop": stopped,
        "seconds": time.time() - t0,
        "gpu_peak_gib": acc.max_memory_allocated() / 2**30 if on_gpu else None,
        "trace": trace,
    }
    if qc.train_norms:
        out["norms"] = norms
    return out


# ---- whole model ---------------------------------------------------------------


# Fields added after the first release, with the value that reproduces the old
# behaviour: at that value they are left out of the fingerprint, so per-layer artifacts
# written before the field existed are still reused.
LEGACY_DEFAULTS = {
    "latent_units": "abs",
    "budget": 1.0,
    "min_gain": 0.0,
    "min_steps": 0,
    "eval_every": 0,
    "train_norms": False,
    "lr_norm": 1e-4,
    "kl_layers": 0,
    "kl_weight": 1.0,
    "kl_tokens": 256,
}


def qad_meta(qc: QadConfig) -> dict:
    """``QadConfig`` for the checkpoint's ``quantization_config``: the low-bit options
    only when they differ from their default (a default run writes the same config)."""
    d = QadConfig()
    new = set(FINAL_FIELDS) | {
        "train_norms",
        "lr_norm",
        "kl_layers",
        "kl_weight",
        "kl_tokens",
    }
    return {k: v for k, v in asdict(qc).items() if k not in new or v != getattr(d, k)}


def qad_fingerprint_fields(qc: QadConfig) -> dict:
    return {
        k: v
        for k, v in asdict(qc).items()
        if LEGACY_DEFAULTS.get(k, object()) != v and k not in FINAL_FIELDS
    }


def fingerprint(obj) -> str:
    return hashlib.sha1(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()


def _avail_ram() -> int:
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return 1 << 62


def buffer_dtype(qc: QadConfig, n_seq: int, seq_len: int, hidden: int, log=print):
    """fp32 hidden-state buffers if they (three sets plus one transient) fit in half the free RAM, else bf16."""
    need = 4 * n_seq * seq_len * hidden * 4  # three buffers + one transient output set
    if qc.buffer_dtype != "auto":
        return getattr(torch, qc.buffer_dtype)
    if need <= 0.5 * _avail_ram():
        return torch.float32
    if need / 2 <= 0.8 * _avail_ram():
        log(
            f"QAD: {need / 2**30:.1f} GiB of fp32 hidden states do not fit; using bf16 buffers"
        )
        return torch.bfloat16
    raise SystemExit(
        f"QAD needs about {need / 2**31:.1f} GiB of RAM for hidden states even in bf16; "
        "lower --qad-samples or --seq-len"
    )


def _propagate(layer, ex, src, kwargs, dst) -> None:
    """``dst[i] = layer(src[i])`` for every sequence, written in place (``dst`` may be
    ``src``: each output is computed before it is stored)."""
    from .layerwise import _layer_pass

    res = _layer_pass(layer, ex.devices, len(src), src, kwargs, {}, {}, False)
    for o, *_ in res:
        for i, v in o.items():
            dst[i].copy_(v)
    del res


def _buffers(hidden, dtype, n_sets: int = 3):
    """``n_sets`` preallocated ``[n, 1, T, H]`` buffers (as lists of per-sequence views),
    each initialised with ``hidden``. They are reused for every layer, so the RAM held by
    hidden states is constant over the run."""
    shape = (len(hidden), *hidden[0].shape)
    out = []
    for _ in range(n_sets):
        buf = torch.empty(shape, dtype=dtype)
        for i, h in enumerate(hidden):
            buf[i].copy_(h)
        out.append(list(buf.unbind(0)))
    return out


class StepBudget:
    """Spreads ``budget x steps x n_layers`` optimizer steps over the trained layers, in
    order. Each layer gets the remaining budget per remaining layer, scaled by how much
    the previous layer gained relative to the mean gain so far (clamped to 0.5x..2x,
    the first layer 2x), within ``[min_steps, steps]``. Steps a layer leaves unused
    (early stop) go to the later layers. ``budget >= 1`` gives every layer ``steps``."""

    def __init__(self, qc: QadConfig, n_layers: int):
        self.cap = qc.steps
        self.floor = min(max(qc.min_steps, 1), qc.steps)
        self.full = qc.budget >= 1
        self.n = n_layers
        self.total = (
            qc.steps * n_layers if self.full else round(qc.budget * qc.steps * n_layers)
        )
        self.left, self.spent = n_layers, 0
        self.gains: list[float] = []
        self.step_sec: list[float] = []

    def allot(self) -> int:
        if self.full:
            return self.cap
        base = (self.total - self.spent) / max(1, self.left)
        w = 2.0
        if self.gains:
            mean = sum(self.gains) / len(self.gains)
            w = min(2.0, max(0.5, self.gains[-1] / mean)) if mean > 0 else 0.5
        return int(max(self.floor, min(self.cap, round(base * w))))

    def record(self, info: dict) -> None:
        self.left -= 1
        steps = info.get("steps", self.cap)
        self.spent += steps
        if "val_before" in info:
            before = max(info["val_before"], 1e-30)
            self.gains.append(max(0.0, 1 - info["val_after"] / before))
        if steps and info.get("seconds"):
            self.step_sec.append(info["seconds"] / steps)

    def summary(self) -> str:
        nominal = self.cap * self.n
        sec = sum(self.step_sec) / len(self.step_sec) if self.step_sec else 0.0
        saved = max(0, nominal - self.spent) * sec
        return (
            f"QAD steps spent {self.spent} of {nominal} ({100 * self.spent / max(nominal, 1):.0f} %), "
            f"saved about {saved / 60:.0f} min"
        )


def split_indices(n: int, qc: QadConfig) -> tuple[list[int], list[int]]:
    """Training and held-out sequence indices; the held-out ones are spread over the set
    (with random sampling the windows are sorted by position, so the tail would be one
    domain)."""
    val_idx = sorted(range(n - 1, -1, -max(1, n // qc.val))[: qc.val])
    train_idx = [i for i in range(n) if i not in set(val_idx)]
    return train_idx, val_idx


@dataclass
class QadResult:
    tensors: dict = field(default_factory=dict)  # checkpoint name -> HI2Tensor
    layers: list = field(default_factory=list)  # per-layer info
    dense: dict = field(default_factory=dict)  # checkpoint name -> trained dense tensor


def final_norm(model: nn.Module, layers_path: str) -> nn.Module:
    """The decoder's final norm (sibling ``norm`` of the layer list)."""
    base = layers_path.rsplit(".", 1)[0]
    for name in (f"{base}.norm", f"{base}.final_layernorm", f"{base}.ln_f"):
        try:
            return model.get_submodule(name)
        except AttributeError:
            continue
    raise SystemExit(f"cannot find the final norm next to {layers_path}")


def qad_layerwise(
    model: nn.Module,
    ids: torch.Tensor,
    tables: dict[str, TensorTable],
    picks: dict[str, torch.Tensor],
    cfg: Config,
    qc: QadConfig,
    ex,
    workdir: Path,
    include,
    exclude,
    resolve,
    fp: str,
    limit_layers: int | None = None,
    log=print,
    qdir: Path | None = None,
) -> QadResult:
    """Train every selected decoder layer in order (see the module docstring); layers
    not selected keep their PTQ encoding. Per-layer results are written to
    ``qdir/layerNNN.pt`` (default ``workdir/qad``) and reused on a re-run with the same
    fingerprint. The PTQ initialisation reads the statistics and Hessians through
    :class:`hi2.cache.StatsView` (the persistent cache or the workdir) and encodes all
    Linears of a layer in one batched :func:`hi2.calibrate.encode_many` call."""
    from .cache import StatsView
    from .calibrate import EncodeRequest, encode_many
    from .checkpoint import is_eligible
    from .layerwise import capture_inputs, find_layers
    from .stats import ColumnStats

    path, layers = find_layers(model)
    prefix = path + "."
    n_layers = len(layers) if limit_layers is None else min(limit_layers, len(layers))
    train_set = parse_layers(qc.layers, len(layers))
    hidden, kwargs = capture_inputs(model, layers, ids)
    n = len(hidden)
    if n < qc.val + 1:
        raise SystemExit(f"QAD needs more than {qc.val} sequences, got {n}")
    dt = buffer_dtype(qc, n, hidden[0].shape[-2], hidden[0].shape[-1], log)
    xt, xs, yt = _buffers(hidden, dt)  # teacher input, student input, teacher target
    del hidden
    train_idx, val_idx = split_indices(n, qc)
    qdir = qdir or workdir / "qad"
    view = StatsView(workdir)
    qdir.mkdir(parents=True, exist_ok=True)
    res = QadResult()
    t_start, trained_time, trained_n = time.time(), 0.0, 0
    todo = sorted(i for i in train_set if i < n_layers)
    budget = StepBudget(qc, len(todo))
    head = None
    if qc.kl_layers > 0:
        head = (final_norm(model, path), model.get_output_embeddings())
    log(
        f"QAD: {n_layers} layers ({len(todo)} trained), {len(train_idx)} train + {len(val_idx)} held-out "
        f"sequences of {ids.shape[1]} tokens, {qc.steps} steps x {qc.batch} seqs per layer, "
        f"lr {qc.lr_w:g}/{qc.lr_p:g}, init {qc.init}, buffers {str(dt).removeprefix('torch.')}, "
        f"training on {ex.primary}"
        + (
            f", step budget {qc.budget:g} (early stop under {qc.min_gain:.1%} per {qc.eval_every or 'eval'} steps, "
            f"at least {budget.floor})"
            if not budget.full or qc.min_gain > 0
            else ""
        )
    )
    for li in range(n_layers):
        layer = layers[li]
        linears = {}
        for name, mod in layer.named_modules():
            full = f"{prefix}{li}.{name}.weight"
            if isinstance(mod, nn.Linear) and is_eligible(
                full, mod.weight.shape, mod.weight.dtype, include, exclude
            ):
                ckpt = resolve(full) if resolve else full
                if ckpt is not None and ckpt in tables:
                    linears[name] = ckpt
        _propagate(layer, ex, xt, kwargs, yt)
        art = qdir / f"layer{li:03d}.pt"
        rec = torch.load(art, weights_only=False) if art.exists() else None
        norms: dict = {}
        if rec is not None and rec.get("fingerprint") == fp:
            tensors, info = rec["tensors"], rec["info"]
            norms = rec.get("norms", {})
            log(f"QAD layer {li}: reusing {art.name}")
            if li in todo:
                budget.record(info)
        else:
            tl = time.time()
            reqs, loaded = [], {}
            for name, ckpt in linears.items():
                w = layer.get_submodule(name).weight.detach().float().cpu()
                sid = view.sigma_id(ckpt)
                if sid is not None and sid not in loaded:  # one load per shared input
                    loaded[sid] = view.sigma(ckpt)
                sigma = loaded.get(sid) if sid is not None else None
                st = view.stats_state(ckpt)
                stats = ColumnStats.from_state(st, "cpu") if st is not None else None
                if cfg.rounding == "gptq" and sigma is None:
                    log(
                        f"warning: {ckpt} has no Hessian; nearest-rounding initialisation"
                    )
                reqs.append(
                    EncodeRequest(
                        w, stats, tables[ckpt], picks[ckpt], sigma, sid, comp_out={}
                    )
                )
            encoded = encode_many(reqs, cfg, ex) if reqs else []
            inits = dict(zip(linears, encoded))
            comps = {name: r.comp_out["comp"] for name, r in zip(linears, reqs)}
            del reqs, loaded
            if li in train_set and linears:
                fq = {
                    name: FakeQuantLinear(
                        layer.get_submodule(name).weight.detach().float().cpu(),
                        inits[name],
                        comps[name],
                        qc.init,
                        qc.latent_units,
                    )
                    for name in linears
                }
                fb = None
                if (
                    qc.init != "ptq"
                ):  # also score the PTQ encoding; kept if training loses
                    fb = {
                        name: FakeQuantLinear(
                            layer.get_submodule(name).weight.detach().float().cpu(),
                            inits[name],
                            comps[name],
                            "ptq",
                        )
                        for name in linears
                    }
                info = train_layer(
                    layer,
                    fq,
                    xs,
                    yt,
                    train_idx,
                    val_idx,
                    kwargs,
                    qc,
                    ex.primary,
                    log,
                    f"L{li}",
                    fb,
                    budget.allot(),
                    head if li >= n_layers - qc.kl_layers else None,
                )
                norms = info.pop("norms", {})
                tensors = {linears[nm]: m.export() for nm, m in fq.items()}
                del fq, fb
                if ex.primary.type == "cuda":
                    torch.accelerator.empty_cache()
                trained_time += time.time() - tl
                trained_n += 1
                budget.record(info)
            else:
                tensors = {linears[nm]: t for nm, t in inits.items()}
                info = {"kept": "ptq", "seconds": time.time() - tl}
            rec = {"fingerprint": fp, "tensors": tensors, "info": info}
            if norms:
                rec["norms"] = norms
            torch.save(rec, art)
            left = len([i for i in todo if i > li])
            eta = (trained_time / max(trained_n, 1)) * left
            if "val_before" in info:
                gain = 1 - info["val_after"] / max(info["val_before"], 1e-30)
                log(
                    f"QAD layer {li}/{n_layers - 1}: {len(linears)} linears, held-out nMSE "
                    f"{info['val_before']:.4e} -> {info['val_after']:.4e} ({-100 * gain:+.1f}%, kept {info['kept']}), "
                    f"{info['steps']}/{info['allotted']} steps{' (early stop)' if info['early_stop'] else ''}, "
                    f"{info['seconds']:.0f} s, elapsed {(time.time() - t_start) / 60:.1f} min, ETA {eta / 60:.1f} min"
                )
            else:
                log(
                    f"QAD layer {li}/{n_layers - 1}: {len(linears)} linears kept at PTQ (not selected)"
                )
        res.tensors.update(tensors)
        res.layers.append(
            {"layer": li, **{k: v for k, v in info.items() if k != "trace"}}
        )
        originals = {}
        for name, ckpt in linears.items():
            mod = layer.get_submodule(name)
            originals[name] = mod.weight.data
            mod.weight.data = dequantize(tensors[ckpt]).to(mod.weight.dtype)
        norm_orig = {}
        for pn, val in norms.items():  # trained norms: the student uses the fp16 value
            prm = layer.get_parameter(pn)
            norm_orig[pn] = prm.data
            prm.data = val.to(torch.float16).to(prm.dtype)
            full = f"{prefix}{li}.{pn}"
            res.dense[(resolve(full) if resolve else full) or full] = val.to(
                torch.float16
            )
        _propagate(layer, ex, xs, kwargs, xs)
        for name, wv in originals.items():  # the model object stays unquantized
            layer.get_submodule(name).weight.data = wv
        for pn, wv in norm_orig.items():
            layer.get_parameter(pn).data = wv
        xt, yt = (
            yt,
            xt,
        )  # the target becomes the next teacher input; reuse the old buffer
        gc.collect()  # training closures/optimizer form reference cycles
    if todo:
        log(budget.summary())
    return res
