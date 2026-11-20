# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
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

import contextlib
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
from .pinned import PinnedWindow, module_to_device
from .runtime import atomic_torch_save

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
    # Math dtype of the layer forward and backward: float32 | float16 | bfloat16. Only the
    # matrix products of the layer change (the latents, the ladder bytes and the loss stay
    # float32), so the matrix units of the GPU can be used. On gfx1030 the float32 step is
    # compute-bound with the matrix path idle (docs/qad.md, "The 27B at four GPUs").
    dtype: str = "float32"
    # Split the sequences of every step over all of the run's devices (each holds a copy of
    # the layer; the decoded weights go out and the gradients come back). Same gradient up
    # to summation order; off for train_norms and the KL term, which run on the first device.
    data_parallel: bool = True
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
    # bytes (CPTC: scales) and the RMSNorm weights, with codes frozen; 0 = off. The presets
    # set it (default and thorough: on; 200 is the Qwen3.5-4B study value). The network is split layer-wise over the devices, and the phase is
    # skipped with a note if it still does not fit.
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
    # ---- CPTC options (--cptc only); every default is the pre-factorial path ----
    # Per-block scale parameterisation of CPTC QAD and final-KL: "log" keeps
    # scale0 * exp(delta) with delta in [-.5, .5]; "row_u8" uses
    # scale0 + delta * (row max block scale / 255), projected to the same
    # relative trust region. No exported scale is floored: zero and subnormal
    # fp16 scales survive an unchanged export exactly.
    cptc_scale_units: str = "log"
    # Bounded projection sweeps of the CPTC code coordinate descent in QAD.
    cptc_projection_sweeps: int = 1
    # Final-KL: train the ONE global mirrored-int8 table pair shared by every CPTC
    # Linear (coefficients and fp16 table_scale stay fixed). Off by default.
    final_cptc_train_tables: bool = False
    # AdamW lr of the shared table entries, in int8 levels per step. This is
    # deliberately independent of final_lr_p (the scale/byte learning rate).
    final_cptc_table_lr: float = 0.02
    # Final-KL: explicit byte budget for immutable CPTC table indices; 0 = off.
    final_cptc_cache_mb: int = 0


# Fields of the final logit-KL phase: never part of the layer-wise fingerprint (they do
# not change the layer artifacts), fingerprinted separately by hi2.finalkl.
FINAL_FIELDS = {f for f in QadConfig.__dataclass_fields__ if f.startswith("final_")}


# Chosen on Qwen3.5-0.8B at 3.0 bpw (docs/qad.md); everything else default.
# fast = half the steps spent where layers still gain, with early stopping (better than a
# uniform 100 steps at the same time); no reduced budget stayed within noise of default.
FINAL_KL_STEPS = 200  # the Qwen3.5-4B study value (docs/lowbit.md)
PRESETS: dict[str, dict] = {
    "fast": dict(budget=0.5, min_gain=0.01, min_steps=75, eval_every=50),
    "default": dict(final_kl=FINAL_KL_STEPS),
    "thorough": dict(steps=600, batch=8, samples=512, final_kl=FINAL_KL_STEPS),
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
    if qc.dtype not in ("float32", "float16", "bfloat16"):
        raise SystemExit("QAD dtype must be 'float32', 'float16' or 'bfloat16'")
    if qc.samples < 1 or qc.steps < 0 or qc.batch < 1 or qc.val < 1:
        raise SystemExit("QAD samples/batch/val must be >= 1 and steps >= 0")
    if qc.cptc_scale_units not in ("log", "row_u8"):
        raise SystemExit("QAD cptc_scale_units must be 'log' or 'row_u8'")
    if qc.cptc_projection_sweeps < 1:
        raise SystemExit("QAD cptc_projection_sweeps must be >= 1")
    if qc.final_cptc_cache_mb < 0:
        raise SystemExit("QAD final_cptc_cache_mb must be >= 0")
    if qc.final_cptc_table_lr <= 0 or not math.isfinite(qc.final_cptc_table_lr):
        raise SystemExit("QAD final_cptc_table_lr must be finite and positive")
    if qc.final_cptc_table_lr != QadConfig.final_cptc_table_lr and not (
        qc.final_cptc_train_tables
    ):
        raise SystemExit(
            "QAD final_cptc_table_lr is only used with final_cptc_train_tables=true"
        )
    if (qc.final_cptc_train_tables or qc.final_cptc_cache_mb) and qc.final_kl <= 0:
        raise SystemExit(
            "QAD final_cptc_train_tables/final_cptc_cache_mb need --qad-final-kl > 0"
        )
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


class _CptcTableGather(torch.autograd.Function):
    """Keep only compact indices for backward; bound temporary int64 indices."""

    @staticmethod
    def forward(ctx, lut, indices):
        ctx.save_for_backward(indices)
        ctx.lut_size = lut.numel()
        ctx.lut_dtype = lut.dtype
        flat = indices.reshape(-1)
        out = torch.empty(indices.shape, dtype=lut.dtype, device=lut.device)
        values = out.reshape(-1)
        chunk = (2 << 20) // 8
        for start in range(0, flat.numel(), chunk):
            end = start + chunk
            values[start:end] = lut[flat[start:end].to(torch.int64)]
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        flat = indices.reshape(-1)
        grad = grad_output.reshape(-1)
        result = torch.zeros(
            ctx.lut_size, dtype=ctx.lut_dtype, device=grad_output.device
        )
        chunk = (2 << 20) // 8
        for start in range(0, flat.numel(), chunk):
            end = start + chunk
            ix = flat[start:end].to(torch.int64)
            if grad_output.is_cuda:
                # Match advanced-index backward's sorted accumulation on HIP/CUDA,
                # rather than contended, order-dependent floating-point atomics.
                result.index_put_((ix,), grad[start:end], accumulate=True)
            else:
                result.index_add_(0, ix, grad[start:end])
        return result, None


# The existing log-scale trust region: a CPTC block scale may move by at most
# exp(+/-0.5) of its PTQ value, and never below zero. "row_u8" only changes the
# parameterisation of the same region (additive uint8-step deltas instead of logs).
_CPTC_SCALE_REL = (math.exp(-0.5) - 1.0, math.exp(0.5) - 1.0)


def _cptc_scale_unit(scale0: torch.Tensor, param_off, row_block) -> torch.Tensor:
    """Per-block "row_u8" unit: the row's initial maximum block scale / 255.

    ``unit`` has the flat layout of ``scale0`` (one value per block, ordered by
    ``param_off``). Every block of a row receives the same unit, so a delta is
    measured in 1/255 of the largest scale that row had at PTQ time. A completely
    zero row gets unit 0, which keeps it exactly zero under every delta.
    """
    unit = torch.zeros_like(scale0)
    bs = row_block.to(torch.int64)
    off = param_off.to(torch.int64)
    for block in sorted(set(bs.tolist())):
        rows = (bs == block).nonzero().flatten()
        if rows.numel() == 0:
            continue
        nblk = int(off[rows[0] + 1] - off[rows[0]])
        pidx = off[rows].unsqueeze(1) + torch.arange(nblk, device=off.device)
        row_max = scale0[pidx].max(dim=1, keepdim=True).values
        unit[pidx] = (row_max / 255.0).expand_as(pidx)
    return unit


def _cptc_scales(scale0, delta, unit, scale_units: str) -> torch.Tensor:
    """The CPTC block scales of a delta state: ``scale0*exp(delta)`` (log) or
    ``scale0 + delta*unit`` (row_u8), within fp16's positive range.

    Zero and subnormal ``scale0`` values are carried through exactly when the
    delta is zero; the export applies its own fp16 rounding, never a floor."""
    if scale_units == "row_u8":
        scale = scale0 + delta * unit
    else:
        scale = scale0 * torch.exp(delta)
    return scale.clamp(0.0, 65504.0)


def _cptc_project_scales(delta, scale0, unit, scale_units: str) -> None:
    """Project a scale delta into the existing relative trust region."""
    if scale_units == "row_u8":
        lo = torch.where(
            unit > 0, scale0 * _CPTC_SCALE_REL[0] / unit.clamp_min(1e-30), 0.0
        )
        hi = torch.where(
            unit > 0, scale0 * _CPTC_SCALE_REL[1] / unit.clamp_min(1e-30), 0.0
        )
        delta.copy_(delta.clamp(min=lo, max=hi))
    else:
        delta.clamp_(-0.5, 0.5)


def _clone_codecs(codecs) -> dict:
    """Fresh codec objects with copied tables, so exports never share or mutate the
    input/PTQ codecs."""
    from .cptc import CptcCodec

    return {
        int(k): CptcCodec(
            c.nbits,
            c.depth,
            c.a0,
            c.a1,
            c.table.cpu().clone(),
            c.table_scale.cpu().clone(),
            c.tap_q,
            c.shift,
        )
        for k, c in codecs.items()
    }


def _same_codec(a, b) -> bool:
    return (
        a.nbits == b.nbits
        and a.depth == b.depth
        and a.tap_q == b.tap_q
        and a.shift == b.shift
        and a.a0 == b.a0
        and a.a1 == b.a1
        and torch.equal(a.table.cpu(), b.table.cpu())
        and torch.equal(a.table_scale.cpu(), b.table_scale.cpu())
    )


def cptc_codecs_from_tensors(tensors: dict) -> dict:
    """The verified per-K codecs of an exported CPTC tensor set.

    A v3.3 checkpoint carries exactly ONE table, coefficient pair and fp16 table
    scale per K, so every tensor must agree. This fails closed instead of writing a
    global config that would silently decode some tensors with the wrong table.
    """
    from .cptc import CptcTensor

    seen: dict = {}
    for name, t in tensors.items():
        if not isinstance(t, CptcTensor):
            continue
        for k, codec in t.codecs.items():
            k = int(k)
            previous = seen.get(k)
            if previous is None:
                seen[k] = codec
            elif not _same_codec(previous, codec):
                raise ValueError(
                    f"CPTC tensors disagree on the K={k} table/coefficients "
                    f"({name}); one global model config cannot represent both"
                )
    return seen


def check_cptc_options(qc: "QadConfig", cptc: bool, final_only: bool = False) -> None:
    """Reject CPTC-specific settings on a non-CPTC run instead of ignoring them."""
    if cptc:
        return
    chosen = []
    if qc.cptc_scale_units != "log":
        chosen.append("cptc_scale_units")
    if not final_only and qc.cptc_projection_sweeps != 1:
        chosen.append("cptc_projection_sweeps")
    if qc.final_cptc_train_tables:
        chosen.append("final_cptc_train_tables")
    if qc.final_cptc_cache_mb:
        chosen.append("final_cptc_cache_mb")
    if (
        qc.final_cptc_table_lr != QadConfig.final_cptc_table_lr
        and not qc.final_cptc_train_tables
    ):
        chosen.append("final_cptc_table_lr")
    if chosen:
        where = "final-KL" if final_only else "QAD"
        raise SystemExit(
            f"{where}: {', '.join(chosen)} only apply to --cptc tensors; "
            "enable --cptc or remove these settings"
        )


def final_fingerprint_fields(qc: "QadConfig") -> dict:
    """Final-phase settings that can change its numerical result.

    ``final_cptc_cache_mb`` is deliberately absent: the cache only stores immutable
    table indices, so changing the budget must not invalidate a finished artifact.
    The CPTC training fields are omitted at their defaults, which leaves the
    fingerprint of a pre-factorial final-KL artifact unchanged.
    """
    legacy = {
        "final_cptc_train_tables": False,
        "final_cptc_table_lr": 0.02,
    }
    return {
        k: v
        for k, v in asdict(qc).items()
        if k in FINAL_FIELDS
        and k != "final_cptc_cache_mb"
        and legacy.get(k, object()) != v
    }


class CptcIndexCache:
    """Explicitly budgeted, training-only cache of immutable CPTC table indices.

    Entries are the exact ``decode_index`` values (0..126) of a tensor's frozen code
    stream, one uint8 per stored weight. They do not depend on table values or block
    scales, so a trainable table can rebuild its LUT every forward without a stale
    decoded-weight cache. The object is shared by every CPTC Linear of a final-KL
    run; ``reserve`` accounts every cache-owned tensor (the uint8 index buffers) and
    refuses allocations beyond ``budget_bytes``. A module/device move replaces its
    slot on the new device, so one logical index group stays one reservation. No
    decoded table, scale or int64 index is kept: row classes and weight bytes are the
    model's normal buffers and exist with or without the cache. It is never used by
    inference.
    """

    def __init__(self, budget_bytes: int = 0):
        self.budget_bytes = max(0, int(budget_bytes))
        self.used_bytes = 0
        self.entries = 0

    def reserve(self, nbytes: int) -> bool:
        nbytes = int(nbytes)
        if nbytes <= 0 or self.used_bytes + nbytes > self.budget_bytes:
            return False
        self.used_bytes += nbytes
        self.entries += 1
        return True

    def summary(self) -> dict:
        return {
            "budget_bytes": self.budget_bytes,
            "used_bytes": self.used_bytes,
            "entries": self.entries,
            "free_bytes": self.budget_bytes - self.used_bytes,
        }


class CptcSharedTables:
    """The ONE global trainable pair of mirrored-int8 CPTC half-tables.

    A final-KL run shares this owner by reference across every CPTC Linear; no
    per-layer codebook is created. Coefficients, depth and the fp16 ``table_scale``
    stay fixed; only the 64 signed-int8 levels per K train. Entries use a
    straight-through rounding and project to [-127, 127] with the last low-half
    entry pinned to 0 (the serialized mirror contract).

    This is deliberately not an ``nn.Module``: attaching the same module to every
    Linear would make model/device staging duplicate or move one global parameter
    set unpredictably. final-KL keeps the one master copy on the first training
    device; ``lut`` materializes its live selected values on each Linear's device so
    sharded gradients return to that master and one optimizer group.
    """

    def __init__(self, codecs: dict):
        if not codecs:
            raise ValueError("CptcSharedTables needs at least one CPTC codec")
        self._codecs = {}
        self.table_u = {}
        for k in sorted(int(k) for k in codecs):
            codec = codecs[k]
            if int(codec.nbits) != k or k not in (2, 4):
                raise ValueError("CPTC shared tables require K=2 and/or K=4 codecs")
            self._codecs[k] = codec
            self.table_u[k] = nn.Parameter(codec.table.float().clone())

    def widths(self) -> tuple[int, ...]:
        return tuple(sorted(self._codecs))

    def parameters(self) -> list[nn.Parameter]:
        return [self.table_u[k] for k in self.widths()]

    def to(self, device) -> "CptcSharedTables":
        """Move the one master copy; module staging must never copy it per Linear."""
        for p in self.parameters():
            p.data = p.data.to(device)
        return self

    def backward(self, output: torch.Tensor, gradient=None) -> None:
        """Accumulate shared FP32 shard gradients in a stable autograd order.

        Parallel engine scheduling can change the addition order at the one
        master parameter even when each shard's gradient is unchanged.
        """
        with torch.autograd.set_multithreading_enabled(False):
            output.backward(gradient)

    def _selected(self, k: int) -> torch.Tensor:
        """The 64 table entries as STE values in [-127, 127], last entry exactly 0."""
        p = self.table_u[int(k)]
        hard = torch.round(p.detach()).clamp(-127, 127)
        hard = torch.cat([hard[:-1], hard.new_zeros(1)])
        soft = torch.cat([p.clamp(-127, 127)[:-1], p.new_zeros(1)])
        return _ste(hard, soft)

    def selected_int8(self, k: int) -> torch.Tensor:
        """The serialized hard table of one K (no graph)."""
        p = self.table_u[int(k)].detach()
        table = torch.round(p).clamp(-127, 127)
        table = torch.cat([table[:-1], table.new_zeros(1)])
        return table.to(torch.int8)

    def lut(self, k: int, device) -> torch.Tensor:
        """The runtime fp16 LUT values carried in fp32 with an fp32 surrogate graph.

        The forward values are exactly the old ``to(torch.float16)`` values, but the
        lookup keeps the live selected fp32 entries and a straight-through fp32 graph
        until the final cast, so tiny gradients are accumulated in fp32 instead of
        underflowing through a half gather.
        """
        k = int(k)
        index = torch.arange(2**7 - 1, device=device, dtype=torch.int64)
        lower = index < 2**6
        pos = index.clamp(max=2**6 - 1)
        neg = (2**7 - 2 - index).clamp(min=0, max=2**6 - 1)
        # Materialize the live master values on the using Linear's device with a
        # differentiable copy: backward sums the shard gradient into the one master.
        table = self._selected(k).to(device)
        i8 = torch.where(lower, table[pos], -table[neg])
        scaled = i8.float() * float(self._codecs[k].table_scale)
        # Exact runtime fp16 forward values, fp32 surrogate gradient for the lookup.
        return _ste(scaled.detach().half().float(), scaled)

    @torch.no_grad()
    def project(self) -> None:
        for k in self.widths():
            p = self.table_u[k].data
            p.clamp_(-127, 127)
            p[-1] = 0

    def codecs_out(self) -> dict:
        """Selected table values as fresh codec objects for every exported tensor."""
        from .cptc import CptcCodec

        return {
            k: CptcCodec(
                c.nbits,
                c.depth,
                c.a0,
                c.a1,
                self.selected_int8(k).cpu(),
                c.table_scale.clone().cpu(),
                c.tap_q,
                c.shift,
            )
            for k, c in self._codecs.items()
        }


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
        self.has_had = bool(
            t0.had.any()
        )  # known on the CPU: no device sync per forward
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
        from . import qad_kernel

        if qad_kernel.available(self.latent.device):  # fused decode and its backward
            return qad_kernel.stored(self, grad)
        parts = [self._group(g, grad)[0] for g in self.groups]
        return torch.cat(parts)[self.inv_perm]

    def forward(self) -> torch.Tensor:
        from . import qad_kernel

        if qad_kernel.available(
            self.latent.device
        ):  # decode, transform and scale fused
            return qad_kernel.original(self)
        w = self.stored(True)
        if self.has_had:
            w = fwht_groups(w, self.had)
        if self.has_scale:
            w = w / self.col_scale.float()
        return w

    @torch.no_grad()
    def export(self, stager=None) -> HI2Tensor:
        pieces = []
        k = self.shape[1]
        # The codes are the big ones (a byte per weight), and they come back to the host
        # here: stage them through the pinned window when there is one.
        to_host = (
            (lambda t: stager.to_host(t)) if stager is not None else (lambda t: t.cpu())
        )
        for g in self.groups:
            _, q, lo_b, st_b = self._group(g, False)
            params = torch.stack([lo_b, st_b], dim=-1).to(torch.uint8)
            pieces.append(
                (
                    to_host(g.rows),
                    to_host(q.reshape(-1, k).to(torch.uint8)),
                    to_host(params),
                    to_host(g.consts),
                )
            )
        return assemble(
            pieces,
            to_host(self.row_nbits),
            to_host(self.row_block),
            to_host(self.had),
            k,
            to_host(self.col_scale) if self.has_scale else None,
        )


class CptcFrozenCodeLinear(nn.Module):
    """The final-KL CPTC student: frozen codes, trainable per-block scales.

    Codes are frozen (1 byte per weight on the device, no fp32 latent). Row classes
    are computed once, and only the exact fp16 scale/LUT product is recomputed per
    forward. ``scale_units`` selects the same log or row_u8 parameterisation as QAD.
    ``tables`` optionally points at the one global trainable table owner; every
    Linear shares it rather than keeping a per-layer codebook. ``index_cache`` is an
    optional explicitly budgeted store of immutable table indices (uint8).
    """

    def __init__(self, t0, scale_units="log", tables=None, index_cache=None):
        super().__init__()
        from .cptc import CptcTensor

        if not isinstance(t0, CptcTensor):
            raise TypeError("CptcFrozenCodeLinear needs a CptcTensor")
        if scale_units not in ("log", "row_u8"):
            raise ValueError("CPTC scale_units must be 'log' or 'row_u8'")
        self.shape = t0.shape
        self.codecs = dict(t0.codecs)
        self.scale_units = scale_units
        self.tables = tables
        self.index_cache = index_cache
        self.register_buffer("codes", t0.codes.clone())
        self.register_buffer("code_off", t0.code_off.clone())
        self.register_buffer("param_off", t0.param_off.clone())
        self.register_buffer("row_nbits", t0.row_nbits.clone())
        self.register_buffer("row_block", t0.row_block.clone())
        self.register_buffer("had", t0.had.clone())
        cs = t0.col_scale
        self.register_buffer(
            "col_scale",
            cs.clone() if cs is not None else torch.ones(0, dtype=torch.float16),
        )
        self.has_scale = cs is not None
        self.has_had = bool(t0.had.any())
        self.register_buffer("scale0", t0.scale.float().clone())
        self.register_buffer(
            "scale_unit",
            _cptc_scale_unit(self.scale0, self.param_off, self.row_block)
            if scale_units == "row_u8"
            else torch.ones(0),
        )
        self.delta = nn.Parameter(torch.zeros_like(self.scale0))
        # Row classes are immutable: never call .tolist()/nonzero() per forward.
        nb = t0.row_nbits.to(torch.int64)
        bl = t0.row_block.to(torch.int64)
        self.groups = []
        self.rows = []
        for nbits in sorted(set(nb.tolist())):
            for bs in sorted(set(bl.tolist())):
                rows = ((nb == nbits) & (bl == bs)).nonzero().flatten()
                if rows.numel() == 0:
                    continue
                self.rows.append(rows)
                self.groups.append((int(nbits), int(bs)))
        self._index_slots = [None] * len(self.groups)

    def scales(self) -> torch.Tensor:
        # Zero and subnormal PTQ scales are valid and must survive a zero-step
        # export exactly; no floor is applied to exported values.
        return _cptc_scales(self.scale0, self.delta, self.scale_unit, self.scale_units)

    def _align_devices(self):
        dev = self.delta.device
        names = [
            "codes", "code_off", "param_off", "row_nbits", "row_block",
            "had", "scale0", "scale_unit",
        ]
        for name in names:
            value = getattr(self, name)
            if value.device != dev:
                setattr(self, name, value.to(dev))
        for i, rows in enumerate(self.rows):
            if rows.device != dev:
                self.rows[i] = rows.to(dev)
        if self.has_scale and self.col_scale.device != dev:
            self.col_scale = self.col_scale.to(dev)
        for i, slot in enumerate(self._index_slots):
            if slot is not None and slot.device != dev:
                # Replace the cache-owned buffer in place: the old device copy is
                # released after this assignment, so used_bytes still counts one
                # logical slot and there is no retained duplicate.
                self._index_slots[i] = slot.to(dev)
                del slot

    def param_groups(self):
        return [], [self.delta]

    @torch.no_grad()
    def project(self):
        _cptc_project_scales(
            self.delta, self.scale0, self.scale_unit, self.scale_units
        )

    def _lut(self, nbits: int, codec, device) -> torch.Tensor:
        """The exact runtime LUT. A fixed table is fp16; the shared table is fp32 with
        its forward values already rounded exactly like the fp16 carrier."""
        if self.tables is not None:
            return self.tables.lut(nbits, device)
        from .cptc import index_table_value

        return index_table_value(
            torch.arange(127, device=device),
            codec.table.to(device),
            float(codec.table_scale),
        )

    def _build_index(self, rows, nbits: int, bs: int, codec) -> torch.Tensor:
        from .cptc import decode_index, unpack_codes, window_indices

        k = self.shape[1]
        nblk = k // bs
        n = rows.numel()
        out = torch.empty(n, nblk, bs, dtype=torch.uint8, device=self.delta.device)
        wr = k * nbits // 32
        # Build in bounded row chunks too: the uint8 cache should not require an
        # equally sized int64 working copy of the whole tensor.
        chunk_rows = max(1, (2 << 20) // max(1, k * 8))
        for c0 in range(0, n, chunk_rows):
            rr = rows[c0 : c0 + chunk_rows]
            widx = self.code_off[rr].long().unsqueeze(1) + torch.arange(
                wr, device=self.delta.device
            )
            codes = unpack_codes(self.codes[widx], nbits).reshape(
                rr.numel(), nblk, bs
            )
            q = window_indices(codes.reshape(-1, bs), nbits, codec.depth, bs)
            idx = decode_index(q, codec.a0, codec.a1, codec.tap_q).reshape(
                rr.numel(), nblk, bs
            )
            out[c0 : c0 + rr.numel()] = idx.to(torch.uint8)
        return out

    def _indices(self, gi: int, rows, nbits: int, bs: int, codec) -> torch.Tensor:
        """Immutable table indices (uint8), cached only under the explicit budget."""
        slot = self._index_slots[gi]
        if slot is not None:
            return slot
        slot = self._build_index(rows, nbits, bs, codec)
        if self.index_cache is not None and self.index_cache.reserve(
            rows.numel() * self.shape[1]
        ):
            self._index_slots[gi] = slot
        return slot

    @staticmethod
    def _table_values(lut: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        if lut.requires_grad and torch.is_grad_enabled():
            return _CptcTableGather.apply(lut, idx)
        chunk_rows = max(1, (2 << 20) // max(1, idx[0].numel() * 8))
        return torch.cat([
            lut[idx[c0 : c0 + chunk_rows].to(torch.int64)]
            for c0 in range(0, idx.shape[0], chunk_rows)
        ], dim=0)

    def stored(self, grad: bool = True) -> torch.Tensor:
        self._align_devices()
        r, k = self.shape
        out = torch.empty(r, k, dtype=torch.float32, device=self.delta.device)
        for gi, (nbits, bs) in enumerate(self.groups):
            rows = self.rows[gi]
            codec = self.codecs[nbits]
            nblk = k // bs
            sc = self.scales()
            idx = self._indices(gi, rows, nbits, bs, codec)
            lut = self._lut(nbits, codec, self.delta.device)
            # The differentiable gather saves uint8 indices, not a full model's
            # worth of int64 indices. Both directions bound their conversion.
            vals = self._table_values(lut, idx)
            if vals.dtype != torch.float32:
                vals = vals.float()
            pidx = self.param_off[rows].long().unsqueeze(1) + torch.arange(
                nblk, device=self.delta.device
            )
            s = sc[pidx].float()
            s_h = s.half().float().unsqueeze(-1)
            # vals already holds the exact fp16 LUT values as fp32; keeping it there
            # preserves fp32 surrogate gradients while leaving every forward value
            # (including the final fp16 product) bit-identical to the old path.
            wq = (s_h * vals.detach()).half().float()
            if grad:
                wq = _ste(wq, s.unsqueeze(-1) * vals)
            out[rows] = wq.reshape(rows.numel(), k)
        return out

    def weight(self) -> torch.Tensor:
        w = self.stored(True)
        if self.has_had:
            w = fwht_groups(w, self.had)
        if self.has_scale:
            w = w / self.col_scale.float()
        return w

    def forward(self) -> torch.Tensor:
        return self.weight()

    def _export_codecs(self) -> dict:
        if self.tables is None:
            return _clone_codecs(self.codecs)
        missing = set(self.codecs) - set(self.tables.widths())
        if missing:
            raise ValueError(
                f"shared CPTC tables have no codec for K={sorted(missing)}"
            )
        tables = self.tables.codecs_out()
        return {k: tables[k] for k in self.codecs}

    @torch.no_grad()
    def export(self, stager=None):
        from .cptc import CptcTensor

        self._align_devices()
        # The returned CptcTensor is the checkpoint artifact: every field must be
        # host-owned, so stage them through the pinned window like the ladder export
        # does (a plain .cpu() without a stager). The shape, dtypes and the exact
        # code/scale bytes are unchanged.
        to_host = (
            (lambda t: stager.to_host(t)) if stager is not None else (lambda t: t.cpu())
        )
        return CptcTensor(
            codes=to_host(self.codes),
            code_off=to_host(self.code_off),
            scale=to_host(self.scales().half()),
            param_off=to_host(self.param_off),
            row_nbits=to_host(self.row_nbits),
            row_block=to_host(self.row_block),
            had=to_host(self.had),
            shape=self.shape,
            codecs=self._export_codecs(),
            col_scale=to_host(self.col_scale) if self.has_scale else None,
        )


class CptcFakeQuantLinear(nn.Module):
    """Full weight-latent QAD for CPTC with deterministic code projection.

    The latent is the stored-basis weight ``H (W s)`` in fp32. Every forward starts the
    block-local coordinate projection from the immutable PTQ code stream, evaluates all
    ``2**K`` symbols for each position in ``t mod period`` phases, and decodes with the
    exact fp16 table/scale roundings the runtime uses. ``scale_units`` selects log deltas
    (unchanged default) or the additive row_u8 parameterisation. The hard codes are a
    straight-through value for the latent and the per-block scales, so the optimizer can
    move both the weights and the code assignment. ``export()`` runs the same projection
    on the final latent/scales; it is not a frozen-code or scale-only path.
    """

    def __init__(self, w, t0, comp=None, init="ptq", units="abs", sweeps=1,
                 scale_units="log"):
        super().__init__()
        from .cptc import (
            CptcTensor,
            dequantize_stored_cptc,
            projection_period,
        )

        if not isinstance(t0, CptcTensor):
            raise TypeError("CptcFakeQuantLinear needs a CptcTensor")
        if scale_units not in ("log", "row_u8"):
            raise ValueError("CPTC scale_units must be 'log' or 'row_u8'")
        r, k = t0.shape
        dev = w.device if w is not None else t0.scale.device
        self.shape = (r, k)
        self.codecs = dict(t0.codecs)
        self.sweeps = int(sweeps)
        self.scale_units = scale_units
        self.register_buffer("codes0", t0.codes.clone())
        self.register_buffer("code_off", t0.code_off.clone())
        self.register_buffer("param_off", t0.param_off.clone())
        self.register_buffer("row_nbits", t0.row_nbits.clone())
        self.register_buffer("row_block", t0.row_block.clone())
        self.register_buffer("had", t0.had.clone())
        cs = None if t0.col_scale is None else t0.col_scale.to(dev)
        self.register_buffer(
            "col_scale",
            cs.clone() if cs is not None else torch.ones(0, dtype=torch.float16),
        )
        self.has_scale = cs is not None
        self.has_had = bool(t0.had.any())
        self.register_buffer("scale0", t0.scale.float().clone())
        self.delta = nn.Parameter(torch.zeros_like(self.scale0))
        self.groups = []
        nb = t0.row_nbits.to(torch.int64)
        bl = t0.row_block.to(torch.int64)
        for nbits in sorted(set(nb.tolist())):
            for bs in sorted(set(bl.tolist())):
                rows = ((nb == nbits) & (bl == bs)).nonzero().flatten()
                if rows.numel() == 0:
                    continue
                projection_period(int(t0.codecs[int(nbits)].depth), int(nbits))
                if k % bs:
                    raise ValueError(f"K={k} is not a multiple of block {bs}")
                self.groups.append((rows, int(nbits), int(bs)))
        perm = torch.cat([g[0] for g in self.groups])
        self.register_buffer("inv_perm", torch.argsort(perm))
        # the latent can be normalised by the initial block scale so one learning rate
        # means the same fraction of a quantisation step across tensors and widths
        self.step_units = units == "step"
        unit = torch.ones(r, k, device=dev)
        if self.step_units:
            for rows, _, bs in self.groups:
                nblk = k // bs
                pidx = (
                    self.param_off[rows].long().unsqueeze(1)
                    + torch.arange(nblk, device=self.param_off.device)
                ).reshape(-1)
                s = self.scale0[pidx].clamp_min(2**-14).reshape(rows.numel(), nblk)
                unit[rows] = s.repeat_interleave(bs, dim=1).to(unit.device)
        self.register_buffer("unit", unit if self.step_units else torch.ones(0))
        self.register_buffer(
            "scale_unit",
            _cptc_scale_unit(self.scale0, self.param_off, self.row_block)
            if self.scale_units == "row_u8"
            else torch.ones(0),
        )
        if init == "weights" or comp is None:
            if w is None:
                v = dequantize_stored_cptc(t0, dev).float()
            else:
                v = w.float()
                if cs is not None:
                    v = v * cs.float()
                v = fwht_groups(v, t0.had.to(dev)) if bool(t0.had.any()) else v
        else:
            v = comp.float()
        v = v / unit if self.step_units else v
        self.latent = nn.Parameter(v.contiguous().clone())

    def scales(self) -> torch.Tensor:
        # Zero and subnormal PTQ scales are valid and must survive a zero-step
        # export exactly; only the latent-normalisation unit uses a 2^-14 floor.
        return _cptc_scales(self.scale0, self.delta, self.scale_unit, self.scale_units)

    def _align_devices(self):
        """QAD may construct this module from host tensors then move it; keep every
        buffer and parameter used by the projection on the latent device even if a
        module move missed one."""
        dev = self.latent.device
        for name in (
            "codes0", "code_off", "param_off", "row_nbits", "row_block",
            "had", "scale0", "unit", "scale_unit", "inv_perm",
        ):
            value = getattr(self, name)
            if value.device != dev:
                setattr(self, name, value.to(dev))
        if self.has_scale and self.col_scale.device != dev:
            self.col_scale = self.col_scale.to(dev)
        if self.delta.device != dev:
            self.delta.data = self.delta.data.to(dev)

    def param_groups(self):
        return [self.latent], [self.delta]

    @torch.no_grad()
    def project(self):
        # Trust region around the PTQ block scale: QAD refines, never re-scales the
        # tensor. "row_u8" projects the additive delta into the same exp(+/-.5)
        # relative band, which also keeps every block scale nonnegative.
        _cptc_project_scales(
            self.delta, self.scale0, self.scale_unit, self.scale_units
        )
        if self.step_units:
            # keep the latent finite when a very aggressive step is requested
            self.latent.data.clamp_(-65504.0, 65504.0)

    def _unpack(self, rows, nbits) -> torch.Tensor:
        from .cptc import unpack_codes

        k = self.shape[1]
        wr = k * nbits // 32
        widx = self.code_off[rows].long().unsqueeze(1) + torch.arange(
            wr, device=self.codes0.device
        )
        return unpack_codes(self.codes0[widx], nbits).reshape(rows.numel(), k)

    def _lut(self, codec):
        from .cptc import _codec_lut

        cache = getattr(self, "_cptc_lut_cache", None)
        if cache is None:
            cache = self._cptc_lut_cache = {}
        key = (codec.nbits, codec.depth, str(self.latent.device))
        lut = cache.get(key)
        if lut is None:
            lut = _codec_lut(codec, self.latent.device)
            cache[key] = lut
        return lut

    def _lut_half(self, codec):
        from .cptc import index_table_value

        cache = getattr(self, "_cptc_lut_half_cache", None)
        if cache is None:
            cache = self._cptc_lut_half_cache = {}
        key = (codec.nbits, codec.depth, str(self.latent.device))
        lut = cache.get(key)
        if lut is None:
            dev = self.latent.device
            lut = index_table_value(
                torch.arange(127, device=dev),
                codec.table.to(dev),
                codec.table_scale,
            )
            cache[key] = lut
        return lut

    def _project_codes(self, w, codec, bs, init, scales):
        if w.is_cuda:
            # Lazy import keeps CPU-only installations free of Triton.  Do not catch
            # launch/compile errors: a fused failure must be visible, not silently
            # downgraded to the CPU/reference path.
            from .cptc_projection_kernel import project_codes_cuda

            lut = self._lut_half(codec)
            n, k = w.shape
            nblk = k // bs
            # The kernel is block-row independent, so one launch covers the whole
            # [rows x blocks, B] flatten instead of one launch (and its own copies) per
            # stored block. ``scales`` is [N, blocks] row-major, i.e. exactly the
            # flatten's row order, and the result reshapes back to the logical [N, K]
            # with the block seams where they were.
            flat = project_codes_cuda(
                w.reshape(n * nblk, bs),
                init.to(torch.uint8).reshape(n * nblk, bs),
                scales.reshape(n * nblk).half(),
                lut,
                codec.a0,
                codec.a1,
                codec.nbits,
                codec.depth,
                bs,
                sweeps=self.sweeps,
            )
            return flat.reshape(n, k)
        from .cptc import project_codes_blocks

        return project_codes_blocks(
            w,
            codec,
            bs,
            init,
            scales,
            sweeps=self.sweeps,
            lut=self._lut(codec),
        )

    def _values(self, codes, codec, scales, block, lut=None):
        from .cptc import decode_index, window_indices_blocks

        dev = codes.device
        if lut is None:
            lut = self._lut(codec)
        q = window_indices_blocks(codes, codec.nbits, codec.depth, block)
        vals = lut[decode_index(q, codec.a0, codec.a1, codec.tap_q)]
        s = scales.to(dev, torch.float16).float()
        s = s.unsqueeze(-1) if s.dim() == 1 else s.repeat_interleave(block, dim=1)
        hard = (s * vals).to(torch.float16).float()
        return hard, vals

    def stored(self, grad: bool = True) -> torch.Tensor:
        self._align_devices()
        r, k = self.shape
        out = torch.empty(r, k, dtype=torch.float32, device=self.latent.device)
        sc = self.scales()
        chunk = max(1, (16 << 20) // (8 * k))
        for rows, nbits, bs in self.groups:
            codec = self.codecs[nbits]
            rows = rows.to(self.latent.device)
            nblk = k // bs
            for start in range(0, rows.numel(), chunk):
                part = rows[start : start + chunk]
                init = self._unpack(part, nbits)
                pidx = (
                    self.param_off[part].long().unsqueeze(1)
                    + torch.arange(nblk, device=part.device)
                ).reshape(part.numel(), nblk)
                s_row = sc[pidx].float()
                v_row = self.latent[part]
                if self.step_units:
                    v_row = v_row * self.unit[part]
                codes = self._project_codes(v_row, codec, bs, init, s_row)
                hard, vals = self._values(
                    codes, codec, s_row, bs, lut=self._lut(codec)
                )
                if grad:
                    soft = s_row.repeat_interleave(bs, dim=1) * vals
                    out[part] = _ste(hard, soft + v_row)
                else:
                    out[part] = hard
        return out

    def weight(self) -> torch.Tensor:
        w = self.stored(True)
        if self.has_had:
            w = fwht_groups(w, self.had)
        if self.has_scale:
            w = w / self.col_scale.float()
        return w

    def forward(self) -> torch.Tensor:
        return self.weight()

    @torch.no_grad()
    def export(self, stager=None):
        self._align_devices()
        from .cptc import assemble_cptc

        r, k = self.shape
        sc = self.scales()
        pieces = []
        to_host = (
            (lambda t: stager.to_host(t)) if stager is not None else (lambda t: t.cpu())
        )
        chunk = max(1, (16 << 20) // (8 * k))
        for rows, nbits, bs in self.groups:
            codec = self.codecs[nbits]
            rows = rows.to(self.latent.device)
            nblk = k // bs
            for start in range(0, rows.numel(), chunk):
                part = rows[start : start + chunk]
                init = self._unpack(part, nbits)
                pidx = (
                    self.param_off[part].long().unsqueeze(1)
                    + torch.arange(nblk, device=part.device)
                ).reshape(part.numel(), nblk)
                s_row = sc[pidx].float()
                v_row = self.latent[part]
                if self.step_units:
                    v_row = v_row * self.unit[part]
                codes_out = self._project_codes(v_row, codec, bs, init, s_row)
                pieces.append(
                    (
                        to_host(part),
                        to_host(codes_out.to(torch.uint8)),
                        to_host(s_row.to(torch.float16)),
                    )
                )
        return assemble_cptc(
            pieces,
            to_host(self.row_nbits),
            to_host(self.row_block),
            to_host(self.had),
            k,
            _clone_codecs(self.codecs),
            to_host(self.col_scale) if self.has_scale else None,
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


def _parallel_step(w, wd, batch, xs, ys, workers, dtype=torch.float32):
    """One data-parallel step: worker ``k`` (device, layer copy, kwargs) takes every
    ``len(workers)``-th sequence of ``batch``, runs its forward and backward on its own
    copy of the decoded weights, and the gradients are summed on the first device.
    Returns ``(loss, [gradient per weight in w's order])``."""
    from concurrent.futures import ThreadPoolExecutor

    from torch.func import functional_call

    n = len(workers)

    def work(k):
        dev, lay, kw = workers[k]
        wk = (
            wd if k == 0 else {a: v.detach().to(dev, dtype=dtype) for a, v in w.items()}
        )
        if k:
            for v in wk.values():
                v.requires_grad_(True)
        tot = 0.0
        stager = PinnedWindow()
        with torch.cuda.device(dev) if dev.type == "cuda" else contextlib.nullcontext():
            for i in batch[k::n]:
                x = stager.to_device(xs[i], dev, dtype=dtype)
                out = _out(functional_call(lay, wk, (x,), kw))
                tgt = stager.to_device(ys[i], dev).float()
                loss = _nmse(out.float(), tgt) / len(batch)
                loss.backward()
                tot += loss.item()
        return wk, tot

    with ThreadPoolExecutor(n) as pool:
        res = list(pool.map(work, range(n)))
    first = workers[0][0]
    grads = []
    for a in w:
        g = None
        for wk, _ in res:
            gk = wk[a].grad
            if gk is not None:
                gk = gk.to(first).float()
                g = gk if g is None else g + gk
        grads.append(torch.zeros_like(wd[a], dtype=torch.float32) if g is None else g)
    return sum(t for _, t in res), grads


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
    replicas: list[torch.device] | None = None,
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
    dt = getattr(torch, qc.dtype)  # the math dtype of the layer forward and backward
    if dt is torch.bfloat16 and dev.type == "cuda" and torch.version.hip:
        arch = torch.cuda.get_device_properties(dev).gcnArchName
        if arch.startswith("gfx10"):
            raise SystemExit(
                f"bfloat16 matrix math is not available on {arch}: RDNA2 has no "
                "fdot2.bf16 instruction and the compiler stops with 'Cannot select: "
                "intrinsic %llvm.amdgcn.fdot2.bf16.bf16'. Use --qad-dtype float16."
            )
    layer = copy.deepcopy(layer)
    reps = []
    if (
        replicas
        and len(replicas) > 1
        and qc.data_parallel
        and not qc.train_norms
        and head is None
    ):
        from .executor import serialize_triton_autotune

        serialize_triton_autotune()
        for d in replicas:
            if d != dev:
                r = module_to_device(copy.deepcopy(layer), d, dtype=dt)
                for p in r.parameters():
                    p.requires_grad_(False)
                reps.append((d, r))
    layer = module_to_device(layer, dev, dtype=dt)
    for p in layer.parameters():
        p.requires_grad_(False)
    for m in fq.values():
        module_to_device(m, dev)
    kw = to_device(kwargs, dev)
    reps = [(d, r, to_device(kwargs, d)) for d, r in reps]
    stager = PinnedWindow()
    norm_params = {}
    if qc.train_norms:
        for n, p in layer.named_parameters():
            if is_norm_param(n, p):
                p.data = p.data.float()
                p.requires_grad_(True)
                norm_params[n] = p
    hd = None
    if head is not None:
        hd = tuple(module_to_device(copy.deepcopy(m), dev).float() for m in head)
        for m in hd:
            for p in m.parameters():
                p.requires_grad_(False)
        kg = torch.Generator().manual_seed(qc.seed + 1)
        n_tok = xs[0].shape[-2]
        k_tok = min(qc.kl_tokens, n_tok) if qc.kl_tokens > 0 else n_tok
        val_pos = {i: torch.randperm(n_tok, generator=kg)[:k_tok] for i in val_idx}

    def loss_fn(out, i, pos=None):
        tgt = stager.to_device(ys[i], dev).float()
        loss = _nmse(out.float(), tgt)
        if hd is not None:
            if pos is None:
                pos = torch.randperm(n_tok, generator=kg)[:k_tok]
            loss = loss + qc.kl_weight * head_kl(hd, out.float(), tgt, pos.to(dev))
        return loss

    def weights(mods=None):
        return {f"{n}.weight": m() for n, m in (mods or fq).items()}

    def run(wd, i):
        return _out(
            functional_call(layer, wd, (stager.to_device(xs[i], dev, dtype=dt),), kw)
        )

    @torch.no_grad()
    def val_loss(mods=None) -> float:
        wd = {k: v.detach().to(dt) for k, v in weights(mods).items()}
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
    try:  # the fused implementation halves the optimizer step on a GPU
        opt = torch.optim.AdamW(groups, weight_decay=0.0, fused=dev.type == "cuda")
    except (RuntimeError, ValueError):
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
        # the detached copies are the leaves the backward writes into: the rounding of the
        # math dtype is a straight-through (the gradient of the rounded weight goes to the
        # float32 latent unchanged)
        wd = {k: v.detach().to(dt).detach().requires_grad_(True) for k, v in w.items()}
        tot = 0.0
        if not reps:
            for i in batch:
                loss = loss_fn(run(wd, i), i) / len(batch)
                loss.backward()
                tot += loss.item()
            grads = [wd[k].grad.float() if wd[k].grad is not None else None for k in w]
        else:
            tot, grads = _parallel_step(
                w, wd, batch, xs, ys, [(dev, layer, kw)] + reps, dtype=dt
            )
        torch.autograd.backward(list(w.values()), grads)
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
            if event := getattr(log, "event", None):
                event(
                    "qad",
                    "running",
                    layer=tag,
                    step=step + 1,
                    total_steps=n_steps,
                    elapsed_s=time.time() - t0,
                    validation_nmse=vl,
                )
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


def _ptq_fallback(layer, linears, inits, comps, qc, stager) -> dict | None:
    """The PTQ encoding scored next to training, as export-ready modules.

    CPTC always gets the exact frozen-code module, whatever ``init``: its export is
    the PTQ code stream and block scales byte for byte, so it is the PTQ candidate
    (and the zero-step artifact) even when QAD itself starts from that encoding. The
    ladder path only scores a separate PTQ candidate when the latent starts away from
    it (``init=ptq`` already is that encoding)."""
    from .cptc import CptcTensor

    if qc.init == "ptq" and not any(isinstance(t, CptcTensor) for t in inits.values()):
        return None
    fb = {}
    for name in linears:
        t = inits[name]
        if isinstance(t, CptcTensor):
            fb[name] = CptcFrozenCodeLinear(t)
        else:
            w = stager.to_host(layer.get_submodule(name).weight.detach().float())
            fb[name] = FakeQuantLinear(w, t, comps[name], "ptq")
    return fb


def _export_layer(fq, fb, info, stager) -> dict:
    """The layer artifact per checkpoint name.

    With zero optimizer steps there is no trained state: for CPTC the checkpoint must
    stay the PTQ encoding byte for byte, so it is exported from the frozen fallback
    instead of re-projecting the untouched initial latent. The projection is free to
    pick another code with the same or lower error, which would change the serialized
    bytes for nothing."""
    zero = info.get("steps", 0) == 0
    out = {}
    for name, m in fq.items():
        exact = fb.get(name) if (zero and fb) else None
        out[name] = (
            exact.export(stager)
            if isinstance(exact, CptcFrozenCodeLinear)
            else m.export(stager)
        )
    return out


# Fields added after the first release, with the value that reproduces the old
# behaviour: at that value they are left out of the fingerprint, so per-layer artifacts
# written before the field existed are still reused.
LEGACY_DEFAULTS = {
    "data_parallel": True,
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
    "dtype": "float32",
    "cptc_scale_units": "log",
    "cptc_projection_sweeps": 1,
}


def qad_meta(qc: QadConfig) -> dict:
    """``QadConfig`` for the checkpoint's ``quantization_config``: the low-bit options
    only when they differ from their default (a default run writes the same config)."""
    d = QadConfig()
    new = set(FINAL_FIELDS) | {
        "data_parallel",
        "train_norms",
        "lr_norm",
        "kl_layers",
        "kl_weight",
        "kl_tokens",
        "cptc_scale_units",
        "cptc_projection_sweeps",
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
    """Bytes of RAM a new allocation can use: ``MemAvailable`` (free memory plus the page cache the
    kernel gives back on demand). ``sysconf(SC_AVPHYS_PAGES)`` counts only never-used pages, which
    on a box that has read a few model files is a small fraction of the real headroom."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
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
    stager = PinnedWindow()
    for o, *_ in res:
        for i, v in o.items():
            dst[i].copy_(stager.to_host(v))
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

    check_cptc_options(qc, bool(getattr(cfg, "cptc", False)))
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
        + (f", math {qc.dtype}" if qc.dtype != "float32" else "")
        + (
            f", step budget {qc.budget:g} (early stop under {qc.min_gain:.1%} per {qc.eval_every or 'eval'} steps, "
            f"at least {budget.floor})"
            if not budget.full or qc.min_gain > 0
            else ""
        )
    )
    wstager = PinnedWindow()
    # Which layers will be trained: the ones whose artifact is missing or stale. The
    # fingerprint is the run's and the same for every layer, so one cheap read per
    # layer decides it. When none of them is trained the propagation is not needed at
    # all: it only feeds the training (the final logit-KL phase computes its own
    # teacher), and a restart that reuses every layer then costs minutes, not an hour.
    trained: set[int] = set()
    for li in todo:
        art = qdir / f"layer{li:03d}.pt"
        rec = torch.load(art, weights_only=False, mmap=True) if art.exists() else None
        if rec is None or rec.get("fingerprint") != fp:
            trained.add(li)
    propagate = bool(trained)
    if todo and not propagate:
        log("QAD: every layer comes from its artifact; the propagation is skipped")
    for li in range(n_layers):
        if event := getattr(log, "event", None):
            event("qad", "running", layer=li + 1, total_layers=n_layers)
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
        if propagate:
            _propagate(layer, ex, xt, kwargs, yt)
        art = qdir / f"layer{li:03d}.pt"
        rec = torch.load(art, weights_only=False) if art.exists() else None
        norms: dict = {}
        if rec is not None and rec.get("fingerprint") == fp:
            tensors, info = rec["tensors"], rec["info"]
            norms = rec.get("norms", {})
            log(f"QAD layer {li}: reusing {art.name}")
            if event := getattr(log, "event", None):
                event(
                    "qad",
                    "reused",
                    layer=li + 1,
                    total_layers=n_layers,
                    artifact=str(art),
                )
            if li in todo:
                budget.record(info)
        else:
            tl = time.time()
            reqs, loaded = [], {}
            for name, ckpt in linears.items():
                w = wstager.to_host(layer.get_submodule(name).weight.detach().float())
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
                from .cptc import CptcTensor

                def _fake_quant(name):
                    w = wstager.to_host(
                        layer.get_submodule(name).weight.detach().float()
                    )
                    if isinstance(inits[name], CptcTensor):
                        return CptcFakeQuantLinear(
                            w,
                            inits[name],
                            comps[name],
                            qc.init,
                            qc.latent_units,
                            sweeps=qc.cptc_projection_sweeps,
                            scale_units=qc.cptc_scale_units,
                        )
                    return FakeQuantLinear(
                        w, inits[name], comps[name], qc.init, qc.latent_units
                    )

                fq = {name: _fake_quant(name) for name in linears}
                fb = _ptq_fallback(layer, linears, inits, comps, qc, wstager)
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
                    ex.devices,
                )
                norms = info.pop("norms", {})
                tensors = {
                    linears[nm]: t
                    for nm, t in _export_layer(fq, fb, info, wstager).items()
                }
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
            atomic_torch_save(rec, art)
            left = len([i for i in todo if i > li])
            eta = (trained_time / max(trained_n, 1)) * left
            if event := getattr(log, "event", None):
                event(
                    "qad",
                    "completed" if li in train_set else "skipped",
                    layer=li + 1,
                    total_layers=n_layers,
                    step=info.get("steps", 0),
                    total_steps=info.get("allotted", 0),
                    eta_s=eta,
                    eta_scope="remaining selected layers",
                    artifact=str(art),
                    gpu_peak_gib=info.get("gpu_peak_gib"),
                )
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
            from .cptc import CptcTensor, dequantize_cptc

            tq = tensors[ckpt]
            mod.weight.data = (
                dequantize_cptc(tq) if isinstance(tq, CptcTensor) else dequantize(tq)
            ).to(mod.weight.dtype)
        norm_orig = {}
        for pn, val in norms.items():  # trained norms: the student uses the fp16 value
            prm = layer.get_parameter(pn)
            norm_orig[pn] = prm.data
            prm.data = val.to(torch.float16).to(prm.dtype)
            full = f"{prefix}{li}.{pn}"
            res.dense[(resolve(full) if resolve else full) or full] = val.to(
                torch.float16
            )
        if propagate:
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
    # The final logit-KL phase needs the teacher's LM-head input (the unquantized
    # model's last hidden state after the final norm). Placing the 27B teacher on the
    # GPUs does not fit four V620s (the fit check skipped it), but the layer-wise
    # propagation already computed it: after the last layer ``xt`` holds the
    # unquantized layer's output. Save it so the phase can run without the teacher
    # placement (hi2.finalkl, ``head_inputs``).
    if propagate and qc.final_kl > 0 and n_layers == len(layers) and xt:
        norm = final_norm(model, path)
        out = torch.empty((len(xt), *xt[0].shape[1:]), dtype=torch.float32)
        with torch.no_grad():
            for i, x in enumerate(xt):
                out[i].copy_(norm(x.float())[0])
        atomic_torch_save(
            {"fingerprint": fp, "head_inputs": out}, qdir / "head_inputs.pt"
        )
        log(
            f"QAD: saved the teacher's head inputs ({out.numel() * 4 / 2**30:.1f} GiB) "
            "for the final logit-KL phase"
        )
        del out
    return res
