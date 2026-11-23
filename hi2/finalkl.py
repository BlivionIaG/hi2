# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Final logit-KL phase after layer-wise QAD (spec 13 step 2, a small-parameter version).

The layer-wise phase fits each decoder layer's output (MSE). This phase distills the
whole network's output distribution instead: ``KL(teacher || student)`` of the next-token
logits on calibration windows, with the full student on one GPU. To fit a 29 GiB V620
with a 4B model it trains only

* the per-block ladder bytes ``(lo_u8, step_u8)`` of every quantized Linear (continuous
  values, rounded with a straight-through estimator, as in the layer-wise phase), with
  the **codes frozen** at the layer-wise result (packed as stored, ``nbits`` per weight
  on the GPU, unpacked per forward; no fp32 latents), and
* the RMSNorm weights (fp32, written to the checkpoint in fp16).

Every stored value is still the exact decode of spec 2.2 (``fp16(lo + code * step)``),
so the export is a normal v3.2 checkpoint. The teacher's last hidden states (the LM
head's input) are computed once with the unquantized model and kept in RAM; the teacher
logits are recomputed from them in chunks, so neither teacher nor student logits are
ever held for a whole sequence. Decoder layers run under activation checkpointing.
Keep-best on held-out windows: the phase never returns a state worse than its start.
"""

import contextlib
import math
import time

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .format import (
    HI2Tensor,
    assemble,
    decode_params,
    fwht_groups,
    ladder_dequantize,
    unpack_codes,
)
from .pinned import PinnedWindow, module_to_device, module_to_host
from .qad import _ste, is_norm_param

LOSS_CHUNK = 256  # positions per LM-head chunk in the loss


class _Stop(Exception):
    pass


class LadderLinear(nn.Module):
    """A quantized Linear with frozen codes and trainable ladder bytes: the weight is the
    exact HI2 decode, mapped back to the original basis (``H^-1(w) / s``)."""

    def __init__(self, t: HI2Tensor, bias: torch.Tensor | None = None):
        super().__init__()

        r, k = t.shape
        self.shape = (r, k)
        self.register_buffer("had", t.had.clone())
        self.has_scale = t.col_scale is not None
        self.register_buffer(
            "col_scale",
            t.col_scale.float() if self.has_scale else torch.ones(0),
        )
        self.register_buffer("row_nbits", t.row_nbits.clone())
        self.register_buffer("row_block", t.row_block.clone())
        nb = t.row_nbits.to(torch.int64)
        bl = t.row_block.to(torch.int64)
        self.cfgs = []
        codes, los, sts, consts, rows_all = [], [], [], [], []
        for nbits in sorted(set(nb.tolist())):
            for bs in sorted(set(bl.tolist())):
                rows = ((nb == nbits) & (bl == bs)).nonzero().flatten()
                if rows.numel() == 0:
                    continue
                wr = k * nbits // 32
                widx = t.code_off[rows].to(torch.int64).unsqueeze(1) + torch.arange(wr)
                # the packed words as stored (int32): ``unpack_codes`` per forward, so a
                # 4-bit row costs half a byte on the device, a 2-bit row a quarter
                codes.append(t.codes[widx].contiguous())
                nblk = k // bs
                pidx = (
                    t.param_off[rows].to(torch.int64).unsqueeze(1) + torch.arange(nblk)
                ).unsqueeze(-1) * 2 + torch.arange(2)
                p = t.blk_params[pidx]
                los.append(nn.Parameter(p[..., 0].float()))
                sts.append(nn.Parameter(p[..., 1].float()))
                consts.append(t.row_consts[rows].to(torch.float16))
                rows_all.append(rows)
                self.cfgs.append((nbits, bs))
        self.lo_u = nn.ParameterList(los)
        self.step_u = nn.ParameterList(sts)
        for i, (c, rc, rw) in enumerate(zip(codes, consts, rows_all)):
            self.register_buffer(f"codes{i}", c)
            self.register_buffer(f"consts{i}", rc)
            self.register_buffer(f"rows{i}", rw)
        self.register_buffer("inv_perm", torch.argsort(torch.cat(rows_all)))
        self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)

    def codes_u8(self, i):
        """The codes of group ``i`` unpacked, ``[rows, K]`` uint8."""
        return unpack_codes(getattr(self, f"codes{i}"), self.cfgs[i][0]).to(torch.uint8)

    def _bytes(self, i):
        return (
            torch.round(self.lo_u[i].detach()).clamp(0, 255),
            torch.round(self.step_u[i].detach()).clamp(0, 255),
        )

    @torch.no_grad()
    def project(self):
        for p in (*self.lo_u, *self.step_u):
            p.clamp_(0, 255)

    def stored(self) -> torch.Tensor:
        """Stored-basis weights ``[R, K]`` (fp32 values of the exact fp16 decode) with
        the straight-through graph to the bytes."""
        k = self.shape[1]
        parts = []
        for i, (_nbits, bs) in enumerate(self.cfgs):
            c = getattr(self, f"consts{i}")
            q = self.codes_u8(i)
            lo_b, st_b = self._bytes(i)
            lo_q, st_q = decode_params(lo_b, st_b, c[:, 0:1], c[:, 1:2], c[:, 2:3])
            qb = q.reshape(q.shape[0], k // bs, bs)
            with torch.no_grad():
                val = ladder_dequantize(qb, lo_q, st_q).float()
            lo_c = self.lo_u[i] * c[:, 1:2].float() + c[:, 0:1].float()
            st_c = self.step_u[i] * c[:, 2:3].float()
            lo_h = _ste(lo_q.float(), lo_c)
            st_h = _ste(st_q.float(), st_c)
            y = lo_h.unsqueeze(-1) + qb.float() * st_h.unsqueeze(-1)
            parts.append(_ste(val, y).reshape(-1, k))
        return torch.cat(parts)[self.inv_perm]

    def weight(self) -> torch.Tensor:
        w = self.stored()
        if bool(self.had.any()):
            w = fwht_groups(w, self.had)
        if self.has_scale:
            w = w / self.col_scale
        return w

    def forward(self, x):
        return F.linear(x, self.weight().to(x.dtype), self.bias)

    @torch.no_grad()
    def export(self, stager=None) -> HI2Tensor:
        pieces = []
        to_host = (
            (lambda t: stager.to_host(t)) if stager is not None else (lambda t: t.cpu())
        )
        for i in range(len(self.cfgs)):
            lo_b, st_b = self._bytes(i)
            pieces.append(
                (
                    to_host(getattr(self, f"rows{i}")),
                    to_host(self.codes_u8(i)),
                    to_host(torch.stack([lo_b, st_b], dim=-1).to(torch.uint8)),
                    to_host(getattr(self, f"consts{i}")),
                )
            )
        return assemble(
            pieces,
            to_host(self.row_nbits),
            to_host(self.row_block),
            to_host(self.had),
            self.shape[1],
            to_host(self.col_scale.to(torch.float16)) if self.has_scale else None,
        )


def _head_input(model: nn.Module, ids: torch.Tensor) -> torch.Tensor:
    """Run ``model`` up to its LM head and return the head's input ``[T, H]``."""
    got = []

    def grab(_m, args):
        got.append(args[0])
        raise _Stop

    h = model.get_output_embeddings().register_forward_pre_hook(grab)
    try:
        with contextlib.suppress(_Stop):
            model(ids, use_cache=False)
    finally:
        h.remove()
    return got[0][0]


class _Logits(nn.Module):
    """The head's input to the logits the model's loss sees: the output embedding, then
    the model's own logit soft-cap (Inkling-2: ``logit_softcap``, ``tanh(x / c) * c`` on
    the float32 logits) when it has one. The children are the model's, not copies."""

    def __init__(self, head: nn.Module, cap: nn.Module | None):
        super().__init__()
        self.head, self.cap = head, cap

    def forward(self, h):
        out = self.head(h)
        return out if self.cap is None else self.cap(out.float())


def _logits_fn(model: nn.Module) -> nn.Module:
    head = model.get_output_embeddings()
    cap = getattr(model, "logit_softcap", None)
    # only a module is the model's own soft-cap op (another family may keep a number)
    return _Logits(head, cap) if isinstance(cap, nn.Module) else head


def _kl_loss(lm: nn.Module, hs: torch.Tensor, ht: torch.Tensor, pos=None, grad=True):
    """Mean over positions of KL(teacher || student); logits in chunks of LOSS_CHUNK,
    each chunk under activation checkpointing so only its hidden states are kept."""
    if pos is not None:
        hs, ht = hs[pos], ht[pos]
    n = hs.shape[0]

    def chunk(a, b):
        with torch.no_grad():
            lt = torch.log_softmax(lm(b).float(), -1)
        ls = torch.log_softmax(lm(a).float(), -1)
        return (lt.exp() * (lt - ls)).sum()

    total = 0.0
    for s in range(0, n, LOSS_CHUNK):
        a, b = hs[s : s + LOSS_CHUNK], ht[s : s + LOSS_CHUNK]
        if grad:
            v = checkpoint(chunk, a, b, use_reentrant=False) / n
            v.backward()
            total += v.item()
        else:
            total += chunk(a, b).item() / n
    return total


@contextlib.contextmanager
def _checkpointed(layers: nn.ModuleList):
    """Run every decoder layer under activation checkpointing (inputs kept, the rest
    recomputed in the backward)."""
    saved = []
    for layer in layers:
        fwd = layer.forward

        def wrapped(*args, _fwd=fwd, **kwargs):
            return checkpoint(_fwd, *args, use_reentrant=False, **kwargs)

        saved.append(fwd)
        layer.forward = wrapped
    try:
        yield
    finally:
        for layer, fwd in zip(layers, saved):
            layer.forward = fwd


def final_kl(
    model: nn.Module,
    ids: torch.Tensor,
    train_idx: list[int],
    val_idx: list[int],
    tensors: dict[str, HI2Tensor],
    dense: dict[str, torch.Tensor],
    qc,
    devs: torch.device | list[torch.device],
    resolve,
    log=print,
    head_inputs: torch.Tensor | None = None,
) -> tuple[dict, dict, dict]:
    """Train ladder bytes and norm weights of the whole student for ``qc.final_kl``
    steps; returns ``(tensors, dense, info)`` with the kept state (``tensors`` and
    ``dense`` updated copies of the inputs). ``model`` is the unquantized model (in RAM);
    it is modified in place (its quantized Linears become :class:`LadderLinear`). With
    several ``devs`` the teacher and the student are split layer-wise over them
    (:func:`hi2.sensitivity.shard_layers`), so models larger than one GPU train too.
    ``head_inputs`` (``[n, 1, T, H]``, the unquantized model's post-final-norm hidden
    states) skips the teacher placement; the layer-wise phase saves it
    (``<workdir>/qad/head_inputs.pt``) because a 27B teacher does not fit the devices."""
    from .layerwise import find_layers
    from .looped import is_looped
    from .moe import UnfusedExperts
    from .sensitivity import shard_layers

    devs = [devs] if isinstance(devs, torch.device) else list(devs)
    dev, last = devs[0], devs[-1]

    def place():
        if len(devs) == 1:
            module_to_device(model, dev)
            return []
        return shard_layers(model, devs)

    t0 = time.time()
    torch.manual_seed(qc.seed)
    lm = _logits_fn(model)
    stager = PinnedWindow()
    # 1. teacher: the LM head's input on every window (unquantized model, on the GPU),
    # or the head inputs the layer-wise phase saved (no teacher placement).
    if head_inputs is not None:
        teach = {i: head_inputs[i] for i in sorted(set(train_idx) | set(val_idx))}
        t_teacher = 0.0
    else:
        hooks = place()
        teach = {}
        with torch.no_grad():
            for i in sorted(set(train_idx) | set(val_idx)):
                teach[i] = stager.to_host(
                    _head_input(model, ids[i : i + 1].to(dev)).float()
                )
        for h in hooks:
            h.remove()
        module_to_host(model, stager)
        t_teacher = time.time() - t0
    # 2. student: quantized Linears -> LadderLinear, norms from the layer-wise phase
    mods = {}
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Linear):
            ckpt = resolve(f"{name}.weight")
            if ckpt in tensors:
                b = mod.bias.detach().float() if mod.bias is not None else None
                ll = LadderLinear(tensors[ckpt], b)
                model.set_submodule(name, ll)
                mods[ckpt] = ll
    if is_looped(model):
        # 128 experts per layer: the weight of every expert Linear is rebuilt from its
        # ladder in the backward instead of being kept (about 6 GiB per layer of the
        # real model), see UnfusedExperts
        for m in model.modules():
            if isinstance(m, UnfusedExperts):
                m.checkpoint_experts = True
    norms = {}
    for name, p in model.named_parameters():
        if is_norm_param(name, p):
            ckpt = resolve(name) or name
            if ckpt in dense:  # trained in the layer-wise phase (stored in fp16)
                p.data = dense[ckpt].to(p.dtype)
            norms[ckpt] = p
    for p in model.parameters():
        p.requires_grad_(False)
    groups = []
    if qc.final_bytes:
        byte_p = [p for m in mods.values() for p in (*m.lo_u, *m.step_u)]
        for p in byte_p:
            p.requires_grad_(True)
        groups.append({"params": byte_p, "lr": qc.final_lr_p})
    if qc.final_norms:
        for p in norms.values():
            p.data = p.data.float()
            p.requires_grad_(True)
        groups.append({"params": list(norms.values()), "lr": qc.final_lr_norm})
    place()
    model.eval()
    _, layers = find_layers(model)
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    steps = qc.final_kl
    warm = max(1, int(qc.warmup * steps))

    def lr_scale(s):
        if s < warm:
            return (s + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)
    trainable = [p for g in groups for p in g["params"]]

    def snapshot():
        return [p.detach().to("cpu", copy=True) for p in trainable]

    def restore(st):
        with torch.no_grad():
            for p, v in zip(trainable, st):
                p.copy_(v.to(p.device))

    n_tok = ids.shape[1]
    k_tok = min(qc.final_tokens, n_tok) if qc.final_tokens > 0 else n_tok
    gen = torch.Generator().manual_seed(qc.seed + 2)

    def positions():
        if k_tok >= n_tok:
            return None
        return torch.randperm(n_tok, generator=gen)[:k_tok].sort().values.to(last)

    @torch.no_grad()
    def val_kl() -> float:
        tot = 0.0
        for i in val_idx:
            hs = _head_input(model, ids[i : i + 1].to(dev)).float()
            tot += _kl_loss(lm, hs, teach[i].to(last), None, grad=False)
        return tot / max(len(val_idx), 1)

    acc = getattr(torch, "accelerator", None)
    on_gpu = dev.type != "cpu" and acc is not None
    if on_gpu:
        for d in devs:
            torch.cuda.reset_peak_memory_stats(d)
    v0 = val_kl()
    best, best_state = v0, snapshot()
    log(
        f"final KL: {len(mods)} quantized Linears, {len(norms)} norms, "
        f"{sum(p.numel() for p in trainable) / 1e6:.1f}M trainable values, {steps} steps x "
        f"{qc.final_batch} windows, held-out KL {v0:.4f} (teacher pass {t_teacher:.0f} s)"
    )
    every = max(1, steps // max(1, qc.final_evals))
    order: list[int] = []
    g = torch.Generator().manual_seed(qc.seed)
    trace = [(0, None, v0)]
    with _checkpointed(layers):
        for step in range(steps):
            if len(order) < qc.final_batch:
                order += [
                    train_idx[j]
                    for j in torch.randperm(len(train_idx), generator=g).tolist()
                ]
            batch, order = order[: qc.final_batch], order[qc.final_batch :]
            tot = 0.0
            for i in batch:
                hs = _head_input(model, ids[i : i + 1].to(dev)).float()
                hd = hs.detach().requires_grad_(True)
                tot += _kl_loss(lm, hd, teach[i].to(last), positions()) / len(batch)
                hs.backward(hd.grad / len(batch))
            # norms outside the decoder path (e.g. a vision tower) get no gradient
            if step == 0:
                touched = {c for c, p in norms.items() if p.grad is not None}
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            for m in mods.values():
                m.project()
            if (step + 1) % every == 0 or step + 1 == steps:
                vl = val_kl()
                trace.append((step + 1, tot, vl))
                if vl < best:
                    best, best_state = vl, snapshot()
                log(
                    f"  final KL step {step + 1}/{steps} train {tot:.4f} held-out {vl:.4f} "
                    f"({(time.time() - t0) / 60:.1f} min)"
                )
                if event := getattr(log, "event", None):
                    event(
                        "final_kl",
                        "running",
                        step=step + 1,
                        total_steps=steps,
                        elapsed_s=time.time() - t0,
                        validation_kld=vl,
                    )
    restore(best_state)
    out_t = dict(tensors)
    for ckpt, m in mods.items():
        out_t[ckpt] = m.export(stager)
    out_d = dict(dense)
    if qc.final_norms and steps > 0:
        for ckpt, p in norms.items():
            if ckpt in touched:
                out_d[ckpt] = p.detach().to(torch.float16).cpu()
    info = {
        "val_before": v0,
        "val_after": best,
        "kept": "init" if best == v0 else "best",
        "steps": steps,
        "seconds": time.time() - t0,
        "teacher_seconds": t_teacher,
        "gpu_peak_gib": max(torch.cuda.max_memory_allocated(d) for d in devs) / 2**30
        if on_gpu
        else None,
        "trace": trace,
    }
    log(
        f"final KL: held-out KL {v0:.4f} -> {best:.4f} ({100 * (best / max(v0, 1e-30) - 1):+.1f}%), "
        f"{info['seconds'] / 60:.1f} min"
    )
    return out_t, out_d, info


ADAM_STATE_BYTES = (
    12  # gradient + exp_avg + exp_avg_sq (fp32 each), beside the parameter
)
# What one decoder layer's backward needs beyond the steady state (rebuilt expert
# weights one expert at a time, the attention and the routed activations of the
# recomputed layer); an estimate, not a measurement: the GPU smoke measures it.
TRANSIENT_GIB = 3.0


def tensor_student_bytes(t: HI2Tensor, train_bytes: bool = True) -> int:
    """Bytes the :class:`LadderLinear` of ``t`` holds on its device: the packed code
    words, the ladder values as fp32 parameters (with their gradient and the two Adam
    moments when trained), and the per-row buffers."""
    r, k = t.shape
    ladder = t.blk_params.numel()  # (lo_u8, step_u8) per block: one fp32 value each
    per_row = 6 + 8 + 8 + 2  # consts fp16 x3, rows and inv_perm int64, nbits and block
    scale = 4 * k if t.col_scale is not None else 0
    return (
        4 * t.codes.numel()
        + ladder * (4 + (ADAM_STATE_BYTES if train_bytes else 0))
        + per_row * r
        + k // 16  # had flags
        + scale
    )


def student_footprint(
    model: nn.Module, tensors, resolve, qc, seq_len: int, n_devices: int
):
    """Where the bytes of the final-KL student go over ``n_devices`` (the split of
    :func:`hi2.sensitivity.shard_layers`), before anything is placed.

    The student is ``model`` with every Linear in ``tensors`` a :class:`LadderLinear`:
    ``static`` is the bytes each device holds all along (parameters, buffers, gradients
    and optimizer state), ``activations`` the inputs the decoder layers keep for one
    sequence's backward (they run under activation checkpointing), ``transient`` the
    extra one layer's backward needs and ``logits`` the loss chunk on the last device.
    ``static`` is exact (a test measures it); the rest is derived."""
    from .layerwise import find_layers
    from .looped import is_looped, saved_input_bytes

    path, layers = find_layers(model)
    prefix = path + "."
    base = path.rsplit(".", 1)[0] if "." in path else ""
    parent = model.get_submodule(base) if base else model
    head = model.get_output_embeddings()
    tails = [
        m
        for m in (
            head,
            getattr(parent, "norm", None),
            getattr(parent, "final_input", None),
        )
        if m is not None
    ]
    tail_ids = {id(p) for m in tails for p in m.parameters()}
    layer_ids = {id(p) for p in layers.parameters()}

    def dense(name, p):
        if qc.final_norms and is_norm_param(name, p):  # fp32 value, gradient, moments
            return p.numel() * (4 + ADAM_STATE_BYTES)
        return p.numel() * p.element_size()

    per_layer = []
    for li, layer in enumerate(layers):
        total = 0
        for name, p in layer.named_parameters():
            full = f"{prefix}{li}.{name}"
            if not (name.endswith(".weight") and resolve(full) in tensors):
                total += dense(full, p)
        for name, mod in layer.named_modules():
            if isinstance(mod, nn.Linear):
                ckpt = resolve(f"{prefix}{li}.{name}.weight")
                if ckpt in tensors:
                    total += tensor_student_bytes(tensors[ckpt], qc.final_bytes)
        per_layer.append(total)
    first = tail = 0
    for name, p in model.named_parameters():
        if id(p) in layer_ids:
            continue
        if id(p) in tail_ids:
            tail += dense(name, p)
        else:
            first += dense(name, p)
    n = n_devices
    device_of = [
        next(k for k in range(n) if i < len(layers) * (k + 1) // n)
        for i in range(len(layers))
    ]
    static = [0] * n
    static[0] += first
    static[-1] += tail
    for li, b in enumerate(per_layer):
        static[device_of[li]] += b
    db = head.weight.element_size()
    hidden = head.weight.shape[1]
    if is_looped(model):
        calls = saved_input_bytes(parent, seq_len, db)
    else:
        calls = [(li, seq_len * hidden * db) for li in range(len(layers))]
    act = [0] * n
    for li, b in calls:
        act[device_of[li]] += b
    if is_looped(model):  # the final attention residual's float32 stack of block states
        blocks = 1 + sum(1 for layer in layers if layer.is_block_boundary)
        act[-1] += 3 * seq_len * blocks * hidden * 4
    logits = 3 * LOSS_CHUNK * head.weight.shape[0] * 4
    return {
        "static": static,
        "activations": act,
        "transient": int(TRANSIENT_GIB * 2**30),
        "logits": logits,
        "devices": [
            s + a + int(TRANSIENT_GIB * 2**30) + (logits if k == n - 1 else 0)
            for k, (s, a) in enumerate(zip(static, act))
        ],
    }
