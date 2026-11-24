# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Layer-wise QAD on a looped model (Inkling-2): teacher-forced layer calls.

:func:`hi2.qad.qad_layerwise` trains a layer from its own input (the quantized prefix's
output) to the unquantized layer's output. A looped model's layer also takes the block
states of its attention residuals, its loop span runs several times, and the later passes
receive messages computed from the pass before (``hi2.looped``): the generic driver
cannot feed those inputs. This module keeps its recipe and trains every layer call
**teacher-forced** from the trajectory of the unquantized layer program:

* one pass of the program (:func:`hi2.looped.run_program`) records the input and output
  of every call the selected layers need. A layer of 0-5 or 18-23 has one call; a layer
  of the loop span has one call per pass and the same :class:`hi2.qad.FakeQuantLinear`
  set, so its samples are ``(window, pass)`` and one optimizer step sees all passes of a
  window (``train_layer(groups=...)``: the same weights, the loss summed over the
  passes). Each pass is fed the teacher's input for that pass; the messages of pass 2
  and later are recomputed from the stored block states and final hidden state of the
  pass before, exactly as :func:`hi2.looped.run_program` does;
* the teacher states are kept in host RAM, in the model's compute dtype (float16 on the
  V620), one tensor per **unique** call input: a target is the next call's input when
  the program continues, pass starts reuse the pre-loop state, and only the end of a
  pass and the final output are extra. The 48 calls of Inkling-2-Mini are 37 states in
  the loop span plus the surrounding layers (49 for every layer, 111 GiB for 264
  windows x 2048 tokens in float16); :func:`_state_count` computes the number before
  the pass and refuses over ``HI2_QAD_STATES_GIB`` (default 128);
* the routing experts of one layer are 19 GiB of trainable state (latent, unit,
  gradient and two Adam moments per weight). With ``QadConfig.expert_groups`` (0: as
  many as keep a group under 300 M weights) they train group by group; the other
  groups stay at their current decode (PTQ, or the trained result of an earlier group)
  and take no optimizer state.

The per-layer artifacts ``qad/layerNNN.pt`` and the resume of :func:`qad_layerwise`
apply to the layers that train; a layer that is only PTQ-encoded is deterministic and is
encoded again on a re-run. With ``final_kl`` the pass that runs to the end of the
program also saves the teacher's head inputs for :mod:`hi2.finalkl`.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import gc
import math
import os
import re
import time
from pathlib import Path

import torch
from torch import nn

from .cache import StatsView
from .calibrate import Config
from .format import dequantize
from .layerwise import (
    _Stop,
    _layer_linears,
    _rss_gib,
    find_layers,
    to_device,
)
from .looped import _Program, run_program, schedule
from .pinned import PinnedWindow
from .qad import (
    QadConfig,
    QadResult,
    StepBudget,
    fake_quants,
    init_layer,
    split_indices,
    train_layer,
)
from .runtime import atomic_torch_save

#: One training group (the live fake-quantised experts) stays under this many weights:
#: 20 bytes each (latent, unit, gradient, two Adam moments) is 6 GB, plus the decode.
GROUP_WEIGHTS = 300_000_000

EXPERT_NAME = re.compile(r"(?:^|\.)experts\.(\d+)\.")


# ---- the calls of the layer program ---------------------------------------------


@dataclasses.dataclass
class _Layout:
    """How the calls of the program relate to each other."""

    calls_of: dict[int, list[int]]  # layer -> call indices, in program order
    blocks_of: list[list[int]]  # call -> call indices whose inputs are its block states
    pass_of: list[int]  # call -> pass over the loop span (0 for the pre-loop calls)
    pass_end: list[int]  # pass p's last span call (p = 0 .. passes - 2)
    prev_blocks: dict[int, list[int]]  # a message call -> the previous pass's blocks
    n_pre: int  # block states before the loop span


def _layout(calls, layers, cfg) -> _Layout:
    """The block-state and message references of :func:`hi2.looped.run_program`,
    without running it: the same reset at a pass start, the same append at a block
    boundary, and the ``n_pre + 1`` slice the model takes for the messages."""
    a, b = cfg.loop_layer_index_start, cfg.loop_layer_index_end
    calls_of: dict[int, list[int]] = {}
    blocks_of: list[list[int]] = []
    pass_of: list[int] = []
    pass_end: list[int] = []
    blocks: list[int] = []
    pre: list[int] | None = None
    p = 0
    blocks_after: dict[int, list[int]] = {}
    for ci, c in enumerate(calls):
        calls_of.setdefault(c.layer, []).append(ci)
        if c.layer == a:
            if not c.messages:
                pre = list(blocks)
                p = 0
            else:
                blocks = list(pre or [])
                p = len(pass_end)
        pass_of.append(p)
        blocks_of.append(list(blocks))
        if layers[c.layer].is_block_boundary:
            blocks.append(ci)
        nxt = calls[ci + 1] if ci + 1 < len(calls) else None
        if nxt is None or nxt.layer != c.layer + 1:
            if a <= c.layer < b:  # a pass of the span ends here
                pass_end.append(ci)
                blocks_after[ci] = list(blocks)
    n_pre = len(pre or [])
    prev_blocks = {}
    for ci, c in enumerate(calls):
        if c.messages:
            end = pass_end[pass_of[ci] - 1]
            prev_blocks[ci] = blocks_after[end][n_pre + 1 :]
    return _Layout(calls_of, blocks_of, pass_of, pass_end, prev_blocks, n_pre)


def _needed(layout: _Layout, calls, selected: list[int]) -> tuple[set[int], set[int]]:
    """The call inputs and outputs the selected layers' training needs, closed over the
    block states of those calls and the messages of the later passes."""
    inputs: set[int] = set()
    outputs: set[int] = set()
    for li in selected:
        inputs.update(layout.calls_of[li])
        outputs.update(layout.calls_of[li])
    changed = True
    while changed:
        changed = False
        for ci in list(inputs):
            for src in layout.blocks_of[ci]:
                if src not in inputs:
                    inputs.add(src)
                    changed = True
            if calls[ci].messages:
                end = layout.pass_end[layout.pass_of[ci] - 1]
                if end not in outputs:
                    outputs.add(end)
                    changed = True
                for src in layout.prev_blocks[ci]:
                    if src not in inputs:
                        inputs.add(src)
                        changed = True
    return inputs, outputs


def _state_count(calls, inputs: set[int], outputs: set[int], a: int) -> int:
    """The number of unique call states the inputs and outputs need (the identity the
    capture dedupes on). A later pass starts from the first pass's pre-loop state; an
    output that the program continues with is the next call's input."""
    first = next(
        (ci for ci, c in enumerate(calls) if c.layer == a and not c.messages),
        None,
    )
    keys: set[tuple] = set()

    def key(ci: int):
        c = calls[ci]
        if c.messages and c.layer == a and first is not None:
            return ("call", first)
        return ("call", ci)

    for ci in inputs:
        keys.add(key(ci))
    for ci in outputs:
        nxt = ci + 1
        if nxt < len(calls) and calls[nxt].layer == calls[ci].layer + 1:
            keys.add(key(nxt))
        else:
            keys.add(("out", ci))
    return len(keys)


class _Teacher:
    """The teacher's per-call states, one CPU tensor per window, deduplicated."""

    def __init__(self, n_seq: int):
        self.n_seq = n_seq
        self.slots: list[list[torch.Tensor]] = []
        self._by = {}
        self.inputs: dict[int, int] = {}  # call -> slot
        self.outputs: dict[int, int] = {}  # call -> slot

    def add(self, states: list[torch.Tensor]) -> int:
        k = tuple(id(t) for t in states)
        slot = self._by.get(k)
        if slot is None:
            slot = len(self.slots)
            self.slots.append(list(states))
            self._by[k] = slot
        return slot

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for s in self.slots for t in s)

    def __del__(self):
        self.slots.clear()


def _capture(prog: _Program, selected: list[int], qc: QadConfig, qdir: Path, fp: str, log):
    """Run the unquantized layer program once and record the states the selected layers
    need. Returns the :class:`_Teacher` (or None when no layer is selected)."""
    if not selected:
        return None
    calls = prog.calls
    cfg = prog.text.config
    layout = _layout(calls, prog.layers, cfg)
    inputs, outputs = _needed(layout, calls, selected)
    n_slots = _state_count(calls, inputs, outputs, cfg.loop_layer_index_start)
    hidden = prog.hidden[0]
    per = hidden[0].numel() * hidden[0].element_size()  # [T, H] of one window
    need = n_slots * prog.n_seq * per
    cap = float(os.environ.get("HI2_QAD_STATES_GIB", "128")) * 2**30
    log(
        f"QAD looped: {len(selected)} layer(s), {len(inputs)} call inputs and "
        f"{len(outputs)} outputs -> {n_slots} unique teacher states, "
        f"{need / 2**30:.1f} GiB of host RAM (cap {cap / 2**30:.0f} GiB)"
    )
    if need > cap:
        raise SystemExit(
            f"the teacher states of this selection need {need / 2**30:.1f} GiB "
            f"(HI2_QAD_STATES_GIB={cap / 2**30:.0f}); lower --qad-samples / --seq-len "
            "or raise HI2_QAD_STATES_GIB"
        )
    teacher = _Teacher(prog.n_seq)
    stop_at = max(inputs | outputs)
    last = len(calls) - 1
    reached_end = stop_at == last
    t0 = time.perf_counter()

    def call(ci, c, h, bs, messages):
        if ci in inputs:
            teacher.inputs[ci] = teacher.add(h)
        out, _ = prog.run(ci, h, bs, messages)
        if ci in outputs:
            teacher.outputs[ci] = teacher.add(out)
        if (ci + 1) % 12 == 0 or ci == stop_at:
            log(
                f"  teacher call {ci + 1}/{stop_at + 1}: {time.perf_counter() - t0:.0f}s, "
                f"{len(teacher.slots)} states"
            )
        if ci == stop_at and not reached_end:
            raise _Stop
        return out

    h, bs = None, None
    with contextlib.suppress(_Stop):
        h, bs = run_program(prog.text, prog.hidden, call)
    got = len(teacher.slots)
    if got > n_slots:
        raise RuntimeError(f"{got} teacher states for the {n_slots} the count promised")
    log(
        f"QAD looped: the teacher pass took {time.perf_counter() - t0:.0f}s; "
        f"{got} states, {teacher.nbytes() / 2**30:.1f} GiB, rss {_rss_gib()} GiB"
    )
    if reached_end and qc.final_kl > 0:
        t1 = time.perf_counter()
        heads = prog.heads(h, bs)
        out = torch.empty((prog.n_seq, *heads[0].shape[1:]), dtype=torch.float32)
        with torch.no_grad():
            for i, x in enumerate(heads):
                out[i].copy_(x.float()[0])
        atomic_torch_save(
            {"fingerprint": fp, "head_inputs": out}, Path(qdir) / "head_inputs.pt"
        )
        log(
            f"QAD looped: saved the teacher's head inputs "
            f"({out.numel() * 4 / 2**30:.1f} GiB) for the final logit-KL phase "
            f"({time.perf_counter() - t1:.0f}s)"
        )
        del out
    return teacher


# ---- training samples ------------------------------------------------------------


class _Samples:
    """The per-sample keyword inputs of a layer's calls: the recorded call kwargs, the
    block states of that call, and the messages of a later pass recomputed from the
    teacher's previous pass (``hi2.looped._Messages`` does the same at run time)."""

    def __init__(self, prog, teacher, layout: _Layout, samples, devices):
        self.prog, self.teacher, self.layout, self.samples = (
            prog,
            teacher,
            layout,
            samples,
        )
        self.a = prog.text.config.loop_layer_index_start
        self.stager = PinnedWindow()
        cis = {ci for _, ci in samples}
        self.base = {
            (ci, d): to_device(prog.calls[ci].kwargs, d) for ci in cis for d in devices
        }
        self.msg = {}
        for d in devices:
            m = copy.deepcopy(prog.text.message_input).to(d).eval()
            for p in m.parameters():
                p.requires_grad_(False)
            m.requires_grad_(False)
            self.msg[d] = m

    def _blocks(self, ci: int, w: int, dev, dtype):
        # a call's input is stored [1, T, H]; a block state is one more dimension
        parts = [
            self.stager.to_device(
                self.teacher.slots[self.teacher.inputs[s]][w], dev
            ).unsqueeze(-2)
            for s in self.layout.blocks_of[ci]
        ]
        x = self.teacher.slots[self.teacher.inputs[ci]][w]
        if parts:
            bs = torch.cat(parts, dim=-2)
        else:
            bs = x.new_empty((x.shape[0], x.shape[1], 0, x.shape[2]), device=dev)
        return bs if bs.dtype == dtype else bs.to(dtype)

    def _messages(self, ci: int, w: int, dev, dtype):
        lay = self.layout
        end = lay.pass_end[lay.pass_of[ci] - 1]
        prev = self.stager.to_device(self.teacher.slots[self.teacher.outputs[end]][w], dev)
        parts = [
            self.stager.to_device(
                self.teacher.slots[self.teacher.inputs[s]][w], dev
            ).unsqueeze(-2)
            for s in lay.prev_blocks[ci]
        ]
        if parts:
            blocks = torch.cat(parts, dim=-2)
        else:
            blocks = prev.new_empty(
                (prev.shape[0], prev.shape[1], 0, prev.shape[2]), device=dev
            )
        k = self.prog.calls[ci].layer - self.a
        with torch.no_grad():
            return self.msg[dev](blocks, prev, k).detach().to(dtype)

    def kwargs(self, i: int, dev, dtype) -> dict:
        w, ci = self.samples[i]
        kw = dict(self.base[(ci, dev)])
        kw["block_states"] = self._blocks(ci, w, dev, dtype)
        if self.prog.calls[ci].messages:
            kw["messages"] = self._messages(ci, w, dev, dtype)
        return kw


def _samples_for(teacher: _Teacher, layout: _Layout, li: int, qc: QadConfig, n_seq: int):
    """``(samples, xs, ys, train_idx, val_idx, groups)`` of one layer: one sample per
    window for a single-call layer, one per (window, pass) for a loop layer."""
    calls = layout.calls_of[li]
    samples = [(w, ci) for w in range(n_seq) for ci in calls]
    xs = [teacher.slots[teacher.inputs[ci]][w] for w, ci in samples]
    ys = [teacher.slots[teacher.outputs[ci]][w] for w, ci in samples]
    tr, va = split_indices(n_seq, qc)
    n = len(calls)
    train_idx = [w * n + p for w in tr for p in range(n)]
    val_idx = [w * n + p for w in va for p in range(n)]
    groups = None if n == 1 else [w for w in range(n_seq) for _ in range(n)]
    return samples, xs, ys, train_idx, val_idx, groups


# ---- the experts of one layer ----------------------------------------------------


def _group_names(layer, linears: dict[str, str], n_groups: int):
    """``(groups, count)``: the module names in the order they train. Experts are
    interleaved over the groups (their routed load differs, their state does not), the
    other Linears of the layer train in the first group; 0 groups means as many as keep
    one under :data:`GROUP_WEIGHTS`."""
    experts: dict[int, list[str]] = {}
    other: list[str] = []
    for name in linears:
        m = EXPERT_NAME.search(name)
        if m:
            experts.setdefault(int(m.group(1)), []).append(name)
        else:
            other.append(name)
    if not experts:
        return [list(linears)] if linears else [], 1
    total = sum(
        layer.get_submodule(n).weight.numel()
        for names in experts.values()
        for n in names
    )
    if n_groups <= 0:
        n_groups = math.ceil(total / GROUP_WEIGHTS)
    n_groups = max(1, min(int(n_groups), len(experts)))
    ids = sorted(experts)
    groups = []
    for g in range(n_groups):
        names = [n for e in ids[g::n_groups] for n in experts[e]]
        if g == 0:
            names = other + names
        groups.append(names)
    return [g for g in groups if g], n_groups


def _merge_infos(infos: list[dict]) -> dict:
    """One layer's info from its groups' (first ``val_before``, last ``val_after``,
    summed steps and seconds, the worst peak, the per-group records)."""
    first, last = infos[0], infos[-1]
    out = {k: v for k, v in first.items() if k != "trace"}
    out.update(
        val_after=last["val_after"],
        val_final=last["val_final"],
        train_last=last["train_last"],
        kept=last["kept"],
        steps=sum(i["steps"] for i in infos),
        allotted=sum(i["allotted"] for i in infos),
        early_stop=any(i["early_stop"] for i in infos),
        seconds=sum(i["seconds"] for i in infos),
        gpu_peak_gib=max(
            (i["gpu_peak_gib"] for i in infos if i["gpu_peak_gib"] is not None),
            default=None,
        ),
        groups=len(infos),
        group_sizes=[i.get("group_size") for i in infos],
        kept_per_group=[i["kept"] for i in infos],
    )
    return out


def _train_groups(
    layer,
    li: int,
    linears: dict[str, str],
    inits: dict,
    comps: dict,
    qc: QadConfig,
    ex,
    steps: int,
    samples,
    xs,
    ys,
    train_idx,
    val_idx,
    groups_of_samples,
    calls,
    log,
    wstager: PinnedWindow,
    layer_prefix: str,
    resolve,
):
    """Train a layer's Linears group by group; the other groups stay at their current
    decode. Returns ``({checkpoint name: HI2Tensor}, info, {param name: fp16 value})``."""
    groups, n_groups = _group_names(layer, linears, qc.expert_groups)
    tensors = {linears[n]: t for n, t in inits.items()}
    if not groups:
        return tensors, {"kept": "ptq", "seconds": 0.0}, {}
    shares = [steps // len(groups)] * len(groups)
    shares[-1] += steps % len(groups)
    orig = {n: layer.get_submodule(n).weight.data for n in linears}
    orig_norms = {}
    current = {
        n: dequantize(inits[n]).to(orig[n].dtype) for n in linears
    }
    dense: dict[str, torch.Tensor] = {}
    infos = []
    try:
        for gi, names in enumerate(groups):
            for n in linears:
                mod = layer.get_submodule(n)
                mod.weight.data = orig[n] if n in names else current[n]
            try:
                fq, fb = fake_quants(layer, names, inits, comps, qc, wstager)
                info = train_layer(
                    layer,
                    fq,
                    xs,
                    ys,
                    train_idx,
                    val_idx,
                    {},
                    qc,
                    ex.primary,
                    log,
                    f"L{li}g{gi}" if len(groups) > 1 else f"L{li}",
                    fb,
                    shares[gi],
                    None,
                    ex.devices,
                    calls=calls,
                    groups=groups_of_samples,
                )
            finally:  # the model layer stays unquantized
                for n in linears:
                    layer.get_submodule(n).weight.data = orig[n]
            norms = info.pop("norms", {})
            info["group_size"] = len(names)
            for nm, m in fq.items():
                tensors[linears[nm]] = m.export(wstager)
                current[nm] = dequantize(tensors[linears[nm]]).to(orig[nm].dtype)
            for pn, val in norms.items():
                prm = layer.get_parameter(pn)
                orig_norms.setdefault(pn, prm.data)
                prm.data = val.to(prm.dtype)
                full = f"{layer_prefix}{li}.{pn}"
                dense[(resolve(full) if resolve else full) or full] = val.to(
                    torch.float16
                )
            infos.append(info)
            peak = info.get("gpu_peak_gib")
            log(
                f"  QAD L{li} group {gi + 1}/{len(groups)}: {len(names)} linears, "
                f"nMSE {info['val_before']:.4e} -> {info['val_after']:.4e} "
                f"(kept {info['kept']}), {info['steps']}/{info['allotted']} steps, "
                f"{info['seconds']:.0f}s"
                + (f", peak {peak:.1f} GiB" if peak is not None else "")
            )
            del fq, fb
            gc.collect()
            if ex.primary.type == "cuda":
                torch.accelerator.empty_cache()
    finally:
        for n, w in orig.items():
            layer.get_submodule(n).weight.data = w
        for pn, w in orig_norms.items():
            layer.get_parameter(pn).data = w
    return tensors, _merge_infos(infos), dense


# ---- the driver ------------------------------------------------------------------


def qad_looped(
    model: nn.Module,
    ids: torch.Tensor,
    tables: dict,
    picks: dict,
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
    """:func:`hi2.qad.qad_layerwise` for a looped model (see the module docstring).
    Teacher-forced: every selected layer is trained from the layer program's inputs and
    outputs, and the final logit-KL phase (``qc.final_kl``) may follow in the pipeline."""
    path, layers = find_layers(model)
    prefix = path + "."
    text = model.get_submodule(path.rsplit(".", 1)[0])
    n_layers = len(layers) if limit_layers is None else min(limit_layers, len(layers))
    from .qad import parse_layers

    selected = sorted(i for i in parse_layers(qc.layers, len(layers)) if i < n_layers)
    calls_per_layer = [0] * len(layers)
    for c in schedule(text.config):
        calls_per_layer[c.layer] += 1
    qdir = Path(qdir or workdir / "qad")
    qdir.mkdir(parents=True, exist_ok=True)
    view = StatsView(workdir)
    wstager = PinnedWindow()
    res = QadResult()
    t_start = time.time()

    def linears_of(li: int) -> dict[str, str]:
        lin = _layer_linears(layers[li], li, prefix, include, exclude, resolve, log)
        return {n: c for n, c in lin.items() if c in tables}

    missing = []
    for li in selected:
        art = qdir / f"layer{li:03d}.pt"
        rec = torch.load(art, weights_only=False, mmap=True) if art.exists() else None
        if rec is None or rec.get("fingerprint") != fp:
            missing.append(li)
    log(
        f"QAD looped: {n_layers} layers ({len(selected)} selected, {len(missing)} to "
        f"train), {len(ids)} sequences of {ids.shape[1]} tokens, "
        f"{qc.steps} steps x {qc.batch} samples per layer, lr {qc.lr_w:g}/{qc.lr_p:g}, "
        f"init {qc.init}, math {qc.dtype}, expert groups "
        f"{qc.expert_groups if qc.expert_groups else 'auto'}"
    )
    prog = None
    teacher = None
    layout = None
    if missing:
        prog = _Program(model, ids, ex, log)
        layout = _layout(prog.calls, prog.layers, text.config)
        teacher = _capture(prog, missing, qc, qdir, fp, log)
    elif selected:
        log(
            "QAD looped: every selected layer comes from its artifact; the teacher "
            "pass is skipped"
        )
    budget = StepBudget(qc, len(selected))
    todo = set(selected)
    trained_time, trained_n = 0.0, 0
    for li in range(n_layers):
        layer = layers[li]
        linears = linears_of(li)
        art = qdir / f"layer{li:03d}.pt"
        rec = torch.load(art, weights_only=False, mmap=True) if art.exists() else None
        norms: dict = {}
        if rec is not None and rec.get("fingerprint") == fp:
            tensors, info = rec["tensors"], rec["info"]
            norms = rec.get("norms", {})
            log(f"QAD layer {li}: reusing {art.name}")
            if li in todo:
                budget.record(info)
        elif li in todo and linears:
            assert prog is not None and teacher is not None and layout is not None
            tl = time.time()
            inits, comps = init_layer(
                layer, linears, tables, picks, view, cfg, ex, wstager, log
            )
            samples, xs, ys, train_idx, val_idx, groups = _samples_for(
                teacher, layout, li, qc, teacher.n_seq
            )
            calls = _Samples(prog, teacher, layout, samples, ex.devices)
            tensors, info, norms = _train_groups(
                layer,
                li,
                linears,
                inits,
                comps,
                qc,
                ex,
                budget.allot(),
                samples,
                xs,
                ys,
                train_idx,
                val_idx,
                groups,
                calls,
                log,
                wstager,
                prefix,
                resolve,
            )
            info["passes"] = calls_per_layer[li]
            del inits, comps, samples, xs, ys, calls
            gc.collect()
            trained_time += time.time() - tl
            trained_n += 1
            rec = {"fingerprint": fp, "tensors": tensors, "info": info}
            if norms:
                rec["norms"] = norms
            atomic_torch_save(rec, art)
            budget.record(info)
            if event := getattr(log, "event", None):
                event(
                    "qad",
                    "completed",
                    layer=li + 1,
                    total_layers=n_layers,
                    step=info.get("steps", 0),
                    total_steps=info.get("allotted", 0),
                    artifact=str(art),
                    gpu_peak_gib=info.get("gpu_peak_gib"),
                )
            gain = 1 - info["val_after"] / max(info["val_before"], 1e-30)
            log(
                f"QAD layer {li}/{n_layers - 1}: {len(linears)} linears, "
                f"{info['passes']} call(s)/layer, held-out nMSE {info['val_before']:.4e} "
                f"-> {info['val_after']:.4e} ({-100 * gain:+.1f}%), "
                f"{info['steps']}/{info['allotted']} steps over {info['groups']} group(s), "
                f"{info['seconds']:.0f} s, rss {_rss_gib()} GiB"
            )
        else:
            tl = time.time()
            inits, comps = init_layer(
                layer, linears, tables, picks, view, cfg, ex, wstager, log
            )
            tensors = {linears[nm]: t for nm, t in inits.items()}
            info = {
                "kept": "ptq",
                "seconds": time.time() - tl,
                "passes": calls_per_layer[li],
            }
            del comps
            if li in todo:
                log(
                    f"QAD layer {li}/{n_layers - 1}: {len(linears)} linears kept at PTQ "
                    "(no eligible Linear)"
                )
            else:
                log(
                    f"QAD layer {li}/{n_layers - 1}: {len(linears)} linears kept at PTQ "
                    "(not selected)"
                )
        if norms:
            for pn, val in norms.items():
                full = f"{prefix}{pn}"
                res.dense[(resolve(full) if resolve else full) or full] = val.to(
                    torch.float16
                )
        res.tensors.update(tensors)
        res.layers.append(
            {"layer": li, **{k: v for k, v in info.items() if k != "trace"}}
        )
        left = len([i for i in selected if i > li])
        if trained_n and left:
            eta = (trained_time / trained_n) * left
            if event := getattr(log, "event", None):
                event(
                    "qad",
                    "running",
                    eta_s=eta,
                    eta_scope="remaining selected layers",
                )
        del tensors
        gc.collect()
    if selected:
        log(budget.summary())
    log(
        f"QAD looped: done in {(time.time() - t_start) / 60:.1f} min, "
        f"{trained_n} layer(s) trained, rss {_rss_gib()} GiB"
    )
    if teacher is not None:
        log(
            f"QAD looped: teacher states held {teacher.nbytes() / 2**30:.1f} GiB "
            "of host RAM"
        )
        del teacher
        gc.collect()
    return res
