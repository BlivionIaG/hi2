# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""The layer program of looped models (Inkling-2): phase 1 and the teacher pass.

:func:`hi2.layerwise.calibrate_layerwise` feeds the decoder layers one after the other
with a hidden state. A looped model's layers also take the block states of its attention
residuals (``block_states``), the layers of the loop span run ``loop_num_passes`` times,
and the passes after the first receive ``messages`` computed from the pass before: the 24
layers of Inkling-2-Mini are 48 layer calls with state in between. This module replays
that program (the control flow of ``Inkling2LoopTextModel.forward``) over per-sequence
states that stay in host memory, one layer at a time on the compute devices, like the
generic loop:

* :func:`record_calls` runs the model's own forward once with every layer replaced by a
  recorder, so what each call is given (attention masks, relative distances, position ids,
  the cache offset of a pass) is read from the model instead of rebuilt here;
* :func:`run_program` is the control flow, parameterised by the function that runs one
  layer call on all sequences;
* :func:`calibrate_looped` is phase 1: the statistics of a Linear are accumulated over
  **every call** of its layer (the loop layers share their weights across passes, so
  every pass is a sample of the same tensor), and the tables are built when the layer's
  last call is done;
* :func:`teacher_head_inputs` is the same program without hooks, giving the unquantized
  model's head inputs for the final logit-KL phase without placing the model on the
  devices.
"""

import contextlib
import dataclasses
import gc
import json
import os
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn

from .calibrate import Config
from .executor import Executor
from .hessian import HessianStats
from .layerwise import (
    WorkdirSink,
    _finish_tables,
    _layer_linears,
    _layer_pass,
    _merge,
    _rss_gib,
    _Stop,
    _tabulate_layer,
    _trim_host,
    capture_inputs,
    find_layers,
)
from .stats import ColumnStats

LOOPED_MODEL_TYPES = ("inkling2_loop",)


def is_looped(model) -> bool:
    """Whether ``model`` is a looped model this module replays (by its ``model_type``;
    every other model keeps the generic layer loop)."""
    return (
        getattr(getattr(model, "config", None), "model_type", None)
        in LOOPED_MODEL_TYPES
    )


def is_looped_source(src) -> bool:
    """:func:`is_looped` for a model directory, before the model is loaded."""
    cfg = json.loads((Path(src) / "config.json").read_text())
    return cfg.get("model_type") in LOOPED_MODEL_TYPES


#: What ``hi2`` quantizes of a looped model when ``--include`` is not given: the routed
#: experts (93.8 % of Inkling-2-Mini's text weights). The attention linears, the controllers
#: (message gates), the sinks, the short convolutions, the router and the vision tower
#: stay as they are (docs/inkling-source.md).
DEFAULT_INCLUDE = r"\.mlp\.experts\."


def default_include(src) -> str | None:
    """The ``--include`` a looped model gets when none is given; None for any other
    model (and for a directory that is not a model, which the caller reports)."""
    try:
        return DEFAULT_INCLUDE if is_looped_source(src) else None
    except (OSError, ValueError):
        return None


def seqs_per_forward() -> int:
    """Sequences per layer forward (``HI2_LOOP_SEQS``, default 4): the attention of a
    full-attention layer and the messages of a loop layer grow with it."""
    return max(1, int(os.environ.get("HI2_LOOP_SEQS", "4")))


@dataclasses.dataclass(frozen=True)
class Call:
    """One layer call of the program: the layer, whether it receives the messages of the
    previous pass, and (once recorded) the keyword arguments the model gave it."""

    layer: int
    messages: bool
    kwargs: dict = dataclasses.field(default_factory=dict, compare=False)


def schedule(config) -> list[Call]:
    """The layer calls of one forward, in order: the layers up to the end of the loop
    span (the first pass), the later passes over the span, the layers after it."""
    a, b = config.loop_layer_index_start, config.loop_layer_index_end
    calls = [Call(i, False) for i in range(b)]
    for _ in range(1, config.loop_num_passes):
        calls += [Call(i, True) for i in range(a, b)]
    return calls + [Call(i, False) for i in range(b, config.num_hidden_layers)]


class _Recorder(nn.Module):
    """Stands in for a decoder layer: records its call and hands the hidden state on."""

    def __init__(self, layer: nn.Module, index: int, seen: list, total: int):
        super().__init__()
        self.layer = layer
        self.index = index
        self.seen = seen
        self.total = total

    def __getattr__(self, name):  # attention_type, is_block_boundary, read by the model
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.layer, name)

    def forward(self, hidden_states, block_states, *args, **kwargs):
        kw = {
            k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
            for k, v in kwargs.items()
        }
        self.seen.append((self.index, bool(args), kw))
        if len(self.seen) == self.total:
            raise _Stop
        return hidden_states


def record_calls(model: nn.Module, layers: nn.ModuleList, text, ids) -> list[Call]:
    """:func:`schedule` with the keyword arguments of every call, read by running the
    model's own forward on ``ids[:1]`` with the layers replaced by recorders. Raises when
    the model calls its layers in another order than the program."""
    sched = schedule(text.config)
    seen: list = []
    originals = list(layers)
    for i, layer in enumerate(originals):
        layers[i] = _Recorder(layer, i, seen, len(sched))
    try:
        with torch.no_grad(), contextlib.suppress(_Stop):
            model(ids[:1], use_cache=False)
    finally:
        for i, layer in enumerate(originals):
            layers[i] = layer
    if [(i, m) for i, m, _ in seen] != [(c.layer, c.messages) for c in sched]:
        raise RuntimeError(
            "the model calls its decoder layers in another order than the looped "
            f"layer program: {[(i, m) for i, m, _ in seen][:8]}... vs "
            f"{[(c.layer, c.messages) for c in sched][:8]}..."
        )
    return [dataclasses.replace(c, kwargs=kw) for c, (_, _, kw) in zip(sched, seen)]


class _Messages:
    """``message_input(previous block states, previous hidden state, loop layer)`` of
    every sequence, computed when the layer's forward asks for it: a pass's messages
    (``[T, 4, H]`` in fp32 per sequence) for all sequences would be gigabytes."""

    def __init__(self, module, blocks, hidden, k):
        self.module, self.blocks, self.hidden, self.k = module, blocks, hidden, k

    def __getitem__(self, i):
        with torch.no_grad():
            return self.module(self.blocks[i], self.hidden[i], self.k)


def run_program(text, hidden: list, call) -> tuple[list, list]:
    """Replay ``Inkling2LoopTextModel.forward`` over per-sequence states.

    ``hidden``: the layer-0 inputs (``[1, T, H]`` each). ``call(index, call, h, bs,
    messages)`` runs layer call ``index`` on all sequences (``h``: their hidden states,
    ``bs``: their block states ``[1, T, blocks, H]``, ``messages``: None or an object
    indexable by sequence) and returns the new hidden states. Returns the final hidden
    states and block states."""
    cfg = text.config
    layers = text.layers
    a, b = cfg.loop_layer_index_start, cfg.loop_layer_index_end
    sched = schedule(cfg)
    index = 0

    def step(li, h, bs, messages=None):
        nonlocal index
        new = call(index, sched[index], h, bs, messages)
        index += 1
        if layers[li].is_block_boundary:  # the layer's input joins the block states
            bs = [torch.cat([s, x.unsqueeze(-2)], dim=-2) for s, x in zip(bs, h)]
        return new, bs

    h = list(hidden)
    bs = [x.new_zeros(*x.shape[:-1], 0, x.shape[-1]) for x in h]
    for li in range(a):
        h, bs = step(li, h, bs)
    pre_h, pre_bs = h, bs
    n_pre = pre_bs[0].shape[-2]
    for li in range(a, b):
        h, bs = step(li, h, bs)
    for _ in range(1, cfg.loop_num_passes):
        # +1 skips the state the first pass of the span appended
        prev_bs = [s[..., n_pre + 1 :, :] for s in bs]
        prev_h = h
        h, bs = pre_h, pre_bs
        for k, li in enumerate(range(a, b)):
            h, bs = step(li, h, bs, _Messages(text.message_input, prev_bs, prev_h, k))
    for li in range(b, cfg.num_hidden_layers):
        h, bs = step(li, h, bs)
    return h, bs


def head_input(text, h: torch.Tensor, bs: torch.Tensor) -> torch.Tensor:
    """The LM head's input of one sequence: the final attention residual over the block
    states, then the final norm (``[1, T, H]`` in the dtype of the hidden state)."""
    with torch.no_grad():
        return text.norm(text.final_input(bs, h)[0]).to(h.dtype)


class _Acc:
    """Statistics of one layer's Linears, accumulated over its calls on the host."""

    def __init__(self, linears: dict, k_of: dict, want_h: bool):
        self.linears, self.k_of, self.want_h = linears, k_of, want_h
        self.lead: dict[str, str] = {}
        self.stats: dict[str, ColumnStats] = {}
        self.hess: dict[str, HessianStats] = {}
        self.calls = 0

    def add(self, results: list, li: int) -> None:
        cpu = torch.device("cpu")
        for _outs, stats, hess, leader_of in results:
            for n, ld in leader_of.items():
                if self.lead.setdefault(n, ld) != ld:
                    raise SystemExit(
                        f"layer {li}: {self.linears[n]} shares its input with "
                        f"{self.linears[ld]} in one call of the layer but not in "
                        "another, so its statistics cannot be pooled over the calls "
                        "(restrict --include to the routed experts)"
                    )
            for ld, st in stats.items():
                self.stats.setdefault(ld, ColumnStats(self.k_of[ld], cpu)).merge(st)
            for ld, hs in hess.items():
                self.hess.setdefault(ld, HessianStats(self.k_of[ld], cpu)).merge(hs)
        self.calls += 1


class _Program:
    """The model, its recorded layer calls and the calibration states."""

    def __init__(self, model, ids, ex: Executor, log):
        self.path, self.layers = find_layers(model)
        self.prefix = self.path + "."
        self.text = model.get_submodule(self.path.rsplit(".", 1)[0])
        self.ex = ex
        self.hidden, _ = capture_inputs(model, self.layers, ids)
        self.n_seq = len(self.hidden)
        self.calls = record_calls(model, self.layers, self.text, ids)
        self.batch = seqs_per_forward()
        log(
            f"captured {self.n_seq} calibration sequences of "
            f"{tuple(self.hidden[0].shape)}; looped program of {len(self.calls)} "
            f"layer calls, {self.batch} sequence(s) per forward"
        )

    def run(self, ci, h, bs, messages, linears=None, k_of=None, want_h=False):
        """Layer call ``ci`` on all sequences: ``(new hidden states, per-device
        results)``; the Linears in ``linears`` collect statistics."""
        extras = {"block_states": bs}
        if messages is not None:
            extras["messages"] = messages
        results = _layer_pass(
            self.layers[self.calls[ci].layer],
            self.ex.devices,
            self.n_seq,
            h,
            self.calls[ci].kwargs,
            linears or {},
            k_of or {},
            want_h,
            False,
            extras=extras,
            batch=self.batch,
            cast=False,
        )
        out: list = [None] * self.n_seq
        for outs, *_ in results:
            for i, o in outs.items():
                out[i] = o
        for d in self.ex.devices:  # account the queued forward/statistics work to it
            if d.type != "cpu":
                torch.accelerator.synchronize(d)
        return out, results

    def heads(self, h, bs) -> list[torch.Tensor]:
        return [head_input(self.text, x, s) for x, s in zip(h, bs)]


def calibrate_looped(
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
    sink=None,
):
    """:func:`hi2.layerwise.calibrate_layerwise` for a looped model: ``{checkpoint
    tensor name: TensorTable}``, the statistics going to ``sink``. The program always
    runs (the layers feed each other); ``limit_layers`` only bounds which layers are
    tabulated, and the run stops after the last call that one needs."""
    sink = sink or WorkdirSink(workdir)
    prog = _Program(model, ids, ex, log)
    n_calls = Counter(c.layer for c in prog.calls)
    want_h = cfg.rounding == "gptq" and not cfg.sequential
    last = len(prog.layers) if limit_layers is None else limit_layers
    stop_after = max(
        (ci for ci, c in enumerate(prog.calls) if c.layer < last), default=-1
    )
    linears_of, k_of_of, accs = {}, {}, {}
    for li, layer in enumerate(prog.layers):
        if li >= last:
            continue
        lin = _layer_linears(layer, li, prog.prefix, include, exclude, resolve, log)
        linears_of[li] = lin
        k_of_of[li] = {n: layer.get_submodule(n).in_features for n in lin}
        accs[li] = _Acc(lin, k_of_of[li], want_h)
    tables: dict = {}
    expert_counts: dict[str, int] = {}
    expert_fallback: set[str] = set()
    tm = {"forward": 0.0, "merge": 0.0, "write": 0.0, "tables": 0.0}

    def call(ci, c, h, bs, messages):
        li = c.layer
        acc = accs.get(li)
        lin = linears_of.get(li, {})
        t0 = time.perf_counter()
        out, results = prog.run(ci, h, bs, messages, lin, k_of_of.get(li), want_h)
        t1 = time.perf_counter()
        tm["forward"] += t1 - t0
        if acc is not None:
            acc.add(results, li)
        del results
        if acc is not None and acc.calls == n_calls[li]:
            _finish_layer(prog.layers[li], li, acc, t1 - t0)
        elif acc is not None:
            log(f"  layer {li} call {acc.calls}/{n_calls[li]}: forward {t1 - t0:.0f}s")
        if ci == stop_after:
            raise _Stop
        return out

    def _finish_layer(layer, li, acc, forward_s):
        t1 = time.perf_counter()
        gc.collect()
        _trim_host()
        lead, merged, hess, counts, fallback = _merge(
            [(None, acc.stats, acc.hess, acc.lead)],
            acc.linears,
            acc.k_of,
            ex.primary,
            want_h,
            cfg.expert_min_tokens,
            log,
        )
        t2 = time.perf_counter()
        if want_h:
            sink.hessians(lead, hess, acc.linears)
        del hess
        accs[li] = None
        tm["merge"] += t2 - t1
        tm["write"] += time.perf_counter() - t2
        elapsed = _tabulate_layer(
            layer,
            li,
            acc.linears,
            lead,
            merged,
            counts,
            fallback,
            cfg,
            ex,
            sink,
            tables,
            expert_counts,
            expert_fallback,
            tm,
            log,
        )
        log(
            f"layer {li}: {len(acc.linears)} linears tabulated over "
            f"{n_calls[li]} call(s) (last forward {forward_s:.0f}s, merge "
            f"{t2 - t1:.0f}s, tables {elapsed:.0f}s, rss {_rss_gib()} GiB)"
        )

    with contextlib.suppress(_Stop):
        run_program(
            prog.text, prog.hidden, lambda ci, c, h, bs, m: call(ci, c, h, bs, m)
        )
    _finish_tables(tm, "in host memory", expert_counts, expert_fallback, cfg, sink, log)
    return tables


def teacher_head_inputs(model: nn.Module, ids: torch.Tensor, ex: Executor, log=print):
    """The unquantized model's LM-head inputs on ``ids``, ``[n, T, H]`` in fp32: the
    teacher of the final logit-KL phase, computed with the layer program (one layer on
    the devices at a time), so the model never has to fit them."""
    prog = _Program(model, ids, ex, log)
    t0 = time.perf_counter()

    def call(ci, c, h, bs, messages):
        out, _ = prog.run(ci, h, bs, messages)
        if (ci + 1) % 12 == 0 or ci + 1 == len(prog.calls):
            log(
                f"  teacher pass: layer call {ci + 1}/{len(prog.calls)} "
                f"({time.perf_counter() - t0:.0f}s)"
            )
        return out

    h, bs = run_program(prog.text, prog.hidden, call)
    heads = prog.heads(h, bs)
    return torch.cat([x.float() for x in heads], dim=0)


def saved_input_bytes(text, seq_len: int, hidden_bytes: int) -> list[tuple[int, int]]:
    """``(layer, bytes)`` of every layer call of the program: the inputs the call keeps
    for the backward when the layers run under activation checkpointing (the hidden
    state, the block states so far and, in the passes after the first, the float32
    messages) for one sequence of ``seq_len`` tokens in ``hidden_bytes`` bytes per value."""
    cfg = text.config
    h = cfg.hidden_size
    a = cfg.loop_layer_index_start
    blocks = pre_blocks = 0
    out = []
    for c in schedule(cfg):
        if c.layer == a and not c.messages:
            pre_blocks = blocks  # the states the later passes restart from
        if c.layer == a and c.messages:
            blocks = pre_blocks
        size = seq_len * h * hidden_bytes * (1 + blocks)
        if c.messages:
            size += seq_len * 4 * h * 4
        out.append((c.layer, size))
        if text.layers[c.layer].is_block_boundary:
            blocks += 1
    return out
