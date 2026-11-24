# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""KL(stock || HI2) of an Inkling-2-Mini quantization on held-out windows (stage 4).

The teacher is the stock model with every routed expert decoded exactly from the stock
tensors (``InklingSource``: NVFP4 / MXFP8 to float32, then ``--compute-dtype``); the student
is the same model with the experts replaced by the decode of an HI2 output
(``hi2 quantize``, read tensor by tensor: no dense export, no second model in RAM). Both run
through the layer program of ``hi2.looped`` (one layer at a time on ``--devices``; the model
is never placed), then the model's own LM head and logit soft-cap give the distributions:
per window and position KL(teacher || student), top-1 agreement, the expected acceptance
of speculative sampling (``sum min(p, q)``) and the next-token NLL of both.

``--groups`` repeats the student pass with the experts of ONE group of layers quantized and
the rest stock (``auto``: before the loop, the loop, then the stretches of one stock format:
Inkling-2-Mini 0-5, 6-17, 18-19, 20-23) on the first ``--breakdown-windows`` windows: where
the damage is. The groups do not add up (the network is not linear); the all-student value
on the same windows is in the result for comparison.

    python experiments/inkling_kl_eval.py --src STOCK --dst HI2 --ids heldout_mixed_ids.npy \\
        --windows 32 --devices cuda:0 cuda:1 --compute-dtype float16 --out kl.json

``--routing-log`` records, in the same student pass, where the router sends every token of every
window (per window, layer and expert) and reports per layer group the share of the
token-expert assignments that go to INT2-class experts (mean code width < 3 bits over the
expert's gate, up and down), to INT4-class ones (< 6) and to INT8-class ones: where the
quantization error is paid. The counts are in ``<out stem>_routing.npz``.

``--int8-class attn|embed|lm_head`` stores ONE bf16 class of the student (the attention
projections, the embedding, the LM head) as HI2 INT8 and reads it back before the student pass
(``experiments/inkling_int8_probe.py``, exploratory); ``--stock-student`` keeps the stock experts,
so the student differs from the teacher by that class alone.

Exit code 0 when every gate held: the student covers every routed expert and nothing else
was quantized, differs from the teacher, every value is finite, no KL is negative, the metric
gives 0 on identical inputs, the groups restored the stock weights, the held-out ids match
``--ids-sha256`` and the KL is within ``--max-kl``. A student that cannot be scored (a
missing expert, another quantized tensor) is refused before the model is loaded.
"""

import argparse
import collections
import copy
import hashlib
import inspect
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

EXPERT_BASE = re.compile(r"\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$")
METRICS = ("kl", "top1", "ear", "nll_teacher", "nll_student")


def auto_groups(num_layers: int, loop: tuple[int, int], formats: list[str]):
    """Contiguous layer ranges cut where the loop starts and ends and the stock format of the
    experts changes: ``[[0..5], [6..17], [18, 19], [20..23]]`` for Inkling-2-Mini."""
    cuts = {loop[0], loop[1]} | {
        i for i in range(1, num_layers) if formats[i] != formats[i - 1]
    }
    bounds = [0, *sorted(c for c in cuts if 0 < c < num_layers), num_layers]
    return [list(range(a, b)) for a, b in zip(bounds, bounds[1:])]


def parse_groups(spec: str, num_layers: int, loop, formats) -> list[list[int]]:
    """``auto`` or ``0-5,6-17,18-19,20-23`` (inclusive ranges, single layers allowed)."""
    if spec == "auto":
        return auto_groups(num_layers, loop, formats)
    groups = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        a, b = int(lo), int(hi or lo)
        if not 0 <= a <= b < num_layers:
            raise SystemExit(f"group {part!r} is outside the {num_layers} layers")
        groups.append(list(range(a, b + 1)))
    return groups


def group_name(layers: list[int]) -> str:
    return f"{layers[0]}-{layers[-1]}" if len(layers) > 1 else str(layers[0])


@torch.no_grad()
def kl_stats(
    lm, hs, ht, ids, device=None, chunk: int = 256, lm_student=None
) -> dict[str, torch.Tensor]:
    """Per window means of the metrics of ``METRICS`` (float64 ``[n]``): ``hs`` and ``ht``
    are ``[n, T, H]`` head inputs of the student and the teacher, ``ids`` the ``[n, T]``
    tokens (the next-token NLL scores positions ``0..T-2``). The head and its soft-cap
    ``lm`` run on ``device`` in chunks of ``chunk`` positions; the student's states go
    through ``lm_student`` when it is given (a head that is not the teacher's)."""
    n, t, _ = hs.shape
    out = {k: torch.zeros(n, dtype=torch.float64) for k in METRICS}
    for i in range(n):
        tot = dict.fromkeys(METRICS, 0.0)
        for s in range(0, t, chunk):
            a, b = hs[i, s : s + chunk], ht[i, s : s + chunk]
            if device is not None:
                a, b = a.to(device), b.to(device)
            lt = torch.log_softmax(lm(b).float(), -1)
            ls = torch.log_softmax(
                (lm if lm_student is None else lm_student)(a).float(), -1
            )
            p = lt.exp()
            tot["kl"] += (p * (lt - ls)).sum().item()
            tot["top1"] += (lt.argmax(-1) == ls.argmax(-1)).sum().item()
            tot["ear"] += torch.minimum(p, ls.exp()).sum().item()
            tgt = ids[i, s + 1 : s + chunk + 1].to(lt.device)
            m = tgt.numel()
            if m:
                tot["nll_teacher"] -= lt[:m].gather(1, tgt[:, None]).sum().item()
                tot["nll_student"] -= ls[:m].gather(1, tgt[:, None]).sum().item()
        for k in METRICS:
            out[k][i] = tot[k] / (max(t - 1, 1) if k.startswith("nll") else t)
    return out


def summarize(stats: dict[str, torch.Tensor]) -> dict:
    """Mean over windows (every window has the same length) with the standard error of the
    mean over windows, the per-window values of the KL and the derived perplexity ratio."""
    kl = stats["kl"]
    n = len(kl)
    sem = (
        float(kl.std(unbiased=True) / n**0.5) if n > 1 else None
    )  # one window: no spread
    d_nll = float((stats["nll_student"] - stats["nll_teacher"]).mean())
    return {
        "windows": n,
        "kl_mean": float(kl.mean()),
        "kl_sem": sem,
        "kl_median_window": float(kl.median()),
        "kl_max_window": float(kl.max()),
        "top1_agreement": float(stats["top1"].mean()),
        "expected_acceptance": float(stats["ear"].mean()),
        "nll_teacher": float(stats["nll_teacher"].mean()),
        "nll_student": float(stats["nll_student"].mean()),
        "delta_nll": d_nll,
        "ppl_ratio": float(np.exp(d_nll)),
        "kl_per_window": [float(x) for x in kl],
    }


class Experts:
    """The routed experts of the model's layers and which weights each holds: the stock
    decode (``teacher``) or the decode of the HI2 output (``student``). ``set`` writes
    only the layers that change, into the very storage the model's Linears view."""

    def __init__(self, model, src, ck, device, log=print):
        from hi2.layerwise import find_layers
        from hi2.moe import ExpertMLP, UnfusedExperts

        self.path, layers = find_layers(model)
        self.src, self.ck, self.device, self.log = src, ck, device, log
        self.mods = {}
        for i, layer in enumerate(layers):
            mod = getattr(getattr(layer, "mlp", None), "experts", None)
            if not isinstance(mod, UnfusedExperts):
                raise SystemExit(
                    f"layer {i}: the experts are not unfused ({type(mod)})"
                )
            # an UnfusedExperts also holds the activation as a child: iterating it
            # yields that too, after the experts
            self.mods[i] = [m for m in mod if isinstance(m, ExpertMLP)]
        self.state: dict[int, str] = {}
        self.installed = 0
        self.changed = 0
        self.missing = 0
        self.weight_error: dict[int, dict] = {}

    @torch.no_grad()
    def _teacher(self, layer: int) -> None:
        mod = self.mods[layer]
        dtype = mod[0].gate_proj.weight.dtype
        gate_up, down = self.src.fused_layer(layer, device=self.device, dtype=dtype)
        for e, mlp in enumerate(mod):
            i = mlp.gate_proj.weight.shape[0]
            mlp.gate_proj.weight.copy_(gate_up[e, :i])
            mlp.up_proj.weight.copy_(gate_up[e, i:])
            mlp.down_proj.weight.copy_(down[e])

    @torch.no_grad()
    def _student(self, layer: int) -> None:
        """Overwrite the layer's experts with the HI2 decode; a tensor the output does not
        hold (a partial student, ``--partial``) keeps the stock weights and is counted."""
        from hi2.format import dequantize

        mod = self.mods[layer]
        fresh = self.state.get(layer) == "teacher" and layer not in self.weight_error
        err = ref = 0.0
        count = changed = missing = 0
        for e, mlp in enumerate(mod):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                lin = getattr(mlp, proj)
                try:
                    t = self.ck.get(f"{self.path}.{layer}.mlp.experts.{e}.{proj}")
                except KeyError:
                    missing += 1
                    continue
                w = dequantize(t, self.device).to(lin.weight.dtype)
                if tuple(w.shape) != tuple(lin.weight.shape):
                    raise SystemExit(
                        f"layer {layer} expert {e} {proj}: HI2 shape {tuple(w.shape)} "
                        f"!= model {tuple(lin.weight.shape)}"
                    )
                w = w.to(lin.weight.device)
                if fresh:  # the model still holds the stock decode: measure the error
                    diff = w.float() - lin.weight.float()
                    err += float((diff**2).sum())
                    ref += float((lin.weight.float() ** 2).sum())
                    changed += bool((diff != 0).any())
                lin.weight.copy_(w)
                count += 1
        if fresh:
            self.weight_error[layer] = {
                "relative_rms": (err / ref) ** 0.5 if ref else None,
                "tensors": count,
                "tensors_changed": changed,
                "tensors_missing": missing,
            }
            self.changed += changed
            self.installed += count
            self.missing += missing

    def set(self, want: dict[int, str]) -> None:
        for layer, kind in sorted(want.items()):
            if self.state.get(layer) == kind:
                continue
            if kind == "student" and self.state.get(layer) != "teacher":
                self._teacher(layer)  # an error is measured against the stock decode
                self.state[layer] = "teacher"
            (self._teacher if kind == "teacher" else self._student)(layer)
            self.state[layer] = kind

    def all(self, kind: str) -> dict[int, str]:
        return dict.fromkeys(self.mods, kind)


def expert_bits(ck, path: str, layers, n_experts: int) -> torch.Tensor:
    """Mean code width of every routed expert of an HI2 output, ``[layers, experts]``: the mean
    over the rows of its gate, up and down (each has as many weights as the others and every
    row of a tensor as many as the rest, so the mean over rows is the mean over weights)."""
    out = torch.zeros(len(layers), n_experts, dtype=torch.float64)
    for li, layer in enumerate(layers):
        for e in range(n_experts):
            bits = [
                ck.get(f"{path}.{layer}.mlp.experts.{e}.{p}").row_nbits.double().mean()
                for p in ("gate_proj", "up_proj", "down_proj")
            ]
            out[li, e] = torch.stack(bits).mean()
    return out


CLASSES = {
    "int2": "mean code width < 3",
    "int4": "3 <= width < 6",
    "int8": "width >= 6",
}
EDGES = [
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    5.0,
    6.0,
    7.0,
    8.01,
]  # of the histogram of expert widths


def class_masks(bits: torch.Tensor) -> dict[str, torch.Tensor]:
    """The INT2-, INT4- and INT8-class experts of ``expert_bits`` (every expert in one)."""
    return {
        "int2": bits < 3.0,
        "int4": (bits >= 3.0) & (bits < 6.0),
        "int8": bits >= 6.0,
    }


class Routing:
    """Where the router of the student sends the tokens of the evaluation windows.

    The layer program runs a layer for all windows at once: the windows are dealt round-robin
    to the devices (one thread each, ``hi2.layerwise._layer_pass``) and a device runs its
    share in batches of ``HI2_LOOP_SEQS`` windows through its own copy of the layer
    (``_device_pass``). ``_device_pass`` is wrapped here so that the thread running a layer
    call knows which windows each of its forwards holds, and a pre-hook on the experts of
    every layer (the copies share it) counts, per window of the forward, the tokens sent to
    each expert. ``counts[window, layer, expert]`` adds up the passes of a loop layer."""

    def __init__(self, model, experts: "Experts", cfg, seq_len: int, n_windows: int):
        from hi2 import layerwise, looped

        self.layerwise = layerwise
        self.seq_len = seq_len
        _, layers = layerwise.find_layers(model)
        self.layers = sorted(experts.mods)
        self.slot = {id(layers[i]): s for s, i in enumerate(self.layers)}
        self.n_experts = len(experts.mods[self.layers[0]])
        self.passes = collections.Counter(c.layer for c in looped.schedule(cfg))
        self.counts = torch.zeros(
            n_windows, len(self.layers), self.n_experts, dtype=torch.int64
        )
        self.top_k: set[int] = set()
        self.forwards = 0
        self.problems: list[str] = []
        self.lock = threading.Lock()
        self.local = threading.local()

        def hook(_module, args, kwargs):
            # a plain function: the deep copy of a layer that every device pass makes
            # shares it (a bound method would be copied with its owner, counts and all)
            self._count(args, kwargs)

        self.handles = [
            layers[i].mlp.experts.register_forward_pre_hook(hook, with_kwargs=True)
            for i in self.layers
        ]
        self.original = layerwise._device_pass
        self.signature = inspect.signature(self.original)
        layerwise._device_pass = self._device_pass

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.layerwise._device_pass = self.original

    def _windows(self, items, batch) -> list[list[int]]:
        """The windows of every forward of a device, as ``layerwise._run`` batches them."""
        if len(items) > 1 and os.environ.get("HI2_LAYER_BATCH", "1") != "0":
            return self.layerwise._chunks(items, batch)
        return [[i] for i in items]

    def _device_pass(self, layer, dev, items, *args, **kwargs):
        slot = self.slot.get(id(layer))
        if slot is None:  # not a layer of the routed experts
            return self.original(layer, dev, items, *args, **kwargs)
        batch = self.signature.bind(layer, dev, items, *args, **kwargs).arguments.get(
            "batch"
        )
        ctx = self.local
        ctx.slot, ctx.forwards, ctx.k = slot, self._windows(items, batch), 0
        try:
            return self.original(layer, dev, items, *args, **kwargs)
        finally:
            if ctx.k != len(ctx.forwards):
                self.problems.append(
                    f"layer {self.layers[slot]}: {ctx.k} expert forwards for "
                    f"{len(ctx.forwards)} batches of windows"
                )
            ctx.slot = None

    def _count(self, args, kwargs):
        ctx = self.local
        if getattr(ctx, "slot", None) is None:
            return
        idx = args[1] if len(args) > 1 else kwargs["top_k_index"]
        if ctx.k >= len(ctx.forwards):
            raise RuntimeError("more expert forwards than batches of windows")
        windows = ctx.forwards[ctx.k]
        ctx.k += 1
        n, k = idx.shape
        if n != len(windows) * self.seq_len:
            raise RuntimeError(
                f"an expert forward holds {n} tokens, not {len(windows)} windows of "
                f"{self.seq_len}"
            )
        rows = len(windows)
        per = torch.zeros(rows, self.n_experts, dtype=torch.int64, device=idx.device)
        sent = idx.reshape(
            rows, -1
        )  # window-major: the tokens of a window are consecutive
        per.scatter_add_(1, sent, torch.ones_like(sent, dtype=torch.int64))
        per = per.cpu()
        with self.lock:
            self.top_k.add(k)
            self.forwards += 1
            for j, w in enumerate(windows):
                self.counts[w, ctx.slot] += per[j]

    def summarize(
        self, bits: torch.Tensor, groups: list[list[int]], group_name
    ) -> dict:
        """The routing against the code widths of the experts, per layer group and window."""
        if len(self.top_k) != 1:
            return {"complete": False, "problem": f"top-k seen: {sorted(self.top_k)}"}
        (k,) = self.top_k  # one top-k for every layer
        want = torch.tensor([self.passes[li] * self.seq_len * k for li in self.layers])
        total = self.counts.sum(-1)  # [windows, layers]
        complete = bool((total == want).all()) and not self.problems
        masks = class_masks(bits)
        counts = self.counts.double()

        def block(slots: list[int]) -> dict:
            c = counts[:, slots]  # [windows, layers, experts]
            tot = c.sum((1, 2))
            out = {
                "layers": [self.layers[s] for s in slots],
                "assignments_per_window": tot.tolist(),
                "experts": {
                    n: float(m[slots].double().mean()) for n, m in masks.items()
                },
                "assignments": {},
                "per_window": {},
                "mean_bits": float((c * bits[slots]).sum() / c.sum()),
                "mean_bits_per_window": ((c * bits[slots]).sum((1, 2)) / tot).tolist(),
            }
            for n, m in masks.items():
                by = (c * m[slots]).sum((1, 2))
                out["assignments"][n] = float(by.sum() / c.sum())
                out["per_window"][n] = (by / tot).tolist()
            return out

        slot_of = {li: s for s, li in enumerate(self.layers)}
        return {
            "complete": complete,
            "top_k": k,
            "classes": CLASSES,
            "passes": {str(li): self.passes[li] for li in self.layers},
            "groups": {
                group_name(g): block([slot_of[li] for li in g if li in slot_of])
                for g in groups
            },
            "all_layers": block(list(range(len(self.layers)))),
            "expert_bits_histogram": {
                "edges": EDGES,
                "experts": [
                    int(((bits >= a) & (bits < b)).sum())
                    for a, b in zip(EDGES[:-1], EDGES[1:])
                ],
            },
        }

    def save(self, path: Path, bits: torch.Tensor) -> None:
        np.savez_compressed(
            path,
            counts=self.counts.numpy().astype(np.int32),
            expert_bits=bits.numpy().astype(np.float32),
            layers=np.array(self.layers),
            passes=np.array([self.passes[li] for li in self.layers]),
            seq_len=self.seq_len,
        )


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.int32).tobytes()).hexdigest()


def fingerprint(src: Path, ids: np.ndarray, dtype: str) -> str:
    """What the teacher's head inputs depend on: the stock model, the tokens, the dtype."""
    h = hashlib.sha256()
    h.update((src / "config.json").read_bytes())
    for f in sorted(src.glob("*.safetensors")):
        h.update(f"{f.name}:{f.stat().st_size}".encode())
    h.update(sha256_array(ids).encode())
    h.update(f"{ids.shape}:{dtype}".encode())
    return h.hexdigest()


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, required=True, help="the stock model directory")
    ap.add_argument("--dst", type=Path, required=True, help="the HI2 output directory")
    ap.add_argument(
        "--ids", type=Path, required=True, help="held-out windows, .npy [n, T]"
    )
    ap.add_argument("--ids-sha256", default=None, help="expected sha256 of ALL windows")
    ap.add_argument("--windows", type=int, default=0, help="first N windows (0: all)")
    ap.add_argument(
        "--seq-len", type=int, default=0, help="crop every window (0: whole)"
    )
    ap.add_argument("--devices", nargs="+", default=["cpu"])
    ap.add_argument("--compute-dtype", default="float16")
    ap.add_argument(
        "--transformers", type=Path, default=None, help="the fork's transformers"
    )
    ap.add_argument("--groups", default="auto", help="auto, none, or 0-5,6-17,...")
    ap.add_argument("--breakdown-windows", type=int, default=8)
    ap.add_argument("--teacher-cache", type=Path, default=None)
    ap.add_argument(
        "--partial", action="store_true", help="allow a student without every expert"
    )
    ap.add_argument(
        "--max-kl", type=float, default=None, help="gate: KL of the whole student"
    )
    ap.add_argument(
        "--routing-log",
        action="store_true",
        help="count where the student's router sends the tokens of every window (per "
        "layer and expert) and report the share going to INT2-class experts per layer "
        "group; the counts go to <out stem>_routing.npz",
    )
    ap.add_argument(
        "--int8-class",
        choices=("attn", "embed", "lm_head"),
        default=None,
        help="store ONE bf16 class of the student as HI2 INT8 and read it back before the "
        "student pass (inkling_int8_probe.py): the attention projections, the embedding or "
        "the LM head; the teacher is untouched; needs --groups none",
    )
    ap.add_argument(
        "--int8-block",
        type=int,
        choices=(32, 64),
        default=64,
        help="the HI2 INT8 candidate (8, B) of --int8-class",
    )
    ap.add_argument(
        "--stock-student",
        action="store_true",
        help="with --int8-class: keep the stock experts in the student, which then differs "
        "from the teacher by that class alone",
    )
    ap.add_argument("--out", type=Path, required=True)
    return ap


def _int8_probe():
    """``experiments/inkling_int8_probe.py``, imported when ``--int8-class`` asks for it."""
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import inkling_int8_probe

    return inkling_int8_probe


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    if a.stock_student and not a.int8_class:
        raise SystemExit(
            "--stock-student needs --int8-class: the student would be the teacher"
        )
    if a.int8_class and a.groups != "none":
        raise SystemExit(
            "--int8-class needs --groups none: a group pass restores the stock class"
        )
    if a.transformers is not None:
        sys.path.insert(0, str(a.transformers))

    from hi2 import looped
    from hi2.executor import Executor
    from hi2.finalkl import _logits_fn
    from hi2.inkling_source import InklingSource
    from hi2.pipeline import RunConfig, _load_model
    from hi2.stack import Hi2Checkpoint

    t_start = time.perf_counter()
    log = lambda *m: print(*m, flush=True)  # noqa: E731
    gates: dict[str, dict] = {}
    result: dict = {
        "src": str(a.src),
        "dst": str(a.dst),
        "dtype": a.compute_dtype,
        "devices": a.devices,
        "seconds": {},
        "stock_student": a.stock_student,
    }

    def gate(name, ok, detail, hard=True):
        gates[name] = {"ok": bool(ok), "hard": hard, "detail": str(detail)}

    def failed() -> bool:
        return not all(g["ok"] for g in gates.values() if g["hard"])

    def finish() -> int:
        result["gates"] = gates
        result["ok"] = not failed()
        result["seconds"]["total"] = round(time.perf_counter() - t_start, 1)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, indent=2))
        if "student" in result:
            s = result["student"]
            sem = "n/a" if s["kl_sem"] is None else f"{s['kl_sem']:.2g}"
            print(
                f"KL(stock||HI2) {s['kl_mean']:.6g} +- {sem} (sem over "
                f"{s['windows']} windows), top-1 {s['top1_agreement']:.4f}, "
                f"perplexity x{s['ppl_ratio']:.4f}"
            )
        bad = {k: g["detail"] for k, g in gates.items() if g["hard"] and not g["ok"]}
        print(f"ok={result['ok']}" + (f" failed: {bad}" if bad else ""))
        return 0 if result["ok"] else 1

    full = np.load(a.ids)
    digest = sha256_array(full)
    if a.ids_sha256:
        gate(
            "ids_match_the_manifest",
            digest == a.ids_sha256,
            f"{digest[:16]} vs {a.ids_sha256[:16]}",
        )
    ids_np = full[: a.windows] if a.windows else full
    if a.seq_len:
        ids_np = ids_np[:, : a.seq_len]
    ids = torch.from_numpy(ids_np.astype(np.int64))
    result |= {
        "windows": int(ids.shape[0]),
        "seq_len": int(ids.shape[1]),
        "ids_sha256_all_windows": digest,
    }
    devices = [torch.device(d) for d in a.devices]
    decode = next((d for d in devices if d.type == "cuda"), torch.device("cpu"))
    try:
        ck = Hi2Checkpoint(a.dst)
        bases = sorted(ck.logical())
    except (OSError, ValueError, KeyError) as e:
        gate("dst_is_an_hi2_checkpoint", False, f"{type(e).__name__}: {e}")
        return finish()
    with InklingSource(a.src, device=decode) as src:
        # the coverage of the student comes first: nothing expensive for one that cannot
        # be scored (the model is not even loaded)
        mine = [b for b in bases if EXPERT_BASE.search(b)]
        other = [b for b in bases if not EXPERT_BASE.search(b)]
        want_n = 3 * src.num_layers * src.n_experts
        gate(
            "student_has_expert_tensors", bool(mine), f"{len(mine)} HI2 expert tensors"
        )
        gate(
            "student_covers_every_expert",
            len(mine) == want_n or a.partial,
            f"{len(mine)} HI2 expert tensors of {want_n}",
        )
        gate(
            "no_other_tensor_is_quantized",
            not other or a.partial,
            f"{len(other)} other HI2 tensors: {other[:3]}",
        )
        if failed():
            return finish()
        ex = Executor(a.devices)
        rc = RunConfig(
            src=a.src,
            dst=a.out.parent / "unused",
            workdir=a.out.parent / "work-unused",
            compute_dtype=a.compute_dtype,
            devices=a.devices,
        )
        model = _load_model(rc, log)
        experts = Experts(model, src, ck, decode, log)
        lm = _logits_fn(model)
        head_dev = devices[0] if devices[0].type != "cpu" else None
        if head_dev is not None:  # a copy: the model keeps its own head where it is
            lm = copy.deepcopy(lm).to(head_dev)

        def timed(name, fn):
            t0 = time.perf_counter()
            r = fn()
            result["seconds"][name] = round(time.perf_counter() - t0, 1)
            log(f"{name}: {result['seconds'][name]} s")
            return r

        def heads(window_ids):
            return looped.teacher_head_inputs(model, window_ids, ex, log=log)

        # --- the teacher
        fp = fingerprint(a.src, ids_np, a.compute_dtype)
        ht = None
        if a.teacher_cache is not None and a.teacher_cache.exists():
            blob = torch.load(a.teacher_cache, map_location="cpu", weights_only=True)
            if blob.get("fingerprint") == fp:
                ht = blob["heads"]
                log(f"teacher head inputs from {a.teacher_cache}")
        timed("install_teacher", lambda: experts.set(experts.all("teacher")))
        if ht is None:
            ht = timed("teacher_program", lambda: heads(ids))
            if a.teacher_cache is not None:
                a.teacher_cache.parent.mkdir(parents=True, exist_ok=True)
                tmp = a.teacher_cache.with_name(f".{a.teacher_cache.name}.tmp")
                torch.save({"fingerprint": fp, "heads": ht}, tmp)
                tmp.replace(a.teacher_cache)
        gate("teacher_finite", torch.isfinite(ht).all(), tuple(ht.shape))
        same = kl_stats(lm, ht[:1], ht[:1], ids[:1], head_dev)
        gate(
            "metric_is_zero_on_identical_inputs",
            float(same["kl"].abs().max()) == 0.0,
            float(same["kl"][0]),
        )
        groups = []
        if a.groups != "none":
            cfg = model.model.language_model.config
            groups = parse_groups(
                a.groups,
                len(experts.mods),
                (cfg.loop_layer_index_start, cfg.loop_layer_index_end),
                src.formats(),
            )
            # the stock model once more on one window, for the restore check below: the
            # same program on the same batch, so only a weight that did not come back
            # can make the two differ
            probe = timed("probe_program", lambda: heads(ids[:1]))
        lm_student = None
        if a.int8_class:
            int8_probe = _int8_probe()
            result["int8"], lm_student = timed(
                "int8_class",
                lambda: int8_probe.apply(
                    model, a.int8_class, a.int8_block, ex, log, lm
                ),
            )
        # --- the student
        if not a.stock_student:
            timed("install_student", lambda: experts.set(experts.all("student")))
        routing = None
        try:
            if a.routing_log:
                routing = Routing(
                    model,
                    experts,
                    model.model.language_model.config,
                    int(ids.shape[1]),
                    int(ids.shape[0]),
                )
            hs = timed("student_program", lambda: heads(ids))
        finally:
            if routing is not None:
                routing.close()
        gate("student_finite", torch.isfinite(hs).all(), tuple(hs.shape))
        head = {} if lm_student is None else {"lm_student": lm_student}
        stats = timed("metrics", lambda: kl_stats(lm, hs, ht, ids, head_dev, **head))
        result["student"] = summarize(stats)
        bw = max(1, min(a.breakdown_windows, ids.shape[0]))
        result["student_on_breakdown_windows"] = summarize(
            {k: v[:bw] for k, v in stats.items()}
        )
        if routing is not None:
            cfg = model.model.language_model.config
            bits = expert_bits(ck, experts.path, routing.layers, routing.n_experts)
            rgroups = parse_groups(
                "auto",
                len(experts.mods),
                (cfg.loop_layer_index_start, cfg.loop_layer_index_end),
                src.formats(),
            )
            result["routing"] = routing.summarize(bits, rgroups, group_name)
            a.out.parent.mkdir(parents=True, exist_ok=True)
            routing.save(a.out.with_name(f"{a.out.stem}_routing.npz"), bits)
            gate(
                "routing_log_complete",
                result["routing"]["complete"],
                f"{routing.forwards} expert forwards counted; {routing.problems[:2]}",
            )
            for name, g in result["routing"].get("groups", {}).items():
                log(
                    f"routing {name}: INT2-class experts {g['experts']['int2']:.3f} of "
                    f"the experts, {g['assignments']['int2']:.3f} of the assignments, "
                    f"{g['mean_bits']:.3f} bits per assignment"
                )
        # --- where the damage is
        if groups:
            result["groups"] = {}
            for layers in groups:
                want = {
                    i: ("student" if i in layers else "teacher") for i in experts.mods
                }
                experts.set(want)
                hg = heads(ids[:bw])
                res = summarize(kl_stats(lm, hg, ht[:bw], ids[:bw], head_dev))
                res["layers"] = layers
                res["formats"] = sorted({src.expert_format(i) for i in layers})
                result["groups"][group_name(layers)] = res
                log(f"group {group_name(layers)}: KL {res['kl_mean']:.6g}")
            experts.set(experts.all("teacher"))
            again = timed("restore_program", lambda: heads(ids[:1]))
            diff = float((again - probe).abs().max())
            exact = all(d.type == "cpu" for d in devices)  # a GPU sums in any order
            tol = 0.0 if exact else 5e-3 * float(probe.abs().max())
            gate(
                "groups_restored_the_stock_weights",
                diff <= tol,
                f"max diff {diff:.3g} (tolerance {tol:.3g})",
            )
        result["weight_error_per_layer"] = {
            str(k): v for k, v in sorted(experts.weight_error.items())
        }
        result["tensors"] = {
            "installed": experts.installed,
            "changed": experts.changed,
            "missing": experts.missing,
            "expected": want_n,
        }
    changed = experts.changed / max(experts.installed, 1)
    if a.int8_class:
        gate(
            "int8_class_changed_the_weights",
            result["int8"]["weight_max_abs_error"] > 0,
            f"{result['int8']['tensors']} tensors, largest weight change "
            f"{result['int8']['weight_max_abs_error']:.3g}",
        )
    if not a.stock_student:
        gate(
            "student_differs_from_the_teacher",
            changed >= 0.5,
            f"{changed:.3f} of {experts.installed} tensors changed",
        )
    kls = torch.tensor(result["student"]["kl_per_window"])
    gate("every_value_finite", bool(torch.isfinite(kls).all()), "per-window KL")
    gate("no_negative_kl", bool(kls.min() >= -1e-6), float(kls.min()))
    if a.max_kl is not None:
        mean = result["student"]["kl_mean"]
        gate("kl_within_the_limit", mean <= a.max_kl, f"{mean:.6g} <= {a.max_kl}")
    return finish()


if __name__ == "__main__":
    sys.exit(main())
