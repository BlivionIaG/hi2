# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Scale and magnitude statistics of the Inkling-2-Mini routed experts.

Per layer and stack (``w13``, ``w2``), measured over every expert:

* the histogram of the block scales as stored: UE8M0 bytes of the MXFP8 layers (the
  exponent is ``u - 127``), E4M3 bytes of the NVFP4 layers, the per-expert ``amax`` and
  the global factor ``amax / 2688``;
* over every dequantized weight (decoded exactly, in float32, ``hi2.inkling_source``):
  the largest magnitude, the histogram of ``floor(log2 |w|) + 1`` (the ``frexp``
  exponent), the count of weights that overflow, underflow or land in the subnormal
  range of float16 and of bfloat16, and the relative energy the cast to each of them
  changes (``sum (w - cast(w))^2 / sum w^2``).

Question answered: is float16 usable for the dequantized teacher on gfx10, where bf16 has
no native GEMM? This script measures the *weights*; the activations of the model are not
looked at here.

    python experiments/inkling_scale_stats.py CKPT --out scale_stats.json [--layers 0,20]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from hi2.inkling_source import InklingSource, e4m3_to_float32

FP16_MAX_ROUND = 65520.0  # |w| >= this rounds to inf in float16
FP16_MIN_NORMAL = 2.0**-14
FP16_HALF_MIN_SUBNORMAL = 2.0**-25  # |w| <= this rounds to 0 (ties to even)
BF16_MAX_ROUND = float(
    2.0**128 * (1 - 2.0**-9)
)  # |w| >= this rounds to inf in bfloat16
BF16_MIN_NORMAL = 2.0**-126
BF16_HALF_MIN_SUBNORMAL = 2.0**-134


def _hist_bytes(raw: torch.Tensor) -> dict[int, int]:
    counts = torch.bincount(raw.view(torch.uint8).reshape(-1).long(), minlength=256)
    return {i: int(c) for i, c in enumerate(counts.tolist()) if c}


def _scale_summary(fmt: str, hist: dict[int, int]) -> dict:
    """Min / max / percentile decoded scale of a byte histogram."""
    codes = torch.tensor(sorted(hist), dtype=torch.uint8)
    counts = torch.tensor([hist[int(c)] for c in codes], dtype=torch.float64)
    if fmt == "mxfp8":
        vals = codes.to(torch.float64) - 127  # the exponent
    else:
        vals = e4m3_to_float32(codes).double()
    keep = ~torch.isnan(vals)  # (E4M3 NaN codes are counted separately below)
    vals, counts = vals[keep], counts[keep]
    order = torch.argsort(vals)
    vals, counts = vals[order], counts[order]
    cum = torch.cumsum(counts, 0) / counts.sum()

    def q(p):
        return float(
            vals[int(torch.searchsorted(cum, torch.tensor(p, dtype=torch.float64)))]
        )

    out = {
        "min": float(vals[0]),
        "p01": q(0.01),
        "median": q(0.5),
        "p99": q(0.99),
        "max": float(vals[-1]),
        "n": int(counts.sum()),
    }
    if (
        fmt == "nvfp4"
    ):  # E4M3: zero, subnormal codes (exponent field 0) and the saturated 448
        sub = sum(c for b, c in hist.items() if b & 0x7F and not (b >> 3) & 0xF)
        out["zero"] = sum(c for b, c in hist.items() if not b & 0x7F)
        out["subnormal_codes"] = sub
        out["at_448"] = sum(c for b, c in hist.items() if b & 0x7F == 0x7E)
        out["nan"] = sum(c for b, c in hist.items() if b & 0x7F == 0x7F)
    else:
        out["u_is_0"] = hist.get(0, 0)
        out["u_is_255"] = hist.get(255, 0)
    return out


class Acc:
    """Accumulates the weight statistics of one stack over its expert chunks."""

    def __init__(self):
        self.n = self.zero = 0
        self.max_abs = 0.0
        self.sumsq = self.err16 = self.errbf = 0.0
        self.counts = dict.fromkeys(
            (
                "fp16_overflow",
                "fp16_subnormal",
                "fp16_underflow_to_zero",
                "bf16_overflow",
                "bf16_subnormal",
                "bf16_underflow_to_zero",
            ),
            0,
        )
        self.exp_hist: dict[int, int] = {}
        self.expert_max: list[float] = []

    def add(self, w: torch.Tensor) -> None:
        """``w``: float32 ``[e, N, K]``."""
        a = w.abs()
        self.expert_max += [float(v) for v in a.amax(dim=(1, 2))]
        self.max_abs = max(self.max_abs, max(self.expert_max[-w.shape[0] :]))
        self.n += w.numel()
        nz = a > 0
        self.zero += int(w.numel() - nz.sum())
        c = self.counts
        c["fp16_overflow"] += int((a >= FP16_MAX_ROUND).sum())
        c["fp16_underflow_to_zero"] += int((nz & (a <= FP16_HALF_MIN_SUBNORMAL)).sum())
        c["fp16_subnormal"] += int(
            ((a > FP16_HALF_MIN_SUBNORMAL) & (a < FP16_MIN_NORMAL)).sum()
        )
        c["bf16_overflow"] += int((a >= BF16_MAX_ROUND).sum())
        c["bf16_underflow_to_zero"] += int((nz & (a <= BF16_HALF_MIN_SUBNORMAL)).sum())
        c["bf16_subnormal"] += int(
            ((a > BF16_HALF_MIN_SUBNORMAL) & (a < BF16_MIN_NORMAL)).sum()
        )
        _, e = torch.frexp(a[nz])
        bins = torch.bincount((e + 200).reshape(-1).long(), minlength=400)
        for i, k in enumerate(bins.tolist()):
            if k:
                self.exp_hist[i - 200] = self.exp_hist.get(i - 200, 0) + k
        self.sumsq += float(torch.sum(w * w, dtype=torch.float64))
        h = w.clamp(-65504.0, 65504.0).half().float()  # an overflow is counted above
        self.err16 += float(torch.sum((w - h) ** 2, dtype=torch.float64))
        b = w.bfloat16().float()
        self.errbf += float(torch.sum((w - b) ** 2, dtype=torch.float64))

    def result(self) -> dict:
        ss = max(self.sumsq, 1e-300)
        return {
            "weights": self.n,
            "exact_zeros": self.zero,
            "max_abs": self.max_abs,
            "rms": (self.sumsq / max(self.n, 1)) ** 0.5,
            **self.counts,
            "fp16_cast_rel_energy": self.err16 / ss,
            "bf16_cast_rel_energy": self.errbf / ss,
            "frexp_exponent_hist": dict(sorted(self.exp_hist.items())),
            "expert_max_abs": {
                "min": min(self.expert_max),
                "median": sorted(self.expert_max)[len(self.expert_max) // 2],
                "max": max(self.expert_max),
            },
        }


def layer_stats(src: InklingSource, layer: int, chunk: int) -> dict:
    fmt = src.expert_format(layer)
    out = {"layer": layer, "format": fmt}
    for which in ("w13", "w2"):
        base = f"model.llm.layers.{layer}.mlp.experts.{which}_weight"
        hist = _hist_bytes(src.tensor(base + "_scale"))
        entry = {"scale_bytes_hist": hist, "scale": _scale_summary(fmt, hist)}
        if fmt == "nvfp4":
            amax = src.tensor(base + "_amax")
            den = float(src.tensor(base + "_outer_scale_denominator")[0])
            entry["amax"] = {
                "min": float(amax.min()),
                "median": float(amax.median()),
                "max": float(amax.max()),
                "denominator": den,
            }
            entry["global_factor"] = {
                "min": float(amax.min()) / den,
                "max": float(amax.max()) / den,
            }
        acc = Acc()
        for a in range(0, src.n_experts, chunk):
            acc.add(src.stack(layer, which, range(a, min(a + chunk, src.n_experts))))
        entry["dequantized"] = acc.result()
        out[which] = entry
    return out


def totals(layers: list[dict]) -> dict:
    keys = (
        "weights",
        "exact_zeros",
        "fp16_overflow",
        "fp16_subnormal",
        "fp16_underflow_to_zero",
        "bf16_overflow",
        "bf16_subnormal",
        "bf16_underflow_to_zero",
    )
    t = dict.fromkeys(keys, 0)
    sumsq = e16 = ebf = 0.0
    max_abs = 0.0
    for lay in layers:
        for which in ("w13", "w2"):
            d = lay[which]["dequantized"]
            for k in keys:
                t[k] += d[k]
            n = d["weights"]
            ss = d["rms"] ** 2 * n
            sumsq += ss
            e16 += d["fp16_cast_rel_energy"] * ss
            ebf += d["bf16_cast_rel_energy"] * ss
            max_abs = max(max_abs, d["max_abs"])
    t["max_abs"] = max_abs
    t["fp16_cast_rel_energy"] = e16 / sumsq
    t["bf16_cast_rel_energy"] = ebf / sumsq
    t["fp16_subnormal_or_underflow_ppm"] = (
        1e6 * (t["fp16_subnormal"] + t["fp16_underflow_to_zero"]) / t["weights"]
    )
    return t


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--out", type=Path, default=Path("inkling_scale_stats.json"))
    ap.add_argument("--layers", default="all", help="comma list or 'all'")
    ap.add_argument("--chunk", type=int, default=8, help="experts decoded at a time")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    layers = []
    with InklingSource(a.ckpt) as src:
        want = (
            list(range(src.num_layers))
            if a.layers == "all"
            else [int(x) for x in a.layers.split(",")]
        )
        for layer in want:
            t0 = time.time()
            st = layer_stats(src, layer, a.chunk)
            layers.append(st)
            w13, w2 = st["w13"]["dequantized"], st["w2"]["dequantized"]
            print(
                f"layer {layer:2d} {st['format']}: max|w| {max(w13['max_abs'], w2['max_abs']):.4g} "
                f"fp16 overflow {w13['fp16_overflow'] + w2['fp16_overflow']} "
                f"subnormal {w13['fp16_subnormal'] + w2['fp16_subnormal']} "
                f"underflow {w13['fp16_underflow_to_zero'] + w2['fp16_underflow_to_zero']} "
                f"[{time.time() - t0:.0f}s]",
                flush=True,
            )
    result = {
        "checkpoint": str(a.ckpt),
        "thresholds": {
            "fp16_overflow_at": FP16_MAX_ROUND,
            "fp16_min_normal": FP16_MIN_NORMAL,
            "fp16_underflow_to_zero_at_or_below": FP16_HALF_MIN_SUBNORMAL,
            "bf16_min_normal": BF16_MIN_NORMAL,
            "bf16_underflow_to_zero_at_or_below": BF16_HALF_MIN_SUBNORMAL,
        },
        "layers": layers,
        "totals": totals(layers),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result["totals"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
