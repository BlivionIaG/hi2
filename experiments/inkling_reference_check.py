# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Bit-exactness of ``hi2.inkling_source`` against the vLLM fork's reference decoders on
every expert of the real Inkling-2-Mini checkpoint.

``tools/rdna/inkling2/tiny_reference.py`` of the fork decodes the stock NVFP4 / MXFP8 expert
tensors to float32 (``dequant_nvfp4``, ``dequant_mxfp8``). For every layer, both stacks and
every expert (a chunk at a time) this script decodes the same bytes with the reference and
with ``InklingSource.stack`` (and the de-interleaved ``gate_up`` / ``down``), compares the
float32 bit patterns, and records the largest absolute difference (0.0 when equal).
``tests/test_inkling_source.py`` does the same for four experts of layers 0 and 20.

    python experiments/inkling_reference_check.py CKPT --tools ~/vllm-inkling2-merge/tools/rdna/inkling2
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

from hi2.inkling_source import InklingSource


def load_reference(tools: Path):
    spec = importlib.util.spec_from_file_location(
        "tiny_reference", tools / "tiny_reference.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tiny_reference"] = mod
    spec.loader.exec_module(mod)
    return mod


def bits(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(torch.int32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt", type=Path)
    ap.add_argument(
        "--tools", type=Path, required=True, help="tools/rdna/inkling2 of the fork"
    )
    ap.add_argument("--layers", default="all", help="comma list or 'all'")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("inkling_reference_check.json"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    ref = load_reference(a.tools)
    shard = a.ckpt / "model-00001-of-00001.safetensors"
    result = {
        "checkpoint": str(a.ckpt),
        "reference": str(a.tools / "tiny_reference.py"),
        "layers": [],
    }
    all_equal = True
    with InklingSource(a.ckpt) as src, safe_open(shard, "pt") as f:
        want = (
            range(src.num_layers)
            if a.layers == "all"
            else [int(x) for x in a.layers.split(",")]
        )
        for layer in want:
            t0 = time.time()
            fmt = src.expert_format(layer)
            base = f"model.llm.layers.{layer}.mlp.experts"
            entry = {"layer": layer, "format": fmt}
            for which in ("w13", "w2"):
                name = f"{base}.{which}_weight"
                w_sl, s_sl = f.get_slice(name), f.get_slice(f"{name}_scale")
                if fmt == "nvfp4":
                    amax_sl = f.get_slice(f"{name}_amax")
                    den = f.get_tensor(f"{name}_outer_scale_denominator")
                equal, diff, n = True, 0.0, 0
                for lo in range(0, src.n_experts, a.chunk):
                    hi = min(lo + a.chunk, src.n_experts)
                    if fmt == "nvfp4":
                        want_t = ref.dequant_nvfp4(
                            w_sl[lo:hi], s_sl[lo:hi], amax_sl[lo:hi], den
                        )
                    else:
                        want_t = ref.dequant_mxfp8(w_sl[lo:hi], s_sl[lo:hi])
                    got = src.stack(layer, which, range(lo, hi))
                    ok = torch.equal(bits(got), bits(want_t))
                    equal &= ok
                    diff = max(diff, float((got - want_t).abs().max()))
                    if (
                        which == "w13"
                    ):  # the de-interleaved views are its even / odd rows
                        gate, up = src.gate_up(layer, range(lo, hi))
                        equal &= torch.equal(bits(gate), bits(want_t[:, 0::2]))
                        equal &= torch.equal(bits(up), bits(want_t[:, 1::2]))
                    else:
                        equal &= torch.equal(
                            bits(src.down(layer, range(lo, hi))), bits(want_t)
                        )
                    n += want_t.numel()
                entry[which] = {
                    "bit_equal": bool(equal),
                    "max_abs_diff": diff,
                    "elements": n,
                }
                all_equal &= equal
            entry["seconds"] = time.time() - t0
            result["layers"].append(entry)
            print(
                f"layer {layer:2d} {fmt}: w13 equal={entry['w13']['bit_equal']} "
                f"w2 equal={entry['w2']['bit_equal']} max abs diff "
                f"{max(entry['w13']['max_abs_diff'], entry['w2']['max_abs_diff'])} "
                f"[{entry['seconds']:.0f}s]",
                flush=True,
            )
    result["all_bit_equal"] = bool(all_equal)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"all bit equal: {all_equal}; wrote {a.out}")
    return 0 if all_equal else 1


if __name__ == "__main__":
    sys.exit(main())
