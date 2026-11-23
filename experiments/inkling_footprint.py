# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Where the bytes of the final-KL student of Inkling-2-Mini go over the devices.

``hi2.finalkl.student_footprint`` on the real configuration: the model is built on the
meta device (the fork's transformers, no weights), its experts are unfused, and every
expert projection gets a stand-in HI2 tensor with an average code width and block size
(the two numbers the allocator's choice comes to; the pipeline's own fit check reads the
real tensors). Prints, per dtype / code width / block / sequence length, the static bytes
of each device (packed codes, trained ladder values with gradient and Adam moments, dense
parameters), the layer inputs one sequence's backward keeps, the transient allowance and
the loss chunk. The static part is exact for the tensors it is given (a test measures it
on the tiny model); the transient allowance is an estimate. No weights are read and no
device is used.

    python experiments/inkling_footprint.py --src DIR --transformers DIR [--out FILE]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import regex as re
import torch

EXPERT = re.compile(r"\.mlp\.experts\.\d+\.(gate|up|down)_proj\.weight$")


class _Count:
    def __init__(self, n: int):
        self.n = n

    def numel(self) -> int:
        return self.n


class _Stand:
    """What ``tensor_student_bytes`` reads of an HI2 tensor."""

    def __init__(self, rows: int, cols: int, code_bits: float, block: int):
        self.shape = (rows, cols)
        self.codes = _Count(math.ceil(rows * cols * code_bits / 32))  # int32 words
        self.blk_params = _Count(
            2 * rows * math.ceil(cols / block)
        )  # lo, step per block
        self.col_scale = None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, required=True, help="stock Inkling-2 directory")
    ap.add_argument(
        "--transformers",
        type=Path,
        required=True,
        help="dir holding the fork's transformers",
    )
    ap.add_argument("--dtypes", nargs="+", default=["float32", "bfloat16"])
    ap.add_argument("--code-bits", type=float, nargs="+", default=[3.0, 3.4])
    ap.add_argument("--blocks", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[512, 1024, 2048])
    ap.add_argument("--devices", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path("inkling_footprint.json"))
    a = ap.parse_args()
    sys.path.insert(0, str(a.transformers))
    from transformers import AutoConfig, AutoModelForCausalLM

    from hi2.checkpoint import make_resolver, source_map
    from hi2.finalkl import student_footprint
    from hi2.inkling_source import register_auto_class, stock_view
    from hi2.moe import unfuse_experts
    from hi2.qad import QadConfig

    register_auto_class(a.src)
    cfg = AutoConfig.from_pretrained(a.src)
    view = stock_view(a.src)
    resolve = make_resolver(source_map(a.src))
    shapes = {
        n: [int(x) for x in view.get_slice(n).get_shape()]
        for n in view.keys()  # noqa: SIM118
        if EXPERT.search(n)
    }
    rows = []
    for dtype in a.dtypes:
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(cfg, dtype=getattr(torch, dtype))
        unfuse_experts(model)
        params = sum(p.numel() for p in model.parameters())
        for bits in a.code_bits:
            for block in a.blocks:
                tensors = {n: _Stand(r, k, bits, block) for n, (r, k) in shapes.items()}
                for seq in a.seq_lens:
                    f = student_footprint(
                        model, tensors, resolve, QadConfig(), seq, a.devices
                    )
                    gib = lambda xs: [round(x / 2**30, 2) for x in xs]  # noqa: E731
                    row = {
                        "dtype": dtype,
                        "avg_code_bits": bits,
                        "block": block,
                        "seq_len": seq,
                        "devices": a.devices,
                        "parameters": params,
                        "quantized_tensors": len(tensors),
                        "static_gib": gib(f["static"]),
                        "saved_inputs_gib": gib(f["activations"]),
                        "transient_gib": round(f["transient"] / 2**30, 2),
                        "loss_chunk_gib": round(f["logits"] / 2**30, 2),
                        "need_gib_per_device": gib(f["devices"]),
                    }
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    a.out.write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
