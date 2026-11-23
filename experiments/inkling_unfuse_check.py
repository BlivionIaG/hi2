# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Does ``hi2.moe.unfuse_experts`` keep a real Inkling-2 model running, and the same?

Builds a tiny ``Inkling2LoopForConditionalGeneration`` (the fork's transformers; the
config and shapes of the fork's ``make_tiny_checkpoint.py``), runs it with its fused 3-D
experts, replaces them with ``hi2.moe.unfuse_experts`` and runs it again. The experts of
Inkling soft-cap a two-argument gated activation, ``act_fn(gate, up) =
4 tanh(gate / 4) * sigmoid(gate) * 25 tanh(up / 25)``, not ``act_fn(gate) * up``.

Exit code 0: the unfused forward runs and its logits equal the fused ones to ``--tol``
(float32 summation order; the fused forward is ``Inkling2LoopExperts``). 1: it raises (what
``hi2/moe.py`` did while it called ``act_fn(gate_proj(x)) * up_proj(x)``) or differs. The
JSON records which.

    python experiments/inkling_unfuse_check.py --tools DIR --transformers DIR
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch

from hi2.moe import unfuse_experts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--tools", type=Path, required=True, help="tools/rdna/inkling2 of the fork"
    )
    ap.add_argument(
        "--transformers",
        type=Path,
        required=True,
        help="dir holding the fork's transformers",
    )
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument(
        "--tol", type=float, default=1e-5, help="largest allowed |logit| difference"
    )
    ap.add_argument(
        "--workdir", type=Path, default=None, help="where the tiny checkpoint goes"
    )
    ap.add_argument("--out", type=Path, default=Path("inkling_unfuse_check.json"))
    a = ap.parse_args()
    sys.path.insert(0, str(a.transformers))
    sys.path.insert(0, str(a.tools))
    import make_tiny_checkpoint as make
    from transformers import AutoConfig
    from transformers.models.inkling2_loop.modeling_inkling2_loop import (
        Inkling2LoopForConditionalGeneration,
    )

    if a.workdir is not None:
        a.workdir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="inkling_unfuse_", dir=a.workdir))
    result: dict = {"layers": a.layers, "tol": a.tol}
    try:
        make.write_checkpoint(work, make.TinySpec(num_hidden_layers=a.layers), seed=11)
        torch.manual_seed(0)
        model = Inkling2LoopForConditionalGeneration(
            AutoConfig.from_pretrained(work)
        ).eval()
        ids = torch.randint(0, 50, (1, 12))
        with torch.no_grad():
            fused = model(input_ids=ids).logits
        result["fused_forward"] = "ok"
        result["unfused_layers"] = unfuse_experts(model)
        experts = model.model.language_model.layers[0].mlp.experts
        result["experts_class"] = type(experts).__name__
        result["act_fn"] = type(experts.act_fn).__name__
        try:
            with torch.no_grad():
                unfused = model(input_ids=ids).logits
            result["unfused_forward"] = "ok"
            result["max_abs_diff"] = float((fused - unfused).abs().max())
            ok = result["max_abs_diff"] <= a.tol
        except Exception as exc:  # the point of the check is to record which one
            result["unfused_forward"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            ok = False
    finally:
        shutil.rmtree(work, ignore_errors=True)
    result["passes"] = ok
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
