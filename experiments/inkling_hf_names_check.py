# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""The reader's names and shapes against the real Hugging Face model class.

Builds ``Inkling2LoopForConditionalGeneration`` of the checkpoint's config on the meta
device (no weights), unfuses its experts with ``hi2.moe.unfuse_experts`` and checks that
every parameter and buffer of the text model is a tensor ``hi2.inkling_source`` can
produce (``hf_names`` for the dense tensors, ``message_input.query`` composed from the
cross-pass tensors, ``hf_expert_names`` for the experts) with the shape the model has,
and the other way round. The vision tower is not read by the reader and is only counted.

The class is not in the pinned transformers: ``--transformers`` is the directory of a
transformers build that has ``models/inkling2_loop`` (the vLLM fork's tfm-main).

    python experiments/inkling_hf_names_check.py CKPT --transformers ~/w4a8_runs/pydeps/tfm-main
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from hi2.inkling_source import MESSAGE_QUERY, InklingSource, expert_name
from hi2.moe import unfuse_experts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt", type=Path)
    ap.add_argument(
        "--transformers",
        type=Path,
        required=True,
        help="dir holding the fork's transformers",
    )
    ap.add_argument("--out", type=Path, default=Path("inkling_hf_names_check.json"))
    a = ap.parse_args()
    sys.path.insert(0, str(a.transformers))
    from transformers import AutoConfig
    from transformers.models.inkling2_loop.modeling_inkling2_loop import (
        Inkling2LoopForConditionalGeneration,
    )

    cfg = AutoConfig.from_pretrained(a.ckpt)
    with torch.device("meta"):
        model = Inkling2LoopForConditionalGeneration(cfg)

    def tensors(m) -> dict[str, tuple[int, ...]]:
        return {
            n: tuple(t.shape) for n, t in [*m.named_parameters(), *m.named_buffers()]
        }

    with InklingSource(a.ckpt) as src:
        e, i, h = src.n_experts, src.inter, src.hidden
        fused = {
            n: s
            for n, s in tensors(model).items()
            if n.endswith(("mlp.experts.gate_up_proj", "mlp.experts.down_proj"))
        }
        fused_ok = all(
            s == ((e, 2 * i, h) if n.endswith("gate_up_proj") else (e, h, i))
            for n, s in fused.items()
        )
        unfused = unfuse_experts(model)
        have = tensors(model)

        produced: dict[str, tuple[int, ...]] = {}
        for serving, hf in src.hf_names().items():
            shape = src.header(serving)[1]
            produced[hf] = (
                (src.unpadded_vocab, *shape[1:]) if hf == "lm_head.weight" else shape
            )
        produced[MESSAGE_QUERY] = tuple(src.message_query().shape)
        for layer in range(src.num_layers):
            for ex in range(e):
                produced[expert_name(layer, ex, "gate_proj")] = (i, h)
                produced[expert_name(layer, ex, "up_proj")] = (i, h)
                produced[expert_name(layer, ex, "down_proj")] = (h, i)

        vision = [n for n in have if n.startswith("model.vision_tower")]
        text = {n: s for n, s in have.items() if n not in set(vision)}
        absent = sorted(n for n in produced if n not in have)
        wrong = {
            n: {"reader": produced[n], "model": s}
            for n, s in text.items()
            if n in produced and produced[n] != s
        }
        unproduced = sorted(n for n in text if n not in produced)
        result = {
            "checkpoint": str(a.ckpt),
            "transformers": str(a.transformers),
            "class": type(model).__name__,
            "model_tensors": len(have),
            "model_parameters_text": len(text),
            "vision_tensors_not_read": len(vision),
            "fused_expert_tensors": len(fused),
            "fused_shapes_are_gate_up_2I_H_and_down_H_I": fused_ok,
            "unfused_layers": unfused,
            "reader_names": len(produced),
            "reader_names_absent_from_model": absent[:20],
            "n_absent": len(absent),
            "shape_mismatches": wrong,
            "n_mismatch": len(wrong),
            "text_tensors_the_reader_does_not_produce": unproduced,
            "n_unproduced": len(unproduced),
            "message_query_shape": produced[MESSAGE_QUERY],
        }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if not isinstance(v, (list, dict)) or len(v) < 5
            },
            indent=1,
        )
    )
    ok = fused_ok and not absent and not wrong and not unproduced
    print("coverage complete:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
