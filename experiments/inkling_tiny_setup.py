# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""A toy Inkling-2 for the CPU dry run of ``inkling_stage4.sh`` (``TINY=1``).

The fork's tiny checkpoint (6 layers, loop over layers 2-4, 8 experts, hidden 256, NVFP4
experts with a last MXFP8 layer, vocabulary 500) in ``--out``, and next to its weights a toy
byte-level BPE tokenizer trained on the texts of an already staged calibration set
(``--staged``, the ``*.jsonl`` of ``inkling_build_calib.py stage``), with every
``<|...|>`` marker of those texts as a special token. The calibration builder, ``hi2
quantize`` and the KL evaluation then run on it unchanged, with scaled-down limits
(``inkling_build_calib.py build --dry-run-scale``) that its manifest records as
non-production.

Needs the fork's ``tools/rdna/inkling2`` (``INKLING2_TOOLS``) and ``tokenizers``.
"""

import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

MARKER = re.compile(r"<\|[a-z_]+\|>")
EOS = "<|content_model_end_sampling|>"
PAD = "<|endoftext|>"


def tools_dir() -> Path:
    """The fork's ``tools/rdna/inkling2`` (``INKLING2_TOOLS`` or the sibling worktree)."""
    cands = (
        [Path(os.environ["INKLING2_TOOLS"])] if os.environ.get("INKLING2_TOOLS") else []
    )
    cands.append(
        Path(__file__).resolve().parents[2] / "vllm-inkling2-merge/tools/rdna/inkling2"
    )
    for c in cands:
        if (c / "make_tiny_checkpoint.py").exists():
            return c
    raise SystemExit("make_tiny_checkpoint.py not found: set INKLING2_TOOLS")


def staged_texts(staged: Path) -> list[str]:
    files = sorted(staged.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"{staged}: no *.jsonl (run inkling_build_calib.py stage)")
    return [
        json.loads(line)["text"]
        for f in files
        for line in f.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def train_tokenizer(texts: list[str], vocab: int):
    """A byte-level BPE of ``vocab`` entries; the markers of ``texts`` are special."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    markers = sorted({m for t in texts for m in MARKER.findall(t)} | {EOS, PAD})
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab,
        special_tokens=markers,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, eos_token=EOS, pad_token=PAD
    ), markers


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True, help="the toy model directory")
    ap.add_argument("--staged", type=Path, required=True, help="calibration src/ dir")
    ap.add_argument("--vocab", type=int, default=480, help="<= 500, the model's")
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args(argv)
    if a.vocab > 500:
        raise SystemExit("--vocab: the toy model has 500 tokens")
    path = tools_dir() / "make_tiny_checkpoint.py"
    spec = importlib.util.spec_from_file_location("hi2_tiny_make", path)
    make = importlib.util.module_from_spec(spec)
    sys.modules["hi2_tiny_make"] = make
    spec.loader.exec_module(make)
    a.out.mkdir(parents=True, exist_ok=True)
    make.write_checkpoint(a.out, make.TinySpec(num_hidden_layers=a.layers), seed=a.seed)
    tok, markers = train_tokenizer(staged_texts(a.staged), a.vocab)
    tok.save_pretrained(a.out)
    print(
        json.dumps(
            {
                "out": str(a.out),
                "layers": a.layers,
                "tokenizer_vocabulary": len(tok),
                "special_markers": markers,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
