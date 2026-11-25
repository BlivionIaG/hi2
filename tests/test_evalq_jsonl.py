# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""The held-out eval JSONL reader splits records on LF only."""
import json
from types import SimpleNamespace

import torch

from hi2.evalq import load_calibration_ids_offset


class _Tokenizer:
    def __init__(self):
        self.raw = None

    def __call__(self, raw, return_tensors=None, add_special_tokens=False):
        self.raw = raw
        return SimpleNamespace(input_ids=torch.zeros(1, 8, dtype=torch.long))


def test_eval_jsonl_splits_only_on_line_feed(tmp_path):
    """A JSON string may contain U+0085/U+2028; str.splitlines() would split the
    record and json.loads would fail on the first fragment."""
    text = "eval\x85with\u2028separators"
    path = tmp_path / "eval.jsonl"
    path.write_text(
        json.dumps({"text": text}, ensure_ascii=False) + "\n"
        + json.dumps({"text": "second"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tok = _Tokenizer()
    ids = load_calibration_ids_offset(path, tok, samples=2, seq_len=4, skip_tokens=0)
    assert tuple(ids.shape) == (2, 4)
    assert tok.raw == text + "\n\nsecond"
