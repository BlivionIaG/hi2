# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""QAD calibration text: the mixed-domain calibration file followed by the calibration region of
the repo-docs corpus (the tokens before the held-out offset), so QAD gets ~4x the PTQ tokens
without touching either held-out text.
usage: make_calib_qad.py <model_dir> <calib_mix.txt> <study_text_all.txt> <docs_tokens> <out.txt>
(docs_tokens must stay below the eval --skip-tokens of the docs text, e.g. 790000 < 800000)."""

import sys

from transformers import AutoTokenizer

model, mix, docs, n_docs, out = sys.argv[1:6]
tok = AutoTokenizer.from_pretrained(model)
with open(docs, encoding="utf-8") as f:
    text = f.read()
enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
n = min(int(n_docs), len(enc.input_ids))
cut = enc.offset_mapping[n - 1][1]
head = text[:cut]
with open(mix, encoding="utf-8") as f:
    mix_text = f.read()
with open(out, "w", encoding="utf-8") as f:
    f.write(mix_text + "\n\n" + head)
n_mix = len(tok(mix_text, add_special_tokens=False).input_ids)
print(f"mix tokens {n_mix}, docs tokens {n} (chars {cut}), written {out}")
