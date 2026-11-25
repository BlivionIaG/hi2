# Data sources and licences

All evaluation data is vendored in this directory (65,664 bytes of JSONL, an
8.5 kB Python module and a 573 kB haystack snapshot). Nothing is downloaded at evaluation time, no
private data is used, and every run stays far below the 2 GiB budget: a full
245-item run JSONL is about 1.8 MB because it keeps the rendered prompts.

Regenerate the three Hub subsets (fixed rows) with:

```bash
HF_DATASETS_OFFLINE=1 python -m tools.inkling_eval.build_data
```

The export used `datasets` 4.8.5 on the GPU box on 2026-11-24, reading the local
Hugging Face caches (no download was needed for these three). Update this file
when the hashes change; `build_data` prints them.

| file | source | rows | licence | bytes | sha256 |
| --- | --- | ---: | --- | ---: | --- |
| `data/gsm8k.jsonl` | `openai/gsm8k`, config `main`, split `test`, rows 0–39; cache revision `740312add88f781978c0658806c59bc2815b9866` | 40 | MIT | 23,884 | `cfc6c975ff60c01b62ff93b1a5c63f2ce4aff033536eb888821d5904d3ae7f7d` |
| `data/mbpp.jsonl` | `google-research-datasets/mbpp`, config `sanitized`, split `test`, rows 0–9; cache revision `4bb6404fdc6cacfda99d4ac4205087b89d32030c` | 10 | CC-BY-4.0 | 5,554 | `26986c6497680bb618c93053bec45b63a1c86d47d78fc7394aae32f3f1de9236` |
| `data/humaneval.jsonl` | `openai/openai_humaneval`, split `test`, rows 0–19; cache revision `7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544` | 20 | MIT | 19,423 | `a33df3d2d7f12748f18eb6ec6eae25824253886e5028f335ab3138d6fe7d2413` |
| `data/c_code.jsonl` | authored for this harness | 5 | Apache-2.0 | 3,561 | `14286849e2da0f4a2d1cb0e4afeae1e0cf9ee7ea111a9b9bf5a5c2aadf0e33cf` |
| `data/ts_code.jsonl` | authored for this harness | 5 | Apache-2.0 | 3,495 | `5909d5e3ae672d30bbf0cf239456d3549dfed74b129f8c3a0a7f73c4e3a29347` |
| `data/math_authored.jsonl` | authored for this harness | 8 | Apache-2.0 | 1,203 | `f94e17b6227f25b6ce100d3c9948bab42612eb3ea47db283f3593688041e6f6d` |
| `multilingual_data.py` | authored for this harness: 14 languages × 2 translation references, 2 numeric and 1 colour question | 70 items | Apache-2.0 | 8,544 | — |
| `data/haystack.txt` | frozen snapshot of this repository's `README.md` and `docs/*.md`, concatenated in sorted order on 2026-11-24 | 1 file | Apache-2.0 | 573,042 | `20f81e6b1aceba112d254322817d7ecf94c4ca8e92f543f9cb339d8cbbab4455` |

The GSM8K and MBPP licence holders do not endorse this evaluation. The
HumanEval `test` and `canonical_solution` fields are executed only inside the
sandbox described in `docs/inkling-task-eval.md`.

## Long-context haystack

`data/haystack.txt` is a **frozen** snapshot of this repository's `README.md`
and `docs/*.md` (Apache-2.0), concatenated in sorted order with an
`<!-- path -->` marker before each document. Freezing it means a documentation
edit never changes the 8k/16k/32k prompts between paired runs; the snapshot hash
is recorded in the run meta and in every long-context item.

To refresh it deliberately, rebuild exactly that concatenation (this repository's
`README.md`, then `docs/*.md` sorted by path, each preceded by
`<!-- relative/path -->` and separated by a blank line), write it here, then
update this table and rerun `plan`; both labels must use the same snapshot.

## Synthetic vision set

`tools/inkling_eval/vision_data.py` generates the 88-item vision set and its 80
PNG images at plan-build time: 24 OCR codes/numbers, 20 bar/line charts, 20
shape/colour counts, 16 table reads and 8 text-only controls. It embeds its own
5x7 bitmap font, uses no Pillow or downloaded font, and writes indexed PNGs with
stored DEFLATE so the bytes and plan hash are independent of the host zlib.
There is no external source, photograph or private data; the generated material
is Apache-2.0. Each image is 480x240 (72 vision cells) and about 115 kB; the
eight text controls carry no image. The mock server validates that every image part
is a real base64 PNG data URL, and the dry-run sends the full set through the
chat endpoint stub.

## Sources considered and not used

* `facebook/flores` (CC-BY-SA-4.0): the Hub dataset is gated, and the box is not
  authenticated, so it cannot be loaded offline.
* `Helsinki-NLP/opus-100` (CC-BY-4.0): cached, but its `test` pairs are
  automatically mined and include misaligned rows (the English side of one row
  describes a different sentence than the reference). chrF against a misaligned
  reference would penalise a correct translation, so the translation set is
  authored instead.
