# Calibration and held-out/OOD sets v2 for Inkling-2-Mini

Status: staged, built and verified on the CPU of the build box on 2026-11-24
(`stage4/calib_v2`). **Nothing here has run on a GPU.** No quality result is reported or
implied: this document describes data, a gate report and cost estimates, and nothing about a
quantized model's output.

Why v2: the stage-4 set (`docs/inkling-source.md`) is 2048-token windows of mostly English
chat, tool use as plain text, arithmetic and technical text, and the only OOD sets are 17
windows of English Wikipedia and 17 of vLLM docs (stage 4, 2048 tokens each). The deployment
target (HI2 at 3.5 bpw, the routed experts quantized, the rest bf16) is used by a coding agent
on long inputs in several languages, so the set that fits the quantizer and the set that judges
it must contain those shapes.

## What is in it

`experiments/inkling_build_calib_v2.py` has four subcommands: `stage` (render and stage the
sources as `<category>.jsonl`), `build` (dedupe, split, window, gate, write the arrays and
`manifest.json`), `ood` (a held-out set from sources that are in no staged category) and
`verify --production` (re-hash every file, re-check the schema and every hard gate).

| Category | Staged records | Sources (licences) |
| --- | --- | --- |
| agentic | 344 | `Team-ACE/ToolACE` (apache-2.0); `NousResearch/hermes-function-calling-v1` (apache-2.0); `tools/agentic_bench` system prompts + tasks of this repository (apache-2.0), rendered as multi-turn tool sessions |
| code | 424 | local project trees: llama.cpp app/common/src (MIT), Linux 6.19 drivers (GPL-2.0), rocprofiler-sdk (MIT), this repository's `hi2`, `experiments`, `bench` (apache-2.0); public: `deepmind/code_contests`, `open-r1/codeforces` (apache-2.0) |
| multilingual | 329 | `wikimedia/wikipedia` 20231101.{ar,de,es,fr,hi,ja,ko,pt,ru,tr,vi,zh} (cc-by-sa-4.0); `Helsinki-NLP/opus-100` en-xx/xx-en pairs (cc-by-4.0) |
| long_context | 102 | `emozilla/pg19` test books (public-domain); llama.cpp `src` (MIT); this repository's `hi2` (apache-2.0); agentic_bench transcripts (apache-2.0) |
| math_reasoning | 1619 | `openai/gsm8k` train (mit); `open-r1/codeforces` (apache-2.0) |
| general_chat | 400 | `timdettmers/openassistant-guanaco` (apache-2.0); `WizardLM/WizardLM_evol_instruct_70k` (mit); `HuggingFaceH4/ultrachat_200k` (mit) |

Every staged record carries `text`, `category`, `language`, `source` and `license`; the
manifest aggregates the same fields per window. "Local project trees" are public source trees
present on the build box; no private data is used. The Linux driver rust files are
GPL-2.0 text: they are not redistributed with the model, but a production set that must avoid
copyleft text should drop the `rust` key and restage. The public code datasets are permissive.

The tool schemas and calls are native turns: a ToolACE record's tools become a `tool_declare`
system turn and its call/result pair becomes `role: tool` turns with `tool_calls`/`tool_call_id`
rendered by the Inkling chat template; the Hermes records already carry those turns. The
agentic_bench system prompts are the real coding-agent ones, not a paraphrase.

Long-context documents are coherent, not random concatenation: PG19 books and repo files are
sliced as one contiguous document, and the agent transcripts are one multi-turn session per
system prompt. A window never mixes documents. Long windows exist at 8192, 16384 and 32768
tokens (the checkpoint's `mtp.config.max_position_embeddings` is 1,048,576, so the cap is the
data, not the model); the same source documents feed the three lengths, and a document shorter
than a length contributes only to the shorter ones.

## Calibration / held-out split

The split is by document, stratified by category and language (`_split`): a document is never
in both splits, and a small language is guaranteed its share instead of losing every document
to the global 20 %. Dedupe first: exact sha256 (a document is never repeated), then
near-duplicate shingles per category (a document whose 64-character shingle set overlaps an
already kept document by 0.5 or more is dropped; shingles are sampled every 8 characters to
keep a 100 MB staging tractable). Measured: **54 exact duplicates dropped, 56 near-duplicates
dropped**. After windowing, every held-out window is checked against the calibration tokens of
its category (32-token shingles, sampled every 8 tokens). A window at or above 0.5 is a leak
that the sampled text dedupe missed; it is dropped and the count recorded. Measured: one held
C window shared 1.0 of its shingles with the calibration and was dropped; the kept maxima are
0.4923 (`code/c`) and 0.4601 (`code/json`). The manifest keeps
`mean`/`p99`/`max` of the kept windows, `leaked_max` before the filter and `dropped`.

Calibration: **714 windows / 3,100,672 tokens**. Held-out: **696 windows / 2,357,248
tokens**.

| Category | Calibration windows | Calibration tokens | Held-out windows | Held-out tokens |
| --- | ---: | ---: | ---: | ---: |
| agentic | 32 | 65,536 | 30 | 61,440 |
| code | 196 | 401,408 | 208 | 425,984 |
| general_chat | 33 | 67,584 | 32 | 65,536 |
| long_context | 96 | 1,835,008 | 57 | 1,048,576 |
| math_reasoning | 32 | 65,536 | 29 | 59,392 |
| multilingual | 325 | 665,600 | 340 | 696,320 |

Long context by length: calibration 32 windows at each of 8192/16384/32768 (262,144 /
524,288 / 1,048,576 tokens); held-out 20 / 20 / 17 windows (163,840 / 327,680 / 557,056
tokens).

Multilingual held-out windows by language: ar 32, de 19, es 26, fr 32, hi 22, ja 38, ko 22,
pt 39, ru 32, tr 21, vi 32, zh 25. Twelve languages across Latin, Cyrillic, Devanagari,
Arabic and CJK scripts; natural text (Wikipedia) and chat/translation turns (OPUS-100).

Code held-out windows by language: c 52, cpp 41, json 40, python 36, rust 33, typescript 6.
The staged `shell` (5 documents) and `sql` (9 documents) documents are too few or too short to
contribute a held-out 2048-token window, and no `.js` file was staged; they are present in
calibration only (or absent). A per-code-language KL claim for those three is not supported.

Out-of-distribution: `ood` adds a separate set of 16 held-out windows of 2048 tokens from
English Wikipedia (`wikimedia/wikipedia` 20231101.en, cc-by-sa-4.0), a source that appears in
no staged category; the manifest's `ood` block records the source, licence and reason. It is
scored with `--split ood`.

The set is 51 `.npy` files (int32 `[windows, length]`) plus `manifest.json`, 21.9 MB. Every
file's sha256, window count and length is in the manifest, and `verify` recomputes them.
`verify --production` prints `{"ok": true, "problems": []}` (exit 0); `manifest.json` sha256
`d12f0223d7fd411a57e097ebb3930af0df1b30fa44106b67ac8d620445bc793b`.

## Minimum detectable difference

The stage-4 mixed-set KL was 0.1197 ± 0.0113 (sem over 96 windows), so the per-window standard
deviation is σ ≈ 0.0113·√96 ≈ 0.111. The values below use that σ for every category
[INFERENCE: long windows may have a smaller σ because the KL is averaged over more positions,
so the long-context MDE may be optimistic; a category with a different distribution can have a
larger σ]. MDE is the 80 %-power, two-sided 5 % detectable difference against a fixed value
(`1.96·σ/√n`) and between two independent equal-n sets (`3.96·σ/√n`, from
(1.96+0.84)·√2); a paired per-window comparison uses the variance of the per-window difference
(usually far below 2σ²) and is the right test once both models are scored on the same windows.

| Set | n | MDE vs fixed | MDE, two sets |
| --- | ---: | ---: | ---: |
| agentic | 30 | 0.0396 | 0.0800 |
| code | 208 | 0.0150 | 0.0304 |
| general_chat | 32 | 0.0384 | 0.0775 |
| long_context | 57 | 0.0287 | 0.0581 |
| math_reasoning | 29 | 0.0403 | 0.0814 |
| multilingual | 340 | 0.0118 | 0.0238 |
| multilingual, de (smallest) | 19 | 0.0498 | 0.1006 |
| code, c (largest) | 52 | 0.0301 | 0.0608 |
| code, typescript | 6 | 0.0886 | 0.1790 |

A category-level KL difference of ~0.015 is detectable for code and multilingual, ~0.03-0.04
for agentic/general/math and long context; per-language claims are at best ~0.03-0.05 and the
typescript point is exploratory only.

## Scoring

`experiments/inkling_kl_eval.py` is extended without changing the old results:

* A single-file `--ids heldout_mixed_ids.npy` still splits the file into windows and scores
  the default groups, routing, tokens and out; the result and the out JSON are unchanged.
* `--ids DIR` (a v2 directory) scores every file of `--split` (default `heldout`; `calib` and
  `ood` too) with its own window length in one model load. The key is the file name
  `split_category__language__length_ids.npy`; every file is checked against the manifest's
  sha256 and size before it is loaded.
* `--position-buckets 0-2048,2048-8192,8192-16384,16384-` adds the KL of each position
  bucket. The ranges are half-open and per window (`0-2048` is tokens 0..2047 of the window,
  not global positions), the last bucket takes every position past its lower bound.
* `--batch-windows N` runs the layer program on at most N windows per forward (0, the
  default, is every window at once, the stage-4 behaviour); the 8k/16k/32k windows need a
  small N to fit the cards.
* `--groups`, `--teacher-cache`, `--routing-log` and `--int8-class` refuse a directory with a
  message: they are per-window or per-file and the set path scores the whole student in one
  load.

`result["eval_set"]` is `{dir, split, sets, category, language}`. `sets[key]` has the file's
category, language, length, windows, the summary and `position_buckets`; `category` and
`language` pool the same windows. The old keys (`student`, `gates`, `seconds`, ...) are
unchanged. Loading a directory refuses a manifest with `ok=false`, a missing gate block or a
file whose sha256/size differs. The tests cover the loader (schema, gates, sha256), the
position buckets and their pooling, and one end-to-end set run with a tiny model and two
windows of different lengths (4 and 8 tokens); the single-file path keeps the old tests.

The production command:
`--src STOCK --dst HI2 --ids <calib_v2> --split heldout --position-buckets
0-2048,2048-8192,8192-16384,16384- --out ...`.

## Costs on the two V620 cards

Measured on the CPU of the build box: 3.66 ms/token for the layer program and 1.9 s / 6.8 s
per forward for 8k / 32k windows (stage-3/4 numbers, 2 x 24 GiB V620, no quantized expert path).
For one 8k window, teacher and student passes: 1.9 s each, ~4 s; 32k: ~14 s. **Estimated**
[INFERENCE] for the v2 held-out set: 57 long windows (20 x 8k, 20 x 16k, 17 x 32k) at 2 passes
and 2 directions ≈ 20·2·1.9 + 20·2·3.8 + 17·2·6.8 ≈ 459 s ≈ 8 min for one model's KL, plus
the 639 short windows ≈ 639·2·0.47 ≈ 10 min; the full 696 held-out windows and 714 calibration
windows with a teacher cache is under an hour on the cards. Memory for one 32k window:
hidden 2304, 24 layers, 8 KV heads of 128; `local_layer_ids` has 20 layers with a 1024
window and 4 global layers (5, 11, 17, 23): the KV cache is
(4·32768 + 20·1024)·8·128·2 B ≈ 0.31 GB, the activation row is 2304·2 B ≈ 4.5 KiB per
token, and the attention scores of the four global layers are 4·24·32768·32768·2 B ≈ 206 GB
if the fused flash kernel is not used. The 32k pass therefore requires FA and a per-layer
streaming/eviction plan; the stage-3/4 measurement (peak state 104,472 MiB, the teacher cache
8.1 GiB for 96 2048 windows) does not cover it.

## GPU smoke checklist (first lease, 2 x 24 GiB; no GPU has run this yet)

1. `experiments/inkling_build_calib_v2.py verify --out <calib_v2> --tokenizer <stock>
   --production` on the box's copy: must exit 0 with `"problems": []`; do this before anything
   GPU.
2. Load the stock checkpoint and run one 2048-token window through the layer program; record
   peak memory and time. Then 8192, 16384, 32768 on a single window with flash attention
   enabled; record peak and whether the 32k pass completes. The expected ceiling is between
   16k and 32k on 2 x 24 GiB if attention is fused and the cache is not materialized
   [INFERENCE]; if 32k does not fit, find the exact max by stepping 1024 tokens.
3. Confirm the layer-program stats pass (`--stats`) covers 32k without NaNs and that the
   float32 evaluator's teacher/student logits agree with the bf16 path to the usual tolerance
   on a 2k window.
4. Check that `--ids <calib_v2> --split heldout` scores at least one window of each length and
   that the per-position buckets are populated; score the same windows one `.npy` file at a
   time through the single-file `--ids` path and compare the per-window KL.
5. Record the actual bytes/seconds per window in the run's `seconds` block and update this
   section with the measured values.

## Limits and risks

* No GPU result: the 3.5 bpw target, the clipping/search and the KL values are untested by a
  quantized model; do not quote a quality number from this set until a lease has run it.
* The agentic set is rendered from real tool schemas and prompts, but the model's own
  generations are not in it; it measures the read of tool turns, not multi-step success.
* The Linux rust text is GPL-2.0 (see above). The per-code-language held-out counts for shell,
  SQL and JavaScript are 0/0/0, so a claim about them is not supported.
* Long windows are 8192-32768 tokens; the checkpoint accepts 1M, so >32k quality is not
  measured.
* The positional buckets are per-window half-open ranges; a 2048-token set has only the first
  bucket, so a per-position claim for the old sets is not possible.
