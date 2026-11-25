# Task-level evaluation of served Inkling checkpoints

Status: **exploratory tooling**, built for the HI2 3.5 bpw Inkling-2-Mini
comparison. It measures task behaviour, not kernels: use it next to the KLD
report and the runtime's own smoke tests, not instead of them.

`tools/inkling_eval/` asks what the quantized checkpoint does to real work:
agentic tool calls, executable code, 14 languages, long-context retrieval and
reasoning. Stock and quantized endpoints get the **same rendered prompt, greedy
decoding and the same scorer**, and the comparison is paired item by item.

## What is measured

| category | items | sub-tasks | `ok` means |
| --- | ---: | --- | --- |
| `agentic` | 60 | 12 multi-turn tool scenarios × 5 value variants | one schema-valid Inkling tool call with the expected function name and exactly the expected arguments |
| `code` | 40 | HumanEval 20, MBPP 10, C 5, TypeScript 5 | the extracted code compiles/runs and the hidden unit tests exit 0 after printing `OK` |
| `multilingual` | 70 | 14 languages × (2 translations, 2 numeric questions, 1 colour question) | translation: chrF2 ≥ 0.20 and the answer is not in a contradicting language; QA: exact numeric/word match |
| `long_context` | 27 | per length 8k/16k/32k: 5 single-needle positions, 3 multi-needle positions, 1 summarize-then-answer | the asked access code is present (multi-needle also reports distractor hits); summarize: ≤ 60-word summary and exact owner |
| `math` | 48 | GSM8K 40, authored 8 | the final number equals the reference |

Total **245 items**. Prompt tokens are estimated with the same tokenizer-free
piece counter the bench `needle` task uses (one piece per 1–6 Latin letters,
digit or punctuation mark); the 32k items render at about 29.4k pieces, so a
server needs `max_model_len >= 32768` for them.
The harness probes `/v1/models`; a length that does not fit is skipped and
recorded in the run meta, never silently scored.

### Answer channels, budgets and truncation

The live run on 2026-11-25 exposed two measurement bugs. First, vLLM's
`/v1/completions` defaults `skip_special_tokens=true`, which stripped
`<|content_thinking|>`, `<|content_text|>` and `<|end_message|>`; the reasoning
preamble was then concatenated with the answer (`... should be "Paris".Paris`).
The client now sends `skip_special_tokens=false` and records it in the meta.
`answer_text` keeps the **last** `<|content_text|>` channel and drops thinking;
`final_segment` takes the last sentence of that channel's last non-empty line.
A completion without markers (older runs or a server that strips anyway) falls
back to the cleaned whole text, so extraction is still defined.

Second, the budgets assumed the model answered immediately. The model writes a
reasoning preamble even at thinking effort 0, so many tasks hit `max_tokens`.
Budgets now cover the preamble plus the answer. They were resized after the
v2 run (stock, stock2 and hi2v2, 245 items each) using the `p99` completion
length of items that stopped before their v2 budget:

| task | budget | v2 evidence |
| --- | ---: | --- |
| agentic | 768 | p99 360; 1/60 truncated at 384 in every label |
| code (Py/C/TS) | 2048 | p99 811; 5/32 stock, 6/32 hi2 truncated at 1024 |
| translation | 384 | p99 156; no truncation |
| qa_numeric / qa_word | 512 | p99 225; no truncation |
| single/multi needle | 2048 | p99 1011 at the old 1024 cap; no truncation but 13 tokens of headroom |
| summarize-then-answer | 2048 | parent of the same long-context budget |
| math | 1536 | p99 720; 1/48 stock truncated at 1024 |

The rule is `ceil(2 * p99 / 128) * 128`, except that code gets 2048: the five
truncated items were still reasoning (the largest complete answer was 958
tokens) and the failure reproduced on both stock runs. Do not lower a budget
below a proven p99 without checking the truncation gate.

`finish_reason == "length"` is now its own record status, `truncated`: it is
excluded from accuracy and from `wrong`, retried on resume, and reported per
category. **More than 5% truncated items in any category makes a run
unreliable**: `run` exits 1 even when every item returned, and `compare` sets
`status: truncated`, prints the failed categories and refuses to present the
deltas as interpretable. The pair set excludes truncated items.

The three code languages are real executions: Python under `python -I -S`, C
compiled with the system `cc`/`gcc`, TypeScript run by `node` (22.6+ native type
stripping), `bun`, `deno`, `qjs`, or the Python `quickjs` module. The latter two
use the harness's own `strip_typescript`, which covers function declarations,
parameter/return/variable annotations, generics, interfaces, type aliases,
optional parameters and `as` casts (the tasks forbid classes, enums and
namespaces). Every child gets an empty directory, a minimal environment, CPU/
memory/file-size limits, a timeout and a killed process group; on Linux it is
also placed in an empty network namespace when the kernel permits. **This is a
guard against mistakes, not a security boundary** — run untrusted output in a
container as well. The scratch directory is `~/.cache/inkling-eval/tmp`
(override with `INKLING_EVAL_SCRATCH`), never the box's small `/tmp`.

## Determinism and recording

* The harness renders the **Inkling chat template itself** (`chat_template.py`,
  a transcription of the model's `chat_template.jinja`) and sends the raw prompt
  to `/v1/completions`. Server-side template versions cannot change the prompt,
  and `tests/test_inkling_eval_template.py` checks the transcription against the
  real Jinja template byte for byte when one is available.
* `temperature=0`, `top_p=1`, `n=1`, `seed=0`, `reasoning_effort=0.0`,
  `skip_special_tokens=false`. Thinking effort 0 does not stop the model from
  writing a reasoning preamble; the channel-aware extraction handles it. Pass
  the same `--reasoning-effort` to both endpoints; it is recorded in the meta
  and compared.
* Every item is one JSONL line: rendered prompt, prompt SHA-256, completion,
  finish reason, usage, latency, score and score detail. A torn final line is
  repaired on the next run.
* Rerunning the same command resumes: complete items with the same prompt hash
  are skipped; missing, errored, changed or truncated items are redone. A
  different label, URL, plan, seed, template, haystack or decoding setting
  refuses to resume (`--force` overrides).
* Exit codes: `0` only when every requested item completed with no category over
  the 5% truncation gate; `1` for an incomplete/unreliable run or comparison;
  `2` for invalid arguments or a resume mismatch. A wrong answer is a completed
  measurement and still exits `0`.

The model's language-ID guard is a documented heuristic: script share for
Russian, Chinese, Japanese, Korean, Arabic and Hindi; function-word hits or
characteristic diacritics for the eight Latin-script languages. It returns
`undetermined` for English text (chrF then rejects the translation) and
`mismatch` only on positive evidence of another language.

## Report and minimum detectable difference

`compare` pairs the non-truncated items both runs completed and reports, per
category and per task: truncation counts, accuracy of both, the delta, a paired
percentile bootstrap 95% CI (10,000 replicates over the discordant counts), the
exact McNemar p, the flip counts and the minimum detectable difference (MDD) at
80% power and α = 0.05. It also recomputes the MDD from the observed
discordance once a real run exists.

The item counts support these differences at 80% power and α = 0.05 with the
report's default 10% discordance assumption (`mdd_assumed_0.10`); `mdd`
recomputes the same bound at the observed discordance and is `null` when the
observed agreement cannot support any bound. Values sit on the 0.5-percentage-
point grid used by `stats.mdd`:

| category | items | MDD |
| --- | ---: | ---: |
| agentic | 60 | 13.0 points |
| code | 40 | 18.5 points |
| multilingual | 70 | 11.0 points |
| long_context | 27 | 27.0 points |
| math | 48 | 16.0 points |
| all items pooled | 245 | 6.0 points |

This table is the **best case, with zero truncated items and the full n**. A
truncated item leaves the paired denominator, so the realized MDD is the
report's `mdd` at the reduced n; a run that fails the truncation gate has no
interpretable MDE at all. Re-check the MDD after a clean rerun, not from a
truncated run.

Per-task claims inside a category are weaker: the agentic scenarios and the
multilingual languages have 5 and 5 items each, so a per-task claim needs several
tasks pooled. Pooling is `all items`; report the category numbers.

`--floor` compares the reference run with a **second stock run** (restart the
stock server or rerun it). Its per-category deltas and flips are the
determinism floor of this stack; a quantized delta inside that floor is not
evidence of a change.

## Agentic probe: reference-to-an-earlier-call

The main plan has five hotel items.  The v2 run exposed a possible
quantization-induced mode flip there: stock made the expected `book_hotel` call
in 4/5 (stock2 identical, floor 0), while hi2 reasoned to the correct arguments
and then emitted a plain `<|content_text|>` answer.  Five items cannot support a
claim, so `--set agentic-probe` adds a **separate** set without touching the 245
plan:

* 80 tool-call items: eight families (`book_hotel`, `rent_car`,
  `schedule_meeting`, `order_food`, `transfer_money`, `create_invoice`,
  `search_flights`, `send_email`) times ten value variants.  Each has 1–3
  earlier calls and asks the model to reuse a value (date, nights, amount,
  route, participants, ...) from one of them.
* 20 plain-chat controls: the same kind of prior call, then a question that
  expects **no tool call**.  The control is `ok` only when the completion has
  no invoke marker or parsed call and contains the asked value, which catches
  the opposite error (inventing an action).

The probe uses the same `agentic` scorer and records its own
`set: agentic-probe` in the plan and run meta; a main-vs-probe comparison is
rejected as mismatched.  Its plan hash is
`284e42e5cf0cf481db396bac1ce39e1007d6cb69ccf1b4eaadefbbb14c5bf9a4`
and its prompt-set hash is
`f3b38f39a8566c9c62e7d2fee0cd162be11660b91f99958a6ba073a1884ae7d2`.
At 100 items and the report's 10% discordance assumption the **MDD is 9.0
percentage points**; the observed hotel discordance (4/5 stock-right → hi2-wrong)
is 80 points, but that is a signal to test, not an estimate.  The eight families
have ten items each, so their per-family MDD is undefined: pool the category.

```bash
$EVAL plan --set agentic-probe
$EVAL dry-run --set agentic-probe   # 100 items, controls included, nothing measured
$EVAL run --set agentic-probe --url http://127.0.0.1:18731 --label probe-stock  --out runs/probe-stock.jsonl  --workers 4
$EVAL run --set agentic-probe --url http://127.0.0.1:18732 --label probe-hi2    --out runs/probe-hi2.jsonl    --workers 4
$EVAL run --set agentic-probe --url http://127.0.0.1:18731 --label probe-stock2 --out runs/probe-stock2.jsonl --workers 4
$EVAL compare --ref runs/probe-stock.jsonl --cand runs/probe-hi2.jsonl \
    --floor runs/probe-stock2.jsonl --out runs/cmp-probe.json --md runs/cmp-probe.md
```

The probe prompts are short (maximum about 1.4k estimated tokens).  Expected
runtime at `--workers 4` is roughly 10–20 minutes per label **[INFERENCE]**.

## Vision set: synthetic images, judged without a model

The `--set vision` plan is separate from both the 245-item main plan and the
agentic probe.  It has **88 items**: 24 OCR-style codes/numbers, 20 bar/line
charts, 20 shape/colour counts, 16 table row reads and 8 text-only controls.
All 80 images are drawn at plan-build time by the pure-Python canvas/PNG writer
in `vision_data.py`; there are no photographs, downloads or external datasets.
The 5x7 bitmap font is embedded, so the PNG bytes and plan hash do not depend on
Pillow, a font package or the host OS.

The set uses the OpenAI multimodal chat shape:

```json
{"role": "user", "content": [
  {"type": "text", "text": "Read the number ..."},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
]}
```

It is sent to `/v1/chat/completions` (the completions transport cannot carry
image parts).  The Inkling template renders the image part as
`<|content_image|><|unused_200054|>` and the Inkling-2 processor expands the pad
to one token per 40x40 cell; a 480x240 image is 72 vision tokens.  The request
sets `reasoning_effort: "none"` (the server's Inkling renderer maps that to
numeric `0.0`), `skip_special_tokens: false`, `temperature: 0`, `top_p: 1`,
`n: 1`, `seed: 0`; the client records `transport: /v1/chat/completions` and
`thinking_effort: none` in the run meta.  The main plan still renders the
template locally and uses `/v1/completions`; the two sets cannot be compared
with each other.

Scorers are exact or bounded, never a judge: OCR/table strings use the same
normalised word match as `qa_word`, counts use `qa_numeric`, and chart values
use `vision_chart` with an absolute tolerance of 2 units.  The text-only
controls carry no image and catch a model that starts calling tools or inventing
visual evidence when none exists.

Plan identity: `plan_sha256` is
`9aef5e57c07f4050ea7310e006ed2f6e4847cf75c5f915b2c4c1638614ec1396` and
`prompt_set_sha256` is
`47a3587e3b3ab3715aa10916448cd179c17d349aa7ac554b515aeb3b2d6bb288`.
At n=88 and the report's 10% discordance assumption the **MDD is 9.5
percentage points**.  The per-family item counts (8–24) do not support a
per-family MDE; pool the `vision` category.  A run with more than 5% truncated
items follows the same rule as every other category: excluded from accuracy and
reported `UNRELIABLE`.

The server must load the vision tower for this session.  The runtime kit's
`serve.sh` enables it with `VISION=1` (which drops `--language-model-only` and
sets `--limit-mm-per-prompt '{"image":1}'`); use that for both the stock and HI2
endpoints.  Every item here carries exactly one image, matching the one-image
cap.  The served directory must be `prep_serve_dir.sh` output so
`processor_config.json` and the vision shard are present.  A text-only boot will
reject or ignore the image parts.

```bash
$EVAL plan --set vision
$EVAL dry-run --set vision          # 88 items through the image-aware mock chat server
$EVAL run --set vision --url http://127.0.0.1:18731 --label vision-stock  --out runs/vision-stock.jsonl  --workers 4
$EVAL run --set vision --url http://127.0.0.1:18732 --label vision-hi2    --out runs/vision-hi2.jsonl    --workers 4
$EVAL run --set vision --url http://127.0.0.1:18731 --label vision-stock2 --out runs/vision-stock2.jsonl --workers 4
$EVAL compare --ref runs/vision-stock.jsonl --cand runs/vision-hi2.jsonl \
    --floor runs/vision-stock2.jsonl --out runs/cmp-vision.json --md runs/cmp-vision.md
```

The images are small but the chat endpoint may be slower than the text
completions path; measure the first probe before committing the full session.
The set is deliberately short enough to run inside the same 2–3 hour budget as
the main plan **[INFERENCE]**.

## Running on the GPU box

The client and every scorer are CPU-only. Do not export GPU visibility for them:

```bash
git -C ~/hi2 fetch origin feat/inkling2-eval
git -C ~/hi2 checkout feat/inkling2-eval
cd ~/hi2
export EVAL="env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= python3 -m tools.inkling_eval.cli"
```

`plan` and the dry-run validate the checkout without a model:

```bash
$EVAL plan                      # counts, contexts, plan hash
$EVAL dry-run                   # 245 items against a canned server, nothing measured
python3 -m pytest -q tests/test_inkling_eval_*.py
python3 -m tools.inkling_eval.mutants
```

The box already has `quickjs` installed user-local (2026-11-24) for the
TypeScript scorer; `gcc`/`cc` covers C. No `node`, `tsc`, `datasets`, `requests`
or other package is needed at evaluation time.

### Exact session commands

Two slots (stock on 18731, quantized on 18732), from the eval checkout:

```bash
$EVAL run --url http://127.0.0.1:18731 --label stock-v2   --out runs/eval-stock-v2.jsonl  --workers 4
$EVAL run --url http://127.0.0.1:18732 --label hi2v2      --out runs/eval-hi2v2.jsonl     --workers 4
$EVAL run --url http://127.0.0.1:18731 --label stock2-v2  --out runs/eval-stock2-v2.jsonl --workers 4
$EVAL compare --ref runs/eval-stock-v2.jsonl --cand runs/eval-hi2v2.jsonl \
    --floor runs/eval-stock2-v2.jsonl \
    --out runs/cmp-hi2v2.json --md runs/cmp-hi2v2.md
```

One slot at a time: run stock, stop the stock server, serve the quantized
checkpoint on the same port and run with `--label hi2v2 --out
runs/eval-hi2v2.jsonl`. Then restart stock for the floor run. Use a new `--out`
per label; the harness refuses to mix configurations in one file. The 2026-11-25
files (`eval-stock.jsonl`, `eval-hi2v1.jsonl`, ...) used the old stripping and
budgets; they are historical and `compare` rejects them on the decoding meta.

Plan identity: `plan_sha256` is
`2ff92e64b0d639e9e3a467539e190965ee80492c13a15b6d7248ea09f0f5babc`;
`prompt_set_sha256` is
`bcafe0f05bec3262e312b3c110d6900ab9a0f95b936838aaa91d018cfd9274c8`, exactly the
old plan hash. The 245 ids, prompts and expected values are unchanged; the new
plan hash differs only because `max_tokens` is now part of the item hash.

Serve the model with at least 32,768 tokens of context
(`vllm ... --max-model-len 32768`) so the 32k items fit; the harness skips and
records longer contexts it cannot reach. Keep `--reasoning-effort 0.0` on both
labels. Concurrency `--workers 4` is the intended ceiling: the served model is
small-context, and more parallel long prompts only queue.

Expected runtime **[INFERENCE]**: about 20–50 minutes per label at `--workers 4`
with the new budgets (the previous HI2 run took 12 minutes; code can now decode
up to 2048, math 1536, agentic 768 and the 27 long items 2048 tokens). Three
labels plus comparison fit inside 2–3 hours; watch the `truncated` counts in the
log and the gate in the summary. Runs are resumable, so a lease end only costs
the unfinished items.
`usage.prompt_tokens` in the JSONL is the real token count; use it to sanity
check the estimate.

`run` writes `runs/eval-<label>.jsonl` and `runs/eval-<label>.jsonl.meta.json`.
Do not edit either between paired runs; `compare` refuses to call mismatched
meta comparable.

## Reading the report

`compare` prints and writes a short table; the JSON has the same numbers plus
per-task rows, the per-category truncation gate and the full checks. `status` is
`complete` only when both runs accounted for every item (complete or truncated),
no category exceeded 5% truncation and the metadata matches. `status: truncated` means the run is
**unreliable**: the table still shows what was measured, but the warning says
not to interpret it. `pairing` reports items present in only one run and the
truncated totals. Undefined statistics are `null`, never `NaN`.

A useful result has: `status: complete` (so every item is accounted for and
truncation is at most 5% in every category), a floor run with small deltas, and a quantized delta whose paired CI
excludes the floor. The report itself does not
gate quality; set acceptance separately.

## Data and licences

See `tools/inkling_eval/data/SOURCES.md`: GSM8K (MIT), MBPP (CC-BY-4.0),
HumanEval (MIT), authored C/TypeScript/math/multilingual tasks (Apache-2.0) and
this repository's markdown as the frozen long-context haystack (Apache-2.0).
The 100 agentic-probe items are generated deterministically from the constants
in `agentic_probe.py`; they use no external data and are Apache-2.0.  The 88
vision items and all of their PNGs are generated by `vision_data.py`; they are
synthetic, use no photographs or external datasets, and are Apache-2.0.
Vendored data is about 640 kB; no network is used at evaluation time.
