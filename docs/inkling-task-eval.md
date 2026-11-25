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
Budgets now cover the preamble plus the answer. They were sized from the 735
completed items of the stock/stock2/hi2-v1 runs (`p99` of completion tokens for
items that stopped before their old budget; censored tasks marked *):

| task | budget | evidence |
| --- | ---: | --- |
| agentic | 384 | p99 189; 6/180 truncated at 200 |
| code (Py/C/TS) | 1024 | p99 506; 83/120 truncated at 512 |
| translation | 384 | p99 156; 7/84 truncated at 160 |
| qa_numeric | 512 | * 0/84 stopped at 32 |
| qa_word | 512 | * 0/42 stopped at 24 |
| single/multi needle | 1024 | * 0/72 stopped at 48 |
| summarize-then-answer | 1024 | * only 4/9 stopped at 320 |
| math | 1024 | p99 464; 14/144 truncated at 512 |

The rule is `min(1024, ceil(2 * p99 / 128) * 128)`; censored short-answer QA
gets 512, censored long-context tasks get the 1024 cap. A later
`max_tokens=1024` probe can tighten this, but do not lower a budget below a
proven p99 without checking the truncation gate.

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
`48c05915e7c0c6ad56d81cefc70ce712dde07f2def674efd73826d35ee652eb7`;
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
with the new budgets (the previous HI2 run took 12 minutes; code, math and the
27 long items can now decode up to 1024 tokens instead of 48–512). Three labels
plus comparison fit inside 2–3 hours; watch the `truncated` counts in the log and
the gate in the summary. Runs are resumable, so a lease end only costs the unfinished items.
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
Vendored data is about 640 kB; no network is used at evaluation time.
