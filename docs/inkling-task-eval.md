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
* `temperature=0`, `top_p=1`, `n=1`, `seed=0`, `reasoning_effort=0.0` (thinking
  off). Pass the same `--reasoning-effort` to both endpoints; it is recorded in
  the meta and compared.
* Every item is one JSONL line: rendered prompt, prompt SHA-256, completion,
  finish reason, usage, latency, score and score detail. A torn final line is
  repaired on the next run.
* Rerunning the same command resumes: complete items with the same prompt hash
  are skipped; missing, errored or changed items are redone. A different label,
  URL, plan, seed, template or haystack refuses to resume (`--force` overrides).
* Exit codes: `0` only when every requested item completed; `1` for an incomplete
  run/comparison; `2` for invalid arguments or a resume mismatch. A wrong answer
  is a completed measurement and still exits `0`.

The model's language-ID guard is a documented heuristic: script share for
Russian, Chinese, Japanese, Korean, Arabic and Hindi; function-word hits or
characteristic diacritics for the eight Latin-script languages. It returns
`undetermined` for English text (chrF then rejects the translation) and
`mismatch` only on positive evidence of another language.

## Report and minimum detectable difference

`compare` pairs the items both runs completed and reports, per category and per
task: accuracy of both, the delta, a paired percentile bootstrap 95% CI (10,000
replicates over the discordant counts), the exact McNemar p, the flip counts and
the minimum detectable difference (MDD) at 80% power and α = 0.05. It also
recomputes the MDD from the observed discordance once a real run exists.

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
$EVAL run --url http://127.0.0.1:18731 --label stock       --out runs/eval-stock.jsonl  --workers 4
$EVAL run --url http://127.0.0.1:18732 --label hi2-3.5bpw --out runs/eval-hi2.jsonl    --workers 4
$EVAL run --url http://127.0.0.1:18731 --label stock2      --out runs/eval-stock2.jsonl --workers 4
$EVAL compare --ref runs/eval-stock.jsonl --cand runs/eval-hi2.jsonl \
    --floor runs/eval-stock2.jsonl \
    --out runs/eval-compare.json --md runs/eval-compare.md
```

One slot at a time: run stock, stop the stock server, serve the quantized
checkpoint on the same port and run with `--label hi2-3.5bpw --out
runs/eval-hi2.jsonl`. Then restart stock for the floor run. Use a new `--out`
per label; the harness refuses to mix configurations in one file.

Serve the model with at least 32,768 tokens of context
(`vllm ... --max-model-len 32768`) so the 32k items fit; the harness skips and
records longer contexts it cannot reach. Keep `--reasoning-effort 0.0` on both
labels. Concurrency `--workers 4` is the intended ceiling: the served model is
small-context, and more parallel long prompts only queue.

Expected runtime **[INFERENCE]**: about 15–40 minutes per label at `--workers 4`
(218 short items dominate the count; the 27 long items dominate the time at
8k–30k prompt tokens each). Three labels plus comparison fit comfortably inside
2–3 hours. Runs are resumable, so a lease end only costs the unfinished items.
`usage.prompt_tokens` in the JSONL is the real token count; use it to sanity
check the estimate.

`run` writes `runs/eval-<label>.jsonl` and `runs/eval-<label>.jsonl.meta.json`.
Do not edit either between paired runs; `compare` refuses to call mismatched
meta comparable.

## Reading the report

`compare` prints and writes a short table; the JSON has the same numbers plus
per-task rows and the full checks. `status` is `complete` only when both runs
completed their whole plan and the metadata matches. `pairing` reports items
present in only one run. Undefined statistics are `null`, never `NaN`.

A useful result has: `status: complete`, a floor run with small deltas, and a
quantized delta whose paired CI excludes the floor. The report itself does not
gate quality; set acceptance separately.

## Data and licences

See `tools/inkling_eval/data/SOURCES.md`: GSM8K (MIT), MBPP (CC-BY-4.0),
HumanEval (MIT), authored C/TypeScript/math/multilingual tasks (Apache-2.0) and
this repository's markdown as the frozen long-context haystack (Apache-2.0).
Vendored data is about 640 kB; no network is used at evaluation time.
