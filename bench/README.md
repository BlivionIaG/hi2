# bench: task-level evaluation

KLD and perplexity say how far a quantized model's next-token distribution moves. This
suite says what that costs on real work: math, code, instruction following, tool calls,
long-context retrieval, multiple choice and agent decisions. It talks to any
OpenAI-compatible server (vLLM; GGUF through a server), so BF16, AWQ, GGUF and HI2
checkpoints are measured the same way, and it stores every item so that two runs can be
compared **paired**: the same items, the same greedy decoding, the flips counted one by one.

```bash
python bench/run.py --url http://127.0.0.1:18731 --model bf16 --workers 16 \
    --tasks all --out results/bf16.json
python bench/run.py --url http://127.0.0.1:18732 --model hi2-3.0 --workers 16 \
    --tasks all --out results/hi2-3.0.json
python bench/compare.py results/bf16.json results/bf16-rerun.json results/hi2-3.0.json \
    --groups --md results/hi2-3.0.md --json results/hi2-3.0.stats.json
```

A second reference run (same server, restarted) gives the spread that greedy decoding has
on the stack (TunableOp, accumulation order): differences inside that spread are noise.

## Tasks

| task | measures | items | `ok` means | details |
| --- | --- | --- | --- | --- |
| `gsm8k` | multi-step arithmetic | 1319 (test) | final number equals the reference | `pred`, `gold` |
| `humaneval` | function synthesis | 164 | the dataset's tests pass (pass@1) | `status`, `stderr` |
| `mbpp` | short programs | 257 (sanitized test) | the three asserts pass | `status`, `stderr` |
| `ifeval` | verifiable instruction following | 500 | every constraint holds (strict) | `frac`, per-constraint `checks` |
| `toolcall` | function name and arguments (ToolACE) | 500 of 9009 eligible | the same calls, order-free | `name_ok`, `arg_f1`, `parsed` |
| `needle` | retrieval at 2k/8k/16k/32k tokens | 160 (40 per length) | the hidden value is in the reply | `distractor` (a wrong key's value) |
| `mc` | ARC-E, ARC-C, HellaSwag, MMLU by log-likelihood | 1799 (500 per set; ARC-C has 299) | best answer by summed log-prob | `acc_norm`, `pick` |
| `decision` | coding-agent tool choice (spec 12.3) | 66 | a well-formed call, no garbage | `tool`, `arg`, `garbage`; agreement with the reference |

`code` is an alias for `humaneval,mbpp`, `all` for every task. `--limit N` caps every task
at N items (a seeded subset that keeps the dataset order; `mc` splits it over its four
sets; `ifeval` and `needle` keep a prefix that still covers every constraint and length).

Details per task:

- **gsm8k**: zero-shot chain of thought, one fixed prompt that asks for a last line "The
  answer is N."; the last such line wins, then `\boxed{}`, then the last number.
- **humaneval / mbpp**: the reply's fenced block (the one defining the entry point, else
  the first) is run with the tests in `sandbox.py`: a child `python -I -S` (standard
  library only, no `sitecustomize`) in an empty temporary directory, a minimal
  environment, CPU/memory/file-size rlimits, sockets, subprocesses and file removal
  disabled, the process group killed after 10 s. It guards
  against accidents, not attacks; run untrusted models in a container. All 164 canonical
  HumanEval solutions and all 257 MBPP reference solutions pass through this path.
- **ifeval**: 17 constraint types (word counts, no commas, lower/upper case, keywords,
  forbidden word, paragraphs with `***`, exact bullet count, end phrase, quotation marks,
  JSON, `<<title>>`, postscript, highlights, placeholders, `SECTION X`) on 40 writing
  tasks; half the prompts carry a second, compatible constraint. Generated from the seed,
  no external package; the verifiers are in `tasks/ifeval.py`.
- **toolcall**: the dataset's own system prompt (function list and the
  `[func(a=1, b="x")]` format) and the first user turn whose reference reply is a call
  list. Replies in that format, in JSON (`{"name", "arguments"}`) or in `<tool_call>`
  blocks are all read. Values compare after normalisation (strings case- and
  space-insensitive, numbers as floats). `arg_f1` is over (function, argument, value)
  triples, with one triple per call for the name.
- **needle**: the haystack is this repository's `README.md` and `docs/*.md`, cut at a
  tokenizer-free count that tracks the Qwen3 token count within about 8%. Half the items
  hide one access code, half hide four key-value pairs and ask for one. The haystack's
  hash is the task fingerprint: `compare.py` warns when the documents changed between
  two runs. The 32k items need `max_model_len` >= 33k.
- **mc**: the logic of `experiments/mc_eval.py`, through the server's `/tokenize` and
  `prompt_logprobs` (vLLM extensions), so no local tokenizer is needed.
- **decision**: `experiments/data/decision_prompts.json` scored with
  `experiments/decision_fidelity.py`. `compare.py` adds the agreement with the reference
  on the first tool and on tool plus key argument, with Wilson intervals.

Chat requests disable the template's thinking mode (`--thinking` keeps it); a
`<think>` block in a reply is stripped before scoring and kept in the stored completion.

## Output and comparison

`run.py` writes one JSON: `config` (model, URL, seed, and per task its version, item
count, `max_tokens` and fingerprint), `results[task][id]` = `{id, group, completion, ok,
details, usage, latency}` (or `error` for a failed request), and `timing` per task. The
file is rewritten every 20 s; rerunning the same command resumes, redoing only missing
and failed items. A changed task version, seed or fingerprint redoes that task.

`compare.py REF CAND...` pairs the items both runs completed and reports per task: each
accuracy, the difference with a paired bootstrap 95% CI (10000 replicates, exact for
binary outcomes through the discordant counts), the exact McNemar p, the flip rates
(reference right and candidate wrong, and the reverse, as shares of the items), the
recovery (candidate / reference), the means of the tasks' numeric details, and the paired
sample size that detects a 1-point difference at 80% power: about 78500 times the
discordance rate, so 8% of items flipping needs about 6300 items. The full suite has
about 4800 items; a 1-point claim on one task of a few hundred items is not testable,
pool tasks or report the CI. `--groups` adds per-group tables (constraint type, context
length, MC set, agent stage).

## Runtime

Estimates per 1000 requests, at an aggregate decode of 150 tok/s and prefill of
2000 tok/s with 16 workers (4x V620, a 27B W4A16 model, roughly the c=8 numbers in the
fork's notes). They scale with the server; `timing` in a result file has the measured
seconds.

| task | prompt / output tokens per request | per 1000 requests | full task |
| --- | --- | --- | --- |
| gsm8k | 200 / 250 | 30 min | 40 min |
| humaneval, mbpp | 200 / 200 | 25 min | 10 min together |
| ifeval | 100 / 350 | 40 min | 20 min |
| toolcall | 1000 / 40 | 12 min | 6 min |
| needle | 14000 / 10 | 2 h (prefill-bound) | 20 min |
| mc | 150 / 1 (prompt log-probs, ~4.5 requests per item) | 6 min | 10 min |
| decision | 2000 / 100 | 30 min | 2 min |

The whole suite is about 2 hours at these rates.

## Datasets

| task | source | license |
| --- | --- | --- |
| gsm8k | `openai/gsm8k` (main, test) | MIT |
| humaneval | `openai/openai_humaneval` (test) | MIT |
| mbpp | `google-research-datasets/mbpp` (sanitized, test) | CC BY 4.0 |
| toolcall | `Team-ACE/ToolACE` (train) | Apache-2.0 |
| mc | llama.cpp validation JSON (`ikawrakow/validation-datasets-for-llama.cpp`, converted with its `convert` tool) of ARC (CC BY-SA 4.0), HellaSwag (MIT) and MMLU (MIT) | as the sources |
| ifeval, needle, decision | generated here / this repository | Apache-2.0 |

The Hugging Face tasks load with `datasets.load_dataset` (`HI2_BENCH_CACHE` sets its
`cache_dir`; `HF_DATASETS_OFFLINE=1` uses the cache only). A client environment without
`datasets` reads jsonl exports instead: run `python -m bench.data export DIR` where
`datasets` works and set `HI2_BENCH_DATA=DIR`. On the build box the exports are in
`/home/chenco_adm/w4a8_runs/hadamard-int2/bench_data/export`. The MC files are found
through `HI2_BENCH_MC_DATA` (default: the build box's copy). A missing dataset skips that
task with a message that says how to get it; the other tasks still run.

Dependencies: the standard library, plus `datasets` for the Hugging Face loaders. The
statistics need nothing else.

## Adding a task

Write `bench/tasks/<name>.py` with `NAME`, `VERSION`, `MAX_TOKENS`, `DEFAULT_LIMIT` and

```python
def load(limit, seed) -> list[dict]: ...  # items with a unique "id", optional "group"
def prompt(item): ...  # message list (chat) or string (completion)
def score(item, completion) -> dict: ...  # {"ok": bool, ...details}
```

Optional: `DATASETS` (for `bench.data export`), `request(client, item)` to replace plain
generation, `fingerprint()`, `AGREE` (detail keys compared with the reference) and
`METRICS` (numeric details to average). Add the name to `NAMES` in
`bench/tasks/__init__.py` and a CPU test of the scoring in `tests/test_bench_*.py`. Bump
`VERSION` whenever the items, the prompt or the scoring change, so stale results are
not compared.
