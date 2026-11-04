# bench: task-level evaluation

KLD and perplexity say how far a quantized model's next-token distribution moves. This
suite says what that costs on real work: math, code, instruction following, tool calls,
long-context retrieval, multiple choice and agent decisions. It uses OpenAI-style
HTTP endpoints, with **additional server capabilities required for some tasks**.
BF16, AWQ, GGUF and HI2 checkpoints can be compared when their serving runtimes
implement the required APIs. Every item is retained for a **paired** comparison:
the same items, the same greedy decoding, the flips counted one by one.

## Installation: checkout tooling, not an installed command

`bench/` and `experiments/` are not included in the installed `hi2` package: package
discovery includes only `hi2*`. There is **no `hi2 bench` command**. Keep a complete
checkout, run the following from its root, and keep it available when using these
entrypoints:

```bash
git clone https://github.com/BlivionIaG/hi2.git
cd hi2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[bench]'
```

Python 3.10 or newer is required. The `bench` extra installs `datasets` in addition
to hi2's base dependencies; it does not install a model server or package `bench/`.
The client, comparison statistics and generated tasks themselves use the standard
library. A standard-library-only client can use prepared JSONL exports instead
of `datasets` (see [Datasets](#datasets)). `decision` also imports code and reads
prompts from this checkout's `experiments/`; `needle` reads the checkout's documents.
Installing hi2 into an unrelated directory does not provide these resources.

### Generated-code execution: read before using `code` or `all`

**`humaneval`, `mbpp`, the `code` alias and `all` execute model-generated Python
on the benchmark client's machine. `bench/sandbox.py` is not a security sandbox.**
Its isolated interpreter, temporary directory, timeouts, resource limits and
monkeypatched functions guard against mistakes, not malicious code. They do not
provide filesystem isolation or enforce a network security boundary, and some
resource limits are platform-dependent. Use a disposable, restricted container
or VM with no credentials, sensitive mounts or unnecessary network access.
The execution helper requires POSIX facilities (`resource`, process groups,
signals); do not assume the code tasks work on Windows.

The examples below avoid code execution. A compatible model server must already
be running, and the Hugging Face datasets must be available or exported locally.

```bash
python bench/run.py --url http://127.0.0.1:18731 --model bf16 --workers 16 \
    --tasks gsm8k,ifeval,toolcall --out results/bf16.json
python bench/run.py --url http://127.0.0.1:18732 --model hi2-3.0 --workers 16 \
    --tasks gsm8k,ifeval,toolcall --out results/hi2-3.0.json
python bench/compare.py results/bf16.json results/hi2-3.0.json \
    --groups --md results/hi2-3.0.md --json results/hi2-3.0.stats.json
```

A second reference run (same server, restarted) gives the spread that greedy decoding has
on the stack (TunableOp, accumulation order): differences inside that spread are noise.

### Required server APIs

Use the base URL, such as `http://127.0.0.1:18731`, **without** a trailing `/v1`.
The built-in client does not add authentication headers; use an appropriately
isolated endpoint or an authenticated proxy configured outside this client.

| entrypoint / task | required capabilities |
| --- | --- |
| chat tasks (`gsm8k`, `humaneval`, `mbpp`, `ifeval`, `toolcall`, `decision`) | `/v1/chat/completions`, greedy generation (`temperature=0`), text `choices[].message.content` |
| `needle` | `/v1/completions`, text `choices[].text`, sufficient context length |
| `mc` | `/tokenize` with text and `add_special_tokens=false`; `/v1/completions` accepting token-ID prompts and returning `prompt_logprobs` for **all** prompt tokens |
| `logprob_gap.py score` | all `mc` extensions, plus `/tokenize` accepting chat messages, `add_generation_prompt` and `chat_template_kwargs` |

The last two rows are vLLM-style extensions, **not guaranteed by an
OpenAI-compatible server**. A GGUF server without them cannot run those paths.
Missing endpoints, rejected requests and truncated log-probability responses
make the evaluation incomplete and nonzero; they are not scored as wrong answers.
Chat generation sends `chat_template_kwargs={"enable_thinking": false}` by default.
`--thinking` omits that override (useful when a server does not accept it), so the
server's template default applies. Record and match this setting across runs.

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
  disabled, the process group killed after 10 s. These are accident guards only;
  see the generated-code warning above before using either task.
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

`run.py` writes schema version 1 JSON with these fields:

- `status`: `running`, `complete` or `incomplete`.
- `completion.requested_tasks`: the full requested task list. No failed task
  disappears from this list.
- `completion.tasks[task]`: `status` (`pending`, `running`, `complete`, `incomplete` or
  `load_error`), `expected`, `completed`, `request_errors`, `score_errors`,
  `missing`, and an error message for a failed load/import. Unknown item counts
  are `null`, not zero.
- `completion.errors`: harness failures or interruption details.
- `config`: model, URL, decoding options, workers, seed and limits, plus each
  loaded task's version, effective limit, item count, dataset declarations,
  optional fingerprint and SHA256 of its selected item content (`items_sha256`).
- `results[task][id]`: raw completion, `ok`, score details, usage and client
  latency, or an error. Item `status` distinguishes `complete`, `request_error`
  and `score_error`; a completed **wrong** answer is still `complete`.
- `timing`: per-task request/scoring wall time and accumulated request latency;
  `_wall` is cumulative across attempts, not a native-kernel timing.
- `attempts`: command arguments when called via the CLI, start/end times and
  attempt status. Programmatic calls without a command record `null`, not an
  invented invocation.

The file is saved before requests, periodically during execution and on final
completion/failure. Partial results are retained. `run.py` exits **0 only when
every requested task/item completed**, 1 for an incomplete evaluation, 2 for
invalid CLI arguments, and 130 for a handled keyboard interrupt. Missing data,
failed imports, an empty task, request failures and scoring/harness exceptions
are incomplete. Zero accuracy by itself still exits 0: there is no implicit
accuracy gate in this harness. Quality acceptance is a separate decision.

Rerunning the same command resumes missing, request-failed and score-failed
items; completed wrong answers are not retried. A changed task version, seed,
fingerprint or selected-item content hash redoes the task. Changing model name,
URL or chat options refuses resume: use a new `--out`. A model name/URL is not a
verified weight identity; if a server changes weights behind the same name, use
a fresh result path and attach the correct manifest when comparing. Older
unrequested tasks may remain in the raw file; completion and comparison use the
current requested list.

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

The comparison includes the **union** of requested tasks, including missing tasks
and load failures. Request/scoring failures are excluded from paired accuracy,
not counted as low-quality answers. Each task has reference/candidate expected
and completed counts, its completion status, and the observed unpaired count.
Version, seed and data fingerprints are checked and differences are reported.
Statistics on a partial intersection are explicitly incomplete, not a claim
about the full requested dataset. Legacy files without enough count information
are `unverified`, never silently called complete. JSON uses `null` for undefined
statistics rather than nonstandard `NaN`.

`compare.py` exits 1 for an incomplete/mismatched comparison while still writing
the report. A legacy `unverified` comparison exits 0 for report generation, but
its explicit status is not evidence of complete execution. Low accuracy or a
recorded unmet quality target does not change the benchmark completion status.

### Reproducible comparisons using existing hi2 outputs

Use `--metadata FILE.json` to attach the **actual JSON outputs** from `hi2 size`,
`hi2 eval`/the quality service and a quantization `hi2_run.json`/`run.json`
manifest. This consumes saved outputs; it does not reimplement evaluation,
run models or launch recipe searches. Provide only outputs for the corresponding
benchmark artifact. Paths inside the mapping, including its run-path keys, are
relative to the metadata file's directory (absolute paths also work):

```json
{
  "bf16.json": {
    "runtime_context": {"kind": "native", "backend": "record-the-actual-server"}
  },
  "hi2-3.0.json": {
    "manifest": "hi2_run.json",
    "size": "hi2-3.0.size.json",
    "quality": "hi2-3.0.quality.json",
    "runtime_context": {"kind": "native", "backend": "record-the-actual-server"}
  }
}
```

`runtime_context` is an operator declaration, not a verified dispatch claim.
Its `kind` is `native`, `dense_numerical` or `unknown` (the default when omitted).
Include actual backend/hardware/version/evidence fields if known; do not label
a dense export as native HI2. Omit unavailable source files and measurements
rather than filling them with guesses.

```bash
python bench/compare.py results/bf16.json results/hi2-3.0.json \
    --metadata results/metadata.json \
    --md results/comparison.md --json results/comparison.json
```

The versioned comparison retains every input's resolved path, SHA256 and raw
JSON, the metadata mapping and comparison command, the benchmark's command
history, and manifest source/artifact identities, resolved recipe, calibration
identities and environment. Supplied quality dataset/artifact identities and
acceptance status are retained independently of task completion.

| measurement | provenance / interpretation |
| --- | --- |
| allocation bpw | `manifest.result.allocation_bpw`, **predicted allocation**, not measured stored bytes |
| stored quantized bpw | `size.linears.bpw`, measured quantized-linear bytes including storage metadata |
| whole-model tensor bytes | `size.tensor_bytes`, all tensor payloads |
| whole-model file bytes | `size.file_bytes`, weight shard files, not ancillary config/tokenizer files |
| quality | existing eval metrics, or `quality.metrics`; dense numerical context remains explicit |
| benchmark wall seconds | actual benchmark `timing._wall`, including client requests/scoring and resumes |
| peak memory / native-kernel time | **missing**: these inputs do not collect them |

Every measurement carries `status` (`measured`, `predicted` or `missing`), `value`,
`unit`, `source` and `scope`. Multi-output manifests with no single
`result.allocation_bpw` leave that measurement missing; the original outputs
remain available in raw JSON. A target bpw is not substituted for an allocation,
and neither is substituted for stored bpw.

Dense-dequantized quality results do **not** establish native loader correctness,
native kernel coverage, memory consumption or speed. This report computes no
native-performance ratio from dense numerical evaluation or client wall time.
Runtime context differences/unknowns are warnings. Compare a bounded set of
already-measured recipe outputs by listing candidate files; repeat inputs and
commands remain reviewable without rerunning a model.

### Teacher-forced scores and shell wrappers

Teacher-forced scoring measures log-probabilities for an existing completion.
It does not generate a new answer. Use a complete reference benchmark and a
server with the extended APIs listed above.

`python bench/logprob_gap.py score --url URL --model NAME --ref-results REF.json
--out LP.json` scores the reference completions.
It defaults to supported chat tasks requested by the reference; `--tasks`
requests an explicit list. No eligible tasks, missing requested reference tasks,
failed data loads, missing completion/token scores or request errors produce an
incomplete report and exit 1. Completed wrong reference answers can still be
scored. Outputs are versioned reports with `results`, `completion` and `status`;
`python bench/logprob_gap.py compare REF_LP.json CAND_LP.json` compares those reports
and reports unpaired items. Rescore older unversioned LP files for this interface.
The reference benchmark must explicitly have `status: complete`; legacy references
without that evidence remain incomplete even if individual token scores succeed.

The comparison field is `mean_logprob_gap` (nats/token), a signed, token-weighted
mean of `log p_ref - log p_candidate` on the scored completions. It is **not KL
divergence**: greedy reference outputs are not samples from the reference
distribution, and a candidate can assign them higher probability, giving a
negative gap. Do not compare this quantity with the full-distribution KL measured
by `hi2 quality`. Matching item IDs and lengths do not prove matching token IDs;
use identical tokenizers and chat templates. This scorer currently forces
`enable_thinking=false`, so reference generation must use that same setting.

`bench/serve_run.sh TAG=MODEL_DIR ...` needs the external fork's
`SERVE_VALIDATE=/path/to/serve_validate.sh`, `GPU` and a compatible server
environment; that driver is not shipped or installed by hi2. Set `PY` to the
client interpreter and `ROOT` to the results directory. `TASKS`, `LIMIT`,
`WORKERS` and `EXTRA` are passed through to the task suite. The hook also runs
teacher-forced scoring by default; `TF_LIMIT` and optional `TF_TASKS` control it.
Set **`TF_LOGPROB_GAP=0`** to omit that stage (for example for `needle,mc`
without chat tasks, or a server lacking teacher-forced APIs).

`hook.sh` prints `PASS`/`FAIL` **and returns the failing stage's nonzero exit
code**. `serve_run.sh` preserves server/hook failure even if the external driver
swallows the hook exit, using an invocation-specific completion marker. It
attempts subsequent arms, copies available partial reports and exits nonzero if
any arm failed. Missing completion markers are failures; a stale previous run
cannot certify a new invocation. Do not use the existence of JSON or a low
accuracy score as a replacement for these status fields and exit codes.

## Runtime

**Predictions, not measurements:** estimates per 1000 requests at an aggregate
decode of 150 tok/s and prefill of 2000 tok/s with 16 workers (4x V620, a 27B W4A16
model, based on rates in the fork's notes). They scale with the server; they are
not populated as measured values in comparison reports. `timing` in an actual
result file records the client-side seconds for that run.

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
`datasets` works and set `HI2_BENCH_DATA=DIR`. The MC files are found through
`HI2_BENCH_MC_DATA`, the directory with the llama.cpp validation JSON files
(`ikawrakow/validation-datasets-for-llama.cpp`, converted with its `convert` tool). A
missing dataset records that task as `load_error` with instructions for obtaining
it; other tasks still run and their partial results are saved, but the overall
run is **incomplete and exits nonzero**. `all` therefore requires MC data and
enables generated-code execution as well as the Hugging Face and local tasks.

For offline use, prepare exports where downloading is permitted, then copy them
to the client:

```bash
python -m bench.data export prepared-data --tasks gsm8k,toolcall
HI2_BENCH_DATA=prepared-data HF_DATASETS_OFFLINE=1 \
    python bench/run.py --url http://127.0.0.1:18731 --model NAME \
    --tasks gsm8k,toolcall --out results/offline.json
```

Exports are preferred when present. If an export is absent, the loader falls
back to `datasets`; `HF_DATASETS_OFFLINE=1` prevents that fallback from
downloading but still requires an existing datasets cache. Record the selected
item hash in the report when comparing runs; dataset names alone are not
immutable revision identities. The MC JSON files are external prerequisites
and are not generated by `bench.data export`.

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
