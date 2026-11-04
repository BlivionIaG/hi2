# Using hi2

This page describes CLI options and the shared recipe/job contracts. Start with
[installation](installation.md) and the [first-model workflow](getting-started.md);
for the reasons behind the defaults, read [results.md](results.md) and
[design-notes.md](design-notes.md). Use `hi2 --help` to discover workflows and
`hi2 COMMAND --help` for every accepted flag. `python -m hi2` is the same entry point.

Terms used below: **PTQ** is post-training quantization. **QAD** is
quantization-aware distillation, which trains the quantized model after PTQ
initialization. **KLD** is Kullback–Leibler divergence between reference and
quantized token distributions. **bpw** means bits per weight.

## The simplest run

```bash
hi2 quantize MODEL OUT --calib calib.txt --target-bpw 3.0
```

The command requests three stages:

1. PTQ initialization.
2. QAD with the `default` preset: 300 steps per layer and 256 training sequences.
3. A 200-step final logit-KL training phase. The fit check can skip this phase.

This is the historical 27B recipe in [qad.md](qad.md), not a guarantee of final
quality. The other defaults are `--devices auto` (every visible GPU),
`--workdir OUT.work`, sensitivity probing with `--sens-dtype auto` (float32,
or float16 when float32 does not fit), smoothing 0.5, and rotation.
Omit `--target-bpw` to select the bit rate with `--max-error-ratio 6`.
Use `--no-qad` for PTQ only. Add `--dry-run` to inspect the plan before execution.

## Commands

| Command | What it does |
| --- | --- |
| `quantize SRC DST` | Convert a Hugging Face checkpoint to a rdna_hint_flex checkpoint. |
| `verify SRC DST` | Sample numerical weight comparisons; report checked/expected coverage. Not complete structural or native-runtime qualification. |
| `validate DST` | Scan all indexed shards, logical tensors, representation fields and report coverage. |
| `eval SRC DST --text T --skip-tokens N` | Measure dense-model PPL, KLD, top-1 agreement and EAR on the selected corpus. EAR is distribution overlap, not measured decode throughput; `lossless` means only KLD < 0.01 on those tokens. |
| `quality SRC DST --text T --max-kld K --max-ppl-ratio R` | Measure the final artifact, persist `hi2_quality.json`, and reject unmet acceptance thresholds. |
| `export-dense SRC DST OUT` | Write a plain fp16 Hugging Face model, not GGUF. llama.cpp needs a separate supported conversion. |
| `export-portable DST OUT --assets-from MODEL` | Copy HI2 weights without re-encoding and materialize trusted tokenizer/config assets with checksums. |
| `restack CKPT OUT --layout stacked\|per-expert` | Convert the expert layout of an MoE checkpoint without re-encoding. |
| `calib OUT SRC[:WEIGHT] ... [--tokens N] [--holdout PATH]` | Build a mixed-domain calibration text. See "Calibration text". |
| `size DST [--src SRC] [--by kind\|layer\|tensor] [--top N] [--json]` | Show where the bytes of a checkpoint go. See "Checking the size". |
| `cache list\|size\|prune` | Manage the persistent cache. |
| `doctor` | Inspect dependency/backend availability without a model or kernel execution. |
| `inspect PATH` | Read model metadata; for HI2, also run structural validation and size accounting. |
| `status WORKDIR` | Read the run manifest and recovery guidance without loading model tensors. |
| `recipe save\|run` | Save a resolved plan or replay a versioned recipe with explicit overrides. |
| `jobs submit\|status\|cancel` | Control durable local jobs; see below and [MCP setup](mcp.md). |

`eval` dequantizes the weights and runs the original architecture. It measures
dense numerical quality, not structural validity or native-kernel behavior.

The usual sequence is `doctor` → `inspect MODEL` → `quantize ... --dry-run` →
`quantize ...` → `validate OUT` → `quality ...` → `export-portable ...`.
These stages answer different questions: dependency visibility, metadata eligibility,
resolved planning, conversion, structural coverage, held-out numerical acceptance,
and distribution. None substitutes for native-runtime qualification. Use
[`status WORKDIR`](#cache-and-work-directory) for core-run recovery,
[`jobs status JOB_ID`](#durable-local-jobs) for detached work, and
[`recipe save/run`](#automation-errors-and-recovery) for replayable settings.

## Calibration text

Calibration needs `--calib-samples` sequences of `--seq-len` tokens. QAD needs more: 264
sequences of 2048 tokens, which is 540 672 tokens. Use text from the domain of your model.

The domain mix matters. A model calibrated only on documentation text had about 3 times the
KLD on chat and tool-calling text. PTQ reads the first tokens of the file and QAD draws
windows from all of it, so the domains must be mixed through the whole file. A file that
holds one domain after the other fails that rule. `hi2 calib` builds a file that follows it:

```bash
hi2 calib calib.txt docs.txt:2 chat.jsonl:1 tools.json:1 --holdout heldout.txt \
    --tokenizer Qwen3.5-4B
```

* A source is a `.txt` file (blocks separated by blank lines), or a `.json` or `.jsonl` file
  whose records have `text`, or chat turns in `messages` or `conversations`.
* `:WEIGHT` sets the share of each source (default 1). The tool interleaves the documents by
  token share. This mixes domains through the file; it does not guarantee exact
  proportions or every domain in each window.
* `--tokens` sets the size (default 540 672). The tool warns when the sources are smaller.
* `--tokenizer` counts real tokens. Without it, 4 characters count as one token.
* `--holdout` writes a second text from documents that the calibration text does not contain
  (20 % of each source; `--holdout-share` changes this), with the same mix, up to
  `--holdout-tokens` (default 200 000). Pass it to `hi2 eval --text`.
* `--min-holdout-tokens` rejects an undersized split (default 1). Set an actual
  evaluation budget, such as `samples * seq_len + skip_tokens`; one token is not
  enough for useful evaluation.
* Chat rendering is opt-in: `--chat-template` uses the local tokenizer's template,
  or `--chat-template-file FILE` supplies one. Both require `--tokenizer MODEL`;
  plain-text records are unchanged. Direct `quantize --calib` accepts text or
  text-field JSONL, not raw chat records; prepare those with `calib` first.

`experiments/build_calib_mix.py` is the script that built the study texts from datasets in the
Hugging Face cache.

Final evaluation must be independent of the entire QAD and tuning inputs, not just the
PTQ prefix. Use a separate held-out corpus with `--skip-tokens 0`. Prefix skipping is
appropriate only when no training/tuning stage has sampled the skipped file's remainder.
See [evaluation.md](evaluation.md) for all four data roles.

Without `--calib`, the tool runs data-free: all column weights are 1, rounding is
nearest, and the model is not loaded. QAD is skipped with a warning.

## Where it runs

| Goal | Flags |
| --- | --- |
| Automatic (default) | `--devices auto`: every visible GPU, otherwise the CPU. Narrow the set with `HIP_VISIBLE_DEVICES`. |
| CPU and RAM only | `--devices cpu` |
| One GPU | `--devices cuda:0` |
| Several GPUs | `--devices cuda:0,cuda:1,...`. The tool splits calibration sequences across devices and shards tensor rows. This is not a guarantee of bitwise equality across device counts. |
| Layer-wise placement | Calibration and layer-wise QAD move active layers to devices and keep hidden states in host RAM. This is not a whole-run VRAM guarantee. |

"Several GPUs" means row-sharded quantization with data-parallel calibration. It is not
a tensor-parallel forward pass.

Sensitivity probing, final-KL training and evaluation have different placement
requirements. See the [phase resource map](operations.md#phase-resource-map)
before running a model larger than device memory.

## Choosing the quality target

Choose an allocation-error budget, a tuning KLD target, or an allocation bit rate.
None guarantees the final checkpoint's quality.

* `--max-error-ratio R` (the default when you set neither option: 6 with QAD, 4 with PTQ
  only; the reasons are in [results.md](results.md#the-default-quality-target)): the
  sum of sensitivity-weighted row errors must stay within R times the sum of each
  row's best candidate error. The allocator spends the fewest bits within that global
  budget. This is not a per-row bound, a KLD target, or a task-quality guarantee.
  Rule of thumb from the measurements: ratio 8 saves about 0.2 bpw and costs about 30 %
  more KLD; ratio 4 costs about 0.27 bpw more and removes about a third of the KLD.
* `--target-kld K` (with `--calib` and `--eval-text`): state the fidelity you need and let the
  tool find the ratio. It quantizes PTQ checkpoints at a few ratios (default 2, 4, 8, 16, one
  shared phase 1, probe and encode pass per round), measures KL(original to quantized) of each
  on the tuning text, fits log KLD against log ratio, and runs with the largest ratio whose
  predicted KLD is within K. If K lies outside the measured range it measures more ratios
  (up to four extensions). `lossless` is only the corpus-specific KLD < 0.01 check;
  historical Qwen3.5-4B measurements were 0.061 at 3.0 bpw with QAD and 0.034 at
  4.0 bpw PTQ. QAD can improve or worsen final KLD; interpolation does not guarantee
  the chosen artifact meets K. Measure it independently with `quality` and an explicit
  gate. Measured tuning points are written to `<dst>.tune.json`.
* `--target-bpw B`: target the allocator's modeled bit rate. The tool spends the bits
  where they help most. This is not the complete serialized checkpoint size; use
  `hi2 size` to measure the written artifact.
* `--max-error-ratio a,b,c`: write one checkpoint per ratio to `<out>-r<ratio>`. The tool
  shares phase 1, the probe, and the encode pass between them.
* `--bits 2,4,8`: allow INT8 rows. They are off by default because they did not help at
  3.0-3.5 bpw ([lowbit.md](lowbit.md)).
* `--bits` and `--blocks` also restrict the candidates. For example, `--bits 4 --blocks 64`
  gives a uniform INT4 reference.

## Rounding and transforms

| Option | Default | Notes |
| --- | --- | --- |
| `--rounding gptq` | on with `--calib` | Hessian-aware rounding. On Qwen3.5-0.8B at 3.0 bpw it takes the perplexity ratio from 1.92 to 1.32. It changes the chosen codes only, never the format. |
| `--rotation on --smooth 0.5` | on | Smooth the activation channels, then rotate. This was the best transform at every bit width. Without `--calib`, smoothing is skipped. |
| `--smooth-search a,b,c` | off | Per-tensor smoothing exponent (spec 8.5): try these values and `--smooth` for every tensor or shared input and keep the lowest INT2/B32 error. Measured worse than the single exponent in all runs ([format-spec.md](format-spec.md), 8.5); keep it off. |
| `--refine-passes N` | 2 without QAD, 0 with QAD | Local code refinement after GPTQ (spec 8.9): N sweeps that move a code one level when the Hessian error falls. Two sweeps lower the KLD of a PTQ model by 3-13 % for about 10 % more encode time (a fused HIP kernel on gfx10/gfx11). QAD retrains the codes, so it is off with QAD. See [format-spec.md](format-spec.md), 8.9. |
| `--sequential` | off | Encode on the quantized prefix. It gave no measurable gain. |
| `--act-order` | off | GPTQ column order by activation size. Not adopted. |
| `--effort fast\|normal\|max` | normal | More effort did not improve quality in our runs. |

The smoothing vector costs `16/R` bits per weight per tensor. The allocation counts it.

On gfx10 and gfx11 the GPTQ column loop and the refinement sweep run as fused HIP kernels.
They give the same result as the torch loops. `HI2_GPTQ_BACKEND=torch` and
`HI2_REFINE_BACKEND=torch` force the torch loops. The kernels build on first use (about 40 s
each) into `~/.cache/hi2/kernels` (`HI2_KERNEL_DIR`).

## Quantization-aware distillation (QAD)

With calibration text, `quantize` distills every decoder layer after PTQ. QAD is on by
default. The checkpoint stays a normal v3.2 checkpoint.

```bash
hi2 quantize <model> <out> --calib calib.txt --max-error-ratio 6     # PTQ + QAD
hi2 quantize <model> <out> --calib calib.txt --dry-run               # plan only
hi2 quantize <model> <out> --calib calib.txt --no-qad                # PTQ only
```

| Option | Meaning |
| --- | --- |
| `--qad-preset fast\|default\|thorough` | Training length. |
| `--qad-budget F` | Fraction of the preset's steps. The tool spends steps where layers still gain, with per-layer early stopping. |
| `--qad-samples N`, `--qad-layers A-B` | Training sequences, layer range. |
| `--qad-final-kl STEPS` | Final logit-KL phase: 200 steps with `default` and `thorough`, 0 with `fast`; 0 turns it off. Split layer-wise over the devices when it does not fit one GPU; skipped with a note if it still does not fit. See [lowbit.md](lowbit.md). |
| `--qad-config file.json` | All settings in a file. |

“Final” in `--qad-final-kl` means the last **training** phase, not final quality
evaluation. Internal validation selects training states. After training, run
`quality` on independent final-test text to accept or reject the published
checkpoint. QAD can improve or worsen that result.

With several `--devices`, QAD can split each training step's sequences across replicas
of the layer. `"data_parallel": false` disables this; norm training and layerwise
logit-KL use the primary-device path. Gradient summation order can change results.
Historical Qwen3.5-0.8B measurements on three GPUs: 0.09 s/step versus 0.16 s, and
1006 s versus 1117 s for the phase; setup and held-out evaluation remained serial.
These measurements do not guarantee speedup or bitwise equality on another recipe.

Runtimes on one V620 (QAD part, `default` and `fast`): Qwen3.5-0.8B 41 and 17 minutes;
Qwen3.5-4B about 3.1 hours and 1.3 hours. PTQ adds 10-50 minutes. The run prints its
expected time first. It is resumable per layer: run the same command again. Details are in
[qad.md](qad.md).

## Mixture-of-experts models

For supported per-expert 2-D source checkpoints, the adapter splits experts fused
in memory by Transformers. Fused 3-D expert tensors on disk are unsupported.
The tool shares gate and up smoothing and counts routed tokens per expert.
The default layout is `stacked`: one tensor group per layer. It loads
about 9 times faster in the vLLM fork. Use `--moe-layout per-expert` for one tensor per
expert. Details: [moe.md](moe.md).

## Cache and work directory

Keep these three locations distinct:

| Location | Purpose |
| --- | --- |
| Output (`DST` or `OUT`; recipe field `dst`) | The quantized checkpoint and its published reports. |
| Work directory (`--workdir`; recipe field `workdir`) | Run identity, phase state and recovery artifacts; default `<out>.work`. |
| Persistent cache (`--cache-dir`; recipe field `cache_dir`) | Compatible phase products that can be reused across runs; default `$HI2_CACHE_DIR` or `~/.cache/hi2`. |

The tool caches statistics, Hessians, error tables, and probe results under `--cache-dir`
(default `$HI2_CACHE_DIR` or `~/.cache/hi2`). Identity includes config/index/weight
contents (streaming SHA-256, one full checkpoint read per fingerprint), calibration
token IDs and product-specific settings. Same-size weight edits invalidate the cache.
Active `HI2_NGRAM_MMAP` inputs also contribute their canonical path and full file
SHA-256. Copied identical checkpoint files can reuse products when the other inputs
match. `STATS_VERSION=3` invalidates products lacking the current content/auxiliary
identity. Logs distinguish cache hits and misses; all inputs must remain stable.

```bash
hi2 cache list
hi2 cache size
hi2 cache prune [--older-than DAYS] [--keep-gb N] [--digest D] [--dry-run]
```

`workdir` (default `<out>.work`) contains `run.json`, shard recovery markers and
`cache_ref.json`. Resume requires matching resolved settings, content identities and
implementation environment. A matching workdir does not authorize replacing an
unowned output: `.hi2-owner.json` claims ownership before writes, and `hi2_run.json`
is published only after full artifact validation. Changed recipes need fresh paths.
QAD layers, tables and metadata use atomic replacement; interrupted temporary files
are not completion evidence. Abrupt termination during publication requires revalidation.
`--no-cache` keeps tables in the workdir. Never unlink permanent run/cache/output locks.
With persistent caching enabled, cache, workdir and output trees must be pairwise
disjoint and must not overlap source/calibration inputs, including through symlinks.
Use sibling directories, not an output or workdir nested inside the cache.

The identity fields answer different questions:

| Field | Identifies | Resume behavior |
| --- | --- | --- |
| `run_id` | The resolved quantization recipe, input identities and implementation environment. | Stays the same for a compatible resumed run. |
| `attempt_id` | One execution of that run. | A new execution gets a new value, even when it resumes the same `run_id`. |
| `job_id` | One durable registry submission, for quantization or another supported operation. | Repeating the same request and idempotency key returns the same job, not a new attempt. |

`run.json` and the published `hi2_run.json` record `run_id` and `attempt_id`.
Use `status WORKDIR` for the run record and `jobs status JOB_ID` for the job.
A successful older attempt does not prove that the current job completed.

When the code that produces a cache entry changes, raise `STATS_VERSION`,
`TABLES_VERSION`, or `PROBE_VERSION` in `hi2/cache.py`. See [speed.md](speed.md).

## Checking the size

`hi2 size` reads only the safetensors headers and the small per-row precision arrays. It
works on both expert layouts and takes a second or two.

```bash
hi2 size out/q4b --src Qwen3.5-4B --top 10
```

The report gives:

* the quantized linears: tensor count, weights, bytes, and the measured bpw;
* the bpw split into codes, block parameters, row constants, smoothing vector, and row
  index (offsets, precisions, block sizes, rotation flags);
* the share of weights at INT2, INT4, and INT8;
* dense tensors in their stored dtype (copied source tensors retain it; trained overrides may be fp16);
* with `--src`, the compression against the source model's tensors;
* a table by tensor family (`--by kind`), by layer, or by tensor.

Measured quantized-tensor bpw includes representation metadata such as row indexes,
unlike predicted allocation bpw. It excludes dense tensors and file/container overhead.
Compare model-level bytes separately and use equivalent tensor/file scopes across formats.

## Examples

```bash
# 4B model, one GPU, quality target 6, held-out check
hi2 quantize Qwen3.5-4B out/q4b --calib calib.txt --max-error-ratio 6 --devices cuda:0
hi2 eval     Qwen3.5-4B out/q4b --text heldout.txt --skip-tokens 0 --device cuda:0

# Dense fp16 Hugging Face copy; convert separately if the consumer requires GGUF
hi2 export-dense Qwen3.5-4B out/q4b out/q4b-dense
```

## Automation, errors and recovery

`--json` emits one versioned result/error object on stdout; help and version remain
plain text. Success keeps operation-specific fields at top level and adds
`schema_version: 1`, `command`, `status`, `phases`, and `artifacts`. Errors include
`error.type`, `error.category`, `error.message`, `error.exit_code`, and `error.remedy`.
Diagnostics go to stderr, including native-library output. `--quiet` suppresses
application progress; `--log-format jsonl` emits structured application events
(native diagnostics may remain text). `--debug` includes tracebacks. Events identify
phase, reuse/skip outcomes, layer/step counters, scoped ETA where available, memory
and artifact paths. `run.json` keeps the latest 256 phase events; durable jobs expose
bounded event pages.

| Exit | Error category | Action |
| ---: | --- | --- |
| 0 | Successful operation | Inspect `state`/quality status; a successful status query is not job completion. |
| 2 | Invalid input or CLI usage | Correct paths, values or flag combinations; run `--dry-run`. |
| 3 | Backend/resource failure | Run `doctor`; check dependencies, device availability, space and permissions. |
| 4 | Incompatible resume | Preserve existing work and use fresh work/output paths. |
| 5 | Final quality rejection | Read `hi2_quality.json`; revise the recipe and re-evaluate independently. |
| 6 | Worker/internal failure | Inspect job diagnostics or rerun with `--debug`. |
| 130 | Interrupted invocation | Inspect `status WORKDIR`; replay unchanged inputs/settings. |

Quality status is `target_met`, `target_not_met`, or `not_measured`; with no thresholds,
a valid measurement is `measured`, not an acceptance claim. Structural success and
numerical quality never imply native-kernel qualification.

Save the resolved configuration and make overrides explicit:

```bash
hi2 quantize MODEL OUT --devices cpu --no-qad --dry-run --json > plan.json
hi2 recipe save --from-plan plan.json --out recipe.json
hi2 recipe run recipe.json --dry-run
hi2 recipe run recipe.json --set 'dst="OUT2"' --set 'workdir="OUT2.work"'
```

Recipes contain exactly `schema_version: 1` and `config`; unknown fields/versions
are rejected. `config` uses shared service fields, not CLI flag names:

| Setting | Recipe representation |
| --- | --- |
| Source/output/work paths | `src`, `dst`, `workdir` (required by CLI recipe replay/jobs); optional `cache_dir` |
| Devices | `devices: ["cpu"]` or a list such as `["cuda:0", "cuda:1"]`, not `"auto"` or a comma-separated string |
| Calibration | `calib`: one file path or a list of file paths |
| Encoding | Nested `cfg`, for example `{"effort": "fast", "rounding": "nearest"}` |
| Distillation | `qad: null` disables QAD; a `qad` object contains resolved distillation settings, not a preset name |
| Allocation | `target_bpw`, `max_error_ratio`, or `max_error_ratios`; choose a bit target or error-ratio mode, not both |

Explicit `--set FIELD=JSON` values win; dotted fields such as `cfg.effort` update one
nested setting. Relative paths resolve from the invocation's working directory,
not from the recipe file's directory; saved resolved plans use absolute paths.
In-memory calibration IDs cannot be exported as replayable recipes. Prefer saving
a CLI dry-run plan: hand-written service recipes do not inherit the CLI's automatic
device selection, QAD preset, sensitivity or cache defaults. CLI `--target-kld`
tuning is not a recipe field. The [MCP recipe schema](mcp.md#tool-contracts) shares
these settings but forbids caller-supplied work/cache paths and assigns them privately.

## Durable local jobs

```bash
hi2 jobs submit recipe.json --operation quantize --idempotency-key run-001 --state-root STATE --json
hi2 jobs status JOB_ID --state-root STATE --cursor 0 --limit 100 --json
hi2 jobs cancel JOB_ID --state-root STATE --json
```

The submit target is a JSON file, not a command string. Accepted operation payloads
are defined in [`jobs._prepare`](../hi2/jobs.py):

| `--operation` | Required JSON fields | Optional fields |
| --- | --- | --- |
| `quantize` (default) | `schema_version: 1`, `config` as above | Optional settings inside `config` |
| `validate` | `path` | None |
| `quality` | `src`, `dst`, `text` | `excluded` (list of paths), `max_kld`, `max_ppl_ratio`, `samples`, `seq_len`, `device`, `dtype` |
| `portable` | `src`, `out` | `assets_from` |

Only quantization uses the recipe envelope. For example, a validation request file
contains `{"path": "/absolute/path/to/OUT"}`. CLI `--exclude-data` becomes `excluded`
in quality-job JSON. Quality defaults to CPU/float32, 8 samples of 2048 tokens.
Unknown operation fields are rejected. Submit success acknowledges durable
registration, not completion; status/cancel can exit 0 even for a failed job.

Jobs require a local POSIX host with SQLite, `fcntl.flock`, inherited file descriptors,
detached subprocess sessions and signals. Keep the launching Python environment and
checkout available until workers finish. Use private operator-owned state on a local
filesystem with working advisory locks; this is a single-host registry, not a
distributed scheduler or OS sandbox. See [test prerequisites](#development-tests).

Persist the request, 1–512-character idempotency key and returned job ID. Keys are
unique across operations within one registry. Retrying the same normalized request
with its key returns the original job, including a terminal job; a key is not a
request to rerun. A different operation/request with that key is rejected. Keep the
same `STATE` across reconnects and preserve the device-visibility environment for
exact retries. Device leases and overlapping work/output paths serialize conflicts;
CPU compute is exclusive within that registry. Separate registries do not share
device arbitration.

Closing the client does not cancel a job. Explicit cancellation records a request;
it does not mean the worker has stopped or released its resource leases.
The supervisor uses SIGINT, then bounded SIGTERM/SIGKILL escalation if needed.
Poll `state` until `succeeded`, `failed`, `cancelled`, or `interrupted`.
`queued`, `running`, and `cancelling` are nonterminal.

For quantization, a validated commit from the job's bound `attempt_id` can win a
late cancellation, but only after worker ownership ends. The real `returncode`
is preserved and can be nonzero despite `succeeded`; orphan recovery can leave
it `null`. A successful artifact from another attempt does not prove that this
job committed.

A stopped orphan retains leases through its ownership descriptor; stale heartbeats
or PIDs are not permission to release them. Resume or terminate the owned process
through the OS rather than deleting locks. A status request can restart an unstarted
queue entry; interrupted started work needs a new idempotency key and the same
identity-compatible recipe. Status pages return at most 100 events, retaining the
latest 1000 with `events_truncated`; use `next_cursor` for the next page. Detailed
worker logs remain separate and are not bounded by event retention.

The [MCP adapter](mcp.md) adds operator-configured model/calibration/output roots and
stable registered IDs. Its `0700` state directory must never accept uploads or copied
`.pt` caches. Path containment is not a pickle sandbox. Credentials remain operator-side;
tool arguments cannot enable remote code, arbitrary commands or untrusted cache paths.

## Quality and portable provenance

```bash
hi2 quality MODEL OUT --text final-test.txt --exclude-data tuning.txt \
    --max-kld 0.1 --max-ppl-ratio 1.1 --samples 8 --seq-len 128 \
    --device cpu --dtype float32 --json
hi2 export-portable OUT PORTABLE --assets-from MODEL --json
hi2 validate PORTABLE --json
```

These thresholds illustrate an application gate, not promised quality. `quality`
structurally validates the artifact, checks the reference fingerprint against
`hi2_run.json` when available, then measures with `skip_tokens=0`. It checks exact
document overlap against recorded calibration paths and each repeated
`--exclude-data` path. Supply tuning/QAD inputs not already recorded. This is not
semantic deduplication or proof that unrecorded training data is independent.
Missing recorded corpora fail the check rather than being silently ignored.
Without recorded source identity, `reference.matches_quantization_source` is
`null`, not proof that the reference is the quantization source.

`hi2_quality.json` records corpus identity, sampling, criteria and numerical results;
a failed acceptance gate can still publish this report. `measured` without thresholds
is not `target_met`. The evidence remains `dense_numerical`, with native execution
`not_measured`. See [evaluation](evaluation.md) for interpreting the metrics.

Portable export requires a new output directory, preserves HI2 weight bytes and the
artifact's own `hi2_run.json`/`hi2_quality.json` when present, and materializes recognized
assets and their local dependencies. `--assets-from` grants trust to an asset source;
it does not replace run/quality evidence or authorize remote code execution.
`hi2_portable_manifest.json` covers copied files with sizes and SHA-256 checksums.
This is neither GGUF conversion nor a bundled native runtime.

Preserved evidence retains original paths/run IDs; export does not measure quality
again or rewrite historical provenance. Calibration/tuning corpora and reference
model weights are **not bundled**. Later quality evaluation still needs the reference
and the recorded corpora at readable paths; on MCP those paths must also lie within
the operator's roots. Portable assets alone do not make evaluation self-contained.

Quality updates hold the artifact output lock and refresh portable checksums, but the
report and checksum manifest are separate atomic replacements, not a jointly
hard-kill-atomic transaction. After an abrupt kill, run `validate` before accepting
or distributing the bundle: a mismatched pair is rejected. Do not bypass the failure
by deleting provenance or checksum evidence. Distribution and recovery details:
[operations](operations.md#distribute-a-checkpoint).

## Development tests

Use the [installation guide](installation.md#checkout-and-virtual-environment)
for test/MCP extras and backend selection. From the checkout, use the same Python
interpreter for installation and tests:

```bash
python -m pytest -q tests
```

Useful focused suites (paths are relative to the checkout):

| Boundary | Test files for `python -m pytest -q ...` |
| --- | --- |
| CLI parsing, structured errors, recipe/service planning | `tests/test_cli_contracts.py tests/test_service.py` |
| Durable processes, ownership, cancellation and reconnect | `tests/test_jobs.py` |
| Root capabilities and ordinary/Tasks MCP contracts | `tests/test_workspace.py tests/test_mcp_server.py tests/test_mcp_tasks.py` |
| Quality acceptance and checkpoint/portable safety | `tests/test_quality.py tests/test_checkpoint_safety.py` |

Job/MCP coverage uses real subprocesses, POSIX advisory locks, FIFOs and signals
(including stop/continue/kill fault cases), not only in-process calls. Use a local
Linux/macOS environment that permits those operations and an importable checkout
in child processes. Restrictive process sandboxes and Windows are not equivalent
test environments. CPU fixtures are generated locally; these suites need no GPU,
model download, external inference service or credentials.

Install the MCP extra before claiming adapter coverage: optional SDK skips are not
passes. Accelerator-dependent skips likewise do not qualify a GPU or native loader.
The [CI workflow](../.github/workflows/tests.yml) defines automated checks; historical
counts and actual SDK/protocol measurements are recorded in
[installation](installation.md#recorded-environment-and-evidence) and
[MCP compatibility](mcp.md#installation-and-compatibility), not asserted by these
commands.
