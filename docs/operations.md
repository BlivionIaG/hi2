# Resources, recovery and distribution

Use [getting started](getting-started.md) for an ordered run and
[usage.md](usage.md) for flags. This page separates implemented lifecycle behavior
from estimates and historical measurements. No new GPU or native-runtime
validation was performed for these documentation changes.

## Read the plan without overreading it

Run the **same intended command** with `--dry-run` first. The resolved pipeline
checks supported numeric/configuration values, incompatible options, requested
CPU/CUDA device availability, input/output path aliases, model metadata and
indexed safetensors presence, and calibration token requirements. It reports
resolved settings, output destinations and requested phases. It does not execute
model forward/backward passes or native loading.

The resource section leaves total runtime and peak GPU memory (VRAM) unknown.
Its QAD fp32 hidden-state estimate is roughly
`4 buffers * (training + validation sequences) * seq_len * hidden_size * 4 bytes`.
This is not the full resident process memory (RSS). Source weights, live layer
copies, Hessians, optimizer state and logits need additional memory.
A successful dry-run does not reserve RAM, VRAM or disk space. Readable metadata
also does not prove that Transformers can execute the architecture.
Source: [`service.resolve_config`](../hi2/service.py) and
[`pipeline.plan`](../hi2/pipeline.py).

Dry-run performs read-only ownership/identity checks for existing work/output
paths, so an incompatible resume can fail during planning. It is not a durable
ownership claim: execution must acquire the locks and repeat those checks before
it can resume or write. Another process can change availability after planning.
There is no promise that the number of requested phases equals the number
actually completed: final-KL can be skipped by its fit check.

## Phase resource map

| Active phase / symptom | Host RAM and device demand | Disk and time considerations |
| --- | --- | --- |
| Input/tokenization, plan | Tokenizer and token IDs; shard/header inspection, not a model forward | Source/tokenizer files must be local and complete. A plan is not a capacity reservation. |
| Calibration/statistics | Source model in host memory; active decoder layers/device replicas and hidden-state buffers; per-input Hessians scale as K squared | Persistent statistics, Hessians and tables can be much larger than the output. Calibration duration is separate from QAD. |
| Sensitivity probe | Different placement from layer training: whole-model or layer-sharded probe, with retained reference probabilities/activations; `sens-dtype auto` may select fp16 for fit | Repeated model forwards; cached probe is cheap to reuse. Reducing QAD buffer dtype does not solve a probe-placement OOM. |
| Allocation | Candidate tables and picks, primarily host data | Usually small compared with probe/training; allocation bpw is not peak RAM or actual file size. |
| GPTQ encode | Active weight/candidate batches, Hessian transforms and factorization; source-shard staging | Reads cached statistics, writes output shards or per-expert staging. First-use fused compilation is extra. |
| Layer-wise QAD | Source model/teacher-student hidden states in RAM; live layer, fp32 latents, ladder parameters, optimizer state and optional replicas on devices | Completed `layerNNN.pt` artifacts; saved teacher `head_inputs.pt` when produced. The printed QAD time estimate covers layer training, not final-KL or total execution. |
| Final logit-KL | Whole-network student split layer-wise; teacher placement avoided when saved head inputs are available; optimizer/activation/logit headroom still required | Separate phase with `final_kl.pt` and report. At 27B this phase took longer than layer-wise QAD. CPU is not protected by a GPU-fit estimate. |
| Publish/restack | Shard/container processing and staging, not just one model layer | Stacked output can coexist with its per-expert staging copy until publication finishes. Plan space for both. |
| Structural validation / sampled `verify` | Scans containers or reconstructs selected tensors; no full model inference | Structural validation and numerical reconstruction coverage are distinct. See [evaluation](evaluation.md#verification-coverage). |
| `eval` | Reference and dense reconstructed models loaded in turn, whole-model or layer-sharded; reference vocabulary log-probabilities retained on CPU | Temporary dense fp16 checkpoint plus its load-time state; this can dominate disk/RAM even if the HI2 file is small. |
| `export-dense` | Dequantized state dictionary and fp16 conversion can coexist in RAM | Full dense checkpoint on disk, not compressed inference. |

For a rough disk envelope, add **source + cache + work artifacts + staging + all
requested outputs + temporary/persistent dense copies + portable asset copies**.
Budget the peak overlap, not only the final output. QAD ratio sweeps repeat the
QAD/final-KL work and keep per-output artifacts; compatible statistics/probes can
be shared. Do not assume a single layerwise memory model applies to every phase.

### Measured examples, not universal estimates

* The [speed study](speed.md#disk) reports persistent statistics of about 0.9 GB
  for Qwen3.5-0.8B, 7.0 GB for 4B and 11.8 GB for OLMoE. These are that study's
  entries, not all working/output disk. Original per-workdir Hessians were larger.
* Historical one-V620 QAD measurements report about 15 GiB process RAM/1.4 GiB
  GPU peak at 0.8B and 46 GiB/6.0 GiB at 4B during layer-wise training. They do
  not bound probe, final-KL or evaluation memory ([qad.md](qad.md)).
* The completed 27B `default` run on four V620s took **247 minutes layer-wise +
  470.8 minutes final-KL**. Saved teacher head inputs were about 10.3 GiB. These
  are distinct from the compressed 15 GiB checkpoint ([surfaces.md](surfaces.md#the-default-27b-run-finished-2026-10-29)).
* `qad.estimate_seconds` scales measured layer-training work by tensor shapes and
  tokens; its CPU multiplier is a heuristic, not a CPU benchmark. Do not report
  it as a measured total-job ETA.

Final-KL means the final **logit-KL training phase**, not independent quality
evaluation. The `default` and `thorough` presets request it. The `fast` preset
and explicit `--qad-final-kl 0` disable it.

On GPU, the fit check can skip this phase if the estimated per-device memory
need remains too large after sharding. The estimate uses nominal capacity and
headroom. It does not guarantee free VRAM or prevent every out-of-memory (OOM)
failure. Retain the skip message and phase status; do not label that output
“final-KL trained.” `final_tokens` reduces positions in the loss, not the memory
needed to place model weights. Even a completed phase needs independent
[final quality evaluation](evaluation.md#allocation-and-kld-targets).

## First-use and slow starts

Fused HIP GPTQ/refinement compilation, model-specific Triton compilation and
GEMM autotuning are different costs. The V620 studies reported roughly 40-50 s
for a fused extension build and around ten minutes for the first linear-attention
backward autotune, sometimes longer for a new dtype. Preserve compatible build
caches; expect a different compiler, device or dtype to require new compilation.
Read [installation](installation.md#first-use-compilation-and-smoke) for eligible
architectures and Torch fallback controls. A long first layer alone is not proof
of a deadlock; inspect the current phase/compiler activity and available memory
before stopping only the process you own.

## Artifact ownership

The **output** is the checkpoint you evaluate or distribute. The **workdir**
holds one run's recovery state. The **persistent cache** holds compatible phase
products for reuse across runs. None of these directories replaces the others.

Keep persistent cache, workdir and output trees separate from one another and from
source/calibration inputs. Preflight rejects equality, containment and aliases
before writes: an output nested under a cache is unsafe because pruning owns cache
products. `--no-cache` instead keeps phase products inside the owned workdir.

| Artifact | Role and retention |
| --- | --- |
| Local source checkpoint, tokenizer and calibration inputs | Immutable inputs for a repeatable run; retain provenance independently of cache/work state. |
| Persistent cache (`--cache-dir`, `$HI2_CACHE_DIR` or `~/.cache/hi2`) | Reusable statistics/Hessians/tables/probe products under their keyed identities; see [speed.md](speed.md#the-cache). |
| `<workdir>/run.json` | Versioned run identity, resolved configuration, outputs, lifecycle/phase record. Do not edit it to bypass ownership checks. |
| `<workdir>/.run.lock` and output-side lock files | Exclusive writer coordination. A lock filename's existence alone does not mean a live owner; do not remove lock files to evade a lock. |
| `cache_ref.json`, phase tables and sensitivity records | References or phase products used by this run. `--no-cache` uses workdir-local products instead. |
| `qad/layerNNN.pt`, `qad/head_inputs.pt`, `qad/final_kl.pt` | Completed layer states, saved teacher head inputs and completed final-KL state; multi-ratio runs use `qad-r<ratio>` namespaces. |
| Per-expert staging and shard/report recovery artifacts | Intermediate publication state; preserve during an unchanged interrupted run. Do not mix it into another run's output. |
| Output `.hi2-owner.json`, `hi2_report.json`, `hi2_run.json` | Ownership, tensor/quality-accounting report and successful run record. A directory's mere presence is not completion. |
| Temporary evaluation/tuning dense products | Scratch space owned by that invocation, not a user checkpoint or a substitute for persistent diagnostics. Never delete similarly named user data as cleanup. |

These are cooperative POSIX locks, not filesystem permissions or an OS sandbox.
Do not unlink their files: a live owner may still hold the old inode while a new
process locks a different one. A recorded PID, heartbeat or `running` state alone
is not proof of liveness. Keep state on a filesystem with working local advisory
locks; multi-host ownership is not qualified.

## Resume and invalidation

Execution takes exclusive ownership of the workdir and output paths.
The `run_id` identifies resolved settings, inputs and implementation environment.
A compatible resume keeps that ID and can reuse committed phase products.
Each execution gets a fresh `attempt_id`. A durable submission has a separate
`job_id`; see [the identity fields](usage.md#cache-and-work-directory).

A different run identity is rejected rather than mixed with prior output.
A nonempty legacy workdir without a valid run manifest is not automatically
adopted. The pipeline structurally validates the output before publishing its
success record. Each artifact replacement is atomic, but publication across
all output markers and `run.json` is not one atomic transaction.
Ordinary publication exceptions restore prior success markers. A hard kill
can leave partial publication or a stale `running` record. A marker alone
neither repairs corruption nor proves that the current attempt finished.

Recovery granularity is phase-specific:

* Calibration/statistics/probe reuse complete compatible cache products.
* QAD reuses completed layers; the interrupted layer starts again. There is no
  optimizer-step resume within that layer.
* Completed final-KL has its own fingerprint and artifact. An interrupted final
  phase restarts the phase, not its last optimizer step.
* Output shard reuse must carry compatible metadata/report coverage. Do not
  create completion markers manually or copy them between destinations.

Keep the source/calibration files, resolved settings, cache and work/output
locations unchanged when rerunning after interruption. For changed recipes,
model weights, tokenizer or calibration, use **new output and work directories**;
compatible persistent cache products may still be reused according to their
keys. Never treat a filename, size, or leftover `.pt` as proof that it belongs to
the new run. Historical caches predating content-based identity need particular
care: their model fingerprint used config bytes and weight/index names and sizes,
not complete weight contents. Do not manually rekey old entries; use fresh state
when changing such inputs.

After stopping your foreground invocation, inspect `hi2 status WORKDIR` and
retain its logs. That command reads recorded state; it does not prove the process
has exited. Once the actual owner has exited, rerun the unchanged command to let
the locked lifecycle validate/reuse compatible state, then run `hi2 validate OUT`
before acceptance. If ownership is still busy, do not delete locks or launch a
second writer. Changes to the recorded implementation/environment can also make
resume incompatible; preserve the old run and choose fresh paths rather than
editing its identity.

For [durable MCP jobs](mcp.md), a cancellation acknowledgement means **requested**,
not **stopped**. Keep polling until the job reaches a terminal state. The worker
must exit before reconciliation releases its resource leases.
A validated commit from the job's bound `attempt_id` can win a late cancellation.
The recorded process `returncode` remains the actual exit code, not a fabricated
zero. An older successful attempt with the same `run_id` cannot complete this job.
Read the persisted job result and the run/artifact evidence together.
Interrupted work is not automatically replayed.

The model fingerprint uses `sha256-content-v1`. It hashes full config/index/shard
bytes and relative filenames, including nested index-referenced shards.
An unchanged copy with the same relative layout matches; same-size weight edits
do not, even with preserved timestamps. Current statistics use
`STATS_VERSION=3`, which also includes active auxiliary model inputs such as
`HI2_NGRAM_MMAP` by canonical path and full SHA-256. Older statistics keys miss,
as do the table/probe keys derived from them.
Hashing costs a full checkpoint read. Keep source and auxiliary files unchanged
throughout a run.

Preview cache pruning with `hi2 cache list`, `hi2 cache size` and
`hi2 cache prune --dry-run`. Remove only the selected obsolete products, never
unrelated checkpoints, all of a user's cache, or repository history. Source:
[`runtime.py`](../hi2/runtime.py), [`service.py`](../hi2/service.py),
[`pipeline.py`](../hi2/pipeline.py), [`cache.py`](../hi2/cache.py).

## Troubleshooting by phase

| Symptom | Focused remedy |
| --- | --- |
| Insufficient calibration/evaluation tokens | Compare available with required **usable** tokens and the model tokenizer. Add independently assigned documents or intentionally change sample/window settings; do not solve it by consuming final-test data. |
| Metadata/shard/decoder discovery failure | Confirm local safetensors layout, complete index and model/Transformers compatibility; consult the support matrix. More VRAM cannot fix an unsupported layout. |
| OOM during probing | Inspect probe dtype and model placement, not only QAD buffer settings. Re-plan with explicit devices and use a fresh run for changed settings. |
| OOM during layer QAD | Inspect source/hidden-state host RAM, layer/optimizer GPU demand and replica count; reducing buffers can change training numerics. |
| Final-KL skipped or OOM | Record the skip/failure. Its full-network placement differs from layer QAD. Disabling it is a deliberate recipe change, not equivalent completion. |
| OOM/disk-full during `eval` or dense export | Budget dense weights, reference vocabulary logits and temporary disk; small compressed file size is not the memory requirement. |
| Fused kernel build failure | Check ROCm Torch/compiler/architecture alignment, or explicitly use the Torch reference path. Do not force gfx1030 binaries on unrelated devices. |
| Incompatible/unowned workdir or output | Preserve it and use fresh named destinations. Do not erase the manifest, copy markers, or bypass the ownership check. |
| Interrupted run or corrupt artifact | Preserve logs/run identity and retry unchanged inputs for normal phase resume. For corrupted committed artifacts, isolate the affected run and use fresh work state rather than trusting or restamping it. |
| Broken tokenizer after moving output | Inspect source-asset symlinks; use a portable copy rather than assuming the HI2 directory is self-contained. |
| `Unknown quantization method: rdna_hint_flex` | The installed inference runtime lacks the needed registration/loader. Dense evaluation cannot fix or certify it; see the [native handoff boundary](getting-started.md#export-and-native-serving). |

## Distribute a checkpoint

Ordinary quantization and dense export can link tokenizer/auxiliary files to the
source directory; restacking can preserve those links. Removing or moving the
source can therefore break the output. Checkpoint tensors retaining a source
dtype is unrelated to this filesystem dependency.

For a self-contained HI2 copy, select a new destination and explicitly trust
the source root for linked assets:

```bash
hi2 export-portable OUT PORTABLE --assets-from MODEL
hi2 validate PORTABLE
```

This copies quantized shards unchanged rather than re-encoding or dequantizing
them, materializes the explicitly allowed local assets/dependencies, and records
hashes in `hi2_portable_manifest.json`.
External linked assets require an explicitly trusted source root; missing,
escaping or remote dependencies must not be silently accepted. The destination
must not exist, even as an empty directory. Export locks source and destination,
validates the source, copies into the destination, writes its manifest, then
validates the result. It is not an atomic directory publication: caught failures
remove the invocation's new directory, but a hard kill can leave a partial copy.
Never distribute a destination still being written. Retain the original until
export has returned successfully and `hi2 validate PORTABLE` passes; after an
interruption preserve diagnostics and use a different new destination.
Portability is not native-loader compatibility.

The copy preserves the artifact's `hi2_run.json` and existing `hi2_quality.json`
when present; `--assets-from` cannot supply or replace that evidence. Recorded
calibration/tuning corpora remain external inputs, not bundled datasets. Later
quality evaluation requires recorded calibration files plus any explicitly
excluded tuning files, and rejects missing inputs or exact-document overlap.
It rejects a different reference checkpoint when the run records its source
fingerprint; an artifact without that provenance does not acquire it merely by
being exported. This overlap check is not semantic deduplication.

Quality evaluation locks the artifact and updates its quality report and portable
checksum manifest together under that lock. Both accepted and rejected quality
gates leave a structurally valid bundle without changing weight bytes. An ordinary
publication exception restores the prior report/manifest. The two filesystem
replacements are not jointly crash-atomic: a hard kill between them can leave a
bundle that fails validation. Do not distribute it or bypass checksum validation;
export a fresh copy from the retained original.

The manifest covers bundle files; it is not a signature, a Python dependency
lock or a code-safety audit. Local custom-code assets may be copied with their
dependencies. Review code before enabling a runtime's trust/custom-code option;
portable packaging does not sandbox it or install external Python libraries.

`export-dense` is the other route: a dense fp16 Hugging Face checkpoint for a
compatible engine, **not GGUF** and not native low-bit HI2 inference. A llama.cpp
handoff requires a separately compatible HF-to-GGUF conversion and its own model
support checks. No version-pinned native HI2 runtime or validated inference
request is supplied by this page.
