# Prepare data and read the results

Use `validate` for structural validity, `verify` for sampled weight comparisons,
and `size` for byte accounting. Use `eval` to measure dense numerical quality,
or `quality` to measure it and apply explicit acceptance thresholds. The
[benchmark harness](../bench/README.md) measures named tasks. None of these
checks replaces the others; dense evaluation does not test native HI2 kernels.

**PTQ** means post-training quantization. **QAD** means quantization-aware
distillation. **KLD** means Kullback–Leibler divergence between the reference
and quantized token distributions. The final logit-KL phase is **training**;
“final test” below means independent evaluation after training and selection.
**bpw** means bits per weight.

## Prepare independent data

Assign document identities to these roles **before** mixing or tokenizing:

| Role | Use | Boundary |
| --- | --- | --- |
| PTQ calibration | Statistics, Hessians, candidate errors and sensitivity probing | PTQ takes a prefix of the calibration input. This is training/selection data, not a test set. |
| QAD training and internal validation | Seeded windows from the entire calibration input; internal validation chooses retained states | Internal validation is used by training and is not untouched final evaluation. It can share the PTQ prefix. |
| Tuning validation | `--eval-text` for KLD-target search, or repeated comparison of ratios/recipes | Once used to choose a checkpoint, it is selection data. Do not report it as an untouched final test. |
| Final test | One independent evaluation after choosing the recipe/checkpoint | Reserve a nonempty corpus outside **all** calibration, QAD and tuning inputs. Do not tune on its scores. |

PTQ and QAD may use the same calibration file. QAD separates its own training
and internal-validation windows, but uses both to produce the model.
Reserve tuning and final-test documents separately. They should represent the
target domain without sharing document identities with training inputs.
This guidance does not establish leakage in every historical experiment.
Retain each result's actual protocol and limitations.

### Token budgets and offsets

These are the current standard-recipe requirements, not additional independent
CLI defaults. [usage.md](usage.md) and command help remain the options reference.

| Phase/recipe | Token requirement at sequence length 2048 |
| --- | ---: |
| PTQ, 128 calibration sequences | 128 x 2048 = 262,144 |
| QAD `fast` or `default`, 256 training + 8 internal validation windows | 264 x 2048 = 540,672 |
| QAD `thorough`, 512 + 8 windows | 520 x 2048 = 1,064,960 |
| Default standalone `eval`, 8 sequences | 8 x 2048 = 16,384 usable input tokens after any offset |

For calibrated QAD, the input must satisfy **both** PTQ and QAD requirements,
not their sum: windows can overlap the PTQ prefix. With balanced multi-file QAD
sampling, each input needs enough full windows for its assigned share; a large
aggregate alone is insufficient. Source: [`pipeline.py`](../hi2/pipeline.py)
(`load_calibration_ids`, `sample_calibration_windows`) and
[`QadConfig`/presets](../hi2/qad.py).

Use offset zero for a separately held-out file:

```bash
hi2 eval MODEL OUT --text final-test.txt --skip-tokens 0 --device cpu
```

Here `MODEL` and `OUT` are the matching local source and completed HI2 checkpoint;
the final-test file must contain at least 16,384 model-tokenizer tokens.
At these settings the scorer evaluates 8 x 2047 = 16,376 next-token positions
(the last position in each window has no target).

An offset of 800,000 is meaningful only for an explicitly larger corpus with at
least 816,384 tokens, where the evaluated documents are independently excluded
from every training/selection input. It is **not** suitable for a dedicated
200,000-token holdout, and skipping the PTQ prefix in a QAD calibration file
never establishes that isolation: QAD samples across the whole file.

### Safe preparation and exact overlap

1. Keep source documents and their provenance. Reserve final-test documents
   first, then tuning documents; never pass either source set to calibration.
2. Normalize records the same way you will feed the model. For an exact-duplicate
   check, hash normalized document text (for example, SHA-256 of UTF-8 text after
   a fixed whitespace normalization). Group equal hashes **globally across all
   sources**, not separately by file, and assign each group to one role only.
3. Check set intersections: calibration/QAD versus tuning, calibration/QAD versus
   final, and tuning versus final must all be empty. Save document hashes and
   split assignments so an overlap decision remains auditable after interleaving.
   Check chunked documents by original document identity as well as text hash;
   different chunks of one document are not independent documents.
4. Mix the calibration documents through the whole output rather than placing
   one domain after another. Run the source model's tokenizer and inspect actual
   token counts, not a character estimate or requested target alone.
5. Check the tuning and final files are nonempty and have enough usable tokens.
   Whole-document selection can overshoot a token target; small or exhausted
   sources can miss it. A successful file write alone is not split validation.

The current `hi2 calib --holdout` implementation assigns exact **prepared-text**
identities globally across its supplied sources, with deterministic SHA-256
ordering. Duplicate occurrences remain in one role even when repeated in other
sources; occurrences within that role are retained, not all removed. A requested
holdout needs at least two unique documents, retains at least one calibration
identity, and rejects an empty/tokenless or below-minimum holdout. For the
standard evaluation budget, request `--min-holdout-tokens 16384` with
`--tokenizer MODEL`; character estimates cannot establish a model-token minimum.
Inspect actual calibration/holdout counts, duplicate counts and realized token
shares in the report ([`calib.py`](../hi2/calib.py)).

This is an implemented correction to the earlier positional splitter, which
could produce empty holdouts or exact overlap across duplicate sources.
It isolates only identities in that invocation, not separate mixer invocations
or externally prepared files. Its equality rule is exact rendered text, not the
stronger optional whitespace/provenance normalization suggested above.
It cannot detect paraphrases, near-duplicates or different chunks of one original
document. Preserve the global split checks across your four roles.

An automatically produced holdout becomes **tuning** data if used to choose a
ratio. Keep the independent final corpus outside that invocation. Different
filenames, positional offsets and distinct JSON records alone are not evidence
of independence.

### Input formats and chat

* `calib` reads plain-text blocks and JSON/JSONL text or chat records (`messages`
  or `conversations`). Its normal chat representation is `role: content`, not
  automatically the model's chat template. Explicit `--chat-template` or
  `--chat-template-file PATH` requires `--tokenizer`; record the tokenizer/template
  identity from the preparation report. Do not silently change formatting between
  calibration, tuning and final testing.
* Direct `quantize --calib` and `eval --text` JSONL input expects a `text` field;
  it is not the same chat-record parser. Preprocess chat before these commands.
  Plain-text output from `calib` is the simplest common input.
* `--tokenizer MODEL` gives real token counts; without it `calib` estimates one
  token per four characters. Mixture weights are token-share goals under
  whole-document selection, not an assurance of exact proportions in every window.

## Allocation and KLD targets

For tensor `t`, row `r` and candidate `c`, let
`e[t,r,c] = sensitivity[t] * candidate_sse[t,r,c] / tensor_energy[t]`
(with the implementation's small-denominator guard). Define:

```text
E(picks) = sum over tensors and rows of e[t,r,picks[t,r]]
E_best   = sum over tensors and rows of min_c e[t,r,c]
quality constraint: E(picks) <= R * E_best
```

`--max-error-ratio R` searches for a low-bit allocation satisfying this **global**
weighted budget ([allocator](../hi2/allocate.py)). Individual rows can exceed R
times their own minimum while the sum meets the constraint. The ratio is
dimensionless and relative to the allowed candidates; changing the candidates
changes the floor. The tables are an output-error proxy, not a KLD measurement,
a per-row bound, or a guaranteed error bound after GPTQ/QAD.

The sensitivity probe normalizes measured relative KLD effects by their median.
If that median is zero but positive effects exist, it uses the positive-effects
median and keeps the configured floor for zero-signal units. An entirely
zero-signal probe keeps neutral unit weights rather than dividing by zero.
Zero measured sensitivity is not proof of losslessness.

`--target-kld K` measures PTQ candidates on tuning text and fits/interpolates a
ratio. The predicted KLD is not a measured score for the final checkpoint.
An out-of-range target may remain unbracketed after extensions; inspect the
reported tuning status and measured points, not just the selected ratio.
QAD and final logit-KL training can improve one corpus and regress another.
The historical [27B fast-recipe regression](qad.md) remains relevant.
A completed final-KL phase is not a final quality result.

Choose acceptance thresholds before inspecting the final corpus. Then measure
the **published final artifact** on that untouched text with `hi2 quality`.
The tuning prediction and internal training scores cannot replace this gate:

```bash
hi2 quality MODEL OUT --text final-test.txt \
  --max-kld "$MAX_KLD" --max-ppl-ratio "$MAX_PPL_RATIO" \
  --samples 8 --seq-len 2048 --device cpu
```

Set `MAX_KLD` and `MAX_PPL_RATIO` to your predeclared acceptance thresholds;
there is no universal recommendation hidden in this example. The gate reports
`target_met`, `target_not_met` or `not_measured`; a gate miss exits nonzero.
Only a measured `target_met` is acceptance. This is dense numerical evaluation,
not native serving validation. Use `quality` instead of a separate `eval` when
you want both a measured result and an explicit threshold decision.

Before scoring, `quality` verifies every recorded `hi2_run.json` calibration input
against its full SHA-256 identity, then checks exact document overlap with the
final corpus. Recorded entries must contain an absolute path and a valid full
SHA-256 digest. Missing files, changed bytes (even with unchanged size/timestamps),
or malformed identities stop evaluation without publishing a new quality report.
Replacing a calibration file with unrelated text cannot make its original
training documents eligible for final evaluation. Keep the original files
available, including when evaluating a relocated portable checkpoint.

Additional `--exclude-data` inputs remain part of the overlap check. Legacy
artifacts without recorded calibration provenance can only check explicitly
supplied exclusions; an empty checked-input list is not evidence that training
or tuning data were disjoint. `semantic_deduplication` remains `not_measured`.
If the original source fingerprint is absent, the report's
`reference.matches_quantization_source` is `null`, not a claim of a match.

The reference identity also includes `source.auxiliary_inputs`, notably the
external table selected by `HI2_NGRAM_MMAP`. `quality` compares the current
canonical path and full SHA-256 against the recorded list before evaluation.
Missing, changed, differently located, or unrecorded external inputs are rejected;
a symlink to the same canonical file is allowed. An absent list means no recorded
auxiliary inputs, so enabling a table for such an artifact is not silently accepted.
An exact match uses the existing dense evaluator's table attachment, and the
report records these identities in `reference.auxiliary_inputs`. Matching only
the model directory fingerprint is insufficient to establish the reference.

Missing, invalid, or nonfinite required measurements produce `not_measured`.
Nonfinite returned metric values are stored as JSON `null`, never replaced with
invented finite scores, and finite measurements are retained. A failed
measurement is published to `hi2_quality.json` before the strict gate raises
`QualityRejected`; the Python API's `strict=False` returns the same rejection report.
For portable outputs, publication also updates the portable manifest so the
rejected result remains structurally verifiable. This is a quality rejection,
not structural acceptance of its numerical result.

Thresholds are corpus/model-specific. `lossless: true` in `eval` means only
`kld_nats_per_token < 0.01` on the measured positions. It does not certify identical
weights, generations, tool choices, task accuracy, or native runtime behavior.

## Read the results

### Verification coverage

`hi2 verify MODEL OUT` reloads stored tensors and compares reconstructed weights
with the matching source. The default `--max-tensors 8` is a **sampled numerical**
check, not whole-model structural validation. The report records checked versus
expected tensor coverage and labels native execution as not run; zero checks or
an incomplete tensor-report mapping cannot establish success.

Complete structural validation is a separate operation implemented by
`hi2.checkpoint.validate_checkpoint`: it scans the shard/index mapping, dense
finite values, HI2 dtypes/shapes/offset bounds/widths/flags and finite decoded
ladders, for either MoE layout, with complete report coverage. The quantization
pipeline invokes it before publishing success. For an independent check:

```bash
hi2 validate OUT
```

Use this all-artifact structural check, not a larger numerical sample count as
a substitute. Structural validity still is not model quality or native loader
validation. A successful command is evidence only when it reports nonempty
expected coverage; retain the report.

Unweighted energy loss is `sum((decoded - source)^2) / sum(source^2)` for a
weight tensor. It is not activation-weighted allocation error, end-to-end KLD,
a task score or a native inference test. Reloaded bpw consistency likewise is
an accounting check, not a quality threshold. Source:
[`checkpoint.py`](../hi2/checkpoint.py), [`stack.py`](../hi2/stack.py).

### Size and bpw

```bash
hi2 size OUT --src MODEL --by kind --top 10
```

* **Allocation bpw** is the allocator's budget/accounting model over eligible
  weights; its compact metadata model is not every byte in the serialized file.
* **Stored linear bpw** is `8 * linears.bytes / linears.weights`, including codes,
  block parameters, row constants, smoothing and row/layout index arrays.
  Precision shares here count **weights**, not the report's fraction of rows.
* **Tensor bytes** include the quantized body and remaining dense tensors.
  Unmodified dense tensors are copied in source dtype; trained overrides may
  replace some norms. They are not universally fp16.
* **File bytes** include the safetensors files and their headers, not necessarily
  every tokenizer/auxiliary asset, cache or workdir. Whole-directory deployment
  footprint additionally includes those assets and any materialized links.
* **Dense export** explicitly writes fp16 dense weights and forfeits compressed
  runtime memory savings. This is distinct from copying source tensors into HI2.

Compare the same parameter set and units across formats. State whether an
embedding, head, vision tower or MTP module is quantized, dense, or excluded.
MiB = 2^20 bytes and GiB = 2^30 bytes: 16 GiB is 16,384 MiB, not 16,000 MiB.
Source: [`size.py`](../hi2/size.py) and [the corrected 27B comparison](results.md#the-quality-side-kld-against-bf16-on-the-same-windows).

### Numerical and task metrics

**PPL** is perplexity; the PPL ratio compares the quantized model with its
reference. **EAR** is the expected acceptance rate computed from distribution
overlap, not a measured serving acceptance rate. **nMSE** is normalized
mean-squared error.

| Metric/check | What it measures | What it does not establish |
| --- | --- | --- |
| PPL and PPL ratio | Next-token likelihood of the chosen text; ratio to the matching reference | Equal PPL does not imply equal distributions or decisions. A ratio below 1 is possible after quantization. |
| KLD | Mean `KL(reference || quantized)` in nats per scored token on that corpus | Not a universal task-accuracy threshold or generation-equivalence certificate. |
| Top-1 agreement | Fraction of evaluated contexts with the same highest-probability token | Not the fraction of full generated answers that will match. |
| EAR | Mean `sum_v min(p_v, q_v)` distribution overlap at evaluated contexts | Not measured speculative-serving acceptance or throughput; scheduling, draft cost and native kernels are absent. |
| Internal QAD nMSE/KL | Training/selection objective on internal validation windows | Not an independent final-corpus score. |
| Task accuracy and decision fidelity | Behavior on the named tasks/prompts and scoring rules | Not universal utility or proof that two sampled runs establish determinism. |
| Native loader plus inference request | The selected runtime actually loads and executes the low-bit checkpoint | Not established by a dense export or `hi2 eval`. Requires its own versioned evidence. |

A **signed log-probability gap** compares reference and candidate log-probabilities
for selected tokens. It can be negative and is not true KLD over the full token
distribution. Do not report such a gap as KLD or apply `--max-kld` thresholds
to it. Use the benchmark harness's metric definitions when comparing task reports
with dense `eval` results.

`eval` exports/dequantizes the checkpoint to dense fp16 HF weights and loads the
original architecture; `--dtype` controls evaluation math, not a native low-bit
kernel. Record model and checkpoint revisions, corpus/split and hash, tokenizer,
offset, scored positions, dtype/devices and evaluator revision. Compare repeated
runs with paired item-level intervals, noting that they do not model every source
of runtime variation. See [`evalq.py`](../hi2/evalq.py).

For task metrics use [bench/README.md](../bench/README.md): the checkout tooling
has task-specific endpoint requirements, optional datasets, incomplete-run
reporting and generated-code execution risks. A generic chat endpoint may not
provide `/tokenize` or prompt log-probabilities. Code-task execution is not a
security sandbox; read the safety guidance before enabling it.
