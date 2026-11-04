# First model: eligibility, quantization and validation

This is an ordered local-model workflow, not a claim that every command below has
been rerun in the currently recorded CPU environment. Historical model evidence
is separated from implementation eligibility. Begin with
[installation and its evidence boundary](installation.md); command defaults live
in [usage.md](usage.md).

## Supported inputs and evidence

Before allocating resources, establish all of the following:

* `SRC` is a **local Hugging Face directory** with `config.json`, tokenizer assets
  and either `model.safetensors` or a complete `model.safetensors.index.json`
  plus its shards. A remote model ID, GGUF, AWQ/GPTQ packed checkpoint, or a
  PyTorch `.bin` file is not this input contract.
* Eligible source tensors are floating-point 2-D `*.weight` matrices whose input
  width is divisible by 64, subject to include/exclude rules. Other tensors are
  copied rather than automatically quantized. Check the resulting tensor coverage.
* Calibrated PTQ/QAD and numerical evaluation must load through
  `AutoModelForCausalLM` and expose a recognized decoder `ModuleList`. The
  supported discovery paths are in [`layerwise.LAYER_PATHS`](../hi2/layerwise.py);
  matching a path is necessary, not proof that every model forward works.
* MoE adaptation can unfuse **in-memory** experts loaded from per-expert 2-D
  source weights. Fused 3-D expert weights **on disk** are not split by the writer.
  These are different cases; inspect actual tensor names/shapes, not just the
  model-family label. See [moe.md](moe.md).

The matrix uses **M** = published historical measurement for the stated case,
**I** = implementation exists but no full-model measurement supplied,
**U** = untested/no qualifying measurement here, and **X** = unsupported input
representation. M is evidence, not a promise that a new runtime or source
revision will behave identically. Exact HF source revisions were not recorded in
the linked model studies unless noted; that provenance gap is not filled in by
inference. “Native” means direct low-bit loader/kernel evidence, not dense export.

| Model / source revision | On-disk input | PTQ | QAD | Final-KL | Dense numerical eval | Native serving | Hardware/environment and evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B / source revision not recorded | Dense 2-D safetensors | M | U | U | M | M, PTQ | V620/gfx1030; [PTQ and fork speed/PPL](results.md#qwen25-05b-against-awq-fork-vllm-held-out-text-unquantized-ppl-1049). No exact reusable runtime/dependency lock. |
| Qwen3.5-0.8B / source revision not recorded | Dense 2-D safetensors, hybrid attention | M | M | M | M | U as a separately version-pinned native recipe; dense-export scores are not native proof | One/three V620 studies; [QAD](qad.md), [final-KL](lowbit.md), [PTQ](results.md#qwen35-08b-in-detail-unquantized-ppl-10874-docs-5483-mix). |
| Qwen3.5-4B / source revision not recorded | Dense 2-D safetensors, copied auxiliary tensors | M | M | M | M | M for published PTQ speed; no generally qualified QAD/runtime-version matrix | V620, TP=1/2/4 runtime measurements; [speed](results.md#speed-of-hi2-against-dense-qwen35-4b-tuned-full-cudagraphs-per-request), [quality](results.md#qwen35-4b-with-mixed-calibration-ptq-and-qad-vllm-harness-same-text-sets), [final-KL](lowbit.md). |
| Qwen3.8-27B / source revision not recorded | Dense 2-D safetensors; BF16 source, copied dense parts | M | M, including measured regressions | M for 2026-10-29 default run; earlier requested phase was skipped | M | M for the recorded fork/checkpoints, not a reproducible public runtime pin | Four V620s for later QAD/final-KL and dense eval; one-card quantized/four-card baseline serving; [dated results](results.md#qwen38-27b-at-30-bpw), [completed run](surfaces.md#the-default-27b-run-finished-2026-10-29). |
| allenai/OLMoE-1B-7B-0924 / exact HF commit not recorded | Per-expert 2-D source; stacked or per-expert HI2 output | M | U | U | M, both layouts | M, both layouts in the historical fork | One V620 quantization/serving; CPU fp32 numerical eval; [2026-10-26 recipe/layouts](moe.md), [native results](results.md#olmoe-1b-7b-unquantized-ppl-6634-docs-3924-mix-whole-file-1384-gb). |
| Qwen2-MoE / Mixtral, no specific revision qualified | Per-expert 2-D source naming/adaptation paths | I | U | U | U | U | [`moe.py`](../hi2/moe.py)/[`stack.py`](../hi2/stack.py); shared experts not qualified by the OLMoE study. |
| Any MoE with fused 3-D experts on disk | `experts.gate_up_proj` / equivalent 3-D tensors | X for expert quantization | U | U | U | U | Source-writer limitation; in-memory unfusing does not make this a supported input. |
| Arbitrary other architecture/revision/backend | Must first meet the input/discovery contract | I, generic path only | U | U | U | U | No blanket dense/MoE completeness claim. Qualify a small local run before a full one. |

The CPU suite also exercises tiny random Qwen2, Llama and OLMoE fixtures
([pipeline tests](../tests/test_pipeline.py), [MoE tests](../tests/test_moe.py)).
The 2026-11-04 CPU/MCP integration gate passed **941 tests**, with 61 skips
requiring GPU/HIP, multiple GPUs, GPU-pinned memory or unavailable Triton.
This is narrow implementation coverage, not real-model qualification.

## Executed no-download CPU smoke

This small workflow exercises the quantizer without fetching a model or selecting
a GPU. It is a randomly initialized Llama, **not a useful language model**.
Run it in an activated [installation](installation.md), in a directory without
an existing `cpu-smoke/`. Dependencies must already be installed; offline mode
below prevents Hub fetching, not package installation. Never use these two-step
training settings or toy quality thresholds as a production recipe.

In a POSIX shell, create the local model, tokenizer and data with:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python - <<'PY'
from pathlib import Path
import json
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

root = Path("cpu-smoke")
root.mkdir()
torch.manual_seed(112)
config = LlamaConfig(
    vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
    num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128,
    tie_word_embeddings=False,
)
LlamaForCausalLM(config).eval().save_pretrained(root / "model")
backend = Tokenizer(WordLevel(
    {"[UNK]": 0, **{f"word{i}": i + 1 for i in range(63)}}, unk_token="[UNK]",
))
backend.pre_tokenizer = Whitespace()
PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]").save_pretrained(root / "model")
documents = [{"text": " ".join(f"word{(j+i)%63}" for j in range(64))} for i in range(20)]
(root / "corpus.jsonl").write_text("".join(json.dumps(doc) + "\n" for doc in documents))
(root / "final.txt").write_text(" ".join(f"word{(j*7)%63}" for j in range(128)))
(root / "qad.json").write_text(json.dumps({
    "steps": 2, "budget": 1.0, "min_steps": 0, "eval_every": 1, "batch": 1,
    "samples": 2, "val": 1, "sampling": "head", "final_kl": 2,
    "final_batch": 1, "final_evals": 2, "dtype": "float32", "buffer_dtype": "float32",
}))
PY
```

Then run in the same shell. The subshell stops on any nonzero exit, so a failed
quality gate does not fall through to portable publication:

```bash
(
set -e
hi2 calib cpu-smoke/calib.txt cpu-smoke/corpus.jsonl \
  --tokenizer cpu-smoke/model --tokens 256 \
  --holdout cpu-smoke/holdout.txt --holdout-tokens 128 --min-holdout-tokens 64
hi2 quantize cpu-smoke/model cpu-smoke/out --workdir cpu-smoke/work \
  --cache-dir cpu-smoke/cache --devices cpu --calib cpu-smoke/calib.txt \
  --calib-samples 2 --seq-len 16 --sensitivity none --max-error-ratio 4 \
  --effort fast --rotation off --qad-config cpu-smoke/qad.json \
  --dry-run --json > cpu-smoke/plan.json
hi2 recipe save --from-plan cpu-smoke/plan.json --out cpu-smoke/recipe.json
hi2 recipe run cpu-smoke/recipe.json --log-format jsonl
hi2 status cpu-smoke/work
hi2 validate cpu-smoke/out
hi2 quality cpu-smoke/model cpu-smoke/out --text cpu-smoke/final.txt \
  --samples 2 --seq-len 32 --device cpu --dtype float32 \
  --max-kld 0.01 --max-ppl-ratio 1.2
hi2 export-portable cpu-smoke/out cpu-smoke/portable --assets-from cpu-smoke/model
hi2 validate cpu-smoke/portable
hi2 recipe run cpu-smoke/recipe.json
)
```

Recorded observations from executing these blocks on **2026-11-04**:
seven quantized linears; calibration, QAD, final-KL, encoding and validation
completed; final quality `target_met`; portable structural validation passed;
replay reused the compatible completed run.
The work manifest records layer/step progress and separates final-KL from
layer-wise QAD. `hi2 validate` establishes structural validity, not quality
acceptance; inspect the explicit quality result before distributing a bundle.

Keep `cpu-smoke/model`, the calibration text and the work/cache paths for replay
or later quality checks. Portable export retains run/quality evidence but does
not bundle the recorded calibration corpus. Do not move the source just to test
portability: ordinary output asset symlinks can still depend on it. Changing a
recipe while retaining its work/output paths is an incompatible-resume error,
not permission to overwrite them. See [safe recovery](operations.md#resume-and-invalidation).
Native inference and GPU fit remain unmeasured.

## First-model walkthrough

Use a locally available **Qwen3.5-0.8B safetensors checkpoint** for this example:
it has published PTQ/QAD evidence, but a new source/dependency combination still
needs the smoke below. Do not download or launch a 27B model just to test the CLI.
Record the model's exact source revision and tokenizer provenance when acquiring
it; the historical matrix cannot supply missing revision information for you.

### 1. Install and choose explicit paths/devices

Complete the [installation smoke](installation.md#first-use-compilation-and-smoke),
then set the model path in your activated environment:

```bash
MODEL=/absolute/path/to/Qwen3.5-0.8B
DEVICES=cpu
```

`MODEL` must exist with the assets listed above. `cpu` avoids accidental use of
all visible GPUs; it does **not** make a full QAD run cheap. On a deliberately
selected, separately qualified GPU installation, set `DEVICES=cuda:0` (ROCm uses
that spelling too). Read [phase resource costs](operations.md#phase-resource-map)
before the full recipe. Use new output/work directories for each recipe; none
may overlap the source or data paths.

### 2. Prepare independent documents, then mix

Prepare `train-docs.jsonl`, `train-chat.jsonl`, `tuning-docs.jsonl` and
`final-docs.jsonl` from your domain, with document identities disjoint across
training, tuning and final roles. Each record should have a nonempty `text`
field; render chat deliberately before this step if template fidelity matters.
Perform and save the [global exact-overlap check](evaluation.md#safe-preparation-and-exact-overlap)
**before** interleaving. The final documents must never be passed to tuning or QAD.

```bash
hi2 calib calib.txt train-docs.jsonl:2 train-chat.jsonl:1 \
  --tokenizer "$MODEL" --tokens 540672
hi2 calib tuning.txt tuning-docs.jsonl --tokenizer "$MODEL" --tokens 32768
hi2 calib final-test.txt final-docs.jsonl --tokenizer "$MODEL" --tokens 32768
```

Success means three nonempty text files with the intended domains and sufficient
actual tokenizer counts: at least 540,672 calibration tokens for the full default
recipe, and at least 16,384 usable tokens in each evaluation file. Inspect the
reported counts/warnings; requested targets alone are not proof. The extra
requested evaluation tokens provide margin, not a disjointness guarantee.
`tuning.txt` is only needed if selecting settings by repeated evaluation or
`--target-kld`; it is not the final test. A mixer-generated `--holdout` file is
another preparation option, but it cannot serve both tuning and untouched testing.

### 3. Plan and run a deliberately small PTQ compatibility smoke

```bash
hi2 quantize "$MODEL" out/smoke --workdir work/smoke --devices "$DEVICES" \
  --calib calib.txt --calib-samples 2 --seq-len 64 --limit-layers 1 \
  --sensitivity none --max-error-ratio 4 --no-qad --dry-run
hi2 quantize "$MODEL" out/smoke --workdir work/smoke --devices "$DEVICES" \
  --calib calib.txt --calib-samples 2 --seq-len 64 --limit-layers 1 \
  --sensitivity none --max-error-ratio 4 --no-qad
hi2 verify "$MODEL" out/smoke --max-tensors 8
hi2 size out/smoke --src "$MODEL"
```

Expected plan: the selected explicit device, 128 required calibration tokens,
one decoder layer, no QAD/final-KL, no sensitivity probe, and the separate
`out/smoke`/`work/smoke` locations. Planning does not prove a model forward,
free-memory sufficiency or native compatibility.

Expected execution artifacts: HI2 safetensors shards/index, `config.json` with
`quant_method: rdna_hint_flex`, `hi2_report.json`, a workdir run record, and a
successful `hi2_run.json` after structural validation. The numerical sample must
check at least one tensor; the size report must show quantized linears. This is a
**partially quantized compatibility artifact**, not a production-quality model
or a calibration/quality result. Do not reuse its work/output paths for the full
recipe or cite its short-input score as final quality.

### 4. Plan and run the full quality recipe

```bash
hi2 quantize "$MODEL" out/model --workdir work/model --devices "$DEVICES" \
  --calib calib.txt --calib-samples 128 --seq-len 2048 \
  --max-error-ratio 6 --sensitivity probe --qad-preset default --dry-run
hi2 quantize "$MODEL" out/model --workdir work/model --devices "$DEVICES" \
  --calib calib.txt --calib-samples 128 --seq-len 2048 \
  --max-error-ratio 6 --sensitivity probe --qad-preset default
```

This explicitly selects calibrated PTQ, every eligible decoder layer, the
sensitivity probe, default layer-wise QAD and its requested 200-step final-KL
phase. Expected calibration requirement is 540,672 tokens. Check whether
final-KL **completed or was skipped**, not only whether output was written.
The layerwise time estimate is not total runtime. To choose PTQ only instead,
use `--no-qad` with fresh paths; do not silently call that the default QAD recipe.

The allocation target is a global weighted error ratio, not a bound on every
row or a promise of end-to-end KLD. If choosing a ratio with `--target-kld`, use
`tuning.txt`, an explicit evaluation device, and the [prediction-versus-measurement
rules](evaluation.md#allocation-and-kld-targets); keep `final-test.txt` untouched.

### 5. Validate the completed artifact

```bash
hi2 validate out/model
hi2 verify "$MODEL" out/model --max-tensors 8
hi2 size out/model --src "$MODEL" --by kind --top 10
hi2 eval "$MODEL" out/model --text final-test.txt --skip-tokens 0 \
  --device "$DEVICES" --dtype float32
```

Quantization runs complete structural validation before its success record;
`validate` repeats that check independently, including for an existing/moved
artifact. `verify --max-tensors 8` remains sampled numerical reconstruction,
not an all-shard integrity gate. Record checked/expected coverage rather than
treating sample success as a full quality certificate.

The size report separates quantized-body bytes from copied dense tensors. The
numerical report should contain finite PPL/KLD/top-1/EAR results and the scored
position count (16,376 for this default evaluation). Compare KLD with a
**preselected corpus-specific threshold**, not an assumed guaranteed value.
If using the result to retune, it is no longer untouched testing: reserve another
final corpus. `eval` needs dense-model RAM/disk even though the checkpoint is
compressed. See [metric boundaries](evaluation.md#read-the-results).

For an explicit final-quality gate, use `hi2 quality` **instead of** the `eval`
command after setting `MAX_KLD` and `MAX_PPL_RATIO` to your preselected thresholds:

```bash
hi2 quality "$MODEL" out/model --text final-test.txt \
  --max-kld "$MAX_KLD" --max-ppl-ratio "$MAX_PPL_RATIO" \
  --samples 8 --seq-len 2048 --device "$DEVICES"
```

Success requires a measured `target_met`; `target_not_met` or `not_measured` is
not acceptance, and a gate miss exits nonzero. This evaluates the final artifact,
unlike PTQ target interpolation, but still does not execute native HI2 kernels.

## Export and native serving

For a compatible dense-HF engine, with a **new** destination and adequate memory:

```bash
hi2 export-dense "$MODEL" out/model out/model-dense
```

Expected artifacts include dense fp16 `model.safetensors` and `config.json`
without HI2 quantization metadata. This output is not GGUF; llama.cpp requires
a compatible separate HF-to-GGUF conversion. It also does not retain HI2 runtime
memory/speed advantages. Ordinary exports can link source assets; use the
[portable distribution guidance](operations.md#distribute-a-checkpoint) before
moving or deleting the source.

For a self-contained low-bit distribution rather than dense conversion:

```bash
hi2 export-portable out/model out/model-portable --assets-from "$MODEL"
hi2 validate out/model-portable
```

`out/model-portable` must be new. Success means unchanged quantized shards,
materialized allowed local assets and `hi2_portable_manifest.json` hashes;
`--assets-from` explicitly trusts the local source root for linked assets.
It retains source-owned run/quality evidence but not the recorded calibration
corpora; keep those external inputs for later quality checks. Do not copy or
publish the destination while export is still running: a hard kill can leave an
incomplete directory. See [publication caveats](operations.md#distribute-a-checkpoint).
This is a portability check, not a native-runtime qualification.

**Native guide qualification remains open.** Published fork results demonstrate
that specific native runs occurred; the historical handover's registration
failure does not prove native serving is universally broken. Conversely, neither
those results nor `hi2 eval` establish that stock vLLM or an unnamed fork revision
will load your checkpoint. No runtime revision or GPU validation is invented here.
A qualified native guide needs all of:

1. An obtainable runtime URL and exact commit, with installation dependencies.
2. The accepted HI2 version, layout, dtype, model/source revision and checkpoint
   provenance, including any assets copied or converted.
3. A loader invocation and log showing the `rdna_hint_flex` registration and
   selected low-bit path, followed by one recorded successful inference request.
4. Hardware, runtime/checkpoint hashes, observed outputs and limitations, tested
   separately from dense export/numerical evaluation.

Until that evidence is supplied, use the historical [runtime result tables](results.md)
as evidence only, not a reproducible installation recipe. Once you have a
qualified server, [bench/README.md](../bench/README.md) describes task evaluation,
API requirements, incomplete-run handling and code-execution risks.
