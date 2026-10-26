# HI2 quantization-aware distillation (QAD)

Tool-side extension of the HI2 v3.2 format (`format-spec.md`, section 13).
QAD changes which codes and block parameters are stored, never the format: a QAD
checkpoint is a normal `rdna_hint_flex` v3.2 checkpoint that `hi2 eval`,
`hi2 export-dense` and the fork's loader read unchanged.

## Quick start

QAD is **on by default** whenever calibration text is given:

```bash
cd .
# HF model -> QAD checkpoint (quality allocation, sensitivity probe, PTQ, then QAD)
python -m hi2 quantize <hf_model_dir> <out_dir> --calib calib.txt --max-error-ratio 6
# PTQ only
python -m hi2 quantize <hf_model_dir> <out_dir> --calib calib.txt --no-qad
# check inputs and print the plan (tokens, RAM, layers) without doing anything
python -m hi2 quantize <hf_model_dir> <out_dir> --calib calib.txt --dry-run
# score it against the unquantized model on held-out text
python -m hi2 eval <hf_model_dir> <out_dir> --text heldout.txt --device cuda:0
# dense fp16 copy for another engine (llama.cpp, the vLLM multiple-choice harness)
python -m hi2 export-dense <hf_model_dir> <out_dir> <dense_dir>
```

At the start the run logs that QAD is on, a rough estimate of its extra time on the
chosen device, and that `--no-qad` turns it off. A run without `--calib` (data-free)
cannot distill: it logs a warning and writes a PTQ checkpoint. `--max-error-ratio 6`
is the recommended size for QAD (see "Recommendation for the default ratio"); without it
the quantizer's `DEFAULT_ERROR_RATIO` applies.

Expected cost on one Radeon PRO V620 (gfx1030), model in RAM:

| Model | Phase 1 + sensitivity | QAD `default` | QAD `fast` | GPU peak (QAD) | RAM (QAD process) |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-0.8B (24 layers) | 8-13 min | 41 min (about 1.7 min per layer) | 17 min | 1.4 GiB | about 15 GiB |
| Qwen3.5-4B (32 layers) | 50 min | 3.1 h (about 5.5 min per layer) | about 1.3 h (estimate) | 6.0 GiB | about 46 GiB |

The first trained layer of a run also pays the one-time Triton autotune of the
linear-attention backward kernels (about 10 minutes on this box, cached afterwards).
`experiments/run_hi2_model.sh` and `run_hi2_moe.sh` now pass `--no-qad` so the PTQ
experiment scripts keep their meaning.

## What it does

The PTQ pipeline runs first, unchanged: phase 1 (statistics, smoothing vectors, rotation
flags, candidate tables, Hessians), the sensitivity probe, the allocation (quality mode
`--max-error-ratio` by default, or `--target-bpw`) and, per layer, the Hessian-aware
encode. QAD then trains the decoder layers one at a time, in order:

* **Inputs and targets.** Student input of layer `L` = the hidden states produced by the
  already quantized layers `0..L-1`; target = the unquantized layer `L` applied to the
  unquantized prefix's hidden states. Both live in RAM; one layer is on the GPU. Training
  against the teacher path asks each layer to absorb the error its prefix left.
* **Trainable, per Linear:** fp32 latent weights in the stored basis `H (W s)` and the
  per-block ladder bytes `(lo_u8, step_u8)` as continuous values. **Frozen:** every row's
  `(nbits, B)` from the PTQ allocation, the per-row fp16 constants
  `(lo_min, lo_scale, step_scale)`, the smoothing vector `s` and the Hadamard flags.
* **Forward = the exact decode of spec 2.2.** Bytes are rounded and clamped to 0..255 and
  decoded with the same `decode_params` as the reader; codes are
  `clamp(round((v - lo) / step))`; the stored value is `fp16(lo + code * step)` computed
  in float64 with one rounding. Straight-through estimators carry the gradient (identity
  through the byte rounding, LSQ-style through the code rounding: the gradient reaches
  the latent inside the ladder, `step` through `code - z`, `lo` through clamped weights).
  The stored-basis weight is mapped back, `W = H^-1(w) / s`, so the layer runs in its
  original basis.
* **Initialisation (`init=weights`, default):** the latents start at the unquantized
  weights (spec 13) and the bytes at the PTQ bytes, so step 0 is nearest rounding on the
  PTQ ladders. `init=ptq` starts the latents at the GPTQ-compensated weights (the values
  each code was rounded from), so step 0 reproduces the PTQ checkpoint bit for bit. On
  0.8B the two are within seed noise; on 4B `init=weights` gained more per layer.
* **Latent units (`latent_units=step`, default):** each latent is stored divided by its
  block's initial quantization step, so `lr_w = 0.01` means 1 % of a step per update on
  any model. With absolute latents the best learning rate on 0.8B (1e-4) is 1 % of an
  INT2 step there but 2.7 % of an INT4 step on 4B, and the first 4B layers over-stepped
  and kept the PTQ encoding.
* **Loss and optimizer:** normalised MSE of the layer output against the target, AdamW
  (no weight decay) with separate learning rates for latents and bytes, linear warm-up
  then cosine decay, gradient accumulation over `batch` sequences per step.
* **Keep-best:** the loss on held-out calibration sequences (spread over the set) is
  measured before training (for both the starting state and the PTQ encoding) and at
  every evaluation; the layer keeps the best of the trained states and the PTQ encoding,
  so QAD never makes a layer worse on held-out calibration data than PTQ did.
* **Step budget and early stop.** A run may spend `budget x steps x layers` optimizer
  steps. Each layer gets the remaining budget per remaining layer, scaled by how much the
  previous layer gained relative to the mean gain so far (0.5x to 2x, the first layer
  2x), between `min_steps` and `steps`; a layer stops when one evaluation interval
  (`eval_every` steps) improves its held-out nMSE by less than `min_gain`, and the
  unspent steps go to later layers. The 4B logs motivated it: early layers gain about
  10-15 %, the last ones 2-3 %. Steps spent per layer and the time saved are logged.
* **Export:** the codes and bytes of the kept state, assembled through the existing
  writer. `dequantize_stored` of the exported tensor equals the forward's stored values
  bit for bit (tested), and the written shards equal the per-layer artifacts (tested).

Calibration data for QAD is a separate draw from the PTQ head: `samples + val` random
non-overlapping `seq_len` windows from the whole calibration text (seeded), so every
domain of a concatenated file is represented. Without calibration text the run falls
back to PTQ with a warning; too little text fails before phase 1 with a message naming
the option to change.

## Interface

| Flag | Meaning |
| --- | --- |
| `--qad` / `--no-qad` | QAD is on by default when `--calib` is given (and implies `--sensitivity probe` unless given); `--no-qad` = PTQ only |
| `--qad-preset fast\|default\|thorough` | steps, budget and sequences per layer (table below) |
| `--qad-budget F` | fraction of `steps x layers` to spend (preset default) |
| `--qad-samples N` | training sequences of `--seq-len` tokens |
| `--qad-layers "0-5,9"` | train only these layers (the others keep PTQ) |
| `--qad-config file.json` | any field of `hi2.qad.QadConfig` (advanced) |
| `--limit-layers N` | smoke test: quantize only the first N layers |
| `--dry-run` | validate and print the plan, write nothing |
| `--devices auto` | default: every visible GPU (narrow with `HIP_VISIBLE_DEVICES`), else CPU |

Hidden flags for quick experiments: `--qad-steps`, `--qad-lr` (latents),
`--qad-lr-params` (bytes, in levels per step), `--qad-batch`, `--qad-init ptq|weights`.

`QadConfig` fields: `steps`, `budget`, `min_gain`, `min_steps`, `eval_every`, `batch`,
`lr_w`, `lr_p`, `samples`, `val`, `init`
(`weights` or `ptq`), `latent_units` (`abs`, or `step`: latents in units of their
block's quantization step, so `lr_w` is a fraction of a step per update), `layers`,
`warmup`, `keep_best`, `buffer_dtype` (`auto` stores hidden states in bf16 when fp32
does not fit in half the free RAM), `seed`, `evals`, `sampling` (`random`, `balanced` =
equal windows from each calibration file, or `head`). Example `--qad-config` file:
`{"sampling": "balanced", "budget": 0.5}`.

| Preset | Steps | Training sequences | 0.8B QAD time | 0.8B result (docs / mix ppl ratio) |
| --- | --- | --- | --- | --- |
| `fast` | budget 0.5 of 300 x layers, early stop under 1 % per 50 steps (min 75) | 256 | 17 min | 1.085 / 1.330 |
| `default` | 300 per layer | 256 | 41 min | 1.061 / 1.286 |
| `thorough` | 600 per layer, 8 sequences per step | 512 | 141 min | 1.047 / 1.267 (absolute latents) |

Other defaults: batch 4 sequences per step, 8 held-out sequences, `lr_w` 0.01 (step
units), `lr_p` 0.05 bytes per step, warm-up 5 % then cosine over each layer's allotment,
`init=weights`, `sampling=random`, seed 0.

**Devices.** Training runs on the first device; with several devices the teacher and
student propagation passes split the sequences across them (training itself is not
data-parallel). The model stays in RAM in every mode.

**Resume.** Every finished layer is written to `<workdir>/qad/layerNNN.pt` with a
fingerprint of everything it depends on (phase-1 settings, allocation, QAD settings);
re-running the same command reuses phase 1, the sensitivity probe and the finished
layers, and continues. A changed setting invalidates exactly what depends on it.

**Progress.** Each layer logs its held-out loss before and after (against the PTQ
encoding), whether it kept the trained state, the steps it spent out of its allotment
(and whether it stopped early), its time, the elapsed time and an ETA; the run ends
with the steps spent out of the nominal `steps x layers` and the time that saved;
`experiments/qad_layers.py <workdir>` prints the per-layer table from the artifacts.

## Measurements

Protocol (spec 12.1): HF evaluator (`hi2 eval`), 8 x 2048 held-out tokens, against the
unquantized model's logits. Two held-out texts: the repo docs (`study_text_all.txt`,
skip 800 000 tokens; unquantized perplexity 10.874 for 0.8B, 7.067 for 4B) and the
mixed docs/chat/tool-calling text (`eval_mix.txt`, skip 0; unquantized perplexity
5.483 for 0.8B, 3.731 for 4B).
"ppl ratio" = quantized / unquantized perplexity, KLD in nats per token, top-1 =
argmax agreement. One seed per cell unless noted; on 0.8B two seeds of the same
configuration differ by about 0.01 in ppl ratio and 0.01 in mix KLD, so smaller
differences are noise.

**Calibration.** `calib_qad.txt` = the fixed mixed calibration file (`calib_mix.txt`,
335k tokens: docs, chat/instruction, tool calling) followed by the docs calibration
region (the first 790k tokens of `study_text_all.txt`, disjoint from its held-out
region): 1.13M tokens. The PTQ phase reads its head (128 x 2048 tokens, all inside
the mixed part, so it is the same data as the user's `q35-o2-mix` runs); QAD draws
264 random non-overlapping 2048-token windows from the whole file (seed 0), so about
70 % of the QAD tokens are docs and 30 % mixed. Both held-out texts are therefore
in-domain for QAD, the docs text more so; the multiple-choice tasks below are the
out-of-domain check. Built with `experiments/make_calib_qad.py`.

### Error ratio and size (quality allocation, sensitivity probe, same calibration)

| `--max-error-ratio` | 0.8B bpw | 0.8B INT2 / INT4 rows | 4B bpw | 4B INT2 / INT4 rows |
| --- | --- | --- | --- | --- |
| 2 | 4.171 | 12.6 / 87.4 % | | |
| 3 | 3.892 | 27.1 / 72.9 % | | |
| 4 (current `DEFAULT_ERROR_RATIO`) | 3.703 | 37.4 / 62.6 % | 3.799 | 29.1 / 70.9 % |
| 5 | 3.557 | 45.2 / 54.8 % | 3.652 | 36.8 / 63.2 % |
| 5.5 | 3.494 | 48.5 / 51.5 % | 3.586 | 40.1 / 59.9 % |
| 6 | 3.435 | 51.4 / 48.6 % | 3.523 | 43.2 / 56.8 % |
| 6.2 | | | 3.499 | 44.4 / 55.6 % |
| 7 | | | 3.405 | 48.8 / 51.2 % |
| 8 | 3.229 | 61.4 / 38.6 % | 3.297 | 54.1 / 45.9 % |
| 10 | 3.057 | 69.3 / 30.7 % | 3.105 | 63.1 / 36.9 % |
| 10.75 | 2.999 | 71.9 / 28.1 % | 3.042 | 65.9 / 34.1 % |
| 11 | 2.980 | 72.7 / 27.3 % | | |
| 11.3 | | | 2.998 | 67.9 / 32.1 % |
| 12 | 2.910 | 75.8 / 24.2 % | 2.946 | 70.0 / 30.0 % |

(`experiments/ratio_table.py <workdir> <ratios>` prints this from a finished phase 1.)
The QAD runs below use 10.75 (0.8B, 3.0 bpw), 11.3 (4B, 3.0 bpw) and 6.2 (4B, 3.5 bpw);
every PTQ comparison row uses the same ratio, tables and sensitivity, so the row
picks are identical and only the codes and block bytes differ.

### Qwen3.5-0.8B at 3.0 bpw: tuning

All rows: ratio 10.75 (2.998 bpw, 71.9 % INT2 rows), default preset unless noted.

| Run | docs ppl ratio | docs KLD | docs top-1 | mix ppl ratio | mix KLD | mix top-1 | wall time (phase 1 reused) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PTQ (same allocation) | 1.268 | 0.234 | 78.6 % | 1.509 | 0.364 | 73.9 % | 5 min |
| PTQ, user's docs-calibrated `q35-b3.0-sens` (`--target-bpw 3.0`) | 1.207 | 0.196 | 80.3 % | 2.049 | 0.650 | 66.5 % | |
| QAD init=ptq, lr 1e-4 | 1.060 | 0.073 | 87.8 % | 1.299 | 0.235 | 80.1 % | 46 min |
| QAD init=weights, lr 1e-4 (absolute latents; the default when run) | **1.054** | 0.073 | 87.4 % | **1.277** | **0.224** | 80.2 % | 47 min |
| same, seed 1 (PTQ fallback on) | 1.065 | 0.074 | 87.5 % | 1.296 | 0.235 | 79.6 % | 47 min |
| lr 3e-5 | 1.083 | 0.087 | 86.6 % | 1.309 | 0.239 | 79.4 % | 47 min |
| lr 3e-4 | 1.054 | 0.080 | 87.2 % | 1.522 | 0.385 | 75.5 % | 46 min |
| 512 training sequences | 1.061 | 0.073 | 88.0 % | 1.300 | 0.238 | 79.8 % | 52 min |
| 800 steps | 1.052 | 0.067 | 88.1 % | 1.318 | 0.258 | 78.8 % | 106 min |
| balanced sampling (half docs, half mixed) | 1.069 | 0.083 | 86.9 % | 1.268 | **0.216** | 80.6 % | 53 min (phase 1 recomputed) |
| uniform 100 steps, 128 seqs (old `fast`) | 1.091 | 0.103 | 85.5 % | 1.352 | 0.264 | 78.5 % | 21 min |
| latents in step units, lr 0.005 | 1.081 | 0.088 | 86.2 % | 1.284 | 0.229 | 80.0 % | 49 min |
| latents in step units, lr 0.01 (**current default**) | 1.061 | 0.077 | 87.5 % | 1.286 | 0.228 | 79.8 % | 48 min |
| latents in step units, lr 0.02 | 1.058 | 0.075 | 87.7 % | 1.322 | 0.250 | 78.8 % | 48 min |
| 600 steps x 8 seqs, 512 seqs (`thorough`, absolute latents) | **1.047** | **0.063** | 88.2 % | **1.267** | 0.223 | 80.0 % | 146 min |

Other sizes (same calibration and probe, default preset):

| Run | bpw | docs ppl ratio | docs KLD | docs top-1 | mix ppl ratio | mix KLD | mix top-1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PTQ, ratio 6 | 3.435 | 1.153 | 0.138 | 82.8 % | 1.265 | 0.199 | 80.7 % |
| **QAD, ratio 6** | 3.435 | **1.030** | **0.046** | **90.0 %** | **1.155** | **0.127** | **85.0 %** |
| PTQ, ratio 4 (current default) | 3.703 | 1.096 | 0.093 | 85.7 % | 1.171 | 0.130 | 83.9 % |

QAD at ratio 6 beats PTQ at ratio 4 on both texts with 0.27 bpw less.

What the tuning says:

* **QAD cuts the KL at fixed size by 69 % on the docs text and 38 % on the mixed text**
  (0.234 to 0.073, 0.364 to 0.224); the perplexity excess falls from 26.8 % to 5.4 % on
  docs and from 50.9 % to 27.7 % on the mixed text. That is far beyond the 3 % bar of
  spec 8.10. On docs, 3.0 bpw QAD beats 3.7 bpw PTQ; on the mixed text it does not.
* **It adapts to its training domain.** A larger latent learning rate (3e-4) or more
  steps (800) keep improving the docs text and make the mixed text worse, at 3e-4
  worse than PTQ; balanced sampling moves the gain from docs to the mixed text. 300
  steps at 1e-4 (absolute) or 0.01 (step units) is the point where both texts improve. Calibrate QAD on
  text from the domain the model will serve, and pass several files with
  `sampling=balanced` to weight domains equally.
* More data (512 sequences) changes nothing at 300 steps; the starting point
  (`init=weights` or `init=ptq`) is within seed noise on 0.8B.

### Qwen3.5-0.8B at 3.0 bpw: step budget and early stopping

Default settings (step-unit latents), ratio 10.75; "QAD time" is the sum of the per-layer
training times from the artifacts (wall times were inflated by other users' jobs on the
same GPUs). The full run reproduces the earlier step-unit run to four digits.

| Run | Steps spent | Early-stopped layers | QAD time | docs ppl ratio | docs KLD | docs top-1 | mix ppl ratio | mix KLD | mix top-1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full, 300 per layer (`default`) | 7200 (100 %) | 0 | 40.7 min | **1.061** | **0.077** | 87.5 % | **1.286** | **0.228** | 79.8 % |
| budget 0.75, early stop 0.5 % / 50 steps | 4554 (63 %) | 4 | 23.4 min | 1.077 | 0.088 | 86.4 % | 1.296 | 0.236 | 79.9 % |
| early stop only (0.5 % / 50 steps, min 150) | 5050 (70 %) | 15 | 25.9 min | 1.071 | 0.087 | 86.6 % | 1.313 | 0.247 | 79.2 % |
| budget 0.5, early stop 1 % (`fast`) | 3093 (43 %) | 4 | 17.4 min | 1.085 | 0.099 | 85.4 % | 1.330 | 0.254 | 79.2 % |
| budget 0.35, early stop 1 % | 2264 (31 %) | 2 | about 12 min | 1.101 | 0.113 | 84.8 % | 1.356 | 0.268 | 78.5 % |
| uniform 100 steps (old `fast`, absolute latents, 128 seqs) | 2400 (33 %) | 0 | 16 min | 1.091 | 0.103 | 85.5 % | 1.352 | 0.264 | 78.5 % |

No reduced budget stays within seed noise (about 0.01 in ppl ratio) of the full run on
both texts: the best trade, budget 0.75, saves 43 % of the QAD time for +0.016 docs and
+0.010 mix ppl ratio. On this model the late layers' small per-layer gains (3-8 %) still
add up at the output, so `default` keeps the full 300 steps per layer and the budget
powers `fast`, where it beats uniformly fewer steps at the same time. On 4B, where the
last layers gain only 2-3 %, a budget may cost less; that is not measured.

### Qwen3.5-4B: fidelity against PTQ

Same calibration, quality allocation and sensitivity probe for every new row; the
user's earlier `q4b-b3.x-sens` checkpoints (docs-only calibration, `--target-bpw`)
are kept for comparison.

| Checkpoint | bpw | INT2 rows | docs ppl ratio | docs KLD | docs top-1 | mix ppl ratio | mix KLD | mix top-1 | quantize time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PTQ `q4b-ptq-b3.0` (ratio 11.3) | 2.998 | 67.9 % | 1.138 | 0.161 | 82.9 % | 1.189 | 0.226 | 81.8 % | 17 min + 50 min phase 1 |
| **QAD `q4b-b3.0-qad`** (ratio 11.3) | 2.998 | 67.9 % | **1.030** | **0.061** | **89.6 %** | **1.165** | **0.195** | **84.3 %** | 3.2 h + 50 min phase 1 |
| QAD, step-unit latents (lr 0.01, now the default) `q4b-b3.0-qad-su` | 2.998 | 67.9 % | 1.034 | 0.068 | 89.4 % | 1.146 | 0.177 | 84.4 % | 3.0 h + 50 min phase 1 |
| PTQ (user) `q4b-b3.0-sens` | 3.000 | | 1.103 | 0.131 | 84.4 % | 1.388 | 0.395 | 76.3 % | |
| PTQ `q4b-ptq-b3.5` (ratio 6.2) | 3.499 | 44.4 % | 1.075 | 0.093 | 86.9 % | 1.087 | 0.117 | 86.7 % | 17 min + 50 min phase 1 |
| **QAD `q4b-b3.5-qad`** (ratio 6.2) | 3.499 | 44.4 % | **1.017** | **0.042** | **91.4 %** | **1.072** | **0.103** | **88.0 %** | 3.3 h + 50 min phase 1 |
| QAD, step-unit latents (lr 0.01, now the default) `q4b-b3.5-qad-su` | 3.499 | 44.4 % | 1.022 | 0.045 | 91.2 % | 1.068 | 0.093 | 88.6 % | 3.1 h + 50 min phase 1 |
| PTQ (user) `q4b-b3.5-sens` | 3.500 | | 1.054 | 0.077 | 88.0 % | 1.178 | 0.212 | 82.6 % | |
| PTQ (user) `q4b-b4.0-sens`, for scale | 4.000 | | 1.025 | 0.034 | 92.2 % | | | | |

`q4b-b3.0-qad` and `q4b-b3.5-qad` were trained with absolute latents (lr 1e-4), the
default when they were run; the `-su` rows use the step-unit latents that are now the
default. Step units move part of the gain from the docs text (the larger share of the QAD
data) to the mixed text: mix KL -9 % / -10 %, docs KL +11 % / +8 %, the sum over both
texts lower at both sizes, and the task accuracy within noise of the absolute runs.

QAD removes 78 % of the docs perplexity excess at 3.0 bpw (13.8 % to 3.0 %) and 77 % at
3.5 bpw (7.5 % to 1.7 %), and cuts the docs KL by 62 % and 55 %. On the mixed text the
gain is smaller (KL -14 % and -12 %), as on 0.8B. QAD at 3.5 bpw is close to PTQ at
4.0 bpw on the docs text (KL 0.042 against 0.034) at 0.5 bpw less.

Per layer on 4B, the held-out layer-output error falls by 2 % to 15 % (larger in the
first half of the network); with the absolute latent learning rate the first layers
(0, 2 at 3.0 bpw; 0, 7 at 3.5 bpw) could not beat the PTQ encoding and kept it.

### Qwen3.5-4B: task accuracy (vLLM harness, `vllm_mc.sh`, `acc`)

Dense fp16 export of each checkpoint (`hi2 export-dense`), served by the fork's vLLM
(`experiments/chain_mc4b_hi2.sh`). n = 570 / 299 / 2000 / 2000; one standard error is
about 1.9 / 2.8 / 1.1 / 1.1 points, differences under two standard errors are not
claimed.

| Checkpoint | ARC-E | ARC-C | HellaSwag | MMLU |
| --- | --- | --- | --- | --- |
| unquantized | 72.3 | 42.5 | 53.2 | 36.5 |
| PTQ `q4b-ptq-b3.0` | 67.7 | 37.1 | 45.1 | 32.6 |
| **QAD `q4b-b3.0-qad`** | 69.1 | 41.5 | 46.5 | 33.1 |
| PTQ (user) `q4b-b3.0-sens` | 67.7 | 36.5 | 44.6 | 32.1 |
| PTQ `q4b-ptq-b3.5` | 68.4 | 40.5 | 48.6 | 35.4 |
| **QAD `q4b-b3.5-qad`** | 70.4 | 41.1 | 49.0 | 34.6 |
| PTQ (user) `q4b-b3.5-sens` | 71.6 | 43.5 | 48.0 | 34.0 |
| EXL3 3.0 bpw (user) | 69.6 | 42.1 | 51.3 | 35.9 |
| QAD step-unit latents `q4b-b3.0-qad-su` | 68.4 | 41.5 | 46.9 | 33.1 |
| QAD step-unit latents `q4b-b3.5-qad-su` | 68.8 | 40.8 | 50.3 | 34.5 |

Against PTQ at the same allocation, QAD moves every task up at 3.0 bpw (+1.4, +4.4, +1.4,
+0.5) and three of four at 3.5 bpw (+2.0, +0.6, +0.4, -0.8), but no single difference
reaches two standard errors (ARC-C +4.4 is 1.6). The HellaSwag gap to the unquantized
model (-6.7 at 3.0 bpw, -4.2 at 3.5 bpw) is the largest remaining loss and QAD barely
moves it; EXL3 3.0 keeps HellaSwag at -1.9. `acc_norm` gives the same picture.

### Recommendation for the default ratio

**With `--qad`, use `--max-error-ratio 6`** (3.43 bpw on Qwen3.5-0.8B and 3.52 bpw on
Qwen3.5-4B, about half the rows INT2). On 0.8B, QAD at ratio 6 beats PTQ at the current
default ratio 4 on both held-out texts with 0.27 bpw less; on 4B, QAD at ratio 6.2
(3.50 bpw) is within 0.01 KLD of PTQ at 4.0 bpw on the docs text and keeps every
multiple-choice task within two standard errors of PTQ at the same size. Ratio 11
(3.0 bpw) is where QAD helps most relative to PTQ (KL -62 % on 4B docs), but its
HellaSwag and ARC-Easy losses against the unquantized model remain (-6.7 and -3.2
points). For PTQ without QAD the placeholder 4 (3.70 / 3.80 bpw) remains a sensible
default; the mapping from ratio to bpw shifts by about 0.05-0.1 bpw between the two
models, so the ratio is a quality knob, not a size knob. Not measured: QAD at ratio 6
on 4B exactly (6.2 was run) and PTQ at ratio 4 on 4B with this calibration.

## What is not done

* A final global phase with a KL loss on the logits (spec 13 step 2's network-level
  objective): the layer-wise MSE objective already moves the KL by 40-70 %, and a
  full-network pass needs the student, teacher and optimizer state together.
* Training the dense norm weights alongside the quantized Linears (they are written
  unchanged from the source checkpoint).
* Data-parallel training across several GPUs (the teacher/student propagation passes
  do split across devices; the training step runs on the first device).
* Decision fidelity (spec 12.3) and a kernel-side check of a QAD checkpoint in the
  fork's HI2 loader (the format is unchanged, so it is expected to load as-is).
