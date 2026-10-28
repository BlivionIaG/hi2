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
The PTQ studies run with `--no-qad`.

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

## The modes

Three independent switches, all on the command line of `hi2 quantize`:

* the **recipe** decides how much the QAD trains (`--qad-preset`, or any field of
  `QadConfig` through `--qad-config` / `--qad-budget`),
* the **math dtype** decides how the layer's matrix products are computed
  (`--qad-dtype`),
* the **buffer dtype** decides how much RAM the hidden states take (`buffer_dtype` in
  `--qad-config`).

### The recipe

| mode | steps per layer | budget | early stop | sequences per layer | final logit-KL | 0.8B time | 0.8B docs/mix ppl ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `fast` | at most 300, weighted by the gain the earlier layers showed | 0.5 | under 1 % per 50 steps (at least 75) | 256 | 0 (the 27B runs set 200) | 17 min | 1.085 / 1.330 |
| `default` | 300 | 1.0 | none | 256 | 200 | 41 min | 1.061 / 1.286 |
| `thorough` | 600 | 1.0 | none | 512 (batch 8) | 200 | 141 min | 1.047 / 1.267 |

The budget spreads `budget x steps x layers` optimizer steps over the layers, weighted by
how much the previous layers gained (clamped to 0.5x..2x, the first layer 2x), and the
early stop ends a layer when one evaluation interval improves its held-out nMSE by less
than `min_gain`. `--qad-budget F` sets the budget alone, `--qad-config file.json` sets
any field (`{"budget": 0.75, "min_gain": 0.005, "buffer_dtype": "float32"}`), and
`--qad-final-kl STEPS` sets the final whole-network logit-KL phase (the `fast` preset
leaves it off, `default` and `thorough` set 200).

What the recipes cost and buy, measured:

| | 0.8B | 4B | 27B |
| --- | --- | --- | --- |
| `default` | 41 min | 3.1 h | 5.3 h (four V620s) |
| `fast` | 17 min | about 1.3 h (estimate) | about 2.6 h (four V620s, measured) |

### The math dtype (`--qad-dtype`)

| dtype | what it does | verdict on gfx1030 |
| --- | --- | --- |
| `float32` | the default: the layer and its input in float32 | **the only one that works here** |
| `float16` | the layer and its input in half precision; the latents, the ladder bytes and the loss stay float32 | implemented and measured: 1.43x faster on the 27B's layer 10 (2019 ms a step against 1414 ms), but the QAD **stops learning** (the held-out nMSE falls 1.8 % against 45 % in float32) and the full-attention layers (every fourth one) do not run at all. Off by default; the measurement is in "The half-precision math, measured" |
| `bfloat16` | as float16 | refused on gfx10: RDNA2 has no `fdot2.bf16` instruction and the compiler stops with "Cannot select: intrinsic %llvm.amdgcn.fdot2.bf16.bf16" |

A float16 QAD is also not bit-comparable with a float32 one, so a checkpoint written
with it cannot be compared with the earlier runs.

### The hidden-state buffers (`buffer_dtype`)

| dtype | what it costs | when |
| --- | --- | --- |
| `float32` | 41 GiB of RAM at the 27B (the hidden states of every calibration window) | what the runs use |
| `bfloat16` | half of that | when the float32 buffers do not fit |
| `auto` | float32 when it fits in half the free RAM, else bfloat16 | the default |

**bfloat16 buffers and a 1 % early stop do not mix**: the rounding of the buffers is
0.4 % of the activation, which is the noise floor the early stop's threshold sits on, so
the stop fires on noise. The first fast run at the 27B (2026-10-27) had bf16 buffers and
its early stop fired at a median layer gain of 1.5 % - the noise, not the layer's
progress. The 27B run of 2026-10-28 uses `--qad-config '{"buffer_dtype": "float32"}'`
for that reason.

### What to pick

| situation | mode |
| --- | --- |
| 4B or smaller, quality first | `default` (41 min at the 0.8B, 3.1 h at the 4B): 2.4x the time of `fast` for a third less error on the mix text |
| 4B or smaller, a checkpoint in minutes | `fast` |
| the last percent at the 0.8B, time no object | `thorough` (141 min): about 1.5 % more than `default` |
| 27B | `default` is the quality-optimal recipe by the small-size studies, but it is 5.3 h and it died twice to GPU page faults (2026-10-28 11:28 and 14:59). The `fast` recipe is 2.6 h; **its mix KLD decides** whether that is acceptable (the earlier fast checkpoint came out worse than PTQ on the mix text, 0.2947 against 0.2171, but with bf16 buffers) |
| any size, a QAD for a *reference* | `default` with float32 buffers and float32 math - the other modes are only worth their time if the mix KLD says so |

## Step time

A QAD optimizer step decodes the fake-quantised weights, runs the layer forward and backward,
backpropagates through the decode, and updates the latents and ladder bytes. Profile of one
Qwen3.5-4B layer (layer 10, 113 M weights in 8 Linears, 2 sequences of 2048 tokens per step, one
V620, `experiments/profile_qad.py`):

| Part | Torch graph | Fused kernels |
| --- | ---: | ---: |
| Decode of the weights (forward) | 70 ms | 6 ms |
| Backward through the decode | 87 ms | 11 ms |
| Optimizer step (AdamW) | 20 ms | 10 ms |
| Layer forward and backward | 375 ms | 376 ms |
| **Whole step** | **550 ms** | **400 ms** |

The decode and its backward were several hundred small torch ops per group. `hi2.qad_kernel`
(gfx10/gfx11 HIP) runs them as one launch per (bits, block) group, one thread per weight so that
every access is coalesced, and fuses the Hadamard transform and the division by the smoothing
vector into a second small kernel; the AdamW is the fused implementation. The forward values are
bit-identical to the torch graph and to the exported tensor (`tests/test_qad_kernel.py`); the
gradients agree with autograd up to float32 summation order (2e-4 of the largest value, all
latent units, blocks outside the clamp range, zero-step blocks). `HI2_QAD_BACKEND=torch` forces
the torch graph. What remains of a step is the layer itself: the fp32 matrix products (42 %) and
the backward of the gated-delta-rule kernels (23 %).

### The 27B at four GPUs (measured on the 2026-10-28 rerun)

The `default` preset on Qwen3.8-27B: 300 steps per layer over 64 layers, about 295 s a layer,
so **1.0 s a step** and 5.3 h for the phase. What that 1.0 s is made of:

* **The step is fp32-compute-bound, and the fp16 matrix units never fire.** `rocm-smi
  --showmetrics` on the live run reports `average_gfx_activity` 99 %, `average_umc_activity`
  14-24 % and **`average_mm_activity` 0 %** on all four cards for the whole phase: the V620's
  V_DOT2 path is idle. The reason is in the code, not in the hardware: the pipeline loads the
  model with `compute_dtype` float32 (`hi2/pipeline.py`), `hi2/qad.py` casts the layer input
  with `.float()`, and the fake-quantised weight the decoder produces is fp32, so every
  matrix product of the layer runs on the fp32 FMA pipes. The ratio between that and V_DOT2
  on a V620 has not been measured here; the 4B profile above puts the fp32 matrix products
  at 42 % of a step, which is the number to beat.
* **The data-parallel split is already near-ideal, so the PCIe traffic is not the problem.**
  The same recipe on *one* V620 took 3.9 s a step (the earlier `fast` run: 6 619 steps in
  7.2 h), so one sequence costs about 0.98 s and four cards in parallel cost 1.0 s: a 3.9x
  speedup from the four cards. The per-step traffic is large (about 1.6 GB of decoded weights
  out to each of the three other cards and the same back for the gradients, on a topology with
  no peer path, so it all crosses the host) but it is hidden behind the compute. Fixing the
  copies alone buys about 2 %, not 50 %.
* **The steps stop paying after the early layers.** The held-out nMSE gain per layer falls
  from -17.9 % (layer 0) and -28.5 % (layer 2) to -7.0 % (layer 9), -3.2 % (layer 12) and
  -0.5 % to -4.1 % over layers 15 to 18. It is not monotonic, but the level is clear: from
  layer 10 on, 300 steps buy a few percent each. `StepBudget` (budget below 1) and the early
  stop (`min_gain`, `min_steps`, `eval_every`) exist to spend the steps where they pay; the
  `fast` preset is budget 0.5 with a 1 % early stop.

**What a cheaper recipe would cost.** `experiments/qad_budget_sim.py` replays the measured
validation curves of the live run through the budget and the early stop of `hi2/qad.py`
(the same arithmetic, `tests/test_qad_budget_sim.py` pins it against `StepBudget`), and
prices the recipes on the 19 layers the run has finished:

| recipe | optimizer steps | optimizer time | held-out nMSE given up |
| --- | ---: | ---: | ---: |
| `default` (the measured run) | 5 700 (100 %) | 1.57 h | 0.0 % |
| budget 0.75 | 3 683 (65 %) | 1.02 h | +0.1 % |
| budget 0.75 + 2 % early stop | 2 672 (47 %) | 0.74 h | +0.2 % |
| budget 0.5 | 2 558 (45 %) | 0.71 h | +0.3 % |
| `fast` (budget 0.5 + 1 % early stop) | 2 161 (38 %) | 0.60 h | +0.3 % |
| 1 % early stop, full budget | 3 450 (61 %) | 0.95 h | +0.1 % |

The layer-wise nMSE is a proxy, and it is a proxy that did not predict the one end-to-end
result we have: the `fast` preset's checkpoint at the 27B came out **worse than PTQ** on the
mix KLD (0.2947 against 0.2171) and on the task suite (pooled -8.2 against -6.2,
[results.md](results.md)). That run differed in three ways at once - a third of the steps,
bf16 hidden-state buffers and one GPU - so it does not say which of the three cost the
quality, but it does say that the layer-wise nMSE is not what to decide on. The mix KLD is.

So the order of work is:

1. **fp16 matrix math for the layer forward and backward** - the same recipe, less time.
   The model is already loaded through `--compute-dtype`, but the QAD path overrides it with
   `.float()` in three places (`hi2/qad.py`: the layer input, the decoded weight and the
   loss target). A dtype option for the QAD phase would let V_DOT2 do the matrix products.
   It needs a check that the fp16 gradients do not drown the signal (the signal is a
   1-4 % nMSE gain in the later layers), which is a 4B run, not a 27B run.
2. **A step budget with an early stop for the 27B** - 2 to 2.6x less optimizer time, and
   the only lever that can be worth more than the time it saves. It changes the result, so
   it has to be judged on the mix KLD, not on the layer nMSE.
3. **Only then the per-step PCIe traffic and the hidden-state buffers** (168 MB a tensor,
   re-uploaded every step; the fp32 buffers need 41 GiB of RAM). The traffic is worth about
   2 % on its own.


## Interface

| Flag | Meaning |
| --- | --- |
| `--qad` / `--no-qad` | QAD is on by default when `--calib` is given (and implies `--sensitivity probe` unless given); `--no-qad` = PTQ only |
| `--qad-preset fast\|default\|thorough` | steps, budget and sequences per layer (see "The modes") |
| `--qad-budget F` | fraction of `steps x layers` to spend (preset default) |
| `--qad-dtype float32\|float16\|bfloat16` | math dtype of the QAD layer forward and backward (default float32); the latents, the ladder bytes and the loss stay float32. Only float32 works on gfx1030 (see "The modes") |
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
does not fit in half the free RAM), `dtype` (`float32`, `float16` or `bfloat16`: the
math dtype of the layer forward and backward), `seed`, `evals`, `sampling` (`random`,
`balanced` = equal windows from each calibration file, or `head`). Example `--qad-config`
file: `{"sampling": "balanced", "budget": 0.5}`.

The three presets are described in "The modes" above, with what they cost and buy.

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

### Qwen3.5-4B: task accuracy (vLLM harness, `mc_eval.py`, `acc`)

Dense fp16 export of each checkpoint (`hi2 export-dense`), served by the fork's vLLM. n = 570 / 299 / 2000 / 2000; one standard error is
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

* A full-network final phase. `--qad-final-kl STEPS` trains a small set of parameters
  (the quantized Linears' ladder bytes and the norms) against the logits; it is on with the
  `default` and `thorough` presets (200 steps) and splits the network layer-wise over the devices when
  it does not fit one GPU ([lowbit.md](lowbit.md)). A pass that trains everything needs the student, teacher and optimizer state together.
* Training the dense norm weights in the layer-wise phase is available as an option
  (`train_norms` in the QAD config) and is off: it was within seed noise
  ([lowbit.md](lowbit.md)).
* Data-parallel training across several GPUs (the teacher/student propagation passes
  do split across devices; the training step runs on the first device).
* Decision fidelity (spec 12.3) and a kernel-side check of a QAD checkpoint in the
  fork's HI2 loader (the format is unchanged, so it is expected to load as-is).

### The half-precision math, measured (2026-10-28)

`--qad-dtype float16` runs the layer's matrix products in half precision: the layer and its
input are cast, the latents, the ladder bytes and the loss stay float32, and the rounding
of the decoded weight is a straight-through (the gradient of the rounded weight reaches
the latent unchanged). `experiments/qad_dtype_check.py` runs one real decoder layer of the
27B (layer 10: 8 quantized Linears, 383 M weights, 6 x 2048 tokens) through the production
training loop in every dtype. One card, 60 steps:

| dtype | ms a step | held-out nMSE | after 60 steps |
| --- | ---: | ---: | --- |
| float32 | 2019 | 2.1598e-03 -> 1.1906e-03 | **-45 %** |
| float16 | 1414 | 2.1599e-03 -> 2.1211e-03 | **-1.8 %** |

Two results, one good and one bad:

* **The speed is 1.43x, not 3x.** The matrix products are about 42 % of a step (the 4B
  profile above), so they got about 3x faster, but the gated-delta-rule recurrence (23 %)
  and the backward through the decode stay float32. The step of the 27B, four cards with
  the data-parallel split, would go from about 1.0 s to about 0.7 s, not to 0.3-0.4 s.
  The reason is the matrix product itself: on this stack (torch 2.12, ROCm 7.14, gfx1030)
  a float16 product of the MLP's shape is only **1.47x** faster than the float32 one
  (57 ms against 84 ms for [2048, 5120] x [5120, 17408] with the backward,
  `experiments/fp16_gemm_check.py`), not the 4x the V_DOT2 peak would suggest. The float16
  product and its backward are numerically fine (relative difference 3.6e-04), so this is
  the speed the hardware gives, not a rounding effect.
* **The QAD does not learn in float16.** The first loss is the same to five digits (the
  forward is fine: 2.1598e-03 against 2.1599e-03), but the training loss *rises* above
  the held-out loss and stalls: the update moves the latent in a direction that does not
  help. That is the signature of a gradient that is mostly noise.

Bfloat16 is not an option on gfx1030 at all: RDNA2 has no `fdot2.bf16` instruction and
the compiler stops with "Cannot select: intrinsic %llvm.amdgcn.fdot2.bf16.bf16"
(`hi2/qad.py` refuses the dtype on gfx10 before that happens).

Two more things the half-precision path costs:

* **The full-attention layers do not run.** The attention of the model computes the
  rotary embedding in float32 and casts q and k back, but v keeps the activation dtype,
  so a float16 layer reaches the scaled dot-product attention with q and k in float32
  and v in float16: "Expected query, key, and value to have the same dtype". That is
  the model's code, not the quantizer's (the 27B has a full-attention layer every
  fourth layer).
* **The float16 kernels of the gated delta rule have to be compiled again.** The Triton
  kernels of the linear attention are compiled per dtype, and the AMD backend needs
  more than ten minutes for the float16 variants (cached afterwards).

So the half-precision QAD is **not** a usable speedup of the 27B QAD on this hardware:
1.43x instead of 3x, and the numerics break (the full-attention layers crash, the
linear-attention layers do not learn). The option stays in the code (`--qad-dtype`,
default float32, so nothing changes) and the measurement is here; a usable version
would keep the attention and the recurrence in float32 and put only the MLP in float16,
which is a per-branch dtype the model code does not offer.

### What is left, in order

1. **The recipe** (the lever that works): `--qad-preset fast` spends the steps where
   they still gain. The 27B rerun of 2026-10-28 runs it with the float32 buffers (the
   earlier fast run had bf16 buffers, whose 0.4 % rounding sits at the noise floor of
   the 1 % early stop, so its early stop fired on noise). It has to be judged on the mix
   KLD: the layer-wise nMSE is a proxy that did not predict the one end-to-end result
   we have.
2. **The copies.** The per-step PCIe traffic is about 9.6 GB (the decoded weights to the
   three workers and the gradients back) and it is hidden behind the compute now (the
   four-way split is within 2 % of ideal). If the workers decoded the weights locally
   from the codes and the ladder - about 110 MB instead of 1.6 GB - the traffic would
   halve, but the gradient that comes back is the same size, so the win is 2x on the
   traffic and nothing on the step until the compute is much faster.
3. **The hidden-state buffers.** 168 MB a tensor, re-uploaded every step (about 1.3 GB
   a step over the four cards, 8 ms a card, 2 % of a step). Keeping them device-resident
   does not fit (264 sequences would need 44 GB); pinned host memory would buy a few
   milliseconds. The real cost is the 41 GiB of RAM they take, which is why the earlier
   fast run fell back to bf16 - and bf16 buffers put the early stop's 1 % threshold at
   the noise floor, so they and the early stop do not mix.
