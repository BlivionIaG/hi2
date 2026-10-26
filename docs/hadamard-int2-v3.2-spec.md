# HadamardINT2-FP16 v3.2 (draft)

Status: **draft for review.** Supersedes v3.1 for the points listed in section 10.
Every number in sections 3, 4 and 6 is produced by a script in this repo
(`./experiments/`) or by a probe that was assembled and **run on a
gfx1030 V620**; where something is a design choice or a guess it says so.

| --- | --- | --- | --- | --- |
| Scope | Weight matrices. Activation handling is defined only for the dot-product input. KV cache is out of scope. |
| Hardware | RDNA2 `gfx1030`, ROCm 7.14 toolchain. |
| Runtime | vLLM. The format, loader and kernels go into the vLLM PR. **Calibration does not**: it lives in the separate `.` quantizer, which produces checkpoints in this format. |
| Format name | `rdna_hint_flex` (quant_method; "HI2" is the short name used in this document), tensor prefix `hi2_`, mode `flex`. v2 and v2i4 are restrictions of flex. The earlier name `hadamard_int2` is kept as a load-time alias because the rows carry INT2, INT4 and INT8. |

## 1. What changed from v3.1, in one table

| Topic | v3.1 | v3.2 | Evidence |
| --- | --- | --- | --- |
| Block parameters | two fp16 per block (`lo`, `step`) | two **8-bit** per block, relative to per-row fp16 constants | no measurable quality change, saves 0.25-1.0 bpw of block parameters (0.5 at B=32) before the 48/K row constants (4.1) |
| Block sizes | 32, 64 (16 dropped: 2 bpw of ladder) | **16, 32, 64** (B=16 costs 1.0 bpw with 8-bit params) | B=16 beats mixing INT2/INT4 rows at ~3.0 bpw in layer-output error (4.3); no end-to-end gain shown yet (4.5) |
| Compute | one path, `v_dot2_f32_f16` | **two modes** on identical stored weights: F16 (`v_dot2`) and I8 (`v_dot4_i32_i8`) | 6 |
| Activation | fp16 | I8 mode: int8 with an fp16 scale per **16** columns | adds ~0.6% output error vs 19-26% from INT2 weights (5) |
| Hadamard | mandatory | **rotation plus smoothing** (default on); rotation alone is not enough | smoothing then rotation helps at every bit width (-30% perplexity at 2.5 bpw); rotation alone hurts a calibrated encoder but helps a data-free one by 31% (4.2, 4.5, 4.6) |
| Encoder | min-max | calibrator-side: least-squares `lo/step` fit, activation-weighted | -39% output error vs min-max at INT2/B32 (4.2) |
| Rounding | nearest | **Hessian-aware error-compensated rounding (GPTQ), default when calibrated** (8.3); offline only, no format or kernel change | removes 66% of the perplexity excess at 3.0 bpw on Qwen3.5-0.8B (1.92 to 1.32) and 57% on Qwen2.5-0.5B (4.8) |
| ISA table | partly wrong | verified by assembler and hardware run | 7 |

## 2. Format

### 2.1 Tensor and blocks

`W[R, K]`. Blocks run along K, size `B` in {16, 32, 64}, chosen per row. Each row
is INT2 (4 levels) or INT4 (16 levels), chosen per row. `K` must be a multiple
of 64.

### 2.2 Affine ladder with hierarchical parameters

Per block the reconstruction is `w = lo + code * step`, `code` in `0..levels-1`.
Rather than storing `lo` and `step` as fp16 per block, each row stores three
fp16 constants and each block stores two bytes:

```text
row:    lo_min, lo_scale, step_scale        (fp16)
block:  lo_u8, step_u8                      (uint8)

lo   = fp16( fma_f32(lo_u8,  lo_scale,   lo_min) )
step = fp16( step_u8 * step_scale )          (fp32 multiply, one rounding to fp16)
w    = fp16( lo + code * step )              (exact in fp64, one rounding; v_fma_f16)
```

`step_u8 = 0` is the degenerate block (constant `lo`). The quantizer **must
encode codes from the decoded fp16 `lo` and `step`**, so offline and device
agree bit for bit. Per-row constants cost `48 / K` bits per weight (0.012 bpw at
K=4096).

Why not a symmetric ladder (`step` only, levels centred): measured worse at equal
bpw (4.1). Why not nvfp4-style FP8 scales: a linear 8-bit step relative to the
row maximum already costs nothing measurable, so there is nothing to recover.

### 2.3 Smoothing and rotation

Two optional transforms act on the input dimension, in this order, on the weights
offline and on the activation at runtime.

**Smoothing** (needs calibration statistics): a per-column fp16 vector `s` with
`s_j = (d_j / geomean(d)) ** (alpha / 2)`, `d_j = E[x_j^2]` and `alpha = 0.5`
(range 0.05 to 20). The weights become `W * s` and the activation `x / s`, so the
product is unchanged. This equalises the channel scales the encoder would otherwise
have to prioritise.

**Rotation**: orthonormal Walsh-Hadamard of order 16 on groups of 16 consecutive
columns, same flags for every row (`K/16` bits per tensor, or 1 bit if uniform).
`H H = I`, so the dot product is preserved exactly; only where quantization error
lands changes.

The stored basis is therefore `H (W * s)` and the runtime activation is `H (x / s)`.
With no smoothing vector `s = 1`. Rotation alone, with an encoder that weights
columns by activation statistics, makes the model worse (it mixes the channel
scales the encoder exploits); rotation after smoothing, or rotation with an
encoder that has no activation statistics, helps (4.5, 4.6). The flags can be all
zero or mixed; the default is all ones.

### 2.4 Layout

Flat per-row buffers as in v3.1 section 6.1 (`codes`, `row_code_off`, `row_nbits`,
`row_block`, `had`), with these replacing `lo` / `step` / `row_scale_off`:

| Buffer | Type | Length |
| --- | --- | --- |
| `blk_params` | `uint8` pairs `(lo_u8, step_u8)` | `2 * sum_r K/B_r` |
| `row_consts` | `fp16` triple | `3 * R` |
| `row_code_off` | `int32`, in 32-bit words | `R+1` |
| `row_param_off` | `int32`, in blocks (each block is one `(lo_u8, step_u8)` pair) | `R+1` |
| `col_scale` (optional) | `fp16` | `K` |

Codes are packed LSB-first in 32-bit words as before (16 INT2, 8 INT4 or 4 INT8 per word). The checkpoint tensors are `hi2_code_off` and `hi2_param_off` (the offsets are `int32`, as written by the tool and read by the fork's kernels).

### 2.5 Bit accounting

`bpw = nbits + 16/B + 48/K + meta/(R*K)` with `meta` as in v3.1 section 4
(Hadamard flags, 1 bit per row when INT2 and INT4 rows are mixed, 2 bits per row when
block sizes differ), plus
`16/R` when the tensor carries a smoothing vector (0.004 at R=4096, but 0.125 for
a 128-row k or v projection). Tensors that share an input (q, k, v; gate, up)
share the same `s`, so it can be stored once or folded into the preceding norm.

| nbits | B=16 | B=32 | B=64 |
| --- | --- | --- | --- |
| 2 | 3.012 | 2.512 | 2.262 |
| 4 | 5.012 | 4.512 | 4.262 |

(K=4096, metadata excluded. The saving over v3.1's fp16 pair is 0.98 bpw at B=16,
0.48 at B=32 and 0.23 at B=64.)

## 3. How these numbers were measured

Model Qwen2.5-0.5B-Instruct (bf16 weights, fp32 math), 5 layers (0, 6, 12, 18,
23) x {q, o, up, down} = 20 matrices, 256 sampled rows each. Activations are
captured by hooking the real forward pass on 4096 tokens of repo documentation.
**The encoder is calibrated on the first 2048 tokens and every error below is
measured on the other 2048** (held out). Metric: relative L2 error of the layer
output `X W_hat^T` against `X W^T`, averaged over matrices. This is output
error, **not KL divergence and not perplexity**; neither has been measured yet.
One small model: treat the *ordering* of options as the finding, not the
absolute values. Script `./experiments/scale_study.py`; raw results
`./experiments/results/qwen2.5-0.5b-scale-study.json`.

## 4. Scale system, encoder, rotation

### 4.1 Scale system (activation-weighted encoder, no rotation, INT2)

| System | B=16 | B=32 | B=64 |
| --- | --- | --- | --- |
| affine, fp16 `lo`+`step` (v3.1) | 4.000 bpw / 0.1905 | 3.000 / 0.2297 | 2.500 / 0.2606 |
| **affine, 8-bit params (v3.2)** | **3.000 / 0.1907** | **2.500 / 0.2296** | **2.250 / 0.2604** |
| symmetric, fp16 `step` | 3.000 / 0.2592 | 2.500 / 0.2844 | 2.250 / 0.3051 |
| symmetric, 8-bit `step` | 2.500 / 0.2593 | 2.250 / 0.2846 | 2.125 / 0.3052 |

(bpw counts code bits plus block parameters only; the row constants add `48/K`,
0.054 at this model's K=896 and 0.012 at K=4096. Output error is relative L2.)
Findings:

* **8-bit parameters are free**: errors match fp16 to the third decimal, at 0.5-1.0
  bpw less. INT4 is the same (e.g. B=64: 4.250 bpw / 0.0664 against 4.5 / 0.0664).
* **Affine beats symmetric at equal bpw** by 9-12% at INT2 (affine B=32 8-bit at
  2.500 bpw: 0.2296; symmetric B=16 8-bit at 2.500 bpw: 0.2593) and about 16% at
  INT4 (4.500 bpw: 0.0553 against 0.0656).

### 4.2 Encoder and rotation (INT2, B=32, fp16 affine, 3.0 bpw)

| Encoder | no rotation | all rotated | per-64-chunk choice |
| --- | --- | --- | --- |
| min-max (v3.1) | 0.3765 | 0.3568 | n/a |
| MSE `lo`/`step` fit | 0.2908 | 0.2718 | n/a |
| MSE, activation-weighted | **0.2297** | 0.2623 | **0.2289** |

INT4 B=32 (5.0 bpw): min-max 0.0741 / 0.0712; MSE 0.0667 / 0.0636; weighted
**0.0552** / 0.0624; per-chunk 0.0549.

Findings:

* The **encoder is the largest lever**: min-max -> MSE fit is -10..-25% output
  error (-23% at INT2/B32, -10% at INT4/B32), adding activation weighting another -21% (no rotation). Format unchanged.
* **Rotation helps an activation-blind encoder (5-7%) but hurts an
  activation-aware one (+14%)**: the weighted fit exploits per-channel activation
  magnitude, which rotation mixes away. Letting the calibrator choose per
  64-column chunk recovers parity (0.2289 vs 0.2297), about a quarter of chunks
  chosen. On this model rotation alone therefore buys almost nothing once activation
  statistics exist (smoothing changes this, 4.6). **This contradicts v3.1's "mandatory; worth 0.6-0.8 bpw"**,
  which was measured on a synthetic tensor with injected column outliers and a
  weight-only calibrator. The correct statement is in 4.5 and 4.6: rotation helps a
  data-free encoder, and helps a calibrated one when paired with smoothing.

### 4.3 Block size and INT4 (8-bit params, weighted, no rotation)

| | bpw | output error |
| --- | --- | --- |
| INT2 B=64 | 2.250 | 0.2604 |
| INT2 B=32 | 2.500 | 0.2296 |
| INT2 B=16 | 3.000 | 0.1907 |
| INT4 B=64 | 4.250 | 0.0664 |
| INT4 B=32 | 4.500 | 0.0553 |
| INT4 B=16 | 5.000 | 0.0437 |

Mixing INT2/B32 and INT4/B64 rows to reach 3.0 bpw would give about 0.197 by
interpolation in squared error, worse than INT2/B16 at 0.1907. So **B=16 earns its
place once its scale cost is 1.0 bpw**, and flex should offer it. This comes from
uniform-configuration points, not from running the allocator; confirm with the
allocator.

### 4.4 Second-level scope: per row or per tensor

The per-row constants cost `48/K` bpw. Per-tensor constants cost nothing, which is
what nvfp4 does (with an FP8 float block scale). Tested with the same held-out
protocol, weighted encoder, no rotation, 8-bit block parameters:

| Variant | INT2 B=16 | INT2 B=32 | INT2 B=64 | INT4 B=32 |
| --- | --- | --- | --- | --- |
| fp16 per block (v3.1) | 0.1905 | 0.2297 | 0.2606 | 0.0552 |
| **per row, linear step, linear `lo`** | 0.1907 | 0.2296 | 0.2604 | **0.0553** |
| per row, linear step, `lo` as z-score | 0.1908 | 0.2298 | 0.2604 | 0.0567 |
| per row, log step, `lo` as z-score | 0.1915 | 0.2301 | 0.2609 | 0.0572 |
| per tensor, linear step, linear `lo` | 0.1910 | 0.2302 | 0.2606 | 0.0561 |
| per tensor, linear step, `lo` as z-score | 0.1911 | 0.2300 | 0.2602 | 0.0569 |
| per tensor, log step, `lo` as z-score | 0.1913 | 0.2302 | 0.2610 | 0.0576 |

At INT2 every variant is within about 0.3%: a tie. At INT4 per-row linear is best
and per-tensor costs +1.4% (linear) to +4% (log). Per tensor saves `48/K` bpw
(0.054 here, 0.012 at K=4096). **Decision: per row, linear, with `lo` in the
affine range** (the form in 2.2). Per tensor is a defensible simplification if the
constants must be a single broadcast, but it trades accuracy for under 0.05 bpw.
`lo` as z-score is slightly worse and cannot represent a constant block, so it is
not used. Raw results:
`./experiments/results/qwen2.5-0.5b-scope-study.json`.

### 4.5 End-to-end quality (Qwen2.5-0.5B, held-out text)

Produced by the reference quantizer `.` (global allocation over the whole
model, calibrated on 32 x 2048 tokens of repo documentation) and `hi2 eval`
(dequantized weights through the original architecture; 8 x 2048 tokens from a
disjoint part of the text; **perplexity of the unquantized model 10.49**). All
linear layers are quantized; embeddings stay dense. One model, one corpus, one run
each: differences under about 5% are within what I would call noise. bpw includes
the row constants, metadata and the smoothing vector. Raw data:
`./experiments/results/e2e_qwen2.5-0.5b.json`.

**Calibrated, smoothing 0.5 plus rotation, against no rotation:**

| bpw | no rotation: perplexity / KL | smoothing + rotation: perplexity / KL |
| --- | --- | --- |
| 2.5 | 38.25 / 1.340 | **26.91 / 0.978** |
| 3.0 | 19.92 / 0.664 | **18.31 / 0.575** |
| 3.5 | 15.11 / 0.379 | **14.43 / 0.333** |
| 4.0 | 12.20 / 0.163 | **12.10 / 0.149** |
| 4.5 | 10.96 / 0.054 | **10.82 / 0.033** |

**Other 3.0 bpw runs:**

| Run | perplexity | KL (nats/token) | top-1 |
| --- | --- | --- | --- |
| data-free, no rotation | 44.28 | 1.472 | |
| data-free, rotation on | **30.71** | 1.100 | 0.576 |
| calibrated, rotation on, no smoothing | 30.50 | 1.062 | 0.583 |
| calibrated, rotation per-chunk auto, no smoothing | 20.75 | 0.700 | 0.651 |
| calibrated, no rotation | 19.92 | 0.664 | 0.656 |
| calibrated, smoothing 0.25 + rotation | 20.66 | 0.695 | 0.648 |
| calibrated, smoothing 0.5 + rotation | **18.31** | **0.575** | 0.679 |
| calibrated, smoothing 0.75 + rotation | 18.42 | 0.583 | 0.675 |
| uniform INT2 B=16, no rotation | 20.65 | 0.719 | 0.633 |
| INT2/INT4 with blocks 32 and 64 only, no rotation | 20.05 | 0.670 | 0.653 |

Findings:

* **Calibration is the biggest win** (data-free to calibrated, both without
  smoothing: 44.3 or 30.7 to 19.9 at 3.0 bpw).
* **Rotation helps a data-free encoder a lot** (-31% perplexity). With calibration
  alone it hurts (30.50 against 19.92), and per-chunk "auto" is slightly worse than
  none. **Smoothing before rotation turns this around**: -8% perplexity and -13% KL
  at 3.0 bpw, -30% perplexity at 2.5 bpw, with the gain shrinking toward 4.5 bpw
  (KL still -39%). Smoothing strength matters: 0.25 is worse than none, 0.5 and 0.75
  are equivalent. See 4.6.
* **Global allocation beats uniform INT2/B16** modestly (19.92 against 20.65 at the
  same bpw). Against a uniform INT4/B64 at 4.3 bpw the flex allocation was not
  better (KL 0.071 against 0.064; measured with rotation per-chunk auto and no
  smoothing, so before the change above). The allocator minimises a proxy (relative
  activation-weighted squared error per tensor); sensitivity-aware allocation is an
  open item (section 9).
* **Block 16 shows no end-to-end benefit** here: restricting flex to blocks 32 and
  64 matches full flex (20.05 against 19.92).
* **More encoder effort did not help** (fast 19.96, normal 20.75, max 21.47, all with
  rotation auto): the extra effort optimises the same proxy.
* The allocation is far from uniform: at 3.0 bpw the k and v projections get about
  5 bpw while the MLP gate gets about 2.5.

### 4.6 Where rotation is applied: smoothing, group size, signs

Layer-output error, held-out, same setup as section 3 (INT2, B=32, weighted encoder;
`./experiments/rotation_study.py`). `alpha` is the smoothing exponent, `G`
the rotation group size:

| Transform | `G=1` (none) | `G=16` | `G=64` signs | `G=128` signs |
| --- | --- | --- | --- | --- |
| no smoothing | **0.2296** | 0.2623 | 0.2718 | 0.2709 |
| smoothing 0.5 | 0.2374 | **0.2207** | 0.2227 | 0.2212 |
| smoothing 1.0 | 0.2925 | 0.2506 | 0.2470 | 0.2514 |

At INT4/B=32 the same pattern: none 0.0553, smoothing 0.5 + `G=16` 0.0516. Random
signs change nothing (G=16: 0.2623 against 0.2624). Larger groups help only after
smoothing and then barely (G=16 is as good as G=128), so **G=16 stays**, which also
keeps the activation transform to four butterfly stages. Every projection type
improves with smoothing 0.5 plus rotation, by 2-7%.

Reading: the activation-weighted encoder prioritises columns that see large
activations; rotating unsmoothed weights spreads that information away. Smoothing
equalises the channels first, so the rotation then only flattens the weight
outliers, which is what it is good at.

### 4.7 Against GGUF and EXL3 (Qwen3.5-0.8B)

Perplexity divided by the unquantized model's, held-out text, details and caveats in
`hadamard-int2-design-notes.md` section 3. EXL3 3inst at 3.0 bpw: **1.12**. HI2
(smoothing + rotation) at 2.5 / 3.0 / 3.5 / 4.0 / 4.5 bpw: 3.12 / 1.92 / 1.46 / 1.19 /
1.05. GGUF Q2_K / Q3_K_S / Q3_K_M / IQ4_XS at 3.26 / 3.46 / 3.96 / 4.57 bpw: 1.74 /
1.37 / 1.27 / 1.044. HI2 is in the range of GGUF's k-quants and needs about 4.2 bpw to
match EXL3 at 3.0, i.e. EXL3 is about 30% smaller at equal quality. Error
compensation (section 8.3) is the first thing to try against that gap.

**Qwen3.5-4B** (hidden 2560, MLP 9216, same text, same protocol; unquantized perplexity
7.067 in both HF and vLLM). EXL3 3.0 bpw, converted on this box with upstream exllamav3
1.6.4 (mul1 codebook, built for gfx1030) and scored in the fork's vLLM: **1.052**. HI2
with Hessian-aware rounding: 1.120 at 3.0 bpw (KL 0.152) and 1.080 at 3.5 bpw (KL
0.099); with sensitivity-aware allocation (8.6) 1.103 at 3.0 bpw (KL 0.131). GGUF in
llama.cpp (f16 perplexity 6.258): Q2_K 1.220 at 3.10 bpw, Q3_K_S 1.104 at 3.44, Q3_K_M
1.066 at 3.88 (1.117 without the imatrix), IQ4_XS 1.021 at 4.44, Q4_K_M 1.012 at 4.88.
HI2 needs about 3.8 bpw to match EXL3's 1.052, so EXL3 is again about 0.8 bpw (20%)
smaller at equal quality, as on the 0.8B model. Against GGUF, HI2 at 3.5 bpw (1.080)
beats Q3_K_S at 3.44 (1.104). Time to quantize the 4B on one gfx1030 GPU: 29 minutes for
HI2 (Hessian-aware, 262k tokens), 50 minutes for the sensitivity probe added on top, and
about 2 hours for EXL3.

### 4.8 Hessian-aware rounding (Qwen3.5-0.8B, Qwen2.5-0.5B)

`hi2 quantize --rounding gptq` (8.3), same held-out protocol as 4.5. Perplexity divided
by the unquantized model's (10.874 for Qwen3.5-0.8B); KL in nats per token. Raw data
in `./experiments/results/e2e_qwen3.5-0.8b.json`.

| bpw | nearest rounding: ratio / KL | Hessian-aware: ratio / KL |
| --- | --- | --- |
| 2.5 | 3.12 / 1.148 | **1.53 / 0.428** |
| 3.0 | 1.92 / 0.642 | **1.32 / 0.267** |
| 3.5 | 1.46 / 0.357 | **1.18 / 0.158** |
| 4.0 | 1.19 / 0.151 | **1.07 / 0.069** |
| 4.5 | 1.05 / 0.041 | **1.02 / 0.019** |

Qwen2.5-0.5B (unquantized 10.49): 3.0 bpw 1.75 to **1.32**, 4.0 bpw 1.15 to **1.08**.

Same engine as the competitors, Qwen3.5-0.8B at 3.0 bpw: vLLM on this fork scores
EXL3 3inst at **1.123** and HI2 with Hessian-aware rounding at **1.316** (nearest
rounding 1.92). Scored in llama.cpp (f16 reference 9.58), HI2 gives 1.565 / 1.334 /
1.197 / 1.074 / 1.023 at 2.5 / 3.0 / 3.5 / 4.0 / 4.5 bpw, against GGUF Q2_K 1.74 at
3.26 bpw, Q3_K_S 1.37 at 3.46, Q3_K_M 1.27 at 3.96 and IQ4_XS 1.044 at 4.57 bpw
(interpolated, HI2 is about 27%, 12% and 15% better than the first three at the same
size). HI2 reaches EXL3's 1.123 at about 3.76 bpw, so EXL3 is now about 0.75 bpw
(20%) smaller at equal quality, down from 1.2 bpw.

Ablations at 3.0 bpw on Qwen3.5-0.8B (ratio; all with Hessian-aware rounding unless
noted):

| Variant | ratio | KL |
| --- | --- | --- |
| default: smoothing 0.5 + rotation | **1.316** | 0.267 |
| same with 524k calibration tokens instead of 65k | 1.303 | 0.250 |
| same with sequential propagation (8.4) | 1.311 | 0.263 |
| rotation only | 1.338 | 0.274 |
| no transform | 1.371 | 0.297 |
| smoothing only | 1.394 | 0.319 |
| smoothing 0.25 / 0.75 + rotation | 1.323 / 1.359 | 0.271 / 0.281 |
| uniform INT2 B=16 (3.05 bpw) | 1.411 | 0.351 |
| blocks 32 and 64 only | 1.314 | 0.266 |

**Calibration size** (Qwen3.5-0.8B, Hessian-aware rounding, ratio / KL):

| Tokens | 3.0 bpw | 4.0 bpw | Time at 3.0 bpw |
| --- | --- | --- | --- |
| 65k (32 x 2048) | 1.316 / 0.267 | 1.068 / 0.069 | 367 s |
| 131k | 1.321 / 0.261 | 1.071 / 0.066 | 381 s |
| **262k (128 x 2048)** | **1.295 / 0.248** | **1.062 / 0.062** | 429 s |
| 524k | 1.303 / 0.250 | not run | 521 s |

Perplexity moves by under 2%, but the KL falls 7% to 10% by 262k tokens and then
flattens. 262k is the default.

**Sensitivity-aware allocation** (8.6), ratio / KL, 3.0 bpw unless noted:

| Model | uniform weights | sensitivity-aware | probe time |
| --- | --- | --- | --- |
| Qwen3.5-0.8B | 1.295 / 0.248 | **1.207 / 0.196** | +254 s |
| Qwen2.5-0.5B | 1.262 / 0.274 | **1.220 / 0.231** | +101 s |
| Qwen3.5-4B | 1.120 / 0.152 | **1.103 / 0.131** | +1278 s |
| Qwen3.5-0.8B at 4.3 bpw | 1.023 / 0.027 | 1.026 / 0.028 | +238 s |

The gain is -31%, -16% and -14% of the perplexity excess on the three models at 3.0
bpw, and nothing at 4.3 bpw where nearly every row is INT4. Read against 8.10 as
written (a 3% cut of the ratio), it passes on the two small models (-6.8% and -3.3%)
and not on the 4B (-1.5%); the ratio is a compressed scale and the gain is consistent
on the excess and on KL, so it is adopted, with that caveat on the record.

Findings:

* **Hessian-aware rounding is the largest single gain after calibration**, on both
  models and at every bit width, and it changes nothing at inference. It costs
  quantization time: 367 s against 99 s for Qwen3.5-0.8B on one gfx1030 GPU, plus about
  2 GB of saved Hessians per run (deleted after the run by the experiment scripts).
* **The transform choice matters much less under compensated rounding**: the four
  cells span 1.32 to 1.39. Smoothing plus rotation is still best, rotation alone now
  helps (1.371 to 1.338), smoothing alone slightly hurts.
* **More calibration tokens help a little** (-1%), **sequential propagation does not**
  (-0.4% at 3.0 bpw, +0.2% at 4.0), and the smoothing exponent is flat between 0.25
  and 0.5 (worse at 0.75). None meets the 3% bar of 8.10.
* **Allocation under compensated rounding**: flex beats uniform INT2/B16 by 7% at 3.0
  bpw, as before, but at 4.3 bpw it does not beat a uniform INT4/B64 (1.030 against
  1.024, KL 0.030 against 0.024). Block 16 again adds nothing (blocks 32 and 64 only:
  1.314 against 1.316).

### 4.9 Qwen3.5-4B against EXL3 and GGUF (measured 2026-10-25)

Same model (Qwen3.5-4B), same 8 x 2048 held-out tokens, HI2 with Hessian rounding, 262k
calibration tokens and the sensitivity probe. Three harnesses, each compared only
with its own unquantized row: HF (HI2 KLD), the fork's vLLM (perplexity and
multiple-choice, `acc`), llama.cpp (GGUF KLD and multiple-choice). Multiple-choice is
n = 570 (ARC-Easy), 299 (ARC-Challenge), 2000 (HellaSwag, MMLU): one standard error is
about 2 points on HellaSwag/MMLU and 3 points on the ARC sets.

**Fidelity.** Unquantized perplexity 7.067. KLD in nats/token against each format's
own unquantized logits.

| Format | Linear-body size (MiB) | Total file (MiB) | PPL (vLLM/HF) | KLD | Top-1 same |
| --- | --- | --- | --- | --- | --- |
| EXL3 3.0 bpw (mul1) | 1336 | 3402 | 7.432 | n/a (not loadable in HF) | n/a |
| HI2 3.0 | 1287 | 3368 | 7.792 | 0.131 | 84.4 % |
| HI2 3.5 | 1499 | 3580 | 7.449 | 0.077 | 88.0 % |
| HI2 4.0 | 1712 | 3793 | 7.245 | 0.034 | 92.2 % |
| GGUF Q2_K | n/a | 1858 | n/a | 0.259 | 79.8 % |
| GGUF Q3_K_S | n/a | 2013 | n/a | 0.143 | 85.3 % |
| GGUF Q3_K_M | n/a | 2201 | n/a | 0.100 | 86.9 % |
| GGUF IQ4_XS | n/a | 2449 | n/a | 0.041 | 91.3 % |
| GGUF Q4_K_M | n/a | 2644 | n/a | 0.033 | 92.3 % |

**Task accuracy, vLLM harness** (`acc`; unquantized in the first row).

| Format | ARC-E | ARC-C | HellaSwag | MMLU |
| --- | --- | --- | --- | --- |
| unquantized | 72.3 | 42.5 | 53.3 | 36.6 |
| EXL3 3.0 | 69.6 | 42.1 | 51.3 | 35.9 |
| HI2 3.0 | 67.7 | 36.5 | 44.6 | 32.2 |
| HI2 3.5 | 71.6 | 43.5 | 48.1 | 34.0 |
| HI2 4.0 | 72.5 | 44.1 | 50.9 | 35.4 |

**Task accuracy, llama.cpp harness** (GGUF, imatrix from the same calibration text;
unquantized F16 in the first row).

| Format | ARC-E | ARC-C | HellaSwag | MMLU |
| --- | --- | --- | --- | --- |
| F16 | 65.4 | 47.2 | 71.0 | 39.0 |
| Q2_K | 64.0 | 44.1 | 64.7 | 35.3 |
| Q3_K_S | 64.7 | 43.8 | 67.8 | 37.8 |
| Q3_K_M | 64.2 | 46.8 | 68.9 | 37.3 |
| IQ4_XS | 66.3 | 45.8 | 70.7 | 38.6 |
| Q4_K_M | 66.1 | 45.5 | 71.0 | 39.0 |

What this says, without rounding in our favour:

* HI2 4.0 matches Q4_K_M on KLD (0.034 against 0.033) and loses nothing beyond noise
  on any task; HI2 3.5 sits between Q3_K_M and IQ4_XS on KLD (0.077 against 0.100 and
  0.041) with ARC/MMLU within noise but HellaSwag 5 points down (2.5 standard errors).
* HI2 3.0 is clearly the weak point: KLD 0.131 is Q3_K_S-class, and it loses 8.7
  HellaSwag points and 6 ARC-Challenge points while EXL3 at the same nominal width
  loses 2.0 and 0.4. The 2-bit-heavy mix hurts exactly the tasks that need the
  sharpest decisions; this is the case the INT8/QAD extensions and a better 2-bit
  encoder have to fix.
* EXL3 3.0 (perplexity 7.43) is about as good as HI2 3.5 (7.45) at a similar body
  size, so the trellis path is worth about 0.5 bpw here.
* The sizes are not yet like-for-like: the GGUF totals include its quantized
  embedding/output matrix, while HI2 and EXL3 keep the 636 M-parameter tied embedding in
  fp16 or 6 bpw and carry a vision tower (HI2 total also includes 232 MiB of unquantized
  MTP head). The "linear-body" column counts only the quantized decoder linears; a per-tensor
  GGUF body size is still to be added before an equal-size claim against GGUF is made.
* Not run: the HI2 dense export through llama.cpp as a bridge row, a KLD for EXL3
  (needs an engine that loads it and a common reference), and decision fidelity (12.3).

## 5. int8 activations (I8 mode input)

Symmetric int8, fp16 scale per group of columns, held-out tokens, fp (unquantized)
weights, so this is the activation error alone:

| Rotation | per token | per 64 | per 32 | per 16 |
| --- | --- | --- | --- | --- |
| none | 3.29% act / 2.91% out | 1.14% / 1.02% | 0.88% / 0.78% | **0.67% / 0.60%** |
| all | 1.31% / 1.18% | 0.59% / 0.54% | 0.49% / 0.45% | 0.41% / 0.38% |

Against INT2 weight output error of 0.19-0.26 the activation term adds in
quadrature (0.1907 with a 0.006 term is still 0.1908). **Group size matters for
INT4 weights**: per-token int8 with no rotation (0.029) would raise INT4/B=64
output error from 0.0664 to about 0.0725 (+9%), while a per-16 group (0.006) adds
0.4%. Decision: **activation scale group A = 16 columns, no rotation needed.**
Re-measured later on real linear inputs after smoothing and rotation (Qwen3.5-0.8B and
4B, 8 x 2048 held-out tokens, perplexity ratio to F16 mode for A = 16, 32, 64 and per
token): 1.0001, 1.00004, 1.0001, 1.0008 on the 0.8B and 1.0001, 0.9998, 1.0000, 1.0019 on the
4B. With smoothing and rotation, A = 64 is free in quality, which is what an
integer-epilogue int8 prefill kernel needs (the checkpoint layout is unchanged; only the
activation group changes and the block size must divide it). The I8 mode itself stays parked:
the kernel is built and correct but not benchmarked against rocBLAS.
int4 activations are rejected (quality), which also rules out `v_dot8_i32_i4`.

## 6. Compute modes

The stored weights are identical; the mode is a kernel choice.

### 6.1 F16 mode

Decode code to fp16 with the magic-number trick (`half(0x6400|c) - 1024`), one
`v_fma_f16`/`v_pk_fma_f16` with the block's fp16 `lo`/`step`, then
`v_dot2_f32_f16` against fp16 activations. Activation quality is unchanged.
`v_dot2c_f32_f16` is fine to use (7): v3.1's warning against it was wrong.

### 6.2 I8 mode

Quantize the (optionally rotated) activation once per layer per token:

```text
sx_g   = fp16( absmax(x_g) / 127 )         per group g of 16 columns
xq_j   = clamp(round(x_j / sx_g), -127, 127)   int8
sumq_g = sum_{j in g} xq_j                  int16
```

These are shared by every output row. For weight block with decoded `lo`, `step`
and 16-column chunk `g` inside it:

```text
S_g   = sum_j code_j * xq_j                 int32, via 4 x v_dot4_i32_i8
y_blk += sx_g * ( step * S_g + lo * sumq_g )     fp32
```

Codes are 0..3 or 0..15, positive int8, so signed `v_dot4_i32_i8` is exact; the
affine offset is carried by `sumq_g`, with no per-weight FMA. Per 16 weights that is
4 dot4 plus 2 fp32 FMAs, against 8 `v_dot2` plus the weight decode (8 packed
FMAs and 8 packed subtracts) in F16 mode.
**Unmeasured**: decode GEMV is usually memory-bound, so the instruction saving
may not show up as speed. A kernel benchmark must decide.

## 7. RDNA2 instruction facts (verified)

Assembled with the ROCm 7.14 clang for `gfx1030`, and the dot results checked by
running `./experiments/isa/run.hip` on a V620:

| Instruction | gfx1030 |
| --- | --- |
| `v_dot2_f32_f16`, `v_dot2c_f32_f16` | yes. `fdot2(100,-200 . 3,4)` returned -500 exactly: **no [-2,2] input clamp** |
| `v_dot4_i32_i8`, `v_dot4c_i32_i8`, `v_dot4_u32_u8` | yes (`sdot4` returned the expected -1) |
| `v_dot8_i32_i4`, `v_dot8_u32_u4` | **yes** (v3.1 said no). Both operands must be int4, so unusable with int8 activations |
| `v_dot2_i32_i16` | yes |
| `v_dot4_f32_f16`, `v_dot2_f32_bf16`, `v_dot4_i32_iu8` | **no** (assembler rejects) |

## 8. Quantization procedure (separate tool, not in the vLLM PR)

`.` produces checkpoints; vLLM only reads them. The contract between them is
the checkpoint layout (section 2.4) plus `quantization_config`
`{quant_method: "rdna_hint_flex", mode, version: "3.2"}`; every per-tensor choice (bits,
block size, flags, smoothing vector) is in the tensors. **Everything in this section
except smoothing and rotation (2.3) lives only in the quantizer: it decides which
values go into the format and never changes the format, the kernels or the vLLM
loader.** Raising quantization quality therefore costs nothing at inference.

### 8.1 Procedure

A conforming quantizer runs three stages over the model, layer by layer (the model
stays in RAM, one layer on the compute device):

* **A. Tabulate (pass 1).** Run the calibration sequences through the *unquantized*
  model and, per Linear, collect the activation statistics (`E[x_j^2]` and the 16x16
  block second moments, enough for the diagonal weights in the rotated basis). From
  them derive the smoothing vector (2.3), the rotation flags, and for every row the
  weighted error of every candidate `(nbits, B)` under nearest rounding.
* **B. Allocate.** Choose every row's `(nbits, B)` globally under one model-wide
  bits-per-weight target (8.6).
* **C. Encode (pass 2).** Run a second layerwise pass. For each layer, accumulate the
  full Hessian of each Linear input in the stored basis from the activations of the
  *already quantized* earlier layers (8.4), then round with error compensation (8.3)
  into the allocated configuration, and quantize that layer before moving on.

The implementation does A and B, and C in two forms. The default collects the full
Hessians during pass 1, from the *unquantized* activations, and encodes with error
compensation while streaming the checkpoint (`--rounding gptq`). `--sequential` runs
the true second layerwise pass of 8.4. Without calibration data it uses nearest
rounding. Both forms are conforming.

### 8.2 Techniques and status

| # | Technique | Stage | Format impact | Status |
| --- | --- | --- | --- | --- |
| 1 | Activation-weighted ladder fit (least squares of `lo`/`step` by `E[x^2]`) | A, C | none | implemented, measured: -39% output error, -35% perplexity vs data-free at 3.0 bpw (4.2, 4.5) |
| 2 | Smoothing then rotation (2.3) | A | in format (`col_scale`, flags) | implemented, measured: -30% perplexity at 2.5 bpw, -8% at 3.0 (4.5, 4.6) |
| 3 | Global bits-per-weight allocation (8.6) | B | none | implemented; beats uniform INT2/B16 by 7% at 3.0 bpw, does not beat uniform INT4/B64 at 4.3 bpw (4.8) |
| 4 | Hessian-aware rounding (8.3) | C | none | **implemented, adopted (default)**: ratio 1.92 to 1.32 at 3.0 bpw (4.8) |
| 5 | Sequential propagation through the quantized prefix (8.4) | C | none | implemented (`--sequential`), **not adopted**: -0.4% at 3.0 bpw (4.8) |
| 6 | Per-tensor smoothing exponent (8.5) | A | none (`col_scale` is already stored) | specified, not implemented; the global exponent is flat between 0.25 and 0.5, so little to gain (4.8) |
| 7 | Sensitivity-aware allocation (8.6) | B | none | **implemented (`--sensitivity probe`), adopted**: -31% / -16% / -14% of the perplexity excess at 3.0 bpw on three models (4.8) |
| 8 | Calibration volume and domain (8.7) | A, C | none | measured: 262k tokens is the default, KL -7% to -10% vs 65k, flat beyond (4.8) |
| 9 | Local code refinement (8.9) | C | none | specified, not implemented |
| 10 | Non-uniform or trellis codebook | format | changes the decode path | out of scope for v3.2 |
| 11 | Quality-driven allocation (`--max-error-ratio`) | B | none | **implemented, default**: fewest bits whose weighted error is at most R x the best achievable; INT2 where it matters less, INT4 elsewhere; bpw is a result (ratio 6 gives 3.43 bpw on the 0.8B, 3.52 on the 4B) |
| 12 | Mixed-domain calibration | A, C | none | **measured, adopted**: docs-only calibration gives 3x the KLD on chat and tool text than on docs (0.650 vs 0.196, 0.8B at 3.0 bpw); a docs, chat and tool-calling mix cuts the out-of-domain KLD by 44% for 20% in-domain (8.7) |
| 13 | Act-order rounding (`--act-order`) | C | none | implemented, **not adopted**: 0.2% better on docs, worse on mixed text (KLD 0.371 vs 0.364) |
| 14 | Quantization-aware distillation (section 13) | C | none | **implemented, on by default**: 4B KLD 0.161 to 0.061 at 3.0 bpw, 0.093 to 0.042 at 3.5 bpw; accuracy gains about 1 to 2 points (13) |
| 15 | Mixture-of-experts quantization | A, B, C | none | **implemented** (routed-token statistics per expert, shared gate/up smoothing, per-layer allocation groups); OLMoE-1B-7B at 3.54 bpw: perplexity x1.011, served in the fork at 2.1x dense decode |
| 16 | Fused encode, statistics cache | C | none | **implemented**: bit-identical output, 0.8B 754 to 266 s cold and 28 s warm, 4B 3073 to 1485 s, OLMoE encode 23x |

Techniques 4 and 5 were implemented and measured in that order; the remaining
specified ones (6, 7, 9) are open. Each is adopted only if it passes 8.10.

### 8.3 Hessian-aware rounding (technique 4, implemented and adopted)

**What it is.** Instead of rounding each weight to its nearest code independently,
round the columns of a row one after another and push each column's rounding error
into the columns not yet rounded, in the direction that least increases the layer's
output error. This is GPTQ, and in the LDLQ form it is what QuIP# and EXL3 use. It is
also the main thing separating HI2 from EXL3 at 3 bits (section 4.7).

**It does not change inference.** It only chooses *which* code each weight gets within
the same ladder. The Hessian exists only inside the quantizer.

**It combines with smoothing and rotation.** The error to minimise is, per row,
`(w_hat - w)^T Hs (w_hat - w)` in the stored basis, where `Hs = E[x' x'^T]` and
`x' = H (x / s)` is exactly the runtime activation of 2.3. So the Hessian is taken
*after* smoothing and rotation. Unlike the diagonal weights used today it is the full
`K x K` matrix: rotation spreads outlier energy across a group but the correlations
between columns are what compensation exploits. It is accumulated once in the original
basis (`Sigma = E[x x^T]`, so it does not depend on the choices of stage A) and
transformed as `Hs = T Sigma T^T` with `T = H diag(1/s)`, where `H` is block-diagonal
so this costs `O(16 K^2)`.

**Algorithm (fixed-ladder GPTQ).**

1. Damp the Hessian: `Hs += 0.01 * mean(diag(Hs)) * I` (the factor is a parameter;
   raise it if the factorisation fails).
2. Take the upper Cholesky factor `U` of `Hs^-1`. One `Hs` serves every tensor that
   shares an input (q, k, v; gate, up).
3. Fit the ladders as now (weighted least squares with `diag(Hs)` weights, 8-bit
   parameters, 2.2). The decoded fp16 `lo`/`step` of each block are **fixed** for the
   rounding pass, so the codes stay consistent with what the decoder reconstructs and
   the 2.2 row constants, which depend on every block, are known before rounding.
4. For each row, for columns `j` in order (blocks of 128 columns with lazy updates of
   the rest, as in GPTQ): `code_j = round((w_j - lo)/step)` with the ladder of the
   block containing `j`; `e = (w_j - w_hat_j) / U_jj`; `w_{j+1:} -= e * U_{j, j+1:}`.
5. Optionally refit `lo`/`step` per block by weighted least squares against the
   compensated targets and repeat from 4 (at most twice), since the best ladder moves
   once the weights have been compensated.

**Processing order.** Columns are processed in storage order by default. Processing in
descending `diag(Hs)` order ("act-order") is allowed and changes nothing in the format
because only the order of rounding changes, not the position or the block of any
column; it is an option to measure, not a default.

Rows are independent given `U`, so the row sharding across devices applies. Cost is
`O(R K^2)` per tensor plus `O(K^3)` for the factorisation per distinct input; memory is
`K^2` floats per Hessian (64 MB at K=4096, 1.2 GB at K=17408 in float32).

**Interactions to re-measure.** The defaults (smoothing 0.5 plus rotation) were tuned
for nearest rounding. The Hessian already encodes channel scale, so smoothing may be
partly redundant, and rotation's incoherence benefit is exactly what the literature
pairs with this algorithm. Run four cells: {nearest, compensated} x {no rotation,
smoothing + rotation}, then re-sweep the smoothing exponent.

### 8.4 Sequential propagation (technique 5, implemented, not adopted)

In stage C, the input activations of layer `l` are produced by running the
calibration sequences through layers `0..l-1` **as already quantized** (dequantized
weights in the original forward). The Hessian and the ladder weights are then those of
the inputs the quantized model will really see, and later layers can absorb earlier
layers' error. This is the standard GPTQ "true sequential" setting (target
`W x_q`). A variant keeps the unquantized model's output `W x_fp` as the target, which
asks later layers to correct earlier errors explicitly; it is left open. Stage A stays
on the unquantized model because the allocation (8.6) must be known first. The cost is
a second layerwise forward pass over the calibration data. Measured: no gain
(4.8), so it is off by default.

### 8.5 Per-tensor smoothing exponent (technique 6, open)

The exponent `alpha` is a single value (0.5) tuned on one model (4.5). Smoothing is
stored per tensor, so `alpha` can be chosen per tensor with no format change: for
`alpha` in a small set (for example 0.25, 0.5, 0.75, 1.0) compute the weighted error
of the encoded tensor (or, with 8.3, of the compensated tensor) and keep the lowest.
q, k and v share an input and so share the vector; the exponent for the group is
chosen on the sum of their errors. Evidence so far argues against expecting much: with nearest rounding `alpha` 0.5 and
0.75 were equivalent and 0.25 was worse than none, and with compensated rounding
0.25 and 0.5 are within 0.5% and 0.75 is 3% worse (4.8).

### 8.6 Allocation (technique 3 and 7)

Every row picks its `(nbits, B)` to minimise `sum_t s_t * sse_t / energy_t` subject to
one model-wide bits-per-weight target (Lagrangian, bisected), counting the 48/K row
constants, the metadata and the smoothing vector. With `s_t = 1` this is implemented
and is a proxy: at 4.3 bpw it did not beat a uniform INT4/B64 (4.5).

**Sensitivity-aware (technique 7, implemented).** `hi2 quantize --sensitivity probe`
quantizes one tensor at a time at a fixed reference configuration (INT2, B=32, nearest
rounding), runs a small calibration batch (8 x 512 tokens) through the whole model, and
takes `s_t = (KL increase) / (proxy error of that tensor)`, normalised to a median of 1
with a floor of 0.05. This turns the proxy into a prediction of the loss. Cost is one
forward pass per tensor, so it needs the model to fit on the compute device. Results in
4.8. The measurement is a stage-A operation and costs nothing at inference.

**What EXL3 does differently, and the next step.** EXL3's `measure_model` does not
extrapolate from one reference point: it substitutes each candidate *precision* into a
module and records the measured KL change for that precision, and `optimize_model`
picks the mix. HI2's probe uses one reference width and scales by the proxy ratio. Probing
each tensor at both INT2 and INT4 (and each block size, or a sample of them) would
replace the extrapolation by measurement; it is not implemented.

Allocation uses nearest-rounding tables even when stage C compensates (8.3); the
compensated errors are lower and differ per candidate, so the allocation is only a
proxy for them. Replacing the tables by compensated errors costs six compensated
passes per tensor and is left open.

### 8.7 Calibration data (technique 8, measured)

The Hessian has `K^2` entries, so it needs far more tokens than the diagonal
statistics. Requirements for a conforming run: report the token count behind every
tensor (a Linear that saw none, such as an unrouted expert, falls back to data-free
weights and must be logged); calibrate on text disjoint from the evaluation text;
default to 128 sequences of 2048 tokens (262k tokens) whenever the Hessian is used, and
say so in the result. Section 4.8 shows the KL improving by 7% to 10% from 65k to 262k
tokens and flat at 524k. The 4B runs also used 262k; EXL3 calibrates on about 500k.
Domain matters, and it matters a lot at low bit width: on Qwen3.5-0.8B at 3.0 bpw,
calibrating on repo docs alone leaves chat and tool-calling text at KLD 0.650 (against
0.196 on docs), while a calibration mix of docs, chat and tool calling gives 0.364 on that
text and 0.234 on docs. The quantizer reads the calibration file from its start, so a mix
must interleave the sources in the first `samples x seq_len` tokens
(`./experiments/build_calib_mix.py` does). Evaluate on text from a different
domain than the calibration text, as well as on matching text; an in-domain-only table
overstates the result.

### 8.8 Interface to the rest of the system

`quantization_config` records `calibrated`, `calib_tokens`, `target_bpw` and the
smoothing and rounding settings that produced the file, for reproducibility only; no
reader depends on them.

### 8.9 Local code refinement (technique 9)

After compensated rounding, a block-wise coordinate descent can try changing a code by
one level and keep the change when the Hessian error `e^T Hs e` falls, using the closed
form for a single-column change (`O(K)` per trial, `O(R K^2)` per pass). The expected
gain over GPTQ is small and uncertain; it is listed so that it is evaluated, not
adopted, after techniques 4 and 5.

### 8.10 Evaluation protocol and acceptance

Every technique is judged on the same measurements as section 4.5:

* Held-out text disjoint from calibration; perplexity divided by the unquantized
  model's, plus KL and top-1 agreement, from the same evaluator for every row of a
  table. bpw counts everything in the linear layers (row constants, metadata,
  smoothing vector).
* Models: Qwen3.5-0.8B (the same-model EXL3 and GGUF comparison of 4.7) and
  Qwen2.5-0.5B, at 2.5, 3.0, 3.5, 4.0 and 4.5 bpw.
* **Adopt** a technique when it lowers the perplexity ratio by at least 3% (about the
  run-to-run noise seen so far) at 3.0 bpw on both models, does not worsen it by more
  than 1% at 4.5 bpw, and keeps quantization time within 10x of the previous
  procedure. Report the numbers either way.
* The headline goal is to close at least half the gap to EXL3 at 3.0 bpw on
  Qwen3.5-0.8B (HI2 1.92, EXL3 1.12: a ratio at or below about 1.5). It is a goal, not
  a prediction: EXL3 also has a trellis codebook, which none of the techniques above
  replicates.

## 9. Open questions

* **The gap to EXL3** (4.7, 4.8): at 3.0 bpw, 1.32 against 1.12 on Qwen3.5-0.8B and
  1.10 against 1.05 on Qwen3.5-4B, about 0.8 bpw at equal quality on both. What remains
  is probably EXL3's trellis codebook and block-parameter-free format (section 11).
* **Accuracy and inference cost are not yet compared.** KL and perplexity are measured;
  multiple-choice accuracy (ARC, HellaSwag, MMLU) is being measured for EXL3, HI2 and
  GGUF, and the inference-cost comparison (decode work per weight, activation
  transforms, kernel variants, measured decode speed of EXL3 and GGUF) is pending.
  No HI2 kernel exists, so HI2's own speed is unknown.
* **Is a trellis path worth its inference cost?** A trellis-coded INT2/INT4 variant
  (11.2) would add a second decode path; it is a proposal, not planned, and would be
  decided against the measured quality gap at the same code width (11.4).
* **Allocation:** sensitivity weighting helps at 3.0 bpw but flex still does not beat a
  uniform INT4/B64 at 4.3 bpw. Measured per-precision sensitivity (8.6) is the next step.
* **Three models only.** Qwen2.5-0.5B, Qwen3.5-0.8B and Qwen3.5-4B, one text. Everything
  was tuned on the small ones.
* The smoothing vector costs `16/R` bpw per tensor, which matters for small tensors;
  sharing it between q/k/v and gate/up (or folding it into the norm) is not
  implemented.
* Calibration and evaluation text is repository documentation, not a standard
  corpus.

## 10. Corrections to v3.1

1. INT4 dot **does** exist on gfx1030 (`v_dot8_i32_i4`); it is unusable for us
   because activations stay int8.
2. `v_dot2c_f32_f16` does not clamp inputs to [-2,2]; measured on hardware.
3. Per-block fp16 `lo`/`step` replaced by 8-bit parameters (2.2).
4. Block size 16 restored (2.1).
5. "Hadamard mandatory" replaced by smoothing plus rotation, default on (2.3, 4.5, 4.6).
6. v3.1's Tables 2-7 are unreproduced; this document does not rely on them.

## 11. Trellis and codebook ideas for the INT2 and INT4 paths (design study, not normative)

The widths stay INT2 and INT4. The question is what the EXL3 trellis codebook (QTIP) can
teach those two paths. Facts below are from the fork's EXL3 kernels
(`csrc/rocm/exl3_dot2_*.cuh`, which already decode them on gfx1030), the conversion plan
printed by upstream exllamav3 1.6.4 (built for gfx1030 in a scratch directory,
`./experiments/exl3_build.sh`) and the checkpoints on the benchmark box.

### 11.1 What the EXL3 format is

* Weights sit in 16 x 16 tiles of `16 K` uint16, `K` bits per weight (1 to 8, and 1.5,
  2.5, 3.5 with the mul1 codebook). **For K = 2 and K = 4 this is the same word-aligned
  stream as HI2's INT2 and INT4 codes.** No new width and no cross-word layout is
  needed.
* There are **no per-block scales or offsets**. The side data are two fp16 vectors per
  tensor (`suh` over the input dimension, `svh` over the output dimension) around a
  128-wide Hadamard on **both** sides, which makes the weights close to Gaussian, so a
  fixed codebook is enough.
* The decode is not a lookup in a ladder. The *state* of weight `i` is the 16-bit window
  of the tile's circular bitstream at bit offset `i K` (parallel funnel shifts, no
  sequential walk; for K = 2 that is 8 consecutive codes, for K = 4 it is 4), and the
  value is a fixed pseudo-random function of the state. `3inst`: `x * 89226354 +
  64248484`, a LOP3 mask, one half add. `mcg`: `x * 0xCBAC1FED`, same epilogue.
  `mul1`: `x * 0x83DCD12D`, a byte-sum (dp4a) and one `hfma`. About four or five
  operations per weight and no table; the dot product is `v_dot2_f32_f16`, as in HI2's
  F16 mode.
* Because consecutive weights share overlapping windows, the code is a *trellis* code:
  the set of reachable value sequences is large (a 2^16-state codebook) while only K
  bits per weight are stored. That is the vector-quantization gain a scalar ladder
  cannot get.
* The encoder is a Viterbi search over the trellis with Hessian-aware (LDLQ) rounding.
  It runs on gfx1030 with upstream exllamav3 >= 1.6, after casting the bf16
  gated-delta-net Triton kernels to fp16 (gfx1030 cannot compile them in bf16).
* **Allocation.** At an integer target, upstream assigns exactly that many bits to every
  language-model tensor (3 bpw to all 32 MLP and attention tensors of Qwen3.5-4B), 4 bpw
  to MTP layers and 6 bpw to the head and vision tower. Its sensitivity-aware path is
  separate: `measure_model` substitutes each candidate precision into one module at a
  time, propagating hidden states, and records the change in output KL divergence;
  `optimize_model` picks the mix that meets the bit budget. The cost is **measured per
  module and per precision**.

### 11.2 What can be taken without changing the widths

1. **A trellis-coded variant of the INT2 and INT4 code stream.** Same storage, same
   per-row width flag, plus one flag that says how the stream is decoded (ladder or
   trellis). The decode adds a window extraction and a procedural codebook in place of
   `lo + code * step`; the fork already has that decode for gfx1030. It is the only
   idea here that attacks the quantizer's rate-distortion efficiency itself.
2. **Drop the block parameters on the trellis path.** A trellis row needs no `lo`/`step`:
   it saves `16/B + 48/K` bits per weight, that is 0.25 to 1.0 bpw plus the row constants,
   so a "2.0 bpw" trellis row is 2.0 bpw and not 2.5. This is where a large part of the
   gap in section 4.7 may sit, because HI2's nominal 3.0 bpw carries only about 2.5
   code bits.
3. **The cost of 1 and 2 is the transform and the encoder.** The fixed codebook is only
   good after a strong rotation of the input dimension (EXL3 uses 128 columns). EXL3
   also rotates the output dimension, which mixes rows and would force the INT2/INT4
   choice to 16-row or 128-row groups. Quantization quality does not obviously need it,
   since the Hessian lives on the input dimension only; whether the Gaussian codebook
   works without it is untested. A trellis runs along a row's bitstream, so per-row
   width stays possible; EXL3's 16 x 16 tiles are a kernel layout choice, not part of
   the idea. The encoder is a Viterbi search plus LDLQ, far heavier than HI2's current
   quantizer, and a row-wise variant would have to be built and validated.
4. **Measured per-precision sensitivity for the allocation** (8.6), the one idea that
   needs none of the above.

### 11.3 What this means for variable INT2/INT4

The identity of HI2 is a variable INT2/INT4 choice per row. Compatible with it: a
row-wise trellis (same storage, a flag for ladder or trellis decode), no block
parameters on trellis rows, measured per-precision sensitivity for the allocation, and
Hessian-aware rounding (already adopted). Not compatible without changing the unit of
the choice: an output-dimension rotation and EXL3's 16-row tiles.

The pieces of the trellis path do not separate cheaply: the codebook, the strong input
rotation, the missing block parameters and the Viterbi encoder are what make it work
together. The cheaper ladder-only alternatives were measured (11.5) and give little.
What a trellis row is worth is therefore the open question, measured next (11.4).

### 11.4 EXL3 against HI2 at the same code width

Qwen3.5-0.8B converted with upstream exllamav3 at exactly 2.0, 3.0 and 4.0 bpw (mul1
codebook, 6-bit head), scored by perplexity in the fork's vLLM on the same held-out
16376 tokens as the HI2 rows (unquantised 10.87; HI2 rows are Hessian-rounded, 128
calibration sequences, nominal bpw including all metadata).

| Target | EXL3 (trellis, no block params) | HI2 (ladder, flex) |
| --- | --- | --- |
| 2.0 bpw | 2 924 962 (collapsed, see below) | not measured (2.5 bpw: 16.66) |
| 3.0 bpw | 12.15 | 14.16 to 14.39 |
| 3.5 bpw | not converted | 12.87 |
| 4.0 bpw | 11.18 | 11.61 |

At equal nominal width the trellis path is ahead by about 2.2 ppl at 3 bpw and 0.4 ppl
at 4 bpw; HI2 needs about 3.5 bpw to match EXL3 at 3.0. The EXL3 2.0 checkpoint
collapsed in this stack, so it is not a quality number for the format: a 0.8B model
with a 2-bit body is expected to be bad but not unscored, and the likeliest cause is
fp16 range loss in the hybrid layers (the converter needed an fp16 patch). It is not
re-examined here. Whether the 2.2 ppl at 3 bpw justifies the trellis decode cost is the
decision this section feeds; the ladder stays the spec's default until that is
decided explicitly.

### 11.5 Fewer block parameters with a stronger rotation (measured)

Qwen3.5-0.8B, MLP projections of layers 0, 8, 16 and 23 (12 matrices, 256 sampled rows
each), activation-weighted encoder with nearest rounding, smoothing 0.5 then input-side
rotation of group `G`, block `B`; held-out relative output error. bpw is code bits plus
`16/B` plus `48/K`. Raw data
`./experiments/results/qwen3.5-0.8b-block-rotation-study.json`.

| `G` | `B` | INT2 error / bpw | INT4 error / bpw |
| --- | --- | --- | --- |
| 16 | 16 | 0.195 / 3.05 | 0.043 / 5.05 |
| 16 | 32 | 0.229 / 2.55 | 0.053 / 4.55 |
| 16 | 64 | 0.252 / 2.30 | 0.062 / 4.30 |
| 128 | 64 | 0.248 / 2.30 | 0.061 / 4.30 |
| 128 | 128 | 0.259 / 2.17 | 0.068 / 4.17 |
| 128 | 256 | 0.269 / 2.11 | 0.074 / 4.11 |
| 256 | 256 | 0.266 / 2.11 | 0.073 / 4.11 |

A 128-wide rotation lowers the error by 1 to 2% at the same block size. Halving the
block parameters (64 to 128) saves 0.125 bpw and costs about 4% more error, which is
roughly what 0.125 bpw buys through the allocation (-4% output error is about -9% KL).
**With a scalar ladder, a stronger rotation does not make the block parameters
free**; removing them needs the fixed Gaussian codebook of 11.1.

### 11.6 Targets and the INT8 row class

Design targets set for the format: better than GGUF at the same size, in the middle of
GGUF and EXL3 on quality, and clearly cheaper to decode than EXL3 (native
`v_dot2_f32_f16`/`v_dot4_i32_i8` on word-aligned codes, no codebook, no 128-wide
output Hadamard). Compression may be traded for quality by mixing INT4 rows into an
INT2 body, which flex already does.

A third row class, INT8 (`nbits = 8`, B = 16, 32 or 64; the tool offers 32 and 64, about 8.4 bpw), is available for
very sensitive projections. It uses the same ladder, params and packing (four codes per
word, one byte each) and is the natural operand of `v_dot4_i32_i8` in the I8 mode. It
is opt-in in the quantizer (`--bits 2,4,8`) and the global allocation decides where it
pays: a mixed row-width stream needs `ceil(log2(3)) = 2` flag bits per row. Measured on
Gaussian rows: relative error 2e-5 at 8.44 bpw (INT4 at 4.44 bpw: 7e-3), so INT8 rows
are effectively lossless and cost about 4 extra bpw over INT4; whether any row earns
that price is a sensitivity-allocation question for the 4B runs.

## 12. How to evaluate a quantized model (normative for this spec's claims)

A format claim ("better than GGUF", "between GGUF and EXL3") is only accepted when it
is measured on all four axes below, on the same model, the same held-out text and the
same evaluator for every row. Perplexity alone is not enough: it can hide the loss of
sharp decisions (which option, which tool, when to stop).

### 12.1 Axis 1: distribution fidelity (KLD)

Held-out text disjoint from calibration (8 x 2048 tokens at a fixed offset). Report
perplexity, perplexity ratio to the unquantized model, KL divergence to the
unquantized logits (nats/token) and top-1 agreement. The evaluator must give the same
numbers for the same weights in two engines (here: HF and the fork's vLLM). The
llama.cpp evaluator scores only the second half of each window, so its perplexity is
compared only with other llama.cpp rows; GGUF rows get KLD from llama.cpp against
logits of the F16 GGUF, and the HI2 dense export is also run through llama.cpp as a
bridge row.

### 12.2 Axis 2: task accuracy

Multiple-choice accuracy through the same harness for every engine: ARC-Easy,
ARC-Challenge, HellaSwag (n = 2000) and MMLU (n = 2000), `acc` and `acc_norm`, using
the `ikawrakow/validation-datasets-for-llama.cpp` sets. Report the difference to the
unquantized model with its binomial standard error; differences below two standard
errors (about 2 points at n = 2000) are not claimed.

### 12.3 Axis 3: decision fidelity (agentic and coding)

Quality as the model is used: on a fixed set of coding-agent prompts (the
`tools/agentic_bench` corpus) with greedy decoding, count (a) the first generated
action (call a tool / which tool / answer / stop) that differs from the unquantized
model, (b) valid, parseable tool-call JSON, (c) the garbage checks (empty, repeat,
non-printable). Report the agreement rate of the decision with the unquantized model.
Greedy output is not bitwise reproducible across runs on this stack, so the baseline
is the unquantized model run twice, and only differences beyond that spread count.
(Motivated by evaluating on the model's decisions rather than on likelihood alone.)

### 12.4 Axis 4: inference cost

Per format, the table of Section 6 filled with: ops per weight in the inner loop,
activation transforms (and their cost per token), number of kernel variants, whether
the decode is a native RDNA2 dot instruction on word-aligned codes, and the measured
decode tokens/s per request and prefill tokens/s on the same GPU (c = 1, 16k/1k and
1k/512, in the fork's vLLM or llama-bench). A format without a kernel is labelled
"estimated" and is never ranked above a measured one.

### 12.5 Acceptance for the stated goal

* Against GGUF: at equal size (bpw including all metadata and the unquantized
  embedding and head, the same on both sides), lower KLD and no accuracy axis worse by
  more than two standard errors.
* Against EXL3: between GGUF and EXL3 on axes 1 to 3 at the same size, with equal or
  lower decode cost on axis 4. Trading some compression for INT4 (and INT8 rows for
  sensitive tensors) to reach that is accepted.
* Not accepted: any claim made on a single model or only on one axis.

## 13. Extension: quantization-aware distillation (QAD), tool-side only

Source idea: NVIDIA's QAT/QAD recipe (a fake-quantized forward pass with a
straight-through estimator, a short fine-tune on a subset of the training data, and for
QAD a distillation loss against the full-precision teacher). They report it recovers
most of the loss PTQ leaves on harder benchmarks for NVFP4 on one Nemotron model; that
is one model and a handful of benchmarks, so it is a lead and not a measurement here.

**No format change.** The extension lives entirely in the quantizer; the container,
kernels and compute modes of Sections 2 and 6 are untouched, and a QAD checkpoint is a
normal v3.2 checkpoint. It is the training-time counterpart of Hessian-aware rounding
(8.3): where 8.3 compensates rounding error layer by layer against a fixed input
second moment, QAD lets the weights move so the whole network's output distribution
matches the teacher.

**Why the format suits it.**

* The forward pass is a plain differentiable map from latent weights to the stored
  grid: smoothing `s` and the order-16 Hadamard `H` are fixed, so the student layer is
  `W_q = fake_quant(H (W s))` with the row's `(nbits, B)` fixed by the allocation, and
  `fake_quant` is the 8-bit-parameter ladder of 2.1 with an STE on the rounding.
* Per-row `(nbits, B)` picks stay frozen from the PTQ allocation (including INT8 rows),
  so the bit budget does not drift during training.
* Teacher and student share tokenizer and architecture, so the teacher is the
  unquantized checkpoint and the loss is `KL(teacher || student)` on logits.

**Procedure.**

1. Run the normal PTQ pipeline (calibrate, allocate, Hessian-rounded encode) and keep
   its picks, smoothing vectors and rotation flags.
2. Initialise the student's latent weights from the unquantized weights (not from the
   dequantized ones) and train with the fake-quantized forward, STE gradients, a small
   learning rate and fp32 master weights; also let the per-block `lo`/`step` follow the
   latents (refit each step, or learn them with the STE).
3. Train on text from the calibration distribution plus agent and code data, for a
   small token budget (hundreds of millions of tokens at most for a 4B model, to be
   measured); hold out the Section 12 evaluation text.
4. Re-encode: round the final latents to codes with the PTQ picks, requantize the
   params, and write the checkpoint through the existing writer.

**Cost and fit.** A layer-by-layer variant (train one decoder layer against the
teacher's hidden states, then move on) fits the box's 29 GiB GPUs for 4B-class models
and keeps the existing layerwise executor; a full-network variant needs the student,
the teacher and optimizer state in memory at once and is practical only with GPU+RAM
offload or TP. Quantization time grows from minutes to hours; the vLLM PR is unchanged.

**Status: implemented and on by default** (`hi2 quantize`; `--no-qad` for PTQ only;
presets `fast`, `default`, `thorough`; layer-wise, resumable; design, defaults and tables
in `docs/hi2-qad.md`). Measured on the allocation of the PTQ run it starts from:

| Model, bpw | KLD docs (PTQ to QAD) | KLD mixed text | Top-1 docs | Task accuracy |
| --- | --- | --- | --- | --- |
| Qwen3.5-0.8B, 3.43 | 0.138 to 0.046 | 0.199 to 0.127 | 82.8 to 90.0 % | not run |
| Qwen3.5-4B, 3.0 | 0.161 to 0.061 | 0.226 to 0.195 | 82.9 to 89.6 % | +0.5 to +4.4 points, none above 2 standard errors |
| Qwen3.5-4B, 3.5 | 0.093 to 0.042 | 0.117 to 0.103 | 86.9 to 91.4 % | -0.8 to +2.0 points |

QAD at ratio 6 beats PTQ at ratio 4 on both texts with 0.27 bpw less (0.8B). Reducing the
step budget costs accuracy beyond noise (budget 0.5: 17 instead of 41 minutes on the
0.8B, KLD 0.099 against 0.077 on docs), so `default` keeps 300 steps per layer; `fast` is
the budgeted run. Distribution fidelity improves much more than multiple-choice accuracy:
HellaSwag and MMLU stay below EXL3 3.0 for every HI2 variant tested. Not done: QAD on a MoE
model, a final logit-KL phase, norm-weight training.
