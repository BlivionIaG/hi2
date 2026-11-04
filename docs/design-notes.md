# HI2 design notes: rotation, and where it stands against GGUF and EXL3

Companion to `format-spec.md` (the format) and `./README.md` (the
quantizer). Everything measured here is Qwen2.5-0.5B-Instruct on held-out text, one
run each, so differences under about 5% are noise. Raw data is in
`experiments/results/`.

## 1. What HI2 is

A weight format for RDNA2 (`gfx1030`): per row INT2 or INT4 codes, blocks of 16, 32
or 64 along the input dimension, an affine ladder `lo + code * step` whose two
parameters are stored as 8 bits relative to three fp16 per-row constants, an optional
per-column smoothing vector, and an optional order-16 Hadamard rotation of the input
dimension. Decode is fixed and branch-free, aimed at `v_dot2_f32_f16` (fp16
activations) or `v_dot4_i32_i8` (int8 activations). Calibration (activation
statistics, encoder, global bits-per-weight allocation) lives in the separate
`hi2` quantizer, not in vLLM.

## 2. Where rotation helps, and where it does not

A rotation spreads a few large weights over a group, which is good for a quantizer
whose step size is set by the block range. It also mixes columns, which is bad for a
quantizer that treats columns differently. What we measured:

| Encoder | rotation off | rotation on |
| --- | --- | --- |
| data-free (no activation statistics), 3.0 bpw | 44.28 ppl | **30.71 ppl** (-31%) |
| calibrated (activation-weighted), 3.0 bpw | **19.92 ppl** | 30.50 ppl (+53%) |
| calibrated, smoothing 0.5 first, 3.0 bpw | n/a | **18.31 ppl** (-8% vs the best unrotated) |

So:

* **Data-free: always rotate.** This is the case the literature (QuIP#, QuaRot,
  EXL3) describes, and it holds here.
* **Calibrated: the encoder already uses activation scale per channel**, similar in
  effect to AWQ. Rotating on top of that destroys what it exploits, so rotation
  alone is worse.
* **Calibrated with smoothing first: best.** Dividing each activation channel by
  `s_j = (E[x_j^2] / geomean)^0.5` (and multiplying the weight column by `s_j`) makes
  the channels comparable; the rotation then only has weight outliers left to flatten.
  This helped at every bit width: perplexity -30% at 2.5 bpw, -8% at 3.0, -5% at 3.5,
  -1% at 4.0, -1% at 4.5 (KL -39% at 4.5). Smoothing strength matters: 0.25 is worse
  than none, 0.5 and 0.75 are equivalent.
* **Group size and signs do not matter much.** After smoothing, groups of 16, 64 and
  128 are within 1%. Random signs (QuIP#-style) made no difference. Group 16 stays,
  because the activation transform is then four butterfly stages and a block never
  splits a group.

Runtime cost of the smoothing and rotation: one divide by `s` plus a 16-point
Walsh-Hadamard per token per layer, fused with the activation quantization in int8
mode. The smoothing vector costs `16/R` bpw per tensor (0.125 bpw for a 128-row
projection in this model, 0.004 at R=4096). q, k and v share an input and so share
`s`; gate and up likewise. Storing it once, or folding it into the preceding RMSNorm
weight (free at runtime), is the obvious next saving and is not implemented. `o_proj`
and `down_proj` inputs have no norm to fold into.

Not tried, and what the literature does that we do not: a full-dimension Hadamard
(EXL3, QuIP#) rather than 16 columns, rotating the residual stream once so most of
the rotations cost nothing at runtime (QuaRot, SpinQuant), and Hessian-aware rounding
(see 4).

## 3. Measured against GGUF and EXL3

Primary comparison: **Qwen3.5-0.8B** (hybrid linear/full attention, hidden 1024, MLP
3584, both multiples of 256 so GGUF needs no fallbacks), one GPU (gfx1030), same
held-out text for everything (8 x 2048 tokens at offset 800000 of the repo
documentation). The EXL3 checkpoint is the one already on the box
(`Qwen3.5-0.8B-exl3-3inst-good`, 3.0 bpw, 3inst codebook, 6-bit head, calibrated on
256 x 2048). Everything below is **perplexity divided by the same model's
unquantized perplexity**. A ratio of 1.00 means equal measured perplexity, not
identical distributions, generations or task accuracy. Cross-evaluator windows
remain a caveat; raw data is in `experiments/results/e2e_qwen3.5-0.8b.json`.

Evaluator check: vLLM (fork, fp16) scores the unquantized model 10.8736 and the HI2
export 20.8727; my HF evaluator scores them 10.8738 and 20.8734. EXL3 and HI2 were
scored in the same vLLM, the GGUF types and HI2 in the same `llama-perplexity`.

| Format | bits per weight (linear layers) | perplexity / unquantized |
| --- | --- | --- |
| **EXL3 3inst** | 3.0 (nominal) | **1.12** (vLLM) |
| HI2, Hessian-aware rounding | 2.5 | 1.53 (HF), 1.57 (llama.cpp) |
| HI2, Hessian-aware rounding | 3.0 | 1.32 (HF), 1.32 (vLLM), 1.33 (llama.cpp) |
| HI2, Hessian-aware rounding | 3.5 | 1.18 (HF), 1.20 (llama.cpp) |
| HI2, Hessian-aware rounding | 4.0 | 1.07 (HF), 1.07 (llama.cpp) |
| HI2, Hessian-aware rounding | 4.5 | 1.02 (HF), 1.02 (llama.cpp) |
| HI2, nearest rounding | 3.0 | 1.92 |
| HI2, nearest rounding | 4.0 | 1.19 |
| GGUF Q2_K | 3.26 | 1.74 |
| GGUF Q3_K_S | 3.46 | 1.37 |
| GGUF Q3_K_M | 3.96 | 1.27 |
| GGUF IQ4_XS | 4.57 | 1.044 |
| GGUF Q4_K_M | 4.97 | 1.042 |
| GGUF Q5_K_M | 5.76 | 1.012 |

GGUF's i-quants (IQ2 and IQ3) fail to quantize this model, and the ROCmFPX
`Q2_0_ROCMFPX` (2.5 bpw) scores a perplexity of 681, so neither is a data point.

What this says:

* **Hessian-aware rounding closed most of the gap to EXL3** (spec 8.3). At 3.0 bpw HI2
  went from 1.92 to 1.32, against EXL3's 1.12. HI2 reaches 1.12 at about 3.76 bpw, so
  EXL3 is now about 0.75 bpw (20%) smaller at equal quality, down from 1.2 bpw.
* **HI2 now beats GGUF's k-quants clearly**, in the same llama.cpp evaluator and
  interpolated to GGUF's bit widths: about 27% better than Q2_K, 12% better than
  Q3_K_S, 15% better than Q3_K_M, and slightly better than IQ4_XS and Q4_K_M with
  fewer bits.
* **Cost:** quantization takes about 3.7x longer than nearest rounding (367 s against
  99 s for this model on one gfx1030 GPU) and needs the full Hessians (about 2 GB of
  scratch for this model). It changes nothing at inference.
* Qwen2.5-0.5B (hidden 896) behaves the same way: 1.75 to 1.32 at 3.0 bpw and 1.15 to
  1.08 at 4.0 bpw. It cannot be used for the GGUF low-bit comparison, because its
  dimensions are not multiples of 256 and GGUF falls back to legacy types.

Caveats: one small hybrid model, one text, one run each. EXL3's "3.0" is its nominal
bit width and the checkpoint also carries per-tensor sign vectors and a 6-bit head;
HI2's bpw includes everything in the linear layers but keeps the embedding and head
dense fp16, so compare body quality, not file size. EXL3 calibrated on 524k tokens;
the HI2 runs above used 65k (524k gave 1% more). Speed and VRAM are not compared.

## 4. What is left between HI2 and EXL3

EXL3 (QTIP) is supported on this fork on RDNA2: native ROCm kernels
(`csrc/rocm/exl3_dot2_*.cu`, `exl3_hadamard.cu`: trellis GEMM, the 3inst and mul1
codebooks, the 128-wide Hadamard). (An earlier version of these notes called EXL3
CUDA-only; that was wrong.) The remaining gap in 3 (1.32 against 1.12 at 3.0 bpw) is
probably, in order of what I can say about it:

* **Codebook.** EXL3 uses a trellis codebook, which gets much closer to the
  rate-distortion bound at 2-3 bits than a uniform scalar ladder. Closing that needs a
  new code type and decode path, not a quantizer change, so it is out of scope for the
  current format. It is the part of the gap HI2's approach cannot recover.
* **Full-dimension rotation.** EXL3 rotates across 128-wide blocks over the matrix; HI2
  mixes 16 columns. Under compensated rounding, rotation matters less (the transform
  cells span 1.32-1.39), so this is probably a small part.
* **Joint ladder and rounding optimisation.** The PTQ encoder fits ladders before
  GPTQ rounding. Later QAD can train ladder bytes and codes, and local PTQ code
  refinement is implemented; a separate PTQ ladder refit after rounding remains
  a distinct proposal ([format-spec.md](format-spec.md), sections 8.9 and 13).

HI2 targets a decode path with a few fixed RDNA2 dot-product instructions rather
than a codebook lookup or trellis walk, input widths divisible by 64 and mixed
INT2/INT4/opt-in INT8 rows. The original PTQ study ran in minutes on one V620;
that is not a time estimate for default QAD plus final-KL. Native HI2 kernels
and loader measurements now exist in the external fork ([results.md](results.md)).
The earlier “no kernel/speed unmeasured” status is superseded, without implying
every model/layout/runtime revision is qualified.

## 5. Research directions and subsequent implementation status

The original priorities below motivated later work; distinguish that completed
work from proposals rather than treating this list as current operating guidance.

1. **Implemented: Hessian-aware rounding** (spec 8.3), the largest single gain
   after calibration in the initial study. Sequential propagation, more
   calibration tokens and a different smoothing exponent did not help there.
2. **Larger-model measurements now exist.** [results.md](results.md) includes 4B,
   OLMoE and 27B runs. Coverage is still finite; Hessian memory scales as `K^2`
   per distinct input, and the [phase resource plan](operations.md) still matters.
3. **Sensitivity-aware allocation is implemented** (spec 8.6). It did not make
   flex beat uniform INT4/B64 at 4.3 bpw in the original study.
4. **Local code refinement is implemented** (spec 8.9). A separate ladder refit
   after PTQ rounding is a proposal, not an enabled encoder mode.
5. **Shared transforms are implemented** for shared inputs; storing their
   smoothing vector once or folding it into RMSNorm remains distinct work.
6. **Native kernels/loader have measured results** in the external fork.
   Reproducible runtime qualification remains a [separate handoff](getting-started.md#export-and-native-serving).
7. **A trellis or vector code type**, if the remaining 0.75 bpw against EXL3 matters
   enough to change the format.
