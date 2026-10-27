# HI2 (`rdna_hint_flex`) against EXL3, GGUF and AWQ: what we measured

Snapshot of 2026-10-26. Every number is from this project's runs on 4x Radeon PRO V620
(gfx1030); "n/m" means not measured. Perplexity ratio (PPL x) is the quantized perplexity
divided by the unquantized perplexity on the same text and the same evaluator.
KLD is in nats/token against the unquantized model's logits (llama.cpp rows: against its
own F16 logits, so KLD is comparable across formats but PPL is not).

Reading guide: "docs" is the repo-docs held-out text, which the early HI2 runs were also
*calibrated* on (flattering); "mix" is a held-out chat, instruction and tool-calling text;
multiple-choice scores are out of domain for everything and are the fairest signal.

## Qwen3.5-4B (unquantized PPL 7.067 on docs)

Sizes: "body" is the quantized decoder linears only; "file" is the whole checkpoint
(HI2 and EXL3 keep the tied embedding in fp16 / 6 bpw and carry a vision tower; the GGUF
file quantizes the embedding, so file sizes are not like-for-like).

| Format | bpw (body) | Body MiB | File MiB | PPL x | KLD | Top-1 | ARC-E | ARC-C | HellaSwag | MMLU | Decode tok/s c=1 | Decode complexity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dense fp16 | 16 | n/a | 8 800 | 1.000 | 0 | 100 % | 72.3 | 42.5 | 53.3 | 36.6 | 38.1 | rocBLAS fp16 |
| HI2 3.0 (docs-calibrated) | 3.0 | 1 287 | 3 368 | 1.103 | 0.131 | 84.4 % | 67.7 | 36.5 | 44.6 | 32.2 | 78.4 | extract + fma + dot2; act. pre-pass |
| HI2 3.5 | 3.5 | 1 499 | 3 580 | 1.054 | 0.077 | 88.0 % | 71.6 | 43.5 | 48.1 | 34.0 | n/m | same |
| HI2 4.0 | 4.0 | 1 712 | 3 793 | 1.025 | 0.034 | 92.2 % | 72.5 | 44.1 | 50.9 | 35.4 | n/m | same |
| EXL3 3.0 (mul1) | 3.0 | 1 336 | 3 402 | 1.052 | n/m | n/m | 69.6 | 42.1 | 51.3 | 35.9 | n/m | trellis codebook (4-5 ops/weight) + two 128-wide Hadamards |

llama.cpp harness (own F16 baseline in the first row; PPL x from llama-perplexity, second
half of each window):

| Format | bpw (file avg) | File MiB | PPL x | KLD | Top-1 | ARC-E | ARC-C | HellaSwag | MMLU | Decode complexity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GGUF F16 | 16 | 8 254 | 1.000 | 0 | 100 % | 65.4 | 47.2 | 71.0 | 39.0 | n/a |
| GGUF Q2_K | 3.60 | 1 858 | 1.220 | 0.259 | 79.8 % | 64.0 | 44.1 | 64.7 | 35.3 | superblock scale unpack + q8 activations |
| GGUF Q3_K_S | 3.90 | 2 013 | 1.104 | 0.143 | 85.3 % | 64.7 | 43.8 | 67.8 | 37.8 | same |
| GGUF Q3_K_M | 4.27 | 2 201 | 1.066 | 0.100 | 86.9 % | 64.2 | 46.8 | 68.9 | 37.3 | same |
| GGUF IQ4_XS | 4.75 | 2 449 | 1.020 | 0.041 | 91.3 % | 66.3 | 45.8 | 70.7 | 38.6 | same |
| GGUF Q4_K_M | 5.13 | 2 644 | 1.012 | 0.033 | 92.3 % | 66.1 | 45.5 | 71.0 | 39.0 | same |
| GGUF IQ2_M | 3.44 | 1 773 | n/m | 0.270 | 78.9 % | 60.0 | 41.8 | 65.8 | 36.1 | codebook (i-quant) |
| GGUF IQ3_XXS | 3.74 | 1 928 | n/m | 0.156 | 83.6 % | 61.6 | 43.8 | 68.2 | 38.1 | codebook (i-quant) |
| GGUF IQ3_XS | 4.06 | 2 093 | n/m | 0.116 | 86.5 % | 62.5 | 41.5 | 69.9 | 37.3 | codebook (i-quant) |
| GGUF IQ3_S | 4.03 | 2 080 | n/m | 0.103 | 87.0 % | 66.1 | 44.5 | 70.5 | 37.8 | codebook (i-quant) |
| GGUF IQ3_M | 4.08 | 2 104 | n/m | 0.100 | 87.7 % | 66.8 | 45.5 | 70.3 | 37.3 | codebook (i-quant) |

The IQ2_M, IQ3_XXS and IQ3_XS rows keep the Qwen3.5 MTP layer (blk.32) at q8_0, because
llama.cpp refuses very low-bit types for a tensor with no importance data (the MTP layer
is never run by the imatrix pass); their file sizes are a little inflated by that.

### Qwen3.5-4B with mixed calibration, PTQ and QAD (vLLM harness, same text sets)

Quality-driven allocation; `bpw` is the HI2 tensors only; docs/mix are the two held-out
texts (unquantized PPL 7.067 on docs); QAD = 300 steps per layer, 540k tokens, layer-wise.

| Run | bpw | INT2 / INT4 rows | PPL x docs | KLD docs | Top-1 docs | ARC-E | ARC-C | HellaSwag | MMLU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PTQ, mixed calibration | 3.0 | 68 % / 32 % | 1.138 | 0.161 | 82.9 % | 67.7 | 37.1 | 45.1 | 32.6 |
| PTQ + QAD | 3.0 | 68 % / 32 % | 1.030 | 0.061 | 89.6 % | 69.1 | 41.5 | 46.5 | 33.1 |
| PTQ + QAD, step-unit latents | 3.0 | 68 % / 32 % | 1.034 | 0.068 | 89.4 % | 68.4 | 41.5 | 46.95 | 33.15 |
| PTQ, mixed calibration | 3.5 | 44 % / 56 % | 1.075 | 0.093 | 86.9 % | 68.4 | 40.5 | 48.6 | 35.4 |
| PTQ + QAD | 3.5 | 44 % / 56 % | 1.017 | 0.042 | 91.4 % | 70.4 | 41.1 | 49.0 | 34.6 |
| PTQ + QAD, step-unit latents | 3.5 | 44 % / 56 % | 1.022 | 0.045 | 91.2 % | 68.8 | 40.8 | 50.3 | 34.5 |

QAD cuts docs KLD by 60 to 55 % and recovers 1.5 to 4 ARC-Challenge points, but the
multiple-choice gains are small (about 1 to 2 points, near the noise of 2 to 3 points):
distribution fidelity improves much more than task accuracy. HellaSwag and MMLU stay
below EXL3 3.0 (51.3 and 35.9) for every HI2 variant at 3.0 and 3.5 bpw.

### Qwen3.5-4B at equal body size (same 3 569.1 M weights, bytes counted the same way)

The tables above mix file sizes that are not like-for-like: GGUF quantizes the tied
embedding and HI2 keeps it in fp16 (1 220 MiB). This table counts only the quantized decoder
linears, with every byte the loader reads (HI2: codes, block parameters, row constants,
smoothing vector, row index, from `hi2 size`; GGUF: the tensor sizes `llama-quantize`
printed for the same 3 569.1 M weights). KLD and top-1 are against each format's own
unquantized model on the same kind of held-out text, so they compare across formats;
multiple-choice scores do not (different harnesses).

| Format | Body MiB | Body bpw | KLD | Top-1 |
| --- | ---: | ---: | ---: | ---: |
| GGUF IQ2_M | 1 230 | 2.89 | 0.270 | 78.9 % |
| HI2 3.0, QAD | about 1 290 | 3.0 | 0.061-0.068 | 89.4-89.6 % |
| GGUF Q2_K | 1 315 | 3.09 | 0.259 | 79.8 % |
| EXL3 3.0 | 1 336 | 3.0 | n/m | n/m |
| GGUF IQ3_XXS | 1 385 | 3.26 | 0.156 | 83.6 % |
| GGUF Q3_K_S | 1 463 | 3.44 | 0.143 | 85.3 % |
| GGUF IQ3_XS | 1 470 | 3.46 | 0.116 | 86.5 % |
| **HI2 3.5, QAD** | **1 499** | **3.52** | **0.042-0.045** | **91.2-91.4 %** |
| GGUF IQ3_S | 1 529 | 3.59 | 0.103 | 87.0 % |
| GGUF IQ3_M | 1 552 | 3.65 | 0.100 | 87.7 % |
| GGUF Q3_K_M | 1 646 | 3.87 | 0.100 | 86.9 % |
| HI2 4.0, PTQ (docs-calibrated) | 1 712 | 4.0 | 0.034 | 92.2 % |
| GGUF IQ4_XS | 1 886 | 4.43 | 0.041 | 91.3 % |
| GGUF Q4_K_M | 2 072 | 4.87 | 0.033 | 92.3 % |

At equal body bytes, HI2 with QAD has less than half the KLD of the GGUF types around 3.5
bpw. HI2 3.5 reaches the KLD of IQ4_XS (0.041) with 21 % fewer body bytes. HI2 3.0 with QAD
(about 1 290 MiB) beats every GGUF type up to Q3_K_M (1 646 MiB) on KLD. The GGUF rows use
llama.cpp quantization with an importance matrix.

## Smaller models are more sensitive (same PTQ recipe, about 3.0 to 3.5 bpw)

| Model | Params | HI2 bpw | PPL x docs | KLD docs | KLD mix | Note |
| --- | --- | --- | --- | --- | --- | --- |
| OLMoE-1B-7B (MoE) | 7 B (1 B active) | 3.54 | 1.011 | 0.017 | 0.048 | quality allocation |
| OLMoE-1B-7B (MoE) | 7 B | 3.00 | 1.021 | 0.031 | 0.091 | |
| Qwen3.5-4B | 4 B | 3.0 / 3.5 | 1.103 / 1.054 | 0.131 / 0.077 | n/m | docs-calibrated |
| Qwen3.5-0.8B | 0.8 B | 3.43 | 1.153 | 0.138 | 0.199 | mixed calibration |
| Qwen3.5-0.8B | 0.8 B | 3.0 | 1.207 | 0.196 | 0.650 | docs-calibrated |
| Qwen2.5-0.5B | 0.5 B | 3.0 | 1.220 | n/m | n/m | |

EXL3 at 3.0 bpw shows the same trend: perplexity x 1.052 on the 4B and 1.117 on the 0.8B.

## Qwen3.5-0.8B in detail (unquantized PPL 10.874 docs, 5.483 mix)

| Variant | bpw | PPL x docs | KLD docs | Top-1 docs | PPL x mix | KLD mix | Top-1 mix |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HI2 3.0, docs calibration, sensitivity | 3.0 | 1.207 | 0.196 | 80.3 % | 2.049 | 0.650 | 66.5 % |
| HI2 3.0, mixed calibration, sensitivity | 3.0 | 1.268 | 0.234 | 78.6 % | 1.509 | 0.364 | 73.9 % |
| HI2 quality mode (ratio 6), mixed calibration, PTQ | 3.43 | 1.153 | 0.138 | 82.8 % | 1.265 | 0.199 | 80.7 % |
| same + QAD (300 steps/layer, 540k tokens) | 3.43 | 1.030 | 0.046 | 90.0 % | 1.155 | 0.127 | 85.0 % |
| HI2 4.0 (docs calibration) | 4.0 | 1.068 | 0.069 | n/m | n/m | n/m | n/m |
| HI2 4.5 (docs calibration) | 4.5 | 1.020 | 0.019 | n/m | n/m | n/m | n/m |
| EXL3 3.0 (vLLM) | 3.0 | 1.117 | n/m | n/m | n/m | n/m | n/m |
| EXL3 4.0 (vLLM) | 4.0 | 1.028 | n/m | n/m | n/m | n/m | n/m |
| EXL3 2.0 (vLLM) | 2.0 | collapsed (PPL 2.9 M) | | | | | |

Calibration domain matters: the docs-calibrated 3.0 run has 3x the KLD on chat and
tool text than on docs (0.650 against 0.196); mixed calibration closes that to 0.364 at a
cost of 20 % in-domain.

## Qwen2.5-0.5B against AWQ (fork vLLM, held-out text, unquantized PPL 10.49)

| Format | bpw | PPL | Decode tok/s c=1 | Decode complexity |
| --- | --- | --- | --- | --- |
| dense fp16 (unquantized) | 16 | 10.49 | 163.7 (speed measured on the dense export of the HI2 weights, same shapes) | rocBLAS fp16 |
| HI2 3.0 | 3.0 | 12.80 | 176.7 | extract + fma + dot2 |
| AWQ int4 g128 | 4.25 | 12.03 | 114.8 | int4 dequant + scale |

## OLMoE-1B-7B (unquantized PPL 6.634 docs, 3.924 mix; whole file 13.84 GB)

| Run | bpw (HI2 tensors) | File | INT2 / INT4 rows | PPL x docs | KLD docs | Top-1 | PPL x mix | KLD mix | Top-1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ratio 4 | 3.54 | 3.44 GB | 40 % / 60 % | 1.011 | 0.0169 | 94.4 % | 1.047 | 0.0480 | 90.2 % |
| 3.0 bpw | 3.00 | 2.98 GB | 67 % / 33 % | 1.021 | 0.0313 | 91.8 % | 1.093 | 0.0910 | 87.0 % |

Served in the fork's vLLM (TP=1 on one V620, full cudagraphs, tuned rows, exclusive GPU; OLMoE's
context is 4096, so 1k/512 and 2k/512). Perplexity through the fork matches the tool's HF
numbers to within 0.03 % (r4: 6.7071 docs, 4.1063 mixed). Per request; wave aggregate in
brackets for prefill and decode:

| Cell | Arm | Prefill tok/s | Decode tok/s | TTFT s | ITL ms |
| --- | --- | --- | --- | --- | --- |
| 1k c=1 | HI2 r4 (3.54 bpw) | 5 096 | 209.1 | 0.20 | 4.8 |
| 1k c=1 | HI2 3.0 bpw | 4 989 | 206.4 | 0.21 | 4.6 |
| 1k c=1 | dense fp16, default MoE backend | 1 371 | 98.8 | 0.75 | 10.1 |
| 1k c=1 | dense fp16, Triton MoE backend | 3 917 | 47.6 | 0.26 | 21.0 |
| 1k c=8 | HI2 r4 | 1 552 | 59.2 | 0.66 | 15.6 |
| 1k c=8 | dense, default backend | 369 | 36.2 | 2.77 | 22.8 |
| 2k c=1 | HI2 r4 | 8 391 | 189.7 | 0.24 | 5.2 |
| 2k c=1 | dense, default backend | 2 588 | 93.8 | 0.79 | 10.5 |
| 2k c=8 | HI2 r4 | 2 420 | 48.9 | 0.85 | 19.1 |
| 2k c=8 | dense, default backend | 733 | 32.2 | 2.79 | 25.5 |

HI2 decodes 2.1x the faster dense backend at c=1 and 1.6x per request at c=8, and its
prefill is ahead of both dense MoE backends once expert tokens are grouped per expert
(without grouping, prefill was 3.8 to 4.4x slower).

## Speed of HI2 against dense (Qwen3.5-4B, tuned, full cudagraphs, per request)

| Cell | TP | HI2 prefill tok/s | HI2 decode tok/s | HI2 TTFT s | HI2 ITL ms | Dense decode tok/s | Decode x |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1k/512 c=1 | 1 | 1 932 | 78.4 | 0.53 | 12.8 | 38.1 | 2.06 |
| 1k/512 c=1 | 2 | 3 436 | 102.7 | 0.30 | 9.7 | 61.4 | 1.67 |
| 1k/512 c=1 | 4 | 6 541 | 113.9 | 0.16 | 8.8 | 93.3 | 1.22 |
| 1k/512 c=8 | 1 | 410 | 35.4 | 2.50 | 26.8 | 27.8 | 1.27 |
| 16k/1k c=1 | 1 | 1 407 | 57.8 | 11.65 | 17.3 | 32.6 | 1.77 |
| 16k/1k c=8 | 1 | 583 | 10.1 | 28.10 | 57.6 | 9.5 | 1.06 |

Prefill is within 4 % of dense everywhere (dequant + GEMM; a fused F16 kernel and an int8
prefill kernel exist but do not beat tuned rocBLAS). Decode wins come from reading 3 to 4
bits per weight; the gain shrinks at high TP and high concurrency, where decode is no longer
bandwidth bound.

## Inference complexity, side by side

| Format | Per-weight decode work | Side computation | Kernel variants | Measured here |
| --- | --- | --- | --- | --- |
| HI2 | code extract, `lo + code*step` fma, `v_dot2_f32_f16` (F16 mode) | divide by smoothing vector + 16-point Hadamard per layer input | 2 widths x 3 blocks (+ INT8) | decode 2.06x dense (4B, TP=1) |
| HI2 I8 mode | code to int8, `v_dot4_i32_i8`, integer epilogue per 64 columns | int8 activation quantization | as above | accuracy cost < 0.2 % PPL; kernel not benchmarked |
| EXL3 | window extract + procedural codebook (about 4 to 5 ops) + dot2 | two 128-wide Hadamards (input and output side) | one per bit width | n/m |
| GGUF K-quants | superblock scale unpacking, mixed int dot | q8 activation quantization | many types | n/m (llama-bench pending) |
| AWQ int4 | int4 dequant + scale fma | none | one | decode 0.65x of HI2 (0.5B) |

## Qwen3.5-0.8B: error-ratio sweep (mixed calibration, sensitivity probe, PTQ)

| Ratio | bpw | INT2 / INT4 rows | KLD docs | Top-1 docs | PPL docs | KLD mix | PPL mix |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4.17 | 13 % / 87 % | 0.041 | 90.6 % | 11.41 | 0.056 | 5.87 |
| 3 | 3.89 | 27 % / 73 % | 0.069 | 87.6 % | 11.69 | 0.094 | 6.15 |
| 4 | 3.70 | 37 % / 63 % | 0.093 | 85.7 % | 11.92 | 0.130 | 6.42 |
| 6 | 3.43 | 51 % / 49 % | 0.138 | 82.8 % | 12.54 | 0.199 | 6.94 |
| 8 | 3.23 | 61 % / 39 % | 0.177 | 80.6 % | 13.05 | 0.265 | 7.43 |
| 12 | 2.91 | 75 % / 25 % | 0.262 | 77.2 % | 14.22 | 0.412 | 8.74 |

KLD roughly doubles for every 0.3 to 0.5 bpw saved; there is no sharp knee between 2.9 and
4.2 bpw on this model.

## Quantizer speed (one V620, bit-identical outputs)

| Run | Old | New, empty cache | New, cache hit |
| --- | --- | --- | --- |
| Qwen3.5-0.8B PTQ | 754 s | 266 s | 28.5 s |
| Qwen3.5-4B PTQ | 3 073 s | 1 485 s | n/m |
| OLMoE encode | about 4 450 s | 192 s | n/m |
| 0.8B, six error ratios in one run | about 3 500 GPU-s | n/m | 84 s |

QAD adds about 41 minutes (0.8B) and 3.0 to 3.3 hours (4B) on one GPU at the default
300 steps per layer; the `fast` preset is about 17 minutes and 1.3 hours.

## What to take from it

* HI2 4.0 matches Q4_K_M on KLD and loses nothing beyond noise on the tasks; HI2 3.5 sits
  between Q3_K_M and IQ4_XS; HI2 3.0 with docs calibration is the weak point.
* Mixed calibration and QAD are the two measured levers for the low end (0.8B: KLD 0.138 to
  0.046 at 3.43 bpw with QAD).
* Smaller models need more bits for the same fidelity; the quality-driven allocation
  (`--max-error-ratio`) adapts bpw to the model, but the ratio itself is not yet tuned and
  is relative to each model's own best achievable error.

## The default quality target

`hi2 quantize` uses `--max-error-ratio 6` when QAD runs and `4` for PTQ only. The ratio is
relative to the best error each model can reach, so one ratio gives a similar quality on
every model and a different bit rate.

What the runs say (docs text, unquantized perplexity as the reference):

| Run | bpw | PPL x | KLD | Note |
| --- | --- | --- | --- | --- |
| 0.8B, ratio 6, PTQ | 3.43 | 1.153 | 0.138 | mixed calibration |
| 0.8B, ratio 6, QAD | 3.43 | 1.030 | 0.046 | the default recipe |
| 0.8B, ratio 4, PTQ | 3.70 | 1.096 | 0.093 | the PTQ-only default |
| 0.8B, ratio 8, PTQ | 3.23 | 1.200 | 0.177 | 0.2 bpw less, 28 % more KLD than ratio 6 |
| 4B, ratio 6.2, PTQ | 3.50 | 1.075 | 0.093 | |
| 4B, ratio 6.2, QAD | 3.50 | 1.017-1.022 | 0.042-0.045 | the default recipe |
| 4B, 3.0 bpw, QAD | 3.00 | 1.030-1.034 | 0.061-0.068 | |
| EXL3 3.0 (4B) | 3.0 | 1.052 | n/m | body 1 336 MiB against 1 499 MiB for ratio 6.2 |
| GGUF Q3_K_M (4B) | 4.27 (file) | 1.066 | 0.100 | file 2 201 MiB against about 3 580 MiB for HI2 with the fp16 embedding |

Why 6 with QAD:

1. **Better than GGUF at equal body size.** At ratio 6 with QAD, the 4B model has a KLD of
   0.042-0.045 with 1 499 MiB of body. GGUF IQ3_XS (1 470 MiB) has 0.116, IQ3_M
   (1 552 MiB) has 0.100, and Q3_K_M (1 646 MiB) has 0.100 (table above). Only IQ4_XS
   (0.041) matches it, with 21 % more body bytes.
2. **Between GGUF and EXL3.** The perplexity ratio (1.02) is better than EXL3 3.0
   (1.052) for 12 % more body bytes. ARC-Easy is level (68.8-70.4 against 69.6).
   HellaSwag (49.0-50.3 against 51.3) and MMLU (34.5-34.6 against 35.9) stay a little
   lower. Closing that gap needs about 4 bpw, which is the price of a middle-ground format
   and of its cheaper decode.
3. **No knee in the curve.** KLD about doubles for every 0.3-0.5 bpw saved between 2.9 and
   4.2 bpw. Ratio 6 is a choice of point on that line, not a measured optimum. Ratio 4
   costs about 8 % more bits for a third less KLD, ratio 8 saves 6 %.
4. **PTQ alone needs more bits.** Without QAD, ratio 6 gives 1.075 x on the 4B model,
   worse than EXL3 3.0. The PTQ-only default is 4.

What the default does not cover: OLMoE was measured at ratio 4 (3.54 bpw) and at a 3.0 bpw
target, not at ratio 6; MoE QAD was not run; the 0.5B model was run only at a 3.0 bpw
target. Choose another ratio when your quality or size limit differs.

## Decision fidelity on Qwen3.5-4B (spec 12.3)

The question: when an agent loop gives the model a coding task, does the quantized model
make the same first decision as the unquantized one? The prompt set has 66 conversations over
this repository's files: 12 tasks at five stages (fresh, after a directory listing, after
reading a file, after a failing test run, after a passing run) and 6 vague requests. The model
must answer with one XML tool call. Decoding is greedy at concurrency 1 in the fork's vLLM,
with thinking off. A run scores the first tool called (or none, when the model answers in
prose), and the same tool with the same argument.

The baseline is the unquantized model booted twice. Both boots made the same first call on
all 66 prompts, so greedy decoding adds no spread on this set, and every difference below is
a difference of the model. With 66 prompts the 95 % interval is about 10 points wide.

| Model (4B) | Same first tool | 95 % interval | Same tool and argument | Well-formed calls | Garbage |
| --- | ---: | ---: | ---: | ---: | ---: |
| unquantized, second boot | 100 % | 94-100 % | 100 % | 72.7 % | 0 |
| HI2 3.5 bpw, QAD, step-unit latents | 77.3 % | 66-86 % | 54.5 % | 60.6 % | 0 |
| HI2 3.5 bpw, QAD, weight-unit latents | 74.2 % | 63-83 % | 54.5 % | 71.2 % | 0 |
| HI2 3.5 bpw, PTQ only | 72.7 % | 61-82 % | 50.0 % | 59.1 % | 0 |
| HI2 3.0 bpw, QAD, step-unit latents | 66.7 % | 55-77 % | 47.0 % | 65.2 % | 0 |

Read this as follows.

1. **About three in four first decisions match at 3.5 bpw.** The first tool agrees on 73-77 %
   of the prompts, and on 50-55 % when the argument must match too (the same file or command).
2. **The variants are not separable at this size.** QAD, PTQ-only and the 3.0 bpw run lie inside
   each other's intervals. QAD does not show a gain on decisions that it shows on KLD (0.093 to
   0.045). A larger prompt set is needed to say more.
3. **The baseline is a weak tool user.** The unquantized 4B model makes a well-formed call
   on only 72.7 % of the prompts; on 27 % it answers in prose or breaks the format. Part of
   each disagreement is a format difference, not a wrong action. No run produced garbage.
4. **The per-stage numbers are too small to read.** Each stage has 12 prompts (6 for the vague
   requests), so one cell moves 8 points per prompt. Agreement by stage runs from 42 % to
   100 % with no consistent pattern; the vague requests agree on every model but the 3.0 bpw
   one (83 %).

Raw decisions of every run are in
[`experiments/results/decision_fidelity_4b_2026-10-27/`](../experiments/results/decision_fidelity_4b_2026-10-27/);
the harness is `experiments/decision_fidelity.py` (see [experiments](../experiments/README.md)).
Not measured: other formats (EXL3, GGUF) on the same prompts, and the 27B model.

## Qwen3.8-27B at 3.0 bpw

The last column is the 4-bit AWQ checkpoint `cyankiwi/Qwen3.8-27B-AWQ-INT4` (group 32), served the same way. It is about 1.5 bpw larger, so it is a quality ceiling for a 4-bit format and not a same-size comparison.

Qwen3.8-27B (BF16, 64 layers, 24.35 B weights in the 496 quantized linears) quantized with
`experiments/run_27b.sh`: target 3.0 bpw, the mixed calibration text, the sensitivity probe,
four V620s for the calibration, probe and encode, and one for QAD. Served in the fork's vLLM
(fp16), one V620 for the quantized models and four (TP=4) for the BF16 baseline.

| | BF16 | HI2 PTQ | HI2 PTQ + QAD | AWQ INT4 (cyankiwi) |
| --- | ---: | ---: | ---: | ---: |
| Quantized linears | 45.4 GiB | 8.5 GiB | 8.5 GiB | not measured |
| Measured bpw (every byte of the linears) | 16 | 3.012 | 3.012 | about 4.5 |
| INT2 / INT4 share of the weights | | 66 % / 34 % | 66 % / 34 % | INT4 only |
| Checkpoint | 51.7 GiB | 14.9 GiB (3.47x smaller) | 14.9 GiB | 20 GB |
| Perplexity, docs text | 4.107 | 4.307 (x 1.049) | 4.282 (x 1.043) | 4.189 (x 1.020) |
| Perplexity, mix text | 5.162 | 4.835 (x 0.937) | 5.574 (x 1.080) | 5.219 (x 1.011) |
| First tool as BF16 (66 agent prompts) | 100 % | 54.5 % (43-66) | 66.7 % (55-77) | **81.8 %** (71-89) |
| Same tool and argument | 100 % | 45.5 % | 56.1 % | **68.2 %** |
| Well-formed tool calls | 87.9 % | 74.2 % | 78.8 % | 83.3 % |

Read this as follows.

1. **The model works.** Both quantized models answer correctly ("Paris", arithmetic) and make
   well-formed tool calls on three prompts in four, with no garbage and no faults.
2. **QAD helps on the decisions.** It matches BF16's first decision on 12 points more prompts
   than PTQ (the intervals overlap; by stage QAD is ahead in four of six, level in one, behind in one) and makes more
   well-formed calls. On docs perplexity the gain is small (x 1.043 against x 1.049).
3. **The mix perplexity is not a fidelity measure here.** The PTQ model scores below BF16
   (x 0.937) on that text, so a ratio near 1 does not mean the same distribution. There is no
   KLD at this size: the BF16 model does not fit one GPU in fp32, which `hi2 eval` needs.
4. **QAD gained little per layer.** The `fast` preset spent 6 619 of 19 200 steps (7.2 h on one
   V620) and early-stopped at a median held-out layer-error gain of 1.5 % (layer 0: 24 %),
   against 15 to 38 % on the 0.8B. With bf16 hidden-state buffers (the fp32 ones need 41 GiB of
   RAM) it is likely undertrained rather than limited; the default preset (three times the steps)
   was not run.

Decisions and perplexities per run: `experiments/results/decision_fidelity_27b_2026-10-27/`
(the harness is `experiments/decision_fidelity.sh`). Not measured: the multiple-choice tasks,
EXL3 or GGUF at this size, and a QAD run with the default preset.

