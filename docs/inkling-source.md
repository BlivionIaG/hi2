# Inkling-2-Mini source reader (NVFP4 / MXFP8)

`hi2.inkling_source` reads the stock Inkling-2-Mini checkpoint, whose routed experts are
already quantized (NVFP4 in layers 0-19, MXFP8 in layers 20-23), and gives `hi2` the
dequantized float32 weights of one layer at a time. It is an optional source adapter:
no default path imports it, nothing in the CLI selects it, and a model of another family
is unaffected. The status of the quantization itself (allocation, PTQ, QAD, final KL) is
not in this note: it records the reader, the checkpoint it reads and the first
weight-only measurements behind the plan.

The teacher of a quantization from this source is the stock model dequantized, so the
KL that is measured is the fidelity to the stock quantized model, not to its bf16
ancestor.

## The checkpoint

`config.json` has hidden size 2304, 24 layers, 128 routed experts of intermediate size
1152, vocabulary 201,024 (200,058 unpadded), full attention on layers 5, 11, 17, 23.
The weights are in a *serving* layout (`model.llm.*`), one `model-00001-of-00001.safetensors`
plus `mtp.safetensors` (the multi-token-prediction head, not indexed).

| Tensor | Layers | Stored as |
| --- | --- | --- |
| `layers.N.mlp.experts.w13_weight` | 0-19 | U8 `[128, 2304, 1152]`, two e2m1 codes per byte |
| `w13_weight_scale` | 0-19 | E4M3 `[128, 2304, 144]`, one scale per 16 weights |
| `w13_weight_amax`, `w13_weight_outer_scale_denominator` | 0-19 | F32 `[128]`, F32 `[1]` (2688) |
| `w2_weight`, `w2_weight_scale`, `w2_weight_amax`, ... | 0-19 | U8 `[128, 2304, 576]`, E4M3 `[128, 2304, 72]`, as above |
| `w13_weight` / `w2_weight` | 20-23 | E4M3 `[128, 2304, 2304]` / `[128, 2304, 1152]` |
| `w13_weight_scale` / `w2_weight_scale` | 20-23 | UE8M0 U8 `[128, 2304, 72]` / `[128, 2304, 36]`, one per 32 weights |
| attention linears `wq_du wk_dv wv_dv wo_ud gate_attn wr_du` | all | bf16 |
| embed, unembed, router, loop controllers, short convs, norms, queries | all | bf16 (queries, router bias fp32) |
| `model.visual.*` | | bf16 (a 23-layer, 1024-wide vision tower) |

The `w13` stack holds gate and up interleaved, rows `[g0, u0, g1, u1, ...]`, in the
weights and in the scales alike. The two are one fused GEMM in the serving runtime, so
`hi2` needs gate-first rows and one transform for both.

**Dequantization**, per element, in float32:

* NVFP4: `e2m1(code) * e4m3(block scale) * (amax[e] / 2688)`. `amax / 2688` is divided
  in float64 and rounded once; the block scale times that factor is rounded to float32,
  then the element is one more float32 product (two roundings, in this order).
* MXFP8: `e4m3(code) * 2^(u - 127)`; `u` is the UE8M0 byte (255 is NaN and rejected).

Both are the arithmetic of the reference decoders of the vLLM fork
(`tools/rdna/inkling2/tiny_reference.py`, `ref_checkpoint.py`), so the result is bit
identical on any device.

## Using it

```python
from hi2.inkling_source import InklingSource

with InklingSource("/path/to/inkling2-mini", device="cuda:0") as src:
    src.formats()                       # ['nvfp4'] * 20 + ['mxfp8'] * 4
    gate, up = src.gate_up(3, experts=[0, 17])      # float32 [2, 1152, 2304] each
    down = src.down(3, experts=[0, 17])             # float32 [2, 2304, 1152]
    for e, g, u, d in src.iter_experts(20, chunk=16):
        ...                             # streams a layer, 16 experts decoded at a time
    gate_up, down = src.fused_layer(0, chunk=16, dtype=torch.float16)  # fused 3-D layout
    w = src.load("model.language_model.layers.5.self_attn.q_proj.linear.weight")
    w = src.load("model.language_model.layers.3.mlp.experts.17.gate_proj.weight")
```

* Only the requested experts are read (safetensors slices); their packed bytes are staged
  once per chunk through `hi2.pinned.PinnedWindow`, then decoded on `device`.
  `raw()` (read + stage) and `decode()` are public so a profile can time them apart.
* The decode is a 256-entry lookup (`F.embedding`, int32 indices) for e2m1 pairs and
  E4M3, and a bit-constructed `2^(u-127)`, so it needs neither an fp8 dtype nor `exp2` on
  the device (gfx1030 has no fp8).
* `fused_layer` returns the fused 3-D layout of transformers 5 (`gate_up [E, 2I, H]` with
  rows `[gate; up]`, `down [E, H, I]`) that `hi2.moe.unfuse_experts` takes; for one layer
  it materialises 4.1 GB in float32 (2.0 GB in float16). `iter_experts` never does.
* `expert_name` and `hf_expert_names` give the per-expert names `hi2.moe` has after
  `unfuse_experts`; `name_map` / `hf_names` map the serving names of the dense tensors
  to the Hugging Face model names (the map of the fork's `ref_checkpoint.build_name_map`).
* `tensor_role` classifies a serving name. The HI2 candidates are `expert` and
  `attn_linear`; norms, `attn_res` and cross-pass queries, `mlp.gate` (router and its fp32
  bias), `attn_sink_proj`, short convolutions, loop controllers and the vision tower are
  not candidates.

## Tests and experiments

`tests/test_inkling_source.py` (CPU, pure torch): the decoders are compared bit for bit with
an independent numpy implementation written from the format definitions; with the fork's
reference decoders (`tiny_reference.py`, `ref_checkpoint.py`) on the tiny checkpoint made by its
`make_tiny_checkpoint.py` (five NVFP4 layers and one MXFP8 layer) and, with `INKLING2_CKPT`
set, on real experts 0, 17, 64 and 127 of layers 0 (NVFP4) and 20 (MXFP8): the largest
absolute difference is 0.0. Tiny checkpoints written by the test (with and without an index
file) cover the de-interleave, chunk and expert-subset invariance, `fused_layer`, the name map
and the `UnfusedExperts` integration. `INKLING2_TOOLS` names the fork's `tools/rdna/inkling2`
directory when the sibling worktree is not at hand; without the fork the reference
comparisons are skipped.

| Script (`experiments/`) | What it records |
| --- | --- |
| `inkling_inventory.py` | Tensor classes, elements, bytes and stock bits per weight from the safetensors headers (main shard, MTP, vision), checked against the design review. |
| `inkling_scale_stats.py` | Histograms of the UE8M0 and E4M3 block scales and of the NVFP4 `amax`; the count of weights that overflow, underflow or fall in the subnormal range of float16 and bfloat16; the relative energy each cast changes. |
| `inkling_alloc_dryrun.py` | HI2's weight-only mode and its allocators on sampled real experts and attention linears: share of INT2/INT4/INT8, block mix, effective bits per weight, implied size. |
| `inkling_dequant_probe.py` | Exactness of the device decode against the CPU, and the cost (read, host-to-device, decode, whole layer, peak memory) of decoding one layer. |

## Measured on the stock checkpoint (2026-11-23)

One build box (CPU torch 2.12.0; the device figures on a Radeon PRO V620, gfx1030), raw output in [`experiments/results/inkling2_mini_2026-11-23/`](../experiments/results/inkling2_mini_2026-11-23/).

**Inventory** (`inkling_inventory.py`, safetensors headers only). Every number of the design
review is reproduced: NVFP4 experts 11,466,178,560 B (weights and scales) at 4.500 bits per
weight, MXFP8 experts 4,204,265,472 B at 8.250, 128 x 7,962,624 weights per layer.

| Class | Tensors | Weights | Bytes | bits/weight |
| --- | --- | --- | --- | --- |
| experts NVFP4 (layers 0-19) | 160 | 20,384,317,440 | 11,466,199,200 (incl. 20,640 of amax and denominators) | 4.500 |
| experts MXFP8 (layers 20-23) | 16 | 4,076,863,488 | 4,204,265,472 | 8.250 |
| attention linears | 144 | 644,087,808 | 1,288,175,616 | 16 |
| embed, unembed | 1 + 1 | 463,159,296 each | 926,318,592 each | 16 |
| vision tower | 422 | 318,256,128 | 636,512,256 | 16 |
| MTP file (3 dense layers 191.1 M, attention 70.8 M, Markov head 56.0 M) | 54 | 317,897,872 | 635,795,744 | 16 |
| loop controllers, router, convs, norms, queries, sinks | 500 | 38,797,824 | 78,081,024 | 16 / 32 |

The main file is 19,525,870,752 B (18.18 GiB, 26.39 B parameters with the vision tower); the
design review's "26.4 B" is that file, without the 0.318 B of the MTP file, and its
"18.68 GiB stock, 14.05 GiB at 3.5 bpw" are the main file plus the MTP file minus the 51.5 M
table `markov_head.markov_w1` (102,924,288 B). Experts alone at 3.5 bpw are 9.967 GiB.

**Every row has unit norm.** The rms of the dequantized `w13` weights of each of the 24 layers is
0.020826-0.020833 (1/sqrt(2304) = 0.020833) and that of `w2` 0.029446-0.029462
(1/sqrt(1152) = 0.029463); the rows of expert 17 of layers 0, 12, 20, 23 have L2 norm 0.983-1.020
and those of the attention linears of layers 5, 12, 20 0.9988-1.0010. So the error of a
tensor relative to its energy is the mean angular error of its rows, whatever the layer.

**Scales and the range of a half-precision teacher** (`inkling_scale_stats.py`, every weight of
every expert, decoded exactly):

* NVFP4 layers: the E4M3 block scale spans 6 to 448 (median 60-160, 1 % quantile 24-64,
  99 % quantile 160-288; 7,489 of the 1.27 G scales sit at 448), no zero, subnormal or NaN
  code; `amax` per expert 0.10-0.74 for `w13`, 0.14-0.85 for `w2`, so `amax / 2688` is
  3.8e-5 to 3.2e-4. 6.9 % of the weights are exactly zero.
* MXFP8 layers: the UE8M0 exponent spans only -14 to -9 (80 % of `w13` scales are 2^-13, 83 %
  of `w2` scales 2^-12); no byte 0 or 255. The E4M3 codes carry the information.
* Largest weight 0.852 (layer 17); the smallest non-zero magnitude is in [2^-22, 2^-21). No
  weight overflows float16 or bfloat16 and none underflows to zero in either. 8,345,812 weights
  (0.2 % of the MXFP8 layers, none in the NVFP4 layers) are subnormal in float16 and none in
  bfloat16.
* Casting the float32 teacher: MXFP8 layers are exact in both float16 and bfloat16 (measured
  zero error: every value is a 4-bit significand times a power of two). NVFP4 layers change
  a relative energy of 4.3e-8 in float16 (relative RMS 2.1e-4) and 2.8e-6 in bfloat16
  (1.7e-3).

So the *weights* are usable in float16, the half type that has a GEMM on gfx1030 (bfloat16 has
none there, see `moe.md`), at a cast error far below any 2-4 bit quantization error. That says
nothing about the activations of the loop layers (3 passes), which are not measured here: with
unit-norm rows, soft-capped gate/up activations and normalized inputs the linear outputs are
structurally bounded [inference], so a float16 teacher forward against a float32 one is the
check to make.

**Weight-only allocation** (`inkling_alloc_dryrun.py`). HI2 has a data-free mode:
`build_table(stats=None)` (what `hi2 quantize` does without `--calib`) scores every
`(nbits, block)` of every row with identity column weights in the rotated basis. The script runs
it, and `allocate_global` / `allocate_by_quality` unmodified, on 8 experts (0, 18, ..., 127) of
layers 0, 6, 12, 17 (standing for the 20 NVFP4 layers) and 20, 23 (the 4 MXFP8 layers) and on
their attention linears, each sampled tensor tiled to the number of real tensors it stands for
(so the bit budget is the model's) and weighted as `hi2.moe.expert_weights` does. First-order:
no activations, no Hessian, no routing statistics, no KL; the split that matters needs
calibration forward passes through the model. Experts alone (attention and the rest bf16), INT2
/INT4/INT8 candidates, by weight:

| Target | INT2 | INT4 | INT8 | block mix (INT2/B16, B32, B64; INT4/B64) | experts | Main file | Review total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3.0 bpw | 77.8 % | 22.2 % | 0 | 17.8, 57.0, 2.9; 22.2 | 8.54 GiB | 12.13 GiB | 12.63 GiB |
| 3.5 bpw | 47.7 % | 52.3 % | 0 | 11.8, 34.4, 1.5; 52.3 | 9.97 GiB | 13.56 GiB | 14.05 GiB |
| 4.0 bpw | 17.2 % | 82.8 % | 0 | 4.4, 12.3, 0.5; 82.8 | 11.39 GiB | 14.98 GiB | 15.48 GiB |

("Main file" is the experts plus every other main-file tensor as stored, vision included;
"Review total" adds the MTP file without its Markov table, the review's definition.)
The default flow, INT2/INT4 and a quality ratio, lands at 3.98 bpw for ratio 6 and 4.13 bpw
for ratio 4 (11.33 and 11.76 GiB of experts).

What it shows, and what it cannot: the weight-space relative error is 6.4e-2, 4.2e-2 and 2.0e-2
at 3.0, 3.5 and 4.0 bpw, and it is the same for NVFP4 and MXFP8 layers and for the bf16
attention linears (every uniform candidate has the same error in all four classes; INT2/B16
6.5e-2, INT4/B64 6.7e-3). The allocation is the time-sharing of two uniform candidates
(INT2/B32 at 2.53 bpw and INT4/B64 at 4.28 bpw) to within a few percent: its error is -2.3 %,
-0.9 % and +4.1 % of the straight line between them at the achieved bpw. The layers differ by
at most 0.05 bpw, `down` differs from `gate`/`up` by +0.07, -0.09 and -0.19 bpw (the same
tensor weight over twice the rows), and INT8 is never chosen for an expert at these targets.
With the attention linears in the same allocation they take 7.7-8.3 bpw (INT8/B64, 0.6 GiB)
and the experts 2.88 / 3.38 / 3.89 bpw at the same targets; that follows from giving each
dense tensor the weight of a whole expert group, not from the data. Weights alone have no
layer or row structure for the allocator to use; whatever mixing gains is in the
activation-weighted cost.

## Known gaps before a quantization run

* `hi2.moe.UnfusedExperts.forward` is a plain SwiGLU. Inkling's experts soft-cap the gate and
  up activations (`mlp_gate_softcap` 4.0, `mlp_up_softcap` 25.0 in `config.json`) and the
  router scales its weights (`route_scale` 8.0); a teacher, or a forward through that view,
  that ignores them is not the stock model. Use the model's own experts module as the teacher.
* `hi2.moe.check_unfused` compares the unfused views with per-expert checkpoint tensors;
  this checkpoint has fused serving tensors, so the comparison is `tests/test_inkling_source.py`'s
  (and the reader's own exactness tests), not that function.
* The Hugging Face class of the model (`Inkling2LoopForConditionalGeneration`) is not in
  the pinned transformers; the fork's tree supplies it.
