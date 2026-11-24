# Inkling-2-Mini source reader (NVFP4 / MXFP8)

`hi2.inkling_source` reads the stock Inkling-2-Mini checkpoint, whose routed experts are
already quantized (NVFP4 in layers 0-19, MXFP8 in layers 20-23), and gives `hi2` the
dequantized float32 weights of one layer at a time. It is an optional source adapter: no
flag selects it, a model of another family never reaches it, and `hi2` uses it only for a
directory that is the stock Inkling layout (see [`hi2` on the stock
directory](#hi2-on-the-stock-directory)). The quantization itself (allocation, PTQ, QAD,
final KL) is not in this note: it records the reader, the checkpoint it reads, the first
measurements behind the plan (exactness, scale ranges, a model assembled from the reader,
weight-only allocation, what the device decode costs), how far `hi2 quantize` gets on
this directory, and what a calibrated run and the final-KL phase needed on this model:
[the layer program](#the-layer-program-hi2loopedpy), [final KL on
Inkling](#final-kl-on-inkling) and [what the runtime needs from the
output](#what-a-loadable-hi2-inkling-directory-needs).

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
* `message_query()` (also `load("model.language_model.message_input.query")`) composes the
  one parameter of the model that is built from two serving tensors: `[12, 4, 2304]` from the
  two cross-pass tensors `[24, 2304]` (rows `2l` and `2l + 1` of loop layer `l`: `q`, `m` in
  the first, `k`, `v` in the second), in the slot order `q, k, v, mlp` of the fork's loader and of
  transformers' own conversion (`Inkling2LoopMergeMessageQueries`).
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
absolute difference is 0.0, and `inkling_reference_check.py` repeats it on every expert of
every layer (all 24.46 G weights: bit-equal). Tiny checkpoints written by the test (with and
without an index file) cover the de-interleave, chunk and expert-subset invariance,
`fused_layer`, the name map, `message_query` and the `UnfusedExperts` integration with a
one-argument and with Inkling's two-argument activation. The decoded view of a stock-layout
directory, `source_map` / `open_source`, `check_unfused` through them, the auto-class
registration and a data-free `hi2.pipeline.run` from the stock layout (equal, tensor for
tensor, to the run from a regular directory holding the same weights) are tested in the same
file. `INKLING2_TOOLS` names the fork's `tools/rdna/inkling2` directory when the sibling
worktree is not at hand; without the fork the reference comparisons are skipped, and the
registration test and the sensitivity-probe test (the probe on the tiny looped model) need the
fork's transformers on `PYTHONPATH`.

`tests/test_looped.py` (the fork's transformers and `tools/rdna/inkling2` on the path, like the
registration test; skipped without them) and `tests/test_layerwise_extras.py` (toy layers, no
fork) cover the layer program, the preset and its guards, the pieces of the final-KL phase and a
CPU end-to-end run on the fork's tiny real model; the vision carry of the output is tested in
`tests/test_inkling_source.py`.

| Script (`experiments/`) | What it records |
| --- | --- |
| `inkling_inventory.py` | Tensor classes, elements, bytes and stock bits per weight from the safetensors headers (main shard, MTP, vision), checked against the design review. |
| `inkling_reference_check.py` | The reader against the fork's reference decoders on all 128 experts of every layer (the tests do four experts of two layers): bit patterns and largest absolute difference. |
| `inkling_teacher_check.py` | The whole stock model assembled from the reader on the CPU and run on a prompt: the top-8 against the fork's oracle, the activation extremes per decoder layer, float16 / bfloat16 against float32 (KL, top-1). |
| `inkling_unfuse_check.py` | `hi2.moe.unfuse_experts` on a tiny real `Inkling2LoopForConditionalGeneration`: does the unfused forward run, and equal the fused one? Exit 1 while `hi2.moe` called `act_fn(gate) * up` (the activation takes two arguments), 0 since `UnfusedExperts` runs it (largest logit difference 8.3e-7). |
| `inkling_hf_names_check.py` | The reader's names and shapes against the real `Inkling2LoopForConditionalGeneration` (built on the meta device from the fork's transformers): every text-model tensor, both ways. |
| `inkling_hf_load_check.py` | `from_pretrained` of the stock directory with the fork's transformers (the class registered by `hi2.inkling_source.register_auto_class`) against the reader: dense tensors and `message_input.query` exactly, the experts of every layer elementwise, and transformers' own dequantization op with its sources cast or as stored. |
| `inkling_scale_stats.py` | Histograms of the UE8M0 and E4M3 block scales and of the NVFP4 `amax`; the count of weights that overflow, underflow or fall in the subnormal range of float16 and bfloat16; the relative energy each cast changes. |
| `inkling_alloc_dryrun.py` | HI2's weight-only mode and its allocators on sampled real experts and attention linears: share of INT2/INT4/INT8, block mix, effective bits per weight, implied size. |
| `inkling_dequant_probe.py` | Exactness of the device decode against the CPU, and the cost (read, host-to-device, decode, whole layer, peak memory) of decoding one layer. |
| `inkling_quantize_dryrun.py` | `hi2 quantize` on the real directory as far as a CPU goes: `--dry-run` plan, a census of every tensor against the default `--exclude`, a data-free `run` on a sample of experts and attention linears, `_load_model` with `check_unfused` and a forward against the saved float32 logits, and `hi2` on a tiny real Inkling model (data-free `run`, the layer-wise calibration driver, the sensitivity probe). |
| `inkling_carry_check.py` | `hi2.inkling_source.carry_vision` on the real directory into a scratch directory, read back tensor by tensor (name, dtype, shape, bit pattern) against the stock vision tower and its index entries. |
| `inkling_footprint.py` | `hi2.finalkl.student_footprint` on the real configuration (meta model, stand-in HI2 tensors at a given code width and block size): the static bytes, saved layer inputs and loss chunk of the final-KL student per device. No weights, no device. |
| `inkling_looped_smoke.py` | The layer program on the real model, for a GPU lease: `teacher_head_inputs` (48 layer calls), optionally `calibrate_looped` over the first layers, the model's own forward on the host over the first `--compare-tokens` positions against the program's head inputs, and with `--student-steps N` a stand-in student (random HI2 tensors of 3.0 code bits for all 9,216 routed-expert Linears) trained for N final-KL steps, with the formula of `student_footprint`, the pipeline's fit line, the time of a step and the memory per device (`--student-full-grads`: a gradient for every trained value, the worst case; `--skip-compare`: the student alone). Time and peak memory per stage and device. `--devices cpu` runs it on the fork's tiny model (measured: ok). Measured on the real model on two V620: [GPU smoke](#gpu-smoke-on-two-v620-measured). |
| `inkling_build_calib.py`, `inkling_kl_eval.py`, `inkling_tiny_setup.py`, `inkling_stage4.sh` | The calibration and held-out set with its manifest, the KL against the stock model, the toy model of the CPU dry run and the driver of the stage-4 run: [Stage 4](#stage-4-calibration-set-pilot-three-encodes-and-their-kl-against-the-stock-model). |
| `inkling_alloc_report.py` | The allocation of a real run, from its `hi2_report.json`: share of INT2/INT4/INT8 by weights and of the blocks by rows, bits per weight and implied size by layer group, projection and layer, against the stock NVFP4 (4.5 bits) and MXFP8 (8.25 bits) experts. `tests/test_inkling_alloc_report.py`. |
| `inkling_target_table.py` | The encodes of a run side by side, from the `hi2_report.json` and `model*.safetensors` of `runs/bpwT` and the evaluation JSONs `eval/bpwT_{mixed,wiki,pydocs,fine}.json`: experts and total GiB with the stock directory's size, bits and shares, KL / top-1 / perplexity per set; the bits per weight of every layer by target; the load of the experts against their bits (tokens per expert, CV, share of the hottest eighth, bits of the coldest and hottest quarter, Spearman of tokens with bits); per group of layers of a fine evaluation its KL, bits, INT2 share, the allocator's predicted relative error, the measured weight error and the KL per squared error, with the Spearman rank correlation of each against the KL. `tests/test_inkling_target_table.py`: 13 tests, 29 mutants tried, all killed. |

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

**Exactness.** All 24,461,180,928 expert weights of the 24 layers (`w13` and `w2` of 128 experts,
20 NVFP4 and 4 MXFP8 layers) decode to the same float32 bits as the fork's `tiny_reference.py`
(`inkling_reference_check.py`, CPU, 534 s; the de-interleaved gate and up rows are its even
and odd rows): largest absolute difference 0.0.

**Names and shapes against the real model class** (`inkling_hf_names_check.py`:
`Inkling2LoopForConditionalGeneration` of the fork's transformers 5.24.0.dev0 built on the meta
device, experts unfused with `hi2.moe.unfuse_experts`). The model has 9,861 text-model
parameters and buffers and the reader produces exactly those names, 644 dense (`hf_names`),
`message_input.query` and 9,216 expert projections, with equal shapes: 0 absent from the model,
0 of a different shape, 0 not produced. The fused expert tensors of the class are
`gate_up_proj [128, 2304, 2304]` and `down_proj [128, 2304, 1152]`, the layout of
`fused_layer`. The 424 tensors of the vision tower are not read.

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
none there, see `moe.md`), at a cast error far below any 2-4 bit quantization error. The
activations are measured by the next paragraph.

**The whole model assembled from the reader, on the CPU** (`inkling_teacher_check.py`, the fork's
transformers). `Inkling2LoopForConditionalGeneration` is built on the meta device, allocated in
float32 / float16 / bfloat16 (105 / 52 GB of host memory) and filled from `InklingSource`:
693 tensors (dense by `load`, `message_input.query`, the experts decoded on the CPU), none of
the text model left empty, 53-78 s for the whole model (the experts 2-3 s per layer).

* Against the fork's CPU oracle (`ref_checkpoint.py`: its own dequantization cache, bfloat16,
  slot order `q, k, v, mlp`), the chat prompt "What is the capital of France?" (22 tokens) gives the
  same top-8 tokens and logits to two decimals when the reader's model is allocated in bfloat16
  (`<|message_model|>` 26.56, then 5.77, 5.62, 5.41, 4.83, 4.68, 4.28, 4.19). The reader, the
  name map and `message_query` give the model the oracle runs. (It does not pin the slot order:
  the oracle's three orders agree on the first token.)
* float16 and bfloat16 against float32, KL(float32 || x) over every position, all 200,058
  logits (CPU forward: float32 6 s, float16 5 s, bfloat16 4 s for 22 tokens; 23 s, 90 s and
  115 s for 512):

  | Input | dtype | largest decoder-layer activation | non-finite | KL mean / max | mean / max abs logit difference | top-1 agrees |
  | --- | --- | --- | --- | --- | --- | --- |
  | chat prompt, 22 tokens | float32 | 348.9 | 0 | | | |
  | | float16 | 349.0 | 0 | 6.6e-4 / 8.6e-3 | 0.039 / 2.8 | 22 / 22 |
  | | bfloat16 | 350.0 | 0 | 1.0e-2 / 6.8e-2 | 0.200 / 3.2 | 21 / 22 |
  | `texts/mixed.txt`: prose, code, French, Chinese, a table; 512 tokens | float32 | 396.3 | 0 | | | |
  | | float16 | 404.2 | 0 | 3.2e-3 / 7.8e-2 | 0.212 / 11.5 | 510 / 512 |
  | | bfloat16 | 384.0 | 0 | 4.2e-3 / 1.1e-1 | 0.249 / 11.4 | 504 / 512 |

* The activations stay far from the float16 limit of 65,504. The output of a decoder layer
  grows inside each of the four blocks (layers 0-5, 6-11, 12-17, 18-23) and peaks at the
  full-attention layers: 396 at layer 17 in the 512-token text (341 at layer 5, 243 at
  layer 11, 149 at layer 23) and 473 in `texts/long.txt`, 2,048 tokens of docs and code of this
  repository (float32 only, 133 s); a mlp output is at most 105 and an attention output at
  most 4.4. Nothing is non-finite in any layer or pass.

So float16 is usable for the teacher: the weights do not overflow, the activations of these
inputs have a headroom of 138-190x, and the half-precision noise (KL 6.6e-4 on the chat prompt,
3.2e-3 on the longer text, about the same as bfloat16 there) is the floor of any KL measured
against a half-precision teacher. That noise is probably router near-ties flipping experts
[inference: not isolated]; in KL it is not larger in float16 than in bfloat16 (the largest
logit difference, 11.5 against 11.4 on the 512 tokens, is the same), and float16 is the dtype
the fork serves the model in on gfx1030. Limits: three inputs of at most 2,048 tokens,
not the long contexts the loop layers see in use; the activation bound is a measurement on
them, not a proof.

**Transformers' own load of the directory** (`inkling_hf_load_check.py`; the fork's transformers
5.24.0.dev0). `AutoModelForCausalLM.from_pretrained(<stock directory>)`, the call of
`hi2.pipeline._load_model`, stops at once with `ValueError: Unrecognized configuration class
Inkling2LoopConfig for this kind of AutoModel: AutoModelForCausalLM`: no auto class knows the
model. With `AutoModelForCausalLM.register(Inkling2LoopConfig, Inkling2LoopForConditionalGeneration)`
it loads in 145 s (float16) and 148 s (bfloat16) on 8 CPU threads: the unknown `modelopt` tag is
skipped with a warning, the model's conversion mapping renames the tensors,
`Inkling2LoopMergeMessageQueries` composes `message_input.query`, and
`Inkling2LoopDequantizeExperts` (marked temporary in its source) dequantizes the NVFP4 and MXFP8
experts to the parameter dtype, 8 experts at a time, in float32 arithmetic with the formulas of the
reader, though not in the reader's order of operations (below). Against the reader:

* 644 of 644 dense tensors and `message_input.query` are identical in both dtypes (so the slot
  order `q, k, v, mlp` is transformers' own), and so are the MXFP8 experts of layers 20-23.
* The NVFP4 experts of layers 0-19 are not: in float16, 22.8 % of `gate_up` (17.0-27.1 % per layer)
  and 22.7 % of `down` differ from the reader's float32 decode rounded once, by one ulp (largest
  difference 4.9e-4); in bfloat16 22.3 % and 22.7 %, largest difference 3.9e-3.
* Cause, in two parts. (1) The loader casts every source tensor of a stack to the parameter dtype
  before the op, `amax` included (isolated by running transformers' op on four experts of layers 0
  and 20: with `amax` as stored the op returns the reader's values rounded to float16 or bfloat16
  bit for bit, 0 of 31.9 M elements differ; with it cast it returns the loaded model's values bit
  for bit). The cast puts a relative error on the global scale of each expert, up to 4.7e-4 (rms
  2.0e-4) in float16 and 3.9e-3 (rms 1.6e-3) in bfloat16 over the 5,120 `amax` values: a per-expert
  gain error as large as the rounding of the weights to that dtype (2.1e-4 and 1.7e-3 relative
  RMS), far below the 4e-2 of a 3-bit quantization. (2) The op multiplies in another order than
  the reader: `(code * block scale) * (amax / 2688)`, an exact product and one rounding, where the
  reader (a bit-exact port of the fork's reference decoders) rounds `block scale * (amax / 2688)`
  first and multiplies the code after, two roundings. A float32 load (`--dtype float32`, 137 s,
  105 GB resident) keeps `amax` exact (maximum relative error 0.0 over the 5,120 values) and still
  differs from the reader in 8.66 % of the NVFP4 `gate_up` elements (8.0-9.5 % per layer) and 9.18 %
  of `down` (8.4-9.8 %), by at most 5.96e-8 (2^-24, the unit in the last place of a weight between
  0.5 and 1; the script records absolute differences only, so one to two units in the last place
  for the smaller weights is [inference]); the MXFP8 layers (an E4M3 code times a power of two:
  exact) and the 644 dense tensors are
  identical. The half-precision comparison above did not show this because the final rounding to
  float16 or bfloat16 hides one float32 ulp.

So transformers can build the teacher in host memory without a staged checkpoint, within one unit
in the last place of the dtype (float32) or two (float16, bfloat16) of the reader's values; the
reader is the decode that matches the fork's reference decoders bit for bit, and the per-layer,
on-the-fly decode on the device is something `from_pretrained` does not offer.

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

**Decoding a layer on the device** (`inkling_dequant_probe.py`; Radeon PRO V620, gfx1030, torch
2.12.0+rocm7.14, one card of slot a2 at a time: PCI d4 alone, then d1; median of three calls after a
warm-up, the file in the host's page cache). The device decode is the host's, bit for bit
(experts 0, 17, 64, 127 of layers 0 and 20, gate/up and down: largest difference 0.0 on both cards),
and the two cards agree within 1-2 ms:

| One layer of 128 experts | NVFP4 (layer 0) | MXFP8 (layer 20) |
| --- | --- | --- |
| decode of bytes already on the device (chunks of 8-32 experts) | 21-26 G weights/s: 39-49 ms | 17 G weights/s: 59-61 ms |
| packed bytes, pinned window and host-to-device copy | 0.57 GB at 6.3-7.1 GB/s: 80-92 ms | 1.05 GB at 6.3-6.9 GB/s: 153-167 ms |
| whole layer from the file into a fused tensor, float32 / float16 | 111-133 / 107-137 ms | 186-206 / 182-211 ms |
| the same streamed (`iter_experts`, nothing kept) | 97-115 ms | 182-187 ms |
| device peak, fused float32 (tensor 3.80 GiB) / float16 (1.90 GiB) | 4.2-5.3 / 2.3-3.4 GiB | 4.2-5.5 / 2.3-3.6 GiB |
| device peak, streamed, chunks of 8 / 16 / 32 experts | 0.54 / 1.06 / 2.12 GiB | 0.58 / 1.15 / 2.30 GiB |

The "on the fly, about 10 ms per layer" figure of the design review is **not met** by this
reader: 39-61 ms for the decode alone and 100-210 ms from the file. About 10 ms is the roofline of
a fused kernel [inference]: 0.57 GB read and 4.1 GB written at the card's 512 GB/s is 9 ms, and
the reader (a lookup, two multiplies and a copy as separate torch operations) is at a fifth to a
quarter of it. With the packed bytes already on the card a teacher forward decodes its 24 layers
in 1.0-1.2 s (2.0-2.4 s if the three passes of the 12 loop layers decode again), and reading
every layer from the host once costs 2.7-3.5 s [inference: sums of the measured per-layer times,
no overlap of copy and compute]. The streamed peak grows with the chunk, 68-74 MiB per expert
in float32, so a chunk of 8-16 experts (0.5-1.2 GiB) is the working size; a fused float16 layer
is 1.9 GiB.

## `hi2` on the stock directory

`hi2` reads source weights by tensor name: a weight map, `safe_open`, and a resolver from model
names to checkpoint names. The stock directory has fused, packed tensors under serving names, so
every source read goes through two calls of `hi2.checkpoint`, `source_map(dir)` and
`open_source(dir, shard)`, which give any other directory its plain `weight_map` / `safe_open`
and this one a decoded view.

* `hi2.inkling_source.stock_view(dir)` returns a `StockView` when `config.json` has `model_type:
  inkling2_loop` and the raw index holds `model.llm.layers.0.mlp.experts.w13_weight`, else `None`.
  The view is the text model under its Hugging Face names, 9,861 tensors: the 644 dense ones
  (`hf_names`), `message_input.query` and the 9,216 expert projections
  `...experts.E.{gate,up,down}_proj.weight` in float32, gate first. They sit in one pseudo-shard
  named like the real main shard; the vision tower (424 tensors) is not in it. `get_slice` gives
  shape and dtype from the headers, `get_tensor` decodes on the CPU one expert at a time (the other
  half of a gate/up pair is kept until it is asked for, so every projection is decoded once).
  Nothing is staged on disk: the decoded experts are 94 GB in float32.
* Converted to `source_map` / `open_source`: `pipeline` (the tables, `_load_model`'s
  `check_unfused`, the sequential, QAD and final-KL resolvers), `checkpoint.write_checkpoints` and
  `verify_checkpoint`, `moe.check_unfused`, `layerwise.tables_from_stats`, `qad.estimate_seconds`,
  `cache.estimate_stats_bytes`, `sensitivity` (the probe's resolver and `fp32_weight_bytes`) and the
  parameter count of `service.inspect_model` (its header walk stays raw, so `tensor_count` and the
  byte sizes still describe the files). Raw on purpose: the byte accounting of `size.py`, the
  identity hash of the files (`cache.model_fingerprint`), the n-gram table, and everything on the
  output side (`stack`, `validate_checkpoint`, `load_dequantized`, `export_*`), which read what `hi2`
  wrote.
* `register_auto_class(dir)` runs before every `from_pretrained` of a source (`pipeline._load_model`,
  `sensitivity.load_probe_model`, `evalq.evaluate`): when `config.json` says `inkling2_loop` it
  registers the fork's `Inkling2LoopForConditionalGeneration` for `AutoModelForCausalLM`, for any
  other model it does nothing. The fork's transformers (`transformers.models.inkling2_loop`) must be
  on `PYTHONPATH`; without it the run stops with a `SystemExit` that says so.
* `check_unfused` compares the unfused views with the checkpoint tensors through the view. The
  model's own loader is a second implementation of the formats (see above), so it allows two units
  in the last place for a stock directory, in every dtype; any other model must match exactly.
* `write_checkpoints` refuses a stock directory in which an expert tensor is left unquantized (an
  `--include` or `--limit-layers` that does not cover them): the view hands the experts out as
  float32 and a passthrough would write them at 4 bytes per weight.
* `_finish_output` links the source's other files (the tokenizer, ...) into the output, but not
  a source `*.safetensors` its index does not list (the stock `mtp.safetensors`:
  `validate_checkpoint` refuses an unindexed shard, and every run on this directory ended in
  `ValueError: shard/index coverage mismatch; unindexed=['mtp.safetensors']`, found by the dry
  run below) and, for the stock Inkling directory only, not `hf_quant_config.json` (it describes
  the NVFP4 / MXFP8 experts the output no longer has). It carries the vision tower:
  `carry_vision` copies the 422 `model.visual.*` tensors bit for bit into
  `model-vision.safetensors` and lists them in the output index
  ([what the runtime needs](#what-a-loadable-hi2-inkling-directory-needs)).

**What the CPU dry run found** (`inkling_quantize_dryrun.py`, the build box, no GPU; results in
[`experiments/results/inkling2_mini_2026-11-23/`](../experiments/results/inkling2_mini_2026-11-23/)):

* `plan` (also `hi2 quantize SRC DST --dry-run --json --devices cpu --include '\.mlp\.experts\.'
  --bits 2,4,8 --target-bpw 3.5`, exit 0): the checkpoint facts and the phases come out (data-free
  `data_free_tables, allocation, encode, validation`; calibrated with QAD `calibration, allocation,
  qad, final_kl, encode, validation`), 24 layers, 26,386,415,616 parameters (the view's logical
  count; the raw headers count about 17.9 G elements of packed tensors), 105.5 GB of float32
  weights, and the tokenizer of the directory reads a 60 KB text of this repository's docs and code
  (20,457 tokens). The repo's QAD estimate for this model, 7,337 s on a GPU, is its generic
  formula [inference: not calibrated on Inkling].
* `census` (9,861 names, 26,068,159,488 elements): experts 24,461,180,928 (93.8 %). The default
  `--exclude` leaves 25,135,497,216 candidates: besides the experts, 644,087,808 elements in the
  six attention linears per layer (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `o_proj.gate_proj`,
  `relative_position_bias.linear`) and **30,228,480 elements that should stay dense**: the loop
  controllers (`*_message_gate.{down,forget_gate,message_gate}_proj`, 28.3 M), `self_attn.sink_proj`
  (1.3 M) and `k_proj` / `v_proj` `.sconv.weight_proj` (0.6 M). `--include '\.mlp\.experts\.'`
  selects exactly the 24,461,180,928 expert weights; attention linears are
  `\.self_attn\.(q|k|v|o)_proj\.linear\.weight$|\.self_attn\.o_proj\.gate_proj\.weight$|\.self_attn\.relative_position_bias\.linear\.weight$`.
* `sample`: a real data-free `run` (tables, global allocation at 3.5 bpw, encoding,
  `validate_checkpoint`, then `verify_checkpoint`) on experts 0 and 64 of layers 0 (NVFP4) and 20
  (MXFP8), 12 projections, and the `q`, `k`, `v`, `o` linears of both layers, with every dense
  tensor passed through: 41.8 s, 4.0 GiB peak resident, 20 tensors quantized at 3.499 bits per
  weight (rows INT2 38.8 %, INT4 61.2 %, INT8 0), the experts at 4.14 and the attention linears at
  2.96 (`q`, `o` INT2 at 2.52; `k`, `v` INT4/B64 at 4.27): a sample of 20 tensors, so the split is
  not the model's. The output has `config.json` rewritten, the index, one shard and the links of
  the tokenizer and the other assets (at that time `hf_quant_config.json`, which lists the stock
  NVFP4 layout, was linked too: it is not any more).
* `model`: `_load_model` on the real directory loads in 142 s in float16 (70 GiB peak resident) and
  134 s in float32 (120.5 GiB), unfuses the experts (`UnfusedExperts`, gated activation) and passes
  `check_unfused` in both; the model has 26,386,412,544 parameters, 3,072 fewer than the view (the
  24 x 128 router biases are buffers). The chat prompt of the teacher check (22 tokens) gives the
  same top-8 as the reader's model (`<|message_model|>` 26.58 in float32, then 5.948, 5.619, 5.517).
  Against the float32 logits of stage 1 (the model the reader filled): float32 4.1e-5 largest logit
  difference and KL 1.1e-7 at its worst position, top-1 equal at all 22; float16 1.00 and 1.3e-4
  (the float16 model of stage 1 had 2.8 and 8.6e-3 at its worst position: one 22-token prompt
  is a noisy measure of half-precision noise, probably router near-ties [inference]).
* `probe`: `hi2` on the fork's tiny real `Inkling2LoopForConditionalGeneration` (six layers, loop
  layers 2-3 with two passes, the stock layout: `make_tiny_checkpoint.py`). A data-free `run`
  works end to end through the view (144 tensors at 3.5 bits per weight, validated). The layer-wise
  driver does not: `capture_inputs` finds the layers (`model.language_model.layers`) and records
  `hidden_states` and the keyword arguments (`attention_mask`, `cache`, `conv_mask`, `position_ids`,
  `relative_distance`, `use_cache`), and the replay `layer(hidden, **kwargs)` fails with
  `TypeError: Inkling2LoopSlidingDecoderLayer.forward() missing 1 required positional argument:
  'block_states'`; `calibrate_layerwise` and a calibrated `run` (`calib_ids`, GPTQ) fail the same
  way. The sensitivity probe loads and runs (full forwards), but its default prefix replay (the
  outputs of the layers before a unit, kept from the reference forward) was silently wrong on the
  looped layers: it kept the output of the last call of each layer, so a unit in the second loop
  layer got the first pass fed with the second pass's output. Largest relative difference to the
  probe without replay 9.7, no error (`quantize_dryrun_probe_before_guard.json`). A layer that runs
  more than once per forward now makes the probe recompute the prefix, 0.0 difference
  (`quantize_dryrun_probe.json`; `tests/test_inkling_source.py`, which fails with the guard off).
  The layer-wise failure is fixed on this branch: [the layer program](#the-layer-program-hi2loopedpy).

## The layer program (`hi2/looped.py`)

The generic loop of `hi2.layerwise` replays `layer(hidden, **kwargs) -> hidden`, one layer after the
other. Inkling-2's text model does not run like that: layers 0-5, then the loop layers 6-17 three
times (the second and third pass restart from the states before the loop and receive `messages =
message_input(previous block states, previous hidden states, loop layer)`), then layers 18-23,
`final_input` and the norm, and every layer takes `(hidden, block_states[, messages], ...)` with
`block_states` `[B, T, blocks, H]` growing at the block-boundary layers: the 24 layers are **48
layer calls**, and a loop layer's weights see three different inputs. For `model_type ==
"inkling2_loop"` (`looped.LOOPED_MODEL_TYPES`) `calibrate_layerwise` hands over to
`looped.calibrate_looped`; every other model keeps the generic loop (the tests of the existing
subset pass unchanged, see the gates).

* `schedule(config)` lists the layer calls with their `messages` flag (the tiny model: layers `[0, 1,
  2, 3, 2, 3, 4, 5]`; Inkling-2-Mini: 48). `record_calls` runs the model's own forward once with
  every layer replaced by a recorder, so the keyword arguments of each call (attention masks,
  relative distances, conv masks, the cache offset of a pass) are read from the model, not rebuilt.
  `run_program` is the control flow over per-sequence states `(hidden, block_states)` that stay in
  host memory, one layer at a time on the devices.
* **Statistics are pooled per tensor across the calls.** A loop layer shares its weights across the
  passes, so every call is a sample of the same Linear: the column statistics and the Hessian sums
  are accumulated on the host (`_Acc`) and the tables are built once, when the layer's last call is
  done (the same `expert_min_tokens` blend and `_tabulate_layer` as the generic loop). A Linear whose
  gate/up sharing leader differs between calls stops the run with a message. The 12 loop layers hold
  their accumulators from the first to the third pass; with GPTQ rounding the Hessians are about
  3.4 GB per layer in host memory (a full fp32 `[K, K]` per shared input), about 41 GB for the 12
  [inference: arithmetic, not run on the real model]. The statistics cache on disk keeps the fp32
  triangle: `cache.estimate_stats_bytes` on the real directory with the experts preset returns
  42,148,823,040 bytes (39.3 GiB, measured; the build box had 138 GB free at 97 % use).
* `teacher_head_inputs` is the same program without hooks: the unquantized model's input to the LM
  head, `[n, T, H]` in float32, for the final-KL phase.
* Refused up front (`pipeline._check_looped`): `--sequential` (it feeds each layer call the quantized
  prefix) and a `--qad-layers` other than none / all (layer-wise QAD training needs a layer's own
  input). The layer-wise phase then runs with `layers="none"`: PTQ per layer, nothing trained; the
  final-KL phase does the training. `HI2_LOOP_SEQS` (default 4) is the number of sequences in one
  layer call.
* `--include` defaults to `\.mlp\.experts\.` for this model family (`looped.DEFAULT_INCLUDE`, set in
  `service.resolve_config`; an explicit `--include` wins; no other model gets a default): the
  attention linears, the loop controllers, the sinks, the short convolutions, the router and the
  vision tower stay as they are (dense bf16), the experts are 93.8 % of the text weights.

**Gates (measured, CPU, the fork's tiny real `Inkling2LoopForConditionalGeneration`: 6 layers, hidden
256, 8 experts, top-2, loop layers 2-3 with two passes; 4 sequences of 12 tokens;
`tests/test_looped.py`).** A calibrated `run` (`calib_ids`, GPTQ) completes; it raised `TypeError:
... missing 1 required positional argument: 'block_states'` before. The statistics of 141 of the 144
expert tensors (three saw no token) equal those of forward hooks on the unfused per-expert linears
of the whole-model forward: column statistics 0.0 and Hessians 2e-7 (relative to the largest entry)
with one sequence per layer call, 7e-6 both with eight (float32 summation order). The head inputs
equal the model's own to 1e-5 (one sequence) and 5e-5 (eight) absolute. A second tiny model has the
loop structure of the real one (12 layers in blocks of three, layers 3-8 run three times, 24 layer
calls: the third pass takes its messages from the second): the schedule equals the order of the
model's own calls, the head inputs equal its forward (1e-4) and the statistics of all 288 expert
tensors equal the hooks' (1e-6); a mutant that is wrong only in the third pass fails there and
passes the two-pass tests.

Existing models are untouched: the seven-file subset (`test_inkling_source`, `test_moe`,
`test_pinned`, `test_allocate_fast`, `test_lowbit`, `test_checkpoint_safety`, `test_stack`) gives 165
passed, 3 skipped (164 before this stage plus the vision-carry test); the whole suite on the build
box, 1011 passed, 60 skipped, 3 failed (the three that also fail on the pristine stage-1 tree:
`test_cache::test_probe_prefix_replay_is_bit_identical`,
`test_cli_contracts::test_version_works_without_compute_dependencies`,
`test_cli_contracts::test_cancelled_concurrent_tuning_runs_only_cleanup_owned_scratch`). Twenty-seven
hand-made mutants (the program: statistics reset per call, no messages, outputs appended in place,
`limit_layers` ignored, the hidden cast always, the batch ignored, the extras mis-indexed, the loop
offset off by one, a stale third pass; the final KL: soft cap dropped, `final_input` and message
placement, expert checkpointing off, the Adam state forgotten, the saved inputs, the unpack width, a
number taken for the soft-cap op, the head-input key; the preset and the guards; the vision carry and
the stale file) each fail a test.

## Final KL on Inkling

`pipeline._qad_encode` runs the layer-wise phase (PTQ per layer), then `hi2.finalkl.final_kl`: the
quantized student's logits against the teacher's, a KL over the sequence, training the per-block
ladder values and the norms. What changed for this model:

* **The loss had no soft cap.** The model's logits are `logit_softcap(lm_head(h).float())`
  (`tanh(x / 30) * 30`); `final_kl` took the raw head's. `_logits_fn(model)` applies the model's cap
  (a model without one keeps its head); a test compares the loss with a manual capped KL and shows
  the uncapped loss is another number.
* **Placement.** `sensitivity.shard_layers` left `message_input` and `final_input` on the first
  device while their inputs live on the devices of the layers before them. `final_input` now goes
  with the last layers, the head and the final norm, `message_input` with the embeddings, both with
  the input-moving pre-hook of a layer (a test records the placement and the device the hooks move
  to).
* **The teacher is not placed.** Its head inputs come from the layer program, are kept in
  `workdir/teacher_head_inputs.pt` (keyed by source, dtype and the token ids) and shared by every
  allocation of the run.
* **Packed code words.** `LadderLinear` kept one `uint8` per weight; it keeps the `int32` words of
  the checkpoint (`nbits` per weight) and unpacks per forward: the experts' codes at 3.5 bits per
  weight are about 10 GiB instead of 24 (the decode, the export and the codes are unchanged:
  `tests/test_lowbit.py::test_ladder_linear_is_the_exact_decode`).
* **Per-expert activation checkpointing** (`UnfusedExperts.checkpoint_experts`, set by `final_kl` for
  a looped model only): the fp32 weight of an expert Linear is rebuilt in the backward, one expert at
  a time, instead of kept (128 experts x 3 projections x 10.6 MB = 4 GiB per layer). Outputs and
  gradients are bit-identical to the unchecked ones and fewer bytes are saved (a test).
* **The fit check.** `pipeline._final_kl_fits` ("codes at one byte per weight", too optimistic for
  this model) is replaced, for a looped model, by `finalkl.student_footprint`: per device the static
  bytes (packed codes, the trained ladder values at 16 bytes each with their gradient and Adam
  moments, buffers, dense parameters), the layer inputs one sequence's backward keeps
  (`looped.saved_input_bytes`), a transient allowance (3 GiB) and the loss chunk on the last device;
  the student must fit within 80 % of the smallest device (92 % until the GPU smoke: the allocator's
  reserve and the context take about 5 GiB more than the formula, see below). On the tiny model the
  static part equals the parameters and buffers of the converted student plus 12 bytes per trained
  value (the ladder linears byte for byte, the whole within the vision tower's norms, 2e-3) and the
  saved inputs equal the sizes read from the layers' own inputs (`tests/test_looped.py`, measured).
  On the real model the static part is exact to 0.7 % and the peak within 3 % of the formula
  (measured, [GPU smoke](#gpu-smoke-on-two-v620-measured)).
* On the real configuration (`experiments/inkling_footprint.py`; meta model, stand-in tensors with
  3.0 code bits per weight in blocks of 32, about 3.5 bits per weight, float32, two devices of
  29.98 GiB): static 20.0 and 18.8 GiB (38.8 GiB: the packed codes are only 8.5 GiB of it, the
  trained ladder values with their gradients and Adam moments about 22.8 GiB, the dense parameters
  7.2 GiB [arithmetic on the numbers above]), saved inputs 1.9 and 2.8 GiB at 2048 tokens (0.5 and
  0.7 at 512), transient 3.0, loss chunk 0.57: about **25.2 GiB on the busiest device, over the
  24.0 GiB of the 80 % limit**, so a float32 student is skipped (it was inside the 27.6 GiB of the
  92 % limit until the GPU smoke showed the card holding about 5 GiB more than the formula: about
  30 GiB for this one [inference: the formula plus the measured 5 GiB]). In float16, the smoke's
  dtype (dense parameters 3.6 GiB), the same student is 22.3 and 22.9 GiB and runs. The student is
  therefore not the 13.6 GiB of packed weights and
  dense parameters: its trained state is about twice that. Blocks of 16 everywhere would need
  36.6 GiB and be skipped (the ladder values are two per block at 16 bytes each, so the size
  depends on the block mix the allocator chose; the check reads the actual tensors); blocks of 64
  everywhere would need 12.2 and 11.6 GiB of static in bfloat16 [inference: the formula, not a run].

**Gate (measured, CPU).** `hi2 quantize` of the tiny real Inkling model with calibration and `--qad`
(`final_kl=2`) runs the layer program, PTQ per layer, the teacher's head inputs and two final-KL
steps and writes a checkpoint that `validate_checkpoint` accepts
(`tests/test_looped.py::test_a_cpu_final_kl_of_two_steps_completes_on_the_looped_model`).

### GPU smoke on two V620 (measured)

`experiments/inkling_looped_smoke.py` on the real model, two Radeon Pro V620 of 29.98 GiB (HIP 0
and 1 together = PCI d1 and d4), under a guard that held the boot lock, read no quiet marker, and
counted the kernel's page faults on both cards: five runs of 4.8 to 9.7 minutes (34.5 minutes in
all), 0 page faults before and after each, at most 62 C. Two random sequences of 2048 tokens,
float16 unless stated.

| stage | float16 | float32 (2 x 128 tokens) |
| --- | --- | --- |
| load: time, host RSS | 130.0-134.5 s, 70.0 GiB | 129.2 s, 120.5 GiB |
| layer program, 48 calls: time, peak per card, host RSS after | 54.1-55.3 s, 3.59 GiB, 101.7-103.5 GiB | 67.4 s, 3.99 GiB, 184.2 GiB |
| the model's own forward on the host, 2 x 128 tokens | 38.9-39.3 s | 18.0 s |
| the program again on the same 128 tokens | 36.6-37.1 s, 2.01 GiB | 63.0 s, 3.99 GiB |
| `calibrate_looped`, the first 2 layers (4096 tokens) | 86.2 s (tables 50 s, then 11 s), 5.69 / 5.66 GiB | |
| the 9,216 stand-in tensors | 44.1 s | |

* **The program is the model's forward.** Float32, the first 128 positions of both sequences: the
  largest difference of the head inputs is 1.6e-4 of a largest value of 94.5 (relative 1.7e-6,
  median position 3.4e-7, 99th percentile 1.3e-6), and the program run twice is identical (0.0).
  Float16: 5.6 of 96.4 (relative 5.8e-2, median position 4.5e-3, 99th percentile 3.4e-2), and the
  program on 2048 tokens differs from the program on its first 128 by 3.0e-2. The float16 difference
  is round-off through 48 layer calls, not the program [inference from the float32 zero; the
  stage-2 float16 against float32 numbers, KL 3.2e-3 and a largest logit difference of 11.5 on 512
  tokens, are the same size]. A float16 student therefore has a noise floor of that order in its
  own KL, and the teacher's head inputs are better computed in float32 [inference] (3.99 GiB per
  card; the host then holds the float32 model: 120.5 GiB, 184.2 GiB after the program).
* **The whole float16 model does not fit the two cards sharded** (run 1: `OutOfMemoryError` at
  23.9 GiB allocated and 5.6 GiB reserved but unallocated on the first card, in `shard_layers`;
  the allocator rounds the 5 MiB expert tensors up inside its segments [inference]), so the
  comparison runs the model's forward on the host. The final-KL phase never places the unquantized
  model.
* **The student** (random HI2 tensors of 3.0 code bits in blocks of 32 for all 9,216 expert Linears:
  24,461,180,928 weights, 1,529.0 M trained values; 3.5 bits per weight with the ladders; the values
  are noise, the bytes and the speed are those of a real student). The fit line printed by the
  pipeline: "the student needs about 22.9 GiB on the busiest of 2 device(s) of 30.0 GiB (static
  ['17.9', '17.3'], saved layer inputs ['1.4', '2.0'], transient 3.0 GiB per device)".

  | per card (cuda:0 / cuda:1), GiB | formula | measured, stand-in's routing | measured, every value with a gradient |
  | --- | --- | --- | --- |
  | static (allocated after a step) | 17.93 / 17.34 | 15.44 / 12.89 | 18.01 / 17.46 |
  | peak allocated | 22.32 / 22.88 | 18.38 / 16.82 | 22.93 / 22.16 |
  | peak reserved by the allocator | | 23.53 / 20.07 | 27.13 / 26.49 |
  | in use on the card (driver) | | 24.1 / 20.7 | 27.8 / 27.1 |

  The formula's static part is exact (0.4 % and 0.7 % over) once every trained value has a
  gradient; the random stand-in's routing reached only 58.2 % of them (890,454,528 of 1,529,005,568
  at the first step), and an unrouted expert has no gradient and no Adam state, so that run is
  not the worst case (`--student-full-grads` gives every value a zero gradient to reach it: the
  same losses to the last digit). The peak allocated is within +2.7 % and -3.1 % of the formula
  (it is larger on cuda:0 than the formula's 22.32, by 0.6 GiB), but the allocator reserves 4.2 and
  4.3 GiB more (1.7 and 1.9 GiB of it split-off free bytes inside segments in use) and the card
  held 27.8 GiB of 29.98: the 92 % limit of the formula's number admitted students that would not
  run, hence 80 %. A float32 dense student (25.2 GiB by the formula) would not fit [inference].
* **A step** (one window of 2048 tokens, batch 1, the layers under checkpointing): 86.2 and 86.6 s
  between two optimizer steps; the first step is done 152 to 154 s after the phase began (the
  placement of the student and the first held-out pass take about 66 s of it) and the last
  held-out pass and the end about 27 s: 265-268 s for two steps. The stand-in's held-out KL is
  10.9556 before and 10.9547 after, finite, as it should be for noise. The host held 126.7-128.8 GiB
  during the phase.
* **Host memory.** The resident size grows by 31.7-33.5 GiB (float16) and 63.7 GiB (float32) during
  the layer program, about 0.6 of the model's size each time, with nothing in the program that
  keeps weights [unexplained: the freed host copies of the layers' weights not returned to the
  system is the first suspect; `calibrate_looped` calls `malloc_trim` per layer,
  `teacher_head_inputs` does not].

## What a loadable HI2 Inkling directory needs

Read from the fork's runtime (`vllm-inkling2-merge` at `ee27e18374`, `vllm/models/inkling2_loop/amd/`,
and the parked `vllm-inkling2-qlinear`), not run; nothing there was edited.

What `hi2 quantize` writes for this model now: the text model under Hugging Face names in
`model-00001-of-00001.safetensors` (HI2 containers for the experts, dense tensors as stored),
`model-vision.safetensors` (the 422 vision tensors, bit for bit, stock names), one index over both,
`config.json` with HI2's `quantization_config` in place of the stock `modelopt` one, and the links of
the tokenizer files. Not written: `hf_quant_config.json` and `mtp.safetensors`.

* **The runtime has no HI2 expert path for Inkling** (none in the eight `vllm-inkling2-*`
  worktrees; the only `rdna_hint_flex` mention under `inkling2_loop` is a comment in `dspark.py`).
  `loader.read_expert_formats` needs `hf_quant_config.json` with `parameter_groups[].compute_dtype`
  in `{nvfp4, mxfp8}` for every `layers.N.mlp.experts.w13_weight` and raises `lists no nvfp4/mxfp8
  experts for layers [...]` otherwise; the experts run on `moe_nvfp4_gemm_rdna2` /
  `moe_mxfp8_gemm_rdna2`. The generic `rdna_hint_flex` method (`quantization/rdna_hint_flex.py`,
  `rdna_hint_flex_moe.py`) serves other models: `quantization_config = {"quant_method":
  "rdna_hint_flex", "version": "3.2"}` and `<name>.hi2_<field>` tensors. An HI2 MoE for Inkling needs
  the soft-capped two-argument gated activation in float32, `route_scale` 8.0 and, as for every HI2
  MoE, gate-first rows with one transform for gate and up (the stock's interleaved `w13` is not
  HI2's).
* **Names.** `Inkling2LoopTextForCausalLM.load_weights` maps serving names (`model.llm.*`) and raises
  `KeyError: unmapped Inkling-2 checkpoint tensor` (or `no such parameter`) for any other, every
  `.hi2_*` field included. The output has Hugging Face names (`model.language_model.*`). Either the
  writer emits serving names (the inverse of `hf_names()` plus a convention for the expert
  containers) or the runtime maps the Hugging Face ones. **Not decided.**
* **Quantization config.** The stock `config.json` says `modelopt` / `NVFP4` for the whole model
  (layers 20-23 are MXFP8); the runtime reads the formats from `hf_quant_config.json`, which
  describes experts the output no longer has: it is not linked any more, and the HI2 formats need a
  declaration the runtime can read (the first item).
* **Vision tower.** `model.visual.backbone.*` (422 tensors, 318,256,128 elements, bf16) is read by
  `Inkling2MultiModalMixin.load_weights`; the output carries it. Measured on the real directory
  (`inkling_carry_check.py`): 422 tensors, 636,512,256 bytes, 0 mismatches, 0.27 s, every name
  indexed.
* **MTP.** The DSpark draft reads `mtp.safetensors` through `allow_patterns_overrides =
  ["mtp.safetensors"]`, which skips the index filter (`filter_duplicate_safetensors_files`) that
  keeps the main loader to the files its index lists. So the file must sit next to the output and
  **stay out of the index** (listed, its tensors would reach the main model's `load_weights`);
  `validate_checkpoint` refuses an unlisted weight file, so `hi2` does not link it: copy or link it
  when deploying [inference: from the loaders, not run].
* **Attention** stays bf16 (the fork's dense quantized-linear packers, `linear_pack.py`, are parked
  in `vllm-inkling2-qlinear`); quantizing it with HI2 needs a runtime path that does not exist.

## Stage 4: calibration set, pilot, three encodes and their KL against the stock model

Prepared and dry-run on the CPU of the build box on 2026-11-24. **Nothing in this section has
run on a GPU** (no lease was taken): the figures are measured where they say so, and
`[inference]` marks the rest.

### The calibration set

`experiments/inkling_build_calib.py` (`stage`, `build`, `verify`) renders the cached
datasets with the chat template and the tokenizer of the stock directory (the specials parse
inside the text: 2.04 % of the tokens), mixes them with `hi2.calib.build_mix` (a document is
never split; the split is by the sha256 of the exact text, 20 % held out), cuts windows of 2048
tokens and writes `manifest.json` with the sources, licences, seeds, counts and the sha256 of
every file. Measured:

| Source | Licence | Staged | Weight | Calibration tokens (documents) | Held-out tokens (documents) |
| --- | --- | --- | --- | --- | --- |
| ToolACE (`Team-ACE/ToolACE`; 552 of 600 records get a native `tool_declare` turn) | apache-2.0 | 600 | 0.30 | 162,751 (191) | 59,275 (66) |
| OpenAssistant-Guanaco (`timdettmers/openassistant-guanaco`, train) | apache-2.0 | 800 | 0.20 | 108,513 (271) | 39,429 (103) |
| WizardLM evol-instruct 70k | mit | 800 | 0.20 | 108,210 (249) | 39,713 (95) |
| GSM8K (`openai/gsm8k` main, train; the solution is the thinking turn) | mit | 500 | 0.05 | 27,113 (143) | 9,840 (55) |
| project docs (vLLM documentation and notes; 921 of 2042 near-duplicate blocks dropped) | apache-2.0 + project | 1,121 | 0.25 | 135,382 (174) | 49,621 (70) |

**542,995 calibration tokens = 265 windows of 2048** (the encodes read the first 256, 524,288
tokens, `--calib-samples 256`; 128 windows, 262,144 tokens, is the floor of the contract);
**held-out `mixed`: 198,266 tokens = 96 windows** from 389 documents that are in no calibration
window, with the same source shares (0.2996 / 0.1993 / 0.2007 / 0.0497 / 0.2508); two
out-of-distribution held-out sets with no overlap at all: `wiki` (17 windows, 35,320 tokens) and
`pydocs` (17 windows, 36,606 tokens). `calib.txt` sha256 `acd2f960…`, `heldout_mixed.txt`
`a543c14e…`, windows (`*_ids.npy`, int32 `[n, 2048]`) `4077eae6…` (calibration), `07360700…`
(mixed), `2e1aa3e1…` (wiki), `e5695509…` (pydocs); the full list is in the manifest. 14 MB.

Hard gates, recorded in the manifest and re-checked by `verify` (every file's sha256, the
windows, the tokenizer and template files): calibration at least 262,144 tokens; at least 32
held-out windows; every source in both splits; no staged document in both splits; no held-out
window shares more than 0.5 of its 32-token shingles with the calibration (measured: mean
0.060, p99 0.163, max 0.172 for `mixed`; the residue is the structure the chat records share:
the tool-use instruction, the template tokens); specials between 0.2 and 20 % of the tokens;
every id inside the vocabulary; `hi2.pipeline.load_calibration_ids` returns the stored
windows (the first 256). Soft: 524,288 tokens, 64 held-out windows (both met).
`verify --production` also refuses a set built with scaled-down limits (`--dry-run-scale`, for
the toy model).

Limits: ToolACE assistant calls stay the dataset's text (`[Name(a=1)]`), not native
`tool_call` turns (the tool lists are native); the only thinking turns are GSM8K's; nothing is
generated by the model itself; the calibration is mostly English chat, tool use, instructions,
arithmetic and technical text, with some chat in other languages (1.7 % of the letters are Cyrillic, 0.34 % CJK and Kana, 0.23 % accented Latin).

### KL against the stock model

`experiments/inkling_kl_eval.py --src STOCK --dst HI2_OUTPUT --ids heldout_mixed_ids.npy`
scores `KL(stock || HI2)` per window. The teacher is the stock model with its experts decoded
by `InklingSource` (exact), the student the same model with every expert Linear overwritten by
`hi2.format.dequantize` of the HI2 tensor, in place and one layer at a time; both run through
`looped.teacher_head_inputs` (the layer program, one layer per device call) and the same head
and soft cap. It reports KL, top-1 agreement, expected acceptance, the perplexity ratio and the
standard error over windows, and (`--groups auto`: layers 0-5, 6-17 and 18-19, 20-23 on the
real model: before, inside and after the loop, and the MXFP8 layers) the KL of a student that
is quantized in one group only, plus the relative weight error of every layer. Gates: the
output is an HI2 directory holding all 9,216 expert tensors and no other quantized tensor, the
windows are the manifest's, the teacher and the student are finite, the metric is exactly 0 on
identical inputs, the groups restore the stock weights (the probe window gives the same head
inputs after them: exactly on the CPU, 5e-3 of the maximum on a GPU), the student differs from
the teacher, no KL is negative, `--max-kl` holds. `tests/test_inkling_kl_eval.py` (CPU, the
tiny model) checks the metric against the textbook formula and a hand-worked case,
the installed weights against `load_dequantized`, the attribution of damage to groups, the
teacher cache and its invalidation, and each gate on an input made to fail it.

### The driver

`experiments/inkling_stage4.sh` runs the steps; `DRY=1` prints the commands, the GPU wrapper
comes from `$STAGE4_ENV` (`experiments/inkling_stage4.env.example`; the driver never takes a GPU
or a lease), one log and one state file
per step, one GPU step at a time, a disk gate in front of every write (`MIN_FREE_GB`, 60) and
the calibration gate (`verify --production`, `SAMPLES` within the windows the build compared
with the pipeline's reader, at least 262,144 tokens for an encode) in front of every GPU step.
Every command it builds is parsed by the real parser of the tool in
`tests/test_inkling_stage4_script.py`.

| Step | Where | What | Disk written |
| --- | --- | --- | --- |
| `calib` | CPU | stage, build, verify the set under `stage4/calib` | 14 MB |
| `plan T` | CPU | `hi2 quantize --dry-run --json` at T bpw: 6 s each on the real directory [measured]; 542,995 tokens available, 524,288 needed | |
| `pilot` | GPU | the whole chain at 3.5 bpw with 4 windows (8,192 tokens) into `runs/pilot`, own cache, then a KL on 4 windows of 512 tokens | cache ~39 GiB + output ~15 GiB [the 2026-11-23 measurement], removed by `clean-pilot` |
| `quantize T` | GPU | calibration (the first call fills `stage4/cache`, the next ones reuse it), allocation, encode, validation | the first: cache + output; then ~15 GiB each |
| `eval T [SET]` | GPU | the KL above in float32 (`EVAL_DTYPE`; the encode keeps float16) on 32 windows (`mixed`, with the group breakdown on 8) or the OOD set; `fine`: the same windows and the same teacher pass as `mixed`, broken down by `FINE_GROUPS` (default eight groups of three layers, `0-2,3-5,...,21-23`) into `eval/bpwT_fine.json`, about 8 x 110 s of group passes [inference from the 3.5 run: four groups took 110-160 s each] | ~1 GiB with the teacher cache |
| `eval T routing` | GPU | the `mixed` windows and teacher pass with `--routing-log` and no breakdown (`eval/bpwT_routing.json`, `bpwT_routing_routing.npz`): the score plus where the router sends the tokens, see [stage 5](#stage-5-a-final-kl-pilot-on-the-35-bpw-encode-and-the-load-flattened-allocation) | ~1 GiB |
| `teacher-heads` | GPU | the stock model's float32 head inputs of `FKL_TRAIN` (48) windows of the calibration text, for the final-KL phase (`teacher/fkl_train_w48_float32.pt`, 0.84 GiB) | 0.84 GiB |
| `finalkl T` | GPU | `FKL_STEPS` (40) final logit-KL steps of `FKL_BATCH` (1) windows at `FKL_LR` (0.07) from the finished output at T, held-out windows `FKL_VAL_OFFSET` (24) onwards of the mixed set -> `runs/bpwT-fkl40` | output ~14 GiB + the phase's resume copy ~14 GiB |

A target may carry a tag after its bits (`3.5-p0.5`): the bits are the part before the dash, the output and the work
directory are `bpw3.5-p0.5`, and `EXTRA_QUANTIZE` adds arguments to the encode command (the load-power sweep passes
`--expert-load-power 0.5`).

The encode command (`hi2 quantize SRC DST --calib calib.txt --calib-samples 256 --seq-len 2048
--include '\.mlp\.experts\.' --bits 2,4,8 --candidates lean --target-bpw T --no-qad
--sensitivity none --devices cuda:0,cuda:1 --compute-dtype float16 --workdir work/bpwT
--cache-dir cache`) keeps the defaults for the rest: GPTQ rounding, two refinement sweeps,
rotation on, smoothing 0.5, effort normal, `--expert-min-tokens 2.0` (an expert Linear with
fewer calibration tokens than twice its input width, 4,608 for gate and up, 2,304 for down,
falls back; 524,288 tokens at 8 of 128 experts per token give 32,768 per expert per layer on
average [arithmetic, the routing is not measured]). `--sensitivity none` because the probe loads
the whole model on the devices.

Dry run on the tiny checkpoint (CPU, 2026-11-24, the same scripts with `TINY=1`:
`inkling_tiny_setup.py` makes the toy model and a toy BPE tokenizer trained on the staged
texts, the set is built with `--dry-run-scale 64 8 4 8`): `tiny` 14 s, `check` 3 s, `plan` 4 s,
`pilot` 56 s, `quantize 3.5` 52 s, `eval 3.5` 13 s, all exit 0 (the toy KL is 6.1 and
meaningless: a random model). The gates that must fail, run on the same directory: a
non-production manifest under `PRODUCTION=1` (exit 1), 9 windows when 8 were compared (1), a
tampered `calib.txt` (1; restored: 0), an output that does not exist (1), a second GPU step
while the lock is held (1), a disk gate that cannot be met (1).
`HI2_INKLING_STAGE4_E2E=1 pytest tests/test_inkling_stage4_script.py -k whole_chain` repeats
the chain in 83 s.

The evaluation also ran on the real stock directory on the CPU (2026-11-24, 12 threads,
float32, `--partial`; the student is the stage-2 sample output: experts 0 and 64 of layers 0 and
20, 12 of 9,216 tensors, its 8 quantized attention tensors ignored by the documented gate):
exit 0 in 267 s, `install_teacher` 69.0 s (all 24 layers decoded to float32), one 128-token
teacher pass 29.9 s (48 layer calls), student pass 29.7 s, KL 4.97e-5, top-1 0.992, perplexity
x1.0001, relative RMS of the installed layer-0 weights 0.121, every gate ok. That checks the
names, shapes and decode of the real model inside the script; the numbers are those of 12
tensors, not of a quantization. A 65k-token pass is not a CPU job.

### The pilot on two V620 (measured, 2026-11-24)

The pilot ran on slot a2 (HIP 0,1) as the `pilot` step of the box chain: 4 windows of 2,048
tokens at 3.5 bpw, `--devices cuda:0,cuda:1`, float16 encode, then the float32 evaluation on
the same cards. Exit 0 for both.

| Phase of `hi2 quantize` | Seconds | Note |
| --- | --- | --- |
| model load and capture | 103 | |
| phase 1: layer program with statistics, merge, write, tables | 1,150 | forward 204 (48 layer calls, 4 windows), merge 264 (1 s on layer 0, 21 s on layer 23), tables 500 (20 s per layer), write 12 |
| allocation | 39 | INT2 41.8 % / INT4 58.2 % / INT8 0 % of the rows at 3.5 bpw with 4-window statistics |
| encode | 628 | 9,216 tensors, 14.7 per second, about 26 s per layer of 384 tensors |
| restack | 1,290, now 14 | see the second defect below |
| validation | 155 | |

Host memory peaked at 131 GiB in the quantize and at 232 GiB in the float32 evaluation (the
formula said 184); VRAM was 10 GiB per card in the quantize and 7.1 and 5.5 GiB in the
evaluation; the statistics cache is 40.3 GiB, the stacked output 13.7 GiB (0.6 of them the
vision tower) and the per-expert staging 14 GiB; 88 GiB stay free after three targets; the
hottest card reached 67 C, no page fault. The evaluation (4 windows of 512 tokens):
`install_teacher` 25 s, teacher program 75 s, `install_student` 58 s, student program 72 s;
KL 0.088 +- 0.056 nats, top-1 0.939, perplexity x1.0009. That is a sample of 2,048 tokens of a
student fitted on 8,192 tokens, where every expert Linear was below the token floor and took
the layer average ("128/128 experts below the token floor"): it checks the chain, it is not a
result.

`experiments/inkling_alloc_report.py RUN` prints the allocation of a finished run by layer group, projection and layer (the pilot's: 3.498 bpw, 9.96 GiB of experts against 14.59 GiB in the stock model, INT2 41.8 % and INT4 58.2 % of the weights, blocks 1.3 / 13.6 / 85.1 % of the rows at 16 / 32 / 64).

The pilot found two defects, both fixed in this tree:

* `hi2.service` refused every `cuda:N` with "requested accelerator is unavailable": on torch
  2.12+rocm7.14 under `HIP_VISIBLE_DEVICES=0,1`, `torch.cuda.is_available()` is true and
  `torch._C._cuda_getDeviceCount()` is 2, but `torch.cuda.device_count()` is 0 until
  `torch.cuda.init()` has run (amdsmi answers before it). `cuda_device_count()` starts the
  runtime first; `resolve_config`, the tuning-evaluation check and `doctor` use it.
* `hi2.stack.restack` copied the key list of the shard for every tensor it read
  (`safe_open(...).keys()`: 60 ms for the 83,589 keys of this shard) and for every expert part it
  planned, quadratic in the tensor count: 21.5 min for the 9,216 tensors, every target. It now
  reads each shard's key set once: 14.4 s for the same staging, and the output is byte-identical
  to the old code's (`cmp` of both shards, the index and `config.json`).

### The 3.5 bpw encode on 256 windows (measured, 2026-11-24)

The `quantize-3.5` step of the chain: 256 windows of 2,048 tokens (524,288 tokens of the 265
built), `--target-bpw 3.5 --bits 2,4,8 --candidates lean --no-qad --sensitivity none`, float16
encode on `cuda:0,cuda:1`. Exit 0 in 3,344 s (55.7 min).

| Phase of `hi2 quantize` | Seconds | Note |
| --- | --- | --- |
| model load and capture | 129 | |
| phase 1: layer program with statistics, merge, write, tables | 2,342 | forward 1,324 (14-17 s per layer call, 36-41 s for the second call of a loop layer), merge 218, write 13, tables 512 (20 s per layer; 57 s on layer 0) |
| allocation | 43 | INT2 42.2 % / INT4 57.8 % / INT8 0 % of the rows |
| encode | 597 | 9,216 tensors, 15.4 per second |
| restack | 11 | the fixed one |
| validation | 174 | |

The forward took 6.5 times the pilot's for 64 times the windows: 4.3 s per layer call at 4
windows (204 s for 48 calls), 27.6 s at 256 (1,324 s), which a fixed part of about 4 s per call
and 0.09 s per window fits [two-point fit, inference]. The two projections made from the pilot
(a flat 13 s and a linear 3.6 h) were off by a factor of 100 and of 10. A further target reuses
the statistics and the tables (the cache keys hold the windows, the candidates and the
rotation, not the target: "statistics hit", "tables hit", phase 1 took 6 s): the 3.0 encode
took 966 s, allocation 75, encode 652, restack 11, validation 179, at 46 GiB of host memory
and 5.2 GiB of VRAM per card [measured]. Resources of the 3.5 step, sampled every 30 s: process RSS up to 182 GiB (131 GiB in the
layer log lines), MemAvailable at least 293 GiB, 15.0 GiB of VRAM per card, 66 C at the hottest
card, 0 page faults, disk free 116 GiB at the end.

Token coverage of the 256-window statistics, by (layer, projection) group of 128 experts: 48 of
the 72 groups have every expert at the token floor, 12 have one below it (topped up with the
layer average), 9 have two and 3 have three; the median expert sees 25,000-32,000 tokens
outside the loop and 43,000-84,000 in the loop layers (three passes), the least routed expert of
a layer 1,900-17,000 (the pilot topped up all 128 of every group).

Allocation (`experiments/inkling_alloc_report.py`; shares of INT2/INT4/INT8 by weights, blocks
by rows, bits per weight of everything stored; stock = the NVFP4 experts at 4.5 bits and the
MXFP8 ones at 8.25):

| Scope | Weights G | INT2 % | INT4 % | INT8 % | B16 % | B32 % | B64 % | bpw | GiB | Stock GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all experts | 24.461 | 42.2 | 57.8 | 0.0 | 1.7 | 15.9 | 82.4 | 3.498 | 9.962 | 14.594 |
| layers 0-5 | 6.115 | 34.6 | 65.4 | 0.0 | 1.8 | 16.0 | 82.3 | 3.649 | 2.598 | 3.204 |
| layers 6-17 (loop) | 12.231 | 46.1 | 53.9 | 0.0 | 1.7 | 16.1 | 82.3 | 3.420 | 4.870 | 6.407 |
| layers 18-19 | 2.038 | 42.0 | 58.0 | 0.0 | 1.6 | 13.1 | 85.4 | 3.493 | 0.829 | 1.068 |
| layers 20-23 (MXFP8) | 4.077 | 41.9 | 58.1 | 0.0 | 2.0 | 16.5 | 81.5 | 3.508 | 1.665 | 3.916 |

No row takes INT8 at 3.5 bpw: the global lambda that meets the budget does not buy 8 bits for
any row. The pilot (four windows, every expert topped up) gave nearly the same split: 41.8 %
INT2, 3.498 bpw; INT4 67.5 % in layers 0-5 against 65.4 %, 56.4 % against 58.0 % in layers
18-19, the same elsewhere. The split says where the cost tables put the bits; whether it is the
split the KL wants is the evaluation's question.

### KL of the 3.5 bpw encode against the stock model (measured, 2026-11-24)

`eval-3.5-mixed` of the chain: the held-out `mixed` set (32 windows of 2,048 tokens = 65,536
tokens, disjoint from the calibration by text hash), float32 on the two V620, the teacher the
stock NVFP4/MXFP8 model decoded to float32. All eleven gates ok. 1,474 s for the step.

| Layers quantized | Windows | KL(stock \|\| HI2) | Median / worst window | Top-1 | NLL stock -> HI2 | Perplexity |
| --- | --- | --- | --- | --- | --- | --- |
| all | 32 | 0.0996 +- 0.0178 | 0.061 / 0.538 | 0.930 | 1.7630 -> 1.7724 | x1.0095 |
| all | 8 of them | 0.1027 +- 0.0156 | 0.085 / 0.177 | 0.926 | 1.9069 -> 1.9139 | x1.0071 |
| 0-5 only | 8 | 0.0717 +- 0.0144 | 0.050 / 0.132 | 0.947 | | x0.9992 |
| 6-17 only | 8 | 0.0861 +- 0.0137 | 0.081 / 0.152 | 0.935 | | x1.0126 |
| 18-19 only | 8 | 0.0042 +- 0.0006 | 0.003 / 0.007 | 0.987 | | x0.9984 |
| 20-23 only | 8 | 0.0085 +- 0.0017 | | 0.978 | | x0.9991 |

The group rows quantize one layer group and restore the stock weights elsewhere (gate: max
difference 0). What they show, all measured on the same 8 windows:

* The damage is in layers 0-17. Layers 18-23 hold 25 % of the expert weights at the same 3.5
  bits and cost 0.004-0.009 nats, against 0.072 and 0.086 for layers 0-5 and 6-17, while the
  relative RMS of the weights is alike (0.24-0.29 in 18-19 and 20-23, 0.25-0.35 in 0-17, per
  layer in the evaluation's JSON). The same weight error costs about 17 times more KL in
  layers 0-5 than in 18-19.
* The groups do not add: 0.171 alone against 0.103 together. The errors of different groups
  overlap rather than sum [the sums are measured, the mechanism is not].
* The allocator gave layers 18-23 the same bits as the others (3.49-3.51 bpw against 3.42-3.65):
  its costs are layer-local and this run has `--sensitivity none`. Moving bits from layers 18-23
  to layers 0-17, which a sensitivity-weighted allocation would do, is the first lever to try
  [inference: no run has tested it].
* The mean is carried by few windows: median 0.061 against a mean of 0.0996 and a worst window
  of 0.538.

The two out-of-distribution sets (17 windows of 2,048 tokens each, `--groups none`, `eval-3.5-wiki`
and `eval-3.5-pydocs`, 595 and 593 s per step, all gates ok):

| Set | KL(stock \|\| HI2) | Median / worst window | Top-1 | NLL stock -> HI2 | Perplexity |
| --- | --- | --- | --- | --- | --- |
| wiki | 0.0468 +- 0.0021 | 0.045 / 0.069 | 0.913 | 1.8376 -> 1.8702 | x1.0331 |
| pydocs | 0.0305 +- 0.0028 | 0.027 / 0.054 | 0.946 | 1.1095 -> 1.1332 | x1.0240 |

The KL is lower than on the mixed set (0.0996) but the cost in the negative log-likelihood of the
true tokens is larger (x1.033 and x1.024 against x1.0095): the KL and the perplexity ratio do
not rank the sets alike, so neither alone says how a target will behave on other text.

The step: `install_teacher` 25 s, teacher program 240 s, probe program 82 s, `install_student`
59 s, student program 240 s, metrics 12 s, four group passes, restore 82 s. A float32 pass over
65,536 tokens (48 layer calls, four windows per forward) costs 240 s: 3.1 times the pilot's 75 s
for 32 times the tokens. Peak host RSS 232 GiB, MemAvailable at least 259 GiB, VRAM 19.1 and 15.5
GiB, 59 C.

### The three targets compared (measured, 2026-11-24)

Same calibration windows, same 32 held-out windows, same float32 teacher (the stock model
decoded); the three encodes share the statistics and the tables, so the targets differ only in
the allocation. The stock experts are 14.594 GiB (NVFP4 4.5 bits in layers 0-19, MXFP8 8.25 bits
in layers 20-23); next to each text file lies the vision tower (0.593 GiB, bit for bit); the
MTP file is not carried.

| Target | Experts GiB | INT2 / INT4 / INT8 % of the weights | B16 / B32 / B64 % of the rows | Text file GiB | Step (s) |
| --- | --- | --- | --- | --- | --- |
| 3.0 | 8.539 | 67.8 / 32.2 / 0 | 2.0 / 20.2 / 77.9 | 11.649 | 966 |
| 3.5 | 9.962 | 42.2 / 57.8 / 0 | 1.7 / 15.9 / 82.4 | 13.073 | 3,344 (fills the cache) |
| 4.0 | 11.385 | 16.2 / 83.8 / 0 | 0.9 / 10.6 / 88.5 | 14.497 | 906 |

| Target | KL(stock \|\| HI2), 32 windows | Median / worst window | Top-1 | Perplexity | NLL stock -> HI2 | Mean relative weight RMS |
| --- | --- | --- | --- | --- | --- | --- |
| 3.0 | 0.1212 +- 0.0193 | 0.079 / 0.582 | 0.915 | x1.0243 | 1.7629 -> 1.7870 | 0.340 |
| 3.5 | 0.0996 +- 0.0178 | 0.061 / 0.538 | 0.930 | x1.0095 | 1.7629 -> 1.7724 | 0.274 |
| 4.0 | 0.0851 +- 0.0175 | 0.052 / 0.521 | 0.942 | x0.9992 | 1.7629 -> 1.7622 | 0.188 |

The sizes and the three held-out sets side by side (the whole directory is the text file and the
vision tower; the stock directory is 18.777 GiB with the 0.592 GiB MTP file, which the output does
not carry; the stock experts are 14.594 GiB):

| Target | bpw | Experts GiB | Whole directory GiB | mixed: KL / top-1 / ppl | wiki: KL / top-1 / ppl | pydocs: KL / top-1 / ppl |
| --- | --- | --- | --- | --- | --- | --- |
| 3.0 | 2.998 | 8.539 | 12.242 | 0.1212 +- 0.0193 / 0.915 / x1.0243 | 0.0875 +- 0.0030 / 0.884 / x1.0744 | 0.0523 +- 0.0048 / 0.928 / x1.0440 |
| 3.5 | 3.498 | 9.962 | 13.666 | 0.0996 +- 0.0178 / 0.930 / x1.0095 | 0.0468 +- 0.0021 / 0.913 / x1.0331 | 0.0305 +- 0.0028 / 0.946 / x1.0240 |
| 4.0 | 3.998 | 11.385 | 15.090 | 0.0851 +- 0.0175 / 0.942 / x0.9992 | 0.0180 +- 0.0008 / 0.948 / x1.0132 | 0.0152 +- 0.0016 / 0.961 / x1.0114 |

The wiki and pydocs sets (17 windows of 2,048 tokens each, `eval-T-wiki` and `eval-T-pydocs`, 7
min per step with the teacher cached; all gates ok) fall with the bits much faster than the
mixed set: between 3.0 and 4.0 the KL falls 4.9 times on wiki and 3.4 times on pydocs, 1.4
times on the mixed set.

KL of one layer group at a time on the first 8 windows (all layers together: 0.1284, 0.1027 and
0.0896):

| Target | Layers 0-5 | Layers 6-17 | Layers 18-19 | Layers 20-23 |
| --- | --- | --- | --- | --- |
| 3.0 | 0.0810 | 0.1056 | 0.0065 | 0.0115 |
| 3.5 | 0.0717 | 0.0861 | 0.0042 | 0.0085 |
| 4.0 | 0.0675 | 0.0743 | 0.0027 | 0.0031 |

What the three runs show:

* **The KL falls slowly with the bits.** From 3.0 to 4.0 bpw the mean KL falls by 30 %
  (0.121 to 0.085), the mean relative weight error by 45 % and the allocator's own error ratio
  (the activation-weighted cost of each layer's output, in the log of each allocation) by a
  factor of 4.2 (1,899 to 447). The worst window hardly moves (0.58, 0.54, 0.52). The measure the
  allocator minimises does not follow the end-to-end KL [measured]; routing flips and the three
  passes of the loop layers are candidates for the difference [inference: nothing here separates
  them].
* **Layers 0-17 hold the KL at every target, and improve least with the bits.** Layers 0-5 and
  6-17 cost 0.081 and 0.106 at 3.0 and 0.068 and 0.074 at 4.0 (down 17 % and 30 %); layers
  18-19 and 20-23 cost 0.0065 and 0.0115 at 3.0 and 0.0027 and 0.0031 at 4.0 (down 58 % and
  73 %).
* **INT8 is a candidate and no row takes it** at 3.0, 3.5 or 4.0 bits (the candidates of the
  lean preset with `--bits 2,4,8` are `(2,16) (2,32) (2,64) (4,32) (4,64) (8,32) (8,64)`):
  the shares of INT2 and INT4 only move. Uniform INT4 in blocks of 64 costs about 4.27 bits, so a
  budget of 4.0 needs about 14 % of the weights in INT2 at blocks of 64 (4.27 (1 - x) + 2.27 x =
  4.0 [arithmetic from the format]; 16.2 % measured). INT8 would appear only at a much higher
  budget [inference, no such run].
* **At 4.0 bits the likelihood of the true tokens is that of the stock model** (NLL 1.7622
  against 1.7629) while the distribution still differs by 0.085 nats and the top-1 choice in
  5.8 % of the positions: KL and perplexity answer different questions.
* This is PTQ with no distillation. A KL of 0.085-0.12 nats leaves the final-KL phase the
  whole distance, and the 0.5 bit between 3.0 and 3.5 or between 3.5 and 4.0 buys 0.02 and
  0.015 nats of it.

#### Bits per layer, and what they follow (measured from the reports of the three encodes)

`python experiments/inkling_target_table.py --runs stage4/runs --evals stage4/eval` prints this
table and the ones around it; the tokens are those the calibration counted per expert (the gate
of the expert; its up and down projections see the same tokens): 524,288 positions with 8
experts each per layer, 32,768 tokens per expert on average, three passes over them in the loop
layers 6-17.

| Layer | Tokens per expert | CV of the load | Hottest eighth, % of tokens | bpw 3.0 | bpw 3.5 | bpw 4.0 | 3.5: coldest / hottest quarter of the experts | rho 3.5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 32768 | 0.58 | 26 | 3.056 | 3.679 | 4.140 | 2.37 / 4.29 | +0.88 |
| 1 | 32768 | 0.72 | 31 | 2.978 | 3.516 | 3.998 | 2.30 / 4.29 | +0.95 |
| 2 | 32768 | 0.65 | 27 | 3.072 | 3.647 | 4.187 | 2.38 / 4.29 | +0.92 |
| 3 | 32768 | 0.40 | 20 | 3.191 | 3.759 | 4.182 | 2.42 / 4.28 | +0.86 |
| 4 | 32768 | 0.46 | 23 | 3.197 | 3.663 | 4.166 | 2.37 / 4.28 | +0.87 |
| 5 | 32768 | 0.51 | 24 | 3.175 | 3.633 | 4.148 | 2.38 / 4.28 | +0.84 |
| 6 | 98304 | 0.81 | 32 | 3.020 | 3.537 | 3.969 | 2.30 / 4.29 | +0.92 |
| 7 | 98304 | 0.66 | 30 | 3.011 | 3.661 | 4.077 | 2.35 / 4.29 | +0.91 |
| 8 | 98304 | 0.85 | 33 | 2.967 | 3.491 | 4.017 | 2.31 / 4.29 | +0.95 |
| 9 | 98304 | 0.72 | 31 | 3.009 | 3.555 | 4.086 | 2.34 / 4.29 | +0.93 |
| 10 | 98304 | 0.77 | 33 | 2.988 | 3.507 | 3.930 | 2.29 / 4.29 | +0.92 |
| 11 | 98304 | 0.90 | 34 | 2.961 | 3.415 | 3.941 | 2.29 / 4.30 | +0.93 |
| 12 | 98304 | 0.72 | 32 | 2.967 | 3.528 | 4.022 | 2.30 / 4.29 | +0.94 |
| 13 | 98304 | 0.89 | 34 | 2.935 | 3.435 | 3.944 | 2.30 / 4.29 | +0.93 |
| 14 | 98304 | 0.84 | 34 | 2.928 | 3.399 | 3.971 | 2.30 / 4.29 | +0.95 |
| 15 | 98304 | 1.07 | 40 | 2.852 | 3.362 | 3.849 | 2.29 / 4.31 | +0.95 |
| 16 | 98304 | 1.12 | 40 | 2.856 | 3.245 | 3.849 | 2.29 / 4.30 | +0.98 |
| 17 | 98304 | 1.51 | 55 | 2.686 | 2.908 | 3.442 | 2.29 / 4.33 | +0.98 |
| 18 | 32768 | 1.14 | 40 | 2.881 | 3.380 | 3.807 | 2.29 / 4.31 | +0.95 |
| 19 | 32768 | 0.63 | 28 | 3.074 | 3.607 | 4.104 | 2.32 / 4.29 | +0.91 |
| 20 | 32768 | 0.64 | 27 | 3.089 | 3.583 | 4.016 | 2.32 / 4.29 | +0.90 |
| 21 | 32768 | 0.74 | 30 | 3.064 | 3.436 | 3.975 | 2.30 / 4.29 | +0.94 |
| 22 | 32768 | 0.82 | 34 | 2.966 | 3.387 | 4.010 | 2.31 / 4.29 | +0.96 |
| 23 | 32768 | 0.59 | 27 | 3.039 | 3.627 | 4.122 | 2.37 / 4.29 | +0.88 |

* **Layer 17 gets the fewest bits at every target** (2.686, 2.908 and 3.442); layers 15, 16 and
  18 come next (15: 2.852, 3.362, 3.849; 16: 2.856, 3.245, 3.849; 18: 2.881, 3.380, 3.807). The
  group of layers 15-17 has 3.172 bits per weight at 3.5 against 3.454-3.685 for the other seven
  groups of three layers.
* **The bits follow the load of the experts.** Inside a layer the allocator gives bits almost in
  the order of the experts' token counts: the Spearman correlation of an expert's tokens with its
  bits, averaged over the 24 layers, is +0.98, +0.92 and +0.74 at 3.0, 3.5 and 4.0. The coldest
  quarter of a layer's experts gets 2.28-2.42 bits per weight (INT2) at 3.0 and at 3.5 against
  3.83-4.33 for its hottest quarter (the low end is layer 17 at 3.0, then layer 15 at 4.20); at
  4.0: 2.31-3.86 against 4.28-4.38.
* **A layer's bits follow the skew of its load.** Over the 24 layers the mean bits per weight
  correlate with the CV of the tokens per expert at -0.92, -0.94 and -0.91 and with the share of
  the tokens that go to the hottest eighth of the experts at -0.95, -0.95 and -0.92 (3.0, 3.5,
  4.0). Layer 17 has the most skewed load: CV 1.51, 55 % of its tokens go to its 16 hottest
  experts (20-40 % in the other layers) and 23 of its 128 experts carry less than a quarter of
  the mean load (0-14 in the others, median 7). Its few bits are the bits of its rarely routed
  experts [measured]; that a cost weighted by the activations of each expert makes them cheap to
  damage is the reading of the allocator [inference].
* **Where the tokens go and where the predicted error sits** (gate experts below 3 bits per
  weight, the INT2 class, counted from the reports): at 3.0, 3.5 and 4.0 bits they are 66 %,
  41 % and 16 % of the experts, receive 41 %, 19 % and 4.8 % of the routed tokens (27 % in layer
  17 at 3.5) and carry 98 %, 96 % and 89 % of the token-weighted predicted squared error
  (`n_tokens * pred_rel_err^2`) [measured].

#### KL by groups of three layers at 3.5 bpw (measured, `eval-3.5-fine`, 2026-11-24)

`inkling_stage4.sh eval 3.5 fine`: the first 8 held-out windows, one group of three layers
quantized at a time and the stock weights everywhere else (the gate: largest difference 0), the
teacher pass cached; 1,670 s for the step (2 min per group), all gates ok, 67 C at most, no page
fault. All 24 layers together on the same 8 windows: 0.1027 +- 0.0156. The columns after the KL
are the allocation at 3.5 (and the bits at the other two targets) and the weight error the
evaluation measured:

| Layers | KL +- sem | Top-1 | bpw 3.0 / 3.5 / 4.0 | INT2 % (3.5) | Predicted rel. error, mean (allocator) | Weight RMS error | KL / predicted rms^2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0-2 | 0.0675 +- 0.0133 | 0.949 | 3.035 / 3.614 / 4.108 | 36.4 | 0.0381 | 0.274 | 21.4 |
| 3-5 | 0.0694 +- 0.0130 | 0.950 | 3.188 / 3.685 / 4.165 | 32.9 | 0.0346 | 0.271 | 25.2 |
| 6-8 | 0.0715 +- 0.0143 | 0.951 | 2.999 / 3.563 / 4.021 | 38.7 | 0.0407 | 0.313 | 20.7 |
| 9-11 | 0.0593 +- 0.0133 | 0.953 | 2.986 / 3.492 / 3.986 | 42.4 | 0.0441 | 0.262 | 15.5 |
| 12-14 | 0.0532 +- 0.0138 | 0.958 | 2.943 / 3.454 / 3.979 | 44.5 | 0.0459 | 0.255 | 13.4 |
| 15-17 | 0.0421 +- 0.0090 | 0.957 | 2.798 / 3.172 / 3.713 | 58.9 | 0.0593 | 0.307 | 7.7 |
| 18-20 | 0.0056 +- 0.0010 | 0.985 | 3.015 / 3.523 / 3.976 | 40.5 | 0.0427 | 0.257 | 1.5 |
| 21-23 | 0.0059 +- 0.0014 | 0.981 | 3.023 / 3.484 / 4.036 | 43.3 | 0.0440 | 0.256 | 1.6 |

(`python experiments/inkling_target_table.py --runs stage4/runs --evals stage4/eval` prints it
with the columns I left out: the root mean square of the predicted error and the KL per squared
weight error.)

* **Layers 15-17 do not carry the KL.** The group has the fewest bits (3.172), the most INT2
  (58.9 %), the highest predicted error and, after layers 6-8, the highest weight error (0.307),
  and it costs 0.0421: less than every group of layers 0-14 (0.053-0.072), 41 % of the all-layer
  KL, 7 times each of layers 18-23 (0.006). The under-allocation of layers 15-17 (layer 17:
  2.908 bits) is real [measured], and it does not explain the KL [measured on these 8 windows].
* **The allocator's predicted error ranks the groups against the KL.** The Spearman rank
  correlation of the eight groups' KL is -0.55 with the predicted relative error, +0.62 with the
  bits per weight (the groups with more bits cost more), -0.62 with the INT2 share and +0.60 with
  the measured weight error. The KL per squared predicted error is 25.2 in layers 3-5 and 1.5 in
  layers 18-20: a factor of 16.6 between groups; per squared weight error the factor is 11.1. The
  allocator's costs are layer-local and `--sensitivity none`, so it cannot see the difference
  between layers [measured: the numbers; inference: the cause]. Eight groups and eight windows
  make the correlations descriptive statistics, not tests.
* **The KL is in layers 0-17 and nearly absent in layers 18-23** (0.0056 and 0.0059 at the same
  3.5 bits and the same weight error as layers 9-14): the profile is flat over layers 0-8
  (0.068-0.072) and falls from layer 9 on.
* **The groups do not add**: the eight sum to 0.375, the same layers together cost 0.103, and
  each of the first six groups alone costs 41-70 % of that. Quantizing three layers is already
  most of the damage. Two readings, which the data do not separate [inference]: the output is
  fragile at some positions or windows and a perturbation of any of the first 18 layers finds
  them (windows 0 and 7 of these 8 are among the heavy ones of the mixed set: 0.177 and 0.151 at
  3.5 bits), or the INT2 experts of every group (a fifth of the routed tokens at 3.5 bits) each
  do the same damage. If the first reading holds, moving bits between layers gains little: along
  the allocator's frontier the KL of layers 0-5 falls only 17 % between 3.0 and 4.0 bits (0.081,
  0.072, 0.068). The load-flattened allocation (bits for the rarely routed experts) and a routing
  log of the held-out windows are the two experiments that would decide.

#### What the mixed-set KL is made of: a few windows that do not improve (measured)

The mixed set's KL per window, for the three encodes (`kl_per_window` in the evaluation JSONs):

* **The same windows are heavy at every target.** The rank correlation of the 32 windows' KL
  between two targets is 0.97 (3.0 and 3.5), 0.96 (3.5 and 4.0) and 0.95 (3.0 and 4.0). Window
  17 costs 0.582, 0.538 and 0.521 nats and window 30 0.365, 0.323 and 0.303; windows 0 and 15
  follow at 0.15-0.21. The three heaviest windows hold 30 %, 33 % and 36 % of the sum of the
  32 windows' KL at 3.0, 3.5 and 4.0.
* **They improve least with the bits.** Between 3.0 and 4.0 the KL of window 17 falls by 10 %, of
  window 30 by 17 % and of window 0 by 23 %; the median of the 32 windows' ratios is 0.61 (range
  0.32-0.89), and the mean of the other 27 windows falls from 0.0855 through 0.0672 to 0.0535
  (-37 %).
* **What they are** (the ids of `heldout_mixed_ids.npy` decoded with the stock tokenizer):
  window 17 is chat in Russian and Spanish (6.7 % of its letters are Cyrillic; presumably the
  OpenAssistant source [inference]), window 0, the third heaviest at 3.5, has 8.1 % Cyrillic
  letters, window 30 is the vLLM documentation in markdown followed by a function-calling record
  (English); the lightest window (26) is Python code with an English exchange. Over the 32
  windows the Pearson correlation of a window's KL with its share of non-English letters
  (Cyrillic, CJK, accented Latin) is 0.51, with its number of special tokens 0.05. The
  calibration is not free of such text (1.7 % of its letters are Cyrillic, 0.34 % CJK and Kana,
  0.23 % accented Latin; 18 of the 256 windows have more than 10 % Cyrillic; the 32 held-out
  windows 0.85 % Cyrillic). That these windows route to the experts the allocator left at INT2
  is the likely cause [inference: the routing of the held-out windows is not recorded; a later
  evaluation can log it].
* The wiki and pydocs sets (English text and Python documentation) behave as the allocator
  predicts: their KL falls 4.9 and 3.4 times between 3.0 and 4.0 (the allocator's error ratio
  4.2 times), against 1.4 times for the mixed set. The floor of 0.085 nats on the mixed set at
  4.0 bits is made of content, not of the bit budget [measured for the windows, inferred for the
  cause].

### Costs and risks

* **GPU needs.** A lease of slot a2 (HIP 0,1, both cards free, 28 GiB each as in the smoke)
  in three blocks [inference, before the pilot]: the pilot (1-2 h; measured: 56 min with the
  old restack, 35 min with the fixed one, plus a 6 min evaluation), the first encode and its
  evaluations (measured: the 3.5 encode on 256 windows took 56 min, of which the forward was
  22 min, see the section above; the evaluations are added), and the 3.0 and 4.0 encodes (the
  statistics are cached: allocation, encode, restack and validation, about 15 min each, with
  their evaluation). The encode of 24.5 G weights with GPTQ and two refinement sweeps
  took 628 s on the two cards [measured, the pilot, above], not the 32 min of the stage-2
  data-free estimate. The fit limit (22.2 / 22.9 / 23.6 GiB of 24.0 for the final-KL student
  at 3.0 / 3.5 / 4.0) matters only for a
  later QAD stage: a `--no-qad` encode trains no student. A step runs for hours, so the
  guard of the box holds its `boot.lock` for the first 600 s of a step only
  (`GUARD_BOOT_SECONDS`; measured with stand-in commands: a second taker waited 2 s of a 12 s
  command with a 3 s window, 7 s of a 9 s command without one).
* **Host memory** 100-130 GiB resident in the program and student phases of the float16
  encode [measured], 131 GiB at the peak of the pilot; a float32 evaluation holds 120 GiB after
  the load and 184 GiB after its first program pass [measured, stage 3, 2 x 128 tokens], 232 GiB
  at the peak of the pilot's evaluation [measured]. The box has 503 GiB.
* **Disk** 170 GiB free at 96 % [measured]: the pilot writes ~54 GiB and is removed before the
  real run; then the cache (39.3 GiB) and three outputs (~15 GiB each) leave ~85 GiB, above
  the 60 GiB spare. The gate refuses a step that would break it.
* **KL cost [inference].** The evaluation runs in float32 (`EVAL_DTYPE`; the encode keeps
  float16): float16 rounding alone is a KL of 3.2e-3 against float32 on the same weights
  [measured, stage 2, 512 tokens], which would be a large part of the signal at 4 bpw. The
  program is transfer-bound at small token counts (52-55 s per forward of two windows in float16
  [measured]; float32 doubles the bytes): the pilot's float32 passes over 2,048 tokens took 75 s
  (teacher) and 72 s (student); a pass over 65,536 tokens took 240 s [measured, the 3.5
  evaluation], 3.1 times for 32 times the tokens. The first student adds the teacher pass
  and the group breakdown (4 groups x 8 windows, 25 min for the whole step), the later ones the
  student pass alone.
* The held-out set is chat, tool use, instructions, arithmetic and docs, mostly English: it holds
  some chat in other languages (window 17: Russian and Spanish) and some code (window 26) in
  proportions nobody counted, and nothing of long context (windows of 2,048 tokens); the OOD
  sets are small (17 windows each).

## Layer-wise QAD on a looped model

`qad.qad_layerwise` trains layer L from the quantized prefix's output to the unquantized layer's output, one
hidden tensor per layer. It does not fit Inkling-2: a layer call is `layer(hidden, block_states[, messages],
**kwargs)` and its 24 layers are 48 calls. `hi2.qad_looped` trains the calls **teacher-forced**: the layer
program (`looped._Program.run`, the unquantized model) records each selected call's input, block states, messages
and output, and `train_layer` runs the fake-quantized layer on exactly those inputs. The design constraints, read
from `qad.py`, `looped.py`, `layerwise.py`:

* **Layers 0-5 and 18-23 (one call each): yes.** The layer program (`looped._Program.run`) hands every call its
  input, block states and output over per-sequence host states, so the teacher's inputs and targets come from one
  pass. The recipe's student input (the quantized prefix's output) needs a second program through the quantized
  layers (for 18-23 through the loop span too: 36 calls, about 19 min at the 3.5 encode's measured 1,324 s forward for
  48 calls); teacher-forced inputs need none.
* **Loop layers 6-17 (three calls, one set of weights): yes, teacher-forced.** One fake-quantized set per layer; each
  pass is its own sample (teacher input, the pass's kwargs, its messages from `looped._Messages` over the stored states
  of the pass before), the loss the sum of the passes' nMSE. No student input inside the span: the messages of pass 2
  come from layers 6-17 of pass 1, and those above L are still PTQ. [INFERENCE] Whether teacher-forced layers help the
  final KL is not known; the one-layer test measures it.
* **Storage.** One state of 264 windows (256 + 8 held-out) x 2048 x 2304 is 2.32 GiB in float16, 4.64 in float32: the 48
  outputs 111 / 223 GiB, the span's 37 states 86 GiB; block states are views, messages (75 MB per window and call) are
  recomputed. The disk has 60 GiB free, so RAM (shared) or 128 windows (44 GiB). A single-pass layer alone needs
  `qad_layerwise`'s three buffers, 7 GiB.
* **VRAM** is the layer's trained state, not the activations: `FakeQuantLinear` keeps a float32 latent, its `unit`, the
  gradient and two Adam moments, 20 B x 1.019 G weights (128 experts x 7.96 M) = 19 GiB, plus 2-4 GiB of decoded weights.
  The torch decode graph (several float32 copies) does not fit 30 GiB; the fused `qad_kernel` might ([INFERENCE], never
  run on 128 experts); else expert groups (32 live, 96 frozen at their PTQ decode).
* **Time.** `estimate_seconds` gives 3.5 s per step of 4 x 2048 tokens at this layer size ([INFERENCE]: scaled from dense
  layers, never run on an MoE one): 300 steps are about 20 min per single-pass layer on one card; a loop layer decodes once
  per step and runs three calls per sequence, at most 3x. Measured bound: a final-KL step costs 3.4 s per layer call and
  window with the recompute (163 s for 48 calls). Whole run: 12 x 20 min + 12 x up to 60 min plus the two program passes,
  about 40 min ([INFERENCE]; two cards help less than 2x).
* **Reusable:** `FakeQuantLinear` with `qad_kernel` and `export`; the AdamW, cosine, keep-best and held-out body of
  `train_layer`; `StepBudget`, `split_indices`, `fingerprint`, the artifact `layerNNN.pt` and `QadResult`; the PTQ
  initialisation (`encode_many` through `StatsView`, its statistics already pooled over the calls). **Not:**
  `capture_inputs`, `_propagate`, `_buffers` (one tensor per layer), the driver loop (layer i+1 reads layer i) and
  `train_layer`'s `functional_call(layer, wd, (x,), kw)`, which takes only a hidden tensor.
* **Hard:** the 19 GiB state in 30 GiB, the span's storage, teacher-forced inputs in the span, the time.

### Implemented, `hi2/qad_looped.py` (2026-11-24)

`pipeline._check_looped` sends a looped model to `qad_looped` when `--qad-layers` is given explicitly (`all`, or a
list); **omitted** keeps the final-KL-only default (PTQ per layer), so the three arms of the comparison are
`--no-qad`, `--qad`, and `--qad --qad-layers 0-5`; the
What is still refused is `--qad-kl-layers` (a decoder layer's output is not the LM head's input) and
`--sequential`; `--qad-layers none` keeps its PTQ-per-layer meaning.

* **Capture.** One pass of the unquantized program, per-sequence host tensors (`_Program` already splits them),
  stops after the last call the selected layers need. A state is stored once: an output the program continues with
  *is* the next call's input (the same tensor object), a later pass's start is the first pass's pre-loop state, and
  the end of a pass plus the final output are the extra states. `_layout` recomputes the block-state append, the
  pass reset and the `n_pre + 1` message slice without running the program; `_state_count` predicts the number of
  slots and refuses a pass over `HI2_QAD_STATES_GIB` (default 128 GiB) before it starts. States stay in the model's
  compute dtype (float16 on the V620): 2.32 GiB per state for 264 windows x 2048 tokens. Messages are never stored;
  `_Samples` recomputes them from the previous pass's stored states with a CPU copy of `message_input`.
* **Training.** A single-call layer has one sample per window; a loop layer one per (window, pass), all of one
  window's samples in one `_GroupBatches` group so the passes reach an optimizer step together and the loss is the
  sum of their nMSE up to the constant `1 / batch` (`train_layer`'s new `groups` hook). `_Samples` supplies the
  recorded call kwargs, the block states and the messages; `functional_call` replaces the layer's weights, so the
  rest of the model is not in the graph. The experts of a layer train group by group (`QadConfig.expert_groups`,
  `--qad-expert-groups`; 0 = enough groups to keep one under 300 M weights; the other Linears train in group 0),
  which keeps one group's latent/unit/gradient/Adam state around 6 GiB instead of 19 GiB.
* **Artifacts.** `qad/layerNNN.pt` per *trained* layer (tensors, info, norms) with a fingerprint that includes the
  pass count, `expert_groups` and the selected layers, so a re-run reuses exact artifacts and training an unselected
  layer is never resumed as if it had trained; unselected layers are PTQ-encoded again (deterministic). The pass
  that runs to the end saves `head_inputs.pt` only when `final_kl > 0`, for `hi2.finalkl`'s placement-free phase.
* **Tests.** `tests/test_qad_looped.py` (CPU, the fork's tiny looped checkpoint): the exact unique-state count, the
  dedupe identity, the memory guard, the head cache, the later passes' messages and block states against a reference
  `run_program`, the group split, the pass groups, a full `hi2 quantize` (layers 0-2, loop layer 2 with two passes)
  and its resume; 21 mutants of the module and its groundwork are killed (QL3-QL20, QG1/QG2/QG4).
* **Not yet measured on the GPU:** one 3.5 bpw run over layers 0-5 (`--qad-layers 0-5 --qad-final-kl 0`) and its
  held-out group KL is the next step; the time [INFERENCE] follows `estimate_seconds`: 3.6 s per 4 x 2048-token
  step at 1.019 G expert weights, 300 steps x 6 layers -> about 2 h on one card, plus the capture and the final
  encode. The final-KL phase separately measured 166-183 s per step with the layer program at 18.5 GiB/card.

## One bf16 class in INT8 (exploratory, eval only)

Quality must not degrade: the parts that are bf16 in the stock checkpoint (attention, embedding, LM head; the vision tower
stays) become INT8 only if that is shown to be free. `experiments/inkling_int8_probe.py` and three options of
`experiments/inkling_kl_eval.py` measure it **one class at a time**, inside the float32 KL eval (no runtime path exists
for INT8 attention, embeddings or head: this is a measurement of the numbers, not of a speed or a format):

* **The scheme** is HI2's own INT8 row candidate `(8, B)` through the repo's data-free encode (`Config(candidates=[(8, B)],
  rounding="nearest", smooth=0.0)`, `build_table`, `encode_with_picks`, `hi2.format.dequantize`), `B` = 64 first (8 + 16/64
  bits per weight + 48 bits per row = 8.25 bpw, 7.75 bits saved per weight), 32 only for a class that fails (8.5 bpw).
  Classes: `attn` (`q/k/v_proj.linear`, `o_proj.linear`, `o_proj.gate_proj`, `relative_position_bias.linear` of the 24
  layers; not `sink_proj` or the short convolutions), `embed` (`embed_tokens`), `lm_head`.
* **The eval** is `inkling_kl_eval.py --groups none --int8-class C --int8-block B [--stock-student]`: the teacher pass
  (or its cache) is the stock model with the class in bf16, the student is the PTQ output (or, with `--stock-student`,
  the stock experts: the class alone) with the class stored as INT8 and read back. For `lm_head` the model keeps its own
  head and the student scores through a copy (`inkling_int8_probe.apply` returns the student's logits function), so the teacher is never
  touched. A gate (`int8_class_changed_the_weights`) fails when the class did not move; `student_differs_from_the_teacher`
  is skipped for a stock student. The result has an `int8` block: tensors, weights, bytes in bf16 / INT8 / saved, bits per
  weight, rms and largest weight error.
* **The rule** (mine, stated before the numbers): a class PASSES when the paired per-window increase of KL over the same
  model with the class in bf16 is at most 1 paired standard error AND the top-1 agreement moved by at most 1e-4, on all
  three held-out sets (mixed, wiki, pydocs). Otherwise it stays bf16. `inkling_int8_probe.py report --eval DIR --student
  bpw3.5 --classes attn embed lm_head --block 64 --json OUT` applies it to the eval files
  (`bpwT-int8CLSB_SET.json` against `bpwT_SET.json`; `stock-int8CLSB_SET.json` gives the class alone and its share of the
  PTQ KL) and always exits 0: the verdicts are in the output. The driver steps are `inkling_stage4.sh int8 T CLASS [SET]
  [stock]` (`INT8_BLOCK`, default 64).
* **Limits.** Eval-only; weight-only data-free INT8 (no activations; a calibrated encode is not tried); one class at
  a time (the classes are not shown to add up); the float32 eval's own noise is the paired standard error above, and a
  class that passes at 1 sem may still cost up to about that much KL: the report prints the increase, not just the verdict.

## Known gaps before a quantization run

* **`hi2.moe.UnfusedExperts` and Inkling (fixed on this branch).** It used to keep the fused
  module's `act_fn` and call `act_fn(gate_proj(x)) * up_proj(x)`, but the experts of Inkling have a
  two-argument gated activation, `Inkling2LoopSoftcappedSwiGLU.forward(gate, up) = 4 tanh(gate / 4)
  * sigmoid(gate) * 25 tanh(up / 25)` (`mlp_gate_softcap` 4.0, `mlp_up_softcap` 25.0 in
  `config.json`), so `unfuse_experts` on a real `Inkling2LoopForConditionalGeneration` raised
  `TypeError: ... missing 1 required positional argument: 'up'` on the first forward (the stage-1
  record of `experiments/inkling_unfuse_check.py`, exit 1). `UnfusedExperts` now detects a
  two-argument `act_fn` from its signature (`hi2.moe.is_gated_activation`) and then calls
  `act_fn(gate, up)` and adds the top-k outputs in float32, as the transformers class does; any
  other activation keeps the one-argument forward and the hidden-dtype sum bit for bit
  (`tests/test_moe.py`: the first forward copied verbatim against the current one in float32,
  bfloat16 and float16). The gate script now exits 0, largest logit difference 8.3e-7 against
  the fused model; the router applies `route_scale` 8.0 itself.
* **Layer program, final KL and preset (fixed on this branch; run on the real model on two V620
  with a stand-in student, not with a calibrated PTQ).** Calibrated PTQ runs on Inkling through
  [the layer program](#the-layer-program-hi2loopedpy) (two layers calibrated on the real model:
  [GPU smoke](#gpu-smoke-on-two-v620-measured));
  [the final-KL phase](#final-kl-on-inkling) has the soft cap, the placement, packed codes,
  per-expert checkpointing and its own fit check (80 % of a card by the formula; two steps of the
  whole real-sized student ran there); `--include '\.mlp\.experts\.'` is the default. Still
  unsupported on this model, and refused up front: `--sequential` and layer-wise QAD training.
* **The default `--exclude` does not keep Inkling's small linears dense** (the census above); the
  preset's include avoids it, an explicit `--include` that selects attention must add its own
  exclusions.
* **The output is not loadable by the fork's runtime yet**: see [what a loadable HI2 Inkling
  directory needs](#what-a-loadable-hi2-inkling-directory-needs) (no HI2 expert path, the names, the
  declaration of the formats, `mtp.safetensors` to deploy beside it).
* `hi2 eval` and `export_dense` write a dense float16 copy next to the output (52.8 GB; the build
  box disk is at 97 %).
* `--compute-dtype float32` holds the teacher at 105 GB and float16 at 52 GB (stage 1: KL against
  float32 6.6e-4 on 22 tokens, 3.2e-3 on 512). Neither dequantizes the NVFP4 experts exactly like the
  reader (about 9 % of their elements differ by a unit in the last place in float32, up to two in
  half precision, see above), so `check_unfused` allows two units for a stock directory.
* The Hugging Face class of the model (`Inkling2LoopForConditionalGeneration`) is not in the
  pinned transformers; the fork's tree supplies it (`~/w4a8_runs/pydeps/tfm-main` on the build
  box, on `PYTHONPATH`) and `register_auto_class` registers it.
* The slot order of `message_input.query` (`q, k, v, mlp`) is the one of transformers' own
  conversion, `Inkling2LoopMergeMessageQueries` ("the query, key, value and MLP queries of each
  loop layer, in that order"), and `message_query()` equals what `from_pretrained` builds bit
  for bit on the real directory (`inkling_hf_load_check.py`). The first token of the fork's CPU
  oracle alone cannot separate it from the two other orders the fork tried (all three give
  `<|message_model|>` at 26.3-28.0), so the served model is the check that remains.
* `config.json` says `quantization_config: {quant_method: modelopt, quant_algo: NVFP4}` for
  the whole model, but layers 20-23 are MXFP8 in the file: the reader decides from the dtypes
  in the header, never from the config.
* The vision tower (`model.visual.*`, 23 layers, 1024 wide) is carried to the output bit for bit
  but not read by `hi2`; the MTP head is neither read nor carried (deploy `mtp.safetensors` beside
  the output).
