# HI2 for mixture-of-experts models

`hi2`'s MoE PTQ path calibrates, allocates and encodes routed experts into the
v3.2 flex format (`quant_method: rdna_hint_flex`). A per-expert projection is an
ordinary `model.layers.N.mlp.experts.E.{gate,up,down}_proj.hi2_*` tensor; the
current default instead groups experts into stacks per layer/projection group.
The tensor decode semantics are unchanged, but a native loader must support the
selected layout. Routers, norms, embeddings and heads stay dense.
This note records the implemented paths and the measured OLMoE-1B-7B-0924 PTQ
case, not general MoE QAD or runtime compatibility. Check the
[support matrix](getting-started.md#supported-inputs-and-evidence) first.

## How experts are handled

| Step | What the tool does | Why |
| --- | --- | --- |
| Model view | transformers 5 builds OLMoE (and Qwen2-MoE, Mixtral, ...) with fused 3-D experts (`experts.gate_up_proj [E, 2I, H]`, `experts.down_proj [E, H, I]`). `hi2.moe.unfuse_experts` replaces that module by a ModuleList of per-expert `gate_proj` / `up_proj` / `down_proj` Linears, views of the same storage, with the same forward. The first expert views are compared bit for bit with the checkpoint tensors before calibration. | The module paths are the checkpoint names, so the dense machinery (hooks, resolver, writer) applies unchanged. Checkpoints with per-expert ModuleLists need no adapter. |
| Statistics | An expert Linear runs only on the tokens routed to it, so its 16x16 block moments and its Hessian are those tokens' only. | Spec 8.7: per-expert statistics. |
| Shared inputs | Linears receiving the very same tensor (q/k/v; gate/up of one expert) are detected on their first call, which for an expert can be any sequence, and share one accumulator and one Hessian (stored once, the others are symlinks). They get one smoothing vector and one set of rotation flags (`calibrate.shared_transform`, chosen on their rows together under `--rotation auto`). | The fork rejects an expert whose gate and up transforms differ, and fuses q/k/v and gate/up into one call when they match. |
| Token floor | An expert that saw fewer than `--expert-min-tokens` (default 2) x its widest input routed tokens keeps its own tokens and fills the missing ones with the layer average of its projection: `S_e + (m - n_e) * S_layer / n_layer`, for the block moments and the Hessian. The affected experts are logged per layer; `workdir/expert_tokens.json` holds every expert's token count, the topped-up list and min / median / max per layer and projection. An expert group with no tokens at all falls back to data-free weights (logged). | A Hessian from a few hundred tokens is rank deficient; the layer average is the best prior for the missing directions. |
| Allocation weight | The allocator minimises `sum_t w_t * rel_err_t`. A dense tensor has `w = 1`; the E experts of one layer and projection get `w_e = share_e / E` with `share_e = n_e / mean(n)` (floored at `--expert-share-floor`, 0.05). | Each token sees a router-weighted average of its experts' errors, so the whole group weighs as one dense projection and an expert counts in proportion to the tokens it serves. Without this, 3072 expert tensors would outvote 64 attention tensors. |
| Sensitivity probe | `--sensitivity probe` quantizes a unit at the reference configuration and measures the KL: a dense tensor alone, or all experts of a layer and projection at once (112 probes for OLMoE instead of 3136). The unit's `s_u = dKL / sum_e w_e rel_err_e` is median-normalised and floored, and each member gets `s_u * w_e`. `--sens-dtype float16` puts a model that does not fit in fp32 on one GPU (bf16 has no GEMM on gfx1030 and is emulated very slowly). | Keeps the probe one forward per unit. |
| Reuse | Compatible persistent statistics/tables/probe entries can serve several allocation targets. Current output/workdir reuse also requires the matching run manifest; use separate output/work directories for a changed allocation rather than relying only on `phase1.json`. | Phase 1 and the probe were 25 of the 100 minutes in the first historical OLMoE run. |
| Writer and export | The checkpoint is streamed as before; `load_dequantized`, `export-dense`, `eval` and `verify` work on the per-expert names (transformers fuses them again on load) in either layout. `eval --dtype` selects the dtype of both models. `--moe-layout stacked` writes one stack per layer and gate_up/down group instead (see "Stacked expert layout"). | |

## Historical PTQ recipe

This reproduces the sampling and PTQ intent of the 2026-10-26 OLMoE experiment,
not a qualified MoE QAD recipe. `$OLMOE` must name a local safetensors checkpoint
with per-expert 2-D weights. `calib_mix.txt` needs at least 364,544 tokens counted
with that model's tokenizer; `eval_mix.txt` is a separate untouched corpus.
The explicit per-expert layout matches the original experiment; new quantizations
otherwise default to stacked.

```bash
python -m hi2 quantize "$OLMOE" out-r4 --workdir olmoe-r4.work --devices cuda:0 \
  --calib calib_mix.txt --calib-samples 178 --seq-len 2048 --no-qad \
  --sensitivity probe --sens-dtype float16 --max-error-ratio 4 --moe-layout per-expert
python -m hi2 quantize "$OLMOE" out-b3.0 --workdir olmoe-b3.0.work --devices cuda:0 \
  --calib calib_mix.txt --calib-samples 178 --seq-len 2048 --no-qad \
  --sensitivity probe --sens-dtype float16 --target-bpw 3.0 --moe-layout per-expert
python -m hi2 eval "$OLMOE" out-r4 --text eval_mix.txt --skip-tokens 0 \
  --device cpu --dtype float32
```

The second allocation can reuse compatible persistent statistics and probe entries,
while keeping separate output/work state. Both commands run PTQ only: no layer-wise
QAD or final-KL. Without `--no-qad`, current default QAD requires 540,672 tokens,
more than the historical corpus. See the [support matrix](getting-started.md#supported-inputs-and-evidence)
and [recovery rules](operations.md#resume-and-invalidation) before changing a recipe.

## OLMoE-1B-7B-0924 (measured 2026-10-26)

Setup: one V620 (gfx1030), calibration on the head of `calib_mix.txt` (docs, chat and
tool-calling text, 178 x 2048 = 364,544 tokens, the whole file), `--rounding gptq`
(default), smoothing 0.5 + rotation on, `--sensitivity probe --sens-dtype float16`,
token floor 2 x input width. Evaluation with `hi2 eval` in fp32 on CPU, 8 x 2048
tokens: `study_text_all.txt` from token 800,000 and `eval_mix.txt` from token 0.
Checkpoints: `olmoe-r4` and `olmoe-b3.0`.

| Run | bpw (HI2 tensors) | bpw (whole file) | INT2 / INT4 rows | experts bpw | attention bpw | allocation error ratio |
| --- | --- | --- | --- | --- | --- | --- |
| `--max-error-ratio 4` | 3.539 | 3.97 | 40.0% / 60.0% | 3.49 | 4.67 | 4.00 |
| `--target-bpw 3.0` | 3.000 | 3.45 | 67.4% / 32.6% | 2.93 | 4.57 | 7.43 |

"Whole file" is the checkpoint size (3.44 GB and 2.98 GB, against 13.84 GB in bf16)
over all 6.92 B parameters, with embedding, head, router and norms kept in bf16.

| Run | study ppl (orig 6.634) | ratio | KLD (nats/token) | top-1 | mix ppl (orig 3.924) | ratio | KLD | top-1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ratio 4 (3.54 bpw) | 6.707 | 1.011 | 0.0169 | 94.4% | 4.108 | 1.047 | 0.0480 | 90.2% |
| 3.0 bpw | 6.773 | 1.021 | 0.0313 | 91.8% | 4.287 | 1.093 | 0.0910 | 87.0% |

The chat and tool-calling text (`eval_mix`) loses about three times as much as the
prose text at both sizes.

Per projection (bpw, ratio 4 / 3.0 bpw): q 4.32 / 4.15, k 4.30 / 4.14, v 5.03 / 4.99,
o 5.03 / 4.97; experts gate 3.29 / 2.79, up 3.52 / 2.94, down 3.66 / 3.07. The probe
(sum over a unit, mean over layers, median 1) rates the expert groups (gate 1.16, up
1.37, down 1.40) and v/o (1.28, 1.15) well above q/k (0.37, 0.43).

### Expert coverage

Routed tokens per expert (gate, up and down of an expert see the same tokens), out of
364,544 x 8 routed slots per layer (45,568 per expert if uniform):

| Layer | min | median | max | Layer | min | median | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 5,582 | 39,860 | 178,164 | 8 | 4,964 | 36,414 | 224,103 |
| 1 | 11,763 | 38,042 | 207,533 | 9 | 9,338 | 36,551 | 218,736 |
| 2 | 8,457 | 37,162 | 241,330 | 10 | 5,251 | 36,785 | 203,442 |
| 3 | 11,036 | 36,860 | 167,767 | 11 | 8,942 | 33,717 | 222,244 |
| 4 | 5,206 | 32,071 | 195,731 | 12 | 5,375 | 35,151 | 202,658 |
| 5 | 9,361 | 31,200 | 227,366 | 13 | 6,921 | 33,321 | 179,215 |
| 6 | 8,113 | 33,054 | 211,453 | 14 | 4,865 | 35,072 | 206,220 |
| 7 | 3,358 | 34,442 | 223,271 | 15 | 6,122 | 31,146 | 207,265 |

Every one of the 1,024 experts received tokens; 12 got fewer than 8,192 and one (layer
7, expert 51, 3,358 tokens) was below the 4,096-token floor and topped up (its three
tensors). Routing is skewed: the busiest expert of a layer sees 4 to 7 times the median.
With the 16k-token calibration of a smoke test, 63 of 64 layer-0 experts were below the
floor, so the full calibration file is needed.

### Cost

| Stage | Time | Peak memory |
| --- | --- | --- |
| Phase 1 (calibration, Hessians) | 15.5 min | host RSS about 58 GiB (fp32 model in RAM) |
| Sensitivity probe (112 units, fp16 model on the GPU) | about 10 min | GPU 13.8 GiB |
| Phase 3, ratio 4 | 72 min | host 37 GiB (run with the probe) |
| Phase 3, 3.0 bpw (phase 1 and probe reused) | 75 min | host 7.5 GiB |

The historical run's per-expert Hessians (gate/up shared, fp32) took about 21 GB
on disk and were deleted after its second allocation; this is not a cleanup
recommendation. The current persistent cache retains reusable statistics; its
measured OLMoE entry is 11.8 GB ([speed.md](speed.md#disk)). The remaining historical
workdir was 0.8 GB. CPU fp32 evaluation took about 15 minutes per text.

## Stacked expert layout (`--moe-layout stacked`)

The historical per-expert layout writes every expert projection as its own HI2
tensor: nine arrays per tensor, 28,307 tensors and a 3.5 MB safetensors header for
OLMoE-1B-7B. In that measurement the fork's loader concatenated 64 experts per
stage and rebased their offsets, taking about 95 s on a quiet host and 350-560 s
under load (dense fp16: 22 s). The **current default is stacked**, which writes
the experts of one layer and projection group as one set of arrays already in
the layout the fork builds in memory. It changes storage only. Every expert's
arrays and dequantized weights matched in the measured round trips below.
The spec container of a single tensor (spec 2.4) is unchanged.

* `hi2 quantize` writes the stacked layout by default (`--moe-layout per-expert` for the
  old one): the fork loader reads stacks, and model load drops from 72-94 s to 9-16 s on
  OLMoE with identical tensors and outputs. The stacked run writes per expert
  into `<workdir>/per-expert/<dst name>`, then restacks into `dst` and removes the
  staging copy.
* `hi2 restack CKPT OUT --layout stacked|per-expert` converts an existing checkpoint
  either way without re-encoding (byte-level data movement). It took 30 s and 4.9 GB RSS
  for `olmoe-r4`.
* `hi2 verify`, `hi2 export-dense`, `hi2 eval` and `checkpoint.load_dequantized` read
  both layouts, and so does `hi2.stack.Hi2Checkpoint`, which gives per-expert access
  by logical name.

### Names

A stack is `<pre>.<group>.hi2_<field>`. `<pre>` is the experts prefix of one layer
(`model.layers.L.mlp.experts`) and `<group>` is one of:

| group | parts, in row order | fork stage |
| --- | --- | --- |
| `gate_up` | `gate_proj`, `up_proj` | `w13` |
| `down` | `down_proj` | `w2` |
| `w13` (Mixtral names) | `w1`, `w3` | `w13` |
| `w2` (Mixtral names) | `w2` | `w2` |

A stack is recognised by its `hi2_part_off` array. A single HI2 tensor never has one.
`config.json` records the layout in `quantization_config`:
`"moe_layout": "stacked"` and `"moe_stacks": {"down": ["down_proj"], "gate_up":
["gate_proj", "up_proj"]}`, the groups used and their part order. `moe_layout` is
`per-expert` (and `moe_stacks` absent) after a restack back. Checkpoints written before
this change have no `moe_layout` key, which means per-expert.

### Arrays

`E` experts, `P` parts per expert (2 for `gate_up`, 1 for `down`), `K` input width
(hidden size for `gate_up`, intermediate size for `down`). Slice `j = e * P + p` is
part `p` of expert `e`, and `R` is the total row count. For OLMoE, `gate_up` has
E = 64, K = 2048 and R = 64 x 2 x 1024 = 131,072, and `down` has K = 1024 and
R = 64 x 2048 = 131,072.

| field | dtype | shape | content |
| --- | --- | --- | --- |
| `hi2_codes` | int32 | `[sum words]` | the slices' `codes`, concatenated in slice order |
| `hi2_code_off` | int32 | `[R + 1]` | word offset of each row in `hi2_codes`, rebased to the concatenation |
| `hi2_blk_params` | uint8 | `[2 * sum blocks]` | the slices' `(lo_u8, step_u8)` pairs, concatenated |
| `hi2_param_off` | int32 | `[R + 1]` | block offset of each row (byte offset / 2), rebased |
| `hi2_row_consts` | fp16 | `[R, 3]` | `lo_min, lo_scale, step_scale` per row, concatenated |
| `hi2_row_nbits` | uint8 | `[R]` | code width per row |
| `hi2_row_block` | uint8 | `[R]` | block size per row |
| `hi2_had` | uint8 | `[E, K/16]` | rotation flags of expert `e` (shared by its parts) |
| `hi2_col_scale` | fp16 | `[E, K]` | smoothing vector of expert `e`; present iff the experts are smoothed |
| `hi2_part_off` | int32 | `[E * P + 1]` | first row of slice `j`; slice `j` is rows `[part_off[j], part_off[j+1])` |

### Invariants a loader can rely on

1. The seven row arrays form one valid flat container of `R` rows (spec 2.4).
   `code_off[0] = param_off[0] = 0`. `code_off[r+1] - code_off[r] = K * nbits_r / 32`
   and `param_off[r+1] - param_off[r] = K / block_r`. `code_off[R]` is the length of
   `hi2_codes` and `2 * param_off[R]` is the length of `hi2_blk_params`. The fork's
   `validate()` passes on them if `K` is taken as `had.shape[1] * 16`, not
   `had.numel() * 16`. All offsets fit int32; the writer keeps a group per expert
   rather than overflow.
2. `E = had.shape[0] = col_scale.shape[0]` and `part_off.numel() = E * P + 1`. Every
   expert and part of a stack has the same `K`.
3. `part_off[0] = 0` and `part_off[E * P] = R`. The parts of a group have equal row
   counts in every model the tool handles: OLMoE gate/up 1024 rows, down 2048 rows. The
   table states the counts, so a loader should read it, not assume them.
4. Row order is expert-major, part-minor. For `gate_up`, expert `e`'s rows are
   `[part_off[2e], part_off[2e+2])`, gate first. This is the fork's in-memory `w13`
   (rows `[2Ne, 2N(e+1))`, gate first), and `down` is its `w2`
   (rows `[He, H(e+1))`).
5. Gate and up of an expert share one transform (decision 6), so `had` and
   `col_scale` have one row per expert, as the fork's `[E, K/16]` / `[E, K]` tables do.
   `col_scale` is all-or-none per stack. If it is absent, there is no smoothing (the
   fork's empty `col_scale`).
6. Slicing a stack gives the per-expert tensors back byte for byte. Rows
   `[part_off[j], part_off[j+1])` of the row arrays, `codes[code_off[r0]:code_off[r1]]`,
   `blk_params[2*param_off[r0]:2*param_off[r1]]`, the offsets minus their first entry,
   `had[e]` and `col_scale[e]` are exactly the per-expert arrays. This is the fork's
   `select_rows`.
7. The two stacks of a layer are in one shard: the shard that held most of that
   layer's expert tensors. Dense tensors keep their shard.
8. A group is stacked only if experts `0..E-1` all have every part, gate and up of
   each expert share `had` and `col_scale`, and smoothing is on for all experts of the
   stack or for none. Otherwise that group stays per expert in the same checkpoint and
   the tool logs it. A loader must accept both forms per layer. No tool-made checkpoint
   has such a group so far.

`hi2_report.json` keeps one entry per expert projection under `tensors`, under the
per-expert names (bpw, error, INT4 share). `summary.moe_layout` and
`summary.moe_stacks` (the stack count) record the layout. `model.safetensors.index.json`
lists the stack arrays, and its `metadata.total_size` is recomputed.

### What the fork loader has to do

* Accept `<pre>.gate_up.hi2_*` and `<pre>.down.hi2_*` (and `w13` / `w2`) as whole-stage
  tensors for the layer's `RoutedExperts`. The `hi2_part_off` array marks a stack.
* At TP = 1 with all experts local, use the arrays as `_rdna_hint_flex_w13` / `_w2`
  directly: `col_scale` is `[E, K]` or absent (pass the empty tensor), and `had` is
  `[E, K/16]`. `stack_experts` and the offset rebase are no longer needed.
* At TP > 1, take each expert's gate and up slices from `part_off`, then select rank
  rows `[rank*n, (rank+1)*n)` of each and the rank columns of `down` with the
  existing `select_rows` / `select_cols`. Concatenate in the same order. With expert
  parallelism, take the local experts' rows `[part_off[e0*P], part_off[e1*P])` and
  `had[e0:e1]`, `col_scale[e0:e1]`.
* Keep the per-expert path for checkpoints without stacks and for groups the writer
  left per expert.

### OLMoE-1B-7B, restacked (measured 2026-10-26)

`hi2 restack` turned the two checkpoints into
`olmoe-r4-stacked` and `olmoe-b3.0-stacked`, both without re-encoding.

| Checkpoint | Layout | Tensors | Safetensors headers | Index file | Shard files |
| --- | --- | --- | --- | --- | --- |
| `olmoe-r4` | per-expert | 28,307 | 3,486,304 B | 2,699,532 B | 3,430,904,600 B |
| `olmoe-r4-stacked` | stacked | **979** | **114,408 B** | **86,500 B** | 3,423,195,424 B |
| `olmoe-b3.0` | per-expert | 28,307 | 3,472,296 B | 2,699,532 B | 2,978,556,352 B |
| `olmoe-b3.0-stacked` | stacked | **979** | **114,176 B** | **86,500 B** | 2,970,860,952 B |

979 tensors = 32 stacks x 10 arrays + 64 attention projections x 9 arrays + 83 dense
tensors (embedding, head, norms, routers). The stacked shards are 7.7 MB smaller. Of
that, 3.4 MB is header. The rest is the transform that gate and up stored twice per
expert and now store once (`col_scale` 4 KB and `had` 128 B per expert). Each layer's
two stacks sit in one of the three shards (365 / 366 / 248 tensors). Each restack took
about 30 s and under 5 GB RSS.

`experiments/compare_layouts.py` checks every one of the 3,136 logical HI2 tensors
(3,072 expert projections and 64 attention projections). For each, all nine arrays are
equal in dtype, shape and bytes, and the float64 dequantization in the original
basis is `torch.equal`. The 83 dense tensors are identical too. The stacked
checkpoint restacked back to per-expert gives the original arrays byte for byte.

`hi2 eval` (fp32 on CPU, 8 x 2048 tokens) of the originals and the stacked copies,
run side by side with the same settings:

| Checkpoint | study ppl | study KLD | study top-1 | mix ppl | mix KLD | mix top-1 |
| --- | --- | --- | --- | --- | --- | --- |
| `olmoe-r4` | 6.706628799 | 0.016882055 | 0.944064485 | 4.107589722 | 0.047960081 | 0.902051783 |
| `olmoe-r4-stacked` | 6.706628799 | 0.016882055 | 0.944064485 | 4.107589722 | 0.047960081 | 0.902051783 |
| `olmoe-b3.0` | 6.772842407 | 0.031327991 | 0.917867611 | 4.287377834 | 0.090997412 | 0.870114802 |
| `olmoe-b3.0-stacked` | 6.772842407 | 0.031327991 | 0.917867611 | 4.287377834 | 0.090997412 | 0.870114802 |

All printed digits are identical, and the values match the references (6.707 / 4.108 and
6.773 / 4.287).

### Layout tests

`tests/test_stack.py` checks the stack invariants (dtypes, shapes, rebased offsets,
`part_off`, one flat container whose rows dequantize as the experts') and the
byte-exact split. It checks the rejection of unstackable groups (rotation or smoothing
that differs, mixed smoothing, different K), and a two-shard checkpoint (one layer
split across shards) converted per-expert -> stacked -> per-expert: equal arrays,
equal `load_dequantized`, one shard per layer, config/report/index. It also checks
that an unstackable group stays per expert. `tests/test_moe.py` quantizes the tiny
OLMoE with `--moe-layout stacked` and per expert from one phase 1. It checks equal
arrays for every tensor, equal `load_dequantized`, `verify` on the stacked checkpoint,
SHA-256-identical `export-dense` files, and `restack` both ways.

## Tests

`tests/test_moe.py` (CPU): a toy MoE with controlled routing checks that each expert's
statistics and Hessian equal those of its routed tokens, that a nearly unused expert is
topped up with exactly `own + (m - n) x layer average` and reported, that q/k/v and
gate/up share smoothing and flags (rotation `on` and `auto`), the allocation weights,
and the unfused forward against the fused one; a tiny OLMoE runs end to end and checks
the tensor names, the dense router, shared transforms in the written file, and the
export round trip through transformers.

## Implementation status and remaining limits

* Fused/batched GPTQ encoding is implemented, not future work: the OLMoE
  measurement fell from roughly 4,450 s to 192 s for 3,136 tensors on one V620,
  with byte-identical output. See [the implemented changes](speed.md#what-changed)
  and the OLMoE table under [speed measurements](speed.md#measurements).
* Native OLMoE serving was measured in the external fork, including both expert
  layouts; see [the recorded serving results](results.md#olmoe-1b-7b-unquantized-ppl-6634-docs-3924-mix-whole-file-1384-gb)
  and the layout measurements above. That is historical evidence, not a
  version-pinned installation recipe for arbitrary vLLM builds.
* Checkpoints storing fused 3-D expert weights on disk are not split by the
  writer. In-memory Transformers fusion adapted from per-expert 2-D source
  tensors is a different, implemented case. Check the actual files rather than
  inferring layout from a model-family name.
* Qwen2-MoE/Mixtral naming and adaptation paths exist, but are not equivalent to
  a measured end-to-end recipe for every revision. Shared experts are ordinary
  dense Linears in the implementation and remain unqualified by the OLMoE study.
* MoE QAD and MoE final-KL have no published end-to-end quality measurements here.
  The dense-model QAD results do not establish those combinations.
