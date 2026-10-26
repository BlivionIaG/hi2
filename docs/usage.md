# Using hi2

This page tells you how to run the tool and what each option does. For the reasons
behind the defaults, read [results.md](results.md) and [design-notes.md](design-notes.md).

## Commands

| Command | What it does |
| --- | --- |
| `quantize SRC DST` | Convert a Hugging Face checkpoint to a rdna_hint_flex checkpoint. |
| `verify SRC DST` | Reload the quantized tensors and compare them with the source weights. |
| `eval SRC DST --text T --skip-tokens N` | Measure perplexity of both models, KL(original to quantized), and top-1 agreement on held-out text. |
| `export-dense SRC DST OUT` | Write the dequantized model as a plain fp16 Hugging Face directory. Use it to score the model in llama.cpp. |
| `restack CKPT OUT --layout stacked\|per-expert` | Convert the expert layout of an MoE checkpoint without re-encoding. |
| `size DST [--src SRC] [--by kind\|layer\|tensor] [--top N] [--json]` | Show where the bytes of a checkpoint go. See "Checking the size". |
| `cache list\|size\|prune` | Manage the persistent cache. |

`eval` dequantizes the weights and runs the original architecture. It checks the format
numerically. It is not a kernel test.

## Calibration text

Calibration needs `--calib-samples` sequences of `--seq-len` tokens. Use text from the
domain of your model. A concatenated multi-domain file works well: QAD draws random
windows from the whole file. `experiments/build_calib_mix.py` shows how the study text was
built.

Evaluate on text that is not in the calibration region. Use `--skip-tokens` to skip it.

Without `--calib`, the tool runs data-free: all column weights are 1, rounding is
nearest, and the model is not loaded. QAD is skipped with a warning.

## Where it runs

| Goal | Flags |
| --- | --- |
| Automatic (default) | `--devices auto`: every visible GPU, otherwise the CPU. Narrow the set with `HIP_VISIBLE_DEVICES`. |
| CPU and RAM only | `--devices cpu` |
| One GPU | `--devices cuda:0` |
| Several GPUs | `--devices cuda:0,cuda:1,...`. The tool splits calibration sequences across the devices and shards the rows of every tensor. The result equals the one-device result. |
| Model larger than VRAM | Automatic. The model stays in RAM, one layer at a time moves to the GPU, and hidden states stay in RAM. |

"Several GPUs" means row-sharded quantization with data-parallel calibration. It is not
a tensor-parallel forward pass.

## Choosing the quality target

Do not choose a bit rate. Choose how much error you accept.

* `--max-error-ratio R` (recommended: 6 with QAD, 4 with PTQ only; the default when you
  set neither option is 4): every row may have at
  most R times the error of its best option. The tool then spends the fewest bits that
  meet this. It reports the bit rate it reached.
* `--target-bpw B`: fix the bit rate. The tool spends the bits where they help most.
* `--max-error-ratio a,b,c`: write one checkpoint per ratio to `<out>-r<ratio>`. The tool
  shares phase 1, the probe, and the encode pass between them.
* `--bits 2,4,8`: allow INT8 rows. They are off by default because they did not help at
  3.0-3.5 bpw ([lowbit.md](lowbit.md)).
* `--bits` and `--blocks` also restrict the candidates. For example, `--bits 4 --blocks 64`
  gives a uniform INT4 reference.

## Rounding and transforms

| Option | Default | Notes |
| --- | --- | --- |
| `--rounding gptq` | on with `--calib` | Hessian-aware rounding. On Qwen3.5-0.8B at 3.0 bpw it takes the perplexity ratio from 1.92 to 1.32. It changes the chosen codes only, never the format. |
| `--rotation on --smooth 0.5` | on | Smooth the activation channels, then rotate. This was the best transform at every bit width. Without `--calib`, smoothing is skipped. |
| `--sequential` | off | Encode on the quantized prefix. It gave no measurable gain. |
| `--act-order` | off | GPTQ column order by activation size. Not adopted. |
| `--effort fast\|normal\|max` | normal | More effort did not improve quality in our runs. |

The smoothing vector costs `16/R` bits per weight per tensor. The allocation counts it.

On gfx10 and gfx11 the GPTQ column loop runs as a fused HIP kernel. It gives the same
result as the torch loop. `HI2_GPTQ_BACKEND=torch` forces the torch loop. The kernel
builds on first use into `~/.cache/hi2/kernels` (`HI2_KERNEL_DIR`).

## Quantization-aware distillation (QAD)

With calibration text, `quantize` distills every decoder layer after PTQ. QAD is on by
default. The checkpoint stays a normal v3.2 checkpoint.

```bash
hi2 quantize <model> <out> --calib calib.txt --max-error-ratio 6     # PTQ + QAD
hi2 quantize <model> <out> --calib calib.txt --dry-run               # plan only
hi2 quantize <model> <out> --calib calib.txt --no-qad                # PTQ only
```

| Option | Meaning |
| --- | --- |
| `--qad-preset fast\|default\|thorough` | Training length. |
| `--qad-budget F` | Fraction of the preset's steps. The tool spends steps where layers still gain, with per-layer early stopping. |
| `--qad-samples N`, `--qad-layers A-B` | Training sequences, layer range. |
| `--qad-final-kl STEPS` | Add a final logit-KL phase. See [lowbit.md](lowbit.md). |
| `--qad-config file.json` | All settings in a file. |

Runtimes on one V620 (QAD part, `default` and `fast`): Qwen3.5-0.8B 41 and 17 minutes;
Qwen3.5-4B about 3.1 hours and 1.3 hours. PTQ adds 10-50 minutes. The run prints its
expected time first. It is resumable per layer: run the same command again. Details are in
[qad.md](qad.md).

## Mixture-of-experts models

The tool splits fused expert tensors, shares the gate and up smoothing, and counts routed
tokens per expert. The default layout is `stacked`: one tensor group per layer. It loads
about 9 times faster in the vLLM fork. Use `--moe-layout per-expert` for one tensor per
expert. Details: [moe.md](moe.md).

## Cache and work directory

The tool caches statistics, Hessians, error tables, and probe results under `--cache-dir`
(default `$HI2_CACHE_DIR` or `~/.cache/hi2`). The key is the model, the calibration token
ids, and the settings each product uses. A second run of the same model and calibration
skips the calibration pass. The log shows `cache: ... hit|miss`.

```bash
hi2 cache list
hi2 cache size
hi2 cache prune [--older-than DAYS] [--keep-gb N] [--digest D] [--dry-run]
```

`workdir` (default `<out>.work`) keeps the shard markers and `cache_ref.json`. A run that
stops can continue because the finished output shards stay. `--no-cache` keeps all tables
in the work directory instead.

When the code that produces a cache entry changes, raise `STATS_VERSION`,
`TABLES_VERSION`, or `PROBE_VERSION` in `hi2/cache.py`. See [speed.md](speed.md).

## Checking the size

`hi2 size` reads only the safetensors headers and the small per-row precision arrays. It
works on both expert layouts and takes a second or two.

```bash
hi2 size out/q4b --src Qwen3.5-4B --top 10
```

The report gives:

* the quantized linears: tensor count, weights, bytes, and the measured bpw;
* the bpw split into codes, block parameters, row constants, smoothing vector, and row
  index (offsets, precisions, block sizes, rotation flags);
* the share of weights at INT2, INT4, and INT8;
* the dense tensors that stay in fp16 (embeddings, `lm_head`, norms, routers);
* with `--src`, the compression against the source model's tensors;
* a table by tensor family (`--by kind`), by layer, or by tensor.

The measured bpw counts every byte the loader reads. It is larger than the allocation
target because it includes the row index. Use it to compare sizes with other formats on
the same footing: add the dense tensors that the other format keeps in fp16, or subtract
them from both.

## Examples

```bash
# 4B model, one GPU, quality target 6, held-out check
hi2 quantize Qwen3.5-4B out/q4b --calib calib.txt --max-error-ratio 6 --devices cuda:0
hi2 eval     Qwen3.5-4B out/q4b --text heldout.txt --skip-tokens 800000 --device cuda:0

# Dense fp16 copy for llama.cpp or another engine
hi2 export-dense Qwen3.5-4B out/q4b out/q4b-dense
```
