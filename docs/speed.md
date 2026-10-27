# HI2 quantizer speed

The HI2 quantizer (`hi2`, spec v3.2 section 8) was made faster without changing
what it writes. Every change below either computes the same values in the same order
(bit-identical output, checked by a test) or is opt-in. This note records where the
time went, what changed, the cache and how to invalidate it, and the measurements.

## Where the time went (profile of real Qwen3.5-4B tensors, one V620)

`experiments/profile_encode.py` times the old per-tensor GPTQ encode of layers 1 and 3
of Qwen3.5-4B (the allocation of a 3.0 bpw run) stage by stage, with a device sync
around every stage:

| Tensor (rows x K) | transform W | Sigma transform | factorise | ladder fit | column loop | block GEMM | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mlp.down_proj (2560 x 9216) | 0.07 | 0.13 | 0.52 | 0.25 | **12.45** | 0.22 | 13.65 s |
| mlp.gate_proj (9216 x 2560) | 0.05 | 0.01 | 0.02 | 0.15 | **2.82** | 0.03 | 3.09 s |
| linear_attn.in_proj_qkv (8192 x 2560) | 0.16 | 0.01 | 0.02 | 0.22 | **10.14** | 0.13 | 10.69 s |
| linear_attn.in_proj_z (4096 x 2560) | 0.04 | 0.01 | 0.02 | 0.10 | **4.19** | 0.03 | 4.39 s |
| self_attn.o_proj (2560 x 4096) | 0.03 | 0.02 | 0.07 | 0.11 | **6.61** | 0.05 | 6.89 s |
| linear_attn.out_proj (2560 x 4096) | 0.07 | 0.02 | 0.07 | 0.14 | **6.77** | 0.05 | 7.12 s |

The column loop is 85-95 % of the encode. It rounds one column at a time with about
20 small torch ops, once per candidate group and row chunk: 52,775 kernel launches for
a 32-row tensor, about 200k per candidate group at K = 9216. It is launch bound: the
time follows the number of candidate groups times K, not the row count. Three smaller
costs showed up once it was gone: packing the codes on the CPU (`format.assemble`, about
1 s per large tensor on the loaded box), the allocator (a strided dim-0 `argmin` per
tensor per bisection step: 3.3 s for 31 tensors, minutes for OLMoE's 3136), and the
Hessians: written to the workdir every run (21 GB for OLMoE, 2 GB for the 0.8B model)
and deleted afterwards.

## What changed

| Change | Where | Result | Exact? |
| --- | --- | --- | --- |
| Fused column loop: one HIP kernel launch per 128-column block for all rows of a batch (one thread per row, the row segment in LDS, the `u` row read from L2; rows may come from different tensors, each with its own `u`, column-to-block map and code count). The cross-block update stays a torch GEMM with the reference's shapes. | `hi2/gptq_kernel.py`, `hessian.encode_jobs_gptq` | 52,775 to 1,568 launches for a small tensor; the kernel itself takes about 1 ms per block launch | yes: same ops in the same order, each product and sum rounded separately (`-ffp-contract=off`, `__f*_rn`, IEEE division, `rintf`, `__float2half_rn`) |
| Batched encode: all candidate groups of many tensors (an OLMoE layer's 64 experts) in one call; the Hessian transform and the Cholesky factor computed once per distinct input (q/k/v, gate/up share one) and Sigma loaded once | `calibrate.encode_many`, `checkpoint.write_checkpoints` | | yes: every row chunk is computed as before |
| Next batch of tensors (weights, statistics, Sigma) read from disk in a background thread while the current one encodes | `checkpoint.write_checkpoints` | | yes |
| Code packing and the flat-buffer scatter on the GPU | `format.assemble` | 1-2 s CPU per large tensor removed | yes: integer ops |
| Allocator: all tables side by side, an argmin as C-1 vector compares | `allocate._Flat` | 10x (3.3 to 0.3 s for 31 tensors) | yes: same per-tensor sums, same picks (test with the old code inlined) |
| Persistent cache of Sigma, block moments, tables and the probe | `hi2/cache.py` | phase 1 and the probe are skipped on a hit | yes: the stored values are the computed ones |
| Hidden states between layers stay on the GPU when they fit (`HI2_RESIDENT_FRAC`, default 0.3 of device memory) | `layerwise._resident` | no host round trip per layer | yes |
| Probe: the decoder layers before a unit's first layer are replayed from the reference pass instead of recomputed; the perturbed tensor is dequantized on the GPU (every step exact in float64 for fp16 data in practice, GPU test) and the reference log-probabilities stay on the GPU | `sensitivity._Replay`, `format.dequantize(t, device)` | probe 1.7-2.5x faster | yes: same sensitivities bit for bit (test, and identical `sensitivity.json` on 0.8B and 4B) |
| `--sens-batch N`: N probe sequences per forward | `sensitivity` | fewer, larger forwards | no: logits round differently; opt-in |
| `--max-error-ratio a,b,c`: several allocations from one phase 1 and probe, written to `<dst>-r<ratio>` in one encode pass (each Sigma loaded and factorised once for all) | `pipeline.run`, `checkpoint.write_checkpoints` | | yes: each output equals its single-ratio run (test) |
| `--candidates lean`: drops INT4/B16 | `__main__` | one candidate less to tabulate | no: changes the candidate set; opt-in |

The torch loop is kept verbatim as `hessian.encode_rows_gptq_reference` (and the
per-tensor encode as `calibrate.encode_with_picks_reference`). It is the fallback on
CPU, CUDA, non-RDNA HIP and when the kernel does not build, and the reference of the
tests. `HI2_GPTQ_BACKEND=torch` forces it, `=kernel` makes a missing kernel an error.
The kernel is compiled on first use with `torch.utils.cpp_extension` (hipcc, about 50 s,
`--offload-arch=gfx1030`) into `$HI2_KERNEL_DIR` (default `~/.cache/hi2/kernels`).

## The cache

`--cache-dir` (default `$HI2_CACHE_DIR`, else `~/.cache/hi2`), `--no-cache` for the old workdir-only layout. Three
kinds of entries, each `<root>/<kind>/<digest>/` with its full key in `key.json`:

| Kind | Holds | Key (digest of) |
| --- | --- | --- |
| `stats` | `sigma/<leader>.pt`: `Sigma = E[x x^T]` per distinct Linear input (q/k/v, gate/up share one), the upper triangle in fp32 when it is exactly symmetric (every Sigma computed on the GPU was), else the full matrix; `colstats/<leader>.pt`: the 16x16 block moments; `layers.json`: which Linears share an input, the token counts; `expert_tokens.json` for MoE | model fingerprint (`config.json` bytes, names and sizes of the weight files and the index), a hash of the calibration token ids (covers the text, the tokenizer, samples and sequence length), compute dtype, device kind and count, `--include` / `--exclude` / `--limit-layers`, `--expert-min-tokens`, `STATS_VERSION` |
| `tables` | the phase-1 tables | the stats digest, the candidates, `--rotation`, `--smooth`, `--effort`, `TABLES_VERSION` |
| `probe` | `sensitivity.json` | the tables digest, a hash of the probe token ids (`--sens-seqs`, `--sens-len`), `--sens-floor`, `--sens-dtype`, `--sens-batch`, `--expert-share-floor`, the reference candidate, `--effort`, device count, `PROBE_VERSION` |

Sigma is taken in the original basis from the unquantized model, so it depends only on
the model, the calibration tokens and how they are run, never on the target bpw, the
error ratio, rounding, smoothing, rotation, candidates or QAD: one stats entry serves
every allocation target, ratio sweep, smoothing exponent or candidate set of a model
and calibration. A change of the phase-1 settings re-tabulates from the cached
statistics without running the model (`layerwise.tables_from_stats`, same tables bit for
bit). The workdir keeps only `cache_ref.json` (which entries it used) plus the small
`sensitivity.json`; `cache.StatsView` reads either layout.

Entries are built in `<digest>.tmp-<pid>` and renamed when complete, under a lock
(`<digest>.lock`), so parallel runs of a sweep wait for one computation instead of
repeating it, and an interrupted build is never used. The log says
`cache: statistics hit|miss`, `tables hit|miss`, `sensitivity probe hit|miss`. Before a
statistics build the tool warns when the free space under the cache root is below the
estimated entry size (`cache.estimate_stats_bytes`).

Invalidation: entries are content-keyed, so a different model, text, tokenizer or
setting simply misses. When the code that produces an entry changes its output, bump
`STATS_VERSION`, `TABLES_VERSION` or `PROBE_VERSION` in `hi2/cache.py`; old entries are
then never hit again. To remove entries:

```bash
python -m hi2 cache list                    # entries, sizes, last use, model, calibration
python -m hi2 cache size
python -m hi2 cache prune                   # interrupted builds only
python -m hi2 cache prune --older-than 14   # unused for 14 days
python -m hi2 cache prune --keep-gb 100     # least recently used first, down to 100 GiB
python -m hi2 cache prune --digest <digest> # one entry
```

`--cache-fp16-above K` stores Sigma in fp16 for inputs wider than K (part of the key).
It halves the disk for the large inputs but is lossy; it is off by default (see the
measurements below).

## With QAD (merged with the QAD branch)

QAD (`docs/qad.md`) is on by default when calibration text is given;
`--no-qad` gives the PTQ flow measured below. Its PTQ initialisation (the encode each
layer starts from) goes through the same machinery: statistics and Hessians are read with
`cache.StatsView` (from the cache, or from the workdir with `--no-cache`), each Sigma is
loaded once per shared input, and all Linears of a layer are encoded in one
`encode_many` call. The values each code was rounded from (`comp`, the latent
initialisation) are produced by the batched engine and the fused kernel as well,
bit-identical to the reference loop. A cached QAD run writes the same checkpoint as the
workdir layout (test).

`--max-error-ratio a,b,c` with QAD on runs one QAD pass per output, in sequence: the run
warns up front that the time is that many QAD passes, keeps the per-layer artifacts of
each output in `<workdir>/qad-r<ratio>` (resumable) and its report in
`qad_report-r<ratio>.json`; each output equals the single-ratio run. For a quick sweep use
`--no-qad` (or sweep with PTQ and run QAD only on the chosen ratio).

Checked after the merge on Qwen3.5-0.8B (`sw08-r6` settings): `--no-qad` is still
byte-identical to `sw08-r6` (cache hit, KLD 0.13820 / 0.19875). A short QAD run
(`--qad-preset fast --qad-layers 0-1 --qad-samples 64`) with the merged code and the
same run with the pre-merge QAD branch (`4ee534b8d8`, no cache) give identical per-layer
artifacts for all 24 layers (PTQ initialisations and the two trained layers) and a
byte-identical checkpoint; `hi2 verify` and `hi2 eval` accept it (KLD 0.13566 / 0.19900).
Time: 188 s with the warm cache against 867 s pre-merge (of which QAD about 75 s in
both). The QAD time is reported as `runtime_s.qad`.

## Measurements

All runs on one V620 (gfx1030) of the shared box, which was heavily loaded by other
jobs at times (load average up to 330 on 96 threads), so timings carry some noise; the
old code was re-run the same afternoon for a like-for-like baseline. Phase times are
from `hi2_report.json` (`runtime_s`, new) or from the run's timestamps and file
modification times (old).

The runs below are PTQ (they predate the QAD merge, which made QAD the default: add
`--no-qad` to reproduce them).

### Qwen3.5-0.8B (settings of `sw08-r6`: `calib_mix.txt`, 128 x 2048, `--sensitivity probe --max-error-ratio 6`)

| Run | phase 1 | probe | allocate | encode | total (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| old code, reference `sw08-r6` | 117 | ~260 | (in encode) | ~382 | 754 |
| old code, re-run today (`o08-r6`) | 112 | ~250 | (in encode) | ~321 | 679 |
| new, empty cache | 103 | 146 | 0.9 | 16.7 | **266** |
| new, cache hit (statistics, tables, probe) | 5.1 | 2.2 | 0.9 | 20.3 | **28.5** |
| new, `--max-error-ratio 2,3,4,6,8,12` (six outputs, cache hit) | 4.3 | 1.8 | 4.4 | 73.9 | **84.4** |

The old sweep of the same six ratios took six runs of 490 to 760 s each (one shared
calibration copied into each workdir), about 3,500 GPU-seconds; the multi-output run
takes 84 s once the cache is warm (350 s from cold).

Equality: the new outputs (empty cache, cache hit, a second empty cache) are
byte-identical to `sw08-r6` (`model.safetensors`, `config.json`); the six multi-output
checkpoints are byte-identical to the old `sw08-r2` ... `sw08-r12`; `sensitivity.json` is
identical to the old one; KLD 0.13820 (docs, skip 800k) and 0.19875 (`eval_mix`) as before.

### Qwen3.5-4B (settings of `q4b-b3.0-sens`: `study_text_all.txt`, 128 x 2048, `--sensitivity probe --target-bpw 3.0`)

| Run | phase 1 | probe | allocate | encode | total (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| old code, reference `q4b-b3.0-sens` (2026-10-25) | ~582 | ~1468 | (in encode) | ~946 | 2999 |
| old code, re-run today (`o4b-b3.0`) | 611 | ~1495 | (in encode) | ~971 | 3073 |
| new, empty cache | 559 | 825 | 3.4 | 97 | **1485** |

The 2.07x is the probe (1.8x, exact levers only) and the encode (10x); phase 1 is the
calibration forward plus the Hessian GEMMs either way. With the cache warm, another
allocation of this model costs the encode, about 100 s, instead of 50 minutes.

Equality: the new checkpoint is byte-identical to the old code's re-run (both shards),
KLD 0.13132 (docs) and 0.39430 (`eval_mix`) in both. The 2026-10-25 reference
`q4b-b3.0-sens` itself is not reproduced by today's base branch either: its tables and
`sensitivity.json` are identical to today's, but the allocation code changed since
(`allocation_bpw` 2.9999991 then, 2.9999999 now from the same inputs), so 128 tensors
have other row picks; today's old-code run is the baseline.

### OLMoE-1B-7B-0924 (settings of `olmoe-b3.0`: `calib_mix.txt`, 178 x 2048, `--sensitivity probe --sens-dtype float16 --target-bpw 3.0`)

| Run | phase 1 | probe | allocate | encode | total (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| old code (first run `olmoe-r4`, then `olmoe-b3.0` reusing phase 1 and the probe) | ~930 | ~603 | (in encode) | 4436-4470 | 4477 (with reuse) |
| new, empty cache | 535 | 245 | 22.4 | **192** | **994** |

Encode: 3136 tensors (3072 experts) in 192 s, **16.3 tensors/s** against 0.70; the
output is byte-identical to the old `olmoe-b3.0` (all three shards). With the cache warm
a further allocation of this model costs about 230 s (allocate and encode).

### Encode per tensor (Qwen3.5-4B layer 3, 3.0 bpw allocation)

| Tensor | old (stage-timed) | new | launches (32-row tensor) |
| --- | ---: | ---: | ---: |
| mlp.down_proj 2560 x 9216 | 13.65 s | 1.71 s | |
| mlp.gate_proj 9216 x 2560 | 3.09 s | 0.44 s | |
| linear_attn.in_proj_a 32 x 2560 | | | 52,775 old, 1,568 new |

What is left of the new encode of a large tensor is the Hessian transform and the
Cholesky factor and inverse (about 0.5 s at K = 9216), the host-to-device copy of Sigma
and the per-candidate ladder fits. `experiments/profile_encode.py` checks every profiled
tensor against the old code: all bit-identical.

### Disk

| Model | Hessians before (per workdir, deleted by the scripts) | cache entry now (kept, shared) |
| --- | ---: | ---: |
| Qwen3.5-0.8B | 1.8 GB | 0.9 GB |
| Qwen3.5-4B | about 14 GB | 7.0 GB |
| OLMoE-1B-7B | 21 GB | 11.8 GB |

Tables add 11 MB / 29 MB / 127 MB, a probe result a few hundred KB.

### Studies (Qwen3.5-0.8B, ratio 6, cache hit for the statistics)

Probe size and batching. A smaller probe changes the allocation a little (the bpw
moves with it at a fixed ratio), so the KLD is compared after correcting for the bpw
with the slope of the old ratio sweep around 3.4 bpw (-0.18 docs, -0.29 mix KLD per bpw):

| Probe | probe s | bpw | KLD docs / mix | at 3.4347 bpw |
| --- | ---: | ---: | --- | --- |
| 8 x 512 (default) | 146 | 3.4347 | 0.1382 / 0.1988 | 0.1382 / 0.1988 |
| 4 x 256 | 50 | 3.4085 | 0.1421 / 0.1959 | 0.1374 / 0.1884 |
| 2 x 256 | 76 (shared GPU) | 3.4122 | 0.1416 / 0.2048 | 0.1376 / 0.1984 |
| 8 x 512, `--sens-batch 8` | 163 | 3.4347 | 0.1382 / 0.1988 | (sensitivities within 0.4 %) |

A 4 x 256 probe (a quarter of the tokens) is as good within noise and three times
faster; it is the recommended setting for sweeps (`--sens-seqs 4 --sens-len 256`).
The default stays 8 x 512 so that default runs reproduce earlier checkpoints bit for
bit. `--sens-batch` does not pay off on this model (the 248k-vocabulary log-softmax of a
batch dominates) and is left at 1.

Candidates. Over 34 earlier checkpoints (0.8B, 4B, OLMoE; `experiments/candidate_usage.py`)
INT4/B16 holds 0.1-0.8 % of the weights at 3-3.5 bpw and up to 5-8 % at 4.3 bpw or ratio
2; no candidate is unused. `--candidates lean` (no INT4/B16) gave no measurable speedup
(the tables of one candidate less, about 4 s here) and, at a fixed ratio, a different
operating point (3.13 bpw, KLD 0.197 / 0.303, on the old sweep's curve), because the
ratio is relative to the best achievable error of the candidate set. Not recommended.

fp16 Sigma (`--cache-fp16-above 2048`, here only the down_proj input, K = 3584): the
entry shrinks from 900 to 606 MB, the allocation is unchanged (every row width
identical; the tables do not use Sigma), the codes of the 24 down_proj tensors change
(4.5 % of all 32-bit code words of the model differ) and the KLD is
0.13805 / 0.19731 against 0.13820 / 0.19875, within noise. It stays opt-in because it is
not bit-exact.

Batched phase-1 tables (one ladder fit per candidate over all experts of a layer)
gave identical tables but were 1.7-1.9x slower than tensor by tensor (64 OLMoE experts:
8.3 against 4.3 s), the fit being bandwidth bound; it was not kept.

## What remains

* The probe is now the largest phase of a cold run (146 s / 825 s / 245 s): about 60 ms
  of host time per forward for the hybrid Qwen3.5 layers (Triton autotuner and launch
  overhead in the linear-attention kernels), eight forwards per unit. The exact levers
  are used (prefix replay, GPU dequantization); the remaining one is a smaller probe
  (above).
* Phase 1 of a cold run is the calibration forward and the Hessian GEMMs
  (`x^T x` per sequence, 45 TFLOP per 4B down_proj input over 128 x 2048 tokens),
  unchanged and exact; a symmetric rank-k update would halve it but changes the
  summation order.
* The Cholesky factor and inverse at K = 9216 (about 0.5 s per input).

## Reproduce

The runs above are plain `hi2 quantize` invocations on the 0.8B and 4B models and an OLMoE model, cold and warm cache, with the options named in the tables. `profile_encode.py`,
`profile_new.py` and `profile_probe.py` are the profilers. Raw numbers:
`experiments/results/hi2_speed_runs_2026-10-26.json`,
`hi2_speed_profile_4b_2026-10-26.json` and `hi2_candidate_usage_2026-10-26.json`.

## Tests

```bash
pytest tests   # CPU; the kernel tests run on a gfx10/gfx11 GPU
```

| Test | Checks |
| --- | --- |
| `test_gptq_fast.py` | batched torch engine and fused kernel equal the reference loop bit for bit: INT2/INT4/INT8, blocks 16/32/64, act-order, K not a multiple of 128, row chunking, several jobs and batches of mixed widths |
| `test_cache.py` | cached runs byte-identical to uncached ones (gptq and nearest), hits on rerun without running the model, new phase-1 settings from cached statistics, probe cached, multi-ratio outputs equal single runs, `encode_many` equals the per-tensor reference (shared inputs, two devices, act-order), triangle storage exact, prune/list, probe prefix replay bit-identical |
| `test_allocate_fast.py` | the vectorised allocator gives the old picks, bpw, multiplier and ratio |
