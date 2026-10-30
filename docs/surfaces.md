# Actionable surfaces for a 27B HI2 run

What can be changed, what the tool already decides, what it leaves to the operator, and what
has been measured at 27B rather than inferred from the 0.8B/4B studies. Written 2026-10-29,
while the first `default` 27B run was in flight; its result is not in this page yet.

## What is measured

| measurement | command | what it says | 27B cost |
| --- | --- | --- | --- |
| corpus KLD | `hi2 eval --text docs\|mix` | distribution distance on one corpus, top-1, EAR | about 20 min (13 min of it is placing the model) |
| domain KLD | `bench/tfkld.py` | KL per task family, teacher-forced on the reference's own completions | hours (needs a TP=4 BF16 serve) |
| task suite | `bench/run.py` | paired accuracy per family, McNemar | 3.5 h (reduced to 766 items at 27B) |

The 27B noise floor is **exactly zero** on the suite (766/766 items identical across two
reference runs) and 2e-05 on the teacher-forced KLD, so one run per candidate is enough and
every difference is the checkpoint's. The layer-wise held-out nMSE is **not** a decision
metric: the `fast` checkpoint matched the previous run's per-layer nMSE within 2 % on 21 of 42
layers and within 5 % on all 42, and came out 33 % worse than PTQ on the mix KLD
(`qad.md`, "The verdict").

## Who decides what

| decision | tool rule | automatic? |
| --- | --- | --- |
| size / quality point | `--target-bpw`, `--max-error-ratio` (default 6 with QAD, 4 PTQ), or `--target-kld` (measures PTQ at several ratios) | the default ratio is automatic; the point is the operator's |
| per-row `(bits, block)` | allocator: Lagrangian on sensitivity-weighted error, subject to the bpw/ratio | yes |
| per-tensor sensitivity | probe: KL cost of each tensor at INT2/B32, median-normalised, floor 0.05 | yes |
| smoothing exponent | 0.5 (`--smooth-search` measured worse) | yes |
| rotation | on, paired with smoothing | yes |
| rounding | GPTQ (Hessian-aware), damp 0.01, act-order off | yes |
| code refinement | 2 sweeps without QAD, off with QAD | yes |
| encoder effort | normal (max measured worse on the 0.5B) | yes |
| candidates | `(2,4) x (16,32,64)`, INT8 only on request | yes |
| QAD recipe | preset: `fast` / `default` / `thorough` | the preset is the operator's |
| QAD data | 264 random windows from the whole calibration text, seed 0 | yes, given the file |
| QAD math / buffers | float32 math (the only one that works here); buffers auto (float32 when it fits) | yes |
| final logit-KL | 200 steps, bytes + norms, lr 0.02 / 5e-5, keep-best on held-out KL | yes |
| devices | every visible GPU, split by rows (PTQ) and sequences (QAD) | yes |
| calibration text | operator-supplied (`.txt` / `.jsonl`, several files) | **no** |
| QAD domain balance | proportional to the file; `sampling=balanced` per file | **no** |

So the tool decides everything *inside* the budget; the operator decides the budget and the
domain mix. The two "no" rows are the only knobs with no automatic rule, and the calibration
domain is the largest measured lever in the whole tool (`format-spec.md` 8.7).

## PTQ surfaces

### Calibration data

| surface | knob | evidence | at 27B |
| --- | --- | --- | --- |
| domain mix | `--calib a.txt,b.txt` (concatenated), or one interleaved file | 0.8B at 3.0 bpw: docs-only calibration leaves chat/tool text at KLD **0.650**; a docs/chat/tools mix gives **0.364** on it and 0.234 on docs | only the mixed file (about 70 % docs, 30 % chat/tools) has been run |
| token count | `--calib-samples` (default 128 x 2048 = 262k) | 0.8B: KLD 0.267 at 65k, 0.248 at 262k (-7 %), 0.250 at 524k (flat) | 262k |
| sequence length | `--seq-len` | not studied; 2048 matches the QAD windows | 2048 |

### Allocation

| surface | knob | evidence | at 27B |
| --- | --- | --- | --- |
| size point | `--target-bpw` vs `--max-error-ratio` vs `--target-kld` | ratio 11.0 = 3.012 bpw at 27B; ratio 6.2 = 3.499 at 4B; ratio 6 is the documented QAD default | target 3.0 (ratio 11.044) |
| sensitivity weights | `--sensitivity probe` (default with QAD) | -31 %, -16 %, -14 % of the perplexity excess at 3.0 bpw on 0.8B/0.5B/4B; nothing at 4.3 bpw | on (fp16 probe, required: fp32 is 102 GiB) |
| probe tokens | `--sens-seqs` (8), `--sens-len` (512) | a 4 x 256 probe is within noise and 3x faster (`speed.md`) | 8 x 512 |
| probe batching | `--sens-batch` | fewer forwards, logits round differently | 1 (exact) |
| INT8 rows | `--bits 2,4,8` | the allocator gives INT8 0.01-0.03 % of the weights, every metric unchanged | not run |
| forced INT8 | `--int8-force N` | worse at every N on the 0.8B | not run |
| expert knobs | `--expert-min-tokens`, `--expert-share-floor` | MoE only | N/A (dense) |

### Encoder

| surface | knob | evidence | at 27B |
| --- | --- | --- | --- |
| GPTQ rounding | `--rounding gptq` (default) | 0.8B at 3.0 bpw: ppl 1.92 (nearest) to 1.32; KLD 0.642 to 0.267 | on |
| damping | `--damp` (0.01) | not swept at model level | default |
| act-order | `--act-order` | not measured at model level | off |
| code refinement | `--refine-passes` (2 without QAD, 0 with) | -3 to -13 % KLD on 0.5B/0.8B; about 1.1x encode time; does not combine with QAD | off (QAD) |
| effort | `--effort fast\|normal\|max` | 0.5B end-to-end: fast 19.96, normal 20.75, max 21.47 (max worse) | normal |
| smoothing | `--smooth` (0.5) | 0.5 and 0.75 equivalent, 0.25 worse than none | 0.5 |
| per-tensor smoothing | `--smooth-search` | worse in 7 of 8 cells, KLD higher in all | off |
| rotation | `--rotation on\|auto\|off` | rotation alone hurts a calibrated encoder; with smoothing it is the best transform at every width | on |
| sequential propagation | `--sequential` | no gain at 3.0 bpw (-0.4 %), +0.2 % at 4.0 | off |
| block candidates | `--blocks`, `--candidates lean` | B=16 shows no end-to-end benefit on the 0.5B; lean drops INT4/B16 (0.1-0.8 % of the weights) | 16/32/64 |

### Format limits (not a running mode)

| limit | consequence |
| --- | --- |
| bits are 2, 4 or 8 | INT3 is not packable (32 is not divisible by 3); the INT3 rows in "not done" need a format change |
| blocks are 16, 32, 64 | 128 is not offered by the encoder or the decoder |
| rotation group is 128 | a wider group lowers error 1-2 % at the same block but is not in the format |
| `embed`, `lm_head`, `norm`, MTP are kept dense | 6.5 GiB of the 15.3 GiB 27B file is fp16; quantizing them (the GGUF does) would need loader support and would buy about 1.1 bpw for the body at the same file size |

## QAD surfaces

### Recipe

| mode | steps | budget | early stop | sequences | final KL | 0.8B time | 0.8B docs/mix ppl |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `fast` | <= 300, weighted | 0.5 | < 1 % / 50 steps (min 75) | 256 | 0 | 17 min | 1.085 / 1.330 |
| `default` | 300 | 1.0 | none | 256 | 200 | 41 min | 1.061 / 1.286 |
| `thorough` | 600 | 1.0 | none | 512, batch 8 | 200 | 141 min | 1.047 / 1.267 |

At 27B the `fast` layer phase spent 4333 steps on 42 layers (2.76x faster than the first run)
and the `default` layer phase is about 290 s a layer (5.3 h for 64). The early stop is inert at
27B with budget 0.5 (the only candidate is step 100, past the allotment), so `fast` == budget
0.5 in practice.

### Optimizer and initialisation

| surface | default | evidence |
| --- | --- | --- |
| `lr_w` | 0.01 in step units | 0.8B: 0.005 -> docs 0.088/mix 0.229; 0.01 -> 0.077/0.228; 0.02 -> 0.075/0.250; 3e-4 absolute -> 0.080/0.385 (adapts to docs, breaks mix) |
| `lr_p` | 0.05 levels/step | 0.8B: 0.02 was best in the final phase; the layer-wise value is not swept |
| `latent_units` | `step` | step units moved gain from docs to mix on the 4B (-9/-10 % mix, +11/+8 % docs) |
| `init` | `weights` | within seed noise on 0.8B; `ptq` with refined codes is much worse |
| `warmup` | 5 % | not swept |
| `batch` | 4 | 8 only in `thorough` |
| `samples` | 256 | 512 gives no gain on the 0.8B |
| `keep_best` | on | on held-out **layer nMSE**, which did not predict the end-to-end KLD |
| `train_norms` | off | within seed noise at 0.8B |
| `kl_layers` / `kl_weight` / `kl_tokens` | 0 / 1.0 / 256 | within seed noise at 0.8B |
| `layers` | all | a subset keeps PTQ on the rest; not tried at 27B |

### Data, math, execution

| surface | default | evidence |
| --- | --- | --- |
| `sampling` | `random` over the whole file | `balanced` gives equal windows per calibration file; `head` takes the first tokens |
| `seed` | 0 | 0.8B seed noise about 0.01 ppl / 0.01 mix KLD |
| `buffer_dtype` | `auto` (float32 at 27B: 41 GiB fits) | bf16 rounds 0.4 % of the activation, which is the noise floor of a 1 % early stop |
| `dtype` | `float32` | float16 is 1.43x faster but does not learn (nMSE -1.8 % against -45 %) and the full-attention layers do not run; bf16 is not an instruction on gfx1030 |
| `data_parallel` | on | same gradient up to summation order; 1.7x a step at 3 GPUs |
| `--qad-final-kl` | preset (200 for default/thorough, 0 for fast) | see below |

### Final logit-KL phase

Trains the per-block ladder bytes and the RMSNorm weights with the codes frozen, against
KL(teacher || student) on the whole network; keep-best on held-out KL.

| surface | default | evidence |
| --- | --- | --- |
| steps | 200 | 0.8B: 100 steps 0.0573/0.1165, 200 0.0556/0.1142; 4B 3.5 bpw: -5.6 % docs, -7.0 % mix |
| `final_lr_p` | 0.02 | 0.8B variants 1 and 2 gave 0.0522/0.1128 and 0.0527/0.1140 (vs 0.0556/0.1142) |
| `final_lr_norm` | 5e-5 | 4B value |
| `final_norms` / `final_bytes` | on / on | the only trainable groups |
| `final_tokens` | 0 (all positions) | reduces the loss chunk, not the model placement |
| `final_batch` | 2 | not swept |
| placement | teacher and student layer-wise | **the 27B teacher in float32 does not fit four V620s** (29 GiB a card against a 27 GiB guard); both 27B QAD runs skipped it. Since 2026-10-29 the layer-wise phase saves the teacher's post-final-norm head inputs (10.3 GiB) and the phase runs from them, needing only the student (about 12 GiB a device) |
| cost | | 4B 3.5 bpw: 200 steps 92.9 min (0.46 min/step). 27B 3.0 bpw: 2.45 min/step, about 8.2 h for 200 steps on four V620s (the student sharded, the dequantize and the checkpointed backward dominate); the phase logs every 50 steps |

## The default 27B run (finished 2026-10-29)

```bash
python -m hi2 quantize MODEL OUT --workdir OUT.work \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --calib calib_qad.txt --calib-samples 128 --seq-len 2048 \
  --sensitivity probe --sens-dtype float16 \
  --target-bpw 3.0 --qad-preset default --qad-final-kl 200
```

Result: `q27b-b3.0-qad-default`, 15.0 GiB, 3.012 bpw body, 67 % INT2 / 33 % INT4 rows, docs
KLD 0.0593 (PTQ 0.0975), mix KLD 0.1837 (PTQ 0.2171), top-1 90.8 % / 86.7 %, EAR 0.918 /
0.874. The layer-wise phase took 247 min (64 layers, the last at −4.3 % held-out nMSE); the
200-step final KL took 470.8 min and moved the held-out KL 0.4122 → 0.2427 (−41.1 %), teacher
pass 0 s from the saved head inputs. Raw: `experiments/results/kld_27b_2026-10-29/`.

Decisions the tool makes: the probe and the per-row allocation at 3.0 bpw (3.012 bpw body,
67 % INT2 / 33 % INT4 rows), smoothing 0.5 + rotation, GPTQ, no refinement (QAD), 300
steps/layer with 256 sequences, float32 buffers, four-GPU data-parallel steps, the 200-step
final KL from the saved head inputs.

Operator decisions: the size point (3.0 bpw) and the calibration text (the mixed file). Both
are deliberate: 3.0 bpw is the compact anchor and the same allocation as the PTQ and `fast`
checkpoints, so the result is a clean comparison at fixed size.

## After this run, in order

1. Score it on the teacher-forced tool-call/math KLD and the reduced suite. Its docs and mix
   KLD are measured and better than PTQ's; the domain texts are where the `fast` checkpoint
   lost, and the default's are not measured.
2. If a domain still loses to PTQ: the answer is a larger size (ratio 6 is about 3.5 bpw on
   the 27B; the tool-call budget on the 4B needed 4.2 bpw) or a domain-calibrated run.
3. A calibration-domain run: a mix-heavy or tool-heavy calibration (the largest evidenced
   lever, untested at 27B) scored on the tool-call text.
4. Final-KL tuning (steps, `final_lr_p`) now that the phase actually runs at 27B.
5. PTQ encoder refinements (`--refine-passes 4`, `--act-order`): cheap, PTQ-only, small.
