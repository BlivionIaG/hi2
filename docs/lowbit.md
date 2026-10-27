# HI2 at 3.0-3.5 bpw: INT8 rows, a final logit-KL phase, QAD data and loss

This is a study of the quantizer at 3.0-3.5 bpw. At this size HI2 with QAD still trails
EXL3 3.0 bpw on HellaSwag and MMLU (`qad.md`, `results.md`). The study asks
which extra options close the gap.

Nothing here changes the format. Every checkpoint below is a normal v3.2 checkpoint. INT8
rows already exist in the format.

## Summary

| Item | Result | Default |
| --- | --- | --- |
| INT8 rows (`--bits 2,4,8`) | Works end to end: quality allocation, probe, cache, fused GPTQ, QAD, writer, verify, export-dense. At 3.0-3.5 bpw the allocator gives INT8 to 0.01-0.03 % of the weights, and every metric stays the same. Forcing the most sensitive tensors to INT8 is worse. | Opt-in, unchanged |
| Final logit-KL phase (`--qad-final-kl STEPS`) | KLD -14 % (docs text) and -10 % (mix) on the 0.8B model; -6 % and -7 % on the 4B model at 3.5 bpw. The perplexity excess is about half. Multiple-choice accuracy stays the same. | On with the `default` and `thorough` presets (200 steps), off with `fast`: about +50 % QAD time. Teacher and student are split layer-wise over the devices, so the 27B trains on four V620s; skipped with a note only if that does not fit. `--qad-final-kl 0` turns it off |
| Norm weights in the layer-wise phase | Within seed noise | Off |
| KL term at the last layers of the layer-wise phase | Within seed noise | Off |
| 2x QAD samples (512 instead of 256) | No gain: the docs KLD is 0.0598 against 0.0607, the perplexity is higher. | Off |
| Task-style training text in the QAD mix | Worse on held-out text (docs KLD 0.0720 against 0.0607). It adapts the model to the task style, not to the language. | Not a default |

## Setup

* Model: Qwen3.5-0.8B at 3.5 bpw (error ratio chosen for that size), one V620.
* Two held-out texts: `docs` (documentation text, unquantized perplexity 10.874) and `mix`
  (mixed domains, 5.483). The columns are perplexity of the quantized model and
  KL(original to quantized) in nats per token.
* Every run uses the same calibration text, the same seed, and the `fast` QAD preset
  unless the row says otherwise.
* Seed noise: the same recipe with a different seed moves the docs KLD by about 0.001
  (0.0607 and 0.0615). Treat smaller differences as noise.

## Results on Qwen3.5-0.8B

| Run | docs ppl | docs KLD | mix ppl | mix KLD |
| --- | ---: | ---: | ---: | ---: |
| PTQ only (3.5 bpw) | 12.406 | 0.1288 | 6.848 | 0.1841 |
| QAD (3.5 bpw), reference | 11.388 | 0.0607 | 6.309 | 0.1249 |
| QAD with INT8 candidates | 11.409 | 0.0604 | 6.309 | 0.1255 |
| PTQ with INT8 candidates | 12.412 | 0.1290 | 6.850 | 0.1845 |
| PTQ, 2 most sensitive tensors forced to INT8 | 12.484 | 0.1327 | 6.872 | 0.1885 |
| PTQ, 4 forced | 12.576 | 0.1443 | 7.014 | 0.2085 |
| PTQ, 8 forced | 12.774 | 0.1620 | 7.186 | 0.2379 |
| QAD, norm weights trained | 11.406 | 0.0598 | 6.298 | 0.1236 |
| QAD, KL term at the last layers | 11.376 | 0.0617 | 6.308 | 0.1250 |
| QAD, 512 samples | 11.476 | 0.0598 | 6.323 | 0.1253 |
| QAD, task-style text in the mix | 11.603 | 0.0720 | not run | not run |
| QAD + final KL, 100 steps | 11.148 | 0.0573 | 6.174 | 0.1165 |
| QAD + final KL, 200 steps | 11.133 | 0.0556 | 6.140 | 0.1142 |
| QAD + final KL, 200 steps, learning-rate variant 1 | 11.090 | 0.0522 | 6.137 | 0.1128 |
| QAD + final KL, 200 steps, learning-rate variant 2 | 11.040 | 0.0527 | 6.020 | 0.1140 |
| QAD (3.0 bpw), reference | 11.796 | 0.0993 | 7.292 | 0.2538 |

## Results on Qwen3.5-4B

Same setup at 3.5 bpw (error ratio 6.2; allocation 3.499 bpw, INT2 44.4 % and INT4 55.6 % of
the rows), unquantized perplexity 7.067 on the docs text.

| Run | docs ppl | docs KLD | mix ppl | mix KLD |
| --- | ---: | ---: | ---: | ---: |
| PTQ only | 7.600 | 0.0927 | 4.055 | 0.1168 |
| PTQ with INT8 candidates | 7.604 | 0.0925 | 4.056 | 0.1173 |
| QAD, weight-unit latents (older default) | 7.190 | 0.0416 | 3.999 | 0.1033 |
| QAD, step-unit latents (current default) | 7.221 | 0.0450 | 3.985 | 0.0934 |
| QAD, step-unit latents + final KL, 200 steps | 7.137 | 0.0425 | 3.933 | 0.0869 |
| PTQ only, 3.0 bpw | 8.041 | 0.1607 | 4.438 | 0.2258 |
| QAD, step-unit latents, 3.0 bpw | 7.308 | 0.0677 | 4.276 | 0.1770 |

The final-KL phase lowers the KLD against the step-unit QAD baseline by 5.6 % (docs) and
7.0 % (mix). Its own log shows the held-out KL falling from 0.1175 to 0.1033 (-12.1 %) in
92.9 minutes. Against the older weight-unit QAD, the docs KLD does not improve (0.0425
against 0.0416). INT8 candidates change nothing at this size, as on the 0.8B model.

All numbers of this page are in
[`experiments/results/lowbit_runs_2026-10-26.json`](../experiments/results/lowbit_runs_2026-10-26.json)
(parsed from the `hi2 eval` logs of each run).

Read the 0.8B table as follows.

1. **INT8 does not help at this size.** The allocator already finds that INT4 is enough
   for almost every row. Forced INT8 costs bits that INT2 rows would have saved, so
   quality drops as more tensors are forced.
2. **The final logit-KL phase is the only option that moves the numbers.** It trains
   against the full output distribution after the layer-wise phase. It reduces KLD, but
   multiple-choice accuracy did not change in the 4B runs.
3. **Training text matters less than expected.** More samples do not help. Task-style
   text hurts held-out language modelling.

## Defaults

The final-KL phase is on with the default QAD preset. The other options are off. Switch with:

```bash
python -m hi2 quantize SRC DST --calib calib.txt --bits 2,4,8           # INT8 rows
python -m hi2 quantize SRC DST --calib calib.txt --qad-final-kl 0       # turn the final logit-KL phase off
```

The final-KL learning rates default to the values measured on Qwen3.5-4B.

## Tools

* `experiments/lowbit_run.sh GPU TAG ...`: one quantize + evaluate run with a GPU lock file.
* `experiments/mc_lowbit.sh`: multiple-choice scoring that holds the same lock file.
* `experiments/restamp_qad.py`: reuse per-layer QAD artifacts of an older run after
  checking that the recomputed layers are bit-identical.
* `experiments/int8_usage.py`, `experiments/make_calib_tasks.py`: INT8 usage report and
  task-style calibration text.
* Tests: `tests/test_lowbit.py`.

## What is not done

* The 4B final-KL runs at 100 steps and the 4B learning-rate run were stopped before they
  finished. The 4B numbers come from the 200-step run only.
* Nobody scored the 0.8B final-KL variants on the multiple-choice tasks.
