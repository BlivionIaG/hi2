# Inkling-2-Mini HI2: the production record (as of 2026-11-26)

Checkpoint: `BlivionIaG/Inkling-2-Mini-HI2-3.5bpw` (not uploaded yet).
Source weights: `~/hfcache/inkling2-mini` (nvfp4 weights + mxfp8 block scales,
bf16 dense tensors). Quantizer: this repo, branch `feat/inkling2-prod` plus
`feat/inkling2-union-arms`.

## What is quantized

Experts only. 9,216 tensors (24.46 B weights) re-encoded into the HI2 INT2/INT4
container, roughly 42 % / 58 %. Everything else is copied unchanged to fp16: the
24 layer norms, routers, the vision tower, MTP, attention, the tokenizer/embedding
side and the lm_head. INT8 promotion of the bf16 classes (attention, embeddings,
lm_head) was measured and **deferred by the user**; none of the three passed the
pre-registered gate (`stage4/results5/int8_classes.json`).

The allocation is weight-error based: `sensitivity: none` in the run config, no
GEMM-informed input. The 27B recipe had a 12-log GEMM study; this model does not.

## Calibration: three candidates, one shipped

| candidate | built from | mixed KL (fp32, 32 win) | wiki | pydocs |
|---|---|---:|---:|---:|
| v1 (stage-4 calib) | English-heavy, 781 windows | 0.09963 | 0.04683 | 0.03054 |
| v2 (`calib_v2_balanced`) | 781 win, 12 languages, long/code/agentic | 0.13864 gate / 0.14068 paired | - | - |
| **union 50/50 (shipped)** | first 128 windows of v1 + `calib_v2_balanced` | **0.10592** | **0.03067** | **0.03039** |

The union is the only candidate that passes the +0.010 mixed-KL allowance
against v1 *and* holds the end-to-end profile: agentic 57/60 vs stock 58/60
(panel recount 58 vs 52 against v1), code 35/40, long 27/27, math 48/48,
multilingual 70/70; vision 85 items 97.7 % (v1 98.8 %, p=1.0). v2 is 0.033-0.040
worse on mixed KL; English-only v1 costs 5-6 agentic items.

On the fp16 instrument the union's mixed KL is 0.09833 / top-1 0.9272 (v1
0.09292) and its long KL is 8k 0.016546 / 16k 0.015680 (v1 0.03634 / 0.03458,
i.e. the union halves long-context KL and gives up 0.0054 on mixed, inside the
allowance). The card quotes the fp32 reference numbers.

The union is built by `experiments/inkling_union_calib.py`: `hi2 quantize
--calib` takes the first `samples * seq_len` tokens of a concatenated file, so a
plain `cat` would take all 256 windows from the first corpus. The script cuts v1
at a token boundary, inserts an optional separator, and verifies from the ids
that windows 0-127 are v1 and 128-255 are v2 before writing
`stage8/prepared/calib_union_50_50.txt`.

## Quality levers: what was and was not run

This checkpoint is **PTQ only**. That was the conclusion of five QAD/final-KL
pilots between 2026-11-25 and 11-26; it is **superseded** and should not be
quoted:

- the pilots spent 20-150 steps per layer against the 27B default preset of
  **300 steps per layer, 256 sequences, float32 buffers, data-parallel steps**,
  plus a **200-step final logit-KL phase**;
- the same repo's 27B record at the same bit budget: docs KLD
  0.09747 -> 0.05934 (**-39 %**), mix 0.21706 -> 0.18372 (**-15 %**, top-1
  86.7 -> 90.8 %);
- the 4B study at our exact 3.5 bpw: the final logit-KL phase alone, at the
  merged `final_lr_p` 0.03 (PR #118), gave **-6 % / -7 %** (docs / mix);
- the "no gain" pilots also ran on a tree that still defaulted `final_lr_p`
  0.02, and one pilot's apparent regression was the PTQ-fallback re-encode bug
  (`bf887d7`), not learning.

A full 24-layer run and a lower-bit ratio-6 arm are therefore being measured
now (2026-11-26, two V620 on `a2`), and this document must be updated with their
numbers before the card is written:

- **arm A**: QAD `default` (300 steps/layer) + final KL 200, on the union's
  3.5 bpw allocation. Gate: fp32 mixed below 0.10592 at the halfway layer
  checkpoint, else the arm aborts.
- **arm B**: fresh PTQ at 3.0 bpw + the same QAD default + final KL. The ratio-6
  result (QAD at ratio 6 beats PTQ at ratio 4 with 0.27 bpw less on 0.8B, 6.2 on
  4B) predicts a smaller file at better quality; it ships only if it wins.
- Both arms run `experiments/inkling_qad_arms.sh` (per-layer artifacts,
  resume by rerun, liveness check, the `steps=0` PTQ control, the halfway gate,
  and a refusal to accept a skipped final phase).

Do not describe the release as final-KL'd or QAD'd until one of those arms has
numbers on the shared instruments; and if one does, update the card, not just
this file.

## Reproducing

```
# data
stage8/prepared/calib_union_50_50.txt   and .json   (inkling_union_calib.py)
# quantize, PTQ only
hi2 quantize ~/hfcache/inkling2-mini <dst> --calib stage8/prepared/calib_union_50_50.txt \
    --calib-samples 256 --seq-len 2048 --target-bpw 3.5 --no-qad
# quantize, QAD default + final 200
hi2 quantize ... --target-bpw 3.5 --qad-preset default --qad-final-kl 200 --workdir <work>
```

Instruments: `experiments/inkling_stage4.sh eval <tag> {mixed,wiki,pydocs}` (fp32
teacher, 32 mixed windows, `student.kl_mean` in `bpw<tag>_<set>.json`) and
`experiments/inkling_kl_eval.py` for the served-path numbers.
