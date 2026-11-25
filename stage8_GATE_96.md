# Stage 8 gate, pre-registered 2026-11-25 by stage8_GATE_96.md, before any new result.
# Subset n=96, committed before the run (budget fallback).
subset n=96 (code 18 = 2 per language, multilingual 24 = 2 per language,
  agentic 14, math_reasoning 14, general_chat 13, long_context 13).
Inputs: same stages 1-4 allocation (target-bpw 3.5, same include/exclude/candidates/
  refine/rotation), new calibration text = 256 windows, 2048 tokens each, drawn
  round-robin from all 27 stage-4 v2 no-copyleft calibration files; QAD none.
  Old = stage4/runs/bpw3.5 (its calibration text was heldout_long_context__en__32768,
  first 256 x 2048), 13.67 GiB. New = stage8/runs/bpw3.5-v2, 3.5 bpw, only the
  calibration text changed. Size gate: the new output directory is within +-1%
  of 13.67 GiB.
Held-out: stage-4 v2 no-copyleft has 27 files/820 windows (2K, 8K, 16K, 32K).
  Native stats preflight is skipped (calibration is 2048 tokens). Only one float32
  eval try at 8192 and one at 16384; 32K is not preflighted and is excluded. If the
  16384 try fails, 16K is excluded too; the 8192 try must pass.
  The 96 rows are materialized from the fixed quotas above, with row indices
  spread over each file; long_context uses 6 rows of 8192 and 7 rows of 16384 when
  the 16384 preflight passes, otherwise 13 rows of 8192. 32K is never selected.
  Float32 evaluation of the old and new student, same rows.
Metric: delta = new KL - old KL per held-out row.
  pooled improves only if mean(delta) < -1.0 * sem(delta),
    sem = sd(delta, ddof=1) / sqrt(n), n = 96.
  category gate: for each remaining held-out category over all its selected rows,
    mean(delta) <= 1.0 * sem(delta).
    The category sem is the paired sem of the windows in that category.
  A category worsens when delta > sem. Any worsening category keeps the old PTQ.
  The gate uses the numbers as found; never recomputed after seeing results.
OOD: stage-4 calib_v2/ood_docs__en__2048_ids.npy, first 8 windows, report only
  (no gate).
Decision: adopt the new PTQ only if it improves beyond sem pooled and no category
  worsens; otherwise keep the old PTQ. Written to $OUT/summary.json (adopt), and
  $OUT/PRODUCTION (old or new path). Never changes `hi2` defaults.
Gate constants are fixed: Pooled sem multiplier = 1.0; category sem multiplier = 1.0;
  OOD included in gate = no; OOD report only.
