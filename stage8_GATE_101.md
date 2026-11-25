# Stage 8 gate, pre-registered 2026-11-25 by stage8_GATE_101.md, before any new result.
# Subset n=101, committed before the run. Long context is NOT in this float32 gate:
# the 8192 float32 preflight OOMed on eager attention softmax (24.9 GiB allocated),
# so all 8K/16K/32K windows are excluded here and gated separately through the served
# path (kl_probe on 8K/16K windows, vLLM, max_model_len 32768), not here.
subset n=101 (code 18 = 2 per language, multilingual 24 = 2 per language,
  agentic 20, math_reasoning 20, general_chat 19, long_context 0).
Inputs: same stages 1-4 allocation (target-bpw 3.5, same include/exclude/candidates/
  refine/rotation), new calibration text = 256 windows, 2048 tokens each, drawn
  round-robin from all 27 stage-4 v2 no-copyleft calibration files; QAD none.
  Old = stage4/runs/bpw3.5 (its calibration text was heldout_long_context__en__32768,
  first 256 x 2048), 13.67 GiB. New = stage8/runs/bpw3.5-v2, 3.5 bpw, only the
  calibration text changed. Size gate: the new output directory is within +-1%
  of 13.67 GiB.
Held-out: stage-4 v2 no-copyleft. The 8192/16384 float32 preflights are single
  tries recording peak. If the 8192 try fails (any OOM), every 8K/16K/32K window is
  excluded from this paired float32 eval and the 101 non-long windows above are
  scored. There is no retry and no loop. The 16384 try runs once only if 8192 passes;
  if it fails, 16K/32K are excluded and 8K long-context rows remain in the n=120 gate.
  The selected rows are spread over each file; 32K is never selected.
  Float32 evaluation of the old and new student, same rows.
Metric: delta = new KL - old KL per held-out row.
  pooled improves only if mean(delta) < -1.0 * sem(delta),
    sem = sd(delta, ddof=1) / sqrt(n), n = 101 (or the subset size if a
    preflight exclusion removes rows; never recomputed after results).
  category gate: for each remaining held-out category over all its selected rows,
    mean(delta) <= 1.0 * sem(delta).
    The category sem is the paired sem of the windows in that category.
  A category worsens when delta > sem. Any worsening category keeps the old PTQ.
  The gate uses the numbers as found; never recomputed after seeing results.
  Long context is gated separately via the served KL probe, not by this table.
OOD: stage-4 calib_v2/ood_docs__en__2048_ids.npy, first 8 windows, report only
  (no gate).
Decision: adopt the new PTQ only if it improves beyond sem pooled and no category
  worsens; otherwise keep the old PTQ. Written to $OUT/summary.json (adopt), and
  $OUT/PRODUCTION (old or new path). Never changes `hi2` defaults.
Gate constants are fixed: Pooled sem multiplier = 1.0; category sem multiplier = 1.0;
  OOD included in gate = no; OOD report only.
Budget: 101 windows x 25 min / 32 windows = 1.315 h/model (2.630 h both).
