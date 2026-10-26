#!/usr/bin/env bash
# shellcheck disable=SC2086  # option strings are split on purpose
# QAD step budget on Qwen3.5-0.8B at ratio 10.75 (3.0 bpw): the full default run against budgeted
# runs with per-layer early stopping. The first run computes phase 1 (Hessians kept); the others
# hard-link it once its phase1.json and sensitivity key exist. Each run starts on an idle GPU.
# usage: chain_budget08.sh <tag prefix> "<hip indices>"   (box paths)
set -u
B=/home/chenco_adm/w4a8_runs/hadamard-int2
P=$1; CANDS=$2
H=$B/qad/experiments
export MODEL=/home/chenco_adm/models/Qwen3.5-0.8B OUT=$B/runs35
C="--calib $B/calib_qad.txt --calib-samples 128 --seq-len 2048 --max-error-ratio 10.75"
echo '{"budget": 0.5, "min_gain": 0.01, "min_steps": 75, "eval_every": 50}' > "$B/qad_b0.5.json"
echo '{"budget": 0.35, "min_gain": 0.01, "min_steps": 75, "eval_every": 50}' > "$B/qad_b0.35.json"
W=$OUT/$P-full.work
( KEEP_HESSIAN=1 bash $H/when_idle.sh "$CANDS" bash $H/run_qad_model.sh "$P-full" $C ) &
until [ -e "$W/phase1.json" ] && [ -e "$W/sensitivity.key.json" ]; do sleep 30; done
sleep 60
for b in 0.5 0.35; do
  ( SHARE_FROM=$W KEEP_WORK=1 bash $H/when_idle.sh "$CANDS" bash $H/run_qad_model.sh "$P-b$b" $C --qad-config "$B/qad_b$b.json"
    rm -rf "$OUT/$P-b$b.work/hessian" ) &
  sleep 120
done
wait
echo "finished" > "$OUT/$P.done"
