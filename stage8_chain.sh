#!/usr/bin/env bash
# Stage 8: v2 calibration, same allocation as the stage-4 3.5 bpw PTQ, QAD none,
# replace only the calibration text, then paired held-out/OOD comparison.
#
# The chain never takes the GPUs until $OUT/GO exists, and it waits behind any
# other a2 lease (HI2Inkling or any other *.a2 / *.a2.request) through
# lease_holder.sh. Every GPU command runs under gpu_guard2.sh with boot.lock,
# the d1/d4 PCI check and a quiet file. Nothing imports torch on the CPU outside
# gpu_guard2.sh.
set -Eeuo pipefail
export LC_ALL=C.UTF-8

W=${W:-$HOME/w4a8_runs/inkling-hi2}
TREE=${TREE:-$W/tree-prod}
OUT=${OUT:-$W/stage8}
SRC=${SRC:-$HOME/hfcache/inkling2-mini}
V2=${V2:-$W/stage4/calib_v2_nocopyleft}
V2_DEFAULT=${V2_DEFAULT:-$W/stage4/calib_v2}
OLD=${OLD:-$W/stage4/runs/bpw3.5}
NEW=${NEW:-$OUT/runs/bpw3.5-v2}
WORK=${WORK:-$OUT/work/bpw3.5-v2}
CACHE=${CACHE:-$OUT/cache}
PREPARED=${PREPARED:-$OUT/prepared}
CALIB_TEXT=${CALIB_TEXT:-$PREPARED/calib_v2_balanced.txt}
CALIB_META=${CALIB_META:-$PREPARED/calib_v2_balanced.json}
CALIB_SAMPLES=${CALIB_SAMPLES:-256}
CALIB_SEQ_LEN=${CALIB_SEQ_LEN:-2048}
SUB=${SUB:-32}
OOD_WINDOWS=${OOD_WINDOWS:-8}
FORCE=${FORCE:-0}
SKIP_PREFLIGHT=${SKIP_PREFLIGHT:-0}
SKIP_CALIB_TEXT=${SKIP_CALIB_TEXT:-0}
SKIP_EVAL=${SKIP_EVAL:-0}
SKIP_OOD=${SKIP_OOD:-0}
BATCH_WINDOWS=${BATCH_WINDOWS:-1}
INCLUDE=${INCLUDE:-'\.mlp\.experts\.'}
EXCLUDE=${EXCLUDE:-'(embed|lm_head|norm|\.mlp\.gate(\.weight)?$|router|indexer|conv1d|\.bias$)'}
TRANSFORMERS=${TRANSFORMERS:-$HOME/w4a8_runs/pydeps/tfm-main}
GPU_GUARD=${GPU_GUARD:-$W/gpu_guard2.sh}
HOLDER=${HOLDER:-$W/lease_holder.sh}
LEASE=${LEASE:-$HOME/w4a8_runs/lanes/lease}
OWNER=${OWNER:-HI2Inkling}
QUIET_FILE=${QUIET_FILE:-$HOME/w4a8_runs/lanes/quiet-a2/Hi2InklingProd}
EXPECT_PCI_BUSES=${EXPECT_PCI_BUSES:-d1,d4}
GUARD_BOOT_SECONDS=${GUARD_BOOT_SECONDS:-600}
GUARD_MEM_FREE_MB=${GUARD_MEM_FREE_MB:-1800}
VENV=${VENV:-$HOME/Apps/vllm/venv-7.14.0_0.34.0-rebase-1e1cac5340}
PY=${PY:-$VENV/bin/python}
GPU_SH=${GPU_SH:-$W/gpu.sh}
PYPATH_EXTRA=${PYPATH_EXTRA:-$HOME/w4a8_runs/pydeps/tfm-main}
REASON=not-started
T0=$(date +%s)
mkdir -p "$OUT"/{logs,preflight,eval,quiet,prepared}

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
say() {
  log "SAY: $*"
}

lease_ok() {
  [[ -n "${HOLDER_PID:-}" && -e "$LEASE/$OWNER.a2" ]] || return 1
  awk -v o="$OWNER" -v p="$HOLDER_PID" 'NF >= 2 && $1 == o && $2 == p { found = 1 } END { exit !found }' "$LEASE/$OWNER.a2" 2>/dev/null
}
other_a2() {
  find "$LEASE" -maxdepth 1 -type f -name '*.a2' \
    ! -name "$OWNER.a2" -print -quit 2>/dev/null
}
lease_end() {
  [[ -n "${HOLDER_PID:-}" ]] || return 0
  local hp=$HOLDER_PID
  kill "$hp" 2>/dev/null || true
  wait "$hp" 2>/dev/null || true
  HOLDER_PID=
  # Only delete the lease if this script's holder owns it; never another owner's file.
  if [[ -e "$LEASE/$OWNER.a2" ]] && awk -v o="$OWNER" -v p="$hp" 'NF >= 2 && $1 == o && $2 == p { found = 1 } END { exit !found }' "$LEASE/$OWNER.a2" 2>/dev/null; then
    rm -f "$LEASE/$OWNER.a2"
  fi
}
cleanup() {
  local rc=$?
  trap - EXIT
  lease_end
  log "FINISH rc=$rc reason=$REASON elapsed=$(( $(date +%s) - T0 ))s"
  say "Stage 8 ${REASON}; exit ${rc}"
  exit "$rc"
}
trap cleanup EXIT

run_gpu() {
  local label=$1; shift
  if ! lease_ok; then
    log "lease $OWNER.a2 not live before $label"
    return 75
  fi
  if [[ -n "$(other_a2)" ]]; then
    log "another a2 lease appeared before $label"
    return 75
  fi
  local q=$QUIET_FILE
  mkdir -p "$(dirname "$q")"
  touch "$q"
  local toucher
  ( while kill -0 $$ 2>/dev/null; do touch "$q"; sleep 300; done ) &
  toucher=$!
  local rc=0
  set +e
  GUARD_QUIET_FILE="$q" GUARD_PCI_BUSES="$EXPECT_PCI_BUSES" \
  GUARD_LEASE_OWNER="$OWNER" GUARD_BOOT_SECONDS="$GUARD_BOOT_SECONDS" \
  GUARD_MEM_FREE_MB="$GUARD_MEM_FREE_MB" \
  OMP_NUM_THREADS=12 PYPATH_EXTRA="$PYPATH_EXTRA" \
    "$GPU_GUARD" 0,1 "$label" -- "$@" 2>&1 | tee "$OUT/logs/$label.log"
  rc=${PIPESTATUS[0]}
  set -e
  kill "$toucher" 2>/dev/null || true
  wait "$toucher" 2>/dev/null || true
  rm -f "$q" "$q.run"
  log "$label rc=$rc"
  return "$rc"
}

# ---- pre-registered gate ----------------------------------------------------
GATE=$OUT/GATE.md
if [[ ! -e "$GATE" ]]; then
  cat > "$GATE" <<'EOF'
# Stage 8 gate, pre-registered 2026-11-25 by stage8_chain.sh, before any new result.
Inputs: same stages 1-4 allocation (target-bpw 3.5, same include/exclude/candidates/
  refine/rotation), new calibration text = 256 windows, 2048 tokens each, drawn
  round-robin from all 27 stage-4 v2 no-copyleft calibration files; QAD none.
  Old = stage4/runs/bpw3.5 (its calibration text was heldout_long_context__en__32768,
  first 256 x 2048), 13.67 GiB. New = stage8/runs/bpw3.5-v2, 3.5 bpw, only the
  calibration text changed. Size gate: the new output directory is within +-3%
  of 13.67 GiB.
Held-out: stage-4 v2 no-copyleft: 8 heldout sets, 48 windows, 32K tokens, one
  held-out example per set. First 32 windows in file order are scored (SUB=32).
  Float32 evaluation of the old and new student, same window indices.
Metric: delta = new KL - old KL per held-out window.
  pooled improves only if mean(delta) < -1.0 * sem(delta),
    sem = sd(delta, ddof=1) / sqrt(n).
  category gate: for each of the 8 held-out categories, mean(delta) <= 1.0 * sem(delta).
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
EOF
fi
log "gate: $GATE"

# ---- calibration text -------------------------------------------------------
if [[ "$SKIP_CALIB_TEXT" != "1" && ! -s "$CALIB_TEXT" ]]; then
  export TREE PYPATH_EXTRA
  "$W/cpu.sh" "$TREE/experiments/inkling_calib_v2_text.py" \
    --calib-dir "$V2" --src "$SRC" --out "$CALIB_TEXT" --meta "$CALIB_META" \
    > "$OUT/logs/calib_text.log" 2>&1
fi
[[ -s "$CALIB_TEXT" ]] || { REASON="calib-text-missing"; exit 1; }
log "calib text: $(wc -c < "$CALIB_TEXT") bytes"

# ---- GO / lease -------------------------------------------------------------
log "waiting for $OUT/GO (a2 untouched)"
while [[ ! -e "$OUT/GO" ]]; do
  [[ -e "$OUT/STOP" ]] && { REASON=stop-before-go; exit 1; }
  sleep 30
done
log "GO present"

if [[ -n "$(other_a2)" ]]; then
  log "another a2 holder exists; waiting for handover"
fi
# lease_holder does not distinguish another holder with the same OWNER; wait for it
# before writing our request, then the pid check in lease_ok protects the rest.
while [[ -e "$LEASE/$OWNER.a2" ]]; do
  [[ -e "$OUT/STOP" ]] && { REASON=stop-before-same-owner-lease; exit 1; }
  read -r _ owner_pid _ < "$LEASE/$OWNER.a2" 2>/dev/null || owner_pid=
  if [[ -n "$owner_pid" ]] && ! kill -0 "$owner_pid" 2>/dev/null; then
    log "removing stale same-owner lease: $(cat "$LEASE/$OWNER.a2" 2>/dev/null)"
    rm -f "$LEASE/$OWNER.a2"
    break
  fi
  log "same-owner lease still live: $(cat "$LEASE/$OWNER.a2" 2>/dev/null)"
  sleep 10
done
if [[ -z "${HOLDER_PID:-}" ]]; then
  TOPIC=HI2Inkling-a2-prod
  NOTE="stage8 v2 PTQ prod"
  LEASE_DIR="$LEASE" HOLD_MAX=21600 WATCH_PID=$$ STOP_FILE="$OUT/STOP" \
    "$HOLDER" "$OWNER" "$TOPIC" "$NOTE" >> "$OUT/logs/lease.log" 2>&1 &
  HOLDER_PID=$!
fi
while ! lease_ok; do
  [[ -e "$OUT/STOP" ]] && { REASON=stop-before-lease; exit 1; }
  kill -0 "$HOLDER_PID" 2>/dev/null || {
    log "lease holder exited before the lease was live"
    REASON=lease-holder-exited
    exit 1
  }
  sleep 5
done
log "lease $OWNER.a2 live; holder pid $HOLDER_PID"
lease_ok || { REASON=lease-not-live; exit 1; }

# ---- preflight --------------------------------------------------------------
PRE=$OUT/preflight
DEC=$PRE/decision.json
stats_try() {
  local L=$1
  local out=$PRE/stats_$L.json
  rm -f "$out"
  run_gpu "preflight_stats_$L" "$GPU_SH" "$TREE/experiments/inkling_looped_smoke.py" \
    --src "$SRC" --transformers "$TRANSFORMERS" \
    --devices cuda:0 cuda:1 --seqs 1 --seq-len "$L" --calib-layers 1 \
    --compute-dtype float16 --skip-compare --text "$CALIB_TEXT" \
    --workdir "$PRE/work" --out "$out"
}
eval_try() {
  local L=$1
  local out=$PRE/eval_$L.json
  rm -f "$out"
  run_gpu "preflight_eval_$L" "$GPU_SH" "$TREE/experiments/inkling_kl_eval.py" \
    --dst "$OLD" --ids "$PRE/window_32768.npy" --windows 1 --seq-len "$L" \
    --devices cuda:0 cuda:1 --compute-dtype float32 --groups none \
    --batch-windows 1 --position-buckets --out "$out"
}
if [[ -s "$DEC" ]]; then
  log "reusing preflight decision: $DEC"
elif [[ "$SKIP_PREFLIGHT" == "1" ]]; then
  [[ -s "$DEC" ]] || cat > "$DEC" <<'EOF'
{"stats_max":null,"eval_max":32768,"batch_windows":1,"assumed":true,"skipped":true}
EOF
  log "preflight skipped by SKIP_PREFLIGHT=1"
else
  STATS_MAX=""
  if stats_try 65536; then STATS_MAX=65536
  elif stats_try 32768; then STATS_MAX=32768
  elif stats_try 31744; then STATS_MAX=31744
  elif stats_try 30720; then STATS_MAX=30720
  fi
  [[ -n "$STATS_MAX" ]] || { REASON=preflight-stats-all-failed; exit 1; }
  log "stats max $STATS_MAX"

  EVAL_MAX=""
  for ((L=32768; L>=2048; L-=1024)); do
    if eval_try "$L"; then EVAL_MAX=$L; break; fi
  done
  [[ -n "$EVAL_MAX" ]] || { REASON=preflight-eval-all-failed; exit 1; }
  log "eval max $EVAL_MAX"
  "$PY" - "$DEC" "$STATS_MAX" "$EVAL_MAX" "$BATCH_WINDOWS" <<'PY'
import json, sys
path, stats, ev, batch = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
with open(path, "w") as f:
    json.dump({"stats_max": stats, "eval_max": ev, "batch_windows": batch,
               "assumed": False, "skipped": False}, f, indent=2)
    f.write("\n")
PY
fi
EVAL_SEQ=$("$PY" - "$DEC" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(int(d.get("eval_max") or 32768))
PY
)
log "preflight decision: $DEC; eval_seq=$EVAL_SEQ"

# ---- quantize ---------------------------------------------------------------
if [[ -f "$NEW/hi2_run.json" && "$FORCE" != "1" ]]; then
  log "quantize already complete: $NEW/hi2_run.json"
else
  run_gpu quantize_v2 /usr/bin/time -v "$GPU_SH" -m hi2 quantize \
    "$SRC" "$NEW" \
    --workdir "$WORK" --cache-dir "$CACHE" \
    --devices cuda:0 cuda:1 --compute-dtype float16 --moe-layout stacked \
    --include "$INCLUDE" --exclude "$EXCLUDE" --target-bpw 3.5 \
    --calib "$CALIB_TEXT" --calib-samples "$CALIB_SAMPLES" --seq-len "$CALIB_SEQ_LEN" \
    --sens-seqs 8 --sens-len 512 --sensitivity none --no-qad || {
      REASON=quantize-v2-failed; exit 1; }
fi
[[ -f "$NEW/hi2_run.json" ]] || { REASON=quantize-v2-no-run-json; exit 1; }

set +e
"$PY" - "$OLD" "$NEW" "$OUT/size.json" <<'PY'
import json, os, sys
old, new, path = sys.argv[1:4]
def size(p):
    return sum(os.path.getsize(os.path.join(p, n)) for n in os.listdir(p)
               if os.path.isfile(os.path.join(p, n)))
a, b = size(old), size(new)
r = {"old_bytes": a, "new_bytes": b, "old_gib": a / 2**30, "new_gib": b / 2**30,
     "delta": b - a, "ratio": b / a, "gate_pm3pct": abs(b / a - 1.0) <= 0.03}
with open(path, "w") as f:
    json.dump(r, f, indent=2); f.write("\n")
print(json.dumps(r, indent=2))
raise SystemExit(0 if r["gate_pm3pct"] else 3)
PY
SIZE_RC=$?
set -e
if [[ "$SIZE_RC" != "0" ]]; then
  REASON=size-gate-failed
  exit 1
fi

# ---- paired evaluation ------------------------------------------------------
if [[ "$SKIP_EVAL" != "1" ]]; then
  mkdir -p "$OUT/eval"
  if [[ -s "$OUT/eval/old_heldout.json" && "$FORCE" != "1" ]]; then
    log "reusing eval/old_heldout.json"
  else
    run_gpu eval_old_heldout "$GPU_SH" "$TREE/experiments/inkling_kl_eval.py" \
      --dst "$OLD" --ids "$V2" --windows 0 --split heldout \
      --seq-len "$EVAL_SEQ" --devices cuda:0 cuda:1 --compute-dtype float32 \
      --groups none --batch-windows "$BATCH_WINDOWS" --position-buckets \
      --out "$OUT/eval/old_heldout.json" || { REASON=eval-old-failed; exit 1; }
  fi
  if [[ -s "$OUT/eval/new_heldout.json" && "$FORCE" != "1" ]]; then
    log "reusing eval/new_heldout.json"
  else
    run_gpu eval_new_heldout "$GPU_SH" "$TREE/experiments/inkling_kl_eval.py" \
      --dst "$NEW" --ids "$V2" --windows 0 --split heldout \
      --seq-len "$EVAL_SEQ" --devices cuda:0 cuda:1 --compute-dtype float32 \
      --groups none --batch-windows "$BATCH_WINDOWS" --position-buckets \
      --out "$OUT/eval/new_heldout.json" || { REASON=eval-new-failed; exit 1; }
  fi
fi
if [[ "$SKIP_OOD" != "1" ]]; then
  if [[ -s "$OUT/eval/old_ood.json" && "$FORCE" != "1" ]]; then
    log "reusing eval/old_ood.json"
  else
    run_gpu eval_old_ood "$GPU_SH" "$TREE/experiments/inkling_kl_eval.py" \
      --dst "$OLD" --ids "$V2_DEFAULT/ood_docs__en__2048_ids.npy" --windows "$OOD_WINDOWS" \
      --devices cuda:0 cuda:1 --compute-dtype float32 --groups none \
      --batch-windows 1 --position-buckets --out "$OUT/eval/old_ood.json" || {
        REASON=eval-old-ood-failed; exit 1; }
  fi
  if [[ -s "$OUT/eval/new_ood.json" && "$FORCE" != "1" ]]; then
    log "reusing eval/new_ood.json"
  else
    run_gpu eval_new_ood "$GPU_SH" "$TREE/experiments/inkling_kl_eval.py" \
      --dst "$NEW" --ids "$V2_DEFAULT/ood_docs__en__2048_ids.npy" --windows "$OOD_WINDOWS" \
      --devices cuda:0 cuda:1 --compute-dtype float32 --groups none \
      --batch-windows 1 --position-buckets --out "$OUT/eval/new_ood.json" || {
        REASON=eval-new-ood-failed; exit 1; }
  fi
fi

# ---- summary -----------------------------------------------------------------
SUMMARY_ARGS=(--old-heldout "$OUT/eval/old_heldout.json"
  --new-heldout "$OUT/eval/new_heldout.json"
  --old-ptq "$OUT/eval/old_heldout.json" --new-ptq "$OUT/eval/new_heldout.json"
  --old-ood "$OUT/eval/old_ood.json" --new-ood "$OUT/eval/new_ood.json"
  --windows "$SUB" --out-json "$OUT/summary.json" --out-md "$OUT/summary.md"
  --run-json "$NEW/hi2_run.json" --calib-meta "$CALIB_META")
if [[ "$SKIP_OOD" == "1" ]]; then
  SUMMARY_ARGS=(--old-heldout "$OUT/eval/old_heldout.json"
    --new-heldout "$OUT/eval/new_heldout.json"
    --windows "$SUB" --out-json "$OUT/summary.json" --out-md "$OUT/summary.md"
    --run-json "$NEW/hi2_run.json" --calib-meta "$CALIB_META")
fi
"$PY" "$TREE/experiments/inkling_v2_compare.py" "${SUMMARY_ARGS[@]}" \
  > "$OUT/logs/summary.log" 2>&1 || { REASON=summary-failed; exit 1; }
cat "$OUT/logs/summary.log"

if "$PY" - "$OUT/summary.json" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1]))["adopt"] else 1)
PY
then
  printf '%s\n' "$NEW" > "$OUT/PRODUCTION"
  REASON=adopted-new-ptq
else
  printf '%s\n' "$OLD" > "$OUT/PRODUCTION"
  REASON=kept-old-ptq
fi
say "Stage 8 $(cat "$OUT/PRODUCTION")"
log "production: $(cat "$OUT/PRODUCTION")"
