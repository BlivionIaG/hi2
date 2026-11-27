#!/usr/bin/env bash
# One-invocation QAD `default` + final logit-KL 200 for the Inkling-2-Mini union
# allocation, then the fp32 instruments.
#
# Why one invocation: hi2's run identity (hi2/service.py run_identity) hashes the
# config, the calibration, the source and the environment. A second invocation with
# a different --qad-layers is a different run and cannot resume the first's workdir
# (`incompatible resume`), and any code edit invalidates an existing workdir too.
# So: 24 layers and the final phase in a single call, no halfway restart.
#
# Launch: setsid nohup ./qad_full.sh > qad/full.log 2>&1 < /dev/null &
set -uo pipefail
W=${W:-$HOME/w4a8_runs/inkling-hi2}
RT=${RT:-$HOME/w4a8_runs/inkling-hi2-runtime}
K=$RT/tree/tools/inkling_hi2
OUT=${OUT:-$RT/qad}
V=${V:-$HOME/Apps/vllm/venv-7.14.0_0.34.0-rebase-1e1cac5340}
NAME=${NAME:-3.5-full}
DST=$W/stage8/runs/$NAME
WORK=$W/stage8/work/$NAME
LOG=$OUT/full.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

export SLOT=a2 HIP_VISIBLE_DEVICES=0,1 EXPECT_PCI_BUSES=d1,d4
export OWNER=HI2Inkling
export LEASE_DIR=$HOME/w4a8_runs/lanes/lease
export QUIET_DIR=$HOME/w4a8_runs/lanes/quiet-a2
export BOOT_LOCK=$HOME/w4a8_runs/boot.lock
export LEASE_FREE_GIB=28 LEASE_POLL_SECONDS=30 LEASE_WAIT_SECONDS=7200
QUIET_FILE=$QUIET_DIR/HI2InklingQAD
export GUARD_QUIET_FILE=$QUIET_FILE
STOP_FILE=$OUT/STOP
rm -f "$STOP_FILE"

WATCH_PID=$$ HOLD_MAX=43200 STOP_FILE=$STOP_FILE NOTE="QAD full: 24 layers + final 200, $NAME" \
    "$W/lease_holder.sh" >>"$LOG" 2>&1 &
lease_pid=$!
TOUCH_PID=
cleanup() {
    rm -f "$QUIET_FILE" 2>/dev/null
    : >"$STOP_FILE" 2>/dev/null
    [[ -n ${TOUCH_PID:-} ]] && kill "$TOUCH_PID" 2>/dev/null
    kill "$lease_pid" 2>/dev/null
}
trap cleanup EXIT
trap 'cleanup; exit 130' TERM INT

say "acquiring a2 as HI2Inkling"
for ((i = 0; i < 240; i++)); do
    [[ -f $LEASE_DIR/HI2Inkling.a2 ]] && break
    kill -0 "$lease_pid" 2>/dev/null || { say "lease holder exited"; exit 3; }
    sleep 5
done
[[ -f $LEASE_DIR/HI2Inkling.a2 ]] || { say "lease not acquired"; exit 3; }
say "lease held: $(cat "$LEASE_DIR/HI2Inkling.a2")"
deadline=$(( $(date +%s) + 1800 ))
while q=$(find "$HOME/w4a8_runs"/lanes/quiet* -type f -mmin -6 2>/dev/null |
    grep -v "HI2InklingQAD" | head -1) && [[ -n $q ]]; do
    (( $(date +%s) >= deadline )) && { say "REFUSING: quiet file $q"; exit 4; }
    say "quiet file $q newer than 6 min; waiting 30 s"; sleep 30
done
mkdir -p "$QUIET_DIR"; : >"$QUIET_FILE"
TOUCH_PID=$(python3 "$K/quiet_touch.py" --file "$QUIET_FILE" --interval 300 --count 0 >/dev/null 2>&1 & echo $!)
say "quiet file re-touched every 300 s: $QUIET_FILE"

mkdir -p "$WORK"
ARGV=( -m hi2 quantize "$HOME/hfcache/inkling2-mini" "$DST"
    --calib "$W/stage8/prepared/calib_union_50_50.txt" --calib-samples 256 --seq-len 2048
    --target-bpw 3.5 --sensitivity none --qad-preset default
    --qad-layers 0-23 --qad-final-kl 200
    --devices cuda:0,cuda:1 --cache-dir "$W/stage8/cache" --workdir "$WORK" )

say "plan check (CPU)"
if ! env -u HIP_VISIBLE_DEVICES -u ROCR_VISIBLE_DEVICES PYTHONPATH="$W/tree" \
    "$V/bin/python" "${ARGV[@]}" --dry-run --json --devices cpu \
    >"$OUT/plan-full.json" 2>"$OUT/plan-full.err"; then
    say "REFUSED: $(tail -2 "$OUT/plan-full.err" | tr '\n' ' ' | cut -c1-300)"
    exit 5
fi
say "plan ok: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print({k: d.get(k) for k in ("target_bpw","layers","calibrated","sensitivity")})' "$OUT/plan-full.json" 2>/dev/null)"

# progress: one line per layer boundary from the QAD log, plus a stall watchdog
( while sleep 60; do
    [[ -f $STOP_FILE ]] && exit 0
    n=$(ls "$WORK"/qad/layer*.pt 2>/dev/null | wc -l)
    say "layers done: $n/24"
    if [[ -f $WORK/qad_report.json ]] && [[ -f $OUT/.last_nmse ]] && \
        cmp -s <(python3 - "$WORK/qad_report.json" <<'PY'
import json,sys
try:
    d=json.load(open(sys.argv[1])); e=d[-1]
    print(round(float(e.get("val_after",0)),9))
except Exception: print("")
PY
) "$OUT/.last_nmse"; then
        say "STALL: the last layer's held-out nMSE did not move; stop with STOP if it repeats"
    fi
  done ) &
WPID=$!

say "running: 24 layers + final 200 (single invocation)"
GUARD_BOOT_SECONDS=600 OMP_NUM_THREADS=12 TREE=$W/tree \
    PYPATH_EXTRA=$HOME/w4a8_runs/pydeps/tfm-main \
    "$W/gpu_guard2.sh" "0,1" "qad-full" -- "$W/gpu.sh" "${ARGV[@]}" >>"$OUT/full-quantize.log" 2>&1
rc=$?
kill $WPID 2>/dev/null
if [[ $rc -ne 0 ]]; then
    say "quantize FAILED rc=$rc (see $OUT/full-quantize.log)"; exit $rc
fi
if grep -qiE "final_kl.*skipped|skipped.*final_kl|fit check failed" "$OUT/full-quantize.log"; then
    say "FINAL PHASE WAS SKIPPED -- refusing the artifact"; exit 6
fi
say "quantize done: $DST"

ln -sfn "$DST" "$W/stage8/runs/bpw$NAME"
for s in mixed wiki pydocs; do
    say "eval $s"
    RUNS=$W/stage8/runs EVALS=$OUT/eval TEACHER=$W/stage4/teacher STAGE4_ENV=$W/stage4.env \
        bash "$W/tree/experiments/inkling_stage4.sh" eval "$NAME" "$s" >>"$LOG" 2>&1 \
        || say "eval $s failed"
done
python3 - "$OUT/eval/bpw${NAME}_mixed.json" "$OUT/eval/bpw${NAME}_wiki.json" "$OUT/eval/bpw${NAME}_pydocs.json" <<'PY' | tee -a "$LOG"
import json,sys
for p in sys.argv[1:]:
    try:
        d=json.load(open(p)); s=d["student"]
        print(f"{p.rsplit('/',1)[-1]}: windows {d.get('windows')} dtype {d.get('dtype')} "
              f"kl_mean {float(s['kl_mean']):.5f} top1 {s.get('top1_mean', s.get('acc_mean', float('nan')))}")
    except Exception as e:
        print(f"{p.rsplit('/',1)[-1]}: unreadable ({type(e).__name__})")
PY
say "union reference: mixed 0.10592 wiki 0.03067 pydocs 0.03039 (fp32, 32/17/17 windows)"
say "done"
