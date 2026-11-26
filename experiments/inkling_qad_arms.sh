#!/usr/bin/env bash
# Two-arm QAD run for the Inkling-2-Mini union checkpoint (Main's decision 2026-11-26):
#   arm A: QAD `default` + 200-step final KL on the union's 3.5 bpw allocation
#   arm B: fresh PTQ at 3.0 bpw + the same QAD/default + final KL
# Per-arm: layer-sharded QAD in two halves (0-11, 12-23) with per-layer artifacts
# (qad/layerNNN.pt) and resume by rerun; the halfway fp32 corpus eval is the abort gate
# against the union's PTQ instrument; the final phase runs from qad/head_inputs.pt and must
# not be skipped (the runner refuses a skipped final phase).
# Lease: HI2Inkling.a2 held for the whole run (lease_holder.sh), quiet file re-touched every
# 300 s, per-step gpu_guard2 boot lock/thermal/page-fault; caller adds the KFD/temps supervisor.
# progress lines: $OUT/progress.md after layers 6/12/18/24, after the final phase, after eval.
set -uo pipefail
W=${W:-$HOME/w4a8_runs/inkling-hi2}
RT=${RT:-$HOME/w4a8_runs/inkling-hi2-runtime}
K=$RT/tree/tools/inkling_hi2
OUT=${OUT:-$RT/qad}
LOG=$OUT/qad.log
mkdir -p "$OUT" "$(dirname "$LOG")"
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
sayp() { say "$*"; echo "- [$(date -u +%H:%M:%S)] $*" >> "$OUT/progress.md"; }

export SLOT=a2 HIP_VISIBLE_DEVICES=0,1 EXPECT_PCI_BUSES=d1,d4
export OWNER=HI2Inkling
export LEASE_DIR=$HOME/w4a8_runs/lanes/lease
export QUIET_DIR=$HOME/w4a8_runs/lanes/quiet-a2
export BOOT_LOCK=$HOME/w4a8_runs/boot.lock
export LEASE_FREE_GIB=28 LEASE_POLL_SECONDS=30 LEASE_WAIT_SECONDS=7200
QUIET_FILE=$QUIET_DIR/HI2InklingQAD
export GUARD_QUIET_FILE=$QUIET_FILE
UNION_MIXED=0.10592   # union PTQ fp32 mixed KLD (Main's instrument), the arm gate
HOLDER=$W/lease_holder.sh
STOP_FILE=$OUT/STOP
rm -f "$STOP_FILE"
WATCH_PID=$$ HOLD_MAX=43200 STOP_FILE=$STOP_FILE \
    NOTE="QAD arm A (3.5 bpw) then arm B (3.0 bpw), default+final-200" \
    "$HOLDER" >>"$LOG" 2>&1 &
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

say "acquiring a2 as HI2Inkling (outer lease for the whole run)"
for ((i = 0; i < 120; i++)); do
    [[ -f $LEASE_DIR/HI2Inkling.a2 ]] && break
    kill -0 "$lease_pid" 2>/dev/null || { say "lease holder exited"; exit 3; }
    sleep 5
done
[[ -f $LEASE_DIR/HI2Inkling.a2 ]] || { say "lease not acquired"; exit 3; }
sayp "lease held: $(cat "$LEASE_DIR/HI2Inkling.a2")"
deadline=$(( $(date +%s) + 1800 ))
while q=$(find "$HOME/w4a8_runs"/lanes/quiet* -type f -mmin -6 2>/dev/null |
    grep -v "HI2InklingQAD" | head -1) && [[ -n $q ]]; do
    (( $(date +%s) >= deadline )) && { say "REFUSING: quiet file $q"; exit 4; }
    say "quiet file $q newer than 6 min; waiting 30 s"; sleep 30
done
mkdir -p "$QUIET_DIR"; : >"$QUIET_FILE"
TOUCH_PID=$(python3 "$K/quiet_touch.py" --file "$QUIET_FILE" --interval 300 --count 0 >/dev/null 2>&1 & echo $!)
say "quiet file declared and re-touched every 300 s: $QUIET_FILE"

V=${V:-$HOME/Apps/vllm/venv-7.14.0_0.34.0-rebase-1e1cac5340}
qad_argv() { # dst work layers final bpw -> QAD_ARGV
    QAD_ARGV=( -m hi2 quantize "$HOME/hfcache/inkling2-mini" "$1"
        --calib "$W/stage8/prepared/calib_union_50_50.txt" --calib-samples 256 --seq-len 2048
        --target-bpw "$5" --sensitivity none --qad-preset default --qad-layers "$3" --qad-final-kl "$4"
        --devices cuda:0,cuda:1 --cache-dir "$W/stage8/cache" --workdir "$2" )
}
qad() { # arm dst work layers final bpw
    local arm=$1
    qad_argv "$2" "$3" "$4" "$5" "$6"
    GUARD_BOOT_SECONDS=600 OMP_NUM_THREADS=12 TREE=$W/tree PYPATH_EXTRA=$HOME/w4a8_runs/pydeps/tfm-main \
        "$W/gpu_guard2.sh" "0,1" "qad-$arm" -- "$W/gpu.sh" "${QAD_ARGV[@]}"
}
qad_plan() { # dst work layers final bpw: resolve the recipe on the CPU before any boot
    qad_argv "$@"
    env -u HIP_VISIBLE_DEVICES -u ROCR_VISIBLE_DEVICES \
        PYTHONPATH="$W/tree${PYPATH_EXTRA:+:$PYPATH_EXTRA}" \
        "$V/bin/python" "${QAD_ARGV[@]}" --dry-run --json --devices cpu
}
progress_watch() { # workdir tag: write boundary lines at 6/12/18/24 layer artifacts
    local work=$1 tag=$2 n
    local -A done=()
    while sleep 20; do
        [[ -f $STOP_FILE ]] && break
        n=$(ls "$work"/qad/layer*.pt 2>/dev/null | wc -l)
        for b in 6 12 18 24; do
            [[ -z ${done[$b]:-} && $n -ge $b ]] || continue
            done[$b]=1
            # last completed layer's val numbers from the report (list of dicts)
            local v=$(python3 - "$work/qad_report.json" <<'PY' 2>/dev/null
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    e=d[-1] if isinstance(d,list) else None
    print("layer %s steps %s val_init %.6g val_after %.6g val_before %.6g gpu_peak %.1f GiB" % (
        e.get("layer"),e.get("steps"),e.get("val_init",0),e.get("val_after",0),e.get("val_before",0),e.get("gpu_peak_gib",0)))
except Exception:
    print("")
PY
)
            sayp "$tag: $n layers checkpointed; $v"
        done
    done
}

qad_supervised() { # tag dst work layers final bpw log
    local tag=$1 dst=$2 work=$3 layers=$4 final=$5 bpw=$6 log=$7 qpid rc
    qad "$tag" "$dst" "$work" "$layers" "$final" "$bpw" >>"$log" 2>&1 &
    qpid=$!
    sleep 90
    if ! kill -0 "$qpid" 2>/dev/null; then
        wait "$qpid"; rc=$?
        say "qad $tag died within 90 s (rc=$rc): $(tail -3 "$log" | tr '\n' ' ')"
        return $rc
    fi
    wait "$qpid"; return $?
}

run_arm() { # arm name target_bpw
    local arm=$1 name=$2 bpw=$3
    local dst=$W/stage8/runs/$name work=$W/stage8/work/$name
    mkdir -p "$work"
    sayp "arm $arm start ($name, target-bpw $bpw): PTQ + QAD default, halves 0-11 / 12-23 + final 200"
    if ! qad_plan "$dst" "$work" 0-11 0 "$bpw" >"$OUT/plan-$arm.json" 2>"$OUT/plan-$arm.err"; then
        if grep -qi "incompatible resume" "$OUT/plan-$arm.err"; then
            local ts; ts=$(date -u +%H%M%S)
            sayp "arm $arm stale resume state in $work; quarantined to .stale-$ts and re-planning"
            mv "$work" "$work.stale-$ts" 2>/dev/null
            [[ -d $dst ]] && mv "$dst" "$dst.stale-$ts" 2>/dev/null
            qad_plan "$dst" "$work" 0-11 0 "$bpw" >"$OUT/plan-$arm.json" 2>"$OUT/plan-$arm.err" || {
                sayp "arm $arm REFUSED after quarantine: $(tail -2 "$OUT/plan-$arm.err" | tr '\n' ' ' | cut -c1-200)"; return 1; }
        else
            sayp "arm $arm REFUSED: the recipe does not resolve on the CPU: $(tail -2 "$OUT/plan-$arm.err" | tr '\n' ' ' | cut -c1-200)"
            return 1
        fi
    fi
    sayp "arm $arm plan ok: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print({k:d.get(k) for k in ('target_bpw','steps','layers','final_kl') if k in d})" "$OUT/plan-$arm.json" 2>/dev/null || echo 'see plan json')"
    progress_watch "$work" "arm $arm" &
    local wp=$!
    local t0=$(date +%s)
    # half 1: layers 0-11, final phase off (the halfway checkpoint is the abort gate)
    qad_supervised "$arm-half1" "$dst" "$work" 0-11 0 "$bpw" "$OUT/logs-$arm-half1.log"
    local rc=$?
    sayp "arm $arm half-1 rc=$rc after $(( $(date +%s)-t0 ))s; exporting $dst"
    [[ $rc -ne 0 ]] && { sayp "arm $arm half-1 FAILED rc=$rc (see $OUT/logs-$arm-half1.log)"; kill $wp; return 1; }
    # halfway gate: fp32 mixed corpus KLD of the 12-layer checkpoint vs the union's PTQ
    RUNS=$W/stage8/runs EVALS=$OUT/eval TEACHER=$W/stage4/teacher STAGE4_ENV=$W/stage4.env \
        bash "$W/tree/experiments/inkling_stage4.sh" eval "$(echo "$name" | sed 's/^bpw//')" mixed \
        >>"$LOG" 2>&1
    local gate=$(python3 - "$OUT/eval/bpw${name#bpw}_mixed.json" "$UNION_MIXED" <<'PY' 2>/dev/null
import json,sys
try:
    d=json.load(open(sys.argv[1])); s=d["student"]
    kl=float(s["kl_mean"]); th=float(sys.argv[2])
    if int(d.get("windows",0)) != 32 or d.get("dtype") != "float32":
        print("BADINSTRUMENT expected 32 windows fp32")
    else:
        print("%.5f %s" % (kl, "PASS" if kl < th else "FAIL"))
except Exception:
    print("NA NA")
PY
)
    sayp "arm $arm halfway gate: mixed KLD $gate vs union $UNION_MIXED"
    case $gate in
        *PASS) : ;;
        *FAIL) sayp "arm $arm ABORTED at the halfway checkpoint (no KL improvement)"; kill $wp; return 1;;
        *) sayp "arm $arm ABORTED: the halfway instrument could not be read ($gate)"; kill $wp; return 1;;
    esac
    # half 2: layers 12-23 + the 200-step final logit-KL phase from qad/head_inputs.pt
    qad_supervised "$arm-half2" "$dst" "$work" 12-23 200 "$bpw" "$OUT/logs-$arm-half2.log"
    rc=$?
    if [[ $rc -ne 0 ]]; then
        sayp "arm $arm final phase rc=$rc (see $OUT/logs-$arm-half2.log)"; kill $wp; return 1
    fi
    if grep -qiE "final_kl.*skipped|skipped.*final_kl|fit check failed" "$OUT/logs-$arm-half2.log"; then
        sayp "arm $arm FINAL PHASE WAS SKIPPED -- refusing (Main: solve with head inputs, never skip)"; kill $wp; return 1
    fi
    sayp "arm $arm final phase done; $dst ready ($(( $(date +%s)-t0 ))s total)"
    for eset in mixed wiki pydocs; do
        RUNS=$W/stage8/runs EVALS=$OUT/eval TEACHER=$W/stage4/teacher STAGE4_ENV=$W/stage4.env \
            bash "$W/tree/experiments/inkling_stage4.sh" eval "$(echo "$name" | sed 's/^bpw//')" "$eset" \
            >>"$LOG" 2>&1 || say "eval $eset failed"
    done
    sayp "arm $arm fp32 evals done: $(ls $OUT/eval/bpw${name#bpw}_*.json 2>/dev/null | tr '\n' ' ')"
    kill $wp
    return 0
}

run_arm A 3.5-union-qad 3.5
run_arm B 3.0-union-qad 3.0
sayp "both arms finished; served KL / longs / paired panel / vision evals are the next stage"
