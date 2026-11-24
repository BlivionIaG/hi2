#!/usr/bin/env bash
# Stage 4 of the Inkling-2-Mini HI2 run: the calibration set, a pilot, the PTQ encodes and
# their KL against the stock model. Nothing starts by itself: every step is a subcommand, DRY=1
# prints the commands and runs none of them (the read-only checks still report).
#
# usage: [VAR=...] inkling_stage4.sh STEP [ARGS]
#   config            the resolved settings
#   check             read-only: disk, calibration manifest (production, SAMPLES), tokenizer
#   calib             (CPU) stage, build and verify the calibration set under $CAL
#   plan T            (CPU) `hi2 quantize --dry-run --json` at T bpw -> $LOGS/plan_T.json
#   pilot             (GPU) PILOT_SAMPLES windows through the whole chain at 3.5 bpw and a KL
#                     on PILOT_EVAL_WINDOWS windows of PILOT_EVAL_LEN tokens; own cache + workdir
#   quantize T        (GPU) calibration (the first call fills $CACHE, the others reuse it),
#                     allocation and encode at T bpw -> $RUNS/bpwT
#   eval T [SET]      (GPU) KL against the stock model on the held-out SET (mixed: with the
#                     per-layer-group breakdown, wiki, pydocs; fine: the mixed windows and their
#                     teacher pass, broken down by FINE_GROUPS; routing: the mixed windows with
#                     the routing log, no breakdown) -> $EVALS/bpwT_SET.json
#   teacher-heads     (GPU) the stock model's float32 head inputs on the FKL_TRAIN training windows
#                     of the final-KL phase -> $TEACHER/fkl_train_wN_float32.pt
#   finalkl T         (GPU) the final logit-KL phase from the finished output at T: FKL_STEPS steps
#                     of FKL_BATCH windows, held-out windows FKL_VAL_OFFSET.. of the mixed set
#                     (their teacher is the evaluation's) -> $RUNS/bpwT-fklSTEPSlrLR
#                     (a target may carry a tag after its bits: 3.5-p0.5 is $RUNS/bpw3.5-p0.5)
#   status            the state of the steps, the results, the disk
#   clean-pilot       remove the pilot's work, cache, output and teacher pass
#   tiny              (CPU, TINY=1) the toy model and its calibration set for the dry run
# Order on the real model: calib, check, plan 3.0/3.5/4.0, pilot, clean-pilot, quantize 3.5,
# eval 3.5, quantize 3.0, eval 3.0, quantize 4.0, eval 4.0. TINY=1 runs the same steps on the
# fork's tiny checkpoint on the CPU (`tiny` builds the toy model and its set instead of `calib`).
#
# The driver never takes a GPU or a lease: `gpu_run LABEL python-args...` runs the GPU steps
# and refuses unless HIP_VISIBLE_DEVICES is set; define it in $STAGE4_ENV (sourced last, it
# may redefine any function) to run a guard instead. `cpu_run python-args...` likewise.
# Settings (defaults in the code): W OUT SRC CAL CACHE WORK RUNS EVALS TEACHER STATE LOGS
# DEVICES DTYPE EVAL_DTYPE SAMPLES SEQ PILOT_SAMPLES EVAL_WINDOWS BREAKDOWN_WINDOWS FINE_GROUPS MIN_FREE_GB
# EVAL_GROUPS FKL_STEPS FKL_TRAIN FKL_VAL FKL_VAL_OFFSET FKL_BATCH FKL_LR FKL_EVALS
# PRODUCTION EXTRA_QUANTIZE MAX_KL TRANSFORMERS STAGE4_ENV DRY TINY.
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$HERE/.." && pwd)}
W=${W:-$HOME/w4a8_runs/inkling-hi2}
DRY=${DRY:-0}
TINY=${TINY:-0}
PY=${PY:-python3}
if [[ $TINY == 1 ]]; then # the dry run on the toy model: small, on the CPU, not production
    OUT=${OUT:-$W/stage4_tiny}
    SRC=${SRC:-$OUT/model}
    DEVICES=${DEVICES:-cpu} DTYPE=${DTYPE:-float32} SEQ=${SEQ:-64} SAMPLES=${SAMPLES:-8}
    EVAL_DTYPE=${EVAL_DTYPE:-float32}
    PRODUCTION=${PRODUCTION:-0} EVAL_WINDOWS=${EVAL_WINDOWS:-6}
    BREAKDOWN_WINDOWS=${BREAKDOWN_WINDOWS:-6} PILOT_EVAL_WINDOWS=${PILOT_EVAL_WINDOWS:-2}
    PILOT_EVAL_LEN=${PILOT_EVAL_LEN:-32} MIN_FREE_GB=${MIN_FREE_GB:-5}
    GIB_CACHE=${GIB_CACHE:-1} GIB_OUT=${GIB_OUT:-1}
else
    OUT=${OUT:-$W/stage4}
    SRC=${SRC:-$HOME/hfcache/inkling2-mini}
    DEVICES=${DEVICES:-cuda:0,cuda:1} DTYPE=${DTYPE:-float16} SEQ=${SEQ:-2048}
    EVAL_DTYPE=${EVAL_DTYPE:-float32} # float16 vs float32 on the same weights: KL 3e-3 (stage 2)
    SAMPLES=${SAMPLES:-256} PRODUCTION=${PRODUCTION:-1} EVAL_WINDOWS=${EVAL_WINDOWS:-32}
    BREAKDOWN_WINDOWS=${BREAKDOWN_WINDOWS:-8} PILOT_EVAL_WINDOWS=${PILOT_EVAL_WINDOWS:-4}
    PILOT_EVAL_LEN=${PILOT_EVAL_LEN:-512} MIN_FREE_GB=${MIN_FREE_GB:-60}
    GIB_CACHE=${GIB_CACHE:-40} GIB_OUT=${GIB_OUT:-16} # measured: 39.3 GiB, ~15 GiB
fi
PILOT_SAMPLES=${PILOT_SAMPLES:-4}
FINE_GROUPS=${FINE_GROUPS:-0-2,3-5,6-8,9-11,12-14,15-17,18-20,21-23} # `eval T fine`: groups of three layers
EVAL_GROUPS=${EVAL_GROUPS:-auto} # `eval T mixed`: the per-layer-group breakdown (none: the scores only)
# `finalkl`: steps, training windows, held-out windows (from FKL_VAL_OFFSET), windows per step, the
# ladder bytes' learning rate (8-bit levels per step; 0.03 is the preset's final_lr_p, the best of the
# one sweep that exists, 4B at 3.0 bpw; nothing was measured on Inkling: pass FKL_LR to compare), held-out evaluations
FKL_STEPS=${FKL_STEPS:-40} FKL_TRAIN=${FKL_TRAIN:-48} FKL_VAL=${FKL_VAL:-8} FKL_VAL_OFFSET=${FKL_VAL_OFFSET:-24}
FKL_BATCH=${FKL_BATCH:-1} FKL_LR=${FKL_LR:-0.03} FKL_EVALS=${FKL_EVALS:-4}
CAL=${CAL:-$OUT/calib}
CACHE=${CACHE:-$OUT/cache}
WORK=${WORK:-$OUT/work}
RUNS=${RUNS:-$OUT/runs}
EVALS=${EVALS:-$OUT/eval}
TEACHER=${TEACHER:-$OUT/teacher}
STATE=${STATE:-$OUT/state}
LOGS=${LOGS:-$OUT/logs}
DOCS=${DOCS:-$HOME/w4a8_runs/hadamard-int2/study_text_all.txt}
OOD_DIR=${OOD_DIR:-$HOME/w4a8_runs/cptc-qwen35-4b-20261119/inputs/confirmatory-20261120}
REAL_STAGED=${REAL_STAGED:-$W/stage4/calib/src}
EXTRA_QUANTIZE=${EXTRA_QUANTIZE:-}
MAX_KL=${MAX_KL:-}
TRANSFORMERS=${TRANSFORMERS:-}

say() { printf '[%s] %s\n' "$(date +%T)" "$*"; }
stop() {
    if ((DRY)); then say "(dry) would stop: $*"; else say "STOP: $*" >&2; exit 1; fi
}
die() { say "STOP: $*" >&2; exit 1; }
run() { # print the command, run it unless DRY
    printf '+'; printf ' %q' "$@"; printf '\n'
    ((DRY)) || "$@"
}
cpu_run() { PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH} nice -n 19 "$PY" "$@"; }
gpu_run() { # gpu_run LABEL python-args...
    local label=$1
    shift
    [[ -n ${HIP_VISIBLE_DEVICES:-} ]] ||
        die "$label: HIP_VISIBLE_DEVICES is not set; this driver never picks a GPU (define gpu_run in \$STAGE4_ENV)"
    PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH} "$PY" "$@"
}
devrun() { # devrun LABEL python-args...: the GPU wrapper for cuda devices, else the CPU one
    if [[ $DEVICES == *cuda* ]]; then gpu_run "$@"; else
        shift
        cpu_run "$@"
    fi
}

step() { # step NAME cmd...: the command with a log and a state file (exit, seconds)
    local name=$1 rc t0 start
    shift
    if ((DRY)); then
        run "$@"
        return 0
    fi
    mkdir -p "$LOGS" "$STATE"
    start=$(date -Is)
    t0=$SECONDS
    {
        printf '+'; printf ' %q' "$@"; printf '\n'
    } | tee -a "$LOGS/$name.log"
    "$@" 2>&1 | tee -a "$LOGS/$name.log"
    rc=${PIPESTATUS[0]}
    printf '{"step": "%s", "start": "%s", "end": "%s", "seconds": %d, "exit": %d}\n' \
        "$name" "$start" "$(date -Is)" $((SECONDS - t0)) "$rc" > "$STATE/$name.json"
    ((rc == 0)) || die "$name: exit $rc (log $LOGS/$name.log)"
}
locked() { # locked NAME cmd...: one at a time (GPU steps share one lock: they share the cards)
    local name=$1
    shift
    if ((DRY)); then "$@"; return; fi
    mkdir -p "$STATE"
    mkdir "$STATE/$name.running" 2> /dev/null ||
        die "$name is running, or died: remove $STATE/$name.running"
    # shellcheck disable=SC2064
    trap "rmdir '$STATE/$name.running' 2> /dev/null" EXIT
    "$@"
}
free_gib() { df -Pk "$1" | awk 'NR==2 {print int($4 / 1048576)}'; }
used_gib() { [[ -d $1 ]] && du -sk "$1" | awk '{print int($1 / 1048576)}' || echo 0; }
need_gib() { # need_gib GIB WHAT: GIB more are written and MIN_FREE_GB must stay free
    local path=$OUT free
    while [[ ! -d $path ]]; do path=$(dirname "$path"); done
    free=$(free_gib "$path")
    say "disk: $free GiB free on $path; $2 writes about $1 GiB and $MIN_FREE_GB GiB must stay free"
    ((free >= $1 + MIN_FREE_GB)) || stop "$2 needs $1 GiB on $path (+$MIN_FREE_GB spare), $free are free"
}
manifest() { # manifest KEY...: a value of $CAL/manifest.json
    [[ -f $CAL/manifest.json ]] || { echo "<$CAL/manifest.json>"; return; }
    "$PY" - "$CAL/manifest.json" "$@" << 'PY'
import json, sys
m = json.load(open(sys.argv[1]))
for k in sys.argv[2:]:
    m = m[k]
print(m)
PY
}
check_calibration() { # check_calibration SAMPLES [0]: SAMPLES windows of SEQ tokens (0: no token floor)
    local samples=$1 floor=${2:-1}
    local args=("$REPO/experiments/inkling_build_calib.py" verify --out "$CAL" --tokenizer "$SRC")
    ((PRODUCTION)) && args+=(--production)
    run cpu_run "${args[@]}" || stop "the calibration set under $CAL does not verify"
    [[ -f $CAL/manifest.json ]] || { stop "$CAL/manifest.json is missing (run: calib)"; return 0; }
    "$PY" - "$CAL/manifest.json" "$samples" "$SEQ" "$PRODUCTION" "$floor" << 'PY' || stop "the calibration set cannot serve $samples windows of $SEQ tokens"
import json, sys
m = json.load(open(sys.argv[1]))
n, seq, prod, floor = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "1", sys.argv[5] == "1"
c, bad = m["calibration"], []
if not m.get("ok"):
    bad.append("the build recorded a failed gate")
if prod and not m.get("thresholds", {}).get("production"):
    bad.append("not a production manifest")
if m["seq_len"] != seq:
    bad.append(f"windows of {m['seq_len']} tokens, not {seq}")
if n > c["pipeline_windows"]:
    bad.append(f"{n} windows asked, the build compared {c['pipeline_windows']} with the reader of hi2")
if n * seq > c["tokens"]:
    bad.append(f"{n * seq} tokens asked, the set has {c['tokens']}")
if prod and floor and n * seq < 262144:
    bad.append(f"{n * seq} tokens < 262144")
print("calibration: %d windows x %d = %d tokens of %d (%d windows, %s)" % (
    n, seq, n * seq, c["tokens"], c["windows"], "production" if m.get("thresholds", {}).get("production") else "dry run"))
if bad:
    print("calibration:", "; ".join(bad), file=sys.stderr)
    sys.exit(1)
PY
}
quantize_cmd() { # quantize_cmd DST WORKDIR CACHEDIR BPW SAMPLES -> Q
    Q=(-m hi2 quantize "$SRC" "$1" --calib "$CAL/calib.txt" --calib-samples "$5" --seq-len "$SEQ"
        --include '\.mlp\.experts\.' --bits 2,4,8 --candidates lean --target-bpw "$4"
        --no-qad --sensitivity none --devices "$DEVICES" --compute-dtype "$DTYPE"
        --workdir "$2" --cache-dir "$3")
    # shellcheck disable=SC2206
    Q+=($EXTRA_QUANTIZE)
}
eval_cmd() { # eval_cmd DST SET WINDOWS LEN GROUPS OUTNAME TEACHERNAME -> E
    local ids=$CAL/heldout_${2}_ids.npy
    # shellcheck disable=SC2206
    E=("$REPO/experiments/inkling_kl_eval.py" --src "$SRC" --dst "$1" --ids "$ids"
        --ids-sha256 "$(manifest heldout "$2" ids_sha256_all_windows)" --windows "$3"
        --devices ${DEVICES//,/ } --compute-dtype "$EVAL_DTYPE" --groups "$5"
        --teacher-cache "$TEACHER/$7.pt" --out "$EVALS/$6.json")
    [[ $4 == 0 ]] || E+=(--seq-len "$4")
    [[ $5 == none ]] || E+=(--breakdown-windows "$BREAKDOWN_WINDOWS")
    [[ -z $MAX_KL ]] || E+=(--max-kl "$MAX_KL")
    [[ -z $TRANSFORMERS ]] || E+=(--transformers "$TRANSFORMERS")
}
target() { # T: bits per weight, then optionally a tag for a variant of the encode (3.5-p0.5)
    [[ ${1:-} =~ ^[0-9]+(\.[0-9]+)?(-[A-Za-z0-9.]+)?$ ]] ||
        die "a target in bits per weight is needed (3.0, 3.5, 4.0; a tag may follow: 3.5-p0.5)"
}

step_config() {
    local v
    for v in REPO W OUT SRC CAL CACHE WORK RUNS EVALS TEACHER STATE LOGS DEVICES DTYPE EVAL_DTYPE SEQ SAMPLES \
        PILOT_SAMPLES PILOT_EVAL_WINDOWS PILOT_EVAL_LEN EVAL_WINDOWS BREAKDOWN_WINDOWS FINE_GROUPS EVAL_GROUPS \
        FKL_STEPS FKL_TRAIN FKL_VAL FKL_VAL_OFFSET FKL_BATCH FKL_LR FKL_EVALS \
        MIN_FREE_GB GIB_CACHE GIB_OUT PRODUCTION EXTRA_QUANTIZE MAX_KL TINY DRY; do
        printf '%-20s %s\n' "$v" "${!v}"
    done
}
step_check() {
    say "disk: $(free_gib "$(dirname "$OUT")") GiB free (spare to keep: $MIN_FREE_GB)"
    [[ -d $SRC ]] || stop "source directory $SRC missing"
    check_calibration "$SAMPLES"
}
step_calib() {
    local ood=()
    [[ -f $OOD_DIR/wiki_test.txt ]] && ood+=(--ood "wiki=$OOD_DIR/wiki_test.txt")
    [[ -f $OOD_DIR/python_docs.txt ]] && ood+=(--ood "pydocs=$OOD_DIR/python_docs.txt")
    need_gib 1 "the calibration set"
    local b=("$REPO/experiments/inkling_build_calib.py")
    step calib-stage cpu_run "${b[@]}" stage --out "$CAL" --tokenizer "$SRC" --docs "$DOCS"
    step calib-build cpu_run "${b[@]}" build --out "$CAL" --tokenizer "$SRC" ${ood[@]+"${ood[@]}"}
    check_calibration "$SAMPLES"
}
step_plan() {
    target "${1:-}"
    quantize_cmd "$RUNS/bpw$1" "$WORK/bpw$1" "$CACHE" "${1%%-*}" "$SAMPLES"
    if ((DRY)); then
        run cpu_run "${Q[@]}" --dry-run --json --devices cpu
        return 0
    fi
    mkdir -p "$LOGS"
    cpu_run "${Q[@]}" --dry-run --json --devices cpu > "$LOGS/plan_$1.json" 2> "$LOGS/plan_$1.err" ||
        die "plan $1: see $LOGS/plan_$1.err"
    say "plan $1: $LOGS/plan_$1.json"
}
step_pilot() {
    check_calibration "$PILOT_SAMPLES" 0 # a pilot is small on purpose
    need_gib $((GIB_CACHE + GIB_OUT)) "the pilot (statistics cache + output)"
    quantize_cmd "$RUNS/pilot" "$WORK/pilot" "$OUT/cache_pilot" 3.5 "$PILOT_SAMPLES"
    step pilot-quantize devrun pilot-quantize "${Q[@]}"
    eval_cmd "$RUNS/pilot" mixed "$PILOT_EVAL_WINDOWS" "$PILOT_EVAL_LEN" none pilot pilot_teacher
    step pilot-eval devrun pilot-eval "${E[@]}"
    say "pilot done: $(used_gib "$OUT") GiB under $OUT; clean-pilot removes the cache and the output"
}
step_quantize() {
    target "${1:-}"
    check_calibration "$SAMPLES"
    local have
    have=$(used_gib "$CACHE")
    ((have > GIB_CACHE)) && have=$GIB_CACHE
    need_gib $((GIB_CACHE - have + GIB_OUT)) "quantize $1 (statistics cache still to fill + output)"
    quantize_cmd "$RUNS/bpw$1" "$WORK/bpw$1" "$CACHE" "${1%%-*}" "$SAMPLES"
    step "quantize-$1" devrun "quantize-$1" "${Q[@]}"
}
step_eval() {
    target "${1:-}"
    local set=${2:-mixed} ids=${2:-mixed} groups=none
    case $set in
        mixed) groups=$EVAL_GROUPS ;;
        fine) ids=mixed groups=$FINE_GROUPS ;; # the held-out mixed windows, a finer breakdown
        routing) ids=mixed ;; # the held-out mixed windows, the scores and the routing log
    esac
    [[ -d $RUNS/bpw$1 ]] || stop "no output at $RUNS/bpw$1: run quantize $1 first"
    check_calibration "$SAMPLES"
    need_gib 2 "the evaluation"
    eval_cmd "$RUNS/bpw$1" "$ids" "$EVAL_WINDOWS" 0 "$groups" "bpw${1}_$set" \
        "${ids}_w${EVAL_WINDOWS}_${EVAL_DTYPE}"
    [[ $set != routing ]] || E+=(--routing-log)
    step "eval-$1-$set" devrun "eval-$1-$set" "${E[@]}"
}
step_teacher_heads() {
    check_calibration "$SAMPLES"
    need_gib 2 "the teacher's head inputs of the final-KL windows"
    step teacher-heads devrun teacher-heads "$REPO/experiments/inkling_final_kl.py" teacher --src "$SRC" \
        --calib "$CAL/calib.txt" --out "$TEACHER/fkl_train_w${FKL_TRAIN}_float32.pt" \
        --train-windows "$FKL_TRAIN" --seq-len "$SEQ" --devices ${DEVICES//,/ }
}
step_finalkl() {
    target "${1:-}"
    local heads=$TEACHER/fkl_train_w${FKL_TRAIN}_float32.pt held=$TEACHER/mixed_w${EVAL_WINDOWS}_${EVAL_DTYPE}.pt
    local name=bpw$1-fkl${FKL_STEPS}lr$FKL_LR cfg=$OUT/fkl_qad_${FKL_STEPS}_lr$FKL_LR.json C=() x
    [[ -d $RUNS/bpw$1 ]] || stop "no output at $RUNS/bpw$1: run quantize $1 first"
    [[ -f $heads ]] || stop "$heads is missing: run teacher-heads first"
    [[ -f $held ]] || stop "$held is missing: the teacher of the held-out windows (run eval $1 mixed first)"
    check_calibration "$SAMPLES"
    need_gib $((GIB_OUT * 2)) "the final-KL output (and the copy the phase keeps to resume from)"
    if ((!DRY)); then
        mkdir -p "$OUT"
        printf '{"final_batch": %s, "final_lr_p": %s, "val": %s, "final_evals": %s, "seed": 0}\n' \
            "$FKL_BATCH" "$FKL_LR" "$FKL_VAL" "$FKL_EVALS" > "$cfg"
    fi
    quantize_cmd "$RUNS/$name" "$WORK/$name" "$CACHE" "${1%%-*}" "$SAMPLES"
    for x in "${Q[@]:2}"; do # Q starts with `-m hi2`: the driver script is the entry point
        if [[ $x == --no-qad ]]; then
            C+=(--qad --qad-samples "$FKL_TRAIN" --qad-final-kl "$FKL_STEPS" --qad-config "$cfg")
        else
            C+=("$x")
        fi
    done
    step "finalkl-$1-lr$FKL_LR" devrun "finalkl-$1-lr$FKL_LR" "$REPO/experiments/inkling_final_kl.py" student \
        --init-from "$RUNS/bpw$1" --heads "$heads" --heldout-ids "$CAL/heldout_mixed_ids.npy" \
        --heldout-teacher "$held" --heldout-windows "$EVAL_WINDOWS" --val-offset "$FKL_VAL_OFFSET" \
        -- "${C[@]}"
}
step_status() {
    df -Ph "$(dirname "$OUT")" | tail -1
    local d f
    for d in "$CAL" "$CACHE" "$WORK" "$RUNS"/* "$TEACHER" "$EVALS"; do
        [[ -e $d ]] && du -sh "$d"
    done
    for f in "$STATE"/*.json; do [[ -f $f ]] && cat "$f"; done
    ls -d "$STATE"/*.running 2> /dev/null | sed 's/^/RUNNING: /'
    for f in "$EVALS"/*.json; do
        [[ -f $f ]] || continue
        "$PY" - "$f" << 'PY'
import json, sys
r = json.load(open(sys.argv[1]))
s = r.get("student")
sem = "n/a" if s and s["kl_sem"] is None else (s and "%.2g" % s["kl_sem"])
print(sys.argv[1].rsplit("/", 1)[-1], "ok" if r.get("ok") else "FAILED",
      "KL %.6g +- %s, top-1 %.4f, ppl x%.4f" % (s["kl_mean"], sem, s["top1_agreement"], s["ppl_ratio"]) if s else "",
      {k: round(v["kl_mean"], 6) for k, v in r.get("groups", {}).items()})
PY
    done
}
step_clean_pilot() {
    local p
    for p in "$RUNS/pilot" "$WORK/pilot" "$OUT/cache_pilot" "$TEACHER/pilot_teacher.pt" "$EVALS/pilot.json"; do
        [[ $p == "$OUT"/* ]] || die "refusing to remove $p: not under $OUT"
        [[ -e $p ]] && run rm -rf "$p"
    done
    return 0
}
step_tiny() {
    [[ $TINY == 1 ]] || die "tiny: set TINY=1"
    [[ -d $REAL_STAGED ]] || die "$REAL_STAGED: stage the real calibration set first (calib)"
    run mkdir -p "$CAL"
    step tiny-model cpu_run "$REPO/experiments/inkling_tiny_setup.py" --out "$SRC" --staged "$REAL_STAGED"
    run rm -rf "$CAL/src"
    run cp -r "$REAL_STAGED" "$CAL/src"
    step tiny-build cpu_run "$REPO/experiments/inkling_build_calib.py" build --out "$CAL" \
        --tokenizer "$SRC" --tokens 40000 --holdout-tokens 12000 --dry-run-scale "$SEQ" 8 4 8
    check_calibration "$SAMPLES"
}

# the environment file comes last: it may redefine cpu_run, gpu_run, devrun or any helper
# shellcheck disable=SC1090
[[ -z ${STAGE4_ENV:-} ]] || source "$STAGE4_ENV"

cmd=${1:-}
shift || true
case $cmd in
    config) step_config ;;
    check) step_check ;;
    calib) locked calib step_calib ;;
    plan) step_plan "$@" ;;
    pilot) locked gpu step_pilot ;;
    quantize) locked gpu step_quantize "$@" ;;
    eval) locked gpu step_eval "$@" ;;
    teacher-heads) locked gpu step_teacher_heads ;;
    finalkl) locked gpu step_finalkl "$@" ;;
    status) step_status ;;
    clean-pilot) step_clean_pilot ;;
    tiny) locked calib step_tiny ;;
    *) sed -n '2,/^set -u/p' "$0" | sed '$d'; [[ -z $cmd ]] || exit 2 ;;
esac
