#!/usr/bin/env bash
# Serve each model in the fork's vLLM and run the bench suite against it; the per-item results
# land in $ROOT/TAG.json (compare them with bench/compare.py).
# usage: SERVE_VALIDATE=<fork>/tools/rdna/port_v031/serve_validate.sh GPU=0 TASKS=gsm8k,mc \
#          bench/serve_run.sh TAG=MODEL_DIR [TAG=MODEL_DIR ...]
#   GPU is one device or a list (GPU=0,1,2,3 TP=4); ROOT=<results dir> (default bench_runs),
#   PORT (default 18731), MAXLEN (default 40000: the 32k needle items need 33k), LIMIT per task,
#   WORKERS, EXTRA (more run.py flags), PY (python of the bench client), REF_RESULTS (the
#   reference run's json: score its completions teacher-forced into TAG.lp.json; leave unset
#   for the reference arm itself) and TF_LIMIT (items per task, default 200).
#   TF_LOGPROB_GAP=0 omits teacher-forced scoring; it is enabled by default.
#   TF_TASKS selects chat tasks for scoring; leave unset to use the reference's tasks.
set -uo pipefail
SERVE_VALIDATE=${SERVE_VALIDATE:?set SERVE_VALIDATE to the serve_validate.sh of the fork}
GPU=${GPU:?GPU (HIP device list) required}
TP=${TP:-1}
PORT=${PORT:-18731}
ROOT=${ROOT:-bench_runs}
MAXLEN=${MAXLEN:-40000}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$ROOT" || exit $?
if (($# == 0)); then
  echo "usage: bench/serve_run.sh TAG=MODEL_DIR [TAG=MODEL_DIR ...]" >&2
  exit 2
fi
export TASKS LIMIT WORKERS EXTRA PY REF_RESULTS TF_LIMIT TF_LOGPROB_GAP TF_TASKS
status=0
for arm in "$@"; do
  tag=${arm%%=*}
  model=${arm#*=}
  if [[ $arm != *=* || -z $tag || -z $model || $tag == */* || $tag == "." || $tag == ".." ]]; then
    echo "invalid arm: $arm (expected TAG=MODEL_DIR, TAG without slashes)" >&2
    exit 2
  fi
  attempt="$$-$RANDOM-$SECONDS"
  arm_status=0
  echo "[$(date +%H:%M:%S)] arm $tag $model GPU=$GPU"
  RECIPE=full MODEL=$model GPUS=$GPU PORT=$PORT TAG=$tag OUT=$ROOT/serve-$tag \
    CACHE=$ROOT/cache-$tag CELLS=none CONC_PROBE=0 PREFIX_PROBE=0 \
    LOAD_HOOK="$HERE/hook.sh" BENCH_ATTEMPT="$attempt" \
    bash "$SERVE_VALIDATE" TP="$TP" MAXLEN="$MAXLEN" SEQS=16 KV=6000000000 TUNABLEOP=1 \
    || arm_status=$?
  # Some external serve wrappers swallow LOAD_HOOK's exit status. Require the marker
  # written by this invocation, not PASS text or an earlier arm's stale files.
  hook_attempt= hook_status= hook_stage=
  if [[ -f "$ROOT/serve-$tag/bench.exit" ]]; then
    read -r hook_attempt hook_status hook_stage < "$ROOT/serve-$tag/bench.exit"
  fi
  if [[ $hook_attempt != "$attempt" || ! $hook_status =~ ^[0-9]+$ ]]; then
    echo "$tag: missing completion marker from benchmark hook" >&2
    if ((arm_status == 0)); then arm_status=1; fi
  elif ((hook_status != 0 && arm_status == 0)); then
    arm_status=$hook_status
  fi
  if [[ $hook_attempt == "$attempt" ]]; then
    if [[ -f "$ROOT/serve-$tag/bench.json" ]]; then
      cp "$ROOT/serve-$tag/bench.json" "$ROOT/$tag.json" || arm_status=$?
    else
      echo "$tag: benchmark report missing" >&2
      if ((arm_status == 0)); then arm_status=1; fi
    fi
    if [[ $hook_stage == teacher_forced ]]; then
      if [[ -f "$ROOT/serve-$tag/lp.json" ]]; then
        cp "$ROOT/serve-$tag/lp.json" "$ROOT/$tag.lp.json" || arm_status=$?
      else
        echo "$tag: teacher-forced report missing" >&2
        if ((arm_status == 0)); then arm_status=1; fi
      fi
    fi
  fi
  echo "[$(date +%H:%M:%S)] arm $tag exit=$arm_status"
  if ((arm_status != 0 && status == 0)); then status=$arm_status; fi
done
exit "$status"
