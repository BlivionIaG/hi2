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
set -u
SERVE_VALIDATE=${SERVE_VALIDATE:?set SERVE_VALIDATE to the fork's serve_validate.sh}
GPU=${GPU:?GPU (HIP device list) required}
TP=${TP:-1}
PORT=${PORT:-18731}
ROOT=${ROOT:-bench_runs}
MAXLEN=${MAXLEN:-40000}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$ROOT"
export TASKS LIMIT WORKERS EXTRA PY REF_RESULTS TF_LIMIT
for arm in "$@"; do
  tag=${arm%%=*}
  model=${arm#*=}
  echo "[$(date +%H:%M:%S)] arm $tag $model GPU=$GPU"
  RECIPE=full MODEL=$model GPUS=$GPU PORT=$PORT TAG=$tag OUT=$ROOT/serve-$tag \
    CACHE=$ROOT/cache-$tag CELLS=none CONC_PROBE=0 PREFIX_PROBE=0 \
    LOAD_HOOK="$HERE/hook.sh" \
    bash "$SERVE_VALIDATE" TP="$TP" MAXLEN="$MAXLEN" SEQS=16 KV=6000000000 TUNABLEOP=1
  cp "$ROOT/serve-$tag/bench.json" "$ROOT/$tag.json" 2> /dev/null
  cp "$ROOT/serve-$tag/lp.json" "$ROOT/$tag.lp.json" 2> /dev/null
  echo "[$(date +%H:%M:%S)] arm $tag done: $(tail -1 "$ROOT/serve-$tag/load_hook.txt" 2> /dev/null)"
done
