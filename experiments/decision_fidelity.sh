#!/usr/bin/env bash
# Serve each model in the fork's vLLM and run the decision-fidelity prompt set against it
# (decision_fidelity.py); the decisions land in $ROOT/TAG.json.
# usage: SERVE_VALIDATE=<fork>/tools/rdna/port_v031/serve_validate.sh GPU=0 \
#          decision_fidelity.sh TAG=MODEL_DIR [TAG=MODEL_DIR ...]
#   GPU is one device, or a list (GPU=0,1,2,3 TP=4) for a model that needs tensor parallelism;
#   ROOT=<results dir> (default decision), PORT (default 18731).
# Compare afterwards: decision_fidelity.py compare $ROOT/a.json $ROOT/b.json ...
set -u
SERVE_VALIDATE=${SERVE_VALIDATE:?set SERVE_VALIDATE to the serve_validate.sh of the fork}
GPU=${GPU:?GPU (HIP device list) required}
TP=${TP:-1}
PORT=${PORT:-18731}
ROOT=${ROOT:-decision}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$ROOT"
for arm in "$@"; do
  tag=${arm%%=*}
  model=${arm#*=}
  echo "[$(date +%H:%M:%S)] arm $tag $model GPU=$GPU"
  RECIPE=full MODEL=$model GPUS=$GPU PORT=$PORT TAG=$tag OUT=$ROOT/serve-$tag \
    CACHE=$ROOT/cache-$tag CELLS=none CONC_PROBE=0 PREFIX_PROBE=0 \
    LOAD_HOOK="$HERE/decision_fidelity_hook.sh" \
    bash "$SERVE_VALIDATE" TP="$TP" MAXLEN=32768 SEQS=4 KV=6000000000 TUNABLEOP=1
  cp "$ROOT/serve-$tag/decisions.json" "$ROOT/$tag.json" 2> /dev/null
  echo "[$(date +%H:%M:%S)] arm $tag done: $(tail -1 "$ROOT/serve-$tag/load_hook.txt" 2> /dev/null)"
done
