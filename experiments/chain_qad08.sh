#!/usr/bin/env bash
# shellcheck disable=SC2086  # option strings are split on purpose
# QAD tuning runs (sensitivity probe, quality allocation, calibration = calib_qad.txt: the mixed
# calibration text first, then the docs calibration region; the PTQ phase reads its head, QAD
# draws random windows from the whole file). Every run hard-links phase 1 (tables, Hessians,
# sensitivity) from SHARE, a finished run with the same calibration settings, so only the
# allocation and QAD run. Runs queue per GPU (mkdir locks).
# usage: chain_qad08.sh <share workdir> <ratio> "<gpus>" <round tag> "<tag|args>" ...   (box paths)
#   e.g. chain_qad08.sh $B/runs35/q35-ptq-mixq.work 5 "4 5" r1 "def|--qad" "lr3e-5|--qad --qad-lr 3e-5"
# MODEL / OUT default to Qwen3.5-0.8B / runs35. QAD is on by default: a PTQ row needs "tag|--no-qad".
set -u
B=/home/chenco_adm/w4a8_runs/hadamard-int2
SHARE=$1; RATIO=$2; read -ra GV <<< "$3"; RT=$4; shift 4
export MODEL=${MODEL:-/home/chenco_adm/models/Qwen3.5-0.8B} OUT=${OUT:-$B/runs35}
R=$B/qad/experiments/run_qad_model.sh
COMMON="--calib $B/calib_qad.txt --calib-samples 128 --seq-len 2048 --sensitivity probe --max-error-ratio $RATIO"
i=0
for spec in "$@"; do
  tag=${spec%%|*}; args=${spec#*|}
  gpu=${GV[$((i % ${#GV[@]}))]}
  ( until mkdir "$B/qad.gpu$gpu.lock" 2> /dev/null; do sleep 30; done
    HIP_VISIBLE_DEVICES=$gpu SHARE_FROM=$SHARE KEEP_WORK=1 $R "q-$RT-$tag" $COMMON $args
    rm -rf "$OUT/q-$RT-$tag.work/hessian"
    rmdir "$B/qad.gpu$gpu.lock" ) &
  i=$((i + 1))
  sleep 2
done
wait
echo "finished" > "$OUT/chain-$RT.done"
