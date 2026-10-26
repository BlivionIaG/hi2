#!/usr/bin/env bash
# Queue low-bit study runs on one GPU, one after the other (lowbit_run.sh, WAIT=1).
# usage: chain_lowbit.sh HIP "<tag|hi2 quantize args>" ...   (box paths, from lowbit/)
#   COMMON = arguments shared by every run (default: calib_qad.txt, 128 x 2048, probe);
#   a spec "tag|args|SHARE_QAD=<workdir>" also copies that run's QAD layer artifacts.
#   MODEL / OUT / KEEP_WORK pass through to lowbit_run.sh.
set -u
HIP=$1; shift
B=/home/chenco_adm/w4a8_runs/hadamard-int2
HERE=$(cd "$(dirname "$0")" && pwd)
COMMON=${COMMON:---calib $B/calib_qad.txt --calib-samples 128 --seq-len 2048 --sensitivity probe}
for spec in "$@"; do
  tag=${spec%%|*}; rest=${spec#*|}; args=${rest%%|*}; extra=""
  [ "$rest" != "$args" ] && extra=${rest#*|}
  # shellcheck disable=SC2086  # option strings are split on purpose
  env WAIT=1 $extra bash "$HERE/lowbit_run.sh" "$HIP" "$tag" $COMMON $args
done
