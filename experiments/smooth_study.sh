#!/usr/bin/env bash
# Per-tensor smoothing exponent study (spec 8.5, accepted by 8.10): PTQ with GPTQ rounding and the
# sensitivity probe on Qwen3.5-0.8B and Qwen2.5-0.5B at 3.0 and 4.5 bpw, the single exponent 0.5
# against a per-tensor choice from {0.25, 0.5, 0.75, 1.0}. One chain per GPU; each run waits for
# its GPU (lowbit_run.sh, WAIT=1) and scores docs and mix held-out text with `hi2 eval`.
# usage: smooth_study.sh HIP_A HIP_B HIP_C        (from the rsynced copy; results in runssmooth/)
set -u
B=/home/chenco_adm/w4a8_runs/hadamard-int2
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$B/runssmooth
mkdir -p "$OUT"
M08=/home/chenco_adm/models/Qwen3.5-0.8B
M05=$(ls -d /home/chenco_adm/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/* | head -1)
COMMON="--calib $B/calib_mix.txt --calib-samples 128 --seq-len 2048 --sensitivity probe --no-qad"
SEARCH="--smooth-search 0.25,0.75,1.0"
chain() {  # chain HIP MODEL TAGPREFIX BPW...
  local hip=$1 model=$2 pre=$3; shift 3
  for bpw in "$@"; do
    MODEL=$model OUT=$OUT COMMON="$COMMON --target-bpw $bpw" \
      bash "$HERE/chain_lowbit.sh" "$hip" "$pre-$bpw-base|" "$pre-$bpw-search|$SEARCH"
  done
}
(chain "$1" "$M08" s08 3.0 > "$OUT/chain-$1.out" 2>&1 &)
(chain "$2" "$M08" s08 4.5 > "$OUT/chain-$2.out" 2>&1 &)
(chain "$3" "$M05" s05 3.0 4.5 > "$OUT/chain-$3.out" 2>&1 &)
