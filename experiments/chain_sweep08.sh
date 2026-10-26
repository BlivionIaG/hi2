#!/usr/bin/env bash
# shellcheck disable=SC2086  # $MIX and $pids are word lists on purpose
# Qwen3.5-0.8B, quality-driven allocation sweep (mixed calibration, sensitivity probe, PTQ): one
# calibration + probe, then an allocation/encode per --max-error-ratio value on separate workdir
# copies; each result is scored on the docs and the mixed held-out text.
# usage: chain_sweep08.sh   (box paths; GPUs: HIP 3 and 4 only)
B=/home/chenco_adm/w4a8_runs/hadamard-int2
export MODEL=/home/chenco_adm/models/Qwen3.5-0.8B TEXT=$B/study_text_all.txt OUT=$B/runs35
R=$B/hi2-q/experiments/run_hi2_model.sh
V=/home/chenco_adm/Apps/vllm/venv-7.14.0_0.31.0-final; SL=$V/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$SL/_rocm_sdk_libraries/lib:$SL/_rocm_sdk_core/lib/host-math/lib:$SL/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SL/_rocm_sdk_core/lib/core/lib:$SL/torch/lib
MIX="--calib $B/calib_mix.txt --calib-samples 128 --seq-len 2048 --sensitivity probe"
W0=$OUT/sw08.work
mixeval() { # tag gpu
  HIP_VISIBLE_DEVICES=$2 PYTHONPATH=$B/hi2-q $V/bin/python -m hi2 eval "$MODEL" "$OUT/$1" --text "$B/eval_mix.txt" --skip-tokens 0 \
    --samples 8 --seq-len 2048 --device cuda:0 > "$OUT/$1.evalmix.log" 2>&1
}
run() { # ratio gpu workdir
  tag=sw08-r$1
  HIP_VISIBLE_DEVICES=$2 WORKDIR=$3 KEEP_HESSIAN=1 $R "$tag" cuda:0 $MIX --max-error-ratio "$1"
  mixeval "$tag" "$2"
}
run 6 3 "$W0"
pids=""
gpu=3
for r in 2 3 4 8 12; do
  rm -rf "$OUT/sw08-r$r.work"
  cp -a "$W0" "$OUT/sw08-r$r.work"
  rm -rf "$OUT/sw08-r$r.work/shards_done"
  run "$r" "$gpu" "$OUT/sw08-r$r.work" &
  pids="$pids $!"
  gpu=$((gpu + 1)); [ $gpu -gt 4 ] && { wait $pids; pids=""; gpu=3; }
done
wait
for r in 2 3 4 8 12; do rm -rf "$OUT/sw08-r$r.work"; done
echo "done" > "$B/chain_sweep08.done"
