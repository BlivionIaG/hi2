#!/usr/bin/env bash
# shellcheck disable=SC2086  # the calibration option strings are split on purpose
# Qwen3.5-0.8B, 3.0 bpw, sensitivity probe: calibration domain and act-order A/B, second run with the calibration mix placed at the file head. Three runs in
# parallel on HIP 2,3,4; each is evaluated on the repo-docs held-out text and on the mixed held-out text.
# usage: chain_opt2.sh   (box paths)
B=/home/chenco_adm/w4a8_runs/hadamard-int2
export MODEL=/home/chenco_adm/models/Qwen3.5-0.8B TEXT=$B/study_text_all.txt OUT=$B/runs35
R=$B/hi2-q/experiments/run_hi2_model.sh
V=/home/chenco_adm/Apps/vllm/venv-7.14.0_0.31.0-final; SL=$V/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$SL/_rocm_sdk_libraries/lib:$SL/_rocm_sdk_core/lib/host-math/lib:$SL/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SL/_rocm_sdk_core/lib/core/lib:$SL/torch/lib
MIX="--calib $B/calib_mix.txt --calib-samples 128 --seq-len 2048"
mixeval() { # tag gpu
  HIP_VISIBLE_DEVICES=$2 PYTHONPATH=$B/hi2-q $V/bin/python -m hi2 eval "$MODEL" "$OUT/$1" --text $B/eval_mix.txt --skip-tokens 0 \
    --samples 8 --seq-len 2048 --device cuda:0 > "$OUT/$1.evalmix.log" 2>&1
}
( HIP_VISIBLE_DEVICES=2 $R q35-o2-mix cuda:0 $MIX --target-bpw 3.0 --sensitivity probe; mixeval q35-o2-mix 2 ) &
( HIP_VISIBLE_DEVICES=3 $R q35-o2-mix-ao cuda:0 $MIX --target-bpw 3.0 --sensitivity probe --act-order; mixeval q35-o2-mix-ao 3 ) &
wait
echo "done" > $B/chain_opt2.done
