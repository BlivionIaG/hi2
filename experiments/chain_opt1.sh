#!/usr/bin/env bash
# shellcheck disable=SC2086  # the calibration option strings are split on purpose
# Qwen3.5-0.8B, 3.0 bpw, sensitivity probe: calibration domain and act-order A/B. Three runs in
# parallel on HIP 2,3,4; each is evaluated on the repo-docs held-out text and on the mixed held-out text.
# usage: chain_opt1.sh   (box paths)
B=/home/chenco_adm/w4a8_runs/hadamard-int2
export MODEL=/home/chenco_adm/models/Qwen3.5-0.8B TEXT=$B/study_text_all.txt OUT=$B/runs35
R=$B/hi2/experiments/run_hi2_model.sh
V=/home/chenco_adm/Apps/vllm/venv-7.14.0_0.31.0-final; SL=$V/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$SL/_rocm_sdk_libraries/lib:$SL/_rocm_sdk_core/lib/host-math/lib:$SL/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SL/_rocm_sdk_core/lib/core/lib:$SL/torch/lib
DOCS="--calib $B/study_text_all.txt --calib-samples 128 --seq-len 2048"
MIX="--calib $B/calib_mix.txt --calib-samples 128 --seq-len 2048"
mixeval() { # tag gpu
  HIP_VISIBLE_DEVICES=$2 PYTHONPATH=$B/hi2 $V/bin/python -m hi2 eval "$MODEL" "$OUT/$1" --text $B/eval_mix.txt --skip-tokens 0 \
    --samples 8 --seq-len 2048 --device cuda:0 > "$OUT/$1.evalmix.log" 2>&1
}
( HIP_VISIBLE_DEVICES=2 $R q35-o1-mix cuda:0 $MIX --target-bpw 3.0 --sensitivity probe; mixeval q35-o1-mix 2 ) &
( HIP_VISIBLE_DEVICES=3 $R q35-o1-ao cuda:0 $DOCS --target-bpw 3.0 --sensitivity probe --act-order; mixeval q35-o1-ao 3 ) &
( HIP_VISIBLE_DEVICES=4 $R q35-o1-mix-ao cuda:0 $MIX --target-bpw 3.0 --sensitivity probe --act-order; mixeval q35-o1-mix-ao 4; mixeval q35-b3.0-sens 4 ) &
wait
echo "done" > $B/chain_opt1.done
