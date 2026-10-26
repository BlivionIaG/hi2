#!/usr/bin/env bash
# shellcheck disable=SC2086  # argument lists on purpose
# Before/after timing runs of the hi2 speed work (docs/rdna2/hi2-speed.md): each job
# waits until one of our GPUs (HIP 2-5) is idle and unlocked, then runs on it.
# usage: chain_fast.sh JOB...   (jobs: see the case list below)
B=/home/chenco_adm/w4a8_runs/hadamard-int2
HERE=$(cd "$(dirname "$0")" && pwd)
M08=/home/chenco_adm/models/Qwen3.5-0.8B
M4B=$(cat $B/m4b.path)
for MOE in /home/chenco_adm/.cache/huggingface/hub/models--allenai--OLMoE-1B-7B-0924/snapshots/*; do break; done
S08="--sensitivity probe --max-error-ratio 6"
S4B="--sensitivity probe --target-bpw 3.0"
gpu() {  # wait for an idle GPU, print its HIP index
  while :; do
    h=$(bash "$HERE/free_gpu.sh")
    [ -n "$h" ] && { echo "$h"; return; }
    sleep 90
  done
}
for job in "$@"; do
  h=$(gpu)
  echo "$(date +%T) $job on HIP $h" >> $B/runsfast/chain_fast.log
  case $job in
    n08)     "$HERE/fast_ptq.sh" $h n08-r6 $M08 $B/calib_mix.txt $S08 ;;
    n08warm) "$HERE/fast_ptq.sh" $h n08-r6-warm $M08 $B/calib_mix.txt $S08 ;;
    o08)     CODE=$B/hi2-fast-base "$HERE/fast_ptq.sh" $h o08-r6 $M08 $B/calib_mix.txt $S08 ;;
    n4b)     "$HERE/fast_ptq.sh" $h n4b-b3.0 $M4B $B/study_text_all.txt $S4B ;;
    o4b)     CODE=$B/hi2-fast-base "$HERE/fast_ptq.sh" $h o4b-b3.0 $M4B $B/study_text_all.txt $S4B ;;
    n08c2)   "$HERE/fast_ptq.sh" $h n08-r6-cold2 $M08 $B/calib_mix.txt $S08 --cache-dir $B/runsfast/cache-c2 ;;
    n08s4)   "$HERE/fast_ptq.sh" $h n08-r6-s4x256 $M08 $B/calib_mix.txt $S08 --sens-seqs 4 --sens-len 256 ;;
    n08s2)   "$HERE/fast_ptq.sh" $h n08-r6-s2x256 $M08 $B/calib_mix.txt $S08 --sens-seqs 2 --sens-len 256 ;;
    n08b8)   "$HERE/fast_ptq.sh" $h n08-r6-b8 $M08 $B/calib_mix.txt $S08 --sens-batch 8 ;;
    n08lean) "$HERE/fast_ptq.sh" $h n08-r6-lean $M08 $B/calib_mix.txt $S08 --candidates lean ;;
    n08f16)  "$HERE/fast_ptq.sh" $h n08-r6-f16 $M08 $B/calib_mix.txt $S08 --cache-fp16-above 2048 ;;
    n08multi) NO_EVAL=1 "$HERE/fast_ptq.sh" $h n08-multi $M08 $B/calib_mix.txt --sensitivity probe --max-error-ratio 2,3,4,6,8,12 ;;
    gputests) HIP_VISIBLE_DEVICES=$h "$HERE/fast_run.sh" $h gputests -- -m pytest -q -p no:cacheprovider "$HERE/../tests/test_gptq_fast.py" ;;
    chk)     for w in n08-r6 n4b-b3.0 nmoe-b3.0; do
               m=$M08; case $w in n4b*) m=$M4B;; nmoe*) m=$MOE;; esac
               "$HERE/fast_run.sh" $h chk-$w -- "$HERE/check_tables.py" $m $B/runsfast/$w.work
             done ;;
    moe)     NO_EVAL=1 CALIB_SAMPLES=178 "$HERE/fast_ptq.sh" $h nmoe-b3.0 $MOE $B/calib_mix.txt --target-bpw 3.0 --sensitivity probe --sens-dtype float16 ;;
  esac
  echo "$(date +%T) $job done rc=$?" >> $B/runsfast/chain_fast.log
done
