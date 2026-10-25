#!/usr/bin/env bash
# llama.cpp scoring suite for GGUF files: perplexity, KL divergence against the f16 model, and the
# four multiple-choice tasks (the .bin files of the llama.cpp validation dataset), same flags for
# every file. Writes logs under <out>/<tag>.*.log and a summary line file <out>/<tag>.summary.txt.
#
#   gguf_suite.sh base  <hf_model> <gguf_f16> <eval_text> <data_dir> <out_dir>   # f16 + base logits
#   gguf_suite.sh score <gguf> <tag> <eval_text> <data_dir> <out_dir>            # any GGUF
# MC_TASKS (default 2000) bounds HellaSwag and MMLU; ARC sets are used whole.
set -u
D=/home/chenco_adm/rocmfpx-rdna2; BIN=$D/build-gfx1030/bin
export LD_LIBRARY_PATH=$BIN:/opt/rocm/core-7.14/lib
MODE=$1; shift
mc() { # gguf tag data out
  local g=$1 t=$2 data=$3 out=$4
  for spec in "arc-easy-validation 570" "arc-challenge-validation 299" "hellaswag-validation ${MC_TASKS:-2000}" "mmlu-test ${MC_TASKS:-2000}"; do
    read -r name tasks <<< "$spec"
    $BIN/llama-perplexity -m "$g" --multiple-choice -bf "$data/$name.bin" --multiple-choice-tasks "$tasks" -c 2048 -ngl 99 > "$out/$t.mc-$name.log" 2>&1
  done
}
score() { # gguf tag text data out
  local g=$1 t=$2 text=$3 data=$4 out=$5
  mkdir -p "$out"
  $BIN/llama-perplexity -m "$g" -f "$text" -c 2048 --chunks 8 -ngl 99 --kl-divergence --kl-divergence-base "$out/base.logits" > "$out/$t.kld.log" 2>&1
  mc "$g" "$t" "$data" "$out"
  echo "done" > "$out/$t.score.done"
}
case $MODE in
  base)
    M=$1; G=$2; TEXT=$3; DATA=$4; OUT=$5; mkdir -p "$OUT"
    V=/home/chenco_adm/Apps/vllm/venv-7.14.0_0.31.0-final
    PYTHONPATH=$D/gguf-py $V/bin/python $D/convert_hf_to_gguf.py "$M" --outtype f16 --outfile "$G" > "$OUT/f16.convert.log" 2>&1
    $BIN/llama-perplexity -m "$G" -f "$TEXT" -c 2048 --chunks 8 -ngl 99 --save-all-logits "$OUT/base.logits" > "$OUT/f16.ppl.log" 2>&1
    mc "$G" f16 "$DATA" "$OUT"
    echo "done" > "$OUT/f16.score.done"
    ;;
  score)
    score "$@"
    ;;
esac
