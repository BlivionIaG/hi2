#!/usr/bin/env bash
# Score hi2 runs in llama.cpp with the same command as the GGUF baselines: export the
# dequantised model as an fp16 HF dir, convert to GGUF f16, run llama-perplexity.
# usage: HIP_VISIBLE_DEVICES=3 HI2_PATH=<tools/hi2> gguf_eval_dense.sh <hf_model_dir> <runs_dir> <gguf_dir> TAG...
set -u
M=$1; RUNS=$2; G=$3; shift 3
D=/home/chenco_adm/rocmfpx-rdna2; BIN=$D/build-gfx1030/bin
V=/home/chenco_adm/Apps/vllm/venv-7.14.0_0.31.0-final
export LD_LIBRARY_PATH=$BIN:/opt/rocm/core-7.14/lib
for TAG in "$@"; do
  PYTHONPATH=${HI2_PATH:-.} $V/bin/python -m hi2 export-dense "$M" "$RUNS/$TAG" "$G/dense-$TAG" > "$G/dense-$TAG.export.log" 2>&1
  PYTHONPATH=$D/gguf-py $V/bin/python $D/convert_hf_to_gguf.py "$G/dense-$TAG" --outtype f16 --outfile "$G/dense-$TAG.gguf" > "$G/dense-$TAG.convert.log" 2>&1
  $BIN/llama-perplexity -m "$G/dense-$TAG.gguf" -f "$G/eval.txt" -c 2048 --chunks 8 -ngl 99 > "$G/dense-$TAG.ppl.log" 2>&1
  rm -rf "$G/dense-$TAG" "$G/dense-$TAG.gguf"   # keep only the logs; the dense copies are large
done
echo "done" > "$G/DENSE_DONE"
