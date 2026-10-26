#!/usr/bin/env bash
# Download a Hugging Face model into the shared HF cache (no --local-dir), log under the runs dir.
# usage: download_hf.sh <repo id> [extra huggingface_hub.snapshot_download patterns...]
set -u
REPO=$1
V=/home/chenco_adm/Apps/vllm/venv-7.14.0_0.31.0-final
LOG=/home/chenco_adm/w4a8_runs/hadamard-int2/dl-$(echo "$REPO" | tr '/' '_').log
"$V/bin/python" - "$REPO" > "$LOG" 2>&1 <<'PY'
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(sys.argv[1], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py"])
print("DOWNLOADED", p)
PY
echo "exit=$?" >> "$LOG"
