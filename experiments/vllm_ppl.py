# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Perplexity of a model served by vLLM, via prompt_logprobs, on the same token slice
as ``hi2 eval``. Lets models that only vLLM can run (EXL3 on this fork) be compared
with the dense and HI2-exported models through one engine.

    python vllm_ppl.py --port 18291 --model-dir <dir with tokenizer> --served-name NAME \
        --text FILE --skip-tokens 800000 --samples 8 --seq-len 2048
"""

import argparse
import json
import math
import urllib.request
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--served-name", required=True)
    ap.add_argument("--text", type=Path, required=True)
    ap.add_argument("--skip-tokens", type=int, default=800000)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=2048)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model_dir)
    ids = tok(a.text.read_text(), add_special_tokens=False).input_ids[a.skip_tokens :]
    need = a.samples * a.seq_len
    assert len(ids) >= need, (len(ids), need)
    nll, n = 0.0, 0
    for s in range(a.samples):
        seq = ids[s * a.seq_len : (s + 1) * a.seq_len]
        body = json.dumps(
            {
                "model": a.served_name,
                "prompt": seq,
                "max_tokens": 1,
                "temperature": 0,
                "prompt_logprobs": 0,
            }
        ).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{a.port}/v1/completions",
            body,
            {"Content-Type": "application/json"},
        )
        out = json.loads(urllib.request.urlopen(req, timeout=900).read())
        plp = out["choices"][0]["prompt_logprobs"]
        for tid, entry in zip(seq[1:], plp[1:]):
            nll -= entry[str(tid)]["logprob"]
            n += 1
        print(f"seq {s + 1}/{a.samples}", flush=True)
    print(json.dumps({"tokens": n, "ppl": math.exp(nll / n)}))


if __name__ == "__main__":
    main()
