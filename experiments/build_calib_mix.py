# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Build a mixed-domain calibration text and a disjoint held-out text from datasets already in the
HF cache (chat, instructions, tool calling) plus a slice of the repo-docs corpus.
usage: build_calib_mix.py <docs_text> <out_calib> <out_eval>
Calibration and evaluation records are disjoint: records are split by index parity of a block of 50."""

import glob
import json
import os
import sys

H = os.path.expanduser("~/.cache/huggingface/hub")


def read(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()


def write(path, text):
    with open(path, "w") as f:
        f.write(text)


def first(pat):
    return sorted(glob.glob(os.path.join(H, pat)))[0]


def guanaco():
    path = first(
        "datasets--timdettmers--openassistant-guanaco/snapshots/*/"
        "openassistant_best_replies_train.jsonl"
    )
    for line in read(path).splitlines():
        yield json.loads(line)["text"]


def wizard():
    path = first(
        "datasets--WizardLM--WizardLM_evol_instruct_70k/snapshots/*/"
        "alpaca_evol_instruct_70k.json"
    )
    for r in json.loads(read(path)):
        yield f"### Instruction:\n{r['instruction']}\n\n### Response:\n{r['output']}"


def toolace():
    path = first("datasets--Team-ACE--ToolACE/snapshots/*/data.json")
    for r in json.loads(read(path)):
        conv = "\n".join(f"{m['from']}: {m['value']}" for m in r["conversations"])
        yield f"{r['system']}\n{conv}"


def take(gen, calib_bytes, eval_bytes):
    cal, ev, nc, ne = [], [], 0, 0
    for i, t in enumerate(gen):
        if (i // 50) % 5 == 4:  # one block in five is held out
            if ne < eval_bytes:
                ev.append(t)
                ne += len(t)
        elif nc < calib_bytes:
            cal.append(t)
            nc += len(t)
        if nc >= calib_bytes and ne >= eval_bytes:
            break
    return cal, ev


docs = read(sys.argv[1])
# the quantizer calibrates on the FIRST tokens of the file, so each source gets an equal
# share of the first ~1.3 MB (about 300k tokens) rather than being appended one after another
cal_parts = [docs[:450_000]]
ev_parts = []
for gen, c, e in (
    (guanaco(), 220_000, 60_000),
    (wizard(), 220_000, 60_000),
    (toolace(), 450_000, 120_000),
):
    c_, e_ = take(gen, c, e)
    cal_parts.append("\n\n".join(c_))
    ev_parts.append("\n\n".join(e_))
write(sys.argv[2], "\n\n".join(cal_parts))
write(sys.argv[3], "\n\n".join(ev_parts))
print("calib bytes", sum(map(len, cal_parts)), "eval bytes", sum(map(len, ev_parts)))
