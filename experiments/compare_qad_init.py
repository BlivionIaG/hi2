# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Compare two QAD runs layer by layer (``<workdir>/qad/layerNNN.pt``): for layers kept
at their PTQ encoding (not trained) the tensors must be identical, for trained layers the
report says whether they are. Also compares the final checkpoints' shards.

usage: python experiments/compare_qad_init.py WORKDIR_A WORKDIR_B [OUT_A OUT_B]
"""

import sys
from pathlib import Path

import torch

from hi2.format import FIELDS


def main():
    a, b = Path(sys.argv[1]), Path(sys.argv[2])
    for pa in sorted((a / "qad").glob("layer*.pt")):
        ra = torch.load(pa, weights_only=False)
        rb = torch.load(b / "qad" / pa.name, weights_only=False)
        same = ra["tensors"].keys() == rb["tensors"].keys() and all(
            torch.equal(getattr(ra["tensors"][n], f), getattr(rb["tensors"][n], f))
            for n in ra["tensors"]
            for f in FIELDS
        )
        print(f"{pa.name}: kept {ra['info'].get('kept')}, identical={same}")
    if len(sys.argv) > 4:
        oa, ob = Path(sys.argv[3]), Path(sys.argv[4])
        for s in sorted(oa.glob("*.safetensors")):
            print(
                s.name,
                "identical"
                if s.read_bytes() == (ob / s.name).read_bytes()
                else "DIFFERS",
            )


if __name__ == "__main__":
    main()
