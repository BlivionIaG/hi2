# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Print the per-layer QAD results (held-out nMSE before/after, time, GPU peak) of a workdir.
usage: qad_layers.py <workdir>"""

import sys
from pathlib import Path

import torch

for p in sorted((Path(sys.argv[1]) / "qad").glob("layer*.pt")):
    info = torch.load(p, weights_only=False)["info"]
    if "val_before" not in info:
        print(f"{p.stem}: kept {info['kept']}")
        continue
    b, a = info["val_before"], info["val_after"]
    peak = info.get("gpu_peak_gib")
    print(
        f"{p.stem}: nMSE {b:.4e} -> {a:.4e} ({100 * (a / b - 1):+.1f}%) kept {info['kept']:7s} "
        f"{info['seconds']:6.0f} s" + (f" peak {peak:.2f} GiB" if peak else "")
    )
