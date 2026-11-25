# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Targeted mutants of the layered-QAD groundwork and controls.

Each mutant edits one source string in place, runs one test that must fail, restores
the file, and exits non-zero if any mutant survives. Run on the build box's CPU:

    TREE=$HOME/w4a8_runs/inkling-hi2/tree-s PYPATH_EXTRA=$HOME/w4a8_runs/pydeps/tfm-main \
        ./cpu.sh experiments/qad_looped_mutants.py

The suite is deliberately narrow: every mutation is a real behaviour the change fixed
(the init-from per-expert mapping and dtype, accumulation, clipping and the saved
per-step trace).
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def mutants():
    q = ROOT / "hi2" / "qad.py"
    LOOP = ROOT / "hi2" / "qad_looped.py"
    loop = ROOT / "tests" / "test_qad_looped.py"
    unit = ROOT / "tests" / "test_qad.py"
    TEST = loop
    return [
        (
            "M1 init-from reads no tensors from the PTQ output",
            LOOP,
            "        ptq = _ptq_inits(qc.init_from, linears) if qc.init_from else {}",
            "        ptq = {}",
            f"{TEST}::test_steps_zero_qad_experts_are_bit_identical_to_ptq",
        ),
        (
            "M2 the QAD initialiser converts the source to float32",
            q,
            "w = wstager.to_host(layer.get_submodule(name).weight.detach())",
            "w = wstager.to_host(layer.get_submodule(name).weight.detach().float())",
            f"{unit}::test_the_ptq_initializer_uses_the_qad_config_and_source_dtype",
        ),
        (
            "M3 accumulation draws one micro-batch",
            q,
            "need = batch * accum",
            "need = batch",
            f"{unit}::test_accumulated_micro_batches_keep_whole_permutations",
        ),
        (
            "M4 gradient clipping is never applied",
            q,
            "if qc.clip > 0 and grad_norm > qc.clip:",
            "if False and grad_norm > qc.clip:",
            f"{unit}::test_gradient_clipping_changes_the_update_and_is_traced",
        ),
        (
            "M5 the per-step trace records no gradient norm",
            q,
            '"step_trace": step_trace,',
            '"step_trace": [],',
            f"{unit}::test_gradient_clipping_changes_the_update_and_is_traced",
        ),
        (
            "M6 the PTQ fallback re-encodes the compensated weights",
            q,
            """        fb = {
            name: FakeQuantLinear(
                dequantize(inits[name]).float(),
                inits[name],
                None,
                "weights",
                "abs",
            )
            for name in names
        }""",
            """        fb = {
            name: FakeQuantLinear(weight(name), inits[name], comps[name], "ptq")
            for name in names
        }""",
            f"{TEST}::test_a_ptq_fallback_is_the_ptq_tensor_not_a_re_encoding",
        ),
    ]


def run_one(label, path, old, new, test):
    source = path.read_text()
    if source.count(old) != 1:
        return f"{label}: source string occurs {source.count(old)} times"
    path.write_text(source.replace(old, new))
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--no-header", test],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
    finally:
        path.write_text(source)
    if proc.returncode == 0:
        return f"{label}: SURVIVED ({test})"
    tail = [line for line in proc.stdout.splitlines() if line.startswith("FAILED")]
    return f"{label}: killed ({tail[-1] if tail else 'exit ' + str(proc.returncode)})"


def main():
    bad = []
    for label, path, old, new, test in mutants():
        result = run_one(label, path, old, new, test)
        print(result)
        if "SURVIVED" in result or "occurs" in result:
            bad.append(result)
    if bad:
        print("\n".join(bad), file=sys.stderr)
        return 1
    print(f"all {len(mutants())} mutants killed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
