# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""The final logit-KL phase on a finished HI2 output of Inkling-2-Mini (stage 5).

``hi2 quantize --qad-final-kl N`` runs the layer-wise phase (for a looped model: the PTQ of
every layer), then ``hi2.finalkl.final_kl``: the quantized student's logits against the
teacher's, training the per-block ladder bytes and the norms. Here the layer-wise phase is
replaced by the tensors of a finished output (``--init-from``: the stage-4 run), so the phase
starts from the very model whose KL was measured, and everything else is the pipeline's:
phase 1 from the cache, the allocation, the fit check, the phase (``pipeline._final_kl``),
the checkpoint writer, the validation. The windows differ from the pipeline's in one way: the
held-out windows of the phase (keep-best, the held-out KL of the log) are windows of the
held-out set of the KL evaluation, with its cached teacher, instead of windows of the
calibration text.

Two steps, each its own process (the host holds one model at a time; the teacher's program is
float32, the student is float16):

    inkling_final_kl.py teacher --src STOCK --calib calib.txt --out heads.pt \\
        --train-windows 48 --devices cuda:0 cuda:1
        the stock model's head inputs (float32, the experts decoded exactly: the teacher of the
        KL evaluation) on the training windows, drawn from the calibration text as the
        pipeline's QAD sampling does

    inkling_final_kl.py student --init-from DIR --heads heads.pt \\
        --heldout-ids heldout_mixed_ids.npy --heldout-teacher teacher/mixed_w32_float32.pt \\
        -- quantize STOCK DST --calib calib.txt --qad --qad-samples 48 --qad-final-kl 40 ...
        ``hi2 quantize`` with its layer-wise phase replaced; the arguments after ``--`` are
        the command's own (the stage-4 quantize command with ``--qad ...`` for ``--no-qad``)
"""

import argparse
import functools
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def _log(*m) -> None:
    print(*m, flush=True)


def kl_eval():
    """``inkling_kl_eval.py`` as a module: its ``Experts`` (the stock decode of the teacher)
    and ``fingerprint`` (what a head-inputs file depends on) are the evaluation's own."""
    name = "inkling_kl_eval"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / "inkling_kl_eval.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def train_windows(src, calib, seq_len: int, n: int, seed: int) -> torch.Tensor:
    """``[n, seq_len]`` windows drawn from the calibration text as the pipeline's QAD
    sampling (``sampling = "random"``) does."""
    from transformers import AutoTokenizer

    from hi2.pipeline import sample_calibration_windows

    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=False)
    return sample_calibration_windows(calib, tok, n, seq_len, seed, False)


def read_output_tensors(dst) -> dict:
    """The HI2 tensors of a finished output by checkpoint name, the dictionary the layer-wise
    phase hands to ``final_kl``."""
    from hi2.stack import Hi2Checkpoint

    ck = Hi2Checkpoint(dst)
    return {f"{base}.weight": ck.get(base) for base in sorted(ck.logical())}


def window_layout(n_train: int, n_val: int):
    """The indices of the training and the held-out windows among the ``n_train + n_val``
    rows of the phase: ``hi2.qad.split_indices``, which spreads the held-out ones."""
    from hi2.qad import QadConfig, split_indices

    return split_indices(n_train + n_val, QadConfig(val=n_val))


def cmd_teacher(a) -> int:
    from hi2 import looped
    from hi2.executor import Executor
    from hi2.inkling_source import InklingSource
    from hi2.pipeline import RunConfig, _load_model

    ke = kl_eval()
    train = train_windows(a.src, [a.calib], a.seq_len, a.train_windows, a.seed)
    fp = ke.fingerprint(a.src, train.numpy(), "float32")
    if a.out.exists():
        blob = torch.load(a.out, map_location="cpu", weights_only=True)
        if blob.get("fingerprint") == fp:
            _log(f"{a.out}: the teacher's head inputs of these windows are there")
            return 0
    devices = [torch.device(d) for d in a.devices]
    decode = next((d for d in devices if d.type == "cuda"), torch.device("cpu"))
    rc = RunConfig(
        src=a.src,
        dst=a.out.parent / "unused",
        workdir=a.out.parent / "work-unused",
        compute_dtype="float32",
        devices=a.devices,
    )
    t0 = time.perf_counter()
    with InklingSource(a.src, device=decode) as src:
        model = _load_model(rc, _log)
        experts = ke.Experts(model, src, None, decode, _log)
        experts.set(experts.all("teacher"))
        _log(f"model and stock experts: {time.perf_counter() - t0:.0f} s")
        heads = looped.teacher_head_inputs(model, train, Executor(a.devices), _log)
    if not torch.isfinite(heads).all():
        _log("STOP: the teacher's head inputs are not finite")
        return 1
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name(f".{a.out.name}.tmp")
    torch.save({"fingerprint": fp, "heads": heads}, tmp)
    tmp.replace(a.out)
    _log(
        f"{a.out}: {tuple(heads.shape)} float32, {heads.numel() * 4 / 2**30:.2f} GiB, "
        f"{time.perf_counter() - t0:.0f} s"
    )
    return 0


def phase_inputs(a, rc, qc):
    """``(ids, head_inputs)`` of the phase, ``[n, T]`` token ids and ``[n, T, H]`` float32:
    the training windows at the places ``split_indices`` gives them with the teacher's head
    inputs of ``teacher``, the held-out windows (``--val-offset`` onwards in the held-out set)
    with the cached teacher of the evaluation. A teacher file made for other windows is
    refused."""
    ke = kl_eval()
    n_train, n_val = qc.samples, qc.val
    train_idx, val_idx = window_layout(n_train, n_val)
    train = train_windows(rc.src, rc.calib, rc.seq_len, n_train, qc.seed)
    blob = torch.load(a.heads, map_location="cpu", weights_only=True)
    if blob.get("fingerprint") != ke.fingerprint(rc.src, train.numpy(), "float32"):
        raise SystemExit(
            f"{a.heads} was made for other training windows (source, text, seed, count or "
            f"length): run `teacher` with --train-windows {n_train}, --seq-len {rc.seq_len}"
        )
    held = np.load(a.heldout_ids)
    want = ke.fingerprint(rc.src, held[: a.heldout_windows], "float32")
    cached = torch.load(a.heldout_teacher, map_location="cpu", weights_only=True)
    if cached.get("fingerprint") != want:
        raise SystemExit(
            f"{a.heldout_teacher} is not the teacher of the first {a.heldout_windows} "
            f"windows of {a.heldout_ids} (float32)"
        )
    if a.val_offset + n_val > a.heldout_windows:
        raise SystemExit(
            f"{n_val} held-out windows from {a.val_offset} do not fit the first "
            f"{a.heldout_windows} windows of the held-out set"
        )
    heads = torch.empty(
        (n_train + n_val, *blob["heads"].shape[1:]), dtype=torch.float32
    )
    ids = torch.empty((n_train + n_val, rc.seq_len), dtype=torch.long)
    heads[train_idx] = blob["heads"]
    ids[train_idx] = train
    cut = slice(a.val_offset, a.val_offset + n_val)
    heads[val_idx] = cached["heads"][cut].float()
    ids[val_idx] = torch.from_numpy(held[cut].astype(np.int64))
    return ids, heads


def resume_key(init_from, ids: torch.Tensor, heads_name: str) -> str:
    """What the phase's resume copy depends on: the finished output it starts from (its shards
    and their sizes), the windows of the phase and the teacher file of its training windows."""
    import hashlib

    from hi2.qad import fingerprint

    shards = sorted(Path(init_from).glob("model*.safetensors"))
    return fingerprint(
        {
            "init": [(p.name, p.stat().st_size) for p in shards],
            "ids": hashlib.sha1(ids.contiguous().numpy().tobytes()).hexdigest(),
            "heads": heads_name,
        }
    )


def qad_from_output(
    a, rc, ex, tables, picks, sens, log, target_bpw=None, ratio=None, tag=""
):
    """The layer-wise phase of ``pipeline._qad_encode``, replaced: the tensors of the finished
    output, then the pipeline's own fit check and final-KL phase. Returns ``(tensors, dense)``."""
    import hi2.pipeline as P

    qc = rc.qad
    if qc.final_kl <= 0:
        raise SystemExit("--qad-final-kl N (N > 0) is what this driver runs")
    log(f"final KL from the finished output {a.init_from}")
    tensors = read_output_tensors(a.init_from)
    log(f"  {len(tensors)} HI2 tensors read")
    ids, heads = phase_inputs(a, rc, qc)
    model = P._load_model(rc, log)
    if not P._final_kl_fits(rc, ex, model, tensors, True, log, ids.shape[1]):
        raise SystemExit(
            "the final-KL student does not fit the devices (the line above): nothing was written"
        )
    fp = resume_key(a.init_from, ids, a.heads.name)
    return P._final_kl(rc, ex, model, ids, tensors, {}, fp, log, tag, heads)


def cmd_student(a, hi2_argv: list[str]) -> int:
    import hi2.pipeline as P
    from hi2.__main__ import main

    if not hi2_argv:
        raise SystemExit("the hi2 command goes after `--`: -- quantize SRC DST ...")
    P._qad_encode = functools.partial(qad_from_output, a)
    return main(hi2_argv)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser(
        "teacher", help="the teacher's head inputs on the training windows"
    )
    t.add_argument("--src", type=Path, required=True, help="the stock model directory")
    t.add_argument("--calib", type=Path, required=True, help="the calibration text")
    t.add_argument("--out", type=Path, required=True, help="the head inputs, .pt")
    t.add_argument("--train-windows", type=int, required=True)
    t.add_argument("--seq-len", type=int, default=2048)
    t.add_argument("--seed", type=int, default=0, help="the QAD seed of the run")
    t.add_argument("--devices", nargs="+", default=["cpu"])
    s = sub.add_parser(
        "student", help="hi2 quantize with the layer-wise phase replaced"
    )
    s.add_argument("--init-from", type=Path, required=True, help="the finished output")
    s.add_argument("--heads", type=Path, required=True, help="`teacher`'s file")
    s.add_argument(
        "--heldout-ids", type=Path, required=True, help="held-out windows, .npy"
    )
    s.add_argument(
        "--heldout-teacher", type=Path, required=True, help="their teacher cache"
    )
    s.add_argument(
        "--heldout-windows", type=int, default=32, help="windows the cache covers"
    )
    s.add_argument(
        "--val-offset", type=int, default=24, help="first held-out window used"
    )
    return ap


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    rest: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, rest = argv[:i], argv[i + 1 :]
    a = parser().parse_args(argv)
    if a.cmd == "teacher":
        return cmd_teacher(a)
    return cmd_student(a, rest)


if __name__ == "__main__":
    sys.exit(main())
