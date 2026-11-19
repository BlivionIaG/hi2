#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU/GPU throughput proof for the CPTC encoder and QAD projection.

CPU mode (default, safe anywhere):
    python bench/cptc_microbench.py --cpu --out bench/results/cptc-cpu.json

Guarded GPU mode (only after the parent grants HIP0/a2): the wrapper checks the
explicit mask, native PCI identity and V620, the lease holder, and free VRAM, then
runs the worker in its own process group under the approved FaultMonitor and
KernelMonitor plus thermal_guard.py (95 C stop, 90 C 5 s mean).  The worker itself
refuses a visible device it was not asked for.
"""
import argparse
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

EXPECTED_PCI = "0000:d1:00.0"
LEASE = Path.home() / "w4a8_runs/lanes/lease/CPTCAllocation.a2"


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod


def _projection_probe(args, torch, dev) -> int:
    """Read-only replay of the phase-1 candidate errors for the preserved dump."""
    from hi2.cptc import decode_index, get_codec, window_indices

    dump = args.probe_dump
    if dump is None:
        dump = Path.home() / "w4a8_runs/cptc-qwen35-4b-20261119/micro/projection-counterexample/counterexample-K2-B64.pt"
    d = torch.load(dump, map_location="cpu", weights_only=False)
    K = int(d["K"])
    depth = int(d["depth"])
    B = int(d["B"])
    row = 849
    codec = get_codec(depth, K)
    span = (depth + K - 1) // K
    period = 1 << (span - 1).bit_length()
    into = torch.arange(span, device=dev)
    shift = into * K
    keep = ~(((1 << K) - 1) << shift)
    mask = (1 << depth) - 1

    # State entering phase 1: phase 0 positions follow the reference; rest initial.
    init = d["init"][row].to(dev).long()
    ref = d["reference"][row].to(dev).long()
    codes = init.clone()
    codes[torch.arange(0, B, period, device=dev)] = ref[torch.arange(0, B, period, device=dev)]
    pos = torch.arange(1, B, period, device=dev)
    affected = (pos[:, None] - into[None, :]) % B
    q = window_indices(codes.unsqueeze(0), K, depth, B)[0]
    qa = q[affected]
    w = d["w"][row].float().to(dev)[affected]
    s = d["scales"].reshape(-1)[row].float().to(dev)
    lut = d["lut"].float().to(dev)

    errs = {}
    terms = {}
    for sym in range(1 << K):
        qc = ((qa & keep) | (sym << shift)) & mask
        idx = decode_index(qc, codec.a0, codec.a1, codec.tap_q)
        vals = (s * lut[idx]).to(torch.float16).float()
        t = (vals - w) ** 2
        terms[sym] = t
        errs[sym] = t.sum(-1)
    col = 33
    k = int((col - 1) // period)
    torch.cuda.synchronize()
    out = {
        "row": row, "col": col, "phase": 1, "position_index": k,
        "terms_sym1": terms[1][k].detach().cpu().tolist(),
        "terms_sym2": terms[2][k].detach().cpu().tolist(),
        "gpu_err_sym1": float(errs[1][k]),
        "gpu_err_sym2": float(errs[2][k]),
        "gpu_err2_minus_err1": float(errs[2][k] - errs[1][k]),
        "double64_err_sym1": float(terms[1][k].detach().cpu().double().sum()),
        "double64_err_sym2": float(terms[2][k].detach().cpu().double().sum()),
    }
    left1 = torch.zeros((), dtype=torch.float32)
    left2 = torch.zeros((), dtype=torch.float32)
    tcpu1 = terms[1][k].detach().cpu()
    tcpu2 = terms[2][k].detach().cpu()
    for j in range(tcpu1.numel()):
        left1 = left1 + tcpu1[j]
        left2 = left2 + tcpu2[j]
    out["left_to_right_fp32_err_sym1"] = float(left1)
    out["left_to_right_fp32_err_sym2"] = float(left2)
    out["left_to_right_fp32_err2_minus_err1"] = float(left2 - left1)
    out["all_gpu_errs"] = {
        str(sym): errs[sym].detach().cpu().tolist() for sym in range(1 << K)
    }
    out["torch_version"] = torch.__version__
    out["dump"] = str(dump)
    import json
    print(json.dumps(out, indent=2), flush=True)
    return 0


def _worker(args) -> int:
    if os.environ.get("HIP_VISIBLE_DEVICES") != str(args.hip):
        raise SystemExit("worker visibility changed")
    import torch

    dev = torch.device("cuda", 0)
    if getattr(args, "probe", False):
        return _projection_probe(args, torch, dev)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hi2.cptc import encode_rows_cptc, get_codec

    rows = {}
    exact_cpu_match = True
    torch.manual_seed(1234)
    base = torch.randn(args.rows, 64)
    for K in (2, 4):
        codec = get_codec(16, K)
        for B in (32, 64):
            n = args.rows
            x = base[:, :B].contiguous().to(dev)
            d = torch.ones(B, device=dev)
            print(f"encode cpu start K{K} B{B} n{n}", flush=True)
            cpu_codes, _, _ = encode_rows_cptc(
                x.cpu(), d.cpu(), K, B, codec, method="viterbi", effort="fast", sweeps=0
            )
            print(f"encode gpu start K{K} B{B}", flush=True)
            gpu_codes, gpu_scales, _ = encode_rows_cptc(
                x, d, K, B, codec, method="viterbi", effort="fast", sweeps=0
            )
            print(f"encode gpu done K{K} B{B}", flush=True)
            exact_cpu_match = exact_cpu_match and torch.equal(cpu_codes, gpu_codes.cpu())
            torch.cuda.synchronize()
            print(f"encode compare done K{K} B{B}", flush=True)
            codes, scales = gpu_codes, gpu_scales
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            codes, scales, sse = encode_rows_cptc(
                x, d, K, B, codec, method="viterbi", effort="fast", sweeps=0
            )
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            rows[f"K{K}_B{B}"] = {
                "values": int(n * B), "seconds": round(dt, 6),
                "values_per_s": round(n * B / dt, 1),
                "sse": float(sse.sum()),
                "finite": bool(torch.isfinite(codes.float()).all()),
            }
    # GPTQ block path (triangular solve) on the same fixed tensor.
    from hi2.cptc import encode_rows_cptc_gptq

    u = torch.triu(torch.randn(64, 64, device=dev) * 0.1)
    u.diagonal().copy_(torch.rand(64, device=dev) + 0.5)
    K, B = 4, 64
    codec = get_codec(16, K)
    gptq_codes, gptq_scales, gptq_sse, _ = encode_rows_cptc_gptq(
        base.to(dev), torch.ones(64, device=dev), u, K, B, codec,
        effort="fast", sweeps=0,
    )
    cpu_gptq = encode_rows_cptc_gptq(
        base, torch.ones(64), u.cpu(), K, B, codec, effort="fast", sweeps=0
    )[0]
    gptq_cpu_match = torch.equal(gptq_codes.cpu(), cpu_gptq)

    # QAD projection throughput (one sweep, the training forward path).
    from hi2.cptc import project_codes

    projection = {}
    for K in (2, 4):
        codec = get_codec(16, K)
        for B in (32, 64):
            x = torch.randn(args.projection_rows, B, device=dev)
            init = torch.randint(
                0, 1 << K, (args.projection_rows, B), device=dev, dtype=torch.uint8
            )
            s = torch.full(
                (args.projection_rows,), 0.02, device=dev, dtype=torch.float16
            )
            print(f"projection reference start K{K} B{B} rows{args.projection_rows}", flush=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            ref = project_codes(x, codec, B, init, s, sweeps=1)
            torch.cuda.synchronize()
            dtp = time.perf_counter() - t0
            print(f"projection reference done K{K} B{B} {dtp:.3f}s", flush=True)
            from hi2.cptc import index_table_value
            from hi2.cptc_projection_kernel import project_codes_cuda

            lut = index_table_value(
                torch.arange(127, device=dev), codec.table.to(dev), codec.table_scale
            )
            print(f"projection fused start K{K} B{B}", flush=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fused = project_codes_cuda(
                x, init, s, lut, codec.a0, codec.a1, K, 16, B, sweeps=1
            )
            torch.cuda.synchronize()
            dtf = time.perf_counter() - t0
            print(f"projection fused done K{K} B{B} {dtf:.3f}s match={torch.equal(ref, fused)}", flush=True)
            match = bool(torch.equal(ref, fused))
            projection[f"K{K}_B{B}"] = {
                "values": int(args.projection_rows * B),
                "seconds": round(dtp, 6),
                "values_per_s": round(args.projection_rows * B / dtp, 1),
                "fused_seconds": round(dtf, 6),
                "fused_values_per_s": round(args.projection_rows * B / dtf, 1),
                "fused_match": match,
            }
            if not match and args.save_projection_dir is not None:
                from pathlib import Path as _Path
                import json as _json
                outd = _Path(args.save_projection_dir)
                outd.mkdir(parents=True, exist_ok=True)
                bad = (ref != fused).any(-1).nonzero().flatten()
                torch.save(
                    {
                        "w": x.cpu(), "init": init.cpu(), "scales": s.cpu(),
                        "lut": lut.cpu(), "reference": ref.cpu(), "fused": fused.cpu(),
                        "K": K, "depth": 16, "B": B, "sweeps": 1,
                        "failing_rows": bad.cpu(), "seed": 1234,
                    },
                    outd / f"counterexample-K{K}-B{B}.pt",
                )
                from hi2.cptc import block_values as _block_values
                refv = _block_values(ref, codec, s, B).float()
                fusedv = _block_values(fused, codec, s, B).float()
                (outd / f"counterexample-K{K}-B{B}.json").write_text(_json.dumps({
                    "seed": 1234, "K": K, "depth": 16, "B": B, "sweeps": 1,
                    "rows": int(args.projection_rows), "failing_rows": bad.tolist(),
                    "ref_sse": float((refv - x).pow(2).sum()),
                    "fused_sse": float((fusedv - x).pow(2).sum()),
                    "first_diffs": [
                        {"row": int(r), "ref": ref[r].tolist(), "fused": fused[r].tolist()}
                        for r in bad[:3]
                    ],
                }, indent=2))
    out = {
        "kind": "gpu-worker", "device": torch.cuda.get_device_name(0),
        "exact_cpu_match": bool(exact_cpu_match),
        "gptq_cpu_match": bool(gptq_cpu_match),
        "gptq": {"sse": float(gptq_sse.sum()), "finite": bool(torch.isfinite(gptq_scales).all())},
        "encode": rows,
        "projection": projection,
        "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
    }
    print(json.dumps(out, indent=2))
    return 0


def _cpu(args) -> int:
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hi2.cptc import encode_rows_cptc, get_codec, project_codes

    out = {"kind": "cpu", "encode": {}, "projection": {}}
    for K in (2, 4):
        codec = get_codec(16, K)
        for B in (32, 64):
            n = args.rows
            x = torch.randn(n, B)
            d = torch.ones(B)
            t0 = time.perf_counter()
            encode_rows_cptc(x, d, K, B, codec, method="viterbi", effort="fast", sweeps=0)
            dt = time.perf_counter() - t0
            out["encode"][f"K{K}_B{B}"] = {
                "values": n * B, "seconds": round(dt, 4),
                "values_per_s": round(n * B / dt, 1),
            }
    K, B = 4, 64
    codec = get_codec(16, K)
    x = torch.randn(args.projection_rows, B)
    init = torch.randint(0, 1 << K, (args.projection_rows, B), dtype=torch.uint8)
    s = torch.full((args.projection_rows,), 0.02, dtype=torch.float16)
    t0 = time.perf_counter()
    project_codes(x, codec, B, init, s, sweeps=1)
    dt = time.perf_counter() - t0
    out["projection"]["K4_B64"] = {
        "values": args.projection_rows * B, "seconds": round(dt, 4),
        "values_per_s": round(args.projection_rows * B / dt, 1),
    }
    out["torch"] = torch.__version__
    print(json.dumps(out, indent=2))
    return 0


def _guarded(args) -> int:
    """Run the worker through the approved isolate_fault.run_step, not a new loop."""
    if args.cwd is not None:
        repo = args.cwd
    else:
        repo = Path(__file__).resolve().parents[1]
    if os.environ.get("HIP_VISIBLE_DEVICES") not in (None, str(args.hip)):
        raise SystemExit(
            f"ambient HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')!r} "
            f"does not match {args.hip}"
        )
    if os.environ.get("ROCR_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise SystemExit("refusing: ROCR/CUDA visibility is set")
    if not args.lease_holder.is_file():
        raise SystemExit(f"refusing: lease holder missing: {args.lease_holder}")
    lease = args.lease_holder.read_text(errors="replace")
    if "a2" not in lease.lower():
        raise SystemExit(f"refusing: lease holder does not name a2: {lease!r}")
    drivers = args.monitor_dir
    for name in ("isolate_fault.py", "run_matrix.py", "verify_env.py", "thermal_guard.py"):
        if not (drivers / name).is_file():
            raise SystemExit(f"refusing: approved driver missing: {drivers / name}")
    isolate = _load(drivers / "isolate_fault.py")
    # SIGTERM/SIGHUP must unwind through run_step's finally, not bypass cleanup.
    isolate._install_signal_handlers()
    # Take the kernel baseline before the verify child; a preflight fault must not be
    # baselined away by reading dmesg after that child.
    dmesg_text, attempts = isolate.dmesg_read()
    lines = dmesg_text.splitlines()
    dmesg_base_count = len(lines)
    dmesg_base_tail = lines[-1] if lines else None
    # Identity and free-VRAM proof happen in a child (no persistent torch context here).
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(args.hip)
    env.pop("ROCR_VISIBLE_DEVICES", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["PYTHONPATH"] = ""
    py = Path(isolate.PY).expanduser()
    env["VIRTUAL_ENV"] = str(py.parent.parent)
    env["PATH"] = str(py.parent) + os.pathsep + env.get("PATH", "")
    verify = subprocess.run(
        [str(isolate.PY), str(drivers / "verify_env.py"), "--no-hash",
         "--no-tensor-check", "--json",
         "--expected-hip-index", str(args.hip), "--expected-pci", args.expected_pci],
        env=env, text=True, capture_output=True, timeout=120,
    )
    if verify.returncode != 0:
        raise SystemExit(f"refusing: verify_env child failed:\n{verify.stdout}\n{verify.stderr}")
    identity = json.loads(verify.stdout[verify.stdout.index("{"):])
    if not identity.get("ok"):
        raise SystemExit(f"refusing: identity not ok: {identity.get('errors')}")
    gpu = identity.get("gpu") or {}
    if gpu.get("mem_free_gib", 0) < args.min_free_gib:
        raise SystemExit(
            f"refusing: free VRAM {gpu.get('mem_free_gib')} GiB < {args.min_free_gib} GiB"
        )
    # Compare the post-verify dmesg against the pre-verify baseline, exactly as the
    # approved isolate caller does; a new fault here refuses before run_step.
    check = isolate.pre_step_dmesg_check(dmesg_base_count, dmesg_base_tail)
    if not check["ok"]:
        raise SystemExit(f"refusing: pre-step dmesg: {check.get('error') or check}")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(args.python), str(Path(__file__).resolve()), "--worker",
           "--hip", str(args.hip), "--rows", str(args.rows),
           "--projection-rows", str(args.projection_rows)]
    if args.probe:
        cmd += ["--probe"]
        if args.probe_dump is not None:
            cmd += ["--probe-dump", str(args.probe_dump)]
    if args.save_projection_dir is not None:
        cmd += ["--save-projection-dir", str(args.save_projection_dir)]
    ld = args.ld_library_path or os.environ.get("LD_LIBRARY_PATH", "")
    overrides = {
        "HIP_VISIBLE_DEVICES": str(args.hip),
        "PYTHONPATH": str(repo),
        "VIRTUAL_ENV": str(Path(args.python).expanduser().parent.parent),
        "OMP_NUM_THREADS": "4",
        "HSA_ENABLE_SDMA": "0",
        "HSA_FORCE_FINE_GRAIN_PCIE": "1",
        "GPU_MAX_HW_QUEUES": "2",
        "TORCH_BLAS_PREFER_HIPBLASLT": "0",
        "PYTORCH_TUNABLEOP_ENABLED": "0",
        "PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED": "0",
    }
    if ld:
        overrides["LD_LIBRARY_PATH"] = ld
    result = isolate.run_step(
        name="cptc-microbench", sequence=1, cmd=cmd, cwd=repo,
        timeout_s=args.max_seconds, stamp=stamp,
        dmesg_before_count=check["after_count"],
        dmesg_before_tail=check["after_tail"],
        parse="auto", env_overrides=overrides,
        thermal_cmd=[
            str(isolate.PY), str(drivers / "thermal_guard.py"),
            "--expected-pci", args.expected_pci,
        ],
    )
    text = Path(result["log"]).read_text(errors="replace")
    payload = None
    start = text.find("{")
    if start >= 0:
        try:
            payload = json.loads(text[start:])
        except json.JSONDecodeError:
            payload = None
    report = {
        "kind": "guarded-gpu", "preflight": identity, "attempts": attempts,
        "command": cmd, "run_step": result, "worker": payload,
    }
    proj_ok = bool(
        payload is not None
        and payload.get("projection")
        and all(v.get("fused_match") is True for v in payload["projection"].values())
    )
    report["ok"] = bool(
        result.get("status") == "pass" and payload is not None
    ) if args.probe else bool(
        result.get("status") == "pass" and payload is not None
        and payload.get("exact_cpu_match") is True
        and payload.get("gptq_cpu_match") is True
        and proj_ok
    )
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--guarded", action="store_true")
    ap.add_argument("--hip", type=int, default=0)
    ap.add_argument("--expected-pci", default=EXPECTED_PCI)
    ap.add_argument("--lease-holder", type=Path, default=LEASE)
    ap.add_argument("--monitor-dir", type=Path,
                    default=Path.home() / "w4a8_runs/triton-pr12747/full-amd/drivers")
    ap.add_argument("--python", type=Path, default=Path(sys.executable))
    ap.add_argument("--cwd", type=Path, default=None)
    ap.add_argument("--ld-library-path", default="")
    ap.add_argument("--min-free-gib", type=float, default=20.0)
    ap.add_argument("--max-seconds", type=int, default=120)
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--projection-rows", type=int, default=128)
    ap.add_argument("--out", type=Path, default=Path("bench/results/cptc-gpu.json"))
    ap.add_argument("--save-projection-dir", type=Path, default=None)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--probe-dump", type=Path, default=None)
    args = ap.parse_args()
    if args.worker:
        return _worker(args)
    if args.guarded:
        return _guarded(args)
    return _cpu(args)


if __name__ == "__main__":
    raise SystemExit(main())
