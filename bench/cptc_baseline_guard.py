#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only supervisor for one persistent full-model baseline quantization step.

The launch shape is:

  setsid nohup env ... python drivers/launch_persistent.py --run-dir RUN -- \
      python baseline/run_baseline_guard.py ...

`run_baseline_guard.py` performs the canonical preflight and calls the approved
isolate_fault.run_step once.  It never retries, resets the GPU, or runs a health
kernel.  A non-pass result writes a new STOP fence and leaves the receipt for
inspection.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

EXPECTED_HIP = "0,1"
EXPECTED_PCI = "0000:d1:00.0"
EXPECTED_PCIS = ("0000:d1:00.0", "0000:d4:00.0")
EXPECTED_ARCH = "gfx1030"
EXPECTED_PIPELINE_SHA = "6305be9c7c0f3ecea9164f94662e4a0538c3feef1ab5158b2d2742a232ec662b"
STUDY_4B = {
    "target_bpw": 3.0,
    "calib_samples": 128,
    "seq_len": 2048,
    "qad_preset": "default",
    "qad_steps": 300,
    "qad_final_kl": 200,
    "qad_samples": 256,
    "qad_batch": 4,
    "qad_budget": 1.0,
    "devices": "cuda:0,cuda:1",
}
SHARED_STOP = Path(
    "/home/chenco_adm/w4a8_runs/triton-pr12747/full-amd/STOP"
)
OWNED_STOP = Path(
    "/home/chenco_adm/w4a8_runs/cptc-qwen35-4b-20261119/baseline/STOP.baseline"
)


def load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod


def sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def donor_ld(python: Path, env: dict) -> tuple[Path, str]:
    """Return the venv site-packages and the exact donor-SDK precedence."""
    probe = subprocess.run(
        [
            str(python),
            "-c",
            "import sysconfig; print(sysconfig.get_paths()['purelib'])",
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    if probe.returncode != 0:
        raise RuntimeError(f"site-packages probe failed: {probe.stderr}")
    sp = Path(probe.stdout.strip())
    relative = (
        "_rocm_sdk_libraries/lib",
        "_rocm_sdk_core/lib/host-math/lib",
        "_rocm_sdk_core/lib/rocm_sysdeps/lib",
        "_rocm_sdk_core/lib",
        "torch/lib",
    )
    paths = [sp / rel for rel in relative]
    missing = [str(p) for p in paths if not p.is_dir()]
    if missing:
        raise RuntimeError(f"donor SDK libraries missing: {missing}")
    return sp, os.pathsep.join(str(p) for p in paths)


def check_ld_provenance(python: Path, source: Path, env: dict, sp: Path) -> dict:
    """CPU-only ldd and loaded-library checks before any GPU context."""
    ldd = subprocess.run(
        ["ldd", str(sp / "torch/lib/libtorch_hip.so")],
        env=env,
        cwd=source,
        text=True,
        capture_output=True,
        timeout=120,
    )
    if ldd.returncode != 0:
        raise RuntimeError(f"ldd failed: {ldd.stderr[-2000:]}")
    if "not found" in ldd.stdout:
        raise RuntimeError(f"ldd has unresolved libraries: {ldd.stdout[-4000:]}")
    ldd_paths = []
    for line in ldd.stdout.splitlines():
        if "=>" not in line:
            continue
        path = line.split("=>", 1)[1].strip().split(" (", 1)[0]
        if path.startswith("/"):
            ldd_paths.append(str(Path(path).resolve()))
    # Only the donor carries these ROCm/HIP libraries on this host.
    for path in ldd_paths:
        name = Path(path).name
        if any(
            token in name
            for token in (
                "libamdhip",
                "libhsa-runtime",
                "libMIOpen",
                "libhip",
                "libroc",
                "libtorch_hip",
                "librocm",
                "librccl",
            )
        ) and not Path(path).is_relative_to(sp):
            raise RuntimeError(f"ldd resolved a ROCm library outside donor: {path}")
    loaded = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import json, pathlib, torch\n"
                "paths = sorted({line.split()[-1] for line in "
                "pathlib.Path('/proc/self/maps').read_text().splitlines() "
                "if line.split() and line.split()[-1].startswith('/')})\n"
                "want = ('libtorch_hip', 'libamdhip64', 'libhsa-runtime64', "
                "'libMIOpen', 'libhip', 'libroc', 'librocm', 'librccl')\n"
                "print(json.dumps([p for p in paths if any(w in pathlib.Path(p).name for w in want)]))\n"
            ),
        ],
        env=env,
        cwd=source,
        text=True,
        capture_output=True,
        timeout=120,
    )
    if loaded.returncode != 0:
        raise RuntimeError(f"loaded-library probe failed: {loaded.stderr[-2000:]}")
    loaded_paths = json.loads(loaded.stdout)
    if not loaded_paths:
        raise RuntimeError("torch import loaded no donor ROCm library")
    bad = [path for path in loaded_paths if not Path(path).resolve().is_relative_to(sp)]
    if bad:
        raise RuntimeError(f"loaded ROCm library outside donor: {bad}")
    return {
        "ldd_hip_libraries": sorted(
            {p for p in ldd_paths if "libtorch_hip" in Path(p).name or "hip" in Path(p).name}
        ),
        "loaded_rocm_libraries": loaded_paths,
    }


def fail(message: str, run_dir: Path, stop: bool = False) -> int:
    write_json(
        run_dir / "baseline-refused.json",
        {"status": "refused", "error": message, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
    )
    if stop:
        OWNED_STOP.write_text(
            "Full 4B baseline stopped by cptc_baseline_guard.py; "
            "no health kernel, reset or retry was run. Parent authorization required.\n"
        )
    print(f"REFUSED: {message}", file=sys.stderr)
    return 2


def study_4b_preflight(args: argparse.Namespace) -> str | None:
    """Refuse any 4B study geometry outside the user's single DEFAULT contract."""
    actual = {
        "target_bpw": args.target_bpw,
        "calib_samples": args.calib_samples,
        "seq_len": args.seq_len,
        "qad_preset": args.qad_preset,
        "qad_steps": args.qad_steps,
        "qad_final_kl": args.qad_final_kl,
        "qad_samples": args.qad_samples,
        "qad_batch": args.qad_batch,
        "qad_budget": args.qad_budget,
        "devices": args.devices,
    }
    wrong = {
        key: {"required": STUDY_4B[key], "actual": actual[key]}
        for key in STUDY_4B
        if actual[key] != STUDY_4B[key]
    }
    if wrong:
        return (
            "4B study preflight refused: preset/steps/final-KL or matched geometry "
            f"mismatch {wrong}; only preset='default', steps=300, final_kl=200, "
            "samples=256, batch=4, budget=1, calib=128x2048, target_bpw=3.0 and "
            "devices='cuda:0,cuda:1' are permitted (no FAST study)"
        )
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--drivers", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--calib", type=Path, required=True)
    ap.add_argument("--python", type=Path, required=True)
    ap.add_argument("--timeout-seconds", type=int, default=24 * 3600)
    ap.add_argument("--min-free-gib", type=float, default=20.0)
    ap.add_argument("--target-bpw", type=float, default=3.0)
    ap.add_argument("--calib-samples", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--qad-preset", default="default")
    ap.add_argument("--qad-steps", type=int, default=300)
    ap.add_argument("--qad-final-kl", type=int, default=200)
    ap.add_argument("--qad-samples", type=int, default=256)
    ap.add_argument("--qad-batch", type=int, default=4)
    ap.add_argument("--qad-budget", type=float, default=1.0)
    ap.add_argument("--devices", default="cuda:0,cuda:1")
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    source = args.source.expanduser().resolve()
    model = args.model.expanduser().resolve()
    out = args.out.expanduser().resolve()
    workdir = args.workdir.expanduser().resolve()
    calib = args.calib.expanduser().resolve()
    drivers = args.drivers.expanduser().resolve()
    # Do not resolve the venv's python symlink: resolving it selects the
    # system interpreter and loses site-packages.
    python = args.python.expanduser()
    run_dir.mkdir(parents=True, exist_ok=True)

    study_error = study_4b_preflight(args)
    if study_error:
        return fail(study_error, run_dir)
    if os.environ.get("HIP_VISIBLE_DEVICES") != EXPECTED_HIP:
        return fail(f"HIP_VISIBLE_DEVICES must be exactly {EXPECTED_HIP}", run_dir)
    if os.environ.get("ROCR_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES"):
        return fail("ROCR/CUDA visibility must be empty", run_dir)
    if OWNED_STOP.is_file():
        return fail(f"baseline-owned STOP fence present: {OWNED_STOP}", run_dir)
    shared_stop = {
        "path": str(SHARED_STOP),
        "present": SHARED_STOP.is_file(),
        "sha256": sha256(SHARED_STOP) if SHARED_STOP.is_file() else None,
    }
    pipeline = source / "hi2" / "pipeline.py"
    if not pipeline.is_file() or sha256(pipeline) != EXPECTED_PIPELINE_SHA:
        return fail("frozen source pipeline.py hash mismatch", run_dir)
    lease = Path.home() / "w4a8_runs/lanes/lease/CPTCAllocation.a2"
    if not lease.is_file() or "a2" not in lease.read_text(errors="replace").lower():
        return fail("CPTCAllocation.a2 lease is absent or does not name a2", run_dir)
    for name in ("isolate_fault.py", "thermal_guard.py", "verify_env.py"):
        if not (drivers / name).is_file():
            return fail(f"approved driver missing: {drivers / name}", run_dir)

    iso = load(drivers / "isolate_fault.py")
    iso._install_signal_handlers()
    dmesg_text, attempts = iso.dmesg_read()
    lines = dmesg_text.splitlines()
    dmesg_before_count = len(lines)
    dmesg_before_tail = lines[-1] if lines else None

    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(EXPECTED_HIP)
    env.pop("ROCR_VISIBLE_DEVICES", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["PYTHONPATH"] = str(source)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["VIRTUAL_ENV"] = str(python.parent.parent)
    env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")
    try:
        sp, ld_path = donor_ld(python, env)
    except (OSError, RuntimeError, ValueError) as exc:
        return fail(f"donor SDK LD probe failed: {exc}", run_dir)
    env["LD_LIBRARY_PATH"] = ld_path
    sdk_hip = sp / "_rocm_sdk_core/lib/libamdhip64.so.7"
    if not sdk_hip.is_file():
        return fail(f"bundled HIP runtime missing: {sdk_hip}", run_dir)
    sdk_alias = run_dir / "sdk-lib/libamdhip64.so"
    sdk_alias.parent.mkdir(parents=True, exist_ok=True)
    if not sdk_alias.is_file():
        sdk_alias.symlink_to(sdk_hip)
    env["TRITON_LIBHIP_PATH"] = str(sdk_alias)
    try:
        ld_provenance = check_ld_provenance(python, source, env, sp)
    except (OSError, RuntimeError, ValueError) as exc:
        return fail(f"library provenance check failed: {exc}", run_dir)
    # verify_env's device_count() is zero with this ROCm/torch build even
    # though device 0 is available and maps to the leased PCI address.  Keep
    # the canonical CPU/source/torch checks, then query the leased properties
    # without launching a kernel.
    verify = subprocess.run(
        [
            str(python),
            str(drivers / "verify_env.py"),
            "--no-hash",
            "--cpu-only",
            "--json",
        ],
        cwd=source,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    if verify.returncode != 0:
        return fail(f"verify_env failed: {verify.stdout[-2000:]} {verify.stderr[-2000:]}", run_dir)
    try:
        identity = json.loads(verify.stdout[verify.stdout.index("{") :])
    except (ValueError, json.JSONDecodeError) as exc:
        return fail(f"verify_env JSON parse failed: {exc}", run_dir)
    if not identity.get("ok"):
        return fail(f"verify_env not ok: {identity.get('errors')}", run_dir)
    probe = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import json, torch\n"
                "devices = []\n"
                "for index in range(2):\n"
                "    p = torch.cuda.get_device_properties(index)\n"
                "    free, total = torch.cuda.mem_get_info(index)\n"
                "    devices.append({\n"
                "      'index': index,\n"
                "      'name': p.name,\n"
                "      'arch': getattr(p, 'gcnArchName', None),\n"
                "      'pci': '%04x:%02x:%02x.0' % (p.pci_domain_id, p.pci_bus_id, p.pci_device_id),\n"
                "      'pci_tuple': [p.pci_domain_id, p.pci_bus_id, p.pci_device_id],\n"
                "      'warp_size': p.warp_size,\n"
                "      'mem_free': free,\n"
                "      'mem_total': total,\n"
                "      'mem_free_gib': round(free / 2**30, 3),\n"
                "    })\n"
                "print(json.dumps({\n"
                "  'available': torch.cuda.is_available(),\n"
                "  'device_count': torch.cuda.device_count(),\n"
                "  'devices': devices,\n"
                "}, sort_keys=True))\n"
            ),
        ],
        cwd=source,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    if probe.returncode != 0:
        return fail(f"torch identity probe failed: {probe.stdout[-2000:]} {probe.stderr[-2000:]}", run_dir)
    try:
        gpu = json.loads(probe.stdout[probe.stdout.index("{") :])
    except (ValueError, json.JSONDecodeError) as exc:
        return fail(f"torch identity JSON parse failed: {exc}", run_dir)
    if not gpu.get("available"):
        return fail(f"torch reports no available GPU: {gpu}", run_dir)
    devices = gpu.get("devices") or []
    if len(devices) != 2:
        return fail(f"expected exactly two visible GPUs: {gpu}", run_dir)
    for expected_pci, device in zip(EXPECTED_PCIS, devices):
        if device.get("pci") != expected_pci or device.get("arch") != EXPECTED_ARCH:
            return fail(f"unexpected GPU identity: {device}", run_dir)
        if "V620" not in str(device.get("name")):
            return fail(f"unexpected device name: {device.get('name')}", run_dir)
        if float(device.get("mem_free_gib", 0)) < args.min_free_gib:
            return fail(
                f"free VRAM below {args.min_free_gib} GiB on cuda:{device['index']}: "
                f"{device.get('mem_free_gib')}",
                run_dir,
            )
    check = iso.pre_step_dmesg_check(dmesg_before_count, dmesg_before_tail)
    if not check["ok"]:
        return fail(f"pre-step dmesg check failed: {check.get('error') or check}", run_dir)
    if args.preflight_only:
        write_json(
            run_dir / "preflight-only.json",
            {
                "status": "ok",
                "verify_env_cpu_only": identity,
                "torch_identity_probe": gpu,
                "dmesg_check": check,
            },
        )
        print(json.dumps({"status": "preflight-ok", "gpu": gpu}, indent=2, sort_keys=True))
        return 0

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    entry = [str(python), "-m", "hi2"]
    command = [
        *entry,
        "quantize",
        str(model),
        str(out),
        "--workdir",
        str(workdir),
        "--devices",
        args.devices,
        "--calib",
        str(calib),
        "--calib-samples",
        str(args.calib_samples),
        "--seq-len",
        str(args.seq_len),
        "--target-bpw",
        str(args.target_bpw),
        "--qad-preset",
        args.qad_preset,
        "--qad-steps",
        str(args.qad_steps),
        "--qad-final-kl",
        str(args.qad_final_kl),
        "--qad-samples",
        str(args.qad_samples),
        "--qad-batch",
        str(args.qad_batch),
        "--qad-budget",
        str(args.qad_budget),
    ]
    plan = {
        "kind": "full-4b-baseline",
        "created_utc": stamp,
        "run_dir": str(run_dir),
        "source": str(source),
        "pipeline_sha256": sha256(pipeline),
        "model": str(model),
        "calib": str(calib),
        "gpu": {"hip": EXPECTED_HIP, "pcis": list(EXPECTED_PCIS), "arch": EXPECTED_ARCH},
        "cptc": False,
        "cptc_flag_absent": True,
        "hi2_entrypoint": entry,
        "command": command,
        "ld_library_path": env["LD_LIBRARY_PATH"],
        "ld_provenance": ld_provenance,
        "shared_stop": shared_stop,
        "study_preflight": {
            "required": STUDY_4B,
            "passed": study_4b_preflight(args) is None,
        },
        "real_defaults": {
            "calib_samples": args.calib_samples,
            "seq_len": args.seq_len,
            "target_bpw": args.target_bpw,
            "qad_preset": args.qad_preset,
            "qad_steps": args.qad_steps,
            "final_kl_steps": args.qad_final_kl,
            "qad_samples": args.qad_samples,
            "qad_batch": args.qad_batch,
            "qad_budget": args.qad_budget,
        },
        "ptq_scale_search": {
            "candidates_per_tensor": 6,
            "candidate_pairs": [[2, 16], [2, 32], [2, 64], [4, 16], [4, 32], [4, 64]],
            "forward_calibration_windows": 128,
            "sensitivity_probe_windows": 8,
            "final_encode_candidates_per_tensor": 1,
            "note": "primitive Viterbi M values/s are not used as an encode-speed estimate",
        },
        "quality_plan": {
            "eval_mix_txt": "/home/chenco_adm/w4a8_runs/cptc-qwen35-4b-20261119/inputs/eval-mix.txt",
            "eval_mix_complete_2048_windows": 28,
            "eval_docs_heldout_txt": "/home/chenco_adm/w4a8_runs/cptc-qwen35-4b-20261119/inputs/eval-docs-heldout.txt",
            "eval_docs_heldout_complete_2048_windows": 7,
            "use_samples": {"eval_mix": 28, "eval_docs_heldout": 7},
            "do_not_use": "default 8 documents",
        },
        "guards": {
            "driver": str(drivers),
            "isolate_fault": str(drivers / "isolate_fault.py"),
            "thermal_guard": str(drivers / "thermal_guard.py"),
            "shared_stop_file": str(SHARED_STOP),
            "baseline_owned_stop_file": str(OWNED_STOP),
            "post_fault_actions": {"health_kernel": False, "reset": False, "retry": False, "rerun": False},
        },
        "dmesg_before": {"count": dmesg_before_count, "attempts": attempts},
        "preflight": {
            "verify_env_cpu_only": identity,
            "torch_identity_probe": gpu,
            "torch_device_count_quirk_observed": gpu.get("device_count") == 0,
        },
    }
    write_json(run_dir / "run-plan.json", plan)

    overrides = {
        "HIP_VISIBLE_DEVICES": str(EXPECTED_HIP),
        "PYTHONPATH": str(source),
        "VIRTUAL_ENV": str(python.parent.parent),
        "OMP_NUM_THREADS": "8",
        "HSA_ENABLE_SDMA": "0",
        "HSA_FORCE_FINE_GRAIN_PCIE": "1",
        "GPU_MAX_HW_QUEUES": "2",
        "HI2_CALIB_MICROBATCH": "8",
        "TORCH_BLAS_PREFER_HIPBLASLT": "0",
        "PYTORCH_TUNABLEOP_ENABLED": "0",
        "PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "LD_LIBRARY_PATH": ld_path,
        "TRITON_LIBHIP_PATH": env["TRITON_LIBHIP_PATH"],
    }
    # run_step appends its module-level EXPECTED_PCI to the thermal guard;
    # point that at the A2 first device and add the second device explicitly.
    iso.EXPECTED_PCI = EXPECTED_PCI
    result = iso.run_step(
        name="baseline-4b",
        sequence=1,
        cmd=command,
        cwd=source,
        timeout_s=args.timeout_seconds,
        stamp=stamp,
        dmesg_before_count=check["after_count"],
        dmesg_before_tail=check["after_tail"],
        parse="auto",
        env_overrides=overrides,
        thermal_cmd=[
            str(python),
            str(drivers / "thermal_guard.py"),
            "--expected-pci",
            EXPECTED_PCIS[1],
        ],
    )
    receipt = {
        "kind": "full-4b-baseline",
        "plan": plan,
        "run_step": result,
        "status": result.get("status"),
        "peak_vram_measured": False,
    }
    write_json(run_dir / "quantization-receipt.json", receipt)
    if result.get("status") != "pass":
        OWNED_STOP.write_text(
            "Full 4B baseline fault stop by cptc_baseline_guard.py; "
            "no health kernel, reset or retry was run. Parent authorization required.\n"
            + json.dumps({"run_dir": str(run_dir), "status": result.get("status")}, sort_keys=True)
            + "\n"
        )
        print(json.dumps({"status": result.get("status"), "receipt": str(run_dir / "quantization-receipt.json")}, indent=2))
        return 4
    print(json.dumps({"status": "pass", "receipt": str(run_dir / "quantization-receipt.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
