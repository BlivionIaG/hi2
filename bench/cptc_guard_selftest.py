#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only proof that the approved isolate_fault.run_step controls work.

No GPU and no torch: the child is a sleeper, dmesg is a deterministic fake, and
the thermal guard is a local Python command.  Cases:
  kernel  a new dmesg gfxhub fault with no child-log marker stops the PGID;
  guard   thermal_guard exiting 3 aborts the child before it finishes;
  cancel  the approved cancellation hook stops the PGID;
  intr    SIGINT to a supervisor running run_step leaves no child PGID.
"""
import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


def load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod


def child_script(tmp: Path) -> Path:
    path = tmp / "child.py"
    path.write_text(
        "import os, signal, sys, time\n"
        "open(sys.argv[1], 'w').write(str(os.getpid()))\n"
        "time.sleep(60)\n"
    )
    return path


def base_args(iso, tmp: Path, cmd, *extra, guard=None):
    return dict(
        name="selftest", sequence=1, cmd=cmd, cwd=tmp, timeout_s=20,
        stamp="selftest", dmesg_before_count=1,
        dmesg_before_tail="baseline",
        thermal_cmd=guard or [sys.executable, "-c", "import sys,time; time.sleep(60)"],
    )


def pgid_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def supervisor_script(tmp: Path) -> Path:
    path = tmp / "supervisor.py"
    path.write_text(
        "import importlib.util, json, sys, time\n"
        "from pathlib import Path\n"
        "drv=Path(sys.argv[1]); out=Path(sys.argv[2])\n"
        "spec=importlib.util.spec_from_file_location('iso', drv/'isolate_fault.py')\n"
        "m=importlib.util.module_from_spec(spec); sys.modules['iso']=m; spec.loader.exec_module(m)\n"
        "cmd=[sys.executable, sys.argv[3], str(out/'intr.pid')]\n"
        "r=m.run_step(name='intr', sequence=1, cmd=cmd, cwd=out, timeout_s=30, stamp='intr',\n"
        "  dmesg_before_count=1, dmesg_before_tail='baseline',\n"
        "  dmesg_reader=lambda: ('baseline', []),\n"
        "  thermal_cmd=[sys.executable,'-c','import time; time.sleep(60)'])\n"
        "print(json.dumps({'status':r['status'],'pgid':r.get('pgid')}))\n"
    )
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drivers", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("bench/results/cptc-guard-selftest.json"))
    ap.add_argument("--interrupt-child", type=Path, help=argparse.SUPPRESS)
    a = ap.parse_args()
    iso = load(a.drivers / "isolate_fault.py")
    tmp = (a.out.parent / "guard-selftest").resolve()
    tmp.mkdir(parents=True, exist_ok=True)
    for d in (iso.LOGS, iso.ARTIFACTS, iso.JUNIT):
        d.mkdir(parents=True, exist_ok=True)
    child = child_script(tmp)
    results = {}

    # 1. kernel-only fault: no child log marker, the fake dmesg gains a page fault.
    state = {"n": 0}

    def fake_dmesg():
        state["n"] += 1
        lines = ["baseline"]
        if state["n"] > 1:
            lines.append("kernel: amdgpu: [gfxhub] page fault (fake)")
        return "\n".join(lines), []

    pidfile = tmp / "kernel.pid"
    r = iso.run_step(
        **base_args(iso, tmp, [sys.executable, str(child), str(pidfile)]),
        dmesg_reader=fake_dmesg,
    )
    time.sleep(0.2)
    results["kernel"] = {
        "status": r["status"], "first_event": r.get("first_event"),
        "dmesg_fault_lines": r.get("dmesg_fault_lines"),
        "log_fault_markers": r.get("log_fault_markers"),
        "child_gone": not pgid_alive(int(pidfile.read_text())) if pidfile.exists() else True,
    }

    # 2. thermal guard exits 3 before the child.
    pidfile = tmp / "guard.pid"
    r = iso.run_step(
        **base_args(
            iso, tmp, [sys.executable, str(child), str(pidfile)],
            guard=[sys.executable, "-c", "import sys; sys.exit(3)"],
        ),
        dmesg_reader=lambda: ("baseline", []),
    )
    time.sleep(0.2)
    results["guard"] = {
        "status": r["status"], "first_event": r.get("first_event"),
        "thermal_guard_exit": r.get("thermal_guard_exit"),
        "child_gone": not pgid_alive(int(pidfile.read_text())) if pidfile.exists() else True,
    }

    # 3. approved cancellation hook.
    pidfile = tmp / "cancel.pid"
    r = iso.run_step(
        **base_args(iso, tmp, [sys.executable, str(child), str(pidfile)]),
        dmesg_reader=lambda: ("baseline", []),
        cancel_check=lambda: "selftest cancel",
    )
    time.sleep(0.2)
    results["cancel"] = {
        "status": r["status"], "first_event": r.get("first_event"),
        "child_gone": not pgid_alive(int(pidfile.read_text())) if pidfile.exists() else True,
    }

    # 4. SIGINT the supervisor; its finally must kill the owned child PGID.
    sup = supervisor_script(tmp)
    intr_pid = tmp / "intr.pid"
    p = subprocess.Popen(
        [sys.executable, str(sup), str(a.drivers.resolve()), str(tmp), str(child)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
    )
    for _ in range(100):
        if intr_pid.is_file():
            break
        time.sleep(0.05)
    child_pgid = int(intr_pid.read_text()) if intr_pid.is_file() else None
    os.kill(p.pid, signal.SIGINT)
    try:
        out, err = p.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        p.kill()
        out, err = p.communicate()
    time.sleep(0.3)
    results["interrupt"] = {
        "supervisor_exit": p.returncode, "stderr_tail": err[-400:],
        "child_pgid": child_pgid,
        "child_gone": child_pgid is None or not pgid_alive(child_pgid),
    }
    results["ok"] = bool(
        results["kernel"]["status"] == "gpu_fault"
        and results["kernel"]["child_gone"]
        and results["guard"]["status"] == "thermal_abort"
        and results["guard"]["child_gone"]
        and results["cancel"]["status"] == "cancelled"
        and results["cancel"]["child_gone"]
        and results["interrupt"]["child_gone"]
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return 0 if results["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
