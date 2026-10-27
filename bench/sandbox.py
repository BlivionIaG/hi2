# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Run model-written Python in a child process: its own temporary directory, an isolated
interpreter without site packages (``-I -S``: the standard library only, and no
``sitecustomize`` that imports torch), a minimal environment, CPU/memory/file-size
limits, sockets and process/file-removal calls disabled, and the whole process group
killed on timeout.

This is a guard against mistakes, not a security boundary: run untrusted output in a
container or VM as well.
"""

import contextlib
import os
import re
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

PRELUDE = """\
import resource as _r
for _lim, _v in ((_r.RLIMIT_CPU, {cpu}), (_r.RLIMIT_FSIZE, 1 << 24), (_r.RLIMIT_CORE, 0),
                 (getattr(_r, "RLIMIT_AS", None), {mem})):
    try:
        if _lim is not None:
            _r.setrlimit(_lim, (_v, _v))
    except (ValueError, OSError):
        pass
import os as _os, shutil as _sh, socket as _s, subprocess as _sp
def _off(*a, **k):
    raise OSError("disabled in the bench sandbox")
for _m, _names in ((_s, ("socket", "create_connection", "socketpair", "getaddrinfo")),
                   (_os, ("system", "fork", "forkpty", "kill", "killpg", "remove",
                          "unlink", "rmdir", "removedirs", "rename", "replace", "putenv",
                          "chmod", "chown", "execv", "execve", "spawnv", "popen")),
                   (_sh, ("rmtree", "move", "chown")),
                   (_sp, ("Popen", "run", "call", "check_call", "check_output"))):
    for _n in _names:
        if hasattr(_m, _n):
            setattr(_m, _n, _off)
del _r, _lim, _v, _m, _names, _n
"""


def run_python(code: str, timeout: float = 10.0, memory_mb: int = 2048) -> dict:
    """Run ``code``; ``status`` is passed (exit 0), failed or timeout."""
    prelude = PRELUDE.format(cpu=int(timeout) + 1, mem=memory_mb << 20)
    with tempfile.TemporaryDirectory(prefix="hi2-bench-") as d:
        prog = Path(d) / "prog.py"
        prog.write_text(prelude + code)
        env = {
            "PATH": os.defpath,
            "HOME": d,
            "TMPDIR": d,
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
        }
        p = subprocess.Popen(
            [sys.executable, "-I", "-S", "-B", str(prog)],
            cwd=d,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            _, err = p.communicate(timeout=timeout)
            status = "passed" if p.returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(p.pid, signal.SIGKILL)
            _, err = p.communicate()
            status = "timeout"
    tail = err.decode(errors="replace").strip().splitlines()[-3:]
    return {"passed": status == "passed", "status": status, "stderr": "\n".join(tail)}


FENCE = re.compile(
    r"```[ \t]*(?:python|py|python3)?[ \t]*\n(.*?)(?:```|\Z)", re.S | re.I
)


def extract_code(text: str, entry: str | None = None) -> str:
    """The code of a reply: the fenced block that defines ``entry``, else the first fenced
    block, else the whole text."""
    blocks = FENCE.findall(text)
    if not blocks:
        return text
    if entry:
        for b in blocks:
            if re.search(rf"\bdef\s+{re.escape(entry)}\s*\(", b):
                return b
    return blocks[0]
