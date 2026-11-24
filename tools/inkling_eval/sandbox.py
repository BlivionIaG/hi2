# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Execute model-written Python, C and TypeScript in child processes.

Every run gets its own empty directory, a minimal environment, CPU/memory/file
size limits, a wall-clock timeout and a killed process group.  On Linux the
child is additionally placed in an empty network namespace (``unshare -n``)
when the kernel allows it.  These are accident guards, not a security
boundary: as ``bench/README.md`` says, untrusted output belongs in a container
or VM as well.

TypeScript has no compiler on the evaluation box.  When ``node`` supports
type stripping (22.6+) or ``bun``/``deno`` is present the file runs natively;
otherwise the harness strips the documented subset of type syntax itself and
runs the resulting JavaScript with ``qjs`` or the Python ``quickjs`` module.
``strip_typescript`` is deliberately small: tasks ask for function
declarations without classes, enums or namespaces.
"""

import importlib.util
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCRATCH_ENV = "INKLING_EVAL_SCRATCH"


def scratch_root() -> Path:
    """Where sandbox directories are made; never the box's tiny ``/tmp`` by default."""
    root = Path(os.environ.get(SCRATCH_ENV) or (Path.home() / ".cache" / "inkling-eval" / "tmp"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, AttributeError):
        proc.kill()


def _wait(proc: subprocess.Popen, timeout: float) -> str:
    try:
        proc.communicate(timeout=timeout)
        return "pass" if proc.returncode == 0 else "fail"
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        return "timeout"


_UNSHARE: list[str] | None = None


def _net_prefix() -> list[str]:
    """``unshare -n --`` when it works here, otherwise nothing (recorded)."""
    global _UNSHARE
    if _UNSHARE is not None:
        return list(_UNSHARE)
    _UNSHARE = []
    exe = shutil.which("unshare")
    if exe and sys.platform.startswith("linux"):
        try:
            probe = subprocess.run(
                [exe, "-n", "--", "true"], capture_output=True, timeout=5
            )
            if probe.returncode == 0:
                _UNSHARE = [exe, "-n", "--"]
        except (OSError, subprocess.SubprocessError):
            pass
    return list(_UNSHARE)


def network_isolation() -> bool:
    return bool(_net_prefix())


def _env(home: Path) -> dict:
    return {
        "PATH": os.defpath + ":/usr/local/bin",
        "HOME": str(home),
        "TMPDIR": str(home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }


PY_PRELUDE = """\
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
    raise OSError("disabled in the eval sandbox")
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

JS_PRELUDE = """\
if (typeof console === "undefined") { globalThis.console = {log: function () {}, error: function () {}}; }
"""


def _run_argv(argv: list[str], cwd: Path, timeout: float, env: dict) -> dict:
    started = time.monotonic()
    try:
        proc = subprocess.Popen(
            _net_prefix() + argv,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            errors="replace",
            start_new_session=True,
        )
    except OSError as e:
        return {"status": "error", "error": str(e), "seconds": 0.0, "stdout": "", "stderr": ""}
    status = _wait(proc, timeout)
    try:
        out, err = proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        out, err = proc.communicate()
    return {
        "status": status,
        "stdout": out[-4000:],
        "stderr": err[-4000:],
        "returncode": proc.returncode,
        "seconds": round(time.monotonic() - started, 3),
    }


def run_python(source: str, tests: str, timeout: float = 10.0, memory_mb: int = 2048) -> dict:
    """Run Python ``source`` plus hidden ``tests``; pass = exit 0 and 'OK' printed."""
    prelude = PY_PRELUDE.format(cpu=int(timeout) + 1, mem=memory_mb << 20)
    with tempfile.TemporaryDirectory(prefix="py-", dir=str(scratch_root())) as d:
        home = Path(d)
        prog = home / "prog.py"
        prog.write_text(prelude + source + "\n" + tests + "\n")
        result = _run_argv(
            [sys.executable, "-S", str(prog)], home, timeout, _env(home)
        )
    result["language"] = "python"
    result["network_isolation"] = network_isolation()
    if result["status"] == "pass" and "OK" not in result.get("stdout", ""):
        result["status"] = "fail"
    return result


C_HEADER = """\
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
"""


def run_c(source: str, test_source: str, timeout: float = 10.0) -> dict:
    """Compile a C translation unit with a hidden test ``main`` and run it."""
    cc = next((shutil.which(name) for name in ("gcc", "cc", "clang") if shutil.which(name)), None)
    if not cc:
        return {"status": "unavailable", "language": "c", "error": "no C compiler"}
    with tempfile.TemporaryDirectory(prefix="c-", dir=str(scratch_root())) as d:
        home = Path(d)
        (home / "candidate.c").write_text(source + "\n")
        (home / "test.c").write_text(C_HEADER + test_source + "\n")
        compile_argv = [
            cc, "-std=c11", "-O0", "-w", "-o", "prog", "candidate.c", "test.c", "-lm",
        ]
        env = _env(home)
        compiled = _run_argv(compile_argv, home, 20.0, env)
        if compiled["status"] != "pass":
            compiled["language"] = "c"
            compiled["stage"] = "compile"
            compiled["network_isolation"] = network_isolation()
            return compiled
        result = _run_argv(["./prog"], home, timeout, env)
    result["language"] = "c"
    result["stage"] = "run"
    result["network_isolation"] = network_isolation()
    if result["status"] == "pass" and "OK" not in result.get("stdout", ""):
        result["status"] = "fail"
    return result


# --------------------------------------------------------------------------- TypeScript


def _node_supports_ts() -> bool:
    exe = shutil.which("node")
    if not exe:
        return False
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=5).stdout
        major, minor = (int(x) for x in out.strip().lstrip("v").split(".")[:2])
        return (major, minor) >= (22, 6)
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


_ENGINE: dict | None = None


def js_engine() -> dict:
    """The first usable JavaScript engine, with how it runs TypeScript."""
    global _ENGINE
    if _ENGINE is not None:
        return dict(_ENGINE)
    forced = os.environ.get("INKLING_EVAL_JS_ENGINE")
    if forced in ("node", "bun", "deno", "qjs", "quickjs"):
        _ENGINE = {"name": forced, "native_ts": forced in ("bun", "deno") or (forced == "node" and _node_supports_ts())}
        return dict(_ENGINE)
    if shutil.which("node"):
        _ENGINE = {"name": "node", "native_ts": _node_supports_ts()}
    elif shutil.which("bun"):
        _ENGINE = {"name": "bun", "native_ts": True}
    elif shutil.which("deno"):
        _ENGINE = {"name": "deno", "native_ts": True}
    elif shutil.which("qjs"):
        _ENGINE = {"name": "qjs", "native_ts": False}
    elif importlib.util.find_spec("quickjs") is not None:
        _ENGINE = {"name": "quickjs", "native_ts": False}
    else:
        _ENGINE = {"name": None, "native_ts": False}
    return dict(_ENGINE)


def _delete_ranges(text: str, ranges: list[tuple[int, int]]) -> str:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    out = text
    for start, end in reversed(merged):
        out = out[:start] + out[end:]
    return out


def _mask(src: str) -> str:
    """A same-length copy with string, template and comment contents blanked."""
    chars = list(src)
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c in ("'", '"', "`"):
            quote = c
            i += 1
            while i < n:
                if src[i] == "\\":
                    chars[i] = " "
                    if i + 1 < n:
                        chars[i + 1] = " "
                    i += 2
                    continue
                if src[i] == quote:
                    break
                chars[i] = " "
                i += 1
            i += 1
        elif src.startswith("//", i):
            while i < n and src[i] != "\n":
                chars[i] = " "
                i += 1
        elif src.startswith("/*", i):
            while i < n and not src.startswith("*/", i):
                if src[i] != "\n":
                    chars[i] = " "
                i += 1
            if i < n:
                chars[i] = chars[i + 1] = " "
                i += 2
        else:
            i += 1
    return "".join(chars)


def _match(masked: str, pos: int, open_ch: str, close_ch: str) -> int:
    depth = 0
    for i in range(pos, len(masked)):
        if masked[i] == open_ch:
            depth += 1
        elif masked[i] == close_ch:
            depth -= 1
            if depth == 0:
                return i
    return -1


def _type_end(masked: str, start: int, stops: str = ",;=\n", stop_arrow: bool = False) -> int:
    """End of a type expression at ``start``; ``{`` starts code, not a type."""
    depth = 0
    i = start
    n = len(masked)
    while i < n:
        c = masked[i]
        if masked.startswith("=>", i) and not stop_arrow:
            i += 2
            continue
        if c in "([<":
            depth += 1
        elif c in ")]>":
            if depth == 0:
                return i
            depth -= 1
        elif c == "{":
            return i
        elif c == "}":
            if depth == 0:
                return i
            depth -= 1
        elif stop_arrow and depth == 0 and masked.startswith("=>", i):
            return i
        elif depth == 0 and c in stops and not (c == "=" and masked.startswith("=>", i)):
            return i
        i += 1
    return n


def _strip_param_list(masked: str, open_paren: int, close_paren: int) -> list[tuple[int, int]]:
    edits = []
    depth = 0
    i = open_paren + 1
    while i < close_paren:
        c = masked[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif depth == 0 and c == ":":
            start = i
            while start > open_paren + 1 and masked[start - 1] in "?!":
                start -= 1
            end = _type_end(masked, i + 1)
            edits.append((start, end))
            i = end
            continue
        i += 1
    return edits


def _strip_arrow_params(masked: str) -> list[tuple[int, int]]:
    edits = []
    for match in re.finditer(r"=>", masked):
        j = match.start() - 1
        while j >= 0 and masked[j].isspace():
            j -= 1
        # walk back over a return annotation to the parameter list's ')'
        k = j
        while k >= 0 and (masked[k].isalnum() or masked[k] in "_$.<>[]|&,: "):
            if masked[k] == ")":
                break
            k -= 1
        if k >= 0 and masked[k] == ")":
            depth = 0
            open_paren = -1
            for j2 in range(k, -1, -1):
                if masked[j2] in ")]}":
                    depth += 1
                elif masked[j2] in "([{":
                    depth -= 1
                    if depth == 0:
                        open_paren = j2
                        break
            if open_paren >= 0 and masked[open_paren] == "(":
                colon = masked.find(":", k + 1, match.start())
                if colon != -1:
                    edits.append((colon, _type_end(masked, colon + 1, stop_arrow=True)))
                edits.extend(_strip_param_list(masked, open_paren, k))
        else:
            # single identifier parameter: `x: T => ...`
            colon = masked.rfind(":", 0, match.start())
            if colon != -1 and masked[colon - 1:colon].strip() == "" or (colon != -1 and masked[colon - 1].isalnum()):
                ident = masked[masked.rfind(" ", 0, colon) + 1:colon].strip()
                if ident and (ident[0].isalpha() or ident[0] in "_$") and all(
                    ch.isalnum() or ch in "_$" for ch in ident
                ):
                    edits.append((colon, _type_end(masked, colon + 1)))
    return edits


def strip_typescript(src: str) -> str:
    """Erase the type syntax of the documented task subset, leaving runnable JS."""
    edits: list[tuple[int, int]] = []
    masked = _mask(src)
    for match in re.finditer(r"\binterface\s+[A-Za-z_$][\w$]*[^{;]*\{", masked):
        close = _match(masked, match.end() - 1, "{", "}")
        if close != -1:
            edits.append((match.start(), close + 1))
    for match in re.finditer(r"(?m)^\s*type\s+[A-Za-z_$][\w$]*\s*=", masked):
        end = masked.find(";", match.end())
        line_end = masked.find("\n", match.end())
        end = line_end if end == -1 else (end if line_end == -1 else min(end, line_end))
        edits.append((match.start(), len(masked) if end == -1 else end + 1))
    for match in re.finditer(r"\bfunction\s+[A-Za-z_$][\w$]*\s*<[^<>]*>", masked):
        lt = masked.find("<", match.start())
        gt = masked.find(">", lt)
        edits.append((lt, gt + 1))
    for match in re.finditer(r"=\s*<[A-Za-z_$][\w$]*(\s*,\s*[A-Za-z_$][\w$]*)*>\s*\(", masked):
        lt = masked.find("<", match.start())
        gt = masked.find(">", lt)
        edits.append((lt, gt + 1))
    for match in re.finditer(r"\bfunction\b[^(]*\(", masked):
        open_paren = match.end() - 1
        close = _match(masked, open_paren, "(", ")")
        if close == -1:
            continue
        edits.extend(_strip_param_list(masked, open_paren, close))
        k = close + 1
        while k < len(masked) and masked[k].isspace():
            k += 1
        if k < len(masked) and masked[k] == ":":
            edits.append((k, _type_end(masked, k + 1, ";\n", stop_arrow=True)))
    edits.extend(_strip_arrow_params(masked))
    for match in re.finditer(r"\b(?:let|const|var)\s+", masked):
        start = match.end()
        end = start
        depth = 0
        while end < len(masked):
            c = masked[end]
            if c in "([{<":
                depth += 1
            elif c in ")]}>":
                depth -= 1
            elif depth == 0 and c in ";\n":
                break
            end += 1
        i = start
        while i < end:
            decl = re.match(r"\s*([A-Za-z_$][\w$]*)\s*([!?])?\s*:", masked[i:end])
            if decl:
                colon = i + decl.end() - 1
                stop = _type_end(masked, colon + 1, ",;=\n")
                edits.append((colon, stop))
                i = stop
            j = i
            depth = 0
            while j < end:
                c = masked[j]
                if c in "([{<":
                    depth += 1
                elif c in ")]}>":
                    depth -= 1
                elif depth == 0 and c == ",":
                    break
                j += 1
            i = j + 1
    for match in re.finditer(r"\bas\s+(?:const|[A-Za-z_$][\w$]*)", masked):
        start = match.start()
        end = _type_end(masked, match.end(), ",;=\n")
        while start > 0 and masked[start - 1] == " ":
            start -= 1
        edits.append((start, end))
    return _delete_ranges(src, edits)


def _js_payload(source: str, tests: str) -> str:
    return JS_PRELUDE + source + "\n" + tests + "\n"


def run_typescript(source: str, tests: str, timeout: float = 10.0) -> dict:
    """Run TypeScript ``source`` plus hidden JavaScript tests; pass = exit 0 and 'OK'."""
    engine = js_engine()
    if not engine["name"]:
        return {"status": "unavailable", "language": "typescript", "error": "no JS engine"}
    payload = _js_payload(source, tests)
    with tempfile.TemporaryDirectory(prefix="ts-", dir=str(scratch_root())) as d:
        home = Path(d)
        env = _env(home)
        if engine["name"] == "node" and engine["native_ts"]:
            path = home / "candidate.ts"
            path.write_text(payload)
            argv = [shutil.which("node"), "--experimental-strip-types", "--no-warnings", str(path)]
            transform = "native"
        elif engine["name"] == "bun":
            path = home / "candidate.ts"
            path.write_text(payload)
            argv = [shutil.which("bun"), "run", str(path)]
            transform = "native"
        elif engine["name"] == "deno":
            path = home / "candidate.ts"
            path.write_text(payload)
            argv = [shutil.which("deno"), "run", "--quiet", "--no-check", str(path)]
            transform = "native"
        elif engine["name"] == "qjs":
            path = home / "candidate.js"
            path.write_text(strip_typescript(payload))
            argv = [shutil.which("qjs"), str(path)]
            transform = "strip"
        else:  # python quickjs
            path = home / "candidate.js"
            path.write_text(strip_typescript(payload))
            runner = home / "quickjs_runner.py"
            runner.write_text(
                "import json, pathlib, sys\n"
                "import quickjs\n"
                "code = pathlib.Path(sys.argv[1]).read_text()\n"
                "ctx = quickjs.Context()\n"
                "ctx.eval('var __out = [];')\n"
                "ctx.eval('if (typeof console === \"undefined\") { console = {'\n"
                "          'log: function (x) { __out.push(String(x)); },'\n"
                "          'error: function (x) { __out.push(String(x)); }}; }')\n"
                "ctx.eval(code)\n"
                "out = json.loads(ctx.eval('JSON.stringify(__out)'))\n"
                "if out:\n"
                "    print('\\n'.join(str(line) for line in out))\n"
            )
            spec = importlib.util.find_spec("quickjs")
            env["PYTHONPATH"] = str(Path(spec.origin).parent.parent)
            argv = [sys.executable, "-S", str(runner), str(path)]
            transform = "strip"
        result = _run_argv(argv, home, timeout, env)
    result["language"] = "typescript"
    result["engine"] = engine["name"]
    result["transform"] = transform
    result["network_isolation"] = network_isolation()
    if result["status"] == "pass" and "OK" not in result.get("stdout", ""):
        result["status"] = "fail"
    return result
