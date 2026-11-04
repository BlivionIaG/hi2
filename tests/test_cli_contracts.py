# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Real CLI subprocess contracts: lightweight help, clean JSON, safe local outputs.

All compute fixtures are local and CPU-only. No model downloads or mocked dispatch.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = (
    "quantize",
    "verify",
    "eval",
    "export-dense",
    "restack",
    "size",
    "calib",
    "cache",
)
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ message['role'] }} >> {{ message['content'] }}\n"
    "{% endfor %}"
)


def _env():
    return {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "PYTHONIOENCODING": "utf-8",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }


def _command(*args, no_torch=False):
    # -S removes site-packages, including Torch, rather than faking its import.
    return [sys.executable, *(["-S"] if no_torch else []), "-m", "hi2", *map(str, args)]


def _cli(*args, no_torch=False):
    return subprocess.run(
        _command(*args, no_torch=no_torch),
        cwd=ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def _result(proc, status="ok", code=0):
    assert proc.returncode == code, (proc.stdout, proc.stderr)
    result = json.loads(proc.stdout)  # Entire stdout, without stripping any log lines.
    assert isinstance(result, dict)
    assert result["schema_version"] == 1
    assert result["status"] == status
    assert isinstance(result["phases"], (list, dict))
    assert isinstance(result["artifacts"], list)
    if status != "ok":
        assert result["error"]["message"]
        assert result["error"]["exit_code"] == code
    return result


@pytest.mark.parametrize("command", [None, *COMMANDS])
def test_help_works_in_interpreter_without_torch(command):
    args = [command, "--help"] if command else ["--help"]
    proc = _cli(*args, no_torch=True)
    assert proc.returncode == 0, proc.stderr
    assert "usage: hi2" in proc.stdout
    assert "--json" in proc.stdout
    assert proc.stderr == ""


def test_version_works_without_compute_dependencies():
    proc = _cli("--version", no_torch=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("hi2 ")
    assert proc.stderr == ""

    from importlib.metadata import version

    installed = _cli("--version")
    assert installed.returncode == 0, installed.stderr
    assert installed.stdout.strip() == f"hi2 {version('hi2')}"


@pytest.mark.parametrize("args", [("--json", "quantize"), ("quantize", "--json")])
def test_usage_errors_are_one_json_object(args):
    result = _result(_cli(*args, no_torch=True), status="error", code=2)
    assert result["command"] == "quantize"
    assert result["error"]["type"] == "UsageError"
    assert "required" in result["error"]["message"]


def test_missing_compute_dependency_has_actionable_json_error(tmp_path):
    proc = _cli("quantize", tmp_path / "src", tmp_path / "dst", "--json", no_torch=True)
    result = _result(proc, status="error", code=3)
    assert result["error"]["category"] == "backend_resource"
    assert result["error"]["type"] == "ModuleNotFoundError"
    assert "torch" in result["error"]["message"]
    assert "torch" in proc.stderr
    assert not (tmp_path / "dst").exists()
    assert not (tmp_path / "dst.work").exists()


@pytest.fixture
def documents(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"text": f"document {i}: " + (f"word{i} " * 24)})
            for i in range(60)
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("global_flag", [False, True])
def test_calib_json_is_top_level_and_distinct_outputs_work(
    tmp_path, documents, global_flag
):
    out, held = tmp_path / "calib.txt", tmp_path / "heldout.txt"
    args = [
        "calib",
        out,
        documents,
        "--tokens",
        "10000",
        "--holdout",
        held,
        "--holdout-tokens",
        "10000",
    ]
    proc = _cli(
        *(["--json", *args] if global_flag else [*args, "--json"]), no_torch=True
    )
    result = _result(proc)
    assert result["command"] == "calib"
    assert result["tokens"] > 0 and result["held_out_tokens"] > 0
    assert str(documents) in result["sources"]
    assert "data" not in result and "result" not in result
    assert out.read_text() and held.read_text()
    assert set(out.read_text().split("\n\n")).isdisjoint(held.read_text().split("\n\n"))
    assert {a["path"] for a in result["artifacts"]} == {str(out), str(held)}
    assert result["preparation"]["chat_format"] == "role-content"


@pytest.mark.parametrize("alias", ["same", "symlink", "hardlink"])
def test_calibration_output_holdout_alias_rejected_before_mutation(
    tmp_path, documents, alias
):
    out = tmp_path / "out.txt"
    out.write_text("existing calibration sentinel", encoding="utf-8")
    held = tmp_path / "held.txt"
    if alias == "same":
        held = out
    elif alias == "symlink":
        held.symlink_to(out)
    else:
        os.link(out, held)
    before_source, before_output = documents.read_bytes(), out.read_bytes()
    proc = _cli(
        "calib",
        out,
        documents,
        "--holdout",
        held,
        "--tokenizer",
        tmp_path / "not-a-tokenizer",
        "--json",
        no_torch=True,
    )
    result = _result(proc, status="error", code=2)
    # This must be path validation, not a later missing tokenizer/compute dependency.
    assert result["error"]["type"] == "ValueError"
    assert str(out) in result["error"]["message"]
    assert documents.read_bytes() == before_source
    assert out.read_bytes() == held.read_bytes() == before_output


@pytest.mark.parametrize("which", ["calibration", "holdout"])
@pytest.mark.parametrize("symlink", [False, True])
def test_calibration_inputs_cannot_be_overwritten(tmp_path, documents, which, symlink):
    collision = documents
    if symlink:
        collision = tmp_path / "input-alias.jsonl"
        collision.symlink_to(documents)
    out, held = tmp_path / "calib.txt", tmp_path / "heldout.txt"
    if which == "calibration":
        out = collision
    else:
        held = collision
    before = documents.read_bytes()
    proc = _cli(
        "calib",
        out,
        f"{documents}:2",
        "--holdout",
        held,
        "--json",
        no_torch=True,
    )
    result = _result(proc, status="error", code=2)
    assert result["error"]["type"] == "ValueError"
    assert str(documents) in result["error"]["message"]
    assert documents.read_bytes() == before
    assert not (
        tmp_path / ("heldout.txt" if which == "calibration" else "calib.txt")
    ).exists()


def test_calibration_ancestor_collision_fails_before_reading_source(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text("not JSON: path preflight must run first", encoding="utf-8")
    result = _result(
        _cli("calib", tmp_path, source, "--json", no_torch=True),
        status="error",
        code=2,
    )
    assert result["error"]["type"] == "ValueError"
    assert str(source) in result["error"]["message"]
    assert source.read_text() == "not JSON: path preflight must run first"


def test_plain_mode_keeps_result_fields_and_readable_errors(tmp_path, documents):
    out = tmp_path / "plain.txt"
    proc = _cli("calib", out, documents, "--tokens", "100", no_torch=True)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["tokens"] > 0
    failure = _cli("calib", out, documents, "--holdout", out, no_torch=True)
    assert failure.returncode == 2
    assert failure.stdout == ""
    assert "hi2:" in failure.stderr and str(out) in failure.stderr


@pytest.mark.parametrize("fail_inside", [False, True])
def test_native_python_and_child_logs_cannot_pollute_result_stdout(
    tmp_path, documents, fail_inside
):
    script = """
import ctypes
import os
import subprocess
import sys
from hi2.__main__ import _stderr_logs, main
libc = ctypes.CDLL('msvcrt' if os.name == 'nt' else None)
libc.printf.argtypes = [ctypes.c_char_p]
try:
    with _stderr_logs():
        print('python chatter')
        os.write(1, b'fd chatter\\n')
        libc.printf(b'buffered C chatter\\n')
        subprocess.run([sys.executable, '-S', '-c', "print('child chatter')"], check=True)
        if sys.argv[3] == 'fail':
            raise ValueError('expected boundary failure')
except ValueError:
    pass
raise SystemExit(main(['calib', sys.argv[1], sys.argv[2], '--tokens', '10', '--json']))
"""
    proc = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            script,
            str(tmp_path / "out.txt"),
            str(documents),
            "fail" if fail_inside else "ok",
        ],
        cwd=ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    _result(proc)
    for text in ("python chatter", "fd chatter", "buffered C chatter", "child chatter"):
        assert text in proc.stderr
        assert text not in proc.stdout


@pytest.fixture(scope="module")
def local_tokenizer(tmp_path_factory):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    path = tmp_path_factory.mktemp("cli-tokenizer")
    backend = Tokenizer(
        WordLevel(
            {"[UNK]": 0, **{f"word{i}": i + 1 for i in range(255)}}, unk_token="[UNK]"
        )
    )
    backend.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", model_max_length=100_000
    )
    tok.chat_template = CHAT_TEMPLATE
    tok.save_pretrained(path)
    return path, tok


@pytest.mark.parametrize("override", [False, True])
def test_explicit_chat_template_and_tokenizer_really_prepare_text(
    tmp_path, local_tokenizer, override
):
    tokenizer_path, tok = local_tokenizer
    source, out = tmp_path / "chat.jsonl", tmp_path / "prepared.txt"
    messages = [
        {"role": "user", "content": "word1 word2"},
        {"role": "assistant", "content": "word3"},
    ]
    plain = "word4  word5\nplain text remains unchanged"
    source.write_text(
        json.dumps({"messages": messages}) + "\n" + json.dumps({"text": plain}),
        encoding="utf-8",
    )
    template = CHAT_TEMPLATE
    flags = ["--chat-template"]
    if override:
        template = CHAT_TEMPLATE.replace(" >> ", " :: ")
        template_file = tmp_path / "chat.jinja"
        template_file.write_text(template, encoding="utf-8")
        flags = ["--chat-template-file", template_file]
    proc = _cli(
        "calib",
        out,
        source,
        "--tokens",
        "1000",
        "--tokenizer",
        tokenizer_path,
        *flags,
        "--json",
    )
    result = _result(proc)
    expected_chat = tok.apply_chat_template(
        messages, chat_template=template, tokenize=False, add_generation_prompt=False
    )
    expected = expected_chat + "\n\n" + plain
    assert out.read_text() == expected
    assert result["tokens"] == len(tok(expected, add_special_tokens=False)["input_ids"])
    assert result["preparation"] == {
        "chat_format": "chat-template",
        "token_count": "tokenizer",
        "tokenizer_id": str(tokenizer_path.resolve()),
        "chat_template_id": "sha256:" + hashlib.sha256(template.encode()).hexdigest(),
    }
    # The same tokenizer alone counts tokens but does not silently enable rendering.
    plain_out = tmp_path / "not-templated.txt"
    plain_result = _result(
        _cli(
            "calib",
            plain_out,
            source,
            "--tokens",
            "1000",
            "--tokenizer",
            tokenizer_path,
            "--json",
        )
    )
    assert plain_out.read_text() == "user: word1 word2\nassistant: word3\n\n" + plain
    assert plain_result["preparation"]["chat_format"] == "role-content"


def test_chat_template_requires_tokenizer_before_output_write(tmp_path, documents):
    out = tmp_path / "out.txt"
    result = _result(
        _cli("calib", out, documents, "--chat-template", "--json", no_torch=True),
        status="error",
        code=2,
    )
    assert "requires --tokenizer" in result["error"]["message"]
    assert not out.exists()


def test_template_file_cannot_be_overwritten(tmp_path, documents, local_tokenizer):
    template = tmp_path / "template.jinja"
    template.write_text(CHAT_TEMPLATE, encoding="utf-8")
    result = _result(
        _cli(
            "calib",
            template,
            documents,
            "--tokenizer",
            local_tokenizer[0],
            "--chat-template-file",
            template,
            "--json",
            no_torch=True,
        ),
        status="error",
        code=2,
    )
    assert result["error"]["type"] == "ValueError"
    assert template.read_text() == CHAT_TEMPLATE


def test_minimum_holdout_is_enforced_before_any_output_write(
    tmp_path, documents, local_tokenizer
):
    out, held = tmp_path / "out.txt", tmp_path / "held.txt"
    result = _result(
        _cli(
            "calib",
            out,
            documents,
            "--holdout",
            held,
            "--tokenizer",
            local_tokenizer[0],
            "--min-holdout-tokens",
            "100000",
            "--json",
        ),
        status="error",
        code=2,
    )
    assert "hold" in result["error"]["message"].lower()
    assert not out.exists() and not held.exists()


@pytest.fixture(scope="module")
def cli_model(local_tokenizer, tmp_path_factory):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    source = tmp_path_factory.mktemp("cli-model") / "model"
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=2048,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(83)
        LlamaForCausalLM(config).eval().save_pretrained(source, safe_serialization=True)
    local_tokenizer[1].save_pretrained(source)
    return source


def _ptq_args(source, dst):
    return [
        "quantize",
        source,
        dst,
        "--devices",
        "cpu",
        "--no-qad",
        "--no-cache",
        "--effort",
        "fast",
        "--rounding",
        "nearest",
        "--rotation",
        "off",
        "--bits",
        "4",
        "--blocks",
        "64",
        "--moe-layout",
        "per-expert",
        "--include",
        r"^model\.layers\.0\.mlp\.down_proj\.weight$",
    ]


def test_cpu_quantize_logs_are_separate_and_size_keeps_human_table(cli_model, tmp_path):
    dst = tmp_path / "quantized"
    proc = _cli(*_ptq_args(cli_model, dst), "--json")
    result = _result(proc)
    assert result["allocation_bpw"] > 0
    assert (dst / "hi2_report.json").is_file()
    assert {"kind": "checkpoint", "path": str(dst)} in result["artifacts"]
    size = _result(_cli("size", dst, "--json"))
    assert size["tensor_bytes"] > 0
    human = _cli("size", dst)
    assert human.returncode == 0, human.stderr
    assert "Quantized linears" in human.stdout
    verified = _result(_cli("verify", cli_model, dst, "--json"))
    assert verified["checked"] > 0


def test_cancelled_concurrent_tuning_runs_only_cleanup_owned_scratch(
    cli_model, tmp_path
):
    import signal
    from concurrent.futures import ThreadPoolExecutor

    dst = tmp_path / "final"
    old_trial = tmp_path / "final-tune-r2"
    old_trial.mkdir()
    sentinel = old_trial / "unrelated.txt"
    sentinel.write_text("must survive both tuning runs", encoding="utf-8")
    work = tmp_path / "caller-work"
    work.mkdir()
    work_sentinel = tmp_path / "caller-work-state.txt"
    work_sentinel.write_text("caller owns this directory", encoding="utf-8")
    calib, held = tmp_path / "calib.txt", tmp_path / "tuning.txt"
    calib.write_text("word1 word2 word3 word4 " * 32, encoding="utf-8")
    held.write_text("word5 word6 " * 8192, encoding="utf-8")
    args = [
        *_ptq_args(cli_model, dst),
        "--workdir",
        work,
        "--calib",
        calib,
        "--calib-samples",
        "2",
        "--seq-len",
        "16",
        "--target-kld",
        "0.01",
        "--tune-ratios",
        "2,4",
        "--eval-text",
        held,
        "--eval-device",
        "cpu",
        "--json",
    ]
    # Interrupt real overlapping trial encodes, after their progress announces encoding.
    processes = [
        subprocess.Popen(
            _command(*args),
            cwd=ROOT,
            env=_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]

    def interrupt_encoding(process):
        diagnostics = []
        for line in process.stderr:
            diagnostics.append(line)
            if "phase 3:" in line:
                process.send_signal(signal.SIGINT)
                break
        stdout, stderr = process.communicate(timeout=180)
        return subprocess.CompletedProcess(
            process.args, process.returncode, stdout, "".join(diagnostics) + stderr
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(interrupt_encoding, process) for process in processes]
        try:
            completed = [future.result(timeout=180) for future in futures]
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
    workspaces, reports = set(), set()
    for proc in completed:
        result = _result(proc, status="interrupted", code=130)
        assert result["error"]["category"] == "interrupted"
        report = Path(
            next(
                a["path"]
                for a in result["artifacts"]
                if a["kind"] == "tune_diagnostics"
            )
        )
        diagnostic = json.loads(report.read_text())
        assert diagnostic["status"] == "interrupted"
        assert (
            diagnostic["trials"] and diagnostic["trials"][-1]["status"] == "interrupted"
        )
        workspace = Path(diagnostic["workspace"])
        assert workspace.parent == dst.parent
        assert not workspace.exists()
        assert all(not Path(trial["path"]).exists() for trial in diagnostic["trials"])
        assert report.is_file() and report.parent == dst.parent
        workspaces.add(workspace)
        reports.add(report)
    assert len(workspaces) == len(reports) == 2
    assert sentinel.read_text() == "must survive both tuning runs"
    assert list(old_trial.iterdir()) == [sentinel]
    assert work_sentinel.read_text() == "caller owns this directory"
    assert list(work.iterdir()) == []
    assert not dst.exists()
    assert not list(tmp_path.glob(".final-tune-*"))


@pytest.mark.parametrize("disabled", [False, True])
def test_calibrated_cli_runs_qad_by_default_and_honors_no_qad(
    cli_model, tmp_path, disabled
):
    calibration = tmp_path / "calibration.txt"
    calibration.write_text(" ".join(f"word{i % 20}" for i in range(128)))
    qad = tmp_path / "qad.json"
    qad.write_text(
        json.dumps(
            {
                "steps": 4,
                "samples": 2,
                "val": 1,
                "batch": 1,
                "min_steps": 1,
                "eval_every": 1,
                "min_gain": 0,
                "sampling": "head",
                "final_kl": 0,
            }
        )
    )
    output, work = tmp_path / "output", tmp_path / "work"
    args = _ptq_args(cli_model, output)
    if not disabled:
        args.remove("--no-qad")
    _result(
        _cli(
            *args,
            "--workdir",
            work,
            "--calib",
            calibration,
            "--calib-samples",
            "2",
            "--seq-len",
            "16",
            "--sens-seqs",
            "1",
            "--sens-len",
            "16",
            "--qad-config",
            qad,
            "--qad-budget",
            "0.25",
            "--json",
        )
    )
    manifest = json.loads((work / "run.json").read_text())
    completed = [
        event for event in manifest["phases"] if event["status"] == "completed"
    ]
    phases = {event["phase"] for event in completed}
    assert manifest["state"] == "succeeded"
    assert {"calibration", "encode", "validation"} <= phases
    if disabled:
        assert "qad" not in phases and "sensitivity" not in phases
        assert not (work / "qad").exists()
    else:
        assert {"qad", "sensitivity"} <= phases
        assert (work / "qad" / "layer000.pt").is_file()
        # The adaptive budget gives the first layer 2x its nominal share:
        # two actual optimizer steps here, rather than the unbudgeted four.
        layer_events = [
            event for event in completed if event["phase"] == "qad" and "layer" in event
        ]
        assert [event["step"] for event in layer_events] == [2]
