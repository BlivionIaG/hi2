# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Plan, resumable runner, paired comparison, CLI and the full canned dry-run."""

import base64
import json
import struct
from pathlib import Path

import pytest

from tools.inkling_eval import cli, compare, mock_server, runner, scoring, tasks
from tools.inkling_eval.client import Client, TransportError


def test_plan_is_deterministic_and_covers_all_five_categories():
    first = tasks.build_plan(max_model_len=32768)
    second = tasks.build_plan(max_model_len=32768)
    assert first["meta"]["plan_sha256"] == second["meta"]["plan_sha256"]
    assert [item["id"] for item in first["items"]] == [item["id"] for item in second["items"]]
    counts = {name: entry["items"] for name, entry in first["meta"]["counts"].items()}
    assert counts == {"agentic": 60, "code": 40, "multilingual": 70, "long_context": 27, "math": 48}
    contexts = {item["meta"]["length"] for item in first["items"] if item["category"] == "long_context"}
    assert contexts == {8192, 16384, 32768}
    assert max(item["approx_prompt_tokens"] for item in first["items"]) < 32768
    # The 2026-11-25 budget change must not touch the 245 ids or prompts.
    assert first["meta"]["prompt_set_sha256"] == "bcafe0f05bec3262e312b3c110d6900ab9a0f95b936838aaa91d018cfd9274c8"


def test_agentic_probe_has_its_own_plan_and_keeps_controls_tool_free():
    probe = tasks.build_plan(set_name="agentic-probe", max_model_len=32768)
    again = tasks.build_plan(set_name="agentic-probe", max_model_len=32768)
    main = tasks.build_plan(max_model_len=32768)
    assert probe["meta"]["plan_sha256"] == again["meta"]["plan_sha256"]
    assert probe["meta"]["plan_sha256"] != main["meta"]["plan_sha256"]
    assert probe["meta"]["set"] == "agentic-probe"
    assert probe["meta"]["categories"] == ["agentic"]
    assert probe["meta"]["probe"]["items"] == len(probe["items"]) == 100
    assert probe["meta"]["probe"]["tool_call_items"] == 80
    assert probe["meta"]["probe"]["control_items"] == 20
    assert probe["meta"]["probe"]["mdd_pp"] == 9.0
    assert not ({item["id"] for item in probe["items"]} & {item["id"] for item in main["items"]})
    controls = [item for item in probe["items"] if item["subcategory"] == "control"]
    calls = [item for item in probe["items"] if item["subcategory"] != "control"]
    assert len(controls) == 20 and len(calls) == 80
    assert all(item["expected"].get("control") for item in controls)
    assert all(item["expected"]["name"] is None for item in controls)
    assert {item["expected"]["name"] for item in calls} == {
        "book_hotel", "rent_car", "schedule_meeting", "order_food",
        "transfer_money", "create_invoice", "search_flights", "send_email",
    }
    assert all(
        "<|content_invoke_tool_json|>" not in item["gold"]
        for item in controls
    )
    for item in probe["items"]:
        assert item["max_tokens"] == 768
        assert (scoring.score(item, item["gold"])["ok"] is True), item["id"]


def test_vision_plan_is_separate_synthetic_and_image_aware():
    vision = tasks.build_plan(set_name="vision", max_model_len=32768)
    again = tasks.build_plan(set_name="vision", max_model_len=32768)
    main = tasks.build_plan(max_model_len=32768)
    probe = tasks.build_plan(set_name="agentic-probe", max_model_len=32768)
    assert vision["meta"]["plan_sha256"] == again["meta"]["plan_sha256"]
    assert vision["meta"]["plan_sha256"] not in {
        main["meta"]["plan_sha256"], probe["meta"]["plan_sha256"]
    }
    assert vision["meta"]["set"] == "vision"
    assert vision["meta"]["categories"] == ["vision"]
    assert vision["meta"]["transport"] == "/v1/chat/completions"
    assert vision["meta"]["thinking_effort"] == "none"
    assert vision["meta"]["vision"]["items"] == len(vision["items"]) == 88
    assert vision["meta"]["vision"]["image_items"] == 80
    assert vision["meta"]["vision"]["text_controls"] == 8
    assert vision["meta"]["vision"]["mdd_pp"] == 9.5
    assert len({item["id"] for item in vision["items"]}) == 88
    assert not ({item["id"] for item in vision["items"]} & {item["id"] for item in main["items"]})
    images = [item for item in vision["items"] if item["image_parts"]]
    controls = [item for item in vision["items"] if not item["image_parts"]]
    assert len(images) == 80 and len(controls) == 8
    assert {item["meta"]["family"] for item in vision["items"]} == {
        "ocr", "charts", "count", "table", "control"
    }
    assert len({item["image_sha256"] for item in images}) == 80
    for item in vision["items"]:
        assert item["transport"] == "chat"
        assert item["max_tokens"] == 768
        assert item["prompt"] == json.dumps(
            item["messages"], sort_keys=True, separators=(",", ":")
        )
        assert scoring.score(item, item["gold"])["ok"] is True, item["id"]
    for item in controls:
        assert "image_url" not in item["prompt"]
    for item in images:
        parts = item["messages"][-1]["content"]
        image = [part for part in parts if part["type"] == "image_url"][0]
        url = image["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        data = base64.b64decode(url.split(",", 1)[1])
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        width, height = struct.unpack(">II", data[16:24])
        assert (width, height) == (480, 240)


def test_v2_truncations_raised_the_main_plan_budgets():
    assert tasks.budget_for("agentic", "hotel") == 768
    assert tasks.budget_for("code", "python_humaneval") == 2048
    assert tasks.budget_for("code", "c") == 2048
    assert tasks.budget_for("long_context", "single_needle") == 2048
    assert tasks.budget_for("math", "gsm8k") == 1536


def test_mdd_is_defined_only_when_the_item_count_can_support_it():
    from tools.inkling_eval import stats

    assert stats.mdd(6, 0.10) is None
    assert stats.mdd(100, 0.10) is not None
    assert stats.mdd(400, 0.02) is not None
    assert stats.mdd(0, 0.10) is None


def test_short_context_skips_the_lengths_it_cannot_serve():
    plan = tasks.build_plan(max_model_len=9000)
    lengths = {item["meta"]["length"] for item in plan["items"] if item["category"] == "long_context"}
    assert lengths == {8192}
    skipped = {entry["length"] for entry in plan["meta"]["skipped_contexts"]}
    assert skipped == {16384, 32768}


def _write_run(path: Path, label: str, oks: dict[str, bool], *, seed: int = 0, set_name: str = "main"):
    records = []
    for item_id, ok in oks.items():
        records.append(
            {
                "item_id": item_id,
                "category": "math",
                "subcategory": "gsm8k",
                "status": "complete",
                "score": {"ok": ok, "detail": {}},
            }
        )
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    meta = {
        "schema": 1,
        "label": label,
        "url": "http://127.0.0.1:1",
        "plan_sha256": "plan",
        "set": set_name,
        "seed": seed,
        "reasoning_effort": 0.0,
        "template": "template",
        "haystack_sha256": None,
        "decoding": {"temperature": 0, "top_p": 1, "n": 1, "seed": seed},
        "plan_counts": {"math": {"items": len(oks)}},
    }
    runner.meta_path(path).write_text(json.dumps(meta))


def test_paired_comparison_and_floor(tmp_path):
    ref = tmp_path / "ref.jsonl"
    cand = tmp_path / "cand.jsonl"
    floor = tmp_path / "floor.jsonl"
    oks = {f"math-{index:02d}": ok for index, ok in enumerate([1, 1, 1, 0, 0, 0])}
    _write_run(ref, "stock", {key: bool(value) for key, value in oks.items()})
    _write_run(cand, "hi2", {f"math-{i:02d}": bool(v) for i, v in enumerate([1, 1, 0, 1, 0, 0])})
    _write_run(floor, "stock-2", {f"math-{i:02d}": bool(v) for i, v in enumerate([0, 1, 1, 0, 0, 0])})
    report = compare.compare(ref, cand, floor_path=floor)
    assert report["status"] == "complete"
    math = report["categories"]["math"]
    assert math["n"] == 6
    assert math["ref_ok"] == 3 and math["cand_ok"] == 3
    assert math["delta"] == 0.0
    assert math["flips_ref_right"] == 1 and math["flips_cand_right"] == 1
    assert math["mcnemar_p"] == 1.0
    assert math["ci_low"] <= 0 <= math["ci_high"]
    # Six items cannot detect even a 50-point difference; report that, do not invent one.
    assert math["mdd"] is None and math["mdd_assumed_0.10"] is None
    assert report["floor"]["max_abs_delta"] == pytest.approx(1 / 6, abs=1e-4)
    assert report["floor"]["total_flips"] == 1
    text = compare.markdown(report)
    assert "determinism floor" in text and "| math | 6 |" in text


def test_comparison_flags_meta_mismatch_and_incomplete(tmp_path):
    ref = tmp_path / "ref.jsonl"
    cand = tmp_path / "cand.jsonl"
    oks = {f"math-{index:02d}": True for index in range(4)}
    _write_run(ref, "stock", oks)
    _write_run(cand, "hi2", oks, seed=1)
    report = compare.compare(ref, cand)
    assert report["status"] == "mismatched"
    assert report["comparable"] is False
    assert report["checks"]["seed"]["same"] is False
    _write_run(cand, "hi2", oks)
    kept = [line for line in cand.read_text().splitlines() if "math-03" not in line]
    cand.write_text("\n".join(kept) + "\n")
    report = compare.compare(ref, cand)
    assert report["status"] == "incomplete"
    assert report["cand"]["complete"] == 3 and report["cand"]["planned"] == 4


def _limited_plan(limit=1):
    return tasks.build_plan(
        categories=list(tasks.CATEGORIES),
        limit=limit,
        max_model_len=32768,
    )


def _serve(plan):
    answers = {mock_server.prompt_key(item["prompt"]): item["gold"] for item in plan["items"]}
    server, thread, url = mock_server.serve(answers)
    return server, thread, url


def test_runner_resumes_and_redoes_a_changed_prompt(tmp_path):
    plan = _limited_plan()
    server, thread, url = _serve(plan)
    try:
        out = tmp_path / "run.jsonl"
        client = Client(url, "canned", timeout=30, retries=0)
        first = runner.run_plan(plan, client, out, label="canned", workers=2, progress_every=100)
        assert first["requested"] == len(plan["items"]) == first["complete"]
        assert first["errors"] == 0 and first["missing"] == 0
        meta = json.loads((tmp_path / "run.jsonl.meta.json").read_text())
        assert meta["decoding"]["skip_special_tokens"] is False
        assert meta["set"] == "main"
        second = runner.run_plan(plan, client, out, label="canned", workers=2, progress_every=100)
        assert second["requested"] == 0 and second["skipped_complete"] == len(plan["items"])
        lines = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
        target = lines[0]["item_id"]
        rewritten = []
        for line in lines:
            if line["item_id"] == target:
                line = dict(line, prompt_sha256="0" * 64)
            rewritten.append(json.dumps(line))
        out.write_text("\n".join(rewritten) + "\n")
        third = runner.run_plan(plan, client, out, label="canned", workers=2, progress_every=100)
        assert third["requested"] == 1
        with pytest.raises(runner.RunMismatch):
            runner.run_plan(plan, client, out, label="other", workers=1, progress_every=100)
        unknown = client.complete("a prompt the canned model never saw", 8)
        assert unknown["text"] == mock_server.UNKNOWN_ANSWER
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_cli_plan_and_limit(tmp_path, capsys):
    assert cli.main(["plan", "--limit", "1"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert sum(entry["items"] for entry in payload["meta"]["counts"].values()) == payload["items"] == 5


def test_cli_compare_rejects_a_main_vs_probe_pair(tmp_path):
    ref = tmp_path / "ref.jsonl"
    cand = tmp_path / "cand.jsonl"
    _write_run(ref, "stock", {"probe-00": True})
    _write_run(cand, "hi2", {"probe-00": True}, set_name="agentic-probe")
    report = compare.compare(ref, cand)
    assert report["status"] == "mismatched"
    assert report["checks"]["set"]["same"] is False


def test_cli_dry_run_vision_uses_chat_and_scores_images(tmp_path, capsys):
    out_dir = tmp_path / "vision"
    assert cli.main(["dry-run", "--set", "vision", "--out-dir", str(out_dir)]) == 0
    output = capsys.readouterr().out
    assert "dry-run OK" in output and '"vision": 88' in output
    records = runner.load_records(out_dir / "dry-run.jsonl")
    meta = json.loads((out_dir / "dry-run.jsonl.meta.json").read_text())
    assert meta["set"] == "vision"
    assert meta["transport"] == "/v1/chat/completions"
    assert meta["thinking_effort"] == "none"
    assert len(records) == 88
    assert sum(1 for record in records.values() if record["transport"] == "chat") == 88
    assert sum(1 for record in records.values() if "data:image/png;base64," in record["prompt"]) == 80
    assert all((record.get("score") or {}).get("ok") for record in records.values())


def test_chat_client_checks_png_images_and_the_mock_is_image_aware(tmp_path):
    plan = tasks.build_plan(set_name="vision", max_model_len=32768)
    answers = {mock_server.prompt_key(item["prompt"]): item["gold"] for item in plan["items"]}
    server, thread, url = mock_server.serve(answers)
    client = Client(url, "canned", timeout=30, retries=0)
    try:
        item = next(item for item in plan["items"] if item["image_parts"])
        reply = client.chat(item["messages"], 64)
        assert reply["text"] == item["gold"]
        assert type(server.RequestHandlerClass).__name__ == "type"
        assert server.RequestHandlerClass.image_requests == 1
        bad = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
            ]}
        ]
        with pytest.raises(TransportError):
            client.chat(bad, 8)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_chat_transport_uses_the_same_truncation_gate(tmp_path):
    full = tasks.build_plan(set_name="vision", max_model_len=32768)
    item = full["items"][0]
    meta = dict(full["meta"])
    meta["counts"] = {"vision": {"items": 1, "subcategories": {item["subcategory"]: 1}}}
    plan = {"items": [item], "meta": meta}
    answers = {mock_server.prompt_key(item["prompt"]): item["gold"]}
    server, thread, url = mock_server.serve(answers, finish_reason="length")
    client = Client(url, "canned", timeout=30, retries=0)
    try:
        out = tmp_path / "vision-truncated.jsonl"
        summary = runner.run_plan(plan, client, out, label="canned", workers=1)
        record = runner.load_records(out)[item["id"]]
        assert record["status"] == "truncated"
        assert record["transport"] == "chat"
        assert summary["truncated"] == 1
        assert summary["reliable"] is False
        assert summary["truncation"]["failed"] == ["vision"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_cli_dry_run_agentic_probe_scores_controls(tmp_path, capsys):
    out_dir = tmp_path / "probe"
    assert cli.main(["dry-run", "--set", "agentic-probe", "--out-dir", str(out_dir)]) == 0
    output = capsys.readouterr().out
    assert "dry-run OK" in output and '"agentic": 100' in output
    records = runner.load_records(out_dir / "dry-run.jsonl")
    assert len(records) == 100
    assert sum(1 for record in records.values() if record["subcategory"] == "control") == 20
    assert all((record.get("score") or {}).get("ok") for record in records.values())


def test_cli_dry_run_covers_every_category(tmp_path, capsys):
    out_dir = tmp_path / "dry"
    assert cli.main(["dry-run", "--out-dir", str(out_dir)]) == 0
    output = capsys.readouterr().out
    assert "dry-run OK" in output
    for name in tasks.CATEGORIES:
        assert f'"{name}"' in output
    records = runner.load_records(out_dir / "dry-run.jsonl")
    assert len(records) == 245
    assert all((record.get("score") or {}).get("ok") for record in records.values())
    assert cli.main(["compare", "--ref", str(out_dir / "dry-run.jsonl"), "--cand", str(out_dir / "dry-run.jsonl"), "--out", str(tmp_path / "cmp.json")]) == 0


def test_cli_compare_reports_incomplete(tmp_path):
    ref = tmp_path / "ref.jsonl"
    cand = tmp_path / "cand.jsonl"
    oks = {f"math-{index:02d}": True for index in range(3)}
    _write_run(ref, "stock", oks)
    _write_run(cand, "hi2", oks)
    cand.write_text("\n".join(json.dumps(json.loads(line)) for line in cand.read_text().splitlines() if "math-00" not in line) + "\n")
    assert cli.main(["compare", "--ref", str(ref), "--cand", str(cand), "--out", str(tmp_path / "cmp.json")]) == 1
    assert json.loads((tmp_path / "cmp.json").read_text())["status"] == "incomplete"


def test_missing_run_file_is_an_error_not_an_empty_run(tmp_path, capsys):
    missing = tmp_path / "missing.jsonl"
    with pytest.raises(FileNotFoundError):
        compare.compare(missing, missing)
    assert (
        cli.main(
            [
                "compare",
                "--ref",
                str(missing),
                "--cand",
                str(missing),
                "--out",
                str(tmp_path / "cmp.json"),
            ]
        )
        == 2
    )
    assert "run file not found" in capsys.readouterr().err
    assert not (tmp_path / "cmp.json").exists()


def test_truncated_items_are_counted_separately_and_fail_the_gate(tmp_path):
    plan = _limited_plan()
    answers = {mock_server.prompt_key(item["prompt"]): item["gold"] for item in plan["items"]}
    server, thread, url = mock_server.serve(answers, finish_reason="length")
    try:
        out = tmp_path / "trunc.jsonl"
        client = Client(url, "canned", timeout=30, retries=0)
        first = runner.run_plan(plan, client, out, label="trunc", workers=2, progress_every=100)
        assert first["truncated"] == 5
        assert first["complete"] == 0 and first["wrong"] == 0
        assert first["reliable"] is False
        assert set(first["truncation"]["failed"]) == set(tasks.CATEGORIES)
        report = compare.compare(out, out)
        assert report["status"] == "truncated"
        assert report["truncation"]["unreliable"] is True
        assert report["categories"]["agentic"]["n"] == 0
        assert report["categories"]["agentic"]["truncated_ref"] == 1
        assert "UNRELIABLE" in compare.markdown(report)
        assert "do not interpret" in compare.markdown(report)
        second = runner.run_plan(plan, client, out, label="trunc", workers=2, progress_every=100)
        assert second["requested"] == 5
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_truncation_gate_is_strictly_above_five_percent():
    records = {
        f"agentic-{index}": {
            "category": "agentic",
            "status": "truncated" if index == 0 else "complete",
        }
        for index in range(20)
    }
    summary = runner.truncation_summary(records, {"agentic": {"items": 20}})
    assert summary["by_category"]["agentic"]["rate"] == 0.05
    assert summary["reliable"] is True
    records["agentic-1"]["status"] = "truncated"
    summary = runner.truncation_summary(records, {"agentic": {"items": 20}})
    assert summary["failed"] == ["agentic"]
    assert summary["reliable"] is False


def test_legacy_length_records_are_excluded_even_with_complete_status(tmp_path):
    ref = tmp_path / "legacy.jsonl"
    oks = {f"math-{index:02d}": True for index in range(20)}
    _write_run(ref, "legacy", oks)
    lines = [json.loads(line) for line in ref.read_text().splitlines()]
    lines[0]["finish_reason"] = "length"
    ref.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    report = compare.compare(ref, ref)
    assert report["status"] == "complete"
    assert report["ref"]["truncated"] == 1
    assert report["ref"]["complete"] == 19
    assert report["pairing"]["paired"] == 19
    assert report["truncation"]["ref"]["by_category"]["math"]["truncated"] == 1
    lines[1]["finish_reason"] = "length"
    ref.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    report = compare.compare(ref, ref)
    assert report["status"] == "truncated"


def test_cli_run_exit_follows_the_truncation_gate(tmp_path, monkeypatch):
    class FakeClient:
        finish_reason = "length"

        def __init__(self, *args, **kwargs):
            self.url = "http://fake"

        def max_model_len(self):
            return 32768

        def model_name(self):
            return "fake"

        def complete(self, prompt, max_tokens):
            return {"text": "x", "finish_reason": self.finish_reason, "usage": {}}

    monkeypatch.setattr(cli, "Client", FakeClient)
    assert (
        cli.main(
            [
                "run",
                "--url",
                "http://fake",
                "--label",
                "truncated",
                "--out",
                str(tmp_path / "gate.jsonl"),
                "--limit",
                "1",
            ]
        )
        == 1
    )
    FakeClient.finish_reason = "stop"
    assert (
        cli.main(
            [
                "run",
                "--url",
                "http://fake",
                "--label",
                "complete",
                "--out",
                str(tmp_path / "ok.jsonl"),
                "--limit",
                "1",
            ]
        )
        == 0
    )


def test_client_requests_special_tokens(monkeypatch):
    client = Client("http://example.invalid", "m")
    captured = {}

    def fake_request(path, body=None):
        captured.update(body or {})
        return {"choices": [{"text": "x", "finish_reason": "stop"}], "usage": {}}

    monkeypatch.setattr(client, "_request", fake_request)
    client.complete("prompt", 16)
    assert captured["skip_special_tokens"] is False
