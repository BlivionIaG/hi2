# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
import json
import math
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from hi2.checkpoint import export_portable, validate_checkpoint
from hi2.quality import (
    QualityRejected,
    assess,
    check_evaluation_split,
    evaluate_quality,
)
from hi2.runtime import file_digest


@pytest.fixture
def measured_metrics():
    return {
        "tokens": 100,
        "ppl_original": 10,
        "ppl_quantized": 11,
        "kld_nats_per_token": 0.02,
    }


def test_acceptance_uses_final_measurements_not_prediction():
    measured = {
        "tokens": 100,
        "ppl_original": 10,
        "ppl_quantized": 11,
        "kld_nats_per_token": 0.02,
    }
    assert assess(measured, max_kld=0.02, max_ppl_ratio=1.1)["status"] == "target_met"
    rejected = assess(measured, max_kld=0.01, max_ppl_ratio=1.05)
    assert rejected["status"] == "target_not_met"
    assert rejected["failures"] == ["max_kld", "max_ppl_ratio"]
    assert rejected["metrics"]["ppl_ratio"] == 1.1


@pytest.mark.parametrize(
    "metrics",
    [
        None,
        {},
        {"tokens": 0},
        {
            "tokens": 10,
            "ppl_original": 5,
            "ppl_quantized": float("inf"),
            "kld_nats_per_token": 0,
        },
    ],
)
def test_unmeasured_or_invalid_quality_cannot_pass(metrics):
    assert assess(metrics, max_kld=0.01)["status"] == "not_measured"


@pytest.mark.parametrize("gate", ["max_kld", "max_ppl_ratio"])
@pytest.mark.parametrize("limit", [float("nan"), float("inf"), -1, True, False, "0.01"])
def test_invalid_gate_is_rejected(gate, limit):
    with pytest.raises(ValueError):
        assess(None, **{gate: limit})


@pytest.mark.parametrize(
    "field", ["tokens", "ppl_original", "ppl_quantized", "kld_nats_per_token"]
)
@pytest.mark.parametrize(
    "value", [None, "1", float("nan"), float("inf"), -float("inf")]
)
def test_each_required_measurement_must_be_finite_numeric(
    measured_metrics, field, value
):
    measured_metrics[field] = value
    report = assess(measured_metrics, max_kld=1)
    assert report["status"] == "not_measured"
    assert report["failures"]
    assert measured_metrics[field] is value
    if isinstance(value, float) and not math.isfinite(value):
        assert report["metrics"][field] is None
    else:
        assert report["metrics"][field] == value
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("field", ["tokens", "ppl_original", "ppl_quantized"])
@pytest.mark.parametrize("value", [0, -1])
def test_nonpositive_tokens_or_perplexity_cannot_pass(measured_metrics, field, value):
    measured_metrics[field] = value
    report = assess(measured_metrics, max_kld=1)
    assert report["status"] == "not_measured"
    assert report["failures"]


@pytest.mark.parametrize(
    "field", ["tokens", "ppl_original", "ppl_quantized", "kld_nats_per_token"]
)
@pytest.mark.parametrize("value", [False, True])
@pytest.mark.parametrize("criteria", [{}, {"max_kld": 100}, {"max_ppl_ratio": 100}])
def test_boolean_measurements_are_not_quality_evidence(
    measured_metrics, field, value, criteria
):
    measured_metrics[field] = value
    report = assess(measured_metrics, **criteria)
    assert report["status"] == "not_measured"
    assert report["failures"]


@pytest.mark.parametrize("criteria", [{}, {"max_kld": 1}, {"max_ppl_ratio": 2}])
def test_overflowed_perplexity_ratio_cannot_pass_or_poison_report(
    measured_metrics, criteria
):
    measured_metrics.update(ppl_original=1e-308, ppl_quantized=1e308)
    report = assess(measured_metrics, **criteria)
    assert report["status"] == "not_measured"
    assert report["failures"]
    assert report["metrics"] == measured_metrics
    json.dumps(report, allow_nan=False)


def test_unmeasured_report_retains_only_finite_optional_measurements(measured_metrics):
    measured_metrics.update(tokens=0, ear=float("nan"), top1_agreement=float("inf"))
    original = json.dumps(measured_metrics, sort_keys=True)
    report = assess(measured_metrics, max_kld=1)
    assert report["status"] == "not_measured"
    assert report["metrics"]["tokens"] == 0
    assert report["metrics"]["ppl_original"] == measured_metrics["ppl_original"]
    assert report["metrics"]["ear"] is None
    assert report["metrics"]["top1_agreement"] is None
    assert json.dumps(measured_metrics, sort_keys=True) == original
    json.dumps(report, allow_nan=False)


def test_perplexity_gate_recomputes_ratio_without_mutating_measurements(
    measured_metrics,
):
    measured_metrics["ppl_ratio"] = 0
    report = assess(measured_metrics, max_ppl_ratio=1.05)
    assert report["status"] == "target_not_met"
    assert report["failures"] == ["max_ppl_ratio"]
    assert report["metrics"]["ppl_ratio"] == 1.1
    assert measured_metrics["ppl_ratio"] == 0


def test_exact_document_leakage_is_rejected_across_files(tmp_path):
    train = tmp_path / "train.jsonl"
    final = tmp_path / "final.jsonl"
    train.write_text(json.dumps({"text": "shared evaluation document"}) + "\n")
    final.write_text(
        json.dumps({"text": "independent document"}) + "\n" + train.read_text()
    )
    with pytest.raises(ValueError, match="overlaps"):
        check_evaluation_split(final, [train])
    final.write_text(json.dumps({"text": "independent document"}) + "\n")
    report = check_evaluation_split(final, [train])
    assert report["documents"] == 1
    assert report["exact_overlap_checked_against"] == [str(train)]
    assert report["semantic_deduplication"] == "not_measured"


def test_empty_evaluation_corpus_is_not_a_held_out_split(tmp_path):
    text = tmp_path / "empty.txt"
    text.write_text(" \n\n\t\n")
    with pytest.raises(ValueError):
        check_evaluation_split(text, [])


def test_evaluation_symlink_cannot_alias_calibration_input(tmp_path):
    calibration = tmp_path / "calibration.jsonl"
    calibration.write_text(json.dumps({"text": "calibration document"}) + "\n")
    alias = tmp_path / "held-out.jsonl"
    alias.symlink_to(calibration)
    with pytest.raises(ValueError):
        check_evaluation_split(alias, [calibration])


@pytest.fixture
def quality_artifact(tmp_path, monkeypatch):
    """A real local causal LM with one deterministically encoded HI2 projection."""
    import torch
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    from hi2.cache import model_fingerprint
    from hi2.checkpoint import to_tensors
    from hi2.encode import encode_rows
    from hi2.format import assemble, tensor_bpw
    from hi2.runtime import RunSession

    monkeypatch.delenv("HI2_NGRAM_MMAP", raising=False)
    src, dst = tmp_path / "source", tmp_path / "quantized"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=8,
                hidden_size=64,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                max_position_embeddings=16,
            )
        ).eval()
    model.save_pretrained(src, safe_serialization=True)
    tokenizer = Tokenizer(
        WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]"
    ).save_pretrained(src)
    state = {
        name: weight.detach().to(torch.float16).contiguous()
        for name, weight in model.state_dict().items()
    }
    name = "model.layers.0.mlp.up_proj.weight"
    weight = state.pop(name).float()
    encoded = encode_rows(weight, torch.ones(64), 2, 32, "fast")
    tensor = assemble(
        [(torch.arange(64), encoded.codes, encoded.params, encoded.consts)],
        torch.full((64,), 2),
        torch.full((64,), 32),
        torch.zeros(4, dtype=torch.uint8),
        64,
    )
    state.update(to_tensors(tensor, name.removesuffix(".weight")))
    calibration = tmp_path / "calibration.jsonl"
    calibration.write_text(
        json.dumps({"text": "calibration only calibration only"}) + "\n"
    )
    text = tmp_path / "held-out.txt"
    text.write_text("hello world hello world hello world hello world")
    identity = {
        "source": {"path": str(src), "fingerprint": model_fingerprint(src)},
        "calibration": {
            "paths": [{"path": str(calibration), "sha256": file_digest(calibration)}]
        },
    }
    with RunSession(tmp_path / "work", identity, [dst]) as session:
        save_file(state, str(dst / "model.safetensors"), metadata={"format": "pt"})
        config = json.loads((src / "config.json").read_text())
        config["quantization_config"] = {
            "quant_method": "rdna_hint_flex",
            "version": "3.2",
            "moe_layout": "per-expert",
        }
        (dst / "config.json").write_text(json.dumps(config))
        (dst / "hi2_report.json").write_text(
            json.dumps(
                {"tensors": {name: {"shape": [64, 64], "bpw": tensor_bpw(tensor)}}}
            )
        )
        session.complete(validate_checkpoint(dst, log=lambda *_: None))
    return src, dst, text, calibration


def _measure(src, dst, text, **kwargs):
    return evaluate_quality(
        src,
        dst,
        text,
        samples=1,
        seq_len=8,
        device="cpu",
        dtype="float32",
        log=lambda *_: None,
        **kwargs,
    )


def _bundle_bytes(dst):
    return {
        path.relative_to(dst).as_posix(): path.read_bytes()
        for path in dst.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("portable", [False, True])
@pytest.mark.parametrize("change", ["replaced", "missing"])
def test_recorded_calibration_is_verified_before_overlap_evaluation(
    quality_artifact, tmp_path, monkeypatch, portable, change
):
    from hi2 import quality

    src, dst, _, calibration = quality_artifact
    text = tmp_path / "leaked.jsonl"
    text.write_text(calibration.read_text() * 2)
    if portable:
        output = tmp_path / "portable"
        export_portable(dst, output, assets_from=src)
        dst = output
    before = _bundle_bytes(dst)
    if change == "replaced":
        stat = calibration.stat()
        calibration.write_text(
            calibration.read_text().replace("calibration", "independent")
        )
        os.utime(calibration, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        assert calibration.stat().st_size == stat.st_size
        error = ValueError
    else:
        calibration.unlink()
        error = FileNotFoundError

    def unexpected_overlap_check(*args, **kwargs):
        pytest.fail("recorded identity must be verified before document comparison")

    monkeypatch.setattr(quality, "check_evaluation_split", unexpected_overlap_check)
    with pytest.raises(error):
        _measure(src, dst, text)
    assert _bundle_bytes(dst) == before
    assert (
        validate_checkpoint(dst, log=lambda *_: None)["portable_manifest_checked"]
        is portable
    )


@pytest.mark.parametrize("digest", [None, True, "", "a" * 63, "g" * 64])
def test_recorded_calibration_requires_a_valid_digest(
    quality_artifact, monkeypatch, digest
):
    from hi2 import quality

    src, dst, text, _ = quality_artifact
    marker = dst / "hi2_run.json"
    manifest = json.loads(marker.read_text())
    identity = manifest["calibration"]["paths"][0]
    if digest is None:
        del identity["sha256"]
    else:
        identity["sha256"] = digest
    marker.write_text(json.dumps(manifest))
    before = _bundle_bytes(dst)

    def unexpected_overlap_check(*args, **kwargs):
        pytest.fail("invalid provenance must not reach document comparison")

    monkeypatch.setattr(quality, "check_evaluation_split", unexpected_overlap_check)
    with pytest.raises(ValueError):
        _measure(src, dst, text)
    assert _bundle_bytes(dst) == before


@pytest.mark.parametrize(
    "calibration",
    [
        None,
        {"paths": None},
        {"paths": [None]},
        {"paths": [{}]},
        {"paths": [{"path": "", "sha256": "0" * 64}]},
        {"paths": [{"path": "relative.jsonl", "sha256": "0" * 64}]},
    ],
)
def test_invalid_recorded_calibration_cannot_be_treated_as_unrecorded(
    quality_artifact, calibration
):
    src, dst, text, _ = quality_artifact
    marker = dst / "hi2_run.json"
    manifest = json.loads(marker.read_text())
    manifest["calibration"] = calibration
    marker.write_text(json.dumps(manifest))
    before = _bundle_bytes(dst)
    with pytest.raises(ValueError):
        _measure(src, dst, text)
    assert _bundle_bytes(dst) == before


@pytest.mark.parametrize("recorded", [False, True])
def test_explicit_exclusions_and_legacy_provenance_remain_honest(
    quality_artifact, tmp_path, recorded
):
    src, dst, text, calibration = quality_artifact
    if not recorded:
        (dst / "hi2_run.json").unlink()
    tuning = tmp_path / "tuning.txt"
    tuning.write_text("selection only selection only")
    report = _measure(src, dst, text, excluded=[tuning])
    assert report["status"] == "measured"
    assert report["reference"]["matches_quantization_source"] is (
        True if recorded else None
    )
    assert report["reference"]["auxiliary_inputs"] == []
    assert report["dataset"]["exact_overlap_checked_against"] == (
        [str(calibration)] if recorded else []
    ) + [str(tuning)]
    assert report["dataset"]["semantic_deduplication"] == "not_measured"
    before = _bundle_bytes(dst)
    tuning.write_text(text.read_text())
    with pytest.raises(ValueError):
        _measure(src, dst, text, excluded=[tuning])
    assert _bundle_bytes(dst) == before


@pytest.fixture
def auxiliary_table(tmp_path):
    import torch
    from safetensors.torch import save_file

    table = tmp_path / "ngram.safetensors"
    save_file({"ngram_embedding.weight": torch.ones(8, 64)}, str(table))
    return table


@pytest.mark.parametrize("portable", [False, True])
def test_quality_records_matching_auxiliary_input_identity(
    quality_artifact, auxiliary_table, tmp_path, monkeypatch, portable
):
    src, dst, text, _ = quality_artifact
    inputs = [
        {
            "role": "ngram_mmap",
            "path": str(auxiliary_table.resolve()),
            "sha256": file_digest(auxiliary_table),
        }
    ]
    marker = dst / "hi2_run.json"
    manifest = json.loads(marker.read_text())
    manifest["source"]["auxiliary_inputs"] = inputs
    marker.write_text(json.dumps(manifest))
    if portable:
        output = tmp_path / "portable"
        export_portable(dst, output, assets_from=src)
        dst = output
    alias = tmp_path / "ngram-alias.safetensors"
    alias.symlink_to(auxiliary_table)
    monkeypatch.setenv("HI2_NGRAM_MMAP", str(alias))
    report = _measure(src, dst, text)
    assert report["status"] == "measured"
    assert report["reference"]["auxiliary_inputs"] == inputs
    assert report["reference"]["matches_quantization_source"] is True
    assert report["metrics"]["tokens"] == 7
    assert json.loads((dst / "hi2_quality.json").read_text()) == report
    assert (
        validate_checkpoint(dst, log=lambda *_: None)["portable_manifest_checked"]
        is portable
    )


@pytest.mark.parametrize("portable", [False, True])
@pytest.mark.parametrize(
    "change", ["unset", "missing", "changed", "different", "unrecorded"]
)
def test_quality_rejects_auxiliary_input_drift_before_evaluation(
    quality_artifact, auxiliary_table, tmp_path, monkeypatch, portable, change
):
    import torch
    from safetensors.torch import save_file

    from hi2 import evalq

    src, dst, text, _ = quality_artifact
    marker = dst / "hi2_run.json"
    manifest = json.loads(marker.read_text())
    if change != "unrecorded":
        manifest["source"]["auxiliary_inputs"] = [
            {
                "role": "ngram_mmap",
                "path": str(auxiliary_table.resolve()),
                "sha256": file_digest(auxiliary_table),
            }
        ]
    marker.write_text(json.dumps(manifest))
    if portable:
        output = tmp_path / "portable"
        export_portable(dst, output, assets_from=src)
        dst = output
    before = _bundle_bytes(dst)
    monkeypatch.setenv("HI2_NGRAM_MMAP", str(auxiliary_table))
    if change == "unset":
        monkeypatch.delenv("HI2_NGRAM_MMAP")
    elif change == "missing":
        auxiliary_table.unlink()
    elif change == "changed":
        stat = auxiliary_table.stat()
        save_file({"ngram_embedding.weight": torch.zeros(8, 64)}, str(auxiliary_table))
        os.utime(auxiliary_table, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        assert auxiliary_table.stat().st_size == stat.st_size
    elif change == "different":
        other = tmp_path / "other-ngram.safetensors"
        shutil.copy2(auxiliary_table, other)
        monkeypatch.setenv("HI2_NGRAM_MMAP", str(other))

    def unexpected_evaluation(*args, **kwargs):
        pytest.fail("auxiliary identity drift must be rejected before evaluation")

    monkeypatch.setattr(evalq, "evaluate", unexpected_evaluation)
    with pytest.raises(FileNotFoundError if change == "missing" else ValueError):
        _measure(src, dst, text, strict=False)
    assert _bundle_bytes(dst) == before
    assert (
        validate_checkpoint(dst, log=lambda *_: None)["portable_manifest_checked"]
        is portable
    )


@pytest.mark.parametrize("portable", [False, True])
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("logit_scale", [1e3, 1e38], ids=["infinite-ppl", "nan-logits"])
def test_nonfinite_evaluation_publishes_json_safe_rejection(
    quality_artifact, tmp_path, portable, strict, logit_scale
):
    from safetensors.torch import load_file, save_file

    from hi2.cache import model_fingerprint

    src, dst, text, _ = quality_artifact
    # Finite weights give constant positive hidden states and a wrong-token logit.
    # The smaller scale overflows perplexity; the larger overflows the logit itself.
    weights = load_file(str(src / "model.safetensors"))
    for weight in weights.values():
        weight.zero_()
    weights["model.embed_tokens.weight"].fill_(1)
    weights["model.norm.weight"].fill_(1)
    weights["lm_head.weight"][0].fill_(logit_scale)
    save_file(weights, str(src / "model.safetensors"), metadata={"format": "pt"})
    marker = dst / "hi2_run.json"
    manifest = json.loads(marker.read_text())
    manifest["source"]["fingerprint"] = model_fingerprint(src)
    marker.write_text(json.dumps(manifest))
    if portable:
        output = tmp_path / "portable"
        export_portable(dst, output, assets_from=src)
        dst = output
    before = _bundle_bytes(dst)
    if strict:
        with pytest.raises(QualityRejected) as rejected:
            _measure(src, dst, text, max_kld=1, strict=True)
        report = rejected.value.report
    else:
        report = _measure(src, dst, text, max_kld=1, strict=False)
    assert report["status"] == "not_measured"
    assert report["failures"]
    assert report["metrics"]["tokens"] == 7
    assert report["metrics"]["ppl_original"] is None
    assert math.isfinite(report["metrics"]["ppl_quantized"])
    assert report["metrics"]["ppl_quantized"] > 0
    assert "ppl_ratio" not in report["metrics"]
    if logit_scale == 1e38:
        assert report["metrics"]["kld_nats_per_token"] is None
        assert report["metrics"]["ear"] is None
    else:
        assert math.isfinite(report["metrics"]["kld_nats_per_token"])
    json.dumps(report, allow_nan=False)
    assert json.loads((dst / "hi2_quality.json").read_text()) == report
    assert (
        validate_checkpoint(dst, log=lambda *_: None)["portable_manifest_checked"]
        is portable
    )
    after = _bundle_bytes(dst)
    for name, content in before.items():
        if name not in {"hi2_quality.json", "hi2_portable_manifest.json"}:
            assert after[name] == content


def test_non_strict_quality_retains_failed_target(quality_artifact):
    src, dst, text, _ = quality_artifact
    report = _measure(src, dst, text, max_ppl_ratio=0, strict=False)
    assert report["status"] == "target_not_met"
    assert report["failures"] == ["max_ppl_ratio"]
    assert report["metrics"]["tokens"] == 7
    assert report["metrics"]["ppl_ratio"] > 0
    assert json.loads((dst / "hi2_quality.json").read_text()) == report


@pytest.mark.parametrize("prior_measurement", [False, True])
def test_relocated_portable_quality_evaluate_validate_evaluate(
    quality_artifact, tmp_path, prior_measurement
):
    src, dst, text, calibration = quality_artifact
    run_bytes = (dst / "hi2_run.json").read_bytes()
    weight_bytes = (dst / "model.safetensors").read_bytes()
    previous_report = None
    if prior_measurement:
        _measure(src, dst, text)
        previous_report = (dst / "hi2_quality.json").read_bytes()
    portable = tmp_path / "portable"
    export_portable(dst, portable, assets_from=src)
    assert (portable / "hi2_run.json").read_bytes() == run_bytes
    if previous_report is not None:
        assert (portable / "hi2_quality.json").read_bytes() == previous_report
    relocated = tmp_path / "relocated"
    portable.rename(relocated)
    reference = tmp_path / "relocated-reference"
    src.rename(reference)
    shutil.rmtree(dst)
    original_manifest = json.loads(
        (relocated / "hi2_portable_manifest.json").read_text()
    )
    for criteria in ({"max_ppl_ratio": 100}, {}):
        report = _measure(reference, relocated, text, **criteria)
        assert report["status"] == ("target_met" if criteria else "measured")
        assert report["metrics"]["tokens"] == 7
        assert report["reference"] == {
            "source_fingerprint": json.loads(run_bytes)["source"]["fingerprint"],
            "auxiliary_inputs": [],
            "matches_quantization_source": True,
        }
        assert report["artifact"] == {
            "path": str(relocated),
            "run_id": json.loads(run_bytes)["run_id"],
        }
        assert report["dataset"]["exact_overlap_checked_against"] == [str(calibration)]
        assert report["native_runtime"] == "not_measured"
        assert json.loads((relocated / "hi2_quality.json").read_text()) == report
        validated = validate_checkpoint(relocated, log=lambda *_: None)
        assert validated["portable_manifest_checked"]
        assert validated["native_runtime"] == "not_run"
        manifest = json.loads((relocated / "hi2_portable_manifest.json").read_text())
        assert manifest["files"]["hi2_quality.json"] == {
            "size_bytes": (relocated / "hi2_quality.json").stat().st_size,
            "sha256": file_digest(relocated / "hi2_quality.json"),
        }
        assert {
            name: info
            for name, info in manifest["files"].items()
            if name != "hi2_quality.json"
        } == {
            name: info
            for name, info in original_manifest["files"].items()
            if name != "hi2_quality.json"
        }
        assert (relocated / "hi2_run.json").read_bytes() == run_bytes
        assert (relocated / "model.safetensors").read_bytes() == weight_bytes


@pytest.mark.parametrize("reason", ["reference", "overlap", "missing_calibration"])
def test_portable_quality_retains_source_and_calibration_rejections(
    quality_artifact, tmp_path, reason
):
    from safetensors.torch import load_file, save_file

    src, dst, text, calibration = quality_artifact
    portable = tmp_path / "portable"
    export_portable(dst, portable, assets_from=src)
    relocated = tmp_path / "relocated"
    portable.rename(relocated)
    shutil.rmtree(dst)
    before = _bundle_bytes(relocated)
    if reason == "reference":
        other = tmp_path / "unrelated-reference"
        shutil.copytree(src, other)
        weights = load_file(str(other / "model.safetensors"))
        weights["model.embed_tokens.weight"].add_(1)
        save_file(weights, str(other / "model.safetensors"), metadata={"format": "pt"})
        src = other
        error, message = ValueError, "reference differs"
    elif reason == "overlap":
        text = tmp_path / "overlap.jsonl"
        text.write_text(
            json.dumps({"text": "independent held-out document"})
            + "\n"
            + calibration.read_text()
        )
        error, message = ValueError, "overlaps"
    else:
        calibration.unlink()
        error, message = FileNotFoundError, "calibration.jsonl"
    with pytest.raises(error, match=message):
        _measure(src, relocated, text)
    assert _bundle_bytes(relocated) == before
    assert validate_checkpoint(relocated, log=lambda *_: None)[
        "portable_manifest_checked"
    ]


def test_rejected_portable_measurement_publishes_a_valid_bundle(
    quality_artifact, tmp_path
):
    src, dst, text, _ = quality_artifact
    portable = tmp_path / "portable"
    export_portable(dst, portable, assets_from=src)
    with pytest.raises(QualityRejected) as rejected:
        _measure(src, portable, text, max_ppl_ratio=0)
    report = rejected.value.report
    assert report["status"] == "target_not_met"
    assert report["failures"] == ["max_ppl_ratio"]
    assert report["metrics"]["tokens"] == 7
    assert report["native_runtime"] == "not_measured"
    assert json.loads((portable / "hi2_quality.json").read_text()) == report
    assert validate_checkpoint(portable, log=lambda *_: None)[
        "portable_manifest_checked"
    ]
    assert _measure(src, portable, text)["status"] == "measured"
    assert validate_checkpoint(portable, log=lambda *_: None)[
        "portable_manifest_checked"
    ]


@pytest.mark.parametrize("prior_measurement", [False, True])
@pytest.mark.parametrize(
    "publication", ["hi2_quality.json", "hi2_portable_manifest.json"]
)
@pytest.mark.parametrize("after_replace", [False, True])
def test_portable_quality_publication_failure_restores_prior_bundle(
    quality_artifact,
    tmp_path,
    monkeypatch,
    prior_measurement,
    publication,
    after_replace,
):
    from hi2 import runtime

    src, dst, text, _ = quality_artifact
    if prior_measurement:
        _measure(src, dst, text)
    portable = tmp_path / "portable"
    export_portable(dst, portable, assets_from=src)
    before = _bundle_bytes(portable)
    original_replace = runtime.os.replace
    injected = False

    def fail_commit(source, destination):
        nonlocal injected
        if Path(destination) == portable / publication and not injected:
            injected = True
            if after_replace:
                original_replace(source, destination)
            raise OSError("injected quality publication failure")
        return original_replace(source, destination)

    monkeypatch.setattr(runtime.os, "replace", fail_commit)
    with pytest.raises(OSError, match="injected quality publication failure"):
        _measure(src, portable, text, max_ppl_ratio=100)
    assert injected
    assert _bundle_bytes(portable) == before
    assert validate_checkpoint(portable, log=lambda *_: None)[
        "portable_manifest_checked"
    ]
    assert _measure(src, portable, text)["status"] == "measured"
    assert validate_checkpoint(portable, log=lambda *_: None)[
        "portable_manifest_checked"
    ]


def test_portable_quality_publication_excludes_other_output_owners(
    quality_artifact, tmp_path, monkeypatch
):
    from hi2 import runtime
    from hi2.runtime import exclusive_lock, object_digest

    src, dst, text, _ = quality_artifact
    portable = tmp_path / "portable"
    export_portable(dst, portable, assets_from=src)
    report_committed, finish = Event(), Event()
    original_replace = runtime.os.replace

    def pause_after_report(source, destination):
        original_replace(source, destination)
        if Path(destination) == portable / "hi2_quality.json":
            report_committed.set()
            assert finish.wait(30), "publication was not released"

    monkeypatch.setattr(runtime.os, "replace", pause_after_report)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(_measure, src, portable, text)
        try:
            assert report_committed.wait(30), "report did not reach its commit boundary"
            lock_name = object_digest(str(portable.resolve()))[:24]
            lock = portable.parent / f".hi2-output-{lock_name}.lock"
            with (
                pytest.raises(RuntimeError, match="resource is busy"),
                exclusive_lock(lock),
            ):
                pass
            with pytest.raises(RuntimeError, match="resource is busy"):
                _measure(src, portable, text)
            with pytest.raises(RuntimeError, match="resource is busy"):
                export_portable(portable, tmp_path / "concurrent-export")
            assert not (tmp_path / "concurrent-export").exists()
        finally:
            finish.set()
        assert future.result(timeout=30)["status"] == "measured"
    assert validate_checkpoint(portable, log=lambda *_: None)[
        "portable_manifest_checked"
    ]


@pytest.mark.parametrize("tamper", ["unlisted_file", "quality", "run"])
def test_quality_publication_does_not_bless_portable_tampering(
    quality_artifact, tmp_path, tamper
):
    src, dst, text, _ = quality_artifact
    _measure(src, dst, text)
    portable = tmp_path / "portable"
    export_portable(dst, portable, assets_from=src)
    if tamper == "unlisted_file":
        (portable / "unlisted.json").write_text("{}\n")
    else:
        (portable / f"hi2_{tamper}.json").write_text("{}\n")
    before = _bundle_bytes(portable)
    with pytest.raises(ValueError, match="portable file coverage|checksum/size"):
        _measure(src, portable, text)
    assert _bundle_bytes(portable) == before
