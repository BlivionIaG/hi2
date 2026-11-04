# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""CPU artifact safety: real tiny safetensors, no downloads or native runtime."""

import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
import torch

pytest.importorskip("safetensors")
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from hi2 import checkpoint  # noqa: E402
from hi2.calibrate import Config, TensorTable  # noqa: E402
from hi2.checkpoint import (  # noqa: E402
    Output,
    export_dense,
    export_portable,
    to_tensors,
    validate_checkpoint,
    verify_checkpoint,
    write_checkpoints,
)
from hi2.encode import encode_rows  # noqa: E402
from hi2.executor import Executor  # noqa: E402
from hi2.format import assemble, dequantize, tensor_bpw  # noqa: E402
from hi2.paths import check_output_paths  # noqa: E402
from hi2.stack import Hi2Checkpoint, build_stack  # noqa: E402


def _json(path, value):
    path.write_text(json.dumps(value))


def _tensor(seed=0, bits=2, block=32):
    weight = torch.randn(2, 64, generator=torch.Generator().manual_seed(seed)) * 0.1
    encoded = encode_rows(weight, torch.ones(64), bits, block, "fast")
    tensor = assemble(
        [(torch.arange(2), encoded.codes, encoded.params, encoded.consts)],
        torch.full((2,), bits),
        torch.full((2,), block),
        torch.zeros(4, dtype=torch.uint8),
        64,
    )
    return weight, tensor


def _bundle(tmp_path, *, stacked=False):
    src, dst = tmp_path / "source", tmp_path / "quantized"
    src.mkdir()
    dst.mkdir()
    source, quantized, report, tensors = {}, {}, {}, {}
    experts = []
    for expert in range(2):
        parts = []
        for i, projection in enumerate(("gate_proj", "up_proj")):
            name = f"model.layers.0.mlp.experts.{expert}.{projection}.weight"
            weight, tensor = _tensor(expert * 2 + i, bits=8 if expert else 2)
            source[name] = weight
            tensors[name] = tensor
            parts.append(tensor)
            report[name] = {"shape": list(tensor.shape), "bpw": tensor_bpw(tensor)}
            if not stacked:
                quantized.update(to_tensors(tensor, name.removesuffix(".weight")))
        experts.append(parts)
    if stacked:
        quantized.update(
            {
                f"model.layers.0.mlp.experts.gate_up.hi2_{field}": value
                for field, value in build_stack(experts).items()
            }
        )
    source["model.norm.weight"] = torch.ones(64)
    quantized["model.norm.weight"] = torch.ones(64, dtype=torch.float16)
    save_file(source, str(src / "model.safetensors"))
    save_file(quantized, str(dst / "model.safetensors"))
    _json(src / "config.json", {"model_type": "llama"})
    _json(
        dst / "config.json",
        {
            "model_type": "llama",
            "quantization_config": {
                "quant_method": "rdna_hint_flex",
                "version": "3.2",
                "moe_layout": "stacked" if stacked else "per-expert",
                **(
                    {"moe_stacks": {"gate_up": ["gate_proj", "up_proj"]}}
                    if stacked
                    else {}
                ),
            },
        },
    )
    _json(
        dst / "model.safetensors.index.json",
        {
            "metadata": {},
            "weight_map": dict.fromkeys(quantized, "model.safetensors"),
        },
    )
    _json(dst / "hi2_report.json", {"summary": {}, "tensors": report})
    return src, dst, tensors


def _mutate_shard(dst, mutate):
    path = dst / "model.safetensors"
    with safe_open(path, "pt") as f:
        names = f.keys()
        values = {name: f.get_tensor(name).clone() for name in names}
    mutate(values)
    save_file(values, str(path))


def _first_field(values, suffix):
    return next(name for name in values if name.endswith(".hi2_" + suffix))


@pytest.mark.parametrize("stacked", [False, True])
def test_full_validation_and_sampled_coverage(tmp_path, stacked):
    src, dst, tensors = _bundle(tmp_path, stacked=stacked)
    full = validate_checkpoint(dst, expected_tensors=list(tensors), log=lambda *_: None)
    assert full["mode"] == "full_structural"
    assert full["checked"] == full["expected"] == 4
    assert full["coverage"] == 1.0 and full["native_runtime"] == "not_run"
    sampled = verify_checkpoint(src, dst, max_tensors=1, log=lambda *_: None)
    assert sampled["checked"] == 1 and sampled["expected"] == 4
    assert sampled["coverage"] == 0.25
    assert sampled["structural"]["status"] == "not_run"
    assert sampled["mode"] == "sampled_numerical"
    assert sampled["worst_unweighted_energy_loss"] >= 0


@pytest.mark.parametrize("problem", ["missing_unsampled", "shape", "nonfinite"])
def test_sampled_verification_rejects_incompatible_source(tmp_path, problem):
    src, dst, tensors = _bundle(tmp_path)
    sampled = next(iter(tensors))

    def corrupt(values):
        if problem == "missing_unsampled":
            del values[list(tensors)[-1]]
        elif problem == "shape":
            values[sampled] = values[sampled].reshape(4, 32)
        else:
            values[sampled][0, 0] = float("nan")

    _mutate_shard(src, corrupt)
    with pytest.raises(ValueError):
        verify_checkpoint(src, dst, max_tensors=1, log=lambda *_: None)


@pytest.mark.parametrize("zero_reconstruction", [False, True])
def test_zero_energy_source_requires_zero_reconstruction(tmp_path, zero_reconstruction):
    src, dst, tensors = _bundle(tmp_path)
    name = next(iter(tensors))
    _mutate_shard(src, lambda values: values[name].zero_())
    if zero_reconstruction:
        # Zero ladder origin and scales encode an exact zero matrix regardless
        # of the packed codes; the report's shape and storage bpw stay unchanged.
        key = name.removesuffix(".weight") + ".hi2_row_consts"
        _mutate_shard(dst, lambda values: values[key].zero_())
        result = verify_checkpoint(src, dst, max_tensors=1, log=lambda *_: None)
        assert result["checked"] == 1
        assert result["worst_unweighted_energy_loss"] == 0
    else:
        with pytest.raises(ValueError, match="zero-energy"):
            verify_checkpoint(src, dst, max_tensors=1, log=lambda *_: None)


@pytest.mark.parametrize(
    "replacement", [{}, {"missing.weight": {"shape": [2, 64], "bpw": 2.0}}]
)
def test_empty_or_incomplete_reports_never_pass(tmp_path, replacement):
    src, dst, _ = _bundle(tmp_path)
    _json(dst / "hi2_report.json", {"tensors": replacement})
    with pytest.raises(ValueError, match="coverage|report shape"):
        validate_checkpoint(dst, log=lambda *_: None)
    with pytest.raises(ValueError, match="coverage"):
        verify_checkpoint(src, dst, log=lambda *_: None)


def test_missing_one_report_entry_and_zero_sample_rejected(tmp_path):
    src, dst, _ = _bundle(tmp_path)
    with pytest.raises(ValueError, match="zero-coverage"):
        verify_checkpoint(src, dst, max_tensors=0, log=lambda *_: None)
    report = json.loads((dst / "hi2_report.json").read_text())
    report["tensors"].pop(next(iter(report["tensors"])))
    _json(dst / "hi2_report.json", report)
    with pytest.raises(ValueError, match="coverage"):
        verify_checkpoint(src, dst, log=lambda *_: None)
    with pytest.raises(ValueError, match="report shape"):
        validate_checkpoint(dst, log=lambda *_: None)


def test_explicit_expected_coverage_and_reportless_structure(tmp_path):
    _, dst, tensors = _bundle(tmp_path)
    with pytest.raises(ValueError, match="expected tensor coverage"):
        validate_checkpoint(dst, expected_tensors=[], log=lambda *_: None)
    with pytest.raises(ValueError, match="expected tensor coverage"):
        validate_checkpoint(
            dst, expected_tensors=list(tensors)[:-1], log=lambda *_: None
        )
    (dst / "hi2_report.json").unlink()
    with pytest.raises(ValueError, match="hi2_report"):
        validate_checkpoint(dst, log=lambda *_: None)
    result = validate_checkpoint(
        dst, expected_tensors=list(tensors), require_report=False, log=lambda *_: None
    )
    assert result["checked"] == 4 and not result["report_checked"]


@pytest.mark.parametrize(
    "field,value", [("shape", [64, 2]), ("bpw", 0), ("bpw", "2.75")]
)
def test_report_measurements_must_match_encoded_tensor(tmp_path, field, value):
    src, dst, tensors = _bundle(tmp_path)
    report = json.loads((dst / "hi2_report.json").read_text())
    report["tensors"][next(iter(tensors))][field] = value
    _json(dst / "hi2_report.json", report)
    with pytest.raises(ValueError, match="report shape|report bpw"):
        validate_checkpoint(dst, log=lambda *_: None)
    with pytest.raises(ValueError, match="report shape|report bpw"):
        verify_checkpoint(src, dst, max_tensors=1, log=lambda *_: None)


@pytest.mark.parametrize(
    "filename",
    ["config.json", "model.safetensors.index.json", "hi2_report.json"],
)
def test_duplicate_json_metadata_keys_are_not_silently_overwritten(tmp_path, filename):
    _, dst, _ = _bundle(tmp_path)
    path = dst / filename
    original = json.loads(path.read_text())
    key, value = next(iter(original.items()))
    path.write_text(
        "{" + json.dumps(key) + ":" + json.dumps(value) + "," + json.dumps(original)[1:]
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_checkpoint(dst, log=lambda *_: None)


def test_quantized_container_cannot_split_fields_across_valid_shards(tmp_path):
    _, dst, _ = _bundle(tmp_path)
    moved = {}

    def split(values):
        key = _first_field(values, "codes")
        moved[key] = values.pop(key)

    _mutate_shard(dst, split)
    save_file(moved, str(dst / "split.safetensors"))
    index = json.loads((dst / "model.safetensors.index.json").read_text())
    index["weight_map"].update(dict.fromkeys(moved, "split.safetensors"))
    _json(dst / "model.safetensors.index.json", index)
    with pytest.raises(ValueError, match="split across shards"):
        validate_checkpoint(dst, log=lambda *_: None)


@pytest.mark.parametrize("stacked", [False, True])
def test_dense_and_quantized_names_cannot_claim_same_logical_tensor(tmp_path, stacked):
    _, dst, tensors = _bundle(tmp_path, stacked=stacked)
    name, tensor = next(iter(tensors.items()))
    _mutate_shard(dst, lambda values: values.update({name: dequantize(tensor)}))
    index = json.loads((dst / "model.safetensors.index.json").read_text())
    index["weight_map"][name] = "model.safetensors"
    _json(dst / "model.safetensors.index.json", index)
    with pytest.raises(ValueError, match="duplicate logical tensor"):
        validate_checkpoint(dst, log=lambda *_: None)


@pytest.mark.parametrize(
    "field,kind,match",
    [
        ("code_off", "offset", "code_off"),
        ("param_off", "offset", "param_off"),
        ("row_nbits", "width", "row_nbits"),
        ("row_block", "width", "row_block"),
        ("had", "flag", "hi2_had"),
        ("row_consts", "nan", "nonfinite"),
        ("row_consts", "overflow", "nonfinite"),
        ("codes", "dtype", "expected torch.int32"),
        ("row_consts", "shape", "shape"),
        ("codes", "missing", "coverage"),
    ],
)
def test_corrupt_quantized_buffers_identify_artifact(tmp_path, field, kind, match):
    _, dst, _ = _bundle(tmp_path)

    def corrupt(values):
        name = _first_field(values, field)
        if kind == "offset":
            values[name][1] = -1
        elif kind == "width":
            values[name][0] = 3
        elif kind == "flag":
            values[name][0] = 2
        elif kind == "nan":
            values[name][0, 0] = float("nan")
        elif kind == "overflow":
            values[name][:, 2] = 65504
            params = name.removesuffix("row_consts") + "blk_params"
            values[params][1::2] = 255
        elif kind == "dtype":
            values[name] = values[name].to(torch.int64)
        elif kind == "shape":
            values[name] = values[name].reshape(-1)
        else:
            del values[name]

    _mutate_shard(dst, corrupt)
    with pytest.raises(ValueError, match=match) as error:
        validate_checkpoint(dst, log=lambda *_: None)
    assert str(dst) in str(error.value)


@pytest.mark.parametrize(
    "kind",
    ["part_start", "part_end", "part_order", "had_shape", "scale_zero", "scale_nan"],
)
def test_stacked_bounds_and_smoothing_are_validated(tmp_path, kind):
    _, dst, _ = _bundle(tmp_path, stacked=True)

    def corrupt(values):
        key = _first_field(values, "part_off")
        if kind == "part_start":
            values[key][0] = 1
        elif kind == "part_end":
            values[key][-1] += 1
        elif kind == "part_order":
            values[key][1] = values[key][0]
        elif kind == "had_shape":
            had = _first_field(values, "had")
            values[had] = values[had].reshape(-1)
        else:
            values[key.removesuffix("part_off") + "col_scale"] = torch.full(
                (2, 64),
                0.0 if kind == "scale_zero" else float("nan"),
                dtype=torch.float16,
            )

    _mutate_shard(dst, corrupt)
    if kind.startswith("scale"):
        index = json.loads((dst / "model.safetensors.index.json").read_text())
        key = next(n for n in index["weight_map"] if n.endswith(".hi2_part_off"))
        index["weight_map"][key.removesuffix("part_off") + "col_scale"] = (
            "model.safetensors"
        )
        _json(dst / "model.safetensors.index.json", index)
    with pytest.raises(ValueError, match="part_off|hi2_had|col_scale"):
        validate_checkpoint(dst, log=lambda *_: None)


@pytest.mark.parametrize(
    "kind",
    [
        "missing_shard",
        "truncated_shard",
        "unindexed_shard",
        "extra_tensor",
        "dense_nan",
        "version",
        "index_escape",
    ],
)
def test_full_validation_checks_whole_artifact(tmp_path, kind):
    _, dst, _ = _bundle(tmp_path)
    if kind == "missing_shard":
        (dst / "model.safetensors").unlink()
    elif kind == "truncated_shard":
        path = dst / "model.safetensors"
        path.write_bytes(path.read_bytes()[:-1])
    elif kind == "unindexed_shard":
        save_file({"extra": torch.ones(1)}, str(dst / "extra.safetensors"))
    elif kind == "extra_tensor":
        _mutate_shard(dst, lambda d: d.update({"extra": torch.ones(1)}))
    elif kind == "dense_nan":
        _mutate_shard(dst, lambda d: d["model.norm.weight"].fill_(float("nan")))
    elif kind == "version":
        cfg = json.loads((dst / "config.json").read_text())
        cfg["quantization_config"]["version"] = "999"
        _json(dst / "config.json", cfg)
    else:
        index = json.loads((dst / "model.safetensors.index.json").read_text())
        index["weight_map"] = dict.fromkeys(
            index["weight_map"], "../source/model.safetensors"
        )
        _json(dst / "model.safetensors.index.json", index)
    with pytest.raises(ValueError) as error:
        validate_checkpoint(dst, log=lambda *_: None)
    assert str(dst) in str(error.value)


def test_zero_quantized_artifact_cannot_pass_reportless_validation(tmp_path):
    _, dst, _ = _bundle(tmp_path)
    save_file({"model.norm.weight": torch.ones(64)}, str(dst / "model.safetensors"))
    _json(
        dst / "model.safetensors.index.json",
        {
            "weight_map": {"model.norm.weight": "model.safetensors"},
        },
    )
    (dst / "hi2_report.json").unlink()
    with pytest.raises(ValueError, match="zero expected"):
        validate_checkpoint(dst, require_report=False, log=lambda *_: None)


def test_shared_path_alias_and_containment_guards_are_pure(tmp_path):
    src = tmp_path / "source"
    src.mkdir()
    sentinel = src / "sentinel"
    sentinel.write_text("untouched")
    alias = tmp_path / "alias"
    alias.symlink_to(src, target_is_directory=True)
    for out in (src, alias, src / "child", tmp_path):
        with pytest.raises(ValueError, match="overlaps"):
            check_output_paths(inputs=[src], outputs=[out])
    with pytest.raises(ValueError, match="overlaps"):
        check_output_paths(inputs=[], outputs=[tmp_path / "a", tmp_path / "a" / "b"])
    with pytest.raises(ValueError, match="overlaps"):
        check_output_paths(inputs=[], outputs=[src, alias])
    hardlink = tmp_path / "hardlink"
    hardlink.hardlink_to(sentinel)
    with pytest.raises(ValueError, match="overlaps"):
        check_output_paths(inputs=[sentinel], outputs=[hardlink])
    check_output_paths(inputs=[src], outputs=[tmp_path / "new"])
    assert sentinel.read_text() == "untouched"
    assert not (tmp_path / "new").exists() and not (src / "child").exists()


def test_export_aliases_and_existing_output_fail_before_writes(tmp_path):
    src, dst, _ = _bundle(tmp_path)
    alias = tmp_path / "alias"
    alias.symlink_to(dst, target_is_directory=True)
    before = (dst / "model.safetensors").read_bytes()
    for out in (src, dst, alias, dst / "child", tmp_path):
        with pytest.raises(ValueError, match="overlaps"):
            export_dense(src, dst, out)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "sentinel").write_text("keep")
    with pytest.raises(ValueError, match="exclusively owned"):
        export_dense(src, dst, occupied)
    assert (occupied / "sentinel").read_text() == "keep"
    assert (dst / "model.safetensors").read_bytes() == before


def _writer_fixture(tmp_path):
    src = tmp_path / "source"
    src.mkdir()
    tables, tensors, picks, wmap = {}, {}, {}, {}
    for i in range(2):
        name = f"model.layers.{i}.self_attn.q_proj.weight"
        weight, tensor = _tensor(i)
        shard = f"model-{i + 1:05d}-of-00002.safetensors"
        save_file({name: weight}, str(src / shard))
        wmap[name] = shard
        tensors[name] = tensor
        picks[name] = torch.zeros(2, dtype=torch.long)
        tables[name] = TensorTable(
            name,
            (2, 64),
            tensor.had,
            ((dequantize(tensor) - weight) ** 2).sum(-1).float()[None, :],
            [(2, 32)],
            float(weight.square().sum()),
            0,
        )
    _json(src / "config.json", {"model_type": "qwen2"})
    _json(src / "model.safetensors.index.json", {"weight_map": wmap})
    return src, tables, tensors, picks, wmap


@pytest.mark.parametrize("staging", [False, True])
def test_writer_restores_complete_reports_and_preserves_shard_bytes(tmp_path, staging):
    src, tables, tensors, picks, wmap = _writer_fixture(tmp_path)
    work = tmp_path / "work"
    out = work / "staging" if staging else tmp_path / "output"
    done = work / "shards_done"
    cfg, ex = Config(rounding="nearest"), Executor("cpu")
    first = Output(out, picks, {}, done, tensors)
    write_checkpoints(src, [first], work, tables, cfg, ex, log=lambda *_: None)
    original_report = json.loads(json.dumps(first.report))
    before = {shard: (out / shard).stat().st_mtime_ns for shard in wmap.values()}
    hashes = {
        shard: hashlib.sha256((out / shard).read_bytes()).hexdigest()
        for shard in wmap.values()
    }
    resumed = Output(out, picks, {}, done, tensors)
    write_checkpoints(src, [resumed], work, tables, cfg, ex, log=lambda *_: None)
    assert resumed.report == original_report
    assert set(resumed.report) == set(tables)
    assert {
        shard: (out / shard).stat().st_mtime_ns for shard in wmap.values()
    } == before
    assert {
        shard: hashlib.sha256((out / shard).read_bytes()).hexdigest()
        for shard in wmap.values()
    } == hashes
    _json(out / "hi2_report.json", {"tensors": resumed.report})
    assert (
        validate_checkpoint(out, expected_tensors=list(tables), log=lambda *_: None)[
            "checked"
        ]
        == 2
    )


@pytest.mark.parametrize(
    "recovery", ["legacy", "missing_report", "altered_shard", "partial"]
)
def test_old_or_interrupted_shards_never_invent_reports(tmp_path, recovery):
    src, tables, tensors, picks, wmap = _writer_fixture(tmp_path)
    work, out = tmp_path / "work", tmp_path / "output"
    done = work / "shards_done"
    cfg, ex = Config(rounding="nearest"), Executor("cpu")
    first = Output(out, picks, {}, done, tensors)
    write_checkpoints(src, [first], work, tables, cfg, ex, log=lambda *_: None)
    shard = next(iter(wmap.values()))
    if recovery == "legacy":
        (done / shard).write_text("ok")
    elif recovery == "missing_report":
        marker = json.loads((done / shard).read_text())
        marker["tensors"] = {}
        _json(done / shard, marker)
    elif recovery == "altered_shard":
        (out / shard).write_bytes(b"corrupt")
    else:
        (out / shard).unlink()
        (done / shard).unlink()
    logs = []
    recovered = Output(out, picks, {}, done, tensors)
    write_checkpoints(src, [recovered], work, tables, cfg, ex, log=logs.append)
    assert json.loads(json.dumps(recovered.report)) == json.loads(
        json.dumps(first.report)
    )
    marker = json.loads((done / shard).read_text())
    assert marker["schema_version"] == 1 and marker["tensors"]
    assert marker["sha256"] == hashlib.sha256((out / shard).read_bytes()).hexdigest()
    if recovery != "partial":
        assert any("re-encoding" in message for message in logs)


def test_writer_alias_preflight_precedes_any_artifact_creation(tmp_path):
    src, tables, tensors, picks, _ = _writer_fixture(tmp_path)
    work = tmp_path / "work"
    alias = tmp_path / "alias"
    alias.symlink_to(src, target_is_directory=True)
    outputs = [
        Output(tmp_path / "would-create", picks, {}, work / "done-a", tensors),
        Output(alias, picks, {}, work / "done-b", tensors),
    ]
    with pytest.raises(ValueError, match="overlaps"):
        write_checkpoints(
            src, outputs, work, tables, Config(rounding="nearest"), Executor("cpu")
        )
    assert not work.exists() and not outputs[0].dst.exists()


def _tokenizer_assets(path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]").save_pretrained(
        path
    )


@pytest.mark.parametrize("stacked", [False, True])
def test_portable_export_relocates_real_tokenizer_and_retains_quantized_bytes(
    tmp_path, stacked
):
    from transformers import AutoTokenizer

    src, dst, tensors = _bundle(tmp_path, stacked=stacked)
    _tokenizer_assets(src)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (dst / name).symlink_to(src / name)
    (src / "unneeded.bin").write_bytes(b"not an ancillary asset")
    (dst / "training-data").symlink_to(src, target_is_directory=True)
    out = tmp_path / "portable"
    original = (dst / "model.safetensors").read_bytes()
    result = export_portable(dst, out, assets_from=src)
    assert result["quantized_weights"] == "byte_identical"
    assert (out / "model.safetensors").read_bytes() == original
    assert not (out / "unneeded.bin").exists() and not (out / "training-data").exists()
    assert all(not entry.is_symlink() for entry in out.rglob("*"))
    relocated = tmp_path / "relocated"
    out.rename(relocated)
    shutil.rmtree(src)
    shutil.rmtree(dst)
    tokenizer = AutoTokenizer.from_pretrained(relocated, local_files_only=True)
    assert tokenizer("hello world", add_special_tokens=False)["input_ids"] == [1, 2]
    assert set(Hi2Checkpoint(relocated).logical()) == {
        n.removesuffix(".weight") for n in tensors
    }
    assert validate_checkpoint(relocated, log=lambda *_: None)[
        "portable_manifest_checked"
    ]
    manifest = json.loads((relocated / "hi2_portable_manifest.json").read_text())
    for name, info in manifest["files"].items():
        assert (
            info["sha256"]
            == hashlib.sha256((relocated / name).read_bytes()).hexdigest()
        )


@pytest.mark.parametrize("source_evidence", [False, True])
def test_portable_evidence_belongs_to_source_not_asset_provider(
    tmp_path, source_evidence
):
    from hi2.cache import model_fingerprint
    from hi2.quality import assess
    from hi2.runtime import file_digest

    src, dst, _ = _bundle(tmp_path)
    _tokenizer_assets(src)
    calibration = tmp_path / "calibration.jsonl"
    calibration.write_text(json.dumps({"text": "known calibration document"}) + "\n")
    run = {
        "schema_version": 1,
        "run_id": "source-owned-run",
        "source": {"path": str(src), "fingerprint": model_fingerprint(src)},
        "calibration": {
            "paths": [{"path": str(calibration), "sha256": file_digest(calibration)}],
            "ids_fingerprint": "source-calibration-ids",
        },
        # Evidence is not a tokenizer config: do not rewrite *_file fields.
        "config": {"calib_file": str(calibration)},
    }
    evidence = {
        "hi2_run.json": json.dumps(run, indent=2).encode() + b"\n",
        "hi2_quality.json": json.dumps(assess(None), indent=2).encode() + b"\n",
    }
    for name, content in evidence.items():
        (src / name).write_text('{"unrelated_asset_provider_evidence": true}\n')
        if source_evidence:
            (dst / name).write_bytes(content)
    out = tmp_path / "portable"
    export_portable(dst, out, assets_from=src)
    relocated = tmp_path / "relocated"
    out.rename(relocated)
    shutil.rmtree(dst)
    manifest = json.loads((relocated / "hi2_portable_manifest.json").read_text())
    for name, content in evidence.items():
        if source_evidence:
            assert (relocated / name).read_bytes() == content
            assert manifest["files"][name] == {
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        else:
            assert not (relocated / name).exists()
            assert name not in manifest["files"]
    assert validate_checkpoint(relocated, log=lambda *_: None)[
        "portable_manifest_checked"
    ]


@pytest.mark.parametrize("name", ["hi2_run.json", "hi2_quality.json"])
@pytest.mark.parametrize("route", ["symlink", "asset_dependency"])
def test_portable_cannot_import_asset_provider_evidence(tmp_path, name, route):
    src, dst, _ = _bundle(tmp_path)
    _tokenizer_assets(src)
    _json(src / name, {"unrelated_evidence": True})
    if route == "symlink":
        (dst / name).symlink_to(src / name)
    else:
        config = json.loads((src / "tokenizer_config.json").read_text())
        config["evidence_file"] = name
        _json(src / "tokenizer_config.json", config)
    out = tmp_path / "portable"
    with pytest.raises(ValueError, match="escapes|collides"):
        export_portable(dst, out, assets_from=src)
    assert not out.exists()


@pytest.mark.parametrize(
    "kind", ["missing", "escape", "reference_escape", "collision", "existing", "alias"]
)
def test_portable_dependencies_fail_before_creating_output(tmp_path, kind):
    src, dst, _ = _bundle(tmp_path)
    out = tmp_path / "portable"
    _tokenizer_assets(dst)
    if kind == "missing":
        (dst / "tokenizer.json").unlink()
    elif kind == "escape":
        (dst / "tokenizer.json").unlink()
        (src / "secret.json").write_text("{}")
        (dst / "tokenizer.json").symlink_to(src / "secret.json")
    elif kind in ("reference_escape", "collision"):
        config = json.loads((dst / "tokenizer_config.json").read_text())
        config["vocab_file"] = (
            "../source/private" if kind == "reference_escape" else "model.safetensors"
        )
        _json(dst / "tokenizer_config.json", config)
    elif kind == "existing":
        out.mkdir()
        (out / "sentinel").write_text("keep")
    else:
        out.symlink_to(dst, target_is_directory=True)
    with pytest.raises(ValueError):
        export_portable(dst, out)
    if kind == "existing":
        assert (out / "sentinel").read_text() == "keep"
    elif kind == "alias":
        assert out.is_symlink() and (dst / "model.safetensors").exists()
    else:
        assert not out.exists()


def test_portable_rewrites_authorized_absolute_asset_dependencies(tmp_path):
    src, dst, _ = _bundle(tmp_path)
    _tokenizer_assets(src)
    config = json.loads((src / "tokenizer_config.json").read_text())
    config["tokenizer_file"] = str(src / "tokenizer.json")
    _json(src / "tokenizer_config.json", config)
    out = tmp_path / "portable"
    export_portable(dst, out, assets_from=src)
    copied = json.loads((out / "tokenizer_config.json").read_text())
    assert copied["tokenizer_file"] == "tokenizer.json"
    (out / "tokenizer.json").write_text("tampered")
    with pytest.raises(ValueError, match="checksum/size"):
        validate_checkpoint(out, log=lambda *_: None)


def _evaluation_fixture(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    src, dst = tmp_path / "eval-source", tmp_path / "eval-quantized"
    cfg = LlamaConfig(
        vocab_size=8,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=16,
    )
    LlamaForCausalLM(cfg).eval().save_pretrained(src, safe_serialization=True)
    tok = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]").save_pretrained(
        src
    )
    dst.mkdir()
    text = tmp_path / "eval.txt"
    text.write_text("hello hello hello hello")
    old = dst.parent / (dst.name + ".eval-dense")
    old.mkdir()
    (old / "sentinel").write_text("keep")
    return src, dst, text, old


def test_failed_eval_export_cleans_only_owned_scratch(tmp_path, monkeypatch):
    from hi2 import evalq

    src, dst, text, old = _evaluation_fixture(tmp_path)
    seen = []

    def fail_export(_src, _dst, out):
        seen.append(out)
        out.mkdir()
        (out / "partial").write_text("partial export")
        raise RuntimeError("injected export failure")

    monkeypatch.setattr(evalq, "export_dense", fail_export)
    with pytest.raises(RuntimeError, match="injected export failure"):
        evalq.evaluate(
            src, dst, text, samples=1, seq_len=4, device="cpu", log=lambda *_: None
        )
    assert len(seen) == 1 and not seen[0].parent.exists()
    assert (old / "sentinel").read_text() == "keep"


def test_concurrent_evaluations_own_distinct_scratch_trees(tmp_path, monkeypatch):
    from hi2 import evalq

    src, dst, text, old = _evaluation_fixture(tmp_path)
    barrier = Barrier(2, timeout=30)
    seen = []

    def concurrent_failure(_src, _dst, out):
        seen.append(out)
        out.mkdir()
        (out / "partial").write_text("owned")
        barrier.wait()
        raise RuntimeError("injected concurrent export failure")

    monkeypatch.setattr(evalq, "export_dense", concurrent_failure)
    with ThreadPoolExecutor(2) as pool:
        futures = [
            pool.submit(
                evalq.evaluate, src, dst, text, 1, 4, "cpu", log=lambda *_: None
            )
            for _ in range(2)
        ]
        for future in futures:
            with pytest.raises(
                RuntimeError, match="injected concurrent export failure"
            ):
                future.result()
    assert len(seen) == len(set(seen)) == 2
    assert all(not path.parent.exists() for path in seen)
    assert (old / "sentinel").read_text() == "keep"


def test_portable_export_follows_only_required_local_code(tmp_path):
    _, dst, _ = _bundle(tmp_path)
    _tokenizer_assets(dst)
    config = json.loads((dst / "tokenizer_config.json").read_text())
    config["auto_map"] = {"AutoTokenizer": ["custom_tokenizer.CustomTokenizer", None]}
    _json(dst / "tokenizer_config.json", config)
    (dst / "custom_tokenizer.py").write_text(
        "from .tokenizer_helpers import helper\n"
        "class CustomTokenizer:\n"
        "    helper = staticmethod(helper)\n"
    )
    (dst / "tokenizer_helpers.py").write_text("def helper():\n    return 1\n")
    (dst / "unrelated.py").write_text("raise RuntimeError('not a dependency')\n")
    out = tmp_path / "portable"
    export_portable(dst, out)
    assert (out / "custom_tokenizer.py").read_bytes() == (
        dst / "custom_tokenizer.py"
    ).read_bytes()
    assert (out / "tokenizer_helpers.py").read_bytes() == (
        dst / "tokenizer_helpers.py"
    ).read_bytes()
    assert not (out / "unrelated.py").exists()


def test_portable_export_failure_removes_only_its_owned_tree(tmp_path, monkeypatch):
    _, dst, _ = _bundle(tmp_path)
    _tokenizer_assets(dst)
    out = tmp_path / "portable"
    sentinel = tmp_path / "sentinel"
    sentinel.write_text("keep")
    original_copy = checkpoint.shutil.copyfile

    def partial_copy(src, dest):
        original_copy(src, dest)
        raise OSError("injected copy failure")

    monkeypatch.setattr(checkpoint.shutil, "copyfile", partial_copy)
    with pytest.raises(OSError, match="injected copy failure"):
        export_portable(dst, out)
    assert not out.exists()
    assert sentinel.read_text() == "keep"
    assert validate_checkpoint(dst, log=lambda *_: None)["checked"] == 4


def test_readonly_source_snapshot_links_remain_supported(tmp_path):
    src, tables, tensors, picks, wmap = _writer_fixture(tmp_path)
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    for shard in wmap.values():
        (src / shard).rename(blobs / shard)
        (src / shard).symlink_to(blobs / shard)
    index_name = "model.safetensors.index.json"
    (src / index_name).rename(blobs / index_name)
    (src / index_name).symlink_to(blobs / index_name)
    assert checkpoint.weight_map(src) == wmap
    work = tmp_path / "work"
    out = Output(tmp_path / "output", picks, {}, work / "done", tensors)
    write_checkpoints(
        src,
        [out],
        work,
        tables,
        Config(rounding="nearest"),
        Executor("cpu"),
        log=lambda *_: None,
    )
    _json(out.dst / "hi2_report.json", {"tensors": out.report})
    assert validate_checkpoint(out.dst, log=lambda *_: None)["checked"] == 2


@pytest.mark.parametrize("name", ["model.safetensors", "model.safetensors.index.json"])
def test_full_output_validation_rejects_external_checkpoint_links(tmp_path, name):
    _, dst, _ = _bundle(tmp_path)
    external = tmp_path / name
    (dst / name).rename(external)
    (dst / name).symlink_to(external)
    with pytest.raises(ValueError, match="escapes"):
        validate_checkpoint(dst, log=lambda *_: None)


def test_index_total_size_and_stack_group_metadata_are_checked(tmp_path):
    _, dst, _ = _bundle(tmp_path, stacked=True)
    index = json.loads((dst / "model.safetensors.index.json").read_text())
    index["metadata"]["total_size"] = 1
    _json(dst / "model.safetensors.index.json", index)
    with pytest.raises(ValueError, match="total_size"):
        validate_checkpoint(dst, log=lambda *_: None)
    index["metadata"] = {}
    _json(dst / "model.safetensors.index.json", index)
    config = json.loads((dst / "config.json").read_text())
    config["quantization_config"]["moe_stacks"] = {}
    _json(dst / "config.json", config)
    with pytest.raises(ValueError, match="moe_stacks"):
        validate_checkpoint(dst, log=lambda *_: None)


def test_eval_model_load_failure_also_cleans_owned_export(tmp_path, monkeypatch):
    from hi2 import evalq

    src, dst, text, old = _evaluation_fixture(tmp_path)
    seen = []

    def invalid_model_export(_src, _dst, out):
        seen.append(out)
        out.mkdir()
        _json(out / "config.json", {"model_type": "invalid-test-architecture"})

    monkeypatch.setattr(evalq, "export_dense", invalid_model_export)
    with pytest.raises(ValueError):
        evalq.evaluate(
            src, dst, text, samples=1, seq_len=4, device="cpu", log=lambda *_: None
        )
    assert len(seen) == 1 and not seen[0].parent.exists()
    assert (old / "sentinel").read_text() == "keep"


def test_portable_manifest_dependency_cannot_overwrite_export_manifest(tmp_path):
    _, dst, _ = _bundle(tmp_path)
    _tokenizer_assets(dst)
    config = json.loads((dst / "tokenizer_config.json").read_text())
    config["vocab_file"] = "hi2_portable_manifest.json"
    _json(dst / "tokenizer_config.json", config)
    # The dependency name is reserved even when it has not been created yet.
    out = tmp_path / "portable"
    with pytest.raises(ValueError, match="collides"):
        export_portable(dst, out)
    assert not out.exists()
