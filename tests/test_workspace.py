# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""CPU-only operator-root, recipe-capability and private-cache contracts."""

import copy
import json
import os
import pickle
import stat
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from hi2.runtime import exclusive_lock
from hi2.workspace import WorkspacePolicy, WorkspacePolicyError


@pytest.fixture
def workspace(tmp_path):
    tmp_path = tmp_path.resolve()
    models, calibration, outputs = [
        tmp_path / name for name in ("models", "calibration", "outputs")
    ]
    for root in (models, calibration, outputs):
        root.mkdir()
    source = models / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "hidden_size": 4,
                "intermediate_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "vocab_size": 8,
            }
        )
    )
    # A real one-element safetensors file, without importing a compute stack.
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    ).encode()
    header += b" " * (-len(header) % 8)
    (source / "model.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + struct.pack("<f", 1)
    )
    corpus = calibration / "input.txt"
    corpus.write_text("Local calibration text.\n")
    state = tmp_path / "state"
    policy = WorkspacePolicy([models], [calibration], [outputs], state)
    return policy, source, corpus, outputs / "result"


def recipe(source, corpus, destination, **settings):
    config = {"src": str(source), "calib": str(corpus), "dst": str(destination)}
    config.update(settings)
    return {"schema_version": 1, "config": config}


def reopen(policy, **overrides):
    values = dict(
        model_roots=list(policy.model_roots),
        calibration_roots=list(policy.calibration_roots),
        output_roots=list(policy.output_roots),
        state_root=policy.state_root,
    )
    values.update(overrides)
    return WorkspacePolicy(**values)


def test_import_and_path_policy_do_not_import_compute_stack(tmp_path):
    program = """
import sys
from pathlib import Path
from hi2.workspace import WorkspacePolicy, WorkspacePolicyError
root = Path(sys.argv[1])
policy = WorkspacePolicy([], [], [], root / 'state')
try:
    policy.validate_recipe({'schema_version': 1, 'config': {'trust_remote_code': True}})
except WorkspacePolicyError:
    pass
else:
    raise AssertionError('remote code was accepted')
assert not any(name in sys.modules for name in ('torch', 'transformers', 'hi2.pipeline', 'hi2.service'))
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


def test_ids_persist_and_remain_kind_scoped(workspace):
    policy, source, corpus, destination = workspace
    artifact = destination.parent / "existing"
    artifact.mkdir()
    (artifact / "hi2_report.json").write_text("{}")
    entries = {
        "model": source,
        "calibration": corpus,
        "output": destination,
        "artifact": artifact,
    }
    ids = {kind: policy.register(kind, path) for kind, path in entries.items()}
    fresh = reopen(policy)
    for kind, path in entries.items():
        assert getattr(fresh, f"resolve_{kind}")(ids[kind]) == path.resolve()
        assert fresh.register(kind, path) == ids[kind]
    assert len(set(ids.values())) == 4
    assert not destination.exists()
    assert stat.S_IMODE(policy.state_root.stat().st_mode) == 0o700
    assert stat.S_IMODE((policy.state_root / "workspace.json").stat().st_mode) == 0o600
    with pytest.raises(WorkspacePolicyError) as error:
        fresh.resolve_output(ids["model"])
    assert error.value.code == "id_kind"
    with pytest.raises(WorkspacePolicyError) as error:
        fresh.resolve_model("model:not-registered")
    assert error.value.code == "unknown_id"


def test_registration_instances_preserve_each_others_entries(workspace):
    first, source, corpus, _ = workspace
    second = reopen(first)
    model_id = first.register("model", source)
    calibration_id = second.register("calibration", corpus)
    assert first.resolve_calibration(calibration_id) == corpus
    assert second.resolve_model(model_id) == source
    assert (
        len(
            json.loads((first.state_root / "workspace.json").read_text())[
                "registrations"
            ]
        )
        == 2
    )


def test_registry_lock_and_failed_atomic_write_preserve_previous_ids(
    workspace, monkeypatch
):
    policy, source, corpus, _ = workspace
    model_id = policy.register("model", source)
    registry = policy.state_root / "workspace.json"
    before = registry.read_bytes()
    with exclusive_lock(policy.state_root / ".workspace.lock"):
        with pytest.raises(WorkspacePolicyError) as error:
            policy.register("calibration", corpus)
        assert error.value.code == "state_busy"
    assert registry.read_bytes() == before
    real_replace = os.replace

    def interrupted_replace(src, dst):
        if Path(dst) == registry:
            raise OSError("simulated registry commit interruption")
        return real_replace(src, dst)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", interrupted_replace)
        with pytest.raises(WorkspacePolicyError) as error:
            policy.register("calibration", corpus)
        assert error.value.code == "invalid_state"
    assert registry.read_bytes() == before
    assert reopen(policy).resolve_model(model_id) == source
    assert {p.name for p in policy.state_root.iterdir()} == {
        "workspace.json",
        ".workspace.lock",
    }


@pytest.mark.parametrize(
    "value", ["", ".", "../source", "models/source", "~/source", "bad\x00path", 1, None]
)
def test_relative_and_malformed_paths_are_rejected(workspace, value):
    policy, _, _, _ = workspace
    for method in (
        policy.resolve_model,
        policy.resolve_calibration,
        policy.resolve_output,
        policy.resolve_artifact,
    ):
        with pytest.raises(WorkspacePolicyError) as error:
            method(value)
        assert error.value.code == "invalid_path"


def test_absolute_parent_traversal_is_rejected_even_when_final_path_is_inside(
    workspace,
):
    policy, source, corpus, destination = workspace
    values = [
        (policy.resolve_model, source.parent / ".." / source.parent.name / source.name),
        (
            policy.resolve_calibration,
            corpus.parent / ".." / corpus.parent.name / corpus.name,
        ),
        (
            policy.resolve_output,
            destination.parent / ".." / destination.parent.name / destination.name,
        ),
    ]
    for resolve, value in values:
        with pytest.raises(WorkspacePolicyError) as error:
            resolve(str(value))
        assert error.value.code == "invalid_path"


def test_unconfigured_roots_and_lookalike_siblings_do_not_grant_access(
    workspace, tmp_path
):
    policy, source, corpus, destination = workspace
    empty = WorkspacePolicy([], [], [], policy.state_root)
    for method, path in (
        (empty.resolve_model, source),
        (empty.resolve_calibration, corpus),
        (empty.resolve_output, destination),
        (empty.resolve_artifact, source),
    ):
        with pytest.raises(WorkspacePolicyError) as error:
            method(str(path))
        assert error.value.code == "unconfigured_root"
    lookalike = tmp_path / "models-untrusted"
    lookalike.mkdir()
    with pytest.raises(WorkspacePolicyError) as error:
        policy.resolve_model(str(lookalike))
    assert error.value.code == "path_escape"
    with pytest.raises(WorkspacePolicyError) as error:
        policy.resolve_output(str(destination.parent))
    assert error.value.code == "input_output_overlap"


def test_safe_symlink_aliases_are_canonical_and_ids_do_not_follow_retargeted_aliases(
    workspace,
):
    policy, source, corpus, destination = workspace
    model_alias = source.parent / "alias"
    model_alias.symlink_to(source, target_is_directory=True)
    text_alias = corpus.parent / "alias.txt"
    text_alias.symlink_to(corpus)
    destination.mkdir()
    output_alias = destination.parent / "alias"
    output_alias.symlink_to(destination, target_is_directory=True)
    model_id = policy.register("model", model_alias)
    assert policy.resolve_model(str(model_alias)) == source
    assert policy.resolve_calibration(str(text_alias)) == corpus
    assert policy.resolve_output(str(output_alias)) == destination
    assert policy.register("model", source) == model_id
    model_alias.unlink()
    model_alias.symlink_to(destination, target_is_directory=True)
    assert policy.resolve_model(model_id) == source
    with pytest.raises(WorkspacePolicyError) as error:
        policy.resolve_model(str(model_alias))
    assert error.value.code == "path_escape"


@pytest.mark.parametrize("kind", ["model", "calibration", "output", "artifact"])
def test_symlink_escapes_are_rejected(workspace, tmp_path, kind):
    policy, source, corpus, destination = workspace
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_text = outside / "outside.txt"
    outside_text.write_text("outside")
    parents = {
        "model": source.parent,
        "calibration": corpus.parent,
        "output": destination.parent,
        "artifact": destination.parent,
    }
    alias = parents[kind] / "escaping"
    alias.symlink_to(
        outside_text if kind == "calibration" else outside,
        target_is_directory=kind != "calibration",
    )
    with pytest.raises(WorkspacePolicyError) as error:
        getattr(policy, f"resolve_{kind}")(str(alias))
    assert error.value.code == "path_escape"


def test_new_output_cannot_escape_through_symlink_parent(workspace, tmp_path):
    policy, _, _, destination = workspace
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = destination.parent / "escaping"
    alias.symlink_to(outside, target_is_directory=True)
    with pytest.raises(WorkspacePolicyError) as error:
        policy.resolve_output(str(alias / "not-created"))
    assert error.value.code == "path_escape"
    assert not (outside / "not-created").exists()


def test_model_assets_may_link_only_into_explicitly_trusted_model_roots(
    workspace, tmp_path
):
    policy, source, corpus, destination = workspace
    blobs = tmp_path / "trusted-blobs"
    blobs.mkdir()
    weights = source / "model.safetensors"
    blob = blobs / "weights.safetensors"
    blob.write_bytes(weights.read_bytes())
    weights.unlink()
    weights.symlink_to(blob)
    with pytest.raises(WorkspacePolicyError) as error:
        policy.resolve_model(str(source))
    assert error.value.code == "path_escape"
    trusted = reopen(policy, model_roots=[source.parent, blobs])
    assert trusted.resolve_model(str(source)) == source
    normalized = trusted.validate_recipe(recipe(source, corpus, destination))
    assert normalized["config"]["src"] == str(source)
    # A trusted directory link must not hide an escaping descendant.
    nested = source / "assets"
    nested.symlink_to(blobs, target_is_directory=True)
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    (blobs / "tokenizer.json").symlink_to(outside)
    with pytest.raises(WorkspacePolicyError) as error:
        trusted.resolve_model(str(source))
    assert error.value.code == "path_escape"


def test_symlink_loop_and_non_regular_assets_are_rejected(workspace):
    policy, source, _, _ = workspace
    loop = source / "loop"
    loop.symlink_to(loop)
    with pytest.raises(WorkspacePolicyError) as error:
        policy.resolve_model(str(source))
    assert error.value.code == "invalid_path"
    loop.unlink()
    os.mkfifo(source / "pipe")
    with pytest.raises(WorkspacePolicyError) as error:
        policy.resolve_model(str(source))
    assert error.value.code == "invalid_path"


def test_registered_paths_are_rechecked_against_new_roots_and_filesystem(
    workspace, tmp_path
):
    policy, source, _, _ = workspace
    identifier = policy.register("model", source)
    outside = tmp_path / "other-models"
    outside.mkdir()
    restricted = reopen(policy, model_roots=[outside])
    with pytest.raises(WorkspacePolicyError) as error:
        restricted.resolve_model(identifier)
    assert error.value.code == "path_escape"
    original = source.with_name("original")
    source.rename(original)
    source.symlink_to(outside, target_is_directory=True)
    with pytest.raises(WorkspacePolicyError) as error:
        policy.resolve_model(identifier)
    assert error.value.code == "path_escape"


@pytest.mark.parametrize(
    "target", ["source", "source-parent", "calibration-parent", "source-child"]
)
def test_outputs_cannot_overwrite_inputs_even_with_overlapping_root_configuration(
    workspace, target
):
    policy, source, corpus, destination = workspace
    broad = reopen(
        policy, output_roots=[source.parent, corpus.parent, destination.parent]
    )
    targets = {
        "source": source,
        "source-parent": source.parent,
        "calibration-parent": corpus.parent,
        "source-child": source / "output",
    }
    before_model, before_text = (
        (source / "model.safetensors").read_bytes(),
        corpus.read_bytes(),
    )
    with pytest.raises(WorkspacePolicyError) as error:
        broad.validate_recipe(recipe(source, corpus, targets[target]))
    assert error.value.code == "input_output_overlap"
    assert (source / "model.safetensors").read_bytes() == before_model
    assert corpus.read_bytes() == before_text
    assert not (policy.state_root / "cache").exists()


@pytest.mark.parametrize("link", ["symbolic", "hard"])
def test_existing_output_tree_cannot_alias_input_files(workspace, link):
    policy, source, corpus, destination = workspace
    destination.mkdir()
    aliased = destination / "input.txt"
    if link == "symbolic":
        aliased.symlink_to(corpus)
    else:
        os.link(corpus, aliased)
    before = corpus.read_bytes()
    with pytest.raises(WorkspacePolicyError) as error:
        policy.validate_recipe(recipe(source, corpus, destination))
    assert error.value.code in {"path_escape", "input_output_overlap"}
    assert corpus.read_bytes() == before
    assert not (policy.state_root / "cache").exists()


def test_all_multi_ratio_outputs_are_root_checked_before_allocating_state(workspace):
    policy, source, corpus, destination = workspace
    expanded = destination.with_name(destination.name + "-r4")
    expanded.symlink_to(source, target_is_directory=True)
    with pytest.raises(WorkspacePolicyError) as error:
        policy.validate_recipe(
            recipe(source, corpus, destination, max_error_ratios=[2, 4])
        )
    assert error.value.code == "path_escape"
    assert not destination.exists()
    assert not (policy.state_root / "work").exists()


def test_valid_typed_recipe_is_normalized_without_loading_models_or_mutating_inputs(
    workspace,
):
    from hi2.calibrate import Config
    from hi2.qad import QadConfig
    from hi2.service import config_from_dict

    policy, source, corpus, destination = workspace
    second = corpus.with_name("more,text.jsonl")
    second.write_text('{"text": "More calibration text"}\n')
    model_id = policy.register("model", source)
    calibration_id = policy.register("calibration", corpus)
    output_id = policy.register("output", destination)
    document = recipe(
        model_id,
        calibration_id,
        output_id,
        devices=["cpu"],
        seq_len=32,
        calib_samples=2,
        cfg={"candidates": [[2, 32], [4, 64]], "smooth_search": [0.25, 0.5]},
        qad={"steps": 2, "samples": 2, "val": 1, "batch": 1},
    )
    document["config"]["calib"] = [calibration_id, str(second)]
    before = copy.deepcopy(document)
    normalized = policy.validate_recipe(document)
    assert document == before
    config = normalized["config"]
    assert normalized["schema_version"] == 1
    assert config["src"] == str(source)
    assert config["calib"] == [str(corpus), str(second)]
    assert config["dst"] == str(destination)
    assert config["cache_dir"] == str(policy.state_root / "cache")
    workdir = Path(config["workdir"])
    assert workdir.parent == policy.state_root / "work"
    assert workdir.is_dir() and list(workdir.iterdir()) == []
    assert stat.S_IMODE(workdir.stat().st_mode) == 0o700
    assert not destination.exists()
    restored = config_from_dict(normalized)
    assert isinstance(restored.cfg, Config) and isinstance(restored.qad, QadConfig)
    assert restored.cfg.candidates == [(2, 32), (4, 64)]
    assert restored.qad.steps == 2
    assert restored.calib == [corpus, second]
    assert reopen(policy).validate_recipe(normalized) == normalized
    changed = copy.deepcopy(document)
    changed["config"]["seq_len"] = 16
    assert policy.validate_recipe(changed)["config"]["workdir"] != config["workdir"]


def test_evaluation_cadence_is_typed_recipe_data_and_replays(workspace):
    policy, source, corpus, destination = workspace
    document = recipe(source, corpus, destination, qad={"eval_every": 2})
    before = copy.deepcopy(document)
    normalized = policy.validate_recipe(document)
    assert document == before
    assert normalized["config"]["qad"]["eval_every"] == 2
    assert reopen(policy).validate_recipe(normalized) == normalized


@pytest.mark.parametrize(
    ("location", "key"),
    [
        ("config", "eval_every"),
        ("cfg", "eval_every"),
        ("config", "eval_unknown"),
        ("cfg", "eval_unknown"),
        ("qad", "eval_unknown"),
    ],
)
def test_evaluation_names_do_not_grant_fields_outside_the_schema(
    workspace, location, key
):
    policy, source, corpus, destination = workspace
    document = recipe(source, corpus, destination)
    target = document["config"]
    if location != "config":
        target = target.setdefault(location, {})
    target[key] = 2
    with pytest.raises(WorkspacePolicyError) as error:
        policy.validate_recipe(document)
    assert error.value.code == "invalid_recipe"
    assert not (policy.state_root / "cache").exists()


def test_data_free_recipe_remains_usable(workspace):
    from hi2.service import config_from_dict

    policy, source, _, destination = workspace
    normalized = policy.validate_recipe(
        {"schema_version": 1, "config": {"src": str(source), "dst": str(destination)}}
    )
    restored = config_from_dict(normalized)
    assert restored.src == source and restored.dst == destination
    assert restored.calib is None
    assert restored.cache_dir == policy.state_root / "cache"
    assert restored.devices == ["cpu"]


@pytest.mark.parametrize(
    "key",
    [
        "trust_remote_code",
        "remote-code",
        "trustRemoteCode",
        "token",
        "HF_TOKEN",
        "api_key",
        "apiKey",
        "password",
        "credentials",
        "authorization",
        "shell",
        "command",
        "function",
        "script",
        "env",
        "eval",
        "eval_command",
        "eval_token",
        "eval_trust_remote_code",
    ],
)
@pytest.mark.parametrize("location", ["document", "config", "cfg", "qad"])
def test_secret_and_executable_fields_are_rejected_without_echo(
    workspace, key, location, capsys
):
    policy, source, corpus, destination = workspace
    document = recipe(source, corpus, destination)
    target = document if location == "document" else document["config"]
    if location in ("cfg", "qad"):
        target = document["config"].setdefault(location, {})
    sentinel = "operator-secret-that-must-not-leak"
    target[key] = sentinel
    with pytest.raises(WorkspacePolicyError) as error:
        policy.validate_recipe(document)
    assert error.value.code == "forbidden_field"
    assert sentinel not in str(error.value)
    assert capsys.readouterr() == ("", "")
    assert not (policy.state_root / "workspace.json").exists()


@pytest.mark.parametrize("flag", [False, None, 0, "false", True])
def test_even_disabled_caller_remote_code_flags_are_not_tool_fields(workspace, flag):
    policy, source, corpus, destination = workspace
    with pytest.raises(WorkspacePolicyError) as error:
        policy.validate_recipe(
            recipe(source, corpus, destination, trust_remote_code=flag)
        )
    assert error.value.code == "forbidden_field"


@pytest.mark.parametrize("location", ["document", "config", "cfg", "qad"])
def test_unknown_fields_are_rejected_without_echoing_field_names_or_values(
    workspace, location
):
    policy, source, corpus, destination = workspace
    document = recipe(source, corpus, destination)
    target = document if location == "document" else document["config"]
    if location in ("cfg", "qad"):
        target = document["config"].setdefault(location, {})
    target["sensitive-unrecognized-name"] = "sensitive-unrecognized-value"
    with pytest.raises(WorkspacePolicyError) as error:
        policy.validate_recipe(document)
    assert error.value.code == "invalid_recipe"
    assert "sensitive-unrecognized" not in str(error.value)
    assert not (policy.state_root / "cache").exists()


@pytest.mark.parametrize(
    "changes",
    [
        {"seq_len": "32"},
        {"seq_len": True},
        {"devices": "cpu"},
        {"cfg": {"smooth": "0.5"}},
        {"cfg": {"sequential": "false"}},
        {"cfg": {"candidates": [[2, "32"]]}},
        {"cfg": []},
        {"qad": {"steps": 1.5}},
        {"qad": {"seed": False}},
        {"qad": []},
        {"qad": {"eval_every": "2"}},
        {"qad": {"eval_every": True}},
        {"max_error_ratio": float("nan")},
        {"max_error_ratios": [float("inf")]},
        {"calib": []},
        {"calib": {"path": "not a path"}},
    ],
)
def test_recipe_values_must_have_declared_json_types(workspace, changes):
    policy, source, corpus, destination = workspace
    with pytest.raises(WorkspacePolicyError):
        policy.validate_recipe(recipe(source, corpus, destination, **changes))
    assert not (policy.state_root / "cache").exists()


def test_non_json_objects_and_recursive_documents_are_rejected_without_execution(
    workspace,
):
    policy, source, corpus, destination = workspace
    ran = []

    def function():
        ran.append(True)

    document = recipe(source, corpus, destination, include=function)
    with pytest.raises(WorkspacePolicyError) as error:
        policy.validate_recipe(document)
    assert error.value.code == "invalid_recipe"
    assert not ran
    document["config"]["include"] = document
    with pytest.raises(WorkspacePolicyError) as error:
        policy.validate_recipe(document)
    assert error.value.code == "invalid_recipe"


def test_operator_environment_secrets_are_not_exported_or_logged(
    workspace, monkeypatch, capsys
):
    policy, source, corpus, destination = workspace
    secret = "operator-only-hf-secret"
    monkeypatch.setenv("HF_TOKEN", secret)
    result = policy.validate_recipe(recipe(source, corpus, destination))
    assert secret not in json.dumps(result)
    assert secret not in (policy.state_root / "workspace.json").read_text()
    assert capsys.readouterr() == ("", "")


class UntrustedPickle:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return Path.write_text, (self.marker, "pickle was executed")


@pytest.mark.parametrize("field", ["cache_dir", "workdir", "calib_ids"])
def test_caller_supplied_pickle_cache_paths_are_never_loaded(
    workspace, tmp_path, field
):
    policy, source, corpus, destination = workspace
    marker = tmp_path / "executed"
    payload = destination.parent / "upload.pt"
    payload.write_bytes(pickle.dumps(UntrustedPickle(marker)))
    with pytest.raises(WorkspacePolicyError) as error:
        policy.validate_recipe(
            recipe(source, corpus, destination, **{field: str(payload)})
        )
    assert error.value.code == "untrusted_cache"
    assert not marker.exists()
    assert not (policy.state_root / "cache").exists()


@pytest.mark.parametrize("where", ["output-root", "private-state"])
@pytest.mark.parametrize("field", ["cache_dir", "workdir"])
def test_root_containment_does_not_authorize_arbitrary_cache_directories(
    workspace, where, field
):
    policy, source, corpus, destination = workspace
    parent = destination.parent if where == "output-root" else policy.state_root
    supplied = parent / "caller-cache"
    supplied.mkdir(mode=0o700)
    (supplied / "tables.pt").write_bytes(b"untrusted")
    with pytest.raises(WorkspacePolicyError) as error:
        policy.validate_recipe(
            recipe(source, corpus, destination, **{field: str(supplied)})
        )
    assert error.value.code == "untrusted_cache"
    assert not (policy.state_root / "cache").exists()


def test_preexisting_unowned_private_cache_is_not_adopted(workspace, tmp_path):
    policy, source, corpus, destination = workspace
    cache = policy.state_root / "cache"
    cache.mkdir(mode=0o700)
    marker = tmp_path / "executed"
    payload = cache / "table.pt"
    contents = pickle.dumps(UntrustedPickle(marker))
    payload.write_bytes(contents)
    with pytest.raises(WorkspacePolicyError) as error:
        policy.validate_recipe(recipe(source, corpus, destination))
    assert error.value.code == "untrusted_cache"
    assert payload.read_bytes() == contents
    assert not marker.exists()
    assert not (policy.state_root / "work").exists()


def test_pickle_files_cannot_be_registered_as_artifacts_or_calibration(workspace):
    policy, _, corpus, destination = workspace
    for kind, parent in (
        ("artifact", destination.parent),
        ("calibration", corpus.parent),
    ):
        payload = parent / "payload.pt"
        payload.write_bytes(b"not a public resource")
        with pytest.raises(WorkspacePolicyError) as error:
            policy.register(kind, payload)
        assert error.value.code == "untrusted_cache"
    with pytest.raises(WorkspacePolicyError) as error:
        policy.register("cache", policy.state_root)
    assert error.value.code == "invalid_kind"


def test_private_state_is_not_exposed_even_under_an_output_root(workspace):
    policy, source, corpus, destination = workspace
    private = WorkspacePolicy(
        list(policy.model_roots),
        list(policy.calibration_roots),
        list(policy.output_roots),
        destination.parent / "private",
    )
    private.validate_recipe(recipe(source, corpus, destination))
    for path in (
        private.state_root,
        private.state_root / "workspace.json",
        private.state_root / "cache",
    ):
        for resolve in (private.resolve_artifact, private.resolve_output):
            with pytest.raises(WorkspacePolicyError) as error:
                resolve(str(path))
            assert error.value.code == "untrusted_cache"


def test_state_requires_private_permissions_and_never_silently_repairs_them(
    workspace, tmp_path
):
    policy, _, _, _ = workspace
    public = tmp_path / "public-state"
    public.mkdir(mode=0o755)
    public.chmod(0o755)
    with pytest.raises(WorkspacePolicyError) as error:
        reopen(policy, state_root=public)
    assert error.value.code == "invalid_state"
    assert stat.S_IMODE(public.stat().st_mode) == 0o755
    alias = tmp_path / "state-alias"
    alias.symlink_to(policy.state_root, target_is_directory=True)
    with pytest.raises(WorkspacePolicyError) as error:
        reopen(policy, state_root=alias)
    assert error.value.code == "invalid_state"
    policy.state_root.chmod(0o755)
    with pytest.raises(WorkspacePolicyError) as error:
        policy.resolve_output(str(tmp_path / "outputs" / "result"))
    assert error.value.code == "invalid_state"


@pytest.mark.parametrize("name", ["workspace.json", ".workspace.lock"])
def test_state_control_files_must_not_be_symlinks(workspace, tmp_path, name):
    policy, source, corpus, _ = workspace
    identifier = policy.register("model", source)
    external = tmp_path / "external.json"
    external.write_text('{"private": "preserve"}')
    entry = policy.state_root / name
    entry.unlink()
    entry.symlink_to(external)
    with pytest.raises(WorkspacePolicyError) as error:
        policy.resolve_model(identifier)
    assert error.value.code == "invalid_state"
    with pytest.raises(WorkspacePolicyError):
        policy.register("calibration", corpus)
    assert external.read_text() == '{"private": "preserve"}'


@pytest.mark.parametrize("link", ["symbolic", "hard", "writable"])
def test_managed_cache_rejects_linked_or_shared_writable_entries(
    workspace, tmp_path, link
):
    policy, source, corpus, destination = workspace
    normalized = policy.validate_recipe(recipe(source, corpus, destination))
    outside = tmp_path / "upload.pt"
    outside.write_bytes(b"external payload")
    entry = Path(normalized["config"]["cache_dir"]) / "entry.pt"
    if link == "symbolic":
        entry.symlink_to(outside)
    elif link == "hard":
        os.link(outside, entry)
    else:
        entry.write_bytes(b"shared writable payload")
        entry.chmod(0o666)
    with pytest.raises(WorkspacePolicyError) as error:
        policy.validate_recipe(normalized)
    assert error.value.code == "invalid_state"


def test_corrupt_registration_data_fails_closed(workspace):
    policy, source, _, _ = workspace
    identifier = policy.register("model", source)
    registry = policy.state_root / "workspace.json"
    for content in (
        "{",
        "[]",
        '{"schema_version": 99}',
        '{"schema_version": 1, "registrations": {}, "managed": "not a list"}',
    ):
        registry.write_text(content)
        with pytest.raises(WorkspacePolicyError) as error:
            policy.resolve_model(identifier)
        assert error.value.code == "invalid_state"


@pytest.mark.parametrize(
    "document",
    [
        [],
        {},
        {"schema_version": True, "config": {}},
        {"schema_version": "1", "config": {}},
        {"schema_version": 2, "config": {}},
        {"schema_version": 1, "config": []},
        {"schema_version": 1, "config": {}},
    ],
)
def test_recipe_requires_exact_versioned_object_shape(workspace, document):
    policy, _, _, _ = workspace
    with pytest.raises(WorkspacePolicyError) as error:
        policy.validate_recipe(document)
    assert error.value.code == "invalid_recipe"
    assert not (policy.state_root / "cache").exists()


def test_resource_kinds_require_existing_inputs_and_directory_outputs(workspace):
    policy, source, corpus, destination = workspace
    destination.write_text("do not overwrite")
    cases = [
        (policy.resolve_model, source / "config.json"),
        (policy.resolve_model, source.parent / "missing"),
        (policy.resolve_calibration, corpus.parent),
        (policy.resolve_calibration, corpus.with_name("missing.txt")),
        (policy.resolve_output, destination),
        (policy.resolve_artifact, destination.with_name("missing-artifact")),
    ]
    for resolve, path in cases:
        with pytest.raises(WorkspacePolicyError) as error:
            resolve(str(path))
        assert error.value.code == "invalid_path"
    assert destination.read_text() == "do not overwrite"


def test_managed_work_parent_cannot_redirect_to_another_private_directory(workspace):
    policy, source, corpus, destination = workspace
    alternate = policy.state_root / "alternate"
    alternate.mkdir(mode=0o700)
    (policy.state_root / "work").symlink_to(alternate, target_is_directory=True)
    with pytest.raises(WorkspacePolicyError) as error:
        policy.validate_recipe(recipe(source, corpus, destination))
    assert error.value.code == "invalid_state"
    assert list(alternate.iterdir()) == []
    assert not (policy.state_root / "cache").exists()


def test_linked_model_assets_remain_read_only_capabilities(workspace):
    policy, source, _, destination = workspace
    (source / "tokenizer.json").write_text("{}")
    destination.mkdir()
    (destination / "tokenizer.json").symlink_to(source / "tokenizer.json")
    assert policy.resolve_output(str(destination)) == destination
    identifier = policy.register("artifact", destination)
    assert reopen(policy).resolve_artifact(identifier) == destination
    # An allowed ancillary link cannot promote model files into writable output
    # files or redirect the named evidence exposed by MCP.
    for name in ("config.json", "model.safetensors", "hi2_report.json"):
        link = destination / name
        link.symlink_to(source / "config.json")
        with pytest.raises(WorkspacePolicyError):
            policy.resolve_output(str(destination))
        with pytest.raises(WorkspacePolicyError):
            policy.resolve_artifact(identifier)
        link.unlink()


def test_linked_assets_do_not_authorize_unconfigured_roots(workspace, tmp_path):
    policy, _, _, destination = workspace
    destination.mkdir()
    outside = tmp_path / "external-tokenizer.json"
    outside.write_text("{}")
    (destination / "tokenizer.json").symlink_to(outside)
    with pytest.raises(WorkspacePolicyError) as error:
        policy.resolve_artifact(str(destination))
    assert error.value.code == "path_escape"


@pytest.mark.parametrize("name", ["workspace.json", ".workspace.lock"])
@pytest.mark.parametrize("unsafe", ["hardlink", "writable"])
def test_state_control_files_reject_aliases_and_shared_write_without_repair(
    workspace, tmp_path, name, unsafe
):
    # The workspace contract requires POSIX ownership, modes and single-link files.
    policy, source, corpus, _ = workspace
    identifier = policy.register("model", source)
    registry = policy.state_root / "workspace.json"
    registry_before = registry.read_bytes()
    entry = policy.state_root / name
    entry_before = entry.read_bytes()
    alias = tmp_path / "aliased-control-file"
    if unsafe == "hardlink":
        os.link(entry, alias)
    else:
        entry.chmod(0o660)

    with pytest.raises(WorkspacePolicyError) as error:
        reopen(policy).resolve_model(identifier)
    assert error.value.code == "invalid_state"
    with pytest.raises(WorkspacePolicyError) as error:
        policy.register("calibration", corpus)
    assert error.value.code == "invalid_state"
    assert registry.read_bytes() == registry_before
    assert entry.read_bytes() == entry_before
    if unsafe == "hardlink":
        assert alias.read_bytes() == entry_before
        assert entry.stat().st_nlink == 2
    else:
        assert stat.S_IMODE(entry.stat().st_mode) == 0o660


def test_changed_recipe_cannot_replay_another_configurations_workdir(workspace):
    policy, source, corpus, destination = workspace
    original = policy.validate_recipe(recipe(source, corpus, destination, seq_len=32))
    work = Path(original["config"]["workdir"])
    marker = work / "owned-cache.pt"
    marker.write_bytes(b"preserve the original recipe's cached state")
    registry = policy.state_root / "workspace.json"
    before = registry.read_bytes()
    changed = copy.deepcopy(original)
    changed["config"]["seq_len"] = 16

    with pytest.raises(WorkspacePolicyError) as error:
        policy.validate_recipe(changed)
    assert error.value.code == "untrusted_cache"
    assert registry.read_bytes() == before
    assert list(work.parent.iterdir()) == [work]
    assert not destination.exists()

    del changed["config"]["workdir"]
    normalized = policy.validate_recipe(changed)
    assert normalized["config"]["workdir"] != str(work)
    assert normalized["config"]["cache_dir"] == original["config"]["cache_dir"]
    assert Path(normalized["config"]["workdir"]).is_dir()
    assert marker.read_bytes() == b"preserve the original recipe's cached state"
    assert reopen(policy).validate_recipe(original) == original


def test_unsafe_nested_cache_prevents_allocating_a_new_workdir(workspace):
    policy, source, corpus, destination = workspace
    original = policy.validate_recipe(recipe(source, corpus, destination, seq_len=32))
    work = Path(original["config"]["workdir"])
    shared = Path(original["config"]["cache_dir"]) / "tables" / "shared"
    shared.mkdir(parents=True)
    shared.chmod(0o777)
    registry = policy.state_root / "workspace.json"
    before = registry.read_bytes()

    with pytest.raises(WorkspacePolicyError) as error:
        policy.validate_recipe(recipe(source, corpus, destination, seq_len=16))
    assert error.value.code == "invalid_state"
    assert registry.read_bytes() == before
    assert list(work.parent.iterdir()) == [work]
    assert stat.S_IMODE(shared.stat().st_mode) == 0o777
    assert not destination.exists()
