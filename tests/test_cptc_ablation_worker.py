# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""CPU lifecycle tests for the CPTC ablation worker.

These exercise the real tiny model/pipeline/TensorTable path: a real QAD parent
with final_kl=0, a real unquantized DEFAULT teacher head, and a real final-only
CPTC export through the worker. They do not run the 4B study or claim timings.
"""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from hi2.calibrate import Config  # noqa: E402
from hi2.pipeline import RunConfig, run  # noqa: E402
from hi2.qad import make_config  # noqa: E402
from hi2.runtime import atomic_json, file_digest  # noqa: E402

WORKER_PATH = (
    Path(__file__).resolve().parents[1] / 'experiments' / 'cptc_ablation_worker.py'
)
_SPEC = importlib.util.spec_from_file_location('cptc_ablation_worker', WORKER_PATH)
worker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(worker)


def _item(path):
    return {'path': str(Path(path).resolve()), 'sha256': file_digest(Path(path))}


@pytest.fixture(scope='module')
def tiny_study(tmp_path_factory, request):
    rng_state = torch.random.get_rng_state()
    request.addfinalizer(lambda: torch.random.set_rng_state(rng_state))
    from transformers import (
        Qwen2Tokenizer,
        Qwen2Config,
        Qwen2ForCausalLM,
    )

    root = tmp_path_factory.mktemp('cptc-ablation-worker')
    model_dir = root / 'model'
    cfg = Qwen2Config(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg).float().eval()
    model.save_pretrained(model_dir, safe_serialization=True)
    from tokenizers.pre_tokenizers import ByteLevel

    vocab = {char: i for i, char in enumerate(sorted(ByteLevel.alphabet()))}
    Qwen2Tokenizer(vocab=vocab, merges=[]).save_pretrained(model_dir)
    calibration = root / 'calib.txt'
    calibration.write_text(
        ' '.join(f'w{(i % 255) + 1}' for i in range(2000)) + '\n'
    )
    cache = root / 'cache'
    base = dict(
        src=model_dir,
        calib=calibration,
        calib_samples=4,
        seq_len=16,
        # Tiny 64-wide rows have >4 bpw of minimum payload/metadata overhead;
        # use a feasible mixed-width allocation, not the full 4B model's budget.
        target_bpw=5.5,
        devices=['cpu'],
        cache_dir=cache,
        cfg=Config(
            candidates=[(2, 16), (4, 16)],
            effort='fast',
            cptc=True,
            cptc_depth=8,
            cptc_table_samples=1024,
        ),
    )
    defaults_rc = RunConfig(
        dst=root / 'out-defaults',
        workdir=root / 'work-defaults',
        qad=make_config(
            'default',
            {
                'steps': 2,
                'samples': 4,
                'val': 2,
                'batch': 2,
                'final_kl': 2,
                'final_lr_p': 0.02,
                'cptc_scale_units': 'log',
                'cptc_projection_sweeps': 1,
                'seed': 0,
            },
        ),
        **base,
    )
    run(defaults_rc, log=lambda *_: None)
    assert (defaults_rc.workdir / 'qad' / 'head_inputs.pt').is_file()
    defaults = json.loads((defaults_rc.workdir / 'run.json').read_text())

    plan = {
        'status': 'factor_design_frozen_not_executed',
        'training_constants': {
            'layers': 2,
            'qad_steps_per_layer': 2,
            'final_kl_steps': 2,
            'final_table_lr_when_enabled': 0.02,
        },
        'quality_factors': {
            'scale_units': ['log', 'row_u8'],
            'projection_sweeps': [1, 4],
            'final_scale_lr': [0.02, 0.002],
            'final_shared_table_learning': [False, True],
        },
        'quality_trials': [],
        'training_efficiency': {'cache_budgets_mb': [0, 4096]},
        'immutable_controls': {
            'quantizer_source_sha': worker.OLD_QUANTIZER_SHA,
            'cptc_artifact_digest': '0' * 64,
        },
    }
    qad = {
        'cptc_scale_units': 'row_u8',
        'cptc_projection_sweeps': 1,
        'steps': 2,
    }
    for lr, fixed in ((0.02, False), (0.002, False)):
        plan['quality_trials'].append(
            {
                'id': f'row_u8-p1-lr{lr}-fixed',
                'qad_parent': 'row_u8-p1',
                'qad_overrides': copy.deepcopy(qad),
                'final_overrides': {
                    'final_kl': 2,
                    'final_lr_p': lr,
                    'final_cptc_train_tables': fixed,
                    'final_cptc_table_lr': 0.02,
                },
            }
        )
    plan_path = root / 'plan.json'
    atomic_json(plan_path, plan)
    holdout = root / 'holdout.json'
    atomic_json(holdout, {'status': 'data_frozen_no_quality_results'})
    source_manifest = root / 'source-manifest.json'
    atomic_json(source_manifest, {'files': []})
    spec = {
        'schema_version': 1,
        'status': 'cpu_verified',
        'snapshot': str(Path(__file__).resolve().parents[1]),
        'factor_plan': _item(plan_path),
        'confirmatory_holdout': _item(holdout),
        'source_manifest': _item(source_manifest),
        'defaults_run': _item(defaults_rc.workdir / 'run.json'),
        'run_root': str(root / 'runs'),
    }
    descriptor_path = root / 'shared-teacher.json'
    worker.prepare_teacher_core(spec, plan, defaults, descriptor_path)
    spec['shared_teacher'] = _item(descriptor_path)
    worker.run_qad_parent(spec, plan, defaults, 'row_u8-p1', log=lambda *_: None)
    parent_work = Path(spec['run_root']) / 'parents' / 'row_u8-p1' / 'work'
    assert not (parent_work / 'qad' / 'head_inputs.pt').exists()
    spec['qad_parent_runs'] = {
        'row_u8-p1': _item(parent_work / 'run.json')
    }
    manifest = worker.parent_manifest_record(spec, plan, defaults, 'row_u8-p1')
    manifest_path = root / 'parent-row_u8-p1.json'
    atomic_json(manifest_path, manifest)
    spec['parent_manifests'] = {'row_u8-p1': _item(manifest_path)}
    return {
        'root': root,
        'defaults': defaults,
        'plan': plan,
        'spec': spec,
        'descriptor': json.loads(descriptor_path.read_text()),
        'manifest': manifest,
        'calibration': calibration,
        'model_dir': model_dir,
        'parent_work': parent_work,
    }


def _trial(spec, plan, trial_id):
    return next(t for t in plan['quality_trials'] if t['id'] == trial_id)


def test_shared_teacher_binds_ids_and_refuses_tampering(tiny_study):
    study = tiny_study
    plan = study['plan']
    spec = study['spec']
    defaults = study['defaults']
    teacher = study['descriptor']
    trial = _trial(spec, plan, 'row_u8-p1-lr0.02-fixed')
    qad_child = copy.deepcopy(defaults['config']['qad'])
    qad_child.update(trial['qad_overrides'])
    qad_child.update(trial['final_overrides'])
    document = worker.expected_config(
        defaults,
        study['root'] / 'id-check' / 'model',
        study['root'] / 'id-check' / 'work',
        qad_child,
    )
    rc = worker.recipe(document)
    from hi2.service import resolve_config, run_identity

    rc = resolve_config(rc, check_ownership=False)
    identity = run_identity(rc)
    assert worker.require_common_identity(teacher, rc, identity).shape == (6, 16)

    tampered = copy.deepcopy(teacher)
    tampered['calibration_ids']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='calibration IDs'):
        worker.require_common_identity(tampered, rc, identity)

    # A changed calibration file cannot silently re-pair different windows.
    changed = study['root'] / 'calib-changed.txt'
    changed.write_text(' '.join(f'w{(i % 255) + 1}' for i in range(2001)) + '\n')
    changed_document = copy.deepcopy(document)
    changed_document['calib'] = str(changed)
    changed_rc = resolve_config(worker.recipe(changed_document), check_ownership=False)
    changed_identity = run_identity(changed_rc)
    with pytest.raises(ValueError, match='calibration'):
        worker.require_common_identity(teacher, changed_rc, changed_identity)


def test_parent_manifest_reuses_common_teacher_without_head(tiny_study):
    study = tiny_study
    record = study['manifest']
    assert record['parent'] == 'row_u8-p1'
    assert record['qad']['final_kl'] == 0
    assert 'head' not in record
    assert record['common_teacher']['head']['sha256'] == (
        study['descriptor']['head']['sha256']
    )
    assert record['common_teacher']['descriptor'] == study['spec']['shared_teacher']
    assert record['layout_signature'] == study['descriptor']['layout_signature']
    assert record['producer']['environment_receipt'] is not None
    assert record['producer']['environment']['tokenizers'] == (
        study['descriptor']['producer_environment']['tokenizers']
    )
    assert not (study['parent_work'] / 'qad' / 'head_inputs.pt').exists()
    repeat = worker.parent_manifest_record(
        study['spec'], study['plan'], study['defaults'], 'row_u8-p1'
    )
    assert repeat == record


def test_preserved_control_is_refused_before_any_work():
    trial = {
        'id': 'log-p1-lr02-fixed',
        'qad_parent': 'log-p1',
        'qad_overrides': {},
        'final_overrides': {},
        'preserved_control': True,
    }
    with pytest.raises(ValueError, match='DEFAULT control'):
        worker.run_final_trial(
            {}, {'quality_trials': [trial]}, {}, 'log-p1-lr02-fixed'
        )


def test_parent_mismatch_and_cache_miss_refuse_before_pipeline(tiny_study):
    study = tiny_study
    plan = study['plan']
    spec = copy.deepcopy(study['spec'])
    defaults = study['defaults']
    trial = _trial(spec, plan, 'row_u8-p1-lr0.02-fixed')

    bad_parent = copy.deepcopy(study['manifest'])
    bad_parent['parent'] = 'row_u8-p4'
    bad_path = study['root'] / 'bad-parent.json'
    atomic_json(bad_path, bad_parent)
    spec['parent_manifests']['row_u8-p1'] = _item(bad_path)
    with pytest.raises(ValueError, match='wrong QAD parent'):
        worker.run_final_trial(spec, plan, defaults, trial['id'])

    # A real cache entry is pinned; a missing/changed entry is refused before
    # the final-KL/export pipeline can run (or rebuild anything).
    teacher = copy.deepcopy(study['descriptor'])
    cache_root = Path(study['defaults']['config']['cache_dir'])
    teacher['cache_ref']['stats']['path'] = str(
        cache_root / 'stats' / 'missing-entry'
    )
    qad_child = copy.deepcopy(defaults['config']['qad'])
    qad_child.update(trial['qad_overrides'])
    qad_child.update(trial['final_overrides'])
    document = worker.expected_config(
        defaults,
        study['root'] / 'cache-check' / 'model',
        study['root'] / 'cache-check' / 'work',
        qad_child,
    )
    rc = worker.recipe(document)
    from hi2.service import resolve_config, run_identity

    rc = resolve_config(rc, check_ownership=False)
    identity = run_identity(rc)
    with pytest.raises(ValueError, match='cache entry changed'):
        worker.cache_preflight(teacher, rc, identity)


def test_real_final_only_export_then_refuses_successful_rerun(tiny_study):
    study = tiny_study
    result = worker.run_final_trial(
        study['spec'],
        study['plan'],
        study['defaults'],
        'row_u8-p1-lr0.02-fixed',
    )
    assert result['structure']['report_checked'] is True
    assert result['structure']['checked'] == result['quantized_tensors']
    assert result['quantized_tensors'] > 0
    assert result['timing']['final_kl_s'] == result['final_kl']['seconds']
    assert result['common_teacher']['head']['path'] == (
        study['descriptor']['head']['path']
    )
    assert 'head' not in study['manifest']
    dst = Path(study['spec']['run_root']) / 'trials' / 'row_u8-p1-lr0.02-fixed'
    qcfg = json.loads((dst / 'model' / 'config.json').read_text())[
        'quantization_config'
    ]
    assert qcfg['qad']['final_kl'] == 2
    assert 'final_cptc_cache_mb' not in qcfg['qad']
    assert qcfg['moe_layout'] == 'stacked'
    assert (dst / 'model' / 'hi2_report.json').is_file()
    run_path = dst / 'work' / 'run.json'
    before = run_path.read_bytes()
    before_record = json.loads(before)
    with pytest.raises(ValueError, match='already complete'):
        worker.run_final_trial(
            study['spec'],
            study['plan'],
            study['defaults'],
            'row_u8-p1-lr0.02-fixed',
        )
    assert run_path.read_bytes() == before
    assert json.loads(run_path.read_text())['state'] == 'succeeded'
    assert json.loads(run_path.read_text())['result'] == before_record['result']


def test_interrupted_final_kl_resumes_without_rewriting_inputs(
    tiny_study, monkeypatch
):
    study = tiny_study
    trial_id = 'row_u8-p1-lr0.002-fixed'
    from hi2 import checkpoint as checkpoint_module

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(checkpoint_module, 'write_checkpoints', interrupted)
    with pytest.raises(KeyboardInterrupt):
        worker.run_final_trial(
            study['spec'], study['plan'], study['defaults'], trial_id
        )
    dst = Path(study['spec']['run_root']) / 'trials' / trial_id
    work = dst / 'work'
    artifact = work / 'qad' / 'final_kl.pt'
    assert artifact.is_file()
    assert json.loads((work / 'run.json').read_text())['state'] == 'interrupted'
    (work / 'final_kl.json').unlink()
    modified = artifact.stat().st_mtime_ns
    monkeypatch.undo()
    logs = []
    result = worker.run_final_trial(
        study['spec'],
        study['plan'],
        study['defaults'],
        trial_id,
        log=lambda *a: logs.append(' '.join(map(str, a))),
    )
    assert any('final KL: reusing' in line for line in logs)
    assert artifact.stat().st_mtime_ns == modified
    assert (work / 'final_kl.json').is_file()
    assert result['timing']['resumed_final_kl'] is True
    assert json.loads((work / 'run.json').read_text())['state'] == 'succeeded'
