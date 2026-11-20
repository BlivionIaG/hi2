#!/usr/bin/env python3
"""Study worker: reuse complete QAD parents, vary only declared final-KL factors.

GPU commands require the existing full4B supervisor grant. This worker does not
lease GPUs, retry failures, or certify performance windows. Parent manifests are
read-only, content-pinned inputs, never recovery caches with rewritten keys.

Shared teacher contract (do not change): the original unquantized DEFAULT run's
``qad/head_inputs.pt`` is frozen once by ``prepare-teacher`` and reused by every
parent and every final trial. It is never copied into a parent workdir, never
rewritten, and never attributed to a new QAD run. New parents deliberately run
with ``final_kl=0`` and therefore do not create their own teacher-head file.

Execution-spec fields used here (all file items are ``{"path", "sha256"}``):

* ``factor_plan``, ``confirmatory_holdout``, ``snapshot``, ``source_manifest``,
  ``defaults_run``, ``qad_parent_runs``, ``parent_manifests``, ``shared_teacher``
* ``original_parent_compatibility_proof`` (required only for ``log-p1``)
* ``training_efficiency_overrides`` and ``training_cache_parity_proof``
* ``run_root``

The shared-teacher descriptor is written by ``prepare-teacher`` and pinned in
the execution spec. Its schema is local to this worker; all numerical products
it names (head inputs, layer caches, scan cache entries) are real files.
"""
import argparse
import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time


SCHEMA_VERSION = 1
PLAN_SHA256 = '4943b5d89a9bf17fecc9449106cab882cb5164caf4b83d344e93910d5d95cf00'
OLD_QUANTIZER_SHA = '995fc26317bd74d5d77653f42c4639086ac0d441'
REQUIRED_HI2_MODULES = (
    'hi2',
    'hi2.cache',
    'hi2.calibrate',
    'hi2.checkpoint',
    'hi2.cptc',
    'hi2.executor',
    'hi2.finalkl',
    'hi2.format',
    'hi2.layerwise',
    'hi2.pipeline',
    'hi2.qad',
    'hi2.runtime',
    'hi2.service',
    'hi2.stack',
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pinned(item):
    path = Path(item['path'])
    if digest(path) != item['sha256']:
        raise ValueError(f'changed study input: {path}')
    return path


def load(path):
    return json.loads(Path(path).read_text())


def object_digest(value):
    from hi2.runtime import object_digest as _object_digest

    return _object_digest(value)


def package_version(name):
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None


def current_environment():
    from hi2.runtime import environment

    value = dict(environment())
    value['tokenizers'] = package_version('tokenizers')
    return value


def environment_view(value):
    """The package/tokenizer identity the shared teacher binds.

    ``code_sha256`` deliberately stays out: it is checked separately (a new
    parent must equal the current implementation; the original parent is
    authorized only by its compatibility proof).
    """
    return {
        key: value.get(key)
        for key in ('python', 'torch', 'transformers', 'safetensors', 'tokenizers')
    }


def run_environment_view(value):
    """Historical run records predate the tokenizers field; compare the package
    versions they do carry. The tokenizer version is bound by the descriptor."""
    return {
        key: value.get(key)
        for key in ('python', 'torch', 'transformers', 'safetensors')
    }


def ids_sha256(ids):
    """SHA256 of the actual token IDs as int64 little-endian bytes."""
    import numpy as np
    import torch

    a = ids.detach().to(torch.int64).contiguous().cpu().numpy()
    return hashlib.sha256(
        np.asarray(a, dtype='<i8').tobytes(order='C')
    ).hexdigest()


def ids_pin(ids, qad, seq_len):
    import torch

    from hi2.cache import ids_fingerprint

    a = ids.detach().to(torch.int64).contiguous().cpu()
    return {
        'shape': list(a.shape),
        'dtype': 'int64',
        'sha256': ids_sha256(a),
        'ids_fingerprint': ids_fingerprint(a),
        'samples': qad.samples,
        'val': qad.val,
        'seq_len': seq_len,
        'seed': qad.seed,
        'sampling': qad.sampling,
    }


def tensor_digest(tensor):
    h = hashlib.sha256()
    a = tensor.detach().contiguous().cpu().numpy()
    h.update(str(tuple(a.shape)).encode())
    h.update(str(a.dtype).encode())
    h.update(a.tobytes(order='C'))
    return h.hexdigest()


def layout_entry(tensor):
    return {
        'shape': list(tensor.shape),
        'row_nbits': tensor_digest(tensor.row_nbits),
        'row_block': tensor_digest(tensor.row_block),
        'had': tensor_digest(tensor.had),
        'col_scale': (
            None if tensor.col_scale is None else tensor_digest(tensor.col_scale)
        ),
    }


def layout_signature(tensors):
    """Deterministic allocation/layout identity; code and scale values are not
    included because those are measured study outputs."""
    return object_digest(
        {name: layout_entry(t) for name, t in sorted(tensors.items())}
    )


def source_config(src):
    cfg = load(Path(src) / 'config.json')
    text = cfg.get('text_config', cfg)
    return int(text['hidden_size']), int(text['num_hidden_layers'])


def expected_layers(plan, src, limit_layers=None):
    _, layers = source_config(src)
    if limit_layers is not None:
        layers = min(layers, limit_layers)
    declared = plan.get('training_constants', {}).get('layers')
    if declared is not None and int(declared) != layers:
        raise ValueError('source layer count differs from the frozen plan')
    return layers


def validate_qad_overrides(plan, overrides):
    allowed = {'cptc_scale_units', 'cptc_projection_sweeps', 'steps'}
    if not isinstance(overrides, dict) or set(overrides) != allowed:
        raise ValueError('QAD parent overrides must be exactly the declared factors')
    factors = plan['quality_factors']
    constants = plan['training_constants']
    if overrides['cptc_scale_units'] not in factors['scale_units']:
        raise ValueError('undeclared QAD scale units')
    if overrides['cptc_projection_sweeps'] not in factors['projection_sweeps']:
        raise ValueError('undeclared QAD projection sweep')
    if overrides['steps'] != constants['qad_steps_per_layer']:
        raise ValueError('QAD step count differs from the frozen plan')
    return dict(overrides)


def validate_final_overrides(plan, overrides):
    allowed = {
        'final_kl',
        'final_lr_p',
        'final_cptc_train_tables',
        'final_cptc_table_lr',
    }
    if not isinstance(overrides, dict) or set(overrides) != allowed:
        raise ValueError('final overrides must be exactly the declared factors')
    factors = plan['quality_factors']
    constants = plan['training_constants']
    if overrides['final_kl'] != constants['final_kl_steps']:
        raise ValueError('final trials must execute the frozen step count')
    if overrides['final_lr_p'] not in factors['final_scale_lr']:
        raise ValueError('undeclared final scale learning rate')
    if overrides['final_cptc_train_tables'] not in factors[
        'final_shared_table_learning'
    ]:
        raise ValueError('undeclared final table mode')
    if overrides['final_cptc_table_lr'] != constants[
        'final_table_lr_when_enabled'
    ]:
        raise ValueError('final table learning rate differs from the frozen plan')
    return dict(overrides)


def declared_parent_qad(plan, parent):
    matches = [t for t in plan['quality_trials'] if t['qad_parent'] == parent]
    if not matches:
        raise ValueError('undeclared QAD parent')
    first = validate_qad_overrides(plan, matches[0]['qad_overrides'])
    if any(validate_qad_overrides(plan, t['qad_overrides']) != first for t in matches):
        raise ValueError('QAD parent settings disagree across its children')
    return first


def assert_defaults_sampling(defaults):
    qad = (defaults.get('config') or {}).get('qad')
    if not isinstance(qad, dict) or qad.get('sampling') != 'random':
        raise ValueError('this experiment requires QAD sampling=random')


def expected_config(defaults, dst, workdir, qad):
    document = copy.deepcopy(defaults['config'])
    document['dst'] = str(dst)
    document['workdir'] = str(workdir)
    document['qad'] = copy.deepcopy(qad)
    return document


def assert_config(document, expected):
    """A run config must match the frozen DEFAULT config on every setting except
    the explicitly substituted destination, workdir and declared QAD values."""
    if set(document) != set(expected):
        raise ValueError('run config keys differ from the DEFAULT config')
    for key in expected:
        if key in ('dst', 'workdir'):
            continue
        if document[key] != expected[key]:
            raise ValueError(f'run config differs from DEFAULT on {key}')


def scan_layer_caches(qdir, layers):
    """Validate real QAD layer caches and return their layout identity.

    Layer-relative trained norms live under ``norms`` and this experiment freezes
    them off; a nonempty entry is refused rather than silently dropped.
    """
    import torch

    paths = sorted(Path(qdir).glob('layer[0-9][0-9][0-9].pt'))
    if len(paths) != layers:
        raise ValueError(f'expected all {layers} QAD layer caches, found {len(paths)}')
    records, names, fingerprints, layout = [], set(), set(), {}
    for path in paths:
        rec = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        if not isinstance(rec, dict) or not isinstance(rec.get('tensors'), dict):
            raise ValueError(f'malformed QAD layer cache: {path}')
        if rec.get('norms'):
            raise ValueError(f'trained norms are not allowed in this study: {path}')
        if rec.get('dense'):
            raise ValueError(f'unexpected dense layer-cache fields: {path}')
        overlap = names.intersection(rec['tensors'])
        if overlap:
            raise ValueError(f'duplicate quantized tensors: {sorted(overlap)[:3]}')
        for name, tensor in rec['tensors'].items():
            if name in layout:
                raise ValueError(f'duplicate layout entry: {name}')
            layout[name] = layout_entry(tensor)
        names.update(rec['tensors'])
        fingerprints.add(rec['fingerprint'])
        records.append({'path': str(path), 'sha256': digest(path)})
    if len(fingerprints) != 1:
        raise ValueError('QAD layer fingerprints disagree')
    return records, names, next(iter(fingerprints)), object_digest(layout)


def pin_entry(path, kind, cfg):
    """Pin one real cache entry: metadata bytes always, and the table tensor
    bytes for a tables entry. Statistics/Hessian files needed by a GPTQ run are
    required to exist and are content-pinned too."""
    path = Path(path).resolve()
    complete = load(path / 'complete.json')
    key = load(path / 'key.json')
    files = {}
    for item in sorted(path.rglob('*')):
        if not item.is_file():
            continue
        rel = item.relative_to(path).as_posix()
        if '.tmp-' in rel or rel.endswith('.lock'):
            continue
        files[rel] = {'bytes': item.stat().st_size, 'sha256': digest(item)}
    for required in ('key.json', 'complete.json'):
        if required not in files:
            raise ValueError(f'cache entry is incomplete: {path}')
    if kind == 'stats':
        if 'layers.json' not in files:
            raise ValueError(f'statistics entry lacks layers.json: {path}')
        leaders = set(load(path / 'layers.json').get('leader', {}).values())
        needs_hessian = cfg.rounding == 'gptq' and not cfg.sequential
        if needs_hessian and complete.get('hessian') is not True:
            raise ValueError(f'statistics entry lacks required Hessians: {path}')
        if complete.get('hessian'):
            for leader in leaders:
                rel = f'sigma/{leader.replace("/", "_")}.pt'
                if rel not in files:
                    raise ValueError(
                        f'statistics entry lacks Sigma for {leader}: {path}'
                    )
        return {
            'path': str(path),
            'complete': complete,
            'key': key,
            'files': files,
            'hessian': bool(complete.get('hessian')),
            'tensors': None,
        }
    names = complete.get('tensors')
    if not isinstance(names, list) or not names:
        raise ValueError(f'tables entry lacks a tensor list: {path}')
    expected = {f'tables/{name.replace("/", "_")}.pt' for name in names}
    if not expected.issubset(files):
        raise ValueError(f'tables entry lacks table tensors: {path}')
    extra = set(files) - expected - {'complete.json', 'key.json'}
    if extra:
        raise ValueError(f'unexpected files in tables entry: {sorted(extra)[:3]}')
    return {
        'path': str(path),
        'complete': complete,
        'key': key,
        'files': files,
        'hessian': bool(complete.get('hessian')),
        'tensors': names,
    }


def verify_entry(pin, kind):
    path = Path(pin['path']).resolve()
    for rel, meta in pin['files'].items():
        item = path / rel
        if (
            not item.is_file()
            or item.stat().st_size != meta['bytes']
            or digest(item) != meta['sha256']
        ):
            raise ValueError(f'cache entry changed: {item}')
    if load(path / 'complete.json') != pin['complete']:
        raise ValueError(f'cache entry metadata changed: {path}')
    if load(path / 'key.json') != pin['key']:
        raise ValueError(f'cache entry key changed: {path}')
    if kind == 'tables':
        names = load(path / 'complete.json').get('tensors')
        if names != pin['tensors']:
            raise ValueError(f'cache table coverage changed: {path}')


def cache_ref_pin(workdir, cfg):
    ref_path = Path(workdir) / 'cache_ref.json'
    if not ref_path.is_file():
        raise ValueError(f'missing cache reference: {ref_path}')
    value = load(ref_path)
    stats = Path(value['stats']).resolve()
    tables = Path(value['tables']).resolve()
    if stats.parent.parent != tables.parent.parent:
        raise ValueError('statistics and tables entries use different cache roots')
    return {
        'path': str(ref_path.resolve()),
        'sha256': digest(ref_path),
        'value': value,
        'stats': pin_entry(stats, 'stats', cfg),
        'tables': pin_entry(tables, 'tables', cfg),
    }


def regenerate_teacher_ids(rc):
    """The actual 264 x 2048 random windows declared by the DEFAULT recipe."""
    from transformers import AutoTokenizer

    from hi2.pipeline import sample_calibration_windows

    if rc.qad is None or rc.qad.sampling != 'random':
        raise ValueError('common teacher requires QAD sampling=random')
    if rc.calib is None:
        raise ValueError('common teacher requires calibration text')
    tokenizer = AutoTokenizer.from_pretrained(rc.src, trust_remote_code=False)
    ids = sample_calibration_windows(
        rc.calib,
        tokenizer,
        rc.qad.samples + rc.qad.val,
        rc.seq_len,
        rc.qad.seed,
        False,
    )
    import torch

    return ids.to(torch.int64)


def load_common_teacher(spec, plan, defaults):
    item = spec['shared_teacher']
    path = pinned(item)
    document = load(path)
    if document.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('unsupported shared-teacher descriptor')
    if document.get('kind') != 'common-unquantized-default-teacher':
        raise ValueError('wrong shared-teacher descriptor kind')
    if document.get('defaults_run') != spec['defaults_run']:
        raise ValueError('shared teacher is not bound to the DEFAULT run')
    if document.get('factor_plan') != spec['factor_plan']:
        raise ValueError('shared teacher is not bound to the frozen factor plan')
    if document.get('defaults', {}).get('run_id') != defaults.get('run_id'):
        raise ValueError('shared teacher names another DEFAULT run')
    if document.get('defaults', {}).get('state') != 'succeeded':
        raise ValueError('shared teacher DEFAULT run did not succeed')
    if document.get('source') != defaults.get('source'):
        raise ValueError('shared teacher source identity changed')
    if document.get('calibration') != defaults.get('calibration'):
        raise ValueError('shared teacher calibration identity changed')
    if document.get('producer_environment', {}).get('code_sha256') != (
        defaults.get('environment', {}).get('code_sha256')
    ):
        raise ValueError('shared teacher producer code hash changed')
    if environment_view(document['producer_environment']) != environment_view(
        current_environment()
    ):
        raise ValueError('current tokenizer/environment differs from the teacher')
    if document.get('compute_dtype') != defaults['config'].get('compute_dtype'):
        raise ValueError('shared teacher compute dtype differs from DEFAULT')
    if document.get('qad_sampling') != 'random':
        raise ValueError('shared teacher did not use random QAD sampling')
    if not math.isfinite(float(document.get('allocation_error_ratio', math.nan))):
        raise ValueError('shared teacher lacks the DEFAULT allocation ratio')
    if float(document['allocation_error_ratio']) != float(
        defaults['result']['allocation_error_ratio']
    ):
        raise ValueError('shared teacher allocation differs from DEFAULT')
    head = document.get('head')
    if not isinstance(head, dict):
        raise ValueError('shared teacher lacks a head-input descriptor')
    head_path = pinned(head)
    if not head_path.is_file():
        raise ValueError('shared teacher head-input file is missing')
    if head.get('fingerprint') != document.get('layerwise_fingerprint'):
        raise ValueError('shared teacher head fingerprint disagrees')
    if not isinstance(head.get('shape'), list) or head.get('dtype') != 'torch.float32':
        raise ValueError('shared teacher head descriptor is malformed')
    ref = document.get('cache_ref')
    if not isinstance(ref, dict) or not isinstance(ref.get('value'), dict):
        raise ValueError('shared teacher lacks pinned cache references')
    ref_path = Path(ref['path'])
    if digest(ref_path) != ref['sha256'] or load(ref_path) != ref['value']:
        raise ValueError('pinned shared cache_ref.json changed')
    verify_entry(ref['stats'], 'stats')
    verify_entry(ref['tables'], 'tables')
    ids = document.get('calibration_ids')
    if not isinstance(ids, dict) or not isinstance(ids.get('shape'), list):
        raise ValueError('shared teacher lacks pinned calibration IDs')
    if len(str(ids.get('sha256', ''))) != 64:
        raise ValueError('shared teacher calibration ID hash is malformed')
    if list(head['shape'][:2]) != [ids['shape'][0], ids['seq_len']]:
        raise ValueError('shared teacher head does not match its calibration IDs')
    return document


def require_common_identity(teacher, rc, identity):
    if identity.get('source') != teacher.get('source'):
        raise ValueError('run source differs from the shared teacher source')
    if identity.get('calibration') != teacher.get('calibration'):
        raise ValueError('run calibration differs from the shared teacher calibration')
    if rc.compute_dtype != teacher.get('compute_dtype'):
        raise ValueError('run compute dtype differs from the shared teacher')
    if rc.qad is None or rc.qad.dtype != teacher.get('qad_dtype'):
        raise ValueError('run QAD dtype differs from the shared teacher')
    if rc.qad.sampling != 'random' or teacher.get('qad_sampling') != 'random':
        raise ValueError('this experiment requires QAD sampling=random')
    for key in ('seq_len', 'samples', 'val', 'seed'):
        expected = teacher['calibration_ids'].get(key)
        actual = getattr(rc, key) if key == 'seq_len' else getattr(rc.qad, key)
        if actual != expected:
            raise ValueError(f'run calibration setting differs on {key}')
    if environment_view(current_environment()) != environment_view(
        teacher['producer_environment']
    ):
        raise ValueError('current tokenizer/environment differs from the teacher')
    ids = regenerate_teacher_ids(rc)
    actual = ids_pin(ids, rc.qad, rc.seq_len)
    if actual != teacher['calibration_ids']:
        raise ValueError('regenerated calibration IDs differ from the shared teacher')
    return ids


def load_teacher_head(teacher):
    import torch

    path = pinned(teacher['head'])
    rec = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if not isinstance(rec, dict) or rec.get('fingerprint') != teacher[
        'layerwise_fingerprint'
    ]:
        raise ValueError('shared teacher head fingerprint changed')
    head = rec.get('head_inputs')
    if head is None or str(head.dtype) != teacher['head']['dtype']:
        raise ValueError('shared teacher head dtype changed')
    if list(head.shape) != teacher['head']['shape']:
        raise ValueError('shared teacher head shape changed')
    return head


def cache_preflight(teacher, rc, identity):
    from hi2 import cache as C
    from hi2 import pipeline

    if rc.cache_dir is None:
        raise ValueError('the study requires the original shared cache directory')
    root = Path(rc.cache_dir).resolve()
    ref = teacher['cache_ref']
    stats = Path(ref['stats']['path']).resolve()
    tables = Path(ref['tables']['path']).resolve()
    if stats.parent.parent != root or tables.parent.parent != root:
        raise ValueError('run cache root differs from the pinned cache root')
    verify_entry(ref['stats'], 'stats')
    verify_entry(ref['tables'], 'tables')
    rc._identity = identity
    ids = pipeline._calib_ids(rc)
    import torch

    devices = [torch.device(d) for d in rc.devices]
    sent = C.Entry(root, C.stats_key(rc, ids, devices))
    tent = C.Entry(root, C.tables_key(sent.key, rc.cfg))
    if sent.path.resolve() != stats:
        raise ValueError('run statistics key does not match the pinned entry')
    if tent.path.resolve() != tables:
        raise ValueError('run tables key does not match the pinned entry')
    need = 'hessian' if rc.cfg.rounding == 'gptq' and not rc.cfg.sequential else None
    if not sent.complete(need):
        raise ValueError(f'pinned statistics cache entry is incomplete: {sent.path}')
    if not tent.complete():
        raise ValueError(f'pinned tables cache entry is incomplete: {tent.path}')
    if tent.info().get('tensors') != ref['tables']['tensors']:
        raise ValueError('pinned tables entry changed its tensor list')
    return stats, tables


def load_pinned_tables(teacher, expected):
    from hi2.calibrate import TensorTable
    import torch

    pin = teacher['cache_ref']['tables']
    verify_entry(pin, 'tables')
    names = list(pin['tensors'])
    if set(names) != set(expected):
        raise ValueError('pinned tables do not cover exactly the parent tensors')
    tables = {}
    for name in names:
        path = Path(pin['path']) / 'tables' / (name.replace('/', '_') + '.pt')
        table = torch.load(path, weights_only=False)
        if not isinstance(table, TensorTable):
            raise ValueError(f'not a real TensorTable: {path}')
        tables[name] = table
    return tables


def load_parent_tensors(parent):
    import torch

    tensors = {}
    for item in parent['layers']:
        rec = torch.load(
            pinned(item), map_location='cpu', weights_only=False, mmap=True
        )
        if rec.get('fingerprint') != parent['layerwise_fingerprint']:
            raise ValueError('QAD parent fingerprint changed')
        if rec.get('norms'):
            raise ValueError('trained norms are not allowed in this study')
        if rec.get('dense'):
            raise ValueError('unexpected dense layer-cache fields')
        overlap = tensors.keys() & rec['tensors'].keys()
        if overlap:
            raise ValueError(f'duplicate QAD parent fields: {sorted(overlap)[:3]}')
        tensors.update(rec['tensors'])
    if len(tensors) != parent['quantized_tensors']:
        raise ValueError('incomplete QAD parent tensor set')
    if layout_signature(tensors) != parent['layout_signature']:
        raise ValueError('QAD parent layout signature changed')
    return tensors


def table_picks(tables, tensors):
    import torch

    picks = {}
    for name, tensor in tensors.items():
        table = tables[name]
        if tuple(table.shape) != tuple(tensor.shape):
            raise ValueError(f'QAD shape differs from calibration: {name}')
        pick = torch.full((tensor.shape[0],), -1, dtype=torch.int64)
        for index, (bits, block) in enumerate(table.candidates):
            pick[(tensor.row_nbits == bits) & (tensor.row_block == block)] = index
        if (pick < 0).any() or not torch.equal(table.had, tensor.had):
            raise ValueError(f'QAD row layout differs from calibration: {name}')
        if (table.col_scale is None) != (tensor.col_scale is None) or (
            table.col_scale is not None
            and not torch.equal(table.col_scale, tensor.col_scale)
        ):
            raise ValueError(f'QAD smoothing differs from calibration: {name}')
        picks[name] = pick
    return picks


def validate_numerical_proof(proof, plan):
    numerical = proof.get('numerical')
    if not isinstance(numerical, dict) or numerical.get('passed') is not True:
        raise ValueError('compatibility proof lacks a passed numerical receipt')
    try:
        worst = float(numerical['max_abs_diff'])
        tolerance = float(numerical['tolerance'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('compatibility proof lacks numerical bounds') from exc
    if not (math.isfinite(worst) and math.isfinite(tolerance)) or worst > tolerance:
        raise ValueError('compatibility proof numerical difference exceeds tolerance')
    if numerical.get('control_artifact_digest') != plan['immutable_controls'][
        'cptc_artifact_digest'
    ]:
        raise ValueError('compatibility proof names another control artifact')
    if len(str(numerical.get('parent_artifact_digest', ''))) != 64:
        raise ValueError('compatibility proof lacks the parent artifact digest')
    if not isinstance(numerical.get('metric'), str) or not numerical['metric']:
        raise ValueError('compatibility proof lacks a named metric')
    receipt = numerical.get('receipt')
    if not isinstance(receipt, dict):
        raise ValueError('compatibility proof lacks a pinned receipt')
    pinned(receipt)


def load_compatibility_proof(
    spec, plan, defaults, parent_run, parent_item=None, require_manifest=True
):
    item = spec['original_parent_compatibility_proof']
    proof = load(pinned(item))
    if proof.get('passed') is not True:
        raise ValueError('original parent lacks a passed compatibility proof')
    if proof.get('old_quantizer_sha') != OLD_QUANTIZER_SHA:
        raise ValueError('compatibility proof names another old quantizer')
    if plan['immutable_controls']['quantizer_source_sha'] != OLD_QUANTIZER_SHA:
        raise ValueError('the frozen plan names another original quantizer')
    if proof.get('old_environment_code_sha256') != parent_run['environment'].get(
        'code_sha256'
    ):
        raise ValueError('compatibility proof names another producer code hash')
    if proof.get('new_source_manifest') != spec['source_manifest']:
        raise ValueError('compatibility proof is not bound to the new source')
    if proof.get('defaults_run') != spec['defaults_run']:
        raise ValueError('compatibility proof is not bound to the DEFAULT run')
    if proof.get('factor_plan') != spec['factor_plan']:
        raise ValueError('compatibility proof is not bound to the frozen plan')
    if proof.get('parent_run') != spec['qad_parent_runs']['log-p1']:
        raise ValueError('compatibility proof is not bound to the log-p1 run')
    if require_manifest and proof.get('parent_manifest') != parent_item:
        raise ValueError('compatibility proof is bound to another parent manifest')
    validate_numerical_proof(proof, plan)
    return item, proof


def load_cache_parity_proof(spec, budget):
    item = spec['training_cache_parity_proof']
    proof = load(pinned(item))
    if proof.get('gpu_passed') is not True:
        raise ValueError('cache needs a passed GPU numerical parity proof')
    if proof.get('new_source_manifest') != spec['source_manifest']:
        raise ValueError('cache proof is not bound to the new source')
    if proof.get('cache_budget_mb') != budget:
        raise ValueError('cache proof is bound to another cache budget')
    if proof.get('receipt') is not None:
        pinned(proof['receipt'])
    return item


def efficiency_budget(spec, plan):
    overrides = spec.get('training_efficiency_overrides', {})
    if set(overrides) - {'final_cptc_cache_mb'}:
        raise ValueError('implementation-only overrides must not change quality factors')
    budget = overrides.get('final_cptc_cache_mb', 0)
    if type(budget) is not int or budget not in plan['training_efficiency'][
        'cache_budgets_mb'
    ]:
        raise ValueError('undeclared training cache budget')
    item = load_cache_parity_proof(spec, budget) if budget else None
    return budget, item


def teacher_descriptor(spec, plan, defaults):
    """Build the immutable shared-teacher descriptor from the original DEFAULT
    run. CPU-only; no QAD, pipeline or GPU call is made."""
    import torch

    from hi2.service import config_from_dict

    assert_defaults_sampling(defaults)
    if defaults.get('state') != 'succeeded':
        raise ValueError('DEFAULT reference run did not complete')
    if defaults['config'].get('qad', {}).get('final_kl', 0) <= 0:
        raise ValueError('DEFAULT run did not execute a final-KL teacher pass')
    rc = config_from_dict({'schema_version': 1, 'config': defaults['config']})
    hidden, _ = source_config(rc.src)
    expected = expected_layers(plan, rc.src, rc.limit_layers)
    qdir = Path(rc.workdir) / 'qad'
    records, names, fingerprint, layout = scan_layer_caches(qdir, expected)
    head_path = qdir / 'head_inputs.pt'
    if not head_path.is_file():
        raise ValueError('DEFAULT run did not save qad/head_inputs.pt')
    head_rec = torch.load(
        head_path, map_location='cpu', weights_only=False, mmap=True
    )
    if not isinstance(head_rec, dict):
        raise ValueError('malformed DEFAULT teacher head-input file')
    if head_rec.get('fingerprint') != fingerprint:
        raise ValueError('DEFAULT head fingerprint differs from its layer caches')
    head = head_rec.get('head_inputs')
    if head is None or head.dtype != torch.float32:
        raise ValueError('DEFAULT teacher head inputs are not float32')
    if len(names) != int(defaults['result']['tensors']):
        raise ValueError('DEFAULT layer caches do not cover its reported tensors')
    ids = regenerate_teacher_ids(rc)
    expected_shape = (ids.shape[0], rc.seq_len, hidden)
    if tuple(head.shape) != expected_shape:
        raise ValueError(
            f'DEFAULT teacher shape {tuple(head.shape)} != {expected_shape}'
        )
    if ids.shape != (rc.qad.samples + rc.qad.val, rc.seq_len):
        raise ValueError('DEFAULT calibration windows have the wrong shape')
    ref = cache_ref_pin(rc.workdir, rc.cfg)
    environment = dict(defaults['environment'])
    environment['tokenizers'] = package_version('tokenizers')
    return {
        'schema_version': SCHEMA_VERSION,
        'kind': 'common-unquantized-default-teacher',
        'defaults_run': spec['defaults_run'],
        'factor_plan': spec['factor_plan'],
        'defaults': {
            'run_id': defaults.get('run_id'),
            'state': defaults.get('state'),
            'dst': defaults['config'].get('dst'),
            'workdir': defaults['config'].get('workdir'),
        },
        'producer_environment': environment,
        'source': defaults['source'],
        'calibration': defaults['calibration'],
        'compute_dtype': defaults['config']['compute_dtype'],
        'qad_dtype': defaults['config']['qad']['dtype'],
        'qad_sampling': defaults['config']['qad']['sampling'],
        'calibration_ids': ids_pin(ids, rc.qad, rc.seq_len),
        'layerwise_fingerprint': fingerprint,
        'layers': records,
        'head': {
            'path': str(head_path.resolve()),
            'sha256': digest(head_path),
            'fingerprint': head_rec['fingerprint'],
            'shape': list(head.shape),
            'dtype': str(head.dtype),
            'bytes': head_path.stat().st_size,
        },
        'cache_ref': ref,
        'allocation_error_ratio': float(
            defaults['result']['allocation_error_ratio']
        ),
        'layout_signature': layout,
        'reuse': (
            'One immutable unquantized DEFAULT teacher-head artifact shared by '
            'every parent and final trial; it is not a per-parent QAD output.'
        ),
    }


def require_parent_environment(producer, teacher):
    """Bind a parent's actual producer tokenizer/packages to the shared teacher.

    New parents record a producer-environment receipt (``hi2.runtime.environment``
    predates the tokenizers field); the original log-p1 run predates the field and
    is authorized by its compatibility proof plus the package versions it does
    carry."""
    if not isinstance(producer, dict):
        raise ValueError('QAD parent lacks a producer environment')
    if 'tokenizers' in producer:
        if environment_view(producer) != environment_view(
            teacher['producer_environment']
        ):
            raise ValueError('QAD parent tokenizer/environment differs from DEFAULT')
    elif run_environment_view(producer) != run_environment_view(
        teacher['producer_environment']
    ):
        raise ValueError('QAD parent tokenizer/environment differs from DEFAULT')


def parent_manifest_record(spec, plan, defaults, parent):
    """Validate one real QAD parent and describe it. No new numerical work."""
    from hi2.qad import qad_fingerprint_fields
    from hi2.service import config_from_dict

    assert_defaults_sampling(defaults)
    declared = declared_parent_qad(plan, parent)
    run_item = spec['qad_parent_runs'][parent]
    run_path = pinned(run_item)
    run = load(run_path)
    if run.get('state') != 'succeeded':
        raise ValueError('QAD parent run did not complete')
    teacher = load_common_teacher(spec, plan, defaults)
    if run.get('source') != teacher['source']:
        raise ValueError('QAD parent source differs from DEFAULT source')
    if run.get('calibration') != teacher['calibration']:
        raise ValueError('QAD parent calibration differs from DEFAULT calibration')
    qad_expected = copy.deepcopy(defaults['config']['qad'])
    qad_expected.update(declared)
    if parent != 'log-p1':
        qad_expected['final_kl'] = 0
    expected = expected_config(
        defaults,
        run['config']['dst'],
        run['config']['workdir'],
        qad_expected,
    )
    assert_config(run['config'], expected)
    rc = config_from_dict({'schema_version': 1, 'config': run['config']})
    if rc.qad is None or rc.qad.sampling != 'random':
        raise ValueError('QAD parent did not use sampling=random')
    expected_rc = config_from_dict({'schema_version': 1, 'config': expected})
    if qad_fingerprint_fields(rc.qad) != qad_fingerprint_fields(expected_rc.qad):
        raise ValueError('QAD parent used different layerwise QAD settings')
    layers = expected_layers(plan, rc.src, rc.limit_layers)
    records, names, fingerprint, layout = scan_layer_caches(
        Path(rc.workdir) / 'qad', layers
    )
    result = run.get('result') or {}
    if result.get('tensors') != len(names):
        raise ValueError('QAD parent report does not cover its layer caches')
    layout_name = run['config'].get('moe_layout', 'per-expert')
    if result.get('moe_layout', 'per-expert') != layout_name:
        raise ValueError('QAD parent layout differs from its run config')
    if float(result.get('allocation_error_ratio', math.nan)) != float(
        teacher['allocation_error_ratio']
    ):
        raise ValueError('QAD parent allocation differs from DEFAULT')
    if layout != teacher['layout_signature']:
        raise ValueError('QAD parent layout differs from the original log-p1 layout')
    ref = cache_ref_pin(rc.workdir, rc.cfg)
    if ref['sha256'] != teacher['cache_ref']['sha256']:
        raise ValueError('QAD parent cache_ref.json differs from DEFAULT')
    if ref['value'] != teacher['cache_ref']['value']:
        raise ValueError('QAD parent cache references differ from DEFAULT')
    if ref['stats']['path'] != teacher['cache_ref']['stats']['path']:
        raise ValueError('QAD parent statistics entry differs from DEFAULT')
    if ref['tables']['path'] != teacher['cache_ref']['tables']['path']:
        raise ValueError('QAD parent tables entry differs from DEFAULT')
    environment = current_environment()
    worker = Path(__file__).resolve()
    if parent == 'log-p1':
        producer_environment = dict(run['environment'])
        environment_receipt = None
        proof_item, _ = load_compatibility_proof(
            spec, plan, defaults, run, require_manifest=False
        )
    else:
        if run['environment'].get('code_sha256') != environment.get('code_sha256'):
            raise ValueError('new QAD parent producer code differs from this worker')
        receipt_path = Path(rc.workdir) / 'study-producer-environment.json'
        if not receipt_path.is_file():
            raise ValueError('new QAD parent lacks a producer environment receipt')
        receipt = load(receipt_path)
        if (
            receipt.get('schema_version') != SCHEMA_VERSION
            or receipt.get('kind') != 'qad-parent-producer-environment'
            or receipt.get('run_id') != run.get('run_id')
            or receipt.get('source') != run.get('source')
            or receipt.get('calibration') != run.get('calibration')
            or receipt.get('config_sha256') != object_digest(run['config'])
            or receipt.get('worker_sha256') != digest(worker)
        ):
            raise ValueError('new QAD parent producer environment receipt changed')
        receipt_env = receipt.get('environment')
        if not isinstance(receipt_env, dict) or receipt_env.get(
            'code_sha256'
        ) != run['environment'].get('code_sha256'):
            raise ValueError('new QAD parent receipt names another producer code')
        require_parent_environment(receipt_env, teacher)
        producer_environment = dict(run['environment'])
        producer_environment['tokenizers'] = receipt_env.get('tokenizers')
        environment_receipt = {
            'path': str(receipt_path.resolve()),
            'sha256': digest(receipt_path),
            'run_id': receipt['run_id'],
            'worker_sha256': receipt['worker_sha256'],
            'environment': receipt_env,
        }
        proof_item = None
    require_parent_environment(producer_environment, teacher)
    return {
        'schema_version': SCHEMA_VERSION,
        'kind': 'complete-pre-final-QAD-parent',
        'parent': parent,
        'producer': {
            'run': run_item,
            'run_id': run.get('run_id'),
            'environment': producer_environment,
            'environment_receipt': environment_receipt,
            'source': run['source'],
            'calibration': run['calibration'],
            'config_sha256': object_digest(run['config']),
            'allocation': {
                'target_bpw': run['config'].get('target_bpw'),
                'max_error_ratio': run['config'].get('max_error_ratio'),
                'allocation_bpw': result.get('allocation_bpw'),
                'allocation_error_ratio': result.get('allocation_error_ratio'),
                'feasible': result.get('feasible'),
            },
        },
        'preparer': {
            'environment': environment,
            'source_manifest': spec['source_manifest'],
            'worker': {'path': str(worker), 'sha256': digest(worker)},
            'snapshot': spec.get('snapshot'),
        },
        'qad': {
            'fingerprint_fields': qad_fingerprint_fields(rc.qad),
            'declared_overrides': declared,
            'final_kl': rc.qad.final_kl,
        },
        'layerwise_fingerprint': fingerprint,
        'layers': records,
        'quantized_tensors': len(names),
        'layout_signature': layout,
        'common_teacher': {
            'descriptor': spec['shared_teacher'],
            'descriptor_sha256': digest(spec['shared_teacher']['path']),
            'head': teacher['head'],
            'fingerprint': teacher['layerwise_fingerprint'],
            'reuse': teacher['reuse'],
        },
        'cache_ref': {
            'path': ref['path'],
            'sha256': ref['sha256'],
            'value': ref['value'],
            'stats': ref['stats']['path'],
            'tables': ref['tables']['path'],
        },
        'compatibility_proof': proof_item,
        'factor_plan': spec['factor_plan'],
        'source_manifest': spec['source_manifest'],
        'scope': (
            'All real layer caches; final_kl=0 for new parents. The shared head '
            'is the unquantized DEFAULT teacher and is not a parent artifact.'
        ),
    }


def prepare_teacher_core(spec, plan, defaults, output):
    output = Path(output)
    if output.exists():
        raise ValueError('refusing to overwrite a shared-teacher descriptor')
    document = teacher_descriptor(spec, plan, defaults)
    from hi2.runtime import atomic_json

    atomic_json(output, document)
    return document


def run_qad_parent(spec, plan, defaults, parent, log=print):
    from hi2.service import execute_run, resolve_config

    if parent == 'log-p1':
        raise ValueError('preserve the original log-p1 control; explicit parity proof governs reuse')
    declared = declared_parent_qad(plan, parent)
    teacher = load_common_teacher(spec, plan, defaults)
    root = Path(spec['run_root']) / 'parents' / parent
    qad_expected = copy.deepcopy(defaults['config']['qad'])
    qad_expected.update(declared)
    qad_expected['final_kl'] = 0
    document = expected_config(defaults, root / 'model', root / 'work', qad_expected)
    rc = resolve_config(recipe(document), log)
    identity = rc._identity
    if identity is None:
        from hi2.service import run_identity

        identity = run_identity(rc)
        rc._identity = identity
    require_common_identity(teacher, rc, identity)
    cache_preflight(teacher, rc, identity)
    run_path = Path(rc.workdir) / 'run.json'
    if run_path.exists() and load(run_path).get('state') == 'succeeded':
        receipt_path = Path(rc.workdir) / 'study-producer-environment.json'
        if not receipt_path.is_file():
            raise ValueError('refusing a completed parent without an environment receipt')
        receipt = load(receipt_path)
        if (
            receipt.get('run_id') != load(run_path).get('run_id')
            or receipt.get('worker_sha256') != digest(Path(__file__).resolve())
            or environment_view(receipt.get('environment') or {})
            != environment_view(current_environment())
        ):
            raise ValueError('refusing a parent produced in another environment')
    result = execute_run(rc, log)
    if float(result.get('allocation_error_ratio', math.nan)) != float(
        teacher['allocation_error_ratio']
    ):
        raise ValueError('new QAD parent allocation differs from DEFAULT')
    from hi2.runtime import atomic_json

    record = load(Path(rc.workdir) / 'run.json')
    atomic_json(
        Path(rc.workdir) / 'study-producer-environment.json',
        {
            'schema_version': SCHEMA_VERSION,
            'kind': 'qad-parent-producer-environment',
            'run_id': record.get('run_id'),
            'source': record.get('source'),
            'calibration': record.get('calibration'),
            'config_sha256': object_digest(record['config']),
            'worker_sha256': digest(Path(__file__).resolve()),
            'environment': current_environment(),
        },
    )
    return result


def expected_final_fingerprint(rc, parent_fingerprint):
    from hi2.qad import final_fingerprint_fields, fingerprint

    return fingerprint(
        {
            'layerwise': parent_fingerprint,
            'final': final_fingerprint_fields(rc.qad),
            'warmup': rc.qad.warmup,
            'seed': rc.qad.seed,
        }
    )


def final_info(rc, resumed, artifact):
    from hi2.runtime import atomic_json
    import torch

    path = Path(rc.workdir) / 'final_kl.json'
    if resumed:
        rec = torch.load(artifact, weights_only=False)
        saved = json.loads(json.dumps(rec['info']))
        if path.exists():
            info = load(path)
            if info != saved:
                raise ValueError('final_kl.json disagrees with the resumed artifact')
        else:
            info = saved
            atomic_json(path, info)
    else:
        info = load(path)
    if info.get('steps') != rc.qad.final_kl:
        raise ValueError('final-KL artifact did not execute the declared steps')
    for key in ('seconds', 'val_after'):
        value = info.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f'final-KL receipt lacks a finite {key}')
    if info['seconds'] < 0:
        raise ValueError('final-KL receipt has negative training seconds')
    return info


def run_final_trial(spec, plan, defaults, trial_id, log=print):
    from hi2 import pipeline
    from hi2.checkpoint import Output, validate_checkpoint, write_checkpoints
    from hi2.cptc import checkpoint_config, get_codec
    from hi2.executor import Executor
    from hi2.qad import (
        cptc_codecs_from_tensors,
        qad_fingerprint_fields,
        qad_meta,
    )
    from hi2.runtime import RunSession, atomic_json, exclusive_lock, read_run
    from hi2.service import config_from_dict, resolve_config, run_identity
    from hi2.stack import restack
    import torch

    phase_start = time.monotonic()
    trial = next((t for t in plan['quality_trials'] if t['id'] == trial_id), None)
    if trial is None:
        raise ValueError('undeclared final-KL trial')
    if trial.get('preserved_control'):
        raise ValueError('the DEFAULT control must be referenced, never regenerated')
    parent_id = trial['qad_parent']
    assert_defaults_sampling(defaults)
    teacher = load_common_teacher(spec, plan, defaults)
    parent_item = spec['parent_manifests'][parent_id]
    parent = load(pinned(parent_item))
    if (
        parent.get('kind') != 'complete-pre-final-QAD-parent'
        or parent.get('parent') != parent_id
    ):
        raise ValueError('wrong QAD parent manifest')
    if parent.get('factor_plan') != spec['factor_plan']:
        raise ValueError('QAD parent manifest is bound to another factor plan')
    if parent.get('source_manifest') != spec['source_manifest']:
        raise ValueError('QAD parent manifest is bound to another source')
    if parent.get('preparer', {}).get('source_manifest') != spec['source_manifest']:
        raise ValueError('QAD parent was prepared against another source')
    if parent.get('common_teacher', {}).get('descriptor') != spec['shared_teacher']:
        raise ValueError('QAD parent names another shared teacher')
    if parent['common_teacher'].get('descriptor_sha256') != digest(
        spec['shared_teacher']['path']
    ):
        raise ValueError('QAD parent shared-teacher descriptor changed')
    if parent.get('layout_signature') != teacher['layout_signature']:
        raise ValueError('QAD parent layout differs from the original log-p1 layout')
    if parent.get('cache_ref', {}).get('sha256') != teacher['cache_ref']['sha256']:
        raise ValueError('QAD parent cache_ref.json differs from DEFAULT')
    if parent['cache_ref'].get('value') != teacher['cache_ref']['value']:
        raise ValueError('QAD parent cache references differ from DEFAULT')
    if parent['cache_ref'].get('stats') != teacher['cache_ref']['stats']['path']:
        raise ValueError('QAD parent statistics entry differs from DEFAULT')
    if parent['cache_ref'].get('tables') != teacher['cache_ref']['tables']['path']:
        raise ValueError('QAD parent tables entry differs from DEFAULT')
    parent_run_item = parent.get('producer', {}).get('run')
    if parent_run_item != spec['qad_parent_runs'][parent_id]:
        raise ValueError('QAD parent manifest names another producer run')
    parent_run = load(pinned(parent_run_item))
    if parent_run.get('state') != 'succeeded':
        raise ValueError('QAD parent run did not complete')
    if object_digest(parent_run['config']) != parent['producer'].get('config_sha256'):
        raise ValueError('QAD parent producer config changed')
    if parent_run.get('source') != teacher['source']:
        raise ValueError('QAD parent source differs from the shared teacher')
    if parent_run.get('calibration') != teacher['calibration']:
        raise ValueError('QAD parent calibration differs from the shared teacher')
    declared = declared_parent_qad(plan, parent_id)
    qad_expected = copy.deepcopy(defaults['config']['qad'])
    qad_expected.update(declared)
    if parent_id != 'log-p1':
        qad_expected['final_kl'] = 0
    expected_parent = expected_config(
        defaults,
        parent_run['config']['dst'],
        parent_run['config']['workdir'],
        qad_expected,
    )
    assert_config(parent_run['config'], expected_parent)
    expected_parent_rc = config_from_dict(
        {'schema_version': 1, 'config': expected_parent}
    )
    parent_rc = config_from_dict(
        {'schema_version': 1, 'config': parent_run['config']}
    )
    if qad_fingerprint_fields(parent_rc.qad) != qad_fingerprint_fields(
        expected_parent_rc.qad
    ):
        raise ValueError('QAD parent used different layerwise settings')
    producer = parent.get('producer') or {}
    receipt_item = producer.get('environment_receipt')
    if receipt_item is not None:
        receipt = load(pinned(receipt_item))
        if (
            receipt.get('run_id') != parent_run.get('run_id')
            or receipt.get('config_sha256') != producer.get('config_sha256')
            or receipt.get('source') != parent_run.get('source')
            or receipt.get('calibration') != parent_run.get('calibration')
            or receipt.get('worker_sha256') != receipt_item.get('worker_sha256')
            or receipt.get('worker_sha256') != digest(Path(__file__).resolve())
            or receipt.get('environment') != receipt_item.get('environment')
        ):
            raise ValueError('QAD parent producer environment receipt changed')
    require_parent_environment(producer.get('environment'), teacher)
    environment = current_environment()
    proof_item = None
    if parent_id == 'log-p1':
        proof_item, _ = load_compatibility_proof(
            spec, plan, defaults, parent_run, parent_item=parent_item
        )
    elif parent_run['environment'].get('code_sha256') != environment.get(
        'code_sha256'
    ):
        raise ValueError('QAD parent producer code differs from this worker')
    allocation = parent['producer'].get('allocation') or {}
    if allocation.get('target_bpw') != parent_run['config'].get('target_bpw'):
        raise ValueError('QAD parent allocation target changed')
    if allocation.get('max_error_ratio') != parent_run['config'].get(
        'max_error_ratio'
    ):
        raise ValueError('QAD parent allocation ratio changed')
    if float(allocation.get('allocation_error_ratio', math.nan)) != float(
        teacher['allocation_error_ratio']
    ):
        raise ValueError('QAD parent allocation differs from DEFAULT')
    if allocation.get('feasible') is not True:
        raise ValueError('QAD parent allocation was not feasible')
    if trial['qad_overrides'] != declared:
        raise ValueError('child QAD settings differ from its parent')
    final_overrides = validate_final_overrides(plan, trial['final_overrides'])
    budget, efficiency_proof = efficiency_budget(spec, plan)
    qad_child = copy.deepcopy(defaults['config']['qad'])
    qad_child.update(declared)
    qad_child.update(final_overrides)
    qad_child['final_cptc_cache_mb'] = budget
    document = expected_config(
        defaults,
        Path(spec['run_root']) / 'trials' / trial_id / 'model',
        Path(spec['run_root']) / 'trials' / trial_id / 'work',
        qad_child,
    )
    rc = resolve_config(recipe(document), log, check_ownership=False)
    if not rc.cfg.cptc:
        raise ValueError('this experiment requires CPTC tensors')
    if rc.target_bpw is None or rc.max_error_ratio is not None:
        raise ValueError('this experiment requires target_bpw and no error ratio')
    if (
        allocation.get('target_bpw') != rc.target_bpw
        or allocation.get('max_error_ratio') != rc.max_error_ratio
    ):
        raise ValueError('child allocation differs from its QAD parent')
    identity = run_identity(rc)
    identity['study_reuse'] = {
        'parent_manifest': parent_item,
        'factor_plan': spec['factor_plan'],
        'source_manifest': spec['source_manifest'],
        'shared_teacher': spec['shared_teacher'],
        'trial': trial['id'],
        'qad_overrides': declared,
        'final_overrides': final_overrides,
        'efficiency': {'final_cptc_cache_mb': budget},
        'compatibility_proof': proof_item,
        'cache_parity_proof': efficiency_proof,
        'worker': {'path': str(Path(__file__).resolve()), 'sha256': digest(Path(__file__).resolve())},
    }
    rc._identity = identity
    if (rc.workdir / 'run.json').exists() and read_run(rc.workdir).get(
        'state'
    ) == 'succeeded':
        raise ValueError('trial already complete; refusing to rewrite it')
    ids = require_common_identity(teacher, rc, identity)
    if qad_fingerprint_fields(parent_rc.qad) != qad_fingerprint_fields(rc.qad):
        raise ValueError('child differs from its parent on layerwise QAD settings')
    head = load_teacher_head(teacher)
    cache_preflight(teacher, rc, identity)
    tensors = load_parent_tensors(parent)
    tables = load_pinned_tables(teacher, tensors.keys())
    picks = table_picks(tables, tensors)
    lock_path = rc.workdir.parent / f'.{rc.workdir.name}.study.lock'
    with exclusive_lock(lock_path):
        if (rc.workdir / 'run.json').exists() and read_run(rc.workdir).get(
            'state'
        ) == 'succeeded':
            raise ValueError('trial already complete; refusing to rewrite it')
        with RunSession(rc.workdir, identity, [rc.dst]) as session:
            shutil.copyfile(teacher['cache_ref']['path'], rc.workdir / 'cache_ref.json')
            ex = Executor(rc.devices)
            artifact = rc.workdir / 'qad' / 'final_kl.pt'
            expected_fingerprint = expected_final_fingerprint(
                rc, parent['layerwise_fingerprint']
            )
            resumed = artifact.exists()
            model = None
            if resumed:
                rec = torch.load(artifact, weights_only=False)
                if rec.get('fingerprint') != expected_fingerprint:
                    raise ValueError('refusing a changed final-KL resume')
                if 'tensors' not in rec or 'dense' not in rec or 'info' not in rec:
                    raise ValueError('final-KL artifact is incomplete')
            else:
                model = pipeline._load_model(rc, log)
            session.phase(
                'final_kl',
                'reused' if resumed else 'running',
                parent=parent_item,
                steps=rc.qad.final_kl,
            )
            tensors, dense = pipeline._final_kl(
                rc,
                ex,
                model,
                ids,
                tensors,
                {},
                parent['layerwise_fingerprint'],
                log,
                head_inputs=head,
            )
            info = final_info(rc, resumed, artifact)
            session.phase('final_kl', 'completed')
            if set(tensors) != set(tables):
                raise ValueError('final-KL changed the quantized tensor set')
            qad_checkpoint = qad_meta(rc.qad)
            qad_checkpoint.pop('final_cptc_cache_mb', None)
            qad_checkpoint['tokens'] = ids.numel()
            if dense:
                qad_checkpoint['dense_trained'] = len(dense)
            meta = {
                'mode': 'flex',
                'target_bpw': rc.target_bpw,
                'max_error_ratio': rc.max_error_ratio,
                'allocation_error_ratio': float(
                    parent['producer']['allocation']['allocation_error_ratio']
                ),
                'calibrated': True,
                'rounding': rc.cfg.rounding,
                'sequential': rc.cfg.sequential,
                'smooth': rc.cfg.smooth,
                **(
                    {'refine_passes': rc.cfg.refine_passes}
                    if rc.cfg.refine_passes
                    else {}
                ),
                **(
                    {'smooth_search': list(rc.cfg.smooth_search)}
                    if rc.cfg.smooth_search
                    else {}
                ),
                'rotation': rc.cfg.rotation,
                'calib_tokens': rc.calib_samples * rc.seq_len,
                'qad': qad_checkpoint,
            }
            codecs = {
                k: get_codec(
                    rc.cfg.cptc_depth,
                    k,
                    rounds=rc.cfg.cptc_table_rounds,
                    samples=rc.cfg.cptc_table_samples,
                )
                for k in (2, 4)
            }
            codecs.update(cptc_codecs_from_tensors(tensors))
            write_to = (
                rc.dst
                if rc.moe_layout == 'per-expert'
                else pipeline.staging_dir(rc, rc.dst)
            )
            out = Output(
                write_to,
                picks,
                meta,
                rc.workdir / 'shards_done',
                tensors,
                dense=dense or None,
                cptc=checkpoint_config(codecs),
            )
            session.phase('encode', 'running')
            export_start = time.monotonic()
            write_checkpoints(
                rc.src, [out], rc.workdir, tables, rc.cfg, ex, print
            )
            summary = pipeline.summarize(out.report)
            atomic_json(
                Path(write_to) / 'hi2_report.json',
                {'summary': summary, 'tensors': out.report},
            )
            if Path(write_to) != rc.dst:
                restack(write_to, rc.dst, rc.moe_layout, print, overwrite=True)
                shutil.rmtree(write_to)
            export_s = time.monotonic() - export_start
            validation = validate_checkpoint(
                rc.dst, expected_tensors=list(tables), log=print
            )
            if (
                validation.get('report_checked') is not True
                or validation.get('checked') != len(tables)
                or validation.get('expected') != len(tables)
            ):
                raise ValueError('final checkpoint validation did not cover the report')
            session.phase(
                'validation',
                'completed',
                report_checked=validation['report_checked'],
                tensors=validation['checked'],
            )
            report_path = rc.dst / 'hi2_report.json'
            result = {
                'trial': trial['id'],
                'parent_manifest': parent_item,
                'parent_run': parent_run_item,
                'common_teacher': {
                    'descriptor': spec['shared_teacher'],
                    'head': teacher['head'],
                    'fingerprint': teacher['layerwise_fingerprint'],
                    'reuse': teacher['reuse'],
                },
                'source': identity['source'],
                'calibration': identity['calibration'],
                'environment': environment,
                'final_kl': info,
                'structure': validation,
                'tensor_summary': summary,
                'report': {'path': str(report_path), 'sha256': digest(report_path)},
                'quantized_tensors': len(tables),
                'cache': {
                    'ref': teacher['cache_ref']['path'],
                    'ref_sha256': teacher['cache_ref']['sha256'],
                    'stats': teacher['cache_ref']['stats']['path'],
                    'tables': teacher['cache_ref']['tables']['path'],
                },
                'efficiency': {
                    'final_cptc_cache_mb': budget,
                    'proof': efficiency_proof,
                },
                'compatibility_proof': proof_item,
                'timing': {
                    'phase_wall_s': time.monotonic() - phase_start,
                    'final_kl_s': info['seconds'],
                    'export_s': export_s,
                    'resumed_final_kl': resumed,
                },
                'scope': (
                    'Whole worker phase wall time includes parent/cache loading, '
                    'tokenization, final-KL, export and validation; final_kl_s is '
                    'the recorded training-phase seconds only. Not whole-model '
                    'quantization or inference throughput. KFD/thermal acceptance '
                    'is external.'
                ),
            }
            atomic_json(rc.workdir / 'ablation-result.json', result)
            session.complete(result)
            print(json.dumps(result, indent=2, allow_nan=False))
            return result


def execution(path):
    spec = load(path)
    if spec.get('schema_version') != SCHEMA_VERSION or spec.get(
        'status'
    ) != 'cpu_verified':
        raise ValueError('execution plan has not passed CPU admission')
    plan = load(pinned(spec['factor_plan']))
    if (
        spec['factor_plan']['sha256'] != PLAN_SHA256
        or plan.get('status') != 'factor_design_frozen_not_executed'
    ):
        raise ValueError('factor design differs from the prospectively frozen plan')
    holdout = load(pinned(spec['confirmatory_holdout']))
    if holdout.get('status') != 'data_frozen_no_quality_results':
        raise ValueError('confirmatory data was not frozen before trials')
    import hi2

    source = Path(hi2.__file__).resolve().parent.parent
    if source != Path(spec['snapshot']).resolve():
        raise ValueError('imported another quantizer source tree')
    manifest = load(pinned(spec['source_manifest']))
    rows = {}
    for row in manifest.get('files', []):
        rel = Path(row['path'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError(f'invalid source-manifest path: {row["path"]}')
        path = source / rel
        if (
            not path.is_file()
            or path.stat().st_size != row['bytes']
            or digest(path) != row['sha256']
        ):
            raise ValueError(f'quantizer source changed: {path}')
        rows[rel.as_posix()] = row
    worker = Path(__file__).resolve()
    try:
        worker_rel = worker.relative_to(source).as_posix()
    except ValueError as exc:
        raise ValueError('worker is outside the pinned source tree') from exc
    if worker_rel not in rows or digest(worker) != rows[worker_rel]['sha256']:
        raise ValueError('worker is not pinned by the source manifest')
    for name in REQUIRED_HI2_MODULES:
        importlib.import_module(name)
    for name, module in list(sys.modules.items()):
        if name != 'hi2' and not name.startswith('hi2.'):
            continue
        filename = getattr(module, '__file__', None)
        if not filename or not filename.endswith('.py'):
            continue
        path = Path(filename).resolve()
        try:
            rel = path.relative_to(source).as_posix()
        except ValueError as exc:
            raise ValueError(f'imported hi2 module outside the source tree: {path}') from exc
        if rel not in rows:
            raise ValueError(f'imported hi2 module is not in the source manifest: {rel}')
    defaults = load(pinned(spec['defaults_run']))
    if defaults.get('state') != 'succeeded':
        raise ValueError('DEFAULT reference run did not complete')
    return spec, plan, defaults


def recipe(document):
    from hi2.service import config_from_dict

    return config_from_dict({'schema_version': SCHEMA_VERSION, 'config': document})


def require_cpu_only():
    if os.environ.get('HIP_VISIBLE_DEVICES') != '':
        raise ValueError('this command is CPU-only with an empty HIP mask')


def gpu_admission():
    if os.environ.get('HIP_VISIBLE_DEVICES') != '0,1':
        raise ValueError('explicit leased HIP_VISIBLE_DEVICES=0,1 required')
    if os.environ.get('ROCR_VISIBLE_DEVICES') or os.environ.get(
        'CUDA_VISIBLE_DEVICES'
    ):
        raise ValueError('ambient ROCR/CUDA masks forbidden')
    if not os.environ.get('CPTC_FULL4B_GRANT'):
        raise ValueError('full4B supervisor grant required')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution', type=Path, required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    teacher = sub.add_parser('prepare-teacher')
    teacher.add_argument('--output', type=Path, required=True)
    manifest = sub.add_parser('make-parent')
    manifest.add_argument('--parent', required=True)
    manifest.add_argument('--output', type=Path, required=True)
    parent = sub.add_parser('qad-parent')
    parent.add_argument('--parent', required=True)
    final = sub.add_parser('final')
    final.add_argument('--trial', required=True)
    args = parser.parse_args()
    spec, plan, defaults = execution(args.execution)
    if args.command == 'prepare-teacher':
        require_cpu_only()
        prepare_teacher_core(spec, plan, defaults, args.output)
    elif args.command == 'make-parent':
        require_cpu_only()
        if args.output.exists():
            raise ValueError('refusing to overwrite a QAD parent manifest')
        record = parent_manifest_record(spec, plan, defaults, args.parent)
        from hi2.runtime import atomic_json

        atomic_json(args.output, record)
        print(json.dumps(record, indent=2))
    elif args.command == 'qad-parent':
        gpu_admission()
        print(json.dumps(run_qad_parent(spec, plan, defaults, args.parent), indent=2))
    else:
        gpu_admission()
        run_final_trial(spec, plan, defaults, args.trial)


if __name__ == '__main__':
    main()
