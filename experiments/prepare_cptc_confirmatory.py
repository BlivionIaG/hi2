#!/usr/bin/env python3
"""Freeze new confirmatory text before optimizer trials; no model execution."""
import array
import concurrent.futures
import hashlib
import json
import re
import ssl
import sys
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
import tokenizers
import truststore


def sha(data):
    return hashlib.sha256(data).hexdigest()


def packed(ids):
    values = array.array('I', ids)
    if sys.byteorder != 'little':
        values.byteswap()
    return values.tobytes()


def main():
    cfg_path = Path(sys.argv[1])
    root = Path(sys.argv[2])
    cfg_bytes = cfg_path.read_bytes()
    cfg = json.loads(cfg_bytes)
    if (root / 'manifest.json').exists():
        raise ValueError('refusing to replace a frozen holdout')
    raw = root / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

    def fetch(url, name, blob=None):
        dst = raw / name
        if dst.exists():
            data = dst.read_bytes()
        else:
            request = urllib.request.Request(url, headers={'User-Agent': 'CPTC-confirmatory-study/1'})
            with urllib.request.urlopen(request, timeout=90, context=context) as response:
                data = response.read()
            with dst.open('xb') as out:
                out.write(data)
        if blob:
            actual = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
            if actual != blob:
                raise ValueError(f'Git blob mismatch: {url}')
        return {'url': url, 'file': str(dst.relative_to(root)), 'bytes': len(data), 'sha256': sha(data), 'git_blob': blob}

    tok_record = fetch(cfg['tokenizer_url'], 'tokenizer.json')
    tok = tokenizers.Tokenizer.from_file(str(root / tok_record['file']))
    # Match AutoTokenizer(..., add_special_tokens=False), without Transformers/GPU.
    def encode(text):
        return tok.encode(text, add_special_tokens=False).ids

    exclusions = []
    for spec in cfg['exclusions']:
        data = Path(spec['path']).read_bytes()
        if sha(data) != spec['sha256']:
            raise ValueError(f'exclusion corpus changed: {spec["path"]}')
        exclusions.append((spec, packed(encode(data.decode()))))

    wiki_record = fetch(cfg['wikitext_url'], 'wikitext-test.parquet')
    rows = pq.read_table(root / wiki_record['file'], columns=['text']).column('text').to_pylist()
    wiki_docs = []
    title, lines = None, []
    for line in rows:
        if re.fullmatch(r'\s*= [^=\n]+ =\s*', line):
            if title is not None:
                wiki_docs.append((title, ''.join(lines)))
            title, lines = line.strip(), [line]
        elif title is not None:
            lines.append(line)
    if title is not None:
        wiki_docs.append((title, ''.join(lines)))
    if not wiki_docs:
        raise ValueError('no top-level Wikitext articles found')

    def doc_source(spec):
        record = fetch(spec['url'], spec['path'].replace('/', '__'), spec['git_blob'])
        return spec['path'], (root / record['file']).read_text(), record

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        docs = list(pool.map(doc_source, cfg['docs']))
    sources = [tok_record, wiki_record] + [entry[2] for entry in docs]
    domains = {'wiki_test': wiki_docs, 'python_docs': [(name, text) for name, text, _ in docs]}
    manifest = {
        'schema_version': 1, 'status': 'data_frozen_no_quality_results',
        'scope': 'Prospective confirmation for validation-selected finalists only; not a tuning set.',
        'factor_plan_sha256': cfg['plan_sha256'], 'source_config_sha256': sha(cfg_bytes),
        'model_revision': cfg['model_revision'], 'tokenizers_version': tokenizers.__version__,
        'selection_seed': cfg['selection_seed'], 'sequence_length': cfg['sequence_length'],
        'source_revisions': {'wikitext': cfg['wikitext_revision'], 'cpython': cfg['cpython_revision']},
        'sources': sources, 'exclusions': [spec for spec, _ in exclusions],
        'deduplication': 'Reject source passages containing a matching 64-token shingle at any 16-token passage offset in calibration or either old held-out corpus. Not semantic deduplication.',
        'interval_scope': 'Paired windows conditional on these corpora; windows can straddle source-document boundaries.',
        'domains': {},
    }
    n, length = cfg['windows_per_domain'], cfg['sequence_length']
    for domain, candidates in domains.items():
        ranked = sorted(candidates, key=lambda row: sha((cfg['selection_seed'] + domain + row[0]).encode()))
        selected, rejected = [], []
        for name, text in ranked:
            ids = encode(text)
            if len(ids) < length + 64:
                rejected.append({'document': name, 'reason': 'short'})
                continue
            passage = ids[:length + 64]
            overlap = None
            for start in range(0, len(passage) - 63, 16):
                needle = packed(passage[start:start + 64])
                for spec, corpus in exclusions:
                    pos = corpus.find(needle)
                    if pos >= 0 and pos % 4 == 0:
                        overlap = spec['path']
                        break
                if overlap:
                    break
            if overlap:
                rejected.append({'document': name, 'reason': '64-token overlap', 'corpus': overlap})
                continue
            selected.append({'document': name, 'text': tok.decode(passage, skip_special_tokens=False), 'source_sha256': sha(text.encode()), 'passage_ids_sha256_u32le': sha(packed(passage))})
            if len(selected) == n:
                break
        if len(selected) != n:
            raise ValueError(f'{domain}: only {len(selected)} eligible documents, need {n}')
        text = '\n\n'.join(row['text'] for row in selected)
        ids = encode(text)
        if len(ids) < n * length:
            raise ValueError(f'{domain}: insufficient re-tokenized corpus')
        corpus_path = root / f'{domain}.txt'
        with corpus_path.open('x') as out:
            out.write(text)
        windows = [ids[i * length:(i + 1) * length] for i in range(n)]
        window_hashes = [sha(b''.join(int(v).to_bytes(8, 'little', signed=True) for v in row)) for row in windows]
        manifest['domains'][domain] = {
            'file': corpus_path.name, 'sha256': sha(corpus_path.read_bytes()),
            'windows': n, 'scored_tokens': n * (length - 1),
            'window_ids_sha256_int64le': window_hashes,
            'selected_documents': [{k: v for k, v in row.items() if k != 'text'} for row in selected],
            'rejected_documents': rejected,
        }
    with (root / 'manifest.json').open('x') as out:
        json.dump(manifest, out, indent=2, allow_nan=False)
        out.write('\n')
    print(json.dumps({'manifest_sha256': sha((root / 'manifest.json').read_bytes()), 'domains': {k: {'windows': v['windows'], 'scored_tokens': v['scored_tokens'], 'sha256': v['sha256']} for k, v in manifest['domains'].items()}}, indent=2))


if __name__ == '__main__':
    main()
