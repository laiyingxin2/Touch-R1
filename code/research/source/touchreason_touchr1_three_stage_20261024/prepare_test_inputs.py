"""Allowlisted test-input export. No annotation answer is passed to a model."""
import collections
import json
from pathlib import Path

from data import ROOT, SNAPSHOT, rows, write, read, key, digest, normalize_bench


def public_input(row, index):
    media = []
    for mid in row['media_ids']:
        value = {'kind': 'public', 'public_id': mid, 'source': index[mid]}
        value['key'] = key(value)[:32]
        media.append(value)
    question = row['messages'][0]['content']
    for i in range(1, len(media)+1):
        question = question.replace('<tactile>', f'Input {i}', 1)
    return {'id': 'public/'+row['id'], 'dataset': row['dataset'], 'split': 'test', 'stage': 'evaluation',
            'task': 'public_json', 'input': {'question': question, 'options': None}, 'media': media}


def prepare():
    folder = ROOT/'test_inputs'
    folder.mkdir(exist_ok=True)
    if (folder/'manifest.json').exists():
        manifest = read(folder/'manifest.json')
        for name, spec in manifest['files'].items():
            assert digest(folder/name) == spec['sha256']
        return manifest
    index = {r['id']: r for r in rows(SNAPSHOT/'public_media_index.jsonl')}
    native = []
    for raw in rows(SNAPSHOT/'test_qa_inputs.jsonl'):
        assert not {'answer', 'gold', 'evaluation'} & set(raw)
        row = normalize_bench(raw, 'test')
        row.pop('answer')
        row.pop('target')
        native.append(row)
    datasets = {'touchreason': native}
    # Public upstream stores question/answer in the same JSONL. This exporter
    # reads that container, selects user text/media only, and never interprets
    # assistant messages/evaluation labels. Scoring reopens gold after sealing
    # the complete prediction set. This distinction is explicit in the manifest.
    public_metadata = []
    for split, name in [('test', 'public'), ('relation_coverage_test', 'public_relation')]:
        datasets[name] = []
        for raw in rows(SNAPSHOT/f'public_{split}.jsonl'):
            row = public_input(raw, index)
            datasets[name].append(row)
            public_metadata.append((row, frozenset(raw['curation']['source_objects'])))
    objects = {r['sample_id']: r for r in rows(SNAPSHOT/'test_objects_200.jsonl')}
    media = {}
    group = {}
    cooccurs = collections.defaultdict(set)
    for row in native:
        assert len(row['media']) == len(row['sample_ids'])
        for m, sid in zip(row['media'], row['sample_ids']):
            meta = objects[sid]
            media[m['key']] = m
            group[m['key']] = ('touchreason/'+meta['sensor'], frozenset([meta['group_id']]))
        for m in row['media']:
            cooccurs[m['key']].update(v['key'] for v in row['media'])
    source_groups = {}
    for row, groups in public_metadata:
        for m in row['media']:
            if m['key'] not in source_groups:
                source_groups[m['key']] = set(groups)
            else:
                source_groups[m['key']].intersection_update(groups)
            media[m['key']] = m
            cooccurs[m['key']].update(v['key'] for v in row['media'])
            group[m['key']] = ('public/'+row['dataset'], source_groups[m['key']])
    assert all(len(v) == 1 for v in source_groups.values()), 'Ambiguous public media object grouping'
    pools = collections.defaultdict(list)
    for mid in sorted(media):
        pools[group[mid][0]].append(mid)
    donors = {}
    for pool in pools.values():
        for i, mid in enumerate(pool):
            eligible = [pool[(i+j) % len(pool)] for j in range(1, len(pool))
                        if not (set(group[mid][1]) & set(group[pool[(i+j) % len(pool)]][1]))]
            assert eligible, f'No different-group, same-source control donor for {mid}'
            donors[mid] = [media[k] for k in eligible]
    within_question_fallbacks = 0
    for records in datasets.values():
        for row in records:
            same_question = {m['key'] for m in row['media']}
            for m in row['media']:
                if not any(d['key'] not in same_question for d in donors[m['key']]):
                    within_question_fallbacks += 1
    files = {}
    for name, records in datasets.items():
        path = folder/(name+'.jsonl')
        path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records))
        files[path.name] = {'sha256': digest(path), 'n': len(records)}
    write(folder/'mismatch_donors.json', donors)
    files['mismatch_donors.json'] = {'sha256': digest(folder/'mismatch_donors.json'), 'n': len(donors)}
    manifest = {'files': files, 'test_answers_passed_to_model': False,
                'public_export_reads_combined_question_answer_container': True,
                'public_export_allowlist': ['id', 'dataset', 'media_ids', 'messages[0].content'],
                'gold_scoring_after_complete_predictions': True,
                'mismatch_policy': 'same source/sensor; disjoint recorded proxy object groups; prefer donor outside question; fall back to another object in question if source pool exhausted',
                'within_question_fallback_streams': within_question_fallbacks,
                'physical_object_identity_verified': False}
    write(folder/'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
