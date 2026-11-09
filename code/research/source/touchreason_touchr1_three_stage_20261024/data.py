"""Immutable Touch-Reason v6 + public tactile QA inputs for three real stages.

Only train/validation answers enter the training database. Test inputs and gold
remain separate frozen files. Paths and source metadata never enter model text.
"""
import collections
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sqlite3

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/mnt/workspace/lyx/touchreason_full_review_20261023_v1/paper_aligned_quick_v6')
SOURCE_ROOT = SOURCE.parent
PUBLIC = Path('/mnt/workspace/lyx/public_tactile_qa_20261022')
BENCH = Path('/mnt/workspace/lyx/touchreason_bench4800_v2_20261023/data')
SNAPSHOT = ROOT/'snapshot'
DB = ROOT/'data.sqlite3'
SEED = 20261024


def rows(path):
    path = Path(path)
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def target(answer, perceive=None, compare=None):
    # Approved annotations are retained; no teacher rationale is invented for
    # weak labels. Empty scaffolds deliberately contain no fabricated evidence.
    def text(value):
        if value is None:
            return ''
        value = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
        return value.replace('<', '&lt;').replace('>', '&gt;')
    return f'<perceive>{text(perceive)}</perceive><compare>{text(compare)}</compare><conclude>{answer}</conclude>'


def media_record(loc, clips):
    loc = dict(loc)
    if loc.get('image_paths'):
        paths = [str(Path(p) if Path(p).is_absolute() else SOURCE/p) for p in loc['image_paths']]
        # Exported reviewed image hashes are checked when first decoded.
        m = {'kind': 'images', 'paths': paths, 'sha256': loc.get('image_sha256'),
             'representation': loc.get('representation', 'recorded tactile evidence')}
    else:
        clip = clips.get(loc.get('evidence_id'), {})
        eid = loc.get('episode_id') or clip.get('episode_id')
        indices = loc.get('frame_indices') or clip.get('frame_indices')
        assert eid and indices, loc
        m = {'kind': 'raw', 'episode_id': eid, 'frame_indices': indices,
             'representation': loc.get('representation', 'recorded tactile frames')}
    m['key'] = key(m)[:32]
    return m


def normalize_touchreason(row, clips, loader):
    prepared = loader.prepare(row, allow_unverified_source=True, release_directory=SOURCE)
    public = json.loads(prepared['model_text'])
    assert '/mnt/' not in prepared['model_text'] and '/tmp/' not in prepared['model_text']
    rationale = row.get('rationale_target') if row.get('machine_consensus_candidate') else None
    answer = str(row['candidate_answer'])
    if rationale:
        # Use the loader's own evidence-target checks before retaining a rationale.
        loader.prepare(row, allow_unverified_source=True, release_directory=SOURCE, target_mode='answer_with_evidence')
    return {'id': 'touchreason/'+row['id'], 'dataset': 'touchreason_v6', 'split': 'train', 'stage': row['stage'],
            'task': row['task'], 'attribute': row.get('attribute'), 'input': public,
            'media': [media_record(loc, clips) for loc in row['evidence_locators']],
            'answer': answer, 'target': target(answer, rationale.get('perceive') if rationale else None,
                                               rationale.get('compare') if rationale else None),
            'label_tier': row.get('label_tier'), 'physical_reference_verified': row.get('physical_reference_verified', False),
            'physical_reward_enabled': False, 'annotation_reward_kind': 'weak_reference_agreement',
            'episode_ids': row.get('episode_ids', []), 'paired_physical_identity_verified': False}


def normalize_bench(row, split):
    media = []
    for paths in row['images']:
        m = {'kind': 'images', 'paths': paths, 'representation': 'recorded tactile modality panels'}
        m['key'] = key(m)[:32]
        media.append(m)
    answer = row.get('answer')
    return {'id': 'touchreason_bench/'+row['qa_id'], 'native_id': row['qa_id'],
            'dataset': 'touchreason_bench', 'split': split, 'stage': 'evaluation',
            'task': row['task'], 'attribute': row.get('attribute'),
            'input': {'question': row['question'], 'options': row.get('options'),
                      'representation': 'Grid panel titles identify recorded tactile modalities; missing panels are blank.'},
            'media': media, 'answer': answer,
            'target': target(str(answer)) if answer is not None else None,
            'sample_ids': row['sample_ids'], 'paired_physical_identity_verified': False}


def normalize_public(row, split, media_index):
    media = []
    for mid in row['media_ids']:
        m = {'kind': 'public', 'public_id': mid, 'source': media_index[mid]}
        m['key'] = key(m)[:32]
        media.append(m)
    question = row['messages'][-2]['content']
    for i in range(1, len(media)+1):
        question = question.replace(f'<tactile {i}>', f'Input {i}')
    answer = json.loads(row['messages'][-1]['content'])
    return {'id': 'public/'+row['id'], 'dataset': row['dataset'], 'split': split, 'stage': 'sft' if split == 'train' else 'evaluation',
            'task': 'public_json', 'native_task': row['task'], 'attribute': row['evaluation'].get('property'),
            'input': {'question': question, 'options': None}, 'media': media,
            'answer': row['messages'][-1]['content'], 'target': target(row['messages'][-1]['content'], answer.get('evidence')),
            'label_tier': 'public_categorical_annotation', 'public_row': row,
            'paired_physical_identity_verified': False}


def freeze():
    SNAPSHOT.mkdir(exist_ok=True)
    manifest = read(SOURCE/'manifest.json')
    assert read(SOURCE/'validation.json')['structural_validation_passed']
    files = {}
    names = [spec['path'] for spec in manifest['quota_files'].values()]
    names += ['manifest.json', 'validation.json', 'quick_candidate_loader.py', 'episodes.jsonl',
              'clips.locators.jsonl.gz', 'source_objects.jsonl', 'test_qa_inputs.jsonl', 'input_quality_checks.json']
    for name in dict.fromkeys(names):
        src, dst = SOURCE/name, SNAPSHOT/name
        expected = digest(src)
        if not dst.exists():
            shutil.copy2(src, dst)
        assert digest(dst) == expected, 'Frozen source changed: '+name
        files[name] = {'sha256': expected, 'bytes': dst.stat().st_size}
    for spec in manifest['quota_files'].values():
        assert files[spec['path']]['sha256'] == spec['sha256']
    # Keep test inputs/gold separate, with their original bytes. No normalization
    # of test labels occurs during dataset construction.
    for source, name in ((BENCH/'val_qa.jsonl', 'touchreason_val_qa.jsonl'),
                         (PUBLIC/'organized/media_index.jsonl', 'public_media_index.jsonl')):
        dst = SNAPSHOT/name
        if not dst.exists():
            shutil.copy2(source, dst)
        assert digest(dst) == digest(source)
        files[name] = {'sha256': digest(dst), 'bytes': dst.stat().st_size}
    for split in ('train', 'val', 'test', 'relation_coverage_test'):
        src = PUBLIC/f'organized/annotations/quality_v3/{split}.jsonl'
        dst = SNAPSHOT/f'public_{split}.jsonl'
        if not dst.exists():
            shutil.copy2(src, dst)
        assert digest(dst) == digest(src)
        files[dst.name] = {'sha256': digest(dst), 'bytes': dst.stat().st_size}
    source_eps = Path('/mnt/workspace/lyx/touchreason_frame_annotation_20261023/episodes.jsonl')
    dst = SNAPSHOT/'all_source_episodes.jsonl'
    if not dst.exists():
        shutil.copy2(source_eps, dst)
    assert digest(dst) == digest(source_eps)
    files[dst.name] = {'sha256': digest(dst), 'bytes': dst.stat().st_size}
    frozen = {'version': 'touchreason_v6_plus_public_v3', 'source_directory': str(SOURCE), 'files': files,
              'latest_completed_at_freeze': 'paper_aligned_quick_v6', 'pending_v7_not_used': True,
              'weak_supervision_explicit': True, 'quality_ready': False,
              'test_gold_not_used_for_training': True}
    if (ROOT/'frozen_data.json').exists():
        assert read(ROOT/'frozen_data.json') == frozen
    else:
        write(ROOT/'frozen_data.json', frozen)
    return frozen


def prepare():
    frozen = freeze()
    if (ROOT/'data_complete.json').exists():
        return read(ROOT/'data_complete.json')
    assert not DB.exists(), 'Preserve incomplete database and inspect before rebuilding'
    loader_spec = importlib.util.spec_from_file_location('frozen_candidate_loader', SNAPSHOT/'quick_candidate_loader.py')
    loader = importlib.util.module_from_spec(loader_spec)
    loader_spec.loader.exec_module(loader)
    clips = {r['clip_id']: r for r in rows(SNAPSHOT/'clips.locators.jsonl.gz')}
    episodes = {r['episode_id']: r for r in rows(SNAPSHOT/'episodes.jsonl')}
    objects = {r['sample_id']: r for r in rows(SNAPSHOT/'source_objects.jsonl')}
    protected = {r['group_id'] for r in objects.values() if r['split'] in ('val', 'test', 'heldout_related')}
    stage_eps = collections.defaultdict(set)
    frame_indices = collections.defaultdict(list)
    for r in rows(SNAPSHOT/'pretrain_frames.1000000.candidates.jsonl.gz'):
        ep = episodes[r['episode_id']]
        assert ep['stage'] == 'pretrain' and ep['split'] == 'train' and ep['group_id'] not in protected
        frame_indices[r['episode_id']].append(r['frame_index'])
        stage_eps['pretrain'].add(r['episode_id'])
    windows = []
    for eid, indices in sorted(frame_indices.items()):
        indices.sort()
        assert len(indices) == len(set(indices))
        if len(indices) < 2:
            continue
        targets = list(range(8, len(indices), 8))
        if not targets or targets[-1] != len(indices)-1:
            targets.append(len(indices)-1)
        for t in targets:
            windows.append({'episode_id': eid, 'context_indices': indices[max(0, t-8):t], 'target_index': indices[t]})
    write(ROOT/'stage1_windows.json', windows)
    del windows, frame_indices
    val_eps = [ep for ep in rows(SNAPSHOT/'all_source_episodes.jsonl') if ep.get('split') == 'val'
               and ep['source'].get('frame_count', len(ep['source'].get('filenames', []))) >= 9]
    by_sensor = collections.defaultdict(list)
    for ep in val_eps:
        by_sensor[ep['sensor']].append(ep)
    val_windows = []
    for sensor, eps in sorted(by_sensor.items()):
        for ep in sorted(eps, key=lambda ep: ep['episode_id'])[:4]:
            episodes[ep['episode_id']] = ep
            val_windows.append({'episode_id': ep['episode_id'], 'context_indices': list(range(8)), 'target_index': 8})
    assert val_windows, 'Need genuine heldout validation episodes for dynamics pretraining'
    write(ROOT/'stage1_val_windows.json', val_windows)
    write(ROOT/'episodes.json', episodes)
    con = sqlite3.connect(DB)
    con.execute('PRAGMA journal_mode=WAL')
    con.execute('CREATE TABLE examples (idx INTEGER PRIMARY KEY, id TEXT UNIQUE, stage TEXT, split TEXT, dataset TEXT, record TEXT)')
    counts, excluded = collections.Counter(), []

    def insert(r):
        con.execute('INSERT INTO examples(id,stage,split,dataset,record) VALUES(?,?,?,?,?)',
                    (r['id'], r['stage'], r['split'], r['dataset'], json.dumps(r, ensure_ascii=False)))
        counts[(r['split'], r['stage'], r['dataset'])] += 1

    for stage in ('sft', 'grpo'):
        for i, r in enumerate(rows(SNAPSHOT/f'{stage}_qa.candidates.jsonl.gz')):
            assert r['stage'] == stage and r['split'] == 'train'
            for eid in r['episode_ids']:
                assert episodes[eid]['split'] == 'train' and episodes[eid]['group_id'] not in protected
                stage_eps[stage].add(eid)
            try:
                insert(normalize_touchreason(r, clips, loader))
            except ValueError as exc:
                if str(exc) != 'unsupported_causal_rationale_requires_correction':
                    raise
                excluded.append({'id': r['id'], 'stage': stage, 'reason': str(exc)})
            if i % 2000 == 0:
                con.commit()
    for a, b in (('pretrain', 'sft'), ('pretrain', 'grpo'), ('sft', 'grpo')):
        assert not stage_eps[a] & stage_eps[b], 'Training-stage episode leak'
    for row in rows(SNAPSHOT/'touchreason_val_qa.jsonl'):
        assert row['split'] == 'val'
        insert(normalize_bench(row, 'val'))
    media_index = {r['id']: r for r in rows(SNAPSHOT/'public_media_index.jsonl')}
    media_by_split = {}
    for split in ('train', 'val'):
        media_by_split[split] = set()
        for row in rows(SNAPSHOT/f'public_{split}.jsonl'):
            insert(normalize_public(row, split, media_index))
            media_by_split[split].update(row['media_ids'])
    assert not media_by_split['train'] & media_by_split['val']
    con.execute('CREATE INDEX stage_split ON examples(stage,split)')
    con.commit()
    con.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    assert con.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    con.close()
    result = {'counts': {'/'.join(k): v for k, v in counts.items()}, 'excluded': excluded,
              'pretrain_recorded_frames': 1000000,
              'stage1_future_windows': len(read(ROOT/'stage1_windows.json')),
              'stage1_validation_windows': len(val_windows),
              'stage_episode_overlap': 0, 'protected_object_groups_in_training': 0,
              'public_train_val_media_overlap': 0, 'test_rows_in_training_database': 0,
              'frozen_manifest_sha256': digest(ROOT/'frozen_data.json'),
              'label_policy': 'Source hypotheses remain weak. Only approved rationales retained. Never claim calibrated physical gold.',
              'test_gold_opened_for_training': False}
    write(ROOT/'data_complete.json', result)
    return result


class Dataset:
    def __init__(self):
        self.con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)

    def indices(self, stage=None, split='train'):
        if stage:
            return [r[0] for r in self.con.execute('SELECT idx FROM examples WHERE stage=? AND split=? ORDER BY idx', (stage, split))]
        return [r[0] for r in self.con.execute('SELECT idx FROM examples WHERE split=? ORDER BY idx', (split,))]

    def get(self, idx):
        return json.loads(self.con.execute('SELECT record FROM examples WHERE idx=?', (idx,)).fetchone()[0])


if __name__ == '__main__':
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
