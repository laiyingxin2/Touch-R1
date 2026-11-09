"""Freeze train/validation pools without reading or adding test targets."""
import collections
import csv
import json
import random
import shutil
import sqlite3
from pathlib import Path

from common import ROOT, V12, PARENT, PUBLIC, read, write, digest, identity
from data import normalize_public, normalize_bench


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def stratified(items, n, key, seed):
    groups = collections.defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    counts = {k: 1 for k in groups}
    assert n >= len(groups)
    while sum(counts.values()) < min(n, len(items)):
        k = max((k for k in groups if counts[k] < len(groups[k])),
                key=lambda k: (n * len(groups[k])/len(items) - counts[k], str(k)))
        counts[k] += 1
    result = []
    for k in sorted(groups, key=str):
        candidates = sorted(groups[k], key=lambda r: identity([seed, r['id']]))
        result.extend(candidates[:counts[k]])
    return result


def main():
    if (ROOT / 'data_manifest.json').exists():
        print('Already frozen', ROOT / 'data_manifest.json')
        return
    data = ROOT / 'data'
    data.mkdir(exist_ok=True)
    files = {}
    for view in ['quality_v3', 'quality_v3_strict']:
        for split in ['train', 'val', 'test']:
            src = PUBLIC / 'organized/annotations' / view / (split + '.jsonl')
            # Test contents are not normalized or parsed into the training DB.
            files[str(src)] = dict(sha256=digest(src), bytes=src.stat().st_size)
            if split != 'test':
                shutil.copyfile(src, data / (view + '_' + split + '.jsonl'))
    media_path = PUBLIC / 'organized/media_index.jsonl'
    files[str(media_path)] = dict(sha256=digest(media_path), bytes=media_path.stat().st_size)
    media_index = {r['id']: r for r in rows(media_path)}
    train_public = rows(data / 'quality_v3_train.jsonl')
    val_public = rows(data / 'quality_v3_val.jsonl')
    public_train_media = {m for r in train_public for m in r['media_ids']}
    public_val_media = {m for r in val_public for m in r['media_ids']}
    assert not public_train_media & public_val_media
    # Test media membership is already fixed in the answer-free media index.
    test_media = {r['id'] for r in media_index.values() if r.get('split') == 'test'}
    assert not public_train_media & test_media
    strict_ids = {r['id'] for r in rows(data / 'quality_v3_strict_train.jsonl')}
    strict_contents = {r['content_sha256'] for r in rows(data / 'quality_v3_strict_train.jsonl')}
    val_src = PARENT / 'snapshot/touchreason_val_qa.jsonl'
    shutil.copyfile(val_src, data / 'native_validation_gold.jsonl')
    files[str(val_src)] = dict(sha256=digest(val_src), bytes=val_src.stat().st_size)
    native_val = [normalize_bench(r, 'val') for r in rows(val_src)]
    db = ROOT / 'train_val.sqlite3'
    assert not db.exists(), 'Preserve partial DB for inspection rather than overwriting'
    con = sqlite3.connect(db)
    con.execute('CREATE TABLE examples(idx INTEGER PRIMARY KEY,id TEXT UNIQUE,split TEXT,pool TEXT,record TEXT)')
    pools = collections.defaultdict(list)
    index_by_id = {}
    def insert(row, pool):
        cur = con.execute('INSERT INTO examples(id,split,pool,record) VALUES(?,?,?,?)',
              (row['id'], row['split'], pool, json.dumps(row, ensure_ascii=False)))
        pools[pool].append(cur.lastrowid)
        index_by_id[row['id']] = cur.lastrowid
        return cur.lastrowid
    strict_indices = []
    for row in train_public:
        normalized = normalize_public(row, 'train', media_index)
        idx = insert(normalized, 'public_train')
        if row['id'] in strict_ids or row['content_sha256'] in strict_contents:
            strict_indices.append(idx)
    assert len(strict_indices) == 9927
    for row in val_public:
        insert(normalize_public(row, 'val', media_index), 'public_val')
    for row in native_val:
        insert(row, 'native_val')
    source = Path('/dev/shm/lyx_v12_training_20261025.sqlite3')
    if not source.exists():
        source = V12 / 'train.sqlite3'
    parent_db = sqlite3.connect(f'file:{source}?mode=ro&immutable=1', uri=True)
    all_indices = [r[0] for r in parent_db.execute('SELECT idx FROM examples ORDER BY idx')]
    assert len(all_indices) == 240000
    replay_indices = random.Random(20261028).sample(all_indices, 8192)
    # Preserve the established source-group separation of the original train
    # records; this branch never imports native val/test rows for supervision.
    protected = {r['sample_id'] for r in rows(PARENT / 'snapshot/source_objects.jsonl')
                 if r['split'] in ['val', 'test', 'heldout_related']}
    for idx in replay_indices:
        row = json.loads(parent_db.execute('SELECT record FROM examples WHERE idx=?', (idx,)).fetchone()[0])
        assert row['split'] == 'train' and row['stage'] == 'sft'
        assert not set(row.get('sample_ids', [])) & protected
        insert(row, 'native_replay')
    parent_db.close()
    con.commit()
    assert con.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    val_rows = native_val + [normalize_public(r, 'val', media_index) for r in val_public]
    screen_native = stratified(native_val, 40, lambda r: (r['task'], r.get('attribute')), 'screen-native-20261028')
    normalized_public_val = [r for r in val_rows if r['task'] == 'public_json']
    screen_public = stratified(normalized_public_val, 24, lambda r: (r['dataset'], r['native_task']), 'screen-public-20261028')
    main_replay_n = len(pools['public_train']) // 3
    splits = dict(pools, strict_public_train=strict_indices,
                  main_epoch=pools['public_train'] + pools['native_replay'][:main_replay_n],
                  screen_validation=[index_by_id[r['id']] for r in screen_native + screen_public],
                  full_validation=pools['native_val'] + pools['public_val'])
    write(ROOT / 'splits.json', splits)
    con.close()
    runtime = Path('/dev/shm/lyx_touchr1_continual_20261028.sqlite3')
    shutil.copyfile(db, runtime)
    assert digest(runtime) == digest(db)
    manifest = dict(database=str(db), runtime_database=str(runtime), database_sha256=digest(db),
         database_bytes=db.stat().st_size, source_files=files, pool_counts={k: len(v) for k, v in pools.items()},
         main_public_fraction=len(pools['public_train'])/len(splits['main_epoch']),
         main_epoch_n=len(splits['main_epoch']), validation_n=len(splits['full_validation']),
         screen_validation_n=len(splits['screen_validation']), split_sha256=digest(ROOT / 'splits.json'),
         public_train_val_media_overlap=0, public_test_media_index_n=len(test_media),
         test_answers_in_training_database=0, protected_native_sample_ids_in_replay=0,
         original_v12_split_audit=str(V12 / 'preflight.json'),
         physical_object_identity_not_independently_verified=True, test_used_for_selection=False,
         same_test_as_official_touchr1_verified=False, v13_not_appended_as_new_independent_data=True)
    write(ROOT / 'data_manifest.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
