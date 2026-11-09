"""Pin the best completed two-seed recipe; normalize only v12 SFT inputs."""
import collections
from concurrent.futures import ThreadPoolExecutor
import fcntl
import gzip
import importlib.util
import json
from pathlib import Path
import shutil
import sqlite3
import stat
import zipfile

from experiment import RUN, PARENT, SEARCH, ARCHIVE, SOURCE, read, write, digest, identity, emit


def lines(path):
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else Path(path).open()) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def freeze_plan():
    if (RUN / 'plan.json').exists():
        return read(RUN / 'plan.json')
    grouped = collections.defaultdict(list)
    for path in SEARCH.glob('trials/*/confirm/result.json'):
        result = read(path)
        assert result['real']['n'] == 192
        config = result['config']
        grouped[config.get('replicate_of', config['id'])].append((path, result))
    completed = []
    for name, pair in grouped.items():
        if len(pair) != 2:
            continue
        base = next(((p, v) for p, v in pair if v['config']['id'] == name), None)
        if base is None or not any(v['config'].get('replicate_of') == name for _, v in pair):
            continue
        native = sum(v['real']['exact_accuracy_per_dataset']['touchreason_bench'] for _, v in pair) / 2
        macro = sum(v['real']['macro_exact_accuracy'] for _, v in pair) / 2
        completed.append((native, macro, name, base, pair))
    assert completed, 'No completed two-seed confirmation'
    native, macro, name, (_, best), pair = max(completed, key=lambda x: (x[0], x[1]))
    checkpoint = read(SEARCH / 'plan.json')['common_stage1_checkpoint']
    plan = {
        'experiment': 'v12_best_confirmed_sft', 'archive': str(ARCHIVE),
        'dataset': 'paper_aligned_quick_v12', 'train_source': 'sft_qa.candidates.jsonl.gz',
        'expected_training_rows': 240000, 'expected_test_rows': 4800,
        'public_training_data_added': False, 'config': best['config'],
        'parameter_selection': 'highest completed two-seed TouchReason validation mean, macro as tie breaker',
        'parameter_selection_frozen_before_v12_test': True,
        'selected_recipe_id': name, 'source_two_seed_native_mean': native,
        'source_two_seed_macro_mean': macro,
        'parameter_source_reports': [{'path': str(p), 'sha256': digest(p)} for p, _ in pair],
        'epochs': 2, 'training_presentations': 480000,
        'epoch_policy': 'Preserve the two passes used in the 1024-unique / 2048-presentation confirmation.',
        'schedule_horizon': 480000,
        'schedule_policy': 'Scale cosine and 3% warmup to the complete fixed training budget; do not decay to zero after 2048 presentations.',
        'model_selection': 'fixed final checkpoint after two epochs; no test or validation checkpoint selection',
        'language_initialization': 'fresh LoRA from base model; do not resume v6 language checkpoints',
        'stage1_checkpoint': checkpoint,
        'stage1_vision_sha256': digest(Path(checkpoint) / 'vision_adapter.pt'),
        'stage1_retrained_on_v12': False, 'stage1_fixed': True,
        'sft_objective': 'assistant-only cross entropy normalized by total target tokens in each global batch',
        'loss_weights_changed': False, 'grpo_enabled': False,
        'test_controls': ['real', 'blank', 'mismatch'],
        'test_gold_opened_only_after_all_predictions_sealed': True,
        'weak_supervision_experiment': True, 'official_paper_reproduction': False,
        'new_user_request_authorizes_this_v12_run_only': True,
        'old_v6_full_training_hold_unchanged': True,
    }
    write(RUN / 'plan.json', plan)
    emit('plan_frozen', recipe=name, native_mean=native, config=plan['config'])
    return plan


def extract():
    marker = RUN / 'archive_verified.json'
    if marker.exists() and (SOURCE / 'sft_qa.candidates.jsonl.gz').exists():
        return read(marker)
    dest = RUN / 'source'
    dest.mkdir(exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as z:
        for item in z.infolist():
            path = Path(item.filename)
            assert not path.is_absolute() and '..' not in path.parts
            assert not stat.S_ISLNK(item.external_attr >> 16)
            target = dest / path
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size == item.file_size:
                # Never trust size alone when resuming extraction.
                import zlib
                crc = 0
                with target.open('rb') as f:
                    for chunk in iter(lambda: f.read(2**20), b''):
                        crc = zlib.crc32(chunk, crc)
                if crc & 0xffffffff == item.CRC:
                    continue
            temp = target.with_suffix(target.suffix + '.extract.tmp')
            with z.open(item) as src, temp.open('wb') as out:
                shutil.copyfileobj(src, out, 2**20)
            temp.replace(target)
    spec = read(SOURCE / 'manifest.json')
    assert read(SOURCE / 'validation.json')['structural_validation_passed']
    verified = {}
    for name in ['sft_qa.candidates.jsonl.gz', 'test_qa.jsonl', 'test_qa_inputs.jsonl', 'test_objects_200.jsonl']:
        expected = next((v['sha256'] for v in spec['quota_files'].values() if v['path'] == name), None)
        expected = expected or spec['frozen_hashes'][name]
        actual = digest(SOURCE / name)
        assert actual == expected, name
        verified[name] = actual
    value = {'archive': str(ARCHIVE), 'archive_sha256': digest(ARCHIVE), 'files': verified,
             'archive_crc_verified_during_extraction': True,
             'test_gold_parsed': False, 'weak_source_opt_in': True}
    write(marker, value)
    return value


def prepare():
    plan = freeze_plan()
    verified = extract()
    if (RUN / 'data_ready.json').exists():
        return read(RUN / 'data_ready.json')
    import data as original
    original.SOURCE = SOURCE
    spec = importlib.util.spec_from_file_location('v12_source_loader', SOURCE / 'quick_candidate_loader.py')
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    episodes = {r['episode_id']: r for r in lines(SOURCE / 'episodes.jsonl')}
    clips = {r['clip_id']: r for r in lines(SOURCE / 'clips.locators.jsonl.gz')}
    objects = {r['sample_id']: r for r in lines(SOURCE / 'source_objects.jsonl')}
    test_objects = {r['sample_id']: r for r in lines(SOURCE / 'test_objects_200.jsonl')}
    protected = {r['group_id'] for r in objects.values() if r['split'] in ['test', 'val', 'heldout_related']}
    protected.update(r['group_id'] for r in test_objects.values())
    test_ids = set(test_objects)
    # Separate answer-free test input export; do not parse test_qa.jsonl here.
    inputs, paths_to_check = [], set()
    for raw in lines(SOURCE / 'test_qa_inputs.jsonl'):
        assert not {'answer', 'gold', 'evaluation'} & set(raw)
        row = original.normalize_bench(raw, 'test')
        row.pop('answer'); row.pop('target')
        assert set(row['sample_ids']) <= test_ids
        inputs.append(row)
        for media in row['media']:
            paths_to_check.update(media['paths'])
    assert len(inputs) == plan['expected_test_rows']
    assert len({r['id'] for r in inputs}) == len(inputs)
    for path in paths_to_check:
        assert Path(path).is_file(), path
    folder = RUN / 'test_inputs'
    folder.mkdir(exist_ok=True)
    with (folder / 'touchreason.jsonl').open('w') as f:
        for row in inputs:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    donors, media_group = {}, {}
    for row in inputs:
        for m, sid in zip(row['media'], row['sample_ids']):
            obj = test_objects[sid]
            media_group[m['key']] = (m, obj['sensor'], obj['group_id'])
    for mid, (_, sensor, group) in sorted(media_group.items()):
        candidates = [m for k, (m, s, g) in sorted(media_group.items()) if k != mid and s == sensor and g != group]
        assert candidates, 'No disjoint-group same-sensor mismatch donor'
        donors[mid] = candidates
    write(folder / 'mismatch_donors.json', donors)
    write(folder / 'manifest.json', {'n': len(inputs), 'sha256': digest(folder / 'touchreason.jsonl'),
          'gold_free': True, 'native_source_hash': verified['files']['test_qa_inputs.jsonl'],
          'mismatch_donors_sha256': digest(folder / 'mismatch_donors.json')})
    temp_db = Path('/dev/shm/lyx_v12_train_preparing_20261025.sqlite3')
    if temp_db.exists():
        temp_db.unlink()
    con = sqlite3.connect(temp_db)
    con.execute('CREATE TABLE examples(idx INTEGER PRIMARY KEY, id TEXT UNIQUE, record TEXT NOT NULL)')
    counts, tiers, train_eps, media_keys = collections.Counter(), collections.Counter(), set(), set()
    raw_sources = set()
    reviewed_hashes_checked = set()
    for n, raw in enumerate(lines(SOURCE / plan['train_source']), 1):
        assert raw['stage'] == 'sft' and raw['split'] == 'train'
        for eid in raw['episode_ids']:
            ep = episodes[eid]
            assert ep['split'] == 'train' and ep['group_id'] not in protected
            assert ep['sample_id'] not in test_ids
            train_eps.add(eid)
        row = original.normalize_touchreason(raw, clips, loader)
        row['dataset'] = 'touchreason_v12'
        for media in row['media']:
            media_keys.add(media['key'])
            if media['kind'] == 'images':
                for i, name in enumerate(media['paths']):
                    assert Path(name).is_file(), name
                    expected = media.get('sha256')
                    if expected and name not in reviewed_hashes_checked:
                        assert digest(name) == expected[i], name
                        reviewed_hashes_checked.add(name)
            else:
                assert media['episode_id'] in raw['episode_ids']
                raw_sources.add(episodes[media['episode_id']]['source']['path'])
        con.execute('INSERT INTO examples(id,record) VALUES(?,?)', (row['id'], json.dumps(row, ensure_ascii=False)))
        counts[row['task']] += 1
        tiers[str(row.get('label_tier'))] += 1
        if n % 10000 == 0:
            con.commit(); emit('data_preparation_progress', n=n)
    emit('checking_raw_media_paths', unique_paths=len(raw_sources))
    def check_path(path):
        assert Path(path).exists(), path
    with ThreadPoolExecutor(max_workers=32) as pool:
        list(pool.map(check_path, raw_sources))
    emit('raw_media_paths_checked', unique_paths=len(raw_sources))
    assert n == plan['expected_training_rows']
    con.commit()
    assert con.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    con.close()
    shutil.copyfile(temp_db, RUN / 'train.sqlite3.preparing')
    (RUN / 'train.sqlite3.preparing').replace(RUN / 'train.sqlite3')
    temp_db.unlink()
    write(RUN / 'episodes.json', episodes)
    # Stage 1 was trained using v6. Verify its actual source episodes against
    # the v12 test proxy groups, rather than relying on a version label.
    old_episodes = read(PARENT / 'episodes.json')
    stage1_ids = {r['episode_id'] for r in read(PARENT / 'stage1_windows.json')}
    assert all(old_episodes[e]['group_id'] not in protected and old_episodes[e]['sample_id'] not in test_ids
               for e in stage1_ids)
    stage1_sft_overlap = len(stage1_ids & train_eps)
    result = {'status': 'ready', 'train_rows': n, 'test_rows': len(inputs), 'task_counts': dict(counts),
              'label_tiers': dict(tiers), 'train_episodes': len(train_eps), 'unique_media': len(media_keys),
              'training_database_sha256': digest(RUN / 'train.sqlite3'),
              'plan_sha256': digest(RUN / 'plan.json'),
              'test_gold_parsed': False, 'protected_test_group_overlap': 0,
              'stage1_test_group_overlap': 0, 'reviewed_image_hashes_checked': len(reviewed_hashes_checked),
              'inherited_stage1_vs_v12_sft_episode_overlap': stage1_sft_overlap,
              'no_training_rows_filtered': True, 'weak_supervision_experiment': True}
    write(RUN / 'data_ready.json', result)
    emit('data_ready', **result)
    return result


if __name__ == '__main__':
    try:
        lock = (RUN / 'prepare.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare()
    except BaseException as exc:
        write(RUN / 'prepare_failure.json', {'type': type(exc).__name__, 'error': str(exc)})
        raise
