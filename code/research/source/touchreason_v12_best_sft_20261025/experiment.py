"""Isolated, fixed-parameter v12 SFT experiment. Does not alter the v6 run."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys

RUN = Path(__file__).resolve().parent
PARENT = Path('/mnt/workspace/lyx/touchreason_touchr1_three_stage_20261024')
SEARCH = PARENT / 'small_parameter_search_20261024'
ARCHIVE = Path('/mnt/workspace/lyx/touchreason_full_review_20261023_v1/touchreason_quick_v12_reviewed.zip')
SOURCE = RUN / 'source' / 'paper_aligned_quick_v12'
PY = '/mnt/workspace/lyx/tactile_world_model_repro/.venv/bin/python'
sys.path.insert(0, str(PARENT))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def resolved_episodes():
    episodes = read(RUN / 'episodes.json')
    path = RUN / 'verified_source_aliases.json'
    aliases = read(path) if path.exists() else {}
    for eid, episode in episodes.items():
        old = episode['source']['path']
        if old in aliases:
            alias = aliases[old]
            assert alias['episode_id'] == eid
            assert alias['verification'] == 'all original 64x64 BGR INTER_AREA SHA256 frame fingerprints'
            episode['source'] = dict(episode['source'], path=alias['path'])
            episode['verified_alias_original_path'] = old
            if episode['source']['kind'] == 'png_sequence':
                episode['verified_alias_frame_sha256'] = alias['frame_sha256']
            else:
                assert episode['source']['kind'] == 'video'
                assert alias['verified_frame_count'] == episode['source']['frame_count']
                episode['verified_alias_video_sha256'] = alias['video_sha256']
    return episodes


def emit(event, **values):
    value = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 event=event, rank=int(os.environ.get('RANK', 0)), **values)
    print(json.dumps(value, ensure_ascii=False), flush=True)
    with (RUN / f"events_rank_{value['rank']}.jsonl").open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False) + '\n')


class Dataset:
    def __init__(self):
        import fcntl
        import shutil
        ready = read(RUN / 'data_ready.json')
        local = Path('/dev/shm/lyx_v12_training_20261025.sqlite3')
        stamp = local.with_suffix('.identity.json')
        with local.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not local.exists() or not stamp.exists() or read(stamp).get('sha256') != ready['training_database_sha256']:
                temp = local.with_suffix(f'.{os.getpid()}.tmp')
                shutil.copyfile(RUN / 'train.sqlite3', temp)
                assert digest(temp) == ready['training_database_sha256']
                temp.replace(local)
                write(stamp, {'sha256': ready['training_database_sha256']})
        self.con = sqlite3.connect(f'file:{local}?mode=ro&immutable=1', uri=True)

    def indices(self):
        return [r[0] for r in self.con.execute('SELECT idx FROM examples ORDER BY idx')]

    def get(self, idx):
        return json.loads(self.con.execute('SELECT record FROM examples WHERE idx=?', (idx,)).fetchone()[0])


def engine_for(plan):
    sys.path.insert(0, str(SEARCH))
    from parameters import method_spec
    import model
    import torch
    config = plan['config']
    model.SPEC = method_spec(config)
    model.rt.SEED = config['seed']
    engine = model.Engine(vision_checkpoint=plan['stage1_checkpoint'])
    assert engine.vision_sha == plan['stage1_vision_sha256']
    # Raw media use the v12 episode inventory. Reviewed PNG media retain their
    # exported bytes; feature identity and media keys include their exact input.
    engine.episodes = resolved_episodes()
    import types
    original_images = engine.images
    checked_alias_images = set()

    def images(self, media):
        if media['kind'] == 'raw':
            ep = self.episodes[media['episode_id']]
            if 'verified_alias_frame_sha256' in ep:
                assert ep['source']['kind'] == 'png_sequence'
                for frame in media['frame_indices']:
                    name = ep['source']['filenames'][frame]
                    path = Path(ep['source']['path']) / name
                    if str(path) not in checked_alias_images:
                        assert digest(path) == ep['verified_alias_frame_sha256'][name], 'Restored image bytes changed'
                        checked_alias_images.add(str(path))
            elif 'verified_alias_video_sha256' in ep:
                path = ep['source']['path']
                if path not in checked_alias_images:
                    assert digest(path) == ep['verified_alias_video_sha256'], 'Restored video bytes changed'
                    checked_alias_images.add(path)
        return original_images(media)

    engine.images = types.MethodType(images, engine)
    engine.cache = Path('/dev/shm/lyx_v12_best_sft_features_20261025') / engine.feature_key
    engine.cache.mkdir(parents=True, exist_ok=True)
    # Reuse verified feature files from the fixed encoder when the media key
    # matches exactly. No unbounded in-memory copy of the training feature set.
    prior_caches = [PARENT / 'features' / engine.feature_key,
                    Path('/dev/shm/lyx_touchreason_small_search_features') / engine.feature_key]
    ram_cache = engine.cache
    disk_cache = RUN / 'features' / engine.feature_key
    original = engine.feature

    def feature(self, media):
        import shutil
        # Leave at least 80 GiB for model staging and other workspace jobs.
        # Existing RAM entries remain usable when new entries spill to disk.
        ram_file = ram_cache / (media['key'] + '.pt')
        if ram_file.exists() or shutil.disk_usage(ram_cache).free >= 80 * 2**30:
            self.cache = ram_cache
        else:
            disk_cache.mkdir(parents=True, exist_ok=True)
            self.cache = disk_cache
        dest = self.cache / (media['key'] + '.pt')
        if not dest.exists():
            for cache in prior_caches:
                src = cache / dest.name
                if src.exists():
                    temp = dest.with_suffix(f'.{os.getpid()}.tmp')
                    shutil.copyfile(src, temp)
                    temp.replace(dest)
                    break
        return original(media)

    engine.feature = types.MethodType(feature, engine)
    assert all(p.dtype == torch.float32 for _, p in engine.parameters)
    return engine
