"""Check uncached inputs before allocating GPU workers; never drop a row."""
from concurrent.futures import ThreadPoolExecutor
import collections
import datetime
import json
import os
from pathlib import Path

from experiment import RUN, PARENT, Dataset, read, write, resolved_episodes


def audit():
    feature_key = read(RUN / 'checkpoints/presentations_0001024/manifest.json')['feature_key'] if (RUN / 'checkpoints/presentations_0001024/manifest.json').exists() else None
    if feature_key is None:
        from data import key
        plan = read(RUN / 'plan.json')
        feature_key = key({'base': '/mnt/workspace/zfc/Qwen/Qwen2.5-VL-7B-Instruct',
            'stage1_vision_sha256': plan['stage1_vision_sha256'], 'image_side': 448,
            'letterbox': 'gray127', 'pooled_tokens': 64, 'version': 2,
            'vision_attention': 'equivalent block-diagonal segmented SDPA'})[:24]
    roots = [PARENT / 'features' / feature_key,
             Path('/dev/shm/lyx_touchreason_small_search_features') / feature_key,
             Path('/dev/shm/lyx_v12_best_sft_features_20261025') / feature_key,
             RUN / 'features' / feature_key]
    cached = set()
    for root in roots:
        if root.exists():
            cached.update(x.name[:-3] for x in os.scandir(root) if x.name.endswith('.pt'))
    episodes = resolved_episodes()
    media, row_media = {}, []
    dataset = Dataset()
    for (value,) in dataset.con.execute('SELECT record FROM examples'):
        row = json.loads(value)
        keys = []
        for item in row['media']:
            if item['key'] in cached:
                continue
            keys.append(item['key'])
            if item['key'] in media:
                continue
            if item['kind'] == 'images':
                paths = item['paths']
            else:
                assert item['kind'] == 'raw'
                source = episodes[item['episode_id']]['source']
                if source['kind'] == 'png_sequence':
                    paths = [str(Path(source['path']) / source['filenames'][i]) for i in item['frame_indices']]
                else:
                    paths = [source['path']]
            media[item['key']] = paths
        row_media.append(keys)
    paths = sorted({path for values in media.values() for path in values})
    with ThreadPoolExecutor(max_workers=24) as pool:
        missing = {p for p, exists in zip(paths, pool.map(lambda p: Path(p).is_file(), paths)) if not exists}
    unavailable = {key: [p for p in paths if p in missing] for key, paths in media.items() if any(p in missing for p in paths)}
    result = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'ready': not unavailable, 'feature_key': feature_key,
              'training_rows': len(row_media), 'uncached_unavailable_media': len(unavailable),
              'affected_training_rows': sum(any(k in unavailable for k in keys) for keys in row_media),
              'missing_paths': sorted(missing), 'unavailable_media': unavailable,
              'training_rows_skipped': False, 'labels_changed': False, 'test_gold_opened': False}
    write(RUN / 'runtime_media_status.json', result)
    return result
