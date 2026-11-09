"""Organize public tactile QA into portable JSONL, preserving official data."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'organized'
SOURCES = [
    ('physiclear_octopi', 'train', 'Octopi/prepared/train_qa.json'),
    ('physiclear_octopi', 'val', 'Octopi/prepared/val_opd_qa.json'),
    ('physiclear_octopi', 'test', 'Octopi/prepared/test_opd_qa.json'),
    ('physiclear_octopi', 'test', 'Octopi/prepared/test_qa.json'),
    ('physiclear_octopi', 'test', 'Octopi/prepared/psr_qa.json'),
    ('vtv_llm', 'train', 'VTVLLM/prepared/vbts_video/train_qa.json'),
    ('vtv_llm', 'val', 'VTVLLM/prepared/vbts_video/val_tfa_qa.json'),
    ('vtv_llm', 'test', 'VTVLLM/prepared/vbts_video/test_qa.json'),
]
TASKS = {
    'object_property_description': 'property_description',
    'tactile_feature_assessment': 'property_description',
    'property_comparison': 'property_comparison',
    'surface_feature_distinction': 'property_comparison',
    'property_superlative_selection': 'property_selection',
    'surface_optimality_identification': 'property_selection',
    'property_object_match': 'object_matching',
    'object_sensation_correlation': 'object_matching',
    'property_scenario_reasoning': 'scenario_reasoning',
}
WRAPPERS = ('<tact_start><img_tokens><tact_end>', '<video_start><video><video_end>')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(dumps(row) + '\n')
    temp.replace(path)


def link(target, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    relative = os.path.relpath(target, path.parent)
    if path.is_symlink():
        if os.readlink(path) == relative:
            return
        raise ValueError(f'Unexpected existing link: {path}')
    if path.exists():
        raise ValueError(f'Existing output is not a link: {path}')
    path.symlink_to(relative, target_is_directory=target.is_dir())


def organize():
    OUT.mkdir(exist_ok=True)
    link(ROOT / 'Octopi/prepared', OUT / 'media/physiclear_frames')
    link(ROOT / 'VTVLLM/extracted/vbts_video/vbts_video', OUT / 'media/vtv_videos')
    source_hashes = {p: digest(ROOT / p) for _, _, p in SOURCES}
    media = {}
    media_by_source = {}
    rows = []
    canonical_groups = defaultdict(list)
    source_summaries = []
    original_prompt = ('A chat between a curious user and an artificial intelligence assistant. '
                       "The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n")

    def register_media(source, dataset):
        if source in media_by_source:
            return media_by_source[source]
        src = Path(source)
        if dataset == 'physiclear_octopi':
            relative = src.relative_to(ROOT / 'Octopi/prepared')
            path = Path('media/physiclear_frames') / relative
            kind = 'frames'
            frames = sorted(x.name for x in src.iterdir() if x.is_file() and x.suffix.lower() in {'.jpg', '.jpeg', '.png'})
            if not frames:
                raise ValueError(f'No tactile frames: {src}')
        else:
            relative = src.relative_to(ROOT / 'VTVLLM/extracted/vbts_video/vbts_video')
            path = Path('media/vtv_videos') / relative
            kind = 'video'
            if not src.is_file() or not src.stat().st_size:
                raise ValueError(f'Invalid video: {src}')
        mid = dataset + '_' + hashlib.sha256(path.as_posix().encode()).hexdigest()[:16]
        item = {'id': mid, 'dataset': dataset, 'kind': kind, 'path': path.as_posix(),
                'source_path': src.relative_to(ROOT).as_posix()}
        if kind == 'frames':
            item.update(frame_count=len(frames), frames=frames)
        else:
            item['bytes'] = src.stat().st_size
        assert (OUT / path).exists()
        media[mid] = item
        media_by_source[source] = mid
        return mid

    for dataset, split, source_file in SOURCES:
        data = json.loads((ROOT / source_file).read_text())
        source_summaries.append({'dataset': dataset, 'split': split, 'file': source_file,
                                 'records': len(data), 'sha256': source_hashes[source_file]})
        for index, record in enumerate(data):
            qtype = record[0]['question_type']
            task = TASKS[re.sub(r'^(train|eval|test)_', '', qtype)]
            rid = f'{dataset}/{Path(source_file).stem}/{index:06d}'
            messages = []
            ordered_media = []
            for turn in record[1:]:
                content = turn['content']
                text = ''.join(content) if isinstance(content, list) else content
                # Preserve all source text, including the source's initial prompt.
                for wrapper in WRAPPERS:
                    text = text.replace(wrapper, '<tactile>')
                mids = [register_media(p, dataset) for p in turn.get('tactile', turn.get('video', []))]
                if text.count('<tactile>') != len(mids):
                    raise ValueError(f'Media placeholder mismatch: {rid}')
                for mid in mids:
                    if mid not in ordered_media:
                        ordered_media.append(mid)
                messages.append({'role': turn['role'].lower(), 'content': text, 'media_ids': mids})
            roles = [m['role'] for m in messages]
            assert roles == ['user', 'assistant'] * int(record[0]['question_steps']), rid
            assert all(m['content'].strip() for m in messages), rid
            canonical = hashlib.sha256(dumps(messages).encode()).hexdigest()
            canonical_groups[canonical].append(rid)
            rows.append({'id': rid, 'dataset': dataset, 'official_split': split, 'task': task,
                         'source_question_type': qtype, 'turn_pairs': len(messages) // 2,
                         'messages': messages, 'media_ids': ordered_media,
                         'source': {'file': source_file, 'index': index, 'qa_provenance': 'official_script_generated'},
                         'content_sha256': canonical})
        print('Organized', source_file, len(data), flush=True)

    by_dataset = defaultdict(list)
    official = defaultdict(list)
    for row in rows:
        by_dataset[(row['dataset'], row['official_split'])].append(row)
        official[row['official_split']].append(row)
    for (dataset, split), data in by_dataset.items():
        write_jsonl(OUT / 'annotations/by_dataset' / dataset / f'{split}.jsonl', data)
    for split, data in official.items():
        write_jsonl(OUT / 'annotations/official' / f'{split}.jsonl', data)
    write_jsonl(OUT / 'media_index.jsonl', [media[k] for k in sorted(media)])

    # Exact duplicate conversations with the same media are retained officially.
    # The optional recommended train file retains only the first such record.
    seen = set()
    train = []
    for row in official['train']:
        if row['content_sha256'] not in seen:
            train.append(row)
            seen.add(row['content_sha256'])
    recommended = {'train': train,
                   'val': by_dataset[('physiclear_octopi', 'val')],
                   'test': official['test']}
    write_jsonl(OUT / 'annotations/recommended/train.jsonl', train)
    link(OUT / 'annotations/by_dataset/physiclear_octopi/val.jsonl', OUT / 'annotations/recommended/val.jsonl')
    link(OUT / 'annotations/official/test.jsonl', OUT / 'annotations/recommended/test.jsonl')
    link(OUT / 'annotations/by_dataset/vtv_llm/val.jsonl', OUT / 'annotations/recommended/auxiliary_vtv_val.jsonl')

    duplicates = [{'content_sha256': h, 'record_ids': ids} for h, ids in canonical_groups.items() if len(ids) > 1]
    (OUT / 'exact_duplicates.json').write_text(json.dumps(duplicates, ensure_ascii=False, indent=2))
    with (OUT / 'records_index.csv').open('w', encoding='utf-8-sig', newline='') as f:
        fields = ['id', 'dataset', 'official_split', 'task', 'turn_pairs', 'media_count',
                  'source_file', 'source_index', 'last_question', 'last_answer']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({'id': row['id'], 'dataset': row['dataset'], 'official_split': row['official_split'],
                             'task': row['task'], 'turn_pairs': row['turn_pairs'], 'media_count': len(row['media_ids']),
                             'source_file': row['source']['file'], 'source_index': row['source']['index'],
                             'last_question': row['messages'][-2]['content'].removeprefix(original_prompt),
                             'last_answer': row['messages'][-1]['content']})

    def overlaps(group):
        pools = {split: {mid for row in data for mid in row['media_ids']} for split, data in group.items()}
        return {f'{a}/{b}': len(pools[a] & pools[b]) for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]}

    summary = {'schema_version': '1.0', 'root': '.', 'qa_records': len(rows),
               'qa_turn_pairs': sum(r['turn_pairs'] for r in rows), 'unique_media': len(media),
               'media_kinds': dict(Counter(m['kind'] for m in media.values())),
               'official_split_counts': {k: len(v) for k, v in official.items()},
               'recommended_split_counts': {k: len(v) for k, v in recommended.items()},
               'recommended_train_exact_duplicates_removed': len(official['train']) - len(train),
               'recommended_auxiliary_vtv_val_records': 75,
               'official_media_overlap': overlaps(official),
               'recommended_media_overlap': overlaps(recommended),
               'task_counts': dict(Counter(r['task'] for r in rows)),
               'dataset_counts': dict(Counter(r['dataset'] for r in rows)),
               'exact_duplicate_groups': len(duplicates),
               'semantic_cleaning_performed': False,
               'quality_audit_report': 'quality_audit/REPORT.txt',
               'sources': source_summaries,
               'limitations': ['Recommended val uses Octopi only; VTV official val shares 75 videos with test.',
                               'Media overlap is checked by media reference identity; different encodings or recordings are not equated.',
                               'Task names do not imply independently verified chain-of-thought labels.',
                               'TouchThinker video-only files are excluded from QA annotations.']}
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    assert all(digest(ROOT / p) == h for p, h in source_hashes.items()), 'Original QA files changed'
    (OUT / 'source_hashes.json').write_text(json.dumps(source_hashes, indent=2))
    print(json.dumps({k: summary[k] for k in ['qa_records', 'unique_media', 'official_split_counts',
                                            'recommended_split_counts', 'recommended_train_exact_duplicates_removed']}, indent=2), flush=True)


if __name__ == '__main__':
    organize()
