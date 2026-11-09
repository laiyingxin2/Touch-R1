"""Create a traceable, rank-consistent derivative; never relabel physical truth."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
ORG = ROOT / 'organized'
OUT = ORG / 'annotations/cleaned'
META = ORG / 'cleaning'
VERSION = 'rank_consistent_v1'


def read(path):
    return [json.loads(line) for line in path.open(encoding='utf-8')]


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(dumps(row) + '\n')
    temp.replace(path)


def constants(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_constants():
    return {'physiclear_octopi': constants('clean_octopi', 'Octopi/upstream/utils/constants.py'),
            'vtv_llm': constants('clean_vtv', 'VTVLLM/upstream/utils/constants_video.py')}


def description_pair(c, dataset, obj):
    h = c.HARDNESS_MAP[c.RANKS['hardness'][obj]]
    if dataset == 'physiclear_octopi':
        r = c.ROUGHNESS_MAP[c.RANKS['roughness'][obj]]
        t = c.TEXTURE_MAP[c.RANKS['texture'][obj]]
        structured = f'It presents a {h} and {r} surface with {t}.'
        original = c.DESCRIPTIONS[obj][0] + f' Overall, it presents a {h} and {r} surface with {t}.'
        return original, structured
    p = c.PROTRUSION_MAP[c.RANKS['protrusion'][obj]]
    e = c.ELASTICITY_MAP[c.RANKS['elasticity'][obj]]
    f = c.FRICTION_MAP[c.RANKS['friction'][obj]]
    original = f'It presents a {h} and {p} surface with {e} and {f}.'
    revised = f'The object is {h}. Its surface has {p}. It has {e} and {f}.'
    return original, revised


def friction_text(text):
    comparative = {'rougher': 'greater', 'smoother': 'lower'}
    superlative = {'roughest': 'greatest', 'smoothest': 'lowest'}
    text = re.sub(r'Is the object in the first video <tactile> (rougher|smoother) than',
                  lambda m: f'Does the object in the first video <tactile> have {comparative[m[1]]} friction than', text)
    text = re.sub(r'(Which one|which one|which object) (?:is|feels) (rougher|smoother)',
                  lambda m: f'{m[1]} has {comparative[m[2]]} friction', text)
    text = re.sub(r'select the (roughest|smoothest) one',
                  lambda m: f'select the object with the {superlative[m[1]]} friction', text)
    text = re.sub(r'Which object is the (roughest|smoothest)',
                  lambda m: f'Which object has the {superlative[m[1]]} friction', text)
    text = re.sub(r'identify the (roughest|smoothest) object',
                  lambda m: f'identify the object with the {superlative[m[1]]} friction', text)
    text = re.sub(r'The (roughest|smoothest) object is',
                  lambda m: f'The object with the {superlative[m[1]]} friction is', text)
    assert not re.search(r'\b(?:rougher|smoother|roughest|smoothest)\b', text), text
    return text


def main():
    META.mkdir(exist_ok=True)
    inputs = [*sorted((ORG / 'annotations/official').glob('*.jsonl')),
              *sorted((ORG / 'annotations/recommended').glob('*.jsonl')),
              ORG / 'media_index.jsonl', ORG / 'quality_audit/media_results.jsonl',
              ROOT / 'Octopi/upstream/utils/constants.py', ROOT / 'VTVLLM/upstream/utils/constants_video.py',
              *[ROOT / p for p in json.loads((ORG / 'source_hashes.json').read_text())]]
    protected = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    write_sources = {s: read(ORG / 'annotations/recommended' / f'{s}.jsonl')
                     for s in ['train', 'val', 'test', 'auxiliary_vtv_val']}
    c = load_constants()
    media = {r['id']: r for r in read(ORG / 'media_index.jsonl')}
    media_results = {r['id']: r for r in read(ORG / 'quality_audit/media_results.jsonl')}
    assert set(media) == set(media_results)
    assert all(not r['errors'] and not r['review_flags'] for r in media_results.values())
    def check_media_bytes(item):
        path = ORG / item['path']
        if item['kind'] == 'video':
            actual = sha(path)
        else:
            ordered = hashlib.sha256()
            for name in item['frames']:
                ordered.update(hashlib.sha256((path / name).read_bytes()).digest())
            actual = ordered.hexdigest()
        assert actual == media_results[item['id']]['encoded_content_sha256'], ('media_changed_since_decode_audit', item['id'])
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(check_media_bytes, media.values()))
    objects = {mid: Path(m['path']).stem.rsplit('_', 1)[0] for mid, m in media.items()}
    content_ids = {mid: m['kind'] + ':' + media_results[mid]['decoded_sequence_sha256'] for mid, m in media.items()}
    media_groups = defaultdict(list)
    for mid, fingerprint in content_ids.items():
        media_groups[fingerprint].append(mid)
    aliases = {}
    for group in media_groups.values():
        assert len({(media[mid]['dataset'], objects[mid]) for mid in group}) == 1, 'Equal media content has different object labels; manual review required'
        representative = min(group, key=lambda mid: media[mid]['path'])
        aliases.update({mid: representative for mid in group})
    changes = []
    removed = []
    outputs = defaultdict(list)
    counts = defaultdict(Counter)

    def correct(parent, input_split):
        row = copy.deepcopy(parent)
        ds = row['dataset']
        pairs = dict(description_pair(c[ds], ds, objects[mid]) for mid in row['media_ids'])
        rules = set()
        friction = ds == 'vtv_llm' and bool(re.search(r'\b(?:rougher|smoother|roughest|smoothest)\b', row['messages'][0]['content']))
        replacements = 0
        for i, message in enumerate(row['messages']):
            before = message['content']
            before_media = message['media_ids'][:]
            text = before
            applied = set()
            if message['role'] == 'assistant':
                for old, new in pairs.items():
                    n = text.count(old)
                    if n:
                        text = text.replace(old, new)
                        replacements += n
                        applied.add('octopi_rank_only_description' if ds == 'physiclear_octopi' else 'vtv_grammatical_rank_description')
            if friction:
                revised = friction_text(text)
                if revised != text:
                    applied.add('ask_friction_instead_of_roughness')
                text = revised
            if row['task'] == 'object_matching' and message['role'] == 'user':
                prefix, options_text = text.rsplit('1) ', 1)
                options = re.split(r', [23]\) ', options_text)
                options[-1] = options[-1].removesuffix('.')
                assert len(options) == 3 and set(options) == {c[ds].OBJECTS[objects[mid]] for mid in row['media_ids']}
                if options != sorted(options):
                    applied.add('sort_candidate_object_names')
                prefix2 = prefix.replace('objects in alphabetical order:', 'objects, listed in alphabetical order:')
                prefix2 = prefix2.replace('match each video to', 'match each video (a, b, c) to')
                if prefix2 != prefix:
                    applied.add('clarify_candidate_order_instruction')
                text = prefix2 + ', '.join(f'{k}) {name}' for k, name in enumerate(sorted(options), 1)) + '.'
            if row['task'] == 'scenario_reasoning' and message['role'] == 'user':
                old = "Use the format 'The most suitable object is x), because xxx'"
                if old in text:
                    text = text.replace(old, "Use the format 'The most suitable object is x).'")
                    applied.add('scenario_choice_only_no_invented_rationale')
            message['media_ids'] = [aliases[mid] for mid in before_media]
            if message['media_ids'] != before_media:
                applied.add('canonicalize_identical_media')
            if text != before or message['media_ids'] != before_media:
                assert applied
                changes.append({'id': row['id'], 'input_split': input_split, 'message_index': i,
                                'role': message['role'], 'rules': sorted(applied), 'before': before, 'after': text,
                                'before_media_ids': before_media, 'after_media_ids': message['media_ids'][:]})
                rules.update(applied)
            message['content'] = text
        assert replacements == len(row['media_ids']), (row['id'], replacements)
        row['media_ids'] = list(dict.fromkeys(mid for message in row['messages'] for mid in message['media_ids']))
        canonical = copy.deepcopy(row['messages'])
        for message in canonical:
            message['media_ids'] = [content_ids[mid] for mid in message['media_ids']]
        row['content_sha256'] = digest(row['messages'])
        row['curation'] = {'version': VERSION, 'parent_view': 'recommended',
                           'parent_content_sha256': parent['content_sha256'],
                           'rules': sorted(rules), 'property_label_source': 'unchanged_upstream_rank_tables',
                           'physical_ground_truth_verified': False,
                           'media_content_equivalence_sha256': digest(canonical)}
        if friction:
            row['curation']['target_property'] = 'friction'
        if row['task'] == 'scenario_reasoning':
            row['curation']['annotation_kind'] = 'choice_only_no_rationale'
        return row

    seen_train = {}
    for split, rows in write_sources.items():
        for parent in rows:
            row = correct(parent, split)
            if split == 'train':
                key = row['curation']['media_content_equivalence_sha256']
                if key in seen_train:
                    removed.append({'id': row['id'], 'input_split': split,
                                    'reason': 'identical_corrected_conversation_and_decoded_media',
                                    'keep_id': seen_train[key], 'content_equivalence_sha256': key,
                                    'parent_content_sha256': parent['content_sha256']})
                    continue
                seen_train[key] = row['id']
            output_split = 'auxiliary_scenario_choice_test' if row['task'] == 'scenario_reasoning' else split
            row['curation']['output_split'] = output_split
            outputs[output_split].append(row)
            counts[output_split].update(row['curation']['rules'])
    for split, rows in outputs.items():
        write(OUT / f'{split}.jsonl', rows)
    write(META / 'correction_log.jsonl', changes)
    write(META / 'removed_training_duplicates.jsonl', removed)
    (META / 'media_aliases.json').write_text(json.dumps({mid: representative for mid, representative in aliases.items() if mid != representative}, indent=2))
    examples = []
    for rule in ['octopi_rank_only_description', 'vtv_grammatical_rank_description', 'ask_friction_instead_of_roughness',
                 'sort_candidate_object_names', 'scenario_choice_only_no_invented_rationale']:
        examples.append({'example_rule': rule, **next(change for change in changes if rule in change['rules'])})
    (META / 'examples.json').write_text(json.dumps(examples, ensure_ascii=False, indent=2))
    summary = {'version': VERSION, 'status': 'awaiting_independent_verification',
               'input_view': 'recommended', 'input_counts': {s: len(rows) for s, rows in write_sources.items()},
               'output_counts': {s: len(rows) for s, rows in outputs.items()},
               'removed_training_duplicate_records': len(removed),
               'current_media_encoded_hashes_verified': len(media),
               'corrected_message_count': len(changes),
               'retained_record_counts_by_rule': {s: dict(count) for s, count in counts.items()},
               'dataset_counts': {s: dict(Counter(r['dataset'] for r in rows)) for s, rows in outputs.items()},
               'task_counts': {s: dict(Counter(r['task'] for r in rows)) for s, rows in outputs.items()},
               'protected_input_sha256': protected,
               'output_sha256': {str(p.relative_to(ORG)): sha(p) for p in sorted(OUT.glob('*.jsonl'))},
               'limitations': ['Physical grade labels remain unverified upstream annotations; text/rank consistency is not physical relabeling.',
                               'Cleaned evaluation prompts and answer templates differ from the official benchmark.',
                               'Validation retains only 36 Octopi description records; no independent VTV validation set is provided.',
                               'Auxiliary VTV validation overlaps test media and must not be used for model selection.',
                               'Auxiliary scenario records contain choices, not validated explanations or CoT.',
                               'Exact decoded media hashes do not detect near-duplicate recordings or insufficient sensor contact.']}
    assert all(sha(ROOT / name) == h for name, h in protected.items()), 'Protected inputs changed'
    (META / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: summary[k] for k in ['version', 'input_counts', 'output_counts', 'removed_training_duplicate_records', 'retained_record_counts_by_rule']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
