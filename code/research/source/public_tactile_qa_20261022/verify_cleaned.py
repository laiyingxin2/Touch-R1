"""Independently verify curated coverage, media identity and rank-based targets."""
from collections import Counter
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
ORG = ROOT / 'organized'
DATA = ORG / 'annotations/cleaned'
META = ORG / 'cleaning'


def read(path):
    return [json.loads(line) for line in path.open(encoding='utf-8')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def property_query(q, dataset, selection):
    rough = 'roughness' if dataset == 'physiclear_octopi' else 'friction'
    patterns = [('hardness', 1, r'\bhardest\b' if selection else r'\bharder\b'),
                ('hardness', -1, r'\bsoftest\b' if selection else r'\bsofter\b'),
                (rough, 1, r'\broughest\b|\bgreatest friction\b' if selection else r'\brougher\b|\bgreater friction\b'),
                (rough, -1, r'\bsmoothest\b|\blowest friction\b' if selection else r'\bsmoother\b|\blower friction\b'),
                ('texture', 1, r'\bbiggest bumps\b' if selection else r'\bbigger bumps\b'),
                ('texture', -1, r'\bsmallest bumps\b' if selection else r'\bsmaller bumps\b'),
                ('elasticity', 1, r'\bmost elastic\b' if selection else r'\bmore elastic\b'),
                ('elasticity', -1, r'\bleast elastic\b' if selection else r'\bless elastic\b'),
                ('protrusion', 1, r'\bmost protruded\b' if selection else r'\bmore protruded\b'),
                ('protrusion', -1, r'\bleast protruded\b' if selection else r'\bless protruded\b')]
    matches = [(p, d) for p, d, pattern in patterns if re.search(pattern, q)]
    assert len(matches) == 1, (q, matches)
    return matches[0]


def main():
    summary = json.loads((META / 'summary.json').read_text())
    for name, fingerprint in summary['protected_input_sha256'].items():
        assert sha(ROOT / name) == fingerprint, ('protected_input_changed', name)
    for name, fingerprint in summary['output_sha256'].items():
        assert sha(ORG / name) == fingerprint, ('output_changed', name)
    constants = {'physiclear_octopi': load('verify_octopi', 'Octopi/upstream/utils/constants.py'),
                 'vtv_llm': load('verify_vtv', 'VTVLLM/upstream/utils/constants_video.py')}
    parents = {r['id']: r for s in summary['input_counts'] for r in read(ORG / 'annotations/recommended' / f'{s}.jsonl')}
    outputs = {s: read(DATA / f'{s}.jsonl') for s in summary['output_counts']}
    rows = [r for records in outputs.values() for r in records]
    removed = read(META / 'removed_training_duplicates.jsonl')
    assert len({r['id'] for r in rows}) == len(rows)
    assert len({r['id'] for r in removed}) == len(removed)
    assert {r['id'] for r in rows}.isdisjoint({r['id'] for r in removed})
    assert {r['id'] for r in rows + removed} == set(parents)
    assert {r['id'] for r in outputs['train']} | {r['id'] for r in removed} == {r['id'] for r in parents.values() if r['official_split'] == 'train'}
    media = {r['id']: r for r in read(ORG / 'media_index.jsonl')}
    mr = {r['id']: r for r in read(ORG / 'quality_audit/media_results.jsonl')}
    obj = {mid: Path(m['path']).stem.rsplit('_', 1)[0] for mid, m in media.items()}
    hashes = {mid: m['kind'] + ':' + mr[mid]['decoded_sequence_sha256'] for mid, m in media.items()}
    aliases = json.loads((META / 'media_aliases.json').read_text())
    for old, new in aliases.items():
        assert hashes[old] == hashes[new] and obj[old] == obj[new]
    logs = {(r['id'], r['message_index']): r for r in read(META / 'correction_log.jsonl')}
    counts = Counter()
    seen_train = set()
    removed_by_id = {r['id']: r for r in removed}
    kept_by_id = {r['id']: r for r in rows}
    for row in rows:
        parent = parents[row['id']]
        c = constants[row['dataset']]
        dataset = row['dataset']
        objects = [obj[mid] for mid in row['media_ids']]
        assert row['source'] == parent['source']
        for key in ['dataset', 'task', 'official_split', 'source_question_type', 'turn_pairs']:
            assert row[key] == parent[key]
        assert row['curation']['parent_content_sha256'] == parent['content_sha256']
        assert row['curation']['physical_ground_truth_verified'] is False
        assert len(row['messages']) == len(parent['messages'])
        assert row['content_sha256'] == digest(row['messages'])
        assert row['media_ids'] == list(dict.fromkeys(mid for m in row['messages'] for mid in m['media_ids']))
        canonical = copy.deepcopy(row['messages'])
        for i, (m, old) in enumerate(zip(row['messages'], parent['messages'])):
            assert m['role'] == old['role']
            assert m['media_ids'] == [aliases.get(mid, mid) for mid in old['media_ids']]
            assert [hashes[mid] for mid in m['media_ids']] == [hashes[mid] for mid in old['media_ids']]
            assert m['content'].strip() and m['content'].count('<tactile>') == len(m['media_ids'])
            assert not re.search(r'<(?:more_property|less_property|most_property|least_property|scenario_question|img_tokens|video_tokens)>', m['content'])
            if m != old:
                change = logs[(row['id'], i)]
                assert change['before'] == old['content'] and change['after'] == m['content']
                assert change['before_media_ids'] == old['media_ids'] and change['after_media_ids'] == m['media_ids']
            assert 'a extremely hard' not in m['content']
            if dataset == 'vtv_llm':
                assert not re.search(r'\b(?:rougher|smoother|roughest|smoothest)\b', m['content'])
            canonical[i]['media_ids'] = [hashes[mid] for mid in m['media_ids']]
        equivalent = digest(canonical)
        assert equivalent == row['curation']['media_content_equivalence_sha256']
        if row['curation']['output_split'] == 'train':
            assert equivalent not in seen_train, row['id']
            seen_train.add(equivalent)

        answer = row['messages'][1]['content']
        prefix, _, conclusion = answer.partition('Conclusion:')
        conclusion = conclusion.strip()
        if row['task'] == 'property_description':
            sections = [answer]
        elif row['task'] == 'property_comparison':
            pattern = r'First object: (.*?) Second object: (.*?)\s*$' if dataset == 'physiclear_octopi' else r'Object 1: (.*?) Object 2: (.*?)\s*$'
            match = re.fullmatch(pattern, prefix)
            assert match, row['id']
            sections = list(match.groups())
        else:
            sections = [text for _, text in re.findall(r'(?:^| )([abc])\) (.*?)(?= [abc]\) |$)', prefix.rstrip())]
        assert len(sections) == len(objects), row['id']
        for section, object_name in zip(sections, objects):
            if dataset == 'physiclear_octopi':
                match = re.fullmatch(r'It presents a (soft|moderately hard|hard) and (smooth|slightly rough|rough) surface with (no bumps|small bumps|big bumps)\.', section)
                properties = [('hardness', c.HARDNESS_MAP), ('roughness', c.ROUGHNESS_MAP), ('texture', c.TEXTURE_MAP)]
            else:
                match = re.fullmatch(r'The object is (highly deformable|moderately deformable|extremely hard)\. Its surface has (no protrusion|moderate protrusion|strong protrusion)\. It has (no elasticity|moderate elasticity|strong elasticity) and (slight friction|moderate friction|strong friction)\.', section)
                properties = [('hardness', c.HARDNESS_MAP), ('protrusion', c.PROTRUSION_MAP), ('elasticity', c.ELASTICITY_MAP), ('friction', c.FRICTION_MAP)]
            assert match, (row['id'], section)
            for phrase, (property_name, mapping) in zip(match.groups(), properties):
                value = {label: grade for grade, label in mapping.items()}[phrase]
                assert value == c.RANKS[property_name][object_name], (row['id'], object_name, property_name)
                counts['individual_attribute_grades'] += 1
        counts['structured_description_records'] += 1
        question = row['messages'][0]['content']
        task = row['task']
        if task in {'property_comparison', 'property_selection'}:
            prop, direction = property_query(question, dataset, task == 'property_selection')
            assert (prop, direction) == property_query(parent['messages'][0]['content'], dataset, task == 'property_selection')
            values = [c.RANKS[prop][name] for name in objects]
            if task == 'property_comparison':
                delta = (values[0] - values[1]) * direction
                if dataset == 'physiclear_octopi':
                    actual = 'tie' if conclusion.startswith('Both objects are similar') else 'yes' if conclusion.startswith('Yes,') else 'no' if conclusion.startswith('No,') else 'unknown'
                    assert actual == ('tie' if delta == 0 else 'yes' if delta > 0 else 'no')
                else:
                    assert delta != 0
                    expected = ('Yes.' if delta > 0 else 'No.') if 'Describe both objects before answering.' in question else ('The first object.' if delta > 0 else 'The second object.')
                    assert conclusion == expected
            else:
                target = (max if direction > 0 else min)(values)
                eligible = [i for i, value in enumerate(values) if value == target]
                match = re.match(r'([abc])\) is the ', conclusion) if dataset == 'physiclear_octopi' else re.search(r'object.* is ([abc])\)', conclusion)
                assert match and len(eligible) == 1 and ord(match[1]) - ord('a') == eligible[0], (row['id'], conclusion)
            counts[task + '_conclusion'] += 1
        elif task == 'object_matching':
            assert all(f'{chr(97+i)}) is {c.OBJECTS[name]}' in conclusion for i, name in enumerate(objects))
            options = re.split(r', [23]\) ', question.rsplit('1) ', 1)[-1])
            options[-1] = options[-1].removesuffix('.')
            assert len(options) == 3 and options == sorted(options)
            assert set(options) == {c.OBJECTS[name] for name in objects}
            counts['object_matching_identity_and_candidates'] += 1
        elif task == 'scenario_reasoning':
            q = row['messages'][-2]['content']
            assert row['curation']['output_split'] == 'auxiliary_scenario_choice_test'
            assert row['curation']['annotation_kind'] == 'choice_only_no_rationale'
            assert 'because' not in q and row['messages'][-1]['content'] == parent['messages'][-1]['content']
            scenario = next(s for s in c.SCENARIOS if q.startswith(s['question']))
            chosen = ord(re.search(r'object is ([ab])\)', row['messages'][-1]['content'])[1]) - ord('a')
            if 'target_objects' in scenario:
                eligible = [name in scenario['target_objects'] for name in objects]
            else:
                eligible = [all(allowed == -1 or c.RANKS[prop][name] in allowed for prop, allowed in zip(['hardness', 'roughness', 'texture'], scenario['target_properties'])) for name in objects]
            assert sum(eligible) == 1 and eligible[chosen]
            counts['scenario_choice_alignment'] += 1
    for removal in removed:
        assert removal['input_split'] == 'train'
        kept = kept_by_id[removal['keep_id']]
        assert removal['content_equivalence_sha256'] == kept['curation']['media_content_equivalence_sha256']
        # Reconstruct the removed record from its full before/after log and prove equivalence.
        messages = copy.deepcopy(parents[removal['id']]['messages'])
        for i, message in enumerate(messages):
            if (removal['id'], i) in logs:
                change = logs[(removal['id'], i)]
                assert change['before'] == message['content'] and change['before_media_ids'] == message['media_ids']
                message['content'] = change['after']
                message['media_ids'] = change['after_media_ids']
            message['media_ids'] = [hashes[mid] for mid in message['media_ids']]
        assert digest(messages) == removal['content_equivalence_sha256']
    pools = {s: {hashes[mid] for row in outputs[s] for mid in row['media_ids']} for s in ['train', 'val', 'test']}
    overlap = {f'{a}/{b}': len(pools[a] & pools[b]) for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]}
    assert not any(overlap.values())
    assert all(not row['task'] == 'scenario_reasoning' for s in ['train', 'val', 'test'] for row in outputs[s])
    assert len(outputs['val']) == 36 and all(r['dataset'] == 'physiclear_octopi' for r in outputs['val'])
    assert len(outputs['auxiliary_vtv_val']) == 75
    assert len(outputs['auxiliary_scenario_choice_test']) == 46
    # Any changed message must have exactly one reversible correction-log entry.
    all_messages = sum(len(r['messages']) for r in rows) + sum(len(parents[r['id']]['messages']) for r in removed)
    assert len(logs) <= all_messages
    report = {'status': 'passed', 'output_counts': {s: len(v) for s, v in outputs.items()},
              'checks': dict(counts), 'protected_input_files_unchanged': len(summary['protected_input_sha256']),
              'reversible_logged_message_changes': len(logs), 'removed_duplicate_equivalence_verified': len(removed_by_id),
              'main_split_exact_media_content_overlap': overlap, 'physical_ground_truth_verified': False,
              'all_output_source_and_media_content_identity_preserved': True,
              'input_rank_tables_sha256': {name: sha(ROOT / name) for name in ['Octopi/upstream/utils/constants.py', 'VTVLLM/upstream/utils/constants_video.py']}}
    (META / 'validation_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    summary['status'] = 'verified_rank_consistent_not_physical_relabeling'
    (META / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
