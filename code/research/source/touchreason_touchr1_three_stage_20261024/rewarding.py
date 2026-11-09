"""Touch-R1 tag/ordinal rewards; weak references never become physical gold."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, '/mnt/workspace/lyx/public_tactile_qa_20261022')
from score_quality_v3 import scales
from score_quality_v2 import normalize

LEVELS = {'hardness': ['soft', 'medium', 'hard'], 'roughness': ['smooth', 'medium', 'rough'],
          'protrusion': ['none', 'small', 'large'], 'elasticity': ['weak', 'medium', 'strong']}
TAGS = ('perceive', 'compare', 'conclude')


def strict_json(value):
    def unique(pairs):
        result = {}
        for k, v in pairs:
            if k in result:
                raise ValueError('Duplicate JSON key')
            result[k] = v
        return result
    return json.loads(value, object_pairs_hook=unique)


def segments(text):
    pattern = r'\s*<perceive>(.*?)</perceive>\s*<compare>(.*?)</compare>\s*<conclude>(.*?)</conclude>\s*'
    match = re.fullmatch(pattern, text, flags=re.S)
    valid = bool(match) and all(text.count(f'<{t}>') == text.count(f'</{t}>') == 1 for t in TAGS)
    return (match.groups() if valid else None)


def answer(text):
    parsed = segments(text)
    if parsed:
        return parsed[2].strip()
    found = re.findall(r'<conclude>(.*?)</conclude>', text, flags=re.S)
    return found[0].strip() if len(found) == 1 else text.strip()


def agreement(row, output):
    value = answer(output)
    expected = row['answer']
    if row['task'] == 'public_json':
        try:
            parsed = strict_json(value)
            target = json.loads(expected)
        except (TypeError, ValueError):
            return -.1, False
        if not isinstance(parsed, dict) or not isinstance(parsed.get('answer'), (str, dict)):
            return -.1, False
        exact = normalize(parsed.get('answer')) == normalize(target['answer'])
        if exact:
            return 1., True
        allowed = scales(row['public_row'])
        if isinstance(target['answer'], dict) and set(target['answer']) <= set(allowed):
            actual = parsed.get('answer')
            if isinstance(actual, dict) and set(actual) == set(target['answer']):
                distances = []
                for prop, gold in target['answer'].items():
                    if actual[prop] not in allowed[prop]:
                        return 0., False
                    distances.append(abs(allowed[prop].index(actual[prop])-allowed[prop].index(gold)))
                return sum(1. if d == 0 else .4 if d == 1 else 0. for d in distances)/len(distances), False
        return 0., False
    if not value:
        return -.1, False
    canonical = lambda text: str(text).strip().lower().removesuffix('.').removesuffix(')')
    options = row['input'].get('options')
    task = row['task'].replace('joint_partial_', 'joint_')
    if task == 'joint_ms':
        choices = [s for s in re.split(r'[,\s]+', value.upper()) if s]
        valid = bool(choices) and len(set(choices)) == len(choices) and all(s in options for s in choices)
        if not valid:
            return -.1, False
        exact = set(choices) == set(str(expected).split(','))
        return float(exact), exact
    if options:
        choice = value.upper().rstrip('.)')
        if choice not in options:
            return -.1, False
    if task == 'ranking':
        ranks = value.split('|')
        if len(ranks) != len(str(expected).split('|')) or any(not re.fullmatch('[1-3]', s.strip()) for s in ranks):
            return -.1, False
        normalized = '|'.join(s.strip() for s in ranks)
        exact = normalized == str(expected)
        return float(exact), exact
    if task == 'joint_em':
        if len(value.split('|')) != len(str(expected).split('|')):
            return -.1, False
    if canonical(value) == canonical(expected):
        return 1., True
    levels = LEVELS.get(row.get('attribute'))
    predicted, gold = canonical(value), canonical(expected)
    if options:
        predicted = canonical(options.get(value.upper(), ''))
        gold = canonical(options.get(str(expected), ''))
    if row['task'] == 'ordinal' and re.fullmatch('[1-3]', str(expected)):
        if not re.fullmatch('[1-3]', value):
            return -.1, False
        distance = abs(int(expected)-int(value))
        return .4 if distance == 1 else 0., False
    if levels and predicted in levels and gold in levels:
        return .4 if abs(levels.index(predicted)-levels.index(gold)) == 1 else 0., False
    if levels and not options and predicted not in levels:
        return -.1, False
    return 0., False


def reward(row, text):
    accuracy, exact = agreement(row, text)
    fmt = bool(segments(text))
    # This release does not establish calibrated, same-physical-object pairs.
    # Do not reinterpret same category, file names, or model consensus as one.
    consistency = 0.
    assert not row.get('paired_physical_identity_verified'), 'Verified new pair types need an explicit ordinal-summary parser'
    return {'total': .6*accuracy+.3*consistency+.1*fmt,
            'annotation_accuracy_reward': accuracy, 'exact_answer': exact, 'format_reward': fmt,
            'cross_sensor_physical_reward': consistency,
            'reference_tier': row.get('label_tier', 'validation_reference'),
            'calibrated_physical_reward': False}


def token_segments(tokenizer, tokens):
    # This mapping affects loss aggregation only. Likelihoods ALWAYS use the
    # exact sampled token IDs, never decoded/re-tokenized completions.
    prefixes = tokenizer.batch_decode([tokens[:i] for i in range(1, len(tokens)+1)],
                                      skip_special_tokens=False, clean_up_tokenization_spaces=False)
    text = prefixes[-1]
    compare, conclude = text.find('<compare>'), text.find('<conclude>')
    if compare < 0 or conclude <= compare:
        return [list(range(len(tokens)))]
    groups = [[], [], []]
    for i, prefix in enumerate(prefixes):
        groups[0 if len(prefix) <= compare else 1 if len(prefix) <= conclude else 2].append(i)
    return [g for g in groups if g]


def segment_mean(value, groups):
    return sum(value[g].mean() for g in groups)/len(groups)
