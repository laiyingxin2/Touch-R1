"""Score typed tactile QA predictions, including all-variants-correct robustness."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def read(path):
    return [json.loads(line) for line in Path(path).open(encoding='utf-8')]


def normalize(answer):
    if isinstance(answer, str):
        return answer.strip().lower().removesuffix('.').removesuffix(')')
    if isinstance(answer, dict):
        return {str(k).strip().lower(): normalize(v) for k, v in answer.items()}
    return answer


def response_object(prediction):
    if 'answer' in prediction:
        return {'answer': prediction['answer'], 'evidence': prediction.get('evidence', {})}
    text = prediction.get('response', '').strip()
    # Accept a JSON response surrounded by markdown or an explanatory preamble.
    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char == '{':
            try:
                value, _ = decoder.raw_decode(text[i:])
                if isinstance(value, dict) and 'answer' in value:
                    return value
            except json.JSONDecodeError:
                pass
    return {'answer': None, 'evidence': {}}


def score(gold, predictions):
    by_id = {p['id']: p for p in predictions}
    assert len(by_id) == len(predictions), 'Duplicate prediction IDs'
    expected_ids = {r['id'] for r in gold}
    assert len(expected_ids) == len(gold)
    assert not set(by_id) - expected_ids, 'Predictions contain unknown IDs'
    correct = {}
    dimensions = {key: defaultdict(list) for key in ['dataset_task', 'property', 'ordinal_gap']}
    evidence_total = evidence_right = 0
    groups = defaultdict(list)
    parsed = 0
    for row in gold:
        prediction = response_object(by_id.get(row['id'], {}))
        target = json.loads(row['messages'][1]['content'])
        good = normalize(prediction['answer']) == normalize(target['answer'])
        parsed += prediction['answer'] is not None
        correct[row['id']] = good
        dimensions['dataset_task'][row['dataset'] + '/' + row['task']].append(good)
        dimensions['property'][str(row['evaluation']['property'])].append(good)
        dimensions['ordinal_gap'][str(row['evaluation']['ordinal_gap'])].append(good)
        supplied = prediction.get('evidence', {})
        if not isinstance(supplied, dict):
            supplied = {}
        for input_name, properties in target['evidence'].items():
            for prop, value in properties.items():
                evidence_total += 1
                actual = supplied.get(input_name, {})
                evidence_right += isinstance(actual, dict) and normalize(actual.get(prop)) == normalize(value)
        if 'group_id' in row['evaluation']:
            groups[row['evaluation']['group_id']].append((good, row['evaluation']['expected_variants']))
    detail = {dimension: {key: {'n': len(flags), 'accuracy': sum(flags) / len(flags)} for key, flags in sorted(bins.items())}
              for dimension, bins in dimensions.items()}
    result = {'n': len(gold), 'predictions_supplied': len(predictions), 'parseable_answers': parsed,
              'answer_accuracy': sum(correct.values()) / len(gold) if gold else None,
              'macro_accuracy_over_dataset_task': sum(x['accuracy'] for x in detail['dataset_task'].values()) / len(detail['dataset_task']) if gold else None,
              'category_evidence_accuracy': evidence_right / evidence_total if evidence_total else None,
              'category_evidence_values': evidence_total, 'breakdown': detail}
    if groups:
        assert all(len(v) == v[0][1] for v in groups.values()), 'Incomplete gold robustness group'
        result['all_variants_correct_accuracy'] = sum(all(good for good, _ in v) for v in groups.values()) / len(groups)
        result['robustness_groups'] = len(groups)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold', required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--output')
    args = parser.parse_args()
    result = score(read(args.gold), read(args.predictions))
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(encoded, encoding='utf-8')
    print(encoded)
