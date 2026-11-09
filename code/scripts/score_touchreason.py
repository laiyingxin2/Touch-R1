"""Score complete predictions with the archived local TouchReason parser."""
import argparse
import gzip
import importlib.util
import json
from pathlib import Path


def rows(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def score(predictions, labels):
    source = Path(__file__).resolve().parents[1] / 'research/source/touchreason_touchr1_three_stage_20261024/native_scoring.py'
    spec = importlib.util.spec_from_file_location('touchr1_native_scoring', source)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    by_id = {}
    for record in predictions:
        identifier = record.get('qa_id', record.get('id', '')).removeprefix('touchreason_bench/')
        if not identifier or identifier in by_id:
            raise ValueError(f'Missing or duplicate prediction ID: {identifier}')
        text = record['response']
        if '<conclude>' in text:
            if text.count('<conclude>') != 1 or text.count('</conclude>') != 1:
                raise ValueError(f'Malformed conclude tags: {identifier}')
            text = text.split('<conclude>', 1)[1].split('</conclude>', 1)[0]
        by_id[identifier] = text
    expected = {record['qa_id'] for record in labels}
    if len(expected) != len(labels) or set(by_id) != expected:
        raise ValueError('Prediction/label IDs must match exactly, with no duplicate labels')
    parsed = [native.parse_result(record, by_id[record['qa_id']]) for record in labels]
    result = native.summarize(parsed)
    result.update(n=len(labels), native_parser_correct_rate=sum(record['correct'] for record in parsed) / len(parsed),
                  correct_rate_definition='Task-specific parser correctness; not literal answer-string exact agreement',
                  protocol='Local reconstructed TouchReason benchmark; not official paper equivalence')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = score(rows(args.predictions), rows(args.labels))
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end='')
