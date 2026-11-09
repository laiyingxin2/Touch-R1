"""Preview one candidate row and produce its explicit model-input contract."""
import argparse
import gzip
import json
from pathlib import Path
from candidate_loader import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('jsonl', type=Path)
    parser.add_argument('--allow-weak-labels', action='store_true',
                        help='Explicitly enable unverified candidate supervision')
    parser.add_argument('--row', type=int, default=0)
    args = parser.parse_args()
    if args.row < 0:
        parser.error('--row must be nonnegative')
    opener = gzip.open if args.jsonl.suffix == '.gz' else open
    with opener(args.jsonl, 'rt', encoding='utf-8') as stream:
        for index, line in enumerate(line for line in stream if line.strip()):
            if index == args.row:
                record = json.loads(line)
                prepared = prepare(record, allow_unverified_source=args.allow_weak_labels,
                                   release_directory=args.jsonl.resolve().parent.parent)
                print(json.dumps(prepared, ensure_ascii=False, indent=2))
                return
    parser.error('row index exceeds the dataset length')


if __name__ == '__main__':
    main()
