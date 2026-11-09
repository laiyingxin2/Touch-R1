"""Inference with an exported visual+language PEFT checkpoint and normalized inputs."""
import argparse
import json
from pathlib import Path
from model import Engine


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--inputs', required=True, help='Answer-free normalized JSONL, as in test_inputs/')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    engine = Engine(vision_checkpoint=Path(a.checkpoint))
    engine.load(a.checkpoint)
    with Path(a.inputs).open() as f, Path(a.output).open('w') as output:
        for line in f:
            row = json.loads(line)
            text, tokens = engine.generate(row, max_new_tokens=384)
            output.write(json.dumps({'id': row['id'], 'response': text, 'token_ids': tokens}, ensure_ascii=False)+'\n')
            output.flush()
