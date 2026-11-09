"""Reproduce QA using the downloaded official scripts; no API or model inference."""
import argparse
from pathlib import Path
import random
import runpy
import sys

ROOT = Path(__file__).resolve().parent


def run(script, args):
    random.seed(0)
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), *args]
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['Octopi', 'VTVLLM'])
    args = parser.parse_args()
    base = ROOT / args.dataset
    if args.dataset == 'Octopi':
        import numpy as np
        np.random.seed(0)
        run(base / 'upstream/utils/process_dataset.py', [
            '--dataset_path', str(base / 'upstream/dataset'),
            '--output_path', str(base / 'prepared')])
        run(base / 'upstream/utils/generate_qa.py', ['--data_path', str(base / 'prepared')])
    else:
        run(base / 'upstream/utils/process_dataset_video.py', [
            '--dataset_path', str(base / 'extracted/vbts_video/vbts_video'),
            '--output_path', str(base / 'prepared/vbts_video'), '--seed', '0'])
        run(base / 'upstream/utils/generate_qa_video.py', [
            '--data_path', str(base / 'prepared'), '--seed', '0'])
