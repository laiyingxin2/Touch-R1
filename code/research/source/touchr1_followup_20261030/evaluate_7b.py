"""Final frozen 7B GRPO/SFT test predictions, then separate label scoring."""
import gc
import json
import os
from pathlib import Path
import sys
from stage_utils import ROOT, TRAIN, read, write, sha, stamp, init_distributed, run_guarded


def combine(folder, rows, world):
    found = {}
    for rank in range(world):
        path = folder / f'rank_{rank}.jsonl'
        assert sha(path) == read(folder / f'rank_{rank}_sealed.json')['sha256']
        for line in path.read_text().splitlines():
            record = json.loads(line)
            assert record['id'] not in found
            found[record['id']] = record
    assert set(found) == {r['id'] for r in rows}
    dest = folder / 'predictions.jsonl'
    dest.write_text(''.join(json.dumps(found[r['id']], ensure_ascii=False) + '\n' for r in rows))
    write(folder / 'sealed.json', dict(utc=stamp(), n=len(rows), sha256=sha(dest), gold_opened=False))
    return list(found.values())


def score_all(stage, specifications, datasets):
    import importlib.util
    sys.path.insert(0, '/mnt/workspace/lyx/touchr1_partial_test_amount_search_20261028')
    from vtv_scoring import score as vtv_score
    from native_scoring import parse_result
    from rewarding import answer
    from scoring import score_predictions
    common_spec = importlib.util.spec_from_file_location('followup_test_common', TRAIN / 'common.py')
    common = importlib.util.module_from_spec(common_spec)
    common_spec.loader.exec_module(common)
    reports = {}
    # Every dataset/model prediction set is sealed before opening labels.
    for model in specifications:
        for name in datasets:
            folder = stage / model['label'] / name
            assert sha(folder / 'predictions.jsonl') == read(folder / 'sealed.json')['sha256']
    for model in specifications:
        reports[model['label']] = {}
        for name in datasets:
            folder = stage / model['label'] / name
            predictions = [json.loads(s) for s in (folder / 'predictions.jsonl').read_text().splitlines()]
            gold = [json.loads(s) for s in (ROOT / 'evaluation_data' / f'{name}_gold.jsonl').read_text().splitlines()]
            by_id = {r['id']: answer(r['response']) for r in predictions}
            if name == 'touchreason':
                metrics = score_predictions(predictions, gold, [])
                parsed = [parse_result(r, by_id['touchreason_bench/' + r['qa_id']]) for r in gold]
                metrics['table'] = common.table_metrics(parsed)
            else:
                metrics = vtv_score(gold, by_id)
            write(folder / 'metrics.json', metrics)
            reports[model['label']][name] = metrics
    write(stage / 'report.json', dict(utc=stamp(), models=specifications, metrics=reports,
        data_manifest=read(ROOT / 'evaluation_data/manifest.json'),
        labels_opened_after_all_predictions_sealed=True, test_used_for_selection=False))


def main():
    import torch
    import torch.distributed as dist
    sys.path.insert(0, str(TRAIN))
    from common import make_engine
    rank, world = init_distributed(29641)
    stage = ROOT / 'stages' / os.environ['FOLLOWUP_STAGE']
    stage.mkdir(parents=True, exist_ok=True)
    data = ROOT / 'evaluation_data'
    manifest = read(data / 'manifest.json')
    datasets = {}
    for name in ['touchreason', 'vtv']:
        path = data / f'{name}_inputs.jsonl'
        assert sha(path) == manifest[name]['inputs_sha256']
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        assert len(rows) == manifest[name]['n']
        assert all(not {'answer', 'target', 'vtv_gold'} & set(r) for r in rows)
        datasets[name] = rows
    if rank == 0:
        complete = read(TRAIN / 'complete.json')
        checkpoints = [('grpo_final', complete['latest']['checkpoint']),
                       ('sft_reference', read(TRAIN / 'plan.json')['sft_reference'])]
        selected = complete['best_validation']['checkpoint']
        if selected not in [path for _, path in checkpoints]:
            checkpoints.insert(0, ('grpo_validation_selected', selected))
        specs = [dict(label=label, checkpoint=path, adaptation_sha256=sha(Path(path) / 'adaptation.pt'))
                 for label, path in checkpoints]
        write(stage / 'frozen_models.json', specs)
    dist.barrier()
    specs = read(stage / 'frozen_models.json')
    engine = make_engine(read(TRAIN / 'plan.json')['config'], specs[0]['checkpoint'])
    engine.peft.gradient_checkpointing_disable()
    for model in specs:
        assert sha(Path(model['checkpoint']) / 'adaptation.pt') == model['adaptation_sha256']
        engine.load(model['checkpoint'])
        engine.mode(False)
        for name, rows in datasets.items():
            folder = stage / model['label'] / name
            folder.mkdir(parents=True, exist_ok=True)
            selected = rows[rank::world]
            path = folder / f'rank_{rank}.jsonl'
            done = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
            assert [r['id'] for r in done] == [r['id'] for r in selected[:len(done)]]
            with path.open('a') as output, torch.no_grad():
                for row in selected[len(done):]:
                    response, tokens = engine.generate(row, max_new_tokens=384)
                    output.write(json.dumps(dict(id=row['id'], response=response, token_ids=tokens), ensure_ascii=False) + '\n')
                    output.flush()
            write(folder / f'rank_{rank}_sealed.json', dict(n=len(selected), sha256=sha(path), gold_opened=False))
            dist.barrier()
            if rank == 0:
                combine(folder, rows, world)
            dist.barrier()
        gc.collect()
        torch.cuda.empty_cache()
    if rank == 0:
        score_all(stage, specs, datasets)
        write(stage / 'complete.json', dict(utc=stamp(), status='complete', world=world,
            native_questions=4800, vtv_questions=550, models=[s['label'] for s in specs]))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    run_guarded(main)
