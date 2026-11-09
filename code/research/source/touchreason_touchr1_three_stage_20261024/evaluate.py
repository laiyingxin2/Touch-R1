"""Freeze model choice, generate answer-free test predictions, then score."""
import json
import os
from pathlib import Path

from data import ROOT, SNAPSHOT, read, write, rows, digest, key
from model import Engine
from distributed import initialize, seed_rank, emit
from prepare_test_inputs import prepare
from scoring import score_predictions
import torch.distributed as dist


def read_predictions(folder):
    result = {}
    for path in sorted(folder.glob('rank_*.jsonl')):
        with path.open('rb') as f:
            content = f.read()
        if content and not content.endswith(b'\n'):
            # A process interruption may leave a final partial record. Retain
            # complete durable records; this repair happens before any writer.
            content = content[:content.rfind(b'\n')+1]
            path.write_bytes(content)
        for line in content.splitlines():
            p = json.loads(line)
            assert p['id'] not in result, 'Duplicate test prediction IDs'
            result[p['id']] = p
    return result


def evaluate_set(engine, records, folder, control, rank, world):
    folder.mkdir(parents=True, exist_ok=True)
    prior = read_predictions(folder) if rank == 0 else None
    shared = [prior]
    dist.broadcast_object_list(shared, src=0)
    done = set(shared[0])
    del shared
    expected = {r['id'] for r in records}
    assert done <= expected and len(expected) == len(records)
    remaining = [r for r in records if r['id'] not in done]
    with (folder/f'rank_w{world}_{rank}.jsonl').open('a') as f:
        for i, row in enumerate(remaining[rank::world]):
            if control == 'mismatch':
                keys = {m['key'] for m in row['media']}
                for m in row['media']:
                    candidates = engine.control_donors[m['key']]
                    engine.mismatch[m['key']] = next((d for d in candidates if d['key'] not in keys), candidates[0])
            text, tokens = engine.generate(row, control=control, max_new_tokens=384)
            value = {'id': row['id'], 'response': text, 'token_ids': tokens, 'control': control}
            f.write(json.dumps(value, ensure_ascii=False)+'\n')
            f.flush()
            if i % 16 == 0:
                emit(ROOT/'test', 'test_prediction_progress', variant=folder.parent.name,
                     control=control, dataset=folder.name, rank=rank, local_generated=i+1)
    dist.barrier()
    if rank == 0:
        predictions = read_predictions(folder)
        assert set(predictions) == expected
        write(folder/'predictions.json', [predictions[r['id']] for r in records])
        write(folder/'sealed.json', {'n': len(records), 'predictions_sha256': digest(folder/'predictions.json'),
                                    'input_id_sha256': key([r['id'] for r in records]),
                                    'gold_scoring_started': False})
    dist.barrier()


def main():
    rank, world = initialize()
    folder = ROOT/'test'
    folder.mkdir(exist_ok=True)
    if rank == 0:
        prepare()
        complete = read(ROOT/'grpo/complete.json')
        models = {'three_stage_last': complete['actual_three_stage_checkpoint']}
        if complete['selected_checkpoint'] != complete['actual_three_stage_checkpoint']:
            models['development_selected'] = complete['selected_checkpoint']
        frozen = {'models': {name: {'checkpoint': path, 'adaptation_sha256': digest(Path(path)/'adaptation.pt'),
                   'vision_sha256': digest(Path(path)/'vision_adapter.pt')} for name, path in models.items()},
                  'test_inputs_manifest_sha256': digest(ROOT/'test_inputs/manifest.json'),
                  'selection_frozen_before_test': True, 'weak_supervision_experiment': True}
        if (folder/'frozen_models.json').exists():
            assert read(folder/'frozen_models.json') == frozen
        else:
            write(folder/'frozen_models.json', frozen)
    dist.barrier()
    frozen = read(folder/'frozen_models.json')
    # Test workers consume only the exported answer-free files.
    datasets = {name: list(rows(ROOT/'test_inputs'/(name+'.jsonl')))
                for name in ['touchreason', 'public', 'public_relation']}
    donors = read(ROOT/'test_inputs/mismatch_donors.json')
    engine = Engine()
    engine.control_donors = donors
    seed_rank(rank)
    for variant, spec in frozen['models'].items():
        engine.load(spec['checkpoint'])
        for control in ['real', 'blank', 'mismatch']:
            for name, records in datasets.items():
                evaluate_set(engine, records, folder/variant/control/name, control, rank, world)
    # Seal ALL variants and controls before opening native test scoring gold.
    if rank == 0:
        for variant in frozen['models']:
            for control in ['real', 'blank', 'mismatch']:
                for name in datasets:
                    assert (folder/variant/control/name/'sealed.json').exists()
        native_gold = list(rows(SNAPSHOT/'test_qa.jsonl'))
        public_gold = list(rows(SNAPSHOT/'public_test.jsonl'))
        relation_gold = list(rows(SNAPSHOT/'public_relation_coverage_test.jsonl'))
        results = {}
        for variant in frozen['models']:
            results[variant] = {}
            for control in ['real', 'blank', 'mismatch']:
                base = folder/variant/control
                combined = read(base/'touchreason/predictions.json')+read(base/'public/predictions.json')
                combined_score = score_predictions(combined, native_gold, public_gold)
                relation_score = score_predictions(read(base/'public_relation/predictions.json'), [], relation_gold)
                results[variant][control] = {'touchreason_plus_public': combined_score, 'public_relation': relation_score}
            real = results[variant]['real']['touchreason_plus_public']['macro_exact_accuracy']
            results[variant]['real_minus_blank_macro_exact'] = real-results[variant]['blank']['touchreason_plus_public']['macro_exact_accuracy']
            results[variant]['real_minus_mismatch_macro_exact'] = real-results[variant]['mismatch']['touchreason_plus_public']['macro_exact_accuracy']
        report = {'status': 'complete', 'models': frozen, 'results': results,
             'numerical_implementation_history': read(ROOT/'checks/numerical_implementation_history.json')
                 if (ROOT/'checks/numerical_implementation_history.json').exists() else None,
             'weak_supervision_experiment': True, 'official_paper_reproduction': False,
             'calibrated_physical_accuracy_claim': False, 'test_used_for_model_selection': False,
             'limits': ['TouchReason v6 contains source hypotheses rather than calibrated physical gold.',
                 'The 4800-question benchmark is a frozen local reconstruction, not an official equivalence claim.',
                 'No verified same-physical-object cross-sensor reward or force conditioning was used.',
                 'Report actual last GRPO separately if development selection prefers Stage 2.']}
        write(folder/'report.json', report)
        write(folder/'complete.json', {'status': 'complete', 'report': str(folder/'report.json'),
                                      'report_sha256': digest(folder/'report.json')})
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        write(ROOT/'test'/f"failure_{os.environ.get('RANK', 'unknown')}.json", {'error': str(exc), 'type': type(exc).__name__})
        raise
