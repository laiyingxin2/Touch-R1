"""Native scorers shared by development and held-out evaluation."""
import collections
import json
import sys

from data import ROOT, Dataset, SNAPSHOT, rows, read, write
from rewarding import answer, segments
from native_scoring import parse_result, summarize
from score_quality_v3 import score as public_score
import torch
import torch.distributed as dist


def score_predictions(predictions, native_gold, public_gold):
    expected = {('touchreason_bench/'+r['qa_id']) for r in native_gold}
    expected.update('public/'+r['id'] for r in public_gold)
    by_id = {p['id']: p for p in predictions}
    assert len(by_id) == len(predictions) and set(by_id) == expected, 'Incomplete or duplicate predictions'
    native = [parse_result(r, answer(by_id['touchreason_bench/'+r['qa_id']]['response'])) for r in native_gold]
    public_predictions = [{'id': r['id'], 'response': answer(by_id['public/'+r['id']]['response'])} for r in public_gold]
    public = public_score(public_gold, public_predictions) if public_gold else None
    # Selection uses exact-answer accuracy averaged across the three datasets.
    # Native ordinal MAE/ranking correlations remain in their native report.
    per_dataset = collections.defaultdict(list)
    for value in native:
        exact = value['correct']
        if value['task'] == 'ranking':
            # Native ranking reports Spearman correlation, which is undefined
            # for all ties and can be 1 for shifted ranks. Keep it unchanged
            # above, but do not mislabel that correlation as exact matching.
            exact = '|'.join(s.strip() for s in value['output'].split('|')) == value['answer']
        per_dataset['touchreason_bench'].append(exact)
    from score_quality_v2 import normalize, response_object
    for row, pred in zip(public_gold, public_predictions):
        gold = json.loads(row['messages'][-1]['content'])['answer']
        per_dataset[row['dataset']].append(normalize(response_object(pred)['answer']) == normalize(gold))
    accuracy = {k: sum(v)/len(v) for k, v in per_dataset.items()}
    return {'n': len(predictions), 'native_touchreason': summarize(native) if native else None,
            'public': public, 'exact_accuracy_per_dataset': accuracy,
            'macro_exact_accuracy': sum(accuracy.values())/len(accuracy),
            'three_tag_format_accuracy': sum(bool(segments(p['response'])) for p in predictions)/len(predictions),
            'label_interpretation': 'Agreement with frozen weak/local/public annotations; not calibrated physical accuracy'}


def development(engine, indices, rank, world, folder, name, generate=False):
    dataset = Dataset()
    engine.mode(False)
    totals = torch.zeros(2, device='cuda:0', dtype=torch.float64)
    predictions = []
    with torch.no_grad():
        for idx in indices[rank::world]:
            row = dataset.get(idx)
            x = engine.encode(row)
            lp = engine.logprobs(x).flatten()
            totals += torch.tensor([-lp.sum().item(), lp.numel()], device='cuda:0')
            if generate:
                text, tokens = engine.generate(row, max_new_tokens=384)
                predictions.append({'id': row['id'], 'response': text, 'token_ids': tokens})
    dist.all_reduce(totals)
    result = {'assistant_token_nll': (totals[0]/totals[1]).item(), 'n': len(indices)}
    if generate:
        shards = [None]*world
        dist.all_gather_object(shards, predictions)
        if rank == 0:
            predictions = [p for shard in shards for p in shard]
            selected = [dataset.get(i) for i in indices]
            native_ids = {r['native_id'] for r in selected if r['dataset'] == 'touchreason_bench'}
            native_gold = [r for r in rows(SNAPSHOT/'touchreason_val_qa.jsonl') if r['qa_id'] in native_ids]
            public_gold = [r['public_row'] for r in selected if r['task'] == 'public_json']
            result.update(score_predictions(predictions, native_gold, public_gold))
            write(folder/(name+'_predictions.json'), predictions)
        shared = [result]
        dist.broadcast_object_list(shared, src=0)
        result = shared[0]
    if rank == 0:
        write(folder/(name+'.json'), result)
    dist.barrier()
    return result
