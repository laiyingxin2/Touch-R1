"""Fixed global-batch SFT followed by a sealed, answer-free 4800-item test."""
import collections
import datetime
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

from experiment import RUN, SOURCE, SEARCH, read, write, digest, identity, emit, Dataset, engine_for


def initial_seed(seed):
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def rng_state():
    import numpy as np
    import torch
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state()}


def restore_rng(value):
    import numpy as np
    import torch
    random.setstate(value['python']); np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch']); torch.cuda.set_rng_state(value['cuda'])


def optimizer_for(engine, config):
    import torch
    language = [p for n, p in engine.parameters if not n.startswith('tactile_adapter.')]
    adapter = [p for n, p in engine.parameters if n.startswith('tactile_adapter.')]
    return torch.optim.AdamW([
        {'params': language, 'lr': config['lr'], 'initial_lr': config['lr']},
        {'params': adapter, 'lr': config['lr'] * config['adapter_ratio'],
         'initial_lr': config['lr'] * config['adapter_ratio']}],
        betas=(.9, config.get('beta2', .95)), weight_decay=config['weight_decay'], foreach=True)


def allreduce_gradients(engine, clip):
    import torch
    import torch.distributed as dist
    # Each local token sum was divided by the GLOBAL token count. Sum the
    # gradients; averaging them again would shrink updates by world size.
    flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                      for _, p in engine.parameters])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    offset = 0
    for _, p in engine.parameters:
        p.grad = flat[offset:offset+p.numel()].view_as(p)
        offset += p.numel()
    norm = torch.nn.utils.clip_grad_norm_([p for _, p in engine.parameters], clip)
    assert torch.isfinite(norm), 'Non-finite global gradient'
    return norm.item()


def checkpoint(engine, optimizer, plan, cursor, step, order_hash, rank, world):
    import torch.distributed as dist
    states = [None] * world
    dist.all_gather_object(states, rng_state())
    folder = RUN / 'checkpoints' / f'presentations_{cursor:07d}'
    if rank == 0:
        engine.save(folder, {'cursor': cursor, 'step': step, 'order_hash': order_hash,
                    'plan_sha256': digest(RUN / 'plan.json'), 'world_size': world, 'rng': states,
                    'optimizer': optimizer.state_dict()})
        # The durable latest pointer changes only AFTER the full checkpoint.
        write(RUN / 'latest.json', {'checkpoint': str(folder), 'cursor': cursor, 'step': step})
        emit('checkpoint_saved', checkpoint=str(folder), cursor=cursor, step=step)
        paths = sorted((RUN / 'checkpoints').glob('presentations_*'))
        keep = set(paths[-2:])
        keep.update(p for p in paths if int(p.name.split('_')[-1]) % plan['expected_training_rows'] == 0)
        for path in paths:
            if path not in keep:
                shutil.rmtree(path)
    dist.barrier()
    return folder


def train(engine, plan, rank, world):
    import torch
    import torch.distributed as dist
    from parameters import learning_rate_factor
    config = plan['config']
    dataset = Dataset()
    indices = dataset.indices()
    assert len(indices) == plan['expected_training_rows']
    orders = []
    for epoch in range(plan['epochs']):
        order = list(indices)
        random.Random(config['seed'] + 991 + epoch).shuffle(order)
        orders.extend(order)
    order_hash = identity(orders)
    optimizer = optimizer_for(engine, config)
    cursor = step = 0
    initial_seed(config['seed'] + rank)
    if (RUN / 'latest.json').exists():
        path = Path(read(RUN / 'latest.json')['checkpoint'])
        engine.load(path)
        state = torch.load(path / 'trainer_state.pt', map_location='cpu', weights_only=False)
        assert state['plan_sha256'] == digest(RUN / 'plan.json') and state['order_hash'] == order_hash
        optimizer.load_state_dict(state['optimizer'])
        for values in optimizer.state.values():
            for name, tensor in values.items():
                if isinstance(tensor, torch.Tensor) and name != 'step':
                    values[name] = tensor.to('cuda:0')
        cursor, step = state['cursor'], state['step']
        if state['world_size'] == world:
            restore_rng(state['rng'][rank])
        elif rank == 0:
            emit('resource_topology_changed', old_world=state['world_size'], new_world=world,
                 stochastic_dropout_schedule_changed=True, global_batch_unchanged=True)
        del state
    started, initial_cursor = time.monotonic(), cursor
    if rank == 0:
        emit('sft_setup', train_unique=len(indices), budget=len(orders), epochs=plan['epochs'],
             initial_cursor=cursor, global_batch=config['batch'], world_size=world,
             trainable_parameters=sum(p.numel() for _, p in engine.parameters), order_hash=order_hash)
    while cursor < len(orders):
        selected = orders[cursor:cursor+config['batch']]
        local = [dataset.get(i) for i in selected[rank::world]]
        counts = [len(engine.tokenizer(r['target']+engine.tokenizer.eos_token,
                                      add_special_tokens=False)['input_ids']) for r in local]
        total = torch.tensor(sum(counts), dtype=torch.float64, device='cuda:0')
        dist.all_reduce(total)
        assert total.item() > 0
        cursor += len(selected); step += 1
        factor = learning_rate_factor(config, cursor, plan['schedule_horizon'])
        for group in optimizer.param_groups:
            group['lr'] = group['initial_lr'] * factor
        optimizer.zero_grad(set_to_none=True)
        engine.mode(True)
        numerator = 0.
        for row, count in zip(local, counts):
            x = engine.encode(row)
            expected = engine.tokenizer(row['target']+engine.tokenizer.eos_token,
                                        add_special_tokens=False)['input_ids']
            assert x['ids'][0, x['prefix']:].tolist() == expected
            lp = engine.logprobs(x).flatten()
            assert lp.numel() == count
            numerator -= lp.detach().sum().item()
            (-lp.sum()/total).backward()
        norm = allreduce_gradients(engine, config.get('gradient_clip', 1.))
        optimizer.step()
        if step == 1 or step % 16 == 0:
            nll = torch.tensor(numerator, dtype=torch.float64, device='cuda:0')
            dist.all_reduce(nll)
            if rank == 0:
                elapsed = time.monotonic()-started
                emit('sft_step', cursor=cursor, budget=len(orders), step=step, token_nll=(nll/total).item(),
                     gradient_norm=norm, learning_rate=config['lr']*factor, seconds=elapsed,
                     presentations_per_second=(cursor-initial_cursor)/max(elapsed, 1e-6),
                     peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
                write(RUN / 'training_progress.json', {'cursor': cursor, 'budget': len(orders),
                      'step': step, 'fraction': cursor/len(orders), 'world_size': world,
                      'presentations_per_second': (cursor-initial_cursor)/max(elapsed, 1e-6),
                      'updated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
        if step == 1 or cursor == 1024 or cursor % 8192 == 0 or cursor % len(indices) == 0:
            path = checkpoint(engine, optimizer, plan, cursor, step, order_hash, rank, world)
            if step == 1:
                # Check the saved adaptation reproduces the actual policy;
                # this consumes training inputs only, never test examples.
                engine.mode(False)
                with torch.no_grad():
                    probe = engine.encode(dataset.get(indices[rank % len(indices)]))
                    before = engine.logprobs(probe).clone()
                    engine.load(path)
                    after = engine.logprobs(probe)
                    assert torch.equal(before, after), 'Checkpoint changed policy probabilities'
                del probe, before, after
                if rank == 0:
                    write(RUN / 'checkpoint_reload_check.json', {'passed': True,
                          'saved_policy_probabilities_bit_identical': True, 'test_inputs_used': False})
            # Resources can grow between checkpoints without changing batch8.
            resize = RUN / 'resize_requested.json'
            stop = int(rank == 0 and resize.exists())
            shared = [stop]; dist.broadcast_object_list(shared, src=0)
            if shared[0] and cursor < len(orders):
                if rank == 0:
                    write(RUN / 'resize_ready.json', {'cursor': cursor, 'checkpoint': str(path)})
                dist.barrier()
                return False
    if rank == 0:
        final = read(RUN / 'latest.json')['checkpoint']
        write(RUN / 'sft_complete.json', {'status': 'complete', 'checkpoint': final,
              'cursor': cursor, 'unique_training_rows': len(indices), 'epochs': plan['epochs'],
              'plan_sha256': digest(RUN / 'plan.json'),
              'model_selection': plan['model_selection'], 'test_used_for_training_or_selection': False})
    dist.barrier()
    return True


def read_predictions(folder):
    values = {}
    for path in sorted(folder.glob('rank_*.jsonl')):
        content = path.read_bytes()
        if content and not content.endswith(b'\n'):
            content = content[:content.rfind(b'\n')+1] if b'\n' in content else b''
            path.write_bytes(content)
        for line in content.splitlines():
            value = json.loads(line)
            assert value['id'] not in values, 'Duplicate test prediction'
            values[value['id']] = value
    return values


def evaluate(engine, plan, rank, world):
    import torch.distributed as dist
    checkpoint_path = Path(read(RUN / 'sft_complete.json')['checkpoint'])
    frozen = {'checkpoint': str(checkpoint_path), 'adaptation_sha256': digest(checkpoint_path / 'adaptation.pt'),
              'plan_sha256': digest(RUN / 'plan.json'), 'input_manifest': read(RUN / 'test_inputs/manifest.json'),
              'config': plan['config'], 'selected_before_test': True}
    if rank == 0:
        if (RUN / 'frozen_test_model.json').exists():
            assert read(RUN / 'frozen_test_model.json') == frozen
        else:
            write(RUN / 'frozen_test_model.json', frozen)
    dist.barrier()
    engine.load(checkpoint_path)
    engine.mode(False)
    test_file = RUN / 'test_inputs/touchreason.jsonl'
    assert digest(test_file) == frozen['input_manifest']['sha256']
    records = [json.loads(line) for line in test_file.read_text().splitlines() if line.strip()]
    assert all('answer' not in r and 'target' not in r for r in records)
    expected = {r['id'] for r in records}
    assert len(expected) == len(records) == plan['expected_test_rows']
    donor_path = RUN / 'test_inputs/mismatch_donors.json'
    assert digest(donor_path) == frozen['input_manifest']['mismatch_donors_sha256']
    donors = read(donor_path)
    for control in plan['test_controls']:
        folder = RUN / 'test' / control
        folder.mkdir(parents=True, exist_ok=True)
        done = [read_predictions(folder) if rank == 0 else None]
        dist.broadcast_object_list(done, src=0)
        assert set(done[0]) <= expected
        remaining = [r for r in records if r['id'] not in done[0]]
        with (folder / f'rank_w{world}_{rank}.jsonl').open('a') as f:
            for n, row in enumerate(remaining[rank::world], 1):
                if control == 'mismatch':
                    in_question = {m['key'] for m in row['media']}
                    for media in row['media']:
                        options = donors[media['key']]
                        engine.mismatch[media['key']] = next((m for m in options if m['key'] not in in_question), options[0])
                response, tokens = engine.generate(row, control=control, max_new_tokens=384)
                f.write(json.dumps({'id': row['id'], 'response': response,
                        'token_ids': tokens, 'control': control}, ensure_ascii=False)+'\n')
                f.flush()
                if n == 1 or n % 16 == 0:
                    emit('test_prediction_progress', control=control, local_completed=n,
                         local_total=len(remaining[rank::world]))
        dist.barrier()
        if rank == 0:
            values = read_predictions(folder)
            assert set(values) == expected
            write(folder / 'predictions.json', [values[r['id']] for r in records])
            write(folder / 'sealed.json', {'n': len(records), 'sha256': digest(folder / 'predictions.json'),
                  'test_gold_parsed': False})
        dist.barrier()
    if rank == 0:
        score(plan, frozen)
    dist.barrier()


def score(plan, frozen):
    from scoring import score_predictions
    from native_scoring import parse_result
    from rewarding import answer
    for control in plan['test_controls']:
        folder = RUN / 'test' / control
        assert read(folder / 'sealed.json')['sha256'] == digest(folder / 'predictions.json')
    assert digest(SOURCE / 'test_qa.jsonl') == read(RUN / 'archive_verified.json')['files']['test_qa.jsonl']
    # This is the first parsing of native test answers in the experiment.
    gold = [json.loads(line) for line in (SOURCE / 'test_qa.jsonl').read_text().splitlines() if line.strip()]
    metrics, correctness = {}, {}
    for control in plan['test_controls']:
        predictions = read(RUN / 'test' / control / 'predictions.json')
        metric = score_predictions(predictions, gold, [])
        by_id = {r['id']: r for r in predictions}
        tasks, correct = collections.defaultdict(list), {}
        for row in gold:
            text = answer(by_id['touchreason_bench/'+row['qa_id']]['response'])
            value = parse_result(row, text)
            exact = value['correct'] if row['task'] != 'ranking' else '|'.join(s.strip() for s in text.split('|')) == row['answer']
            tasks[row['task']].append(exact); correct[row['qa_id']] = bool(exact)
        metric['per_task_exact_accuracy'] = {k: {'n': len(v), 'correct': sum(v), 'accuracy': sum(v)/len(v)} for k, v in tasks.items()}
        metric['correct'] = sum(correct.values())
        metric['exact_accuracy'] = sum(correct.values())/len(gold)
        metrics[control], correctness[control] = metric, correct
    gaps = {}
    for control in ['blank', 'mismatch']:
        gaps[control] = {'real_minus_control_percentage_points': 100*(metrics['real']['exact_accuracy']-metrics[control]['exact_accuracy']),
            'real_only_correct': sum(correctness['real'][k] and not correctness[control][k] for k in correctness['real']),
            'control_only_correct': sum(not correctness['real'][k] and correctness[control][k] for k in correctness['real'])}
    report = {'status': 'complete', 'dataset': 'paper_aligned_quick_v12', 'n': len(gold),
              'frozen_model': frozen, 'metrics': metrics, 'paired_controls': gaps,
              'all_predictions_sealed_before_gold_scoring': True,
              'weak_supervision_experiment': True, 'test_used_for_selection': False,
              'stage1_retrained_on_v12': False, 'grpo_trained': False,
              'official_benchmark_equivalence_claim': False,
              'interpretation': 'Agreement with frozen local weak labels; not calibrated physical accuracy.'}
    write(RUN / 'test_report.json', report)
    write(RUN / 'complete.json', {'status': 'complete', 'report': str(RUN / 'test_report.json'),
          'report_sha256': digest(RUN / 'test_report.json')})
    emit('test_complete', accuracy=metrics['real']['exact_accuracy'], paired_controls=gaps)


def main():
    import torch
    import torch.distributed as dist
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    assert world in [1, 2, 4, 8]
    torch.cuda.set_device(0)
    dist.init_process_group('nccl', init_method=os.environ['DIST_URL'], rank=rank, world_size=world,
                            timeout=datetime.timedelta(minutes=45))
    plan = read(RUN / 'plan.json')
    assert read(RUN / 'data_ready.json')['plan_sha256'] == digest(RUN / 'plan.json')
    assert plan['config']['batch'] == 8 and not plan['grpo_enabled']
    sys.path.insert(0, str(SEARCH))
    engine = engine_for(plan)
    # Initialization is identical across ranks; explicitly broadcast trainable
    # tensors before the first optimizer step to guard against loader changes.
    for _, p in engine.parameters:
        dist.broadcast(p.data, src=0)
    ready = (RUN / 'sft_complete.json').exists() or train(engine, plan, rank, world)
    if ready:
        evaluate(engine, plan, rank, world)
    dist.destroy_process_group()


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        write(RUN / f"failure_rank_{os.environ.get('RANK', 'unknown')}.json", {
              'error': str(exc), 'type': type(exc).__name__,
              'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
        raise
