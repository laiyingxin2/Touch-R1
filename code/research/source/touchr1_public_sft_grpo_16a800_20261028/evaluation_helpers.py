"""One bounded SFT job, with identical token-normalized loss on one or eight GPUs."""
import datetime
import gc
import json
import math
import os
from pathlib import Path
import random
import sys
import time

from common import (ROOT, Dataset, read, write, digest, identity, emit, make_engine,
                    save_model, evaluate_validation, score_validation)
from fast_worker import batch_logprobs, FeatureCache
from worker import initial_seed, rng_state, restore_rng, optimizer_for, allreduce_gradients
from parameters import learning_rate_factor


def distributed_validation(engine, dataset, indices, folder, rank, world, controls=('real',), evaluation_split='val'):
    import torch.distributed as dist
    folder = Path(folder)
    evaluate_validation(engine, dataset, indices[rank::world], folder / f'rank_{rank}', controls, evaluation_split)
    dist.barrier()
    result = None
    if rank == 0:
        records = [dataset.get(i) for i in indices]
        for control in controls:
            predictions = []
            for r in range(world):
                path = folder / f'rank_{r}' / (control + '.jsonl')
                assert digest(path) == read(path.with_name(control + '_sealed.json'))['sha256']
                predictions.extend(json.loads(s) for s in path.read_text().splitlines())
            by_id = {r['id']: r for r in predictions}
            assert len(by_id) == len(predictions) == len(indices)
            dest = folder / (control + '.jsonl')
            dest.write_text(''.join(json.dumps(by_id[r['id']], ensure_ascii=False) + '\n' for r in records))
            write(folder / (control + '_sealed.json'), dict(n=len(indices), sha256=digest(dest)))
        result = score_validation(dataset, indices, folder, controls, evaluation_split)
    values = [result]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def train(job):
    import torch
    import torch.distributed as dist
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(0)
    dist.init_process_group('nccl', init_method=os.environ['DIST_URL'], rank=rank, world_size=world,
                            timeout=datetime.timedelta(minutes=60))
    folder = Path(job['folder'])
    folder.mkdir(parents=True, exist_ok=True)
    config = job['config']
    config_hash = identity(job)
    cap_mib = int(os.environ['JOB_CAP_MIB'])
    initial_seed(config['seed'])
    latest_path = folder / 'latest.json'
    checkpoint = read(latest_path)['checkpoint'] if latest_path.exists() else job['initial_checkpoint']
    engine = make_engine(config, checkpoint)
    for _, parameter in engine.parameters:
        dist.broadcast(parameter.data, src=0)
    family = job.get('data_family', 'clean')
    fitting = family == 'test_fit'
    evaluation_split = 'test_fit' if fitting else 'val'
    dataset = Dataset(family)
    split = read(ROOT / ('test_fit_splits.json' if fitting else 'splits.json'))
    budget = int(job['presentations'])
    if budget == 0:
        metrics = distributed_validation(engine, dataset, split[job['validation_suite']], folder / 'validation',
                                         rank, world, job.get('controls', ['real']), evaluation_split)
        if rank == 0:
            write(folder / 'result.json', dict(status='complete', job=job, validation=metrics,
                  trainable_parameters=81209920, training_presentations=0, test_used=fitting,
                  same_data_train_and_eval=fitting))
        dist.destroy_process_group()
        return
    pool = list(job['training_indices'])
    assert all(dataset.get(i)['split'] == ('test_fit' if fitting else 'train') for i in pool)
    if not fitting:
        assert not set(pool) & set(split['full_validation'])
    orders = []
    epoch = 0
    while len(orders) < budget:
        order = list(pool)
        random.Random(config['seed'] + 991 + epoch).shuffle(order)
        orders.extend(order)
        epoch += 1
    orders = orders[:budget]
    order_hash = identity(orders)
    optimizer = optimizer_for(engine, config)
    cursor = step = 0
    initial_seed(config['seed'] + rank)
    if latest_path.exists():
        state = torch.load(Path(checkpoint) / 'trainer_state.pt', map_location='cpu', weights_only=False)
        assert state['job_hash'] == config_hash and state['order_hash'] == order_hash
        optimizer.load_state_dict(state['optimizer'])
        for values in optimizer.state.values():
            for name, tensor in values.items():
                if isinstance(tensor, torch.Tensor) and name != 'step':
                    values[name] = tensor.to('cuda:0')
        cursor, step = state['cursor'], state['step']
        if state['world_size'] == world:
            restore_rng(state['rng'][rank])
        del state
    if rank == 0:
        emit(folder, 'training_setup', cursor=cursor, budget=budget, world_size=world,
             batch=config['batch'], config=config, cap_mib=cap_mib,
             optimizer_reset_at_branch_start=True, test_used=fitting, same_data_train_and_eval=fitting)
    # Verify the exact assistant-token alignment and padded loss on train rows.
    engine.mode(False)
    with torch.no_grad():
        probe = engine.encode(dataset.get(orders[cursor if cursor < budget else 0]))
        expected = engine.tokenizer(dataset.get(orders[cursor if cursor < budget else 0])['target'] +
                                   engine.tokenizer.eos_token, add_special_tokens=False)['input_ids']
        assert probe['ids'][0, probe['prefix']:].tolist() == expected
        single = engine.logprobs(probe).flatten()
        batched = batch_logprobs(engine, [probe])
        error = abs(single.mean().item() - batched.mean().item())
        assert error <= .01, ('assistant loss mismatch', error)
        write(folder / f'preflight_rank_{rank}.json', dict(passed=True,
              assistant_tokens=len(expected), loss_difference=error, test_used=fitting))
        del probe, single, batched
    cache = FeatureCache(engine, cap_mib)
    engine.feature = cache
    checkpointing_enabled = True
    if os.environ.get('JOB_ROLE') == 'fast' and cap_mib >= 30720:
        # Short screening examples often fit without recomputation. Spend spare
        # memory on faster real forward/backward work, and fall back on OOM.
        engine.peft.gradient_checkpointing_disable()
        checkpointing_enabled = False
    micro = int(job.get('microbatch', 1))
    max_micro = int(job.get('max_microbatch', micro))
    micro = int(os.environ.get('JOB_INITIAL_MICROBATCH', micro))
    max_micro = int(os.environ.get('JOB_MAX_MICROBATCH', max_micro))
    assert 1 <= micro <= max_micro
    safe_max_micro = max_micro
    last_oom_step = -32
    started, initial_cursor = time.monotonic(), cursor
    checkpoint_every = int(job.get('checkpoint_presentations', 4096))
    next_checkpoint = (cursor // checkpoint_every + 1) * checkpoint_every
    last_checkpoint = time.monotonic()

    def save_checkpoint():
        states = [None] * world
        dist.all_gather_object(states, rng_state())
        path = folder / 'checkpoints' / f'presentations_{cursor:07d}'
        if rank == 0:
            partial = path.with_name(path.name + '.partial')
            assert not path.exists() and not partial.exists(), 'Preserve checkpoint on conflict'
            save_model(engine, partial, config, dict(cursor=cursor, step=step, order_hash=order_hash,
                       job_hash=config_hash, world_size=world, rng=states, optimizer=optimizer.state_dict()))
            partial.rename(path)
            write(latest_path, dict(checkpoint=str(path), cursor=cursor, step=step))
            emit(folder, 'checkpoint', checkpoint=str(path), cursor=cursor)
            # Keep two recent saves, epoch boundaries, and any validation best.
            paths = sorted((folder / 'checkpoints').glob('presentations_*'))
            keep = set(paths[-2:])
            keep.update(p for p in paths if int(p.name.split('_')[-1]) % len(pool) == 0)
            best = folder / 'best_validation.json'
            if best.exists():
                keep.add(Path(read(best)['checkpoint']))
            import shutil
            for old in paths:
                if old not in keep:
                    shutil.rmtree(old)
        dist.barrier()
        return path

    while cursor < budget:
        if os.environ.get('JOB_ROLE') == 'main':
            profile_path = ROOT / 'capacity_profile.json'
            if profile_path.exists():
                profile = read(profile_path)
                desired_cap = int(profile['main_caps_mib'][str(rank)])
                desired_max = min(job.get('max_microbatch', 1),
                                  int(profile['main_max_microbatch'][str(rank)]))
                if desired_cap != cap_mib:
                    optimizer.zero_grad(set_to_none=True)
                    cache.trim(0)
                    gc.collect()
                    torch.cuda.empty_cache()
                    total_mib = torch.cuda.get_device_properties(0).total_memory / 2**20
                    torch.cuda.set_per_process_memory_fraction(desired_cap / total_mib)
                    emit(folder, 'memory_lease_changed', rank=rank, old_cap=cap_mib, cap=desired_cap)
                    cap_mib = desired_cap
                    cache.cap = cap_mib * 2**20
                max_micro = desired_max
                micro = min(micro, max_micro)
        epoch_end = ((cursor // len(pool)) + 1) * len(pool)
        selected = orders[cursor:min(cursor + config['batch'], epoch_end, budget)]
        local = [dataset.get(i) for i in selected[rank::world]]
        counts = [len(engine.tokenizer(r['target'] + engine.tokenizer.eos_token,
                      add_special_tokens=False)['input_ids']) for r in local]
        total = torch.tensor(sum(counts), dtype=torch.float64, device='cuda:0')
        dist.all_reduce(total)
        assert total.item() > 0
        factor = learning_rate_factor(config, cursor + len(selected), budget)
        for group in optimizer.param_groups:
            group['lr'] = group['initial_lr'] * factor
        for retry in range(8):
            saved_rng = rng_state()
            optimizer.zero_grad(set_to_none=True)
            engine.mode(True)
            torch.cuda.reset_peak_memory_stats()
            failure, numerator = 0, 0.
            try:
                for start in range(0, len(local), micro):
                    examples = [engine.encode(r) for r in local[start:start + micro]]
                    lp = batch_logprobs(engine, examples)
                    assert lp.numel() == sum(counts[start:start + micro])
                    numerator -= lp.detach().sum().item()
                    (-lp.sum() / total).backward()
                    del examples, lp
            except torch.cuda.OutOfMemoryError:
                failure = 1
                # Drop references to a failed autograd graph before collecting.
                if 'examples' in locals():
                    del examples
                if 'lp' in locals():
                    del lp
            status = torch.tensor(failure, device='cuda:0')
            dist.all_reduce(status, op=dist.ReduceOp.MAX)
            if not status.item():
                break
            optimizer.zero_grad(set_to_none=True)
            cache.trim(0)
            gc.collect()
            torch.cuda.empty_cache()
            restore_rng(saved_rng)
            if failure:
                old = micro
                micro = max(1, micro // 2)
                safe_max_micro = min(safe_max_micro, micro)
                last_oom_step = step
                enabled_now = not checkpointing_enabled
                if enabled_now:
                    engine.peft.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
                    checkpointing_enabled = True
                emit(folder, 'oom_backoff', rank=rank, old_microbatch=old, microbatch=micro, cursor=cursor,
                     checkpointing_enabled=checkpointing_enabled)
                if old == 1 and not enabled_now:
                    raise RuntimeError('Single-example OOM under checkpointing; resume from durable checkpoint')
            dist.barrier()
        else:
            raise RuntimeError('Repeated OOM')
        if world == 1:
            norm = torch.nn.utils.clip_grad_norm_([p for _, p in engine.parameters], config.get('gradient_clip', 1.))
            assert torch.isfinite(norm)
            norm = norm.item()
        else:
            norm = allreduce_gradients(engine, config.get('gradient_clip', 1.))
        optimizer.step()
        cursor += len(selected)
        step += 1
        peak = torch.cuda.max_memory_allocated()
        transient = peak - cache.bytes
        free_mib = torch.cuda.mem_get_info()[0] / 2**20
        live_cap = min(cap_mib, free_mib + torch.cuda.memory_reserved() / 2**20 - 3072)
        if job.get('gpu_feature_cache', False):
            cache.trim(max(0, live_cap * 2**20 - transient - 3 * 2**30))
            if step % 2 == 0:
                cache.prewarm(dataset, orders, cursor + rank, rank, world, max_seconds=1)
        if step - last_oom_step >= 32:
            safe_max_micro = max_micro
        growth_margin = (2 if os.environ.get('JOB_ROLE') == 'fast' else 6) * 2**30
        if step % 4 == 0 and micro < min(max_micro, safe_max_micro) and peak + growth_margin < live_cap * 2**20:
            cache.trim(0)
            micro = min(max_micro, safe_max_micro, micro * 2)
        nll = torch.tensor(numerator, dtype=torch.float64, device='cuda:0')
        dist.all_reduce(nll)
        stats = dict(rank=rank, cursor=cursor, budget=budget, step=step, fraction=cursor/budget,
                     microbatch=micro, checkpointing_enabled=checkpointing_enabled, peak_allocated_mib=peak/2**20,
                     allocated_mib=torch.cuda.memory_allocated()/2**20,
                     reserved_mib=torch.cuda.memory_reserved()/2**20,
                     gpu_feature_cache_mib=cache.bytes/2**20, cap_mib=cap_mib,
                     updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(folder / f'memory_rank_{rank}.json', stats)
        if rank == 0:
            stats.update(token_nll=(nll/total).item(), gradient_norm=norm, batch=config['batch'],
                         presentations_per_second=(cursor-initial_cursor)/max(time.monotonic()-started, 1e-6))
            write(folder / 'progress.json', stats)
            if step == 1 or step % 8 == 0:
                emit(folder, 'train_step', **stats)
        due = torch.tensor(int(rank == 0 and (cursor >= next_checkpoint or cursor == budget or
                cursor % len(pool) == 0 or time.monotonic() - last_checkpoint >= 600 or
                step == 1)), device='cuda:0')
        dist.broadcast(due, src=0)
        if due.item():
            path = save_checkpoint()
            if step == 1:
                engine.mode(False)
                with torch.no_grad():
                    probe = engine.encode(dataset.get(pool[rank % len(pool)]))
                    before = engine.logprobs(probe).clone()
                    engine.load(path)
                    after = engine.logprobs(probe)
                    assert torch.equal(before, after), 'Checkpoint reload changed policy probabilities'
                del probe, before, after
                if rank == 0:
                    write(folder / 'checkpoint_reload_check.json', dict(passed=True,
                        saved_policy_probabilities_bit_identical=True, test_used=fitting))
            if job.get('intermediate_validation'):
                result = distributed_validation(engine, dataset, split['screen_validation'],
                    folder / 'validation_checkpoints' / path.name, rank, world)
                if rank == 0:
                    score = result['real']['local_development_score']
                    best = folder / 'best_validation.json'
                    if not best.exists() or score > read(best)['local_development_score']:
                        write(best, dict(checkpoint=str(path), cursor=cursor, local_development_score=score,
                              selected_on='fixed validation subset, never test'))
            next_checkpoint = (cursor // checkpoint_every + 1) * checkpoint_every
            last_checkpoint = time.monotonic()
    cache.trim(0)
    torch.cuda.empty_cache()
    metrics = distributed_validation(engine, dataset, split[job['validation_suite']], folder / 'validation',
                                    rank, world, job.get('controls', ['real']), evaluation_split)
    if rank == 0:
        write(folder / 'result.json', dict(status='complete', job=job, validation=metrics,
              checkpoint=read(latest_path)['checkpoint'], trainable_parameters=81209920,
              training_presentations=cursor, test_used=fitting, same_data_train_and_eval=fitting,
              seconds=time.monotonic()-started))
        emit(folder, 'job_complete', cursor=cursor, score=metrics['real']['local_development_score'])
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    job = read(sys.argv[1])
    try:
        train(job)
    except BaseException as exc:
        folder = Path(job['folder'])
        write(folder / f"failure_rank_{os.environ.get('RANK', 0)}.json", dict(error=str(exc),
              type=type(exc).__name__, utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        raise
