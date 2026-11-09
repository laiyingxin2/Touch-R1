"""Resume v12 with real padded minibatches and bounded, adaptive GPU caching."""
import collections
import datetime
import gc
import math
import os
import random
import time
from pathlib import Path

import worker as original
from experiment import RUN, Dataset, digest, emit, identity, read, write


def batch_logprobs(engine, examples):
    import torch
    import torch.nn.functional as F
    lengths = [x['ids'].shape[1] for x in examples]
    width = max(lengths)
    embeddings = torch.nn.utils.rnn.pad_sequence(
        [engine.embeds(x)[0] for x in examples], batch_first=True)
    positions = torch.cat([F.pad(x['positions'], (0, width - n))
                           for x, n in zip(examples, lengths)], dim=1)
    mask = torch.arange(width, device=embeddings.device)[None, :] < torch.tensor(
        lengths, device=embeddings.device)[:, None]
    hidden = engine.base.model(inputs_embeds=embeddings, position_ids=positions,
        attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state
    selected = torch.cat([hidden[i, x['prefix'] - 1:n - 1]
                          for i, (x, n) in enumerate(zip(examples, lengths))])
    labels = torch.cat([x['ids'][0, x['prefix']:] for x in examples])
    # Project assistant tokens only; chunking bounds the large vocabulary tensor.
    values = []
    for start in range(0, len(labels), 128):
        logits = engine.base.lm_head(selected[start:start + 128]).float()
        values.append(-F.cross_entropy(logits, labels[start:start + 128], reduction='none'))
    return torch.cat(values)


class FeatureCache:
    def __init__(self, engine, cap_mib):
        self.engine = engine
        self.original = engine.feature
        self.values = collections.OrderedDict()
        self.bytes = 0
        self.cap = int(cap_mib * 2**20)
        self.limit = 0

    def trim(self, limit):
        self.limit = max(0, int(limit))
        while self.values and self.bytes > self.limit:
            _, value = self.values.popitem(last=False)
            self.bytes -= value['features'].numel() * value['features'].element_size()

    def __call__(self, media):
        import torch
        key = media['key']
        if key in self.values:
            self.values.move_to_end(key)
            return self.values[key]
        value = self.original(media)
        size = value['features'].numel() * value['features'].element_size()
        if size <= self.limit:
            self.trim(self.limit - size)
            self.limit += size
            value = dict(value, features=value['features'].to('cuda:0'))
            self.values[key] = value
            self.bytes += size
        return value

    def prewarm(self, dataset, orders, cursor, rank, world, max_seconds=5):
        """Populate spare VRAM with upcoming, already encoded training inputs."""
        import torch
        started = time.monotonic()
        position = max(cursor, getattr(self, 'prefetch_cursor', cursor))
        loaded = 0
        while self.bytes < self.limit - 8 * 2**20 and position < len(orders):
            if time.monotonic() - started > max_seconds:
                break
            row = dataset.get(orders[position])
            position += world
            for media in row['media']:
                key = media['key']
                if key in self.values:
                    continue
                path = self.engine.cache / (key + '.pt')
                # Do not decode new videos or touch NAS in a speculative prefetch.
                if not str(path).startswith('/dev/shm/') or not path.is_file():
                    continue
                value = torch.load(path, map_location='cpu', weights_only=True)
                assert value['feature_key'] == self.engine.feature_key
                size = value['features'].numel() * value['features'].element_size()
                if self.bytes + size > self.limit:
                    continue
                try:
                    value = dict(value, features=value['features'].to('cuda:0'))
                except torch.cuda.OutOfMemoryError:
                    self.trim(self.bytes - 2**30)
                    torch.cuda.empty_cache()
                    self.prefetch_cursor = position
                    return loaded
                self.values[key] = value
                self.bytes += size
                loaded += size
        self.prefetch_cursor = position
        return loaded


def fast_train(engine, plan, rank, world):
    import torch
    import torch.distributed as dist
    from parameters import learning_rate_factor
    runtime = read(RUN / 'fast_runtime.json')
    if runtime.get('disable_gradient_checkpointing'):
        engine.peft.gradient_checkpointing_disable()
    checkpointing_disabled = bool(runtime.get('disable_gradient_checkpointing'))
    config = dict(plan['config'], batch=runtime['global_batch'])
    micro = int(os.environ.get('V12_MICROBATCH', runtime['initial_microbatch']))
    cap_mib = int(os.environ['V12_CAP_MIB'])
    dataset = Dataset()
    indices = dataset.indices()
    assert len(indices) == plan['expected_training_rows']
    orders = []
    for epoch in range(plan['epochs']):
        order = list(indices)
        random.Random(config['seed'] + 991 + epoch).shuffle(order)
        orders.extend(order)
    order_hash = identity(orders)
    optimizer = original.optimizer_for(engine, config)
    cursor = step = 0
    original.initial_seed(config['seed'] + rank)
    if (RUN / 'latest.json').exists():
        path = Path(read(RUN / 'latest.json')['checkpoint'])
        engine.load(path)
        state = torch.load(path / 'trainer_state.pt', map_location='cpu', weights_only=False)
        assert state['plan_sha256'] == digest(RUN / 'plan.json')
        assert state['order_hash'] == order_hash
        optimizer.load_state_dict(state['optimizer'])
        for values in optimizer.state.values():
            for name, tensor in values.items():
                if isinstance(tensor, torch.Tensor) and name != 'step':
                    values[name] = tensor.to('cuda:0')
        cursor, step = state['cursor'], state['step']
        if state['world_size'] == world:
            original.restore_rng(state['rng'][rank])
        del state
    # Validate actual unequal-length training inputs before using batched updates.
    engine.mode(False)
    with torch.no_grad():
        probes = [engine.encode(dataset.get(i)) for i in orders[cursor:cursor + 2]]
        single = torch.cat([engine.logprobs(x).flatten() for x in probes])
        batched = batch_logprobs(engine, probes)
        error = (single - batched).abs()
        # BF16 GEMMs use different reduction orders at different batch sizes.
        # Compare the probabilities and the normalized loss, including tiny
        # likelihoods where a log-probability relative tolerance is misleading.
        probability_error = (single.exp() - batched.exp()).abs().max().item()
        nll_error = (single.mean() - batched.mean()).abs().item()
        passed = probability_error <= .025 and nll_error <= .01
        write(RUN / f'fast_validation_rank_{rank}.json', dict(passed=passed,
            max_abs_error=error.max().item(), mean_abs_error=error.mean().item(),
            max_probability_error=probability_error, mean_nll_error=nll_error,
            lengths=[x['ids'].shape[1] for x in probes], target_tokens=single.numel(),
            test_inputs_used=False))
        assert passed, 'Batched probabilities differ from single-example probabilities'
        del probes, single, batched, error
    dist.barrier()
    cache = FeatureCache(engine, cap_mib)
    engine.feature = cache
    started, initial_cursor = time.monotonic(), cursor
    if rank == 0:
        emit('fast_sft_setup', initial_cursor=cursor, budget=len(orders), world_size=world,
             global_batch=config['batch'], old_global_batch=plan['config']['batch'],
             runtime=runtime, optimizer_resumed=True, order_hash=order_hash,
             batch_change_authorized_by_user=True)
    next_checkpoint = (cursor // runtime['checkpoint_presentations'] + 1) * runtime['checkpoint_presentations']
    last_checkpoint = time.monotonic()
    stable = 0
    while cursor < len(orders):
        epoch_end = ((cursor // len(indices)) + 1) * len(indices)
        selected = orders[cursor:min(cursor + config['batch'], epoch_end)]
        local = [dataset.get(i) for i in selected[rank::world]]
        counts = [len(engine.tokenizer(r['target'] + engine.tokenizer.eos_token,
                      add_special_tokens=False)['input_ids']) for r in local]
        total = torch.tensor(sum(counts), dtype=torch.float64, device='cuda:0')
        dist.all_reduce(total)
        assert total.item() > 0
        factor = learning_rate_factor(config, cursor + len(selected), plan['schedule_horizon'])
        for group in optimizer.param_groups:
            group['lr'] = group['initial_lr'] * factor
        retry = 0
        while True:
            saved_rng = original.rng_state()
            optimizer.zero_grad(set_to_none=True)
            engine.mode(True)
            torch.cuda.reset_peak_memory_stats()
            failure = 0
            numerator = 0.
            try:
                numerator = run_local(engine, local, counts, micro, total)
            except torch.cuda.OutOfMemoryError:
                failure = 1
            status = torch.tensor(failure, device='cuda:0')
            dist.all_reduce(status, op=dist.ReduceOp.MAX)
            if not status.item():
                break
            optimizer.zero_grad(set_to_none=True)
            cache.trim(0)
            gc.collect()
            torch.cuda.empty_cache()
            original.restore_rng(saved_rng)
            if failure:
                old = micro
                micro = max(1, micro // 2)
                emit('fast_microbatch_oom_backoff', old_microbatch=old, microbatch=micro,
                     cursor=cursor, cap_mib=cap_mib)
                if old == 1:
                    if checkpointing_disabled:
                        engine.peft.gradient_checkpointing_enable(
                            gradient_checkpointing_kwargs={'use_reentrant': False})
                        checkpointing_disabled = False
                        emit('long_context_checkpointing_fallback', cursor=cursor)
                    else:
                        raise RuntimeError('Single-example OOM; durable checkpoint preserved')
            retry += 1
            stable = 0
            assert retry <= 8, 'Repeated OOM; durable checkpoint preserved'
            dist.barrier()
        norm = original.allreduce_gradients(engine, config.get('gradient_clip', 1.))
        optimizer.step()
        cursor += len(selected)
        step += 1
        stable += 1
        # Cache real features only after measured training allocations fit.
        peak = torch.cuda.max_memory_allocated()
        transient = peak - cache.bytes
        free_mib = torch.cuda.mem_get_info()[0] / 2**20
        live_cap_mib = min(cap_mib, free_mib + torch.cuda.memory_reserved() / 2**20 - 4096)
        # Preserve 3 GiB inside this process's cap for sequence-length variation.
        cache.trim(min(runtime['max_gpu_feature_cache_mib'] * 2**20,
                       live_cap_mib * 2**20 - transient - 3 * 2**30))
        base = torch.cuda.memory_allocated() - cache.bytes
        activation_per_sample = max(256 * 2**20, (transient - base) / micro)
        estimate = max(1, int((live_cap_mib * 2**20 - base - 4 * 2**30) /
                             (activation_per_sample * 1.3)))
        candidate = min(runtime['max_microbatch'], len(local), 2 ** int(math.log2(estimate)))
        if stable >= 1 and candidate > micro:
            cache.trim(0)
            micro = min(micro * 2, candidate)
            stable = 0
        elif stable >= 2:
            cache.prewarm(dataset, orders, cursor + rank, rank, world,
                          max_seconds=runtime.get('feature_prefetch_seconds', 5))
        nll = torch.tensor(numerator, dtype=torch.float64, device='cuda:0')
        dist.all_reduce(nll)
        stats = dict(rank=rank, microbatch=micro, peak_allocated_mib=peak / 2**20,
            allocated_mib=torch.cuda.memory_allocated() / 2**20,
            reserved_mib=torch.cuda.memory_reserved() / 2**20,
            gpu_feature_cache_mib=cache.bytes / 2**20, cap_mib=cap_mib)
        write(RUN / f'fast_memory_rank_{rank}.json', stats)
        if rank == 0:
            elapsed = time.monotonic() - started
            rate = (cursor - initial_cursor) / max(elapsed, 1e-6)
            emit('fast_sft_step', cursor=cursor, budget=len(orders), step=step,
                token_nll=(nll / total).item(), gradient_norm=norm,
                global_batch=len(selected), microbatch=micro,
                presentations_per_second=rate, seconds=elapsed,
                peak_allocated_mib=peak / 2**20)
            write(RUN / 'training_progress.json', dict(cursor=cursor, budget=len(orders),
                step=step, fraction=cursor / len(orders), world_size=world,
                global_batch=config['batch'], microbatch_rank0=micro,
                presentations_per_second=rate,
                updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        # Every rank makes the same checkpoint decision using rank zero's clock.
        due = torch.tensor(int(rank == 0 and (cursor >= next_checkpoint or
            cursor % len(indices) == 0 or cursor == len(orders) or
            time.monotonic() - last_checkpoint > 300)), device='cuda:0')
        dist.broadcast(due, src=0)
        if due.item():
            original.checkpoint(engine, optimizer, plan, cursor, step, order_hash, rank, world)
            if rank == 0:
                path = Path(read(RUN / 'latest.json')['checkpoint'])
                write(path / 'fast_runtime.json', runtime)
            dist.barrier()
            next_checkpoint = (cursor // runtime['checkpoint_presentations'] + 1) * runtime['checkpoint_presentations']
            last_checkpoint = time.monotonic()
    if rank == 0:
        write(RUN / 'sft_complete.json', dict(status='complete',
            checkpoint=read(RUN / 'latest.json')['checkpoint'], cursor=cursor,
            unique_training_rows=len(indices), epochs=plan['epochs'],
            plan_sha256=digest(RUN / 'plan.json'), fast_runtime=runtime,
            model_selection=plan['model_selection'], test_used_for_training_or_selection=False))
    dist.barrier()
    return True


def run_local(engine, rows, counts, micro, total):
    numerator = 0.
    for start in range(0, len(rows), micro):
        examples = [engine.encode(row) for row in rows[start:start + micro]]
        lp = batch_logprobs(engine, examples)
        assert lp.numel() == sum(counts[start:start + micro])
        numerator -= lp.detach().sum().item()
        (-lp.sum() / total).backward()
        del examples, lp
    return numerator


if __name__ == '__main__':
    original.train = fast_train
    try:
        original.main()
    except BaseException as exc:
        write(RUN / f"failure_rank_{os.environ.get('RANK', 'unknown')}.json",
              dict(error=str(exc), type=type(exc).__name__,
                   utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        raise
