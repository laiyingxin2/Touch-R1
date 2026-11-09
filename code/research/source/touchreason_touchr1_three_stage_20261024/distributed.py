"""Small-adapter distributed optimization with topology-independent batches."""
import datetime
import json
import os
import random
import time
from pathlib import Path

from data import ROOT, SEED, Dataset, read, write, key
import numpy as np
import torch
import torch.distributed as dist


def emit(folder, event, **kw):
    if int(os.environ.get('RANK', 0)):
        return
    value = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), event=event, **kw)
    print(json.dumps(value, ensure_ascii=False), flush=True)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'events.jsonl').open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False)+'\n')


def initialize():
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(0)
    dist.init_process_group('nccl', init_method=os.environ['DIST_URL'], rank=rank, world_size=world)
    return rank, world


def seed_rank(rank):
    random.seed(SEED+rank)
    np.random.seed(SEED+rank)
    torch.manual_seed(SEED+rank)
    torch.cuda.manual_seed_all(SEED+rank)


def gradients(engine):
    # Accumulation already divides by global examples/tokens; SUM is required.
    flat = torch.cat([(p.grad.float() if p.grad is not None else torch.zeros_like(p, dtype=torch.float32)).flatten()
                      for _, p in engine.parameters])
    dist.all_reduce(flat)
    offset = 0
    for _, p in engine.parameters:
        p.grad = flat[offset:offset+p.numel()].reshape_as(p).to(p.dtype).clone()
        offset += p.numel()
    norm = torch.nn.utils.clip_grad_norm_([p for _, p in engine.parameters], 1.)
    assert torch.isfinite(norm), 'Non-finite global gradient'
    return norm.item()


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state()}


def save(folder, engine, optimizer, state, rank, world):
    states = [None]*world
    dist.all_gather_object(states, rng_state())
    path = folder/'checkpoints'/f"examples_{state['cursor']:07d}"
    if rank == 0:
        engine.save(path, dict(state, optimizer=optimizer.state_dict(), rng=states, world_size=world))
        write(folder/'latest.json', {'checkpoint': str(path)})
    dist.barrier()
    return path


def resume(folder, engine, optimizer, expected_order_hash, rank, world):
    if not (folder/'latest.json').exists():
        seed_rank(rank)
        return None
    path = Path(read(folder/'latest.json')['checkpoint'])
    engine.load(path)
    state = torch.load(path/'trainer_state.pt', map_location='cpu', weights_only=False)
    assert state['order_hash'] == expected_order_hash
    optimizer.load_state_dict(state['optimizer'])
    for value in optimizer.state.values():
        for name, tensor in value.items():
            if isinstance(tensor, torch.Tensor):
                value[name] = tensor.to('cuda:0')
    if rank < len(state['rng']):
        r = state['rng'][rank]
        random.setstate(r['python'])
        np.random.set_state(r['numpy'])
        torch.set_rng_state(r['torch'])
        torch.cuda.set_rng_state(r['cuda'])
    else:
        seed_rank(rank)
    if state['world_size'] != world:
        emit(folder, 'resource_topology_changed', old_world=state['world_size'], new_world=world,
             cursor=state['cursor'], stochastic_rng_schedule_changed=True)
    return state


def dev_indices():
    """Pin a 96-question balanced development subset before any model selection."""
    dataset = Dataset()
    groups = {}
    for idx in dataset.indices(split='val'):
        row = dataset.get(idx)
        groups.setdefault((row['dataset'], row.get('native_task', row['task'])), []).append(idx)
    rng = random.Random(SEED)
    for group in groups.values():
        rng.shuffle(group)
    result = []
    while len(result) < 96:
        for name in sorted(groups):
            if groups[name] and len(result) < 96:
                result.append(groups[name].pop())
    return result


def pin_dev(rank):
    result = dev_indices()
    if rank == 0:
        p = ROOT/'development_subset.json'
        value = {'indices': result, 'sha256': key(result), 'selection': 'seeded dataset/task round-robin',
                 'test_used': False}
        if p.exists():
            assert read(p) == value
        else:
            write(p, value)
    dist.barrier()
    return result
