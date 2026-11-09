"""Stage 1: actual future tactile-token prediction with trainable ViT LoRA."""
import argparse
import collections
import datetime
import json
import math
import os
from pathlib import Path
import random
import time

from data import ROOT, SEED, read, write, key
from vision import VisionEncoder, DynamicsPredictor, adapter_state, decode, VISION_SPEC
import torch
import torch.distributed as dist
import torch.nn.functional as F

RUN = ROOT/'stage1'


def emit(event, **kwargs):
    obj = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'event': event, **kwargs}
    print(json.dumps(obj), flush=True)
    if int(os.environ.get('RANK', 0)) == 0:
        RUN.mkdir(exist_ok=True)
        with (RUN/'events.jsonl').open('a') as f:
            f.write(json.dumps(obj)+'\n')


class WindowLoader:
    def __init__(self, windows):
        self.episodes = read(ROOT/'episodes.json')
        wanted = collections.defaultdict(set)
        for w in windows:
            wanted[w['episode_id']].update(w['context_indices']+[w['target_index']])
        self.wanted = {k: sorted(v) for k, v in wanted.items()}
        self.cached_id, self.cached = None, None

    def get(self, w):
        ep = self.episodes[w['episode_id']]
        assert all(i < w['target_index'] for i in w['context_indices'])
        if ep['source']['kind'] == 'tac3d_matrix':
            images = decode(ep, w['context_indices']+[w['target_index']], history_scale_indices=w['context_indices'])
        else:
            if self.cached_id != w['episode_id']:
                self.cached = decode(ep, self.wanted[w['episode_id']])
                self.cached_id = w['episode_id']
            images = self.cached
        return [images[i] for i in w['context_indices']], images[w['target_index']]


def loss_for(encoder, predictor, loader, row):
    images, future = loader.get(row)
    z = encoder(images)
    with encoder.teacher():
        target = F.layer_norm(encoder([future])[0].float(), (3584,))
    prediction = predictor(z)
    loss = F.mse_loss(prediction, target)
    baseline = F.mse_loss(F.layer_norm(z[-1].float(), (3584,)), target)
    assert torch.isfinite(loss)
    return loss, baseline


def losses_for(encoder, predictor, loader, batch):
    # Batch independent windows through the same visual tower, retaining each
    # window's past-only scale and separate temporal context.
    decoded = [loader.get(row) for row in batch]
    lengths = [len(images) for images, _ in decoded]
    encoded = encoder([im for images, _ in decoded for im in images])
    with encoder.teacher():
        targets = F.layer_norm(encoder([future for _, future in decoded]).float(), (3584,))
    losses, offset = [], 0
    for length, target in zip(lengths, targets):
        z = encoded[offset:offset+length]
        offset += length
        loss = F.mse_loss(predictor(z), target)
        assert torch.isfinite(loss)
        losses.append(loss)
    return torch.stack(losses)


def parameters(encoder, predictor):
    return [(n, p) for n, p in encoder.named_parameters() if p.requires_grad]+[
        ('future_predictor.'+n, p) for n, p in predictor.named_parameters() if p.requires_grad]


def restore(encoder, predictor, checkpoint):
    from vision import attach
    attach(encoder.model, torch.load(checkpoint/'vision_adapter.pt', map_location='cpu', weights_only=True), trainable=True)
    predictor.load_state_dict(torch.load(checkpoint/'future_predictor.pt', map_location='cpu', weights_only=True))


def save(encoder, predictor, checkpoint, state=None):
    checkpoint.mkdir(parents=True, exist_ok=True)
    torch.save(adapter_state(encoder.modules_lora), checkpoint/'vision_adapter.pt')
    torch.save(predictor.state_dict(), checkpoint/'future_predictor.pt')
    if state is not None:
        torch.save(state, checkpoint/'trainer_state.pt')
    write(checkpoint/'manifest.json', {'complete': True, 'objective': 'future_tactile_token_prediction',
                                      'vision_spec': VISION_SPEC, 'teacher': 'fixed original ViT with LoRA disabled',
                                      'uses_force_labels': False, 'future_frame_not_in_context': True,
                                      'tac3d_display_scale_fitted_on_history_only': True})


def smoke():
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    encoder = VisionEncoder()
    predictor = DynamicsPredictor().to('cuda:0')
    encoder.model.train()
    windows = read(ROOT/'stage1_windows.json')[:32]
    loader = WindowLoader(windows)
    initial = adapter_state(encoder.modules_lora)
    probe = encoder.model.patch_embed.proj.weight.detach().flatten()[:1024].cpu().clone()
    optimizer = torch.optim.AdamW([p for _, p in parameters(encoder, predictor)], lr=1e-4)
    for row in windows:
        optimizer.zero_grad(set_to_none=True)
        loss, baseline = loss_for(encoder, predictor, loader, row)
        loss.backward()
        grads = [p.grad for _, p in parameters(encoder, predictor) if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        if any(m.B.grad is not None and m.B.grad.abs().sum().item() > 0 for m in encoder.modules_lora.values()):
            optimizer.step()
            break
    else:
        raise AssertionError('No real ViT LoRA gradient on any smoke training window')
    assert any(not torch.equal(initial[n], value) for n, value in adapter_state(encoder.modules_lora).items())
    assert torch.equal(probe, encoder.model.patch_embed.proj.weight.detach().flatten()[:1024].cpu())
    encoder.model.eval()
    predictor.eval()
    with torch.no_grad():
        expected = loss_for(encoder, predictor, loader, row)[0].cpu()
    checkpoint = RUN/'checks/smoke_checkpoint'
    save(encoder, predictor, checkpoint)
    with torch.no_grad():
        for _, p in parameters(encoder, predictor):
            p.zero_()
    restore(encoder, predictor, checkpoint)
    with torch.no_grad():
        after = loss_for(encoder, predictor, loader, row)[0].cpu()
    assert torch.equal(expected, after), (expected, after)
    write(ROOT/'checks/stage1_gpu_smoke.json', {'passed': True, 'real_vit_lora_gradient': True,
        'frozen_patch_embed_unchanged': True, 'future_input_not_in_context': True,
        'save_reload_loss_identical': True, 'smoke_loss': loss.item(), 'persistence_mse': baseline.item(),
        'trainable_parameters': sum(p.numel() for _, p in parameters(encoder, predictor)),
        'not_a_model_quality_result': True})
    emit('stage1_gpu_smoke_passed', loss=loss.item())


def train():
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.20)
    dist.init_process_group('nccl', init_method=os.environ['DIST_URL'], rank=rank, world_size=world)
    encoder = VisionEncoder()
    predictor = DynamicsPredictor().to('cuda:0')
    named = parameters(encoder, predictor)
    for _, p in named:
        dist.broadcast(p.data, src=0)
    windows = read(ROOT/'stage1_windows.json')
    val = read(ROOT/'stage1_val_windows.json')
    grouped = collections.defaultdict(list)
    for i, row in enumerate(windows):
        grouped[row['episode_id']].append(i)
    episodes = sorted(grouped)
    random.Random(SEED).shuffle(episodes)
    order = [i for eid in episodes for i in grouped[eid]]
    order_hash = key(order)
    loader = WindowLoader(windows+val)
    batch_size = 16
    optimizer = torch.optim.AdamW([p for _, p in named], lr=1e-4, weight_decay=.01, betas=(.9, .95))
    cursor = step = 0
    best, bestpath = float('inf'), None
    pointer = RUN/'latest.json'
    if pointer.exists():
        path = Path(read(pointer)['checkpoint'])
        restore(encoder, predictor, path)
        state = torch.load(path/'trainer_state.pt', map_location='cpu', weights_only=False)
        assert state['order_hash'] == order_hash
        if rank == 0 and state['world_size'] != world:
            emit('topology_migration', old_world=state['world_size'], new_world=world, restored_cursor=state['cursor'], global_batch=batch_size)
        optimizer.load_state_dict(state['optimizer'])
        for value in optimizer.state.values():
            for name, tensor in value.items():
                if isinstance(tensor, torch.Tensor):
                    value[name] = tensor.to('cuda:0')
        cursor, step, best, bestpath = state['cursor'], state['step'], state['best'], state['bestpath']
        if rank < len(state['rng']):
            torch.set_rng_state(state['rng'][rank]['torch'])
            torch.cuda.set_rng_state(state['rng'][rank]['cuda'])
        else:
            torch.manual_seed(SEED+rank)
            torch.cuda.manual_seed_all(SEED+rank)
    started = time.time()
    initial_step, initial_cursor = step, cursor
    while cursor < len(order):
        batch = order[cursor:cursor+batch_size]
        cursor += len(batch)
        step += 1
        encoder.model.train()
        predictor.train()
        optimizer.zero_grad(set_to_none=True)
        progress = cursor/len(order)
        factor = min(1., progress/.03)*(.1+.9*.5*(1+math.cos(math.pi*progress)))
        for g in optimizer.param_groups:
            g['lr'] = 1e-4*factor
        numerator = 0.
        local = [windows[index] for index in batch[rank::world]]
        if local:
            losses = losses_for(encoder, predictor, loader, local)
            numerator = losses.detach().sum().item()
            (losses.sum()/len(batch)).backward()
        flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for _, p in named])
        dist.all_reduce(flat)
        offset = 0
        for _, p in named:
            p.grad = flat[offset:offset+p.numel()].reshape_as(p).clone()
            offset += p.numel()
        norm = torch.nn.utils.clip_grad_norm_([p for _, p in named], 1.)
        assert torch.isfinite(norm)
        optimizer.step()
        if step == initial_step+1 or step % 8 == 0:
            total = torch.tensor(numerator, device='cuda:0')
            dist.all_reduce(total)
            if rank == 0:
                emit('stage1_step', step=step, windows_seen=cursor, total_windows=len(order),
                     future_token_mse=total.item()/len(batch), seconds=time.time()-started,
                     resumed_from_windows=initial_cursor,
                     cumulative_windows_per_second=(cursor-initial_cursor)/(time.time()-started),
                     visual_attention='segmented_sdpa')
        validation_due = step == 1 or cursor % 2048 == 0 or cursor == len(order)
        if validation_due:
            encoder.model.eval()
            predictor.eval()
            values = torch.zeros(3, device='cuda:0', dtype=torch.float64)
            with torch.no_grad():
                for row in val[rank::world]:
                    loss, baseline = loss_for(encoder, predictor, loader, row)
                    values += torch.tensor([loss.item(), baseline.item(), 1], device='cuda:0')
            dist.all_reduce(values)
            nll = (values[0]/values[2]).item()
            path = RUN/'checkpoints'/f'windows_{cursor:07d}'
            if nll < best:
                best, bestpath = nll, str(path)
            if rank == 0:
                write(RUN/'selected.json', {'checkpoint': bestpath, 'validation_future_token_mse': best,
                                           'validation_persistence_mse': (values[1]/values[2]).item()})
                emit('stage1_validation', windows_seen=cursor, mse=nll, best=best)
        if validation_due or cursor % 256 == 0:
            path = RUN/'checkpoints'/f'windows_{cursor:07d}'
            rng = {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state()}
            rngs = [None]*world
            dist.all_gather_object(rngs, rng)
            if rank == 0:
                save(encoder, predictor, path, {'cursor': cursor, 'step': step, 'optimizer': optimizer.state_dict(),
                    'order_hash': order_hash, 'world_size': world, 'rng': rngs, 'best': best, 'bestpath': bestpath})
                write(pointer, {'checkpoint': str(path)})
            dist.barrier()
    if rank == 0:
        write(RUN/'complete.json', {'status': 'complete', 'windows_seen': cursor,
             'recorded_frame_pool': 1000000, 'best_checkpoint': bestpath,
             'validation_future_token_mse': best, 'world_size': world,
             'trainable_parameters': sum(p.numel() for _, p in named),
             'force_conditioning_verified': False, 'test_used_for_selection': False})
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    RUN.mkdir(exist_ok=True)
    try:
        smoke() if args.smoke else train()
    except BaseException as exc:
        write(RUN/f"failure_{os.environ.get('RANK','smoke')}.json", {'error': str(exc), 'type': type(exc).__name__})
        raise
