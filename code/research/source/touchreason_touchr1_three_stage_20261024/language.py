"""Stages 2/3: full weak-source SFT and Touch-R1 distribution-grounded GRPO."""
import argparse
import math
import os
from pathlib import Path
import random
import time

from data import ROOT, SEED, Dataset, read, write, key
from model import Engine
from distributed import initialize, seed_rank, gradients, save, resume, emit, pin_dev
from scoring import development
from rewarding import reward, token_segments, segment_mean
import torch
import torch.distributed as dist


def make_optimizer(engine, config):
    # The adapter and LoRA are FP32; the frozen 7B backbone stays BF16.
    assert all(p.dtype == torch.float32 for _, p in engine.parameters)
    return torch.optim.AdamW(engine.groups(config['learning_rate'], config['learning_rate']*config.get('adapter_lr_ratio', 1)),
                             weight_decay=config.get('weight_decay', .01), betas=(.9, .95))


def sft(engine, dataset, plan, rank, world, dev):
    folder = ROOT/'sft'
    config = plan['sft']
    indices = dataset.indices(stage='sft')
    random.Random(SEED).shuffle(indices)
    order_hash = key(indices)
    optimizer = make_optimizer(engine, config)
    previous = resume(folder, engine, optimizer, order_hash, rank, world)
    cursor = previous['cursor'] if previous else 0
    step = previous['step'] if previous else 0
    best = previous['best'] if previous else float('inf')
    bestpath = previous['bestpath'] if previous else None
    started = time.time()
    while cursor < len(indices):
        selected = indices[cursor:cursor+config['global_batch']]
        local = [dataset.get(i) for i in selected[rank::world]]
        # Count exact standalone target tokens, then assert their boundary below.
        counts = [len(engine.tokenizer(r['target']+engine.tokenizer.eos_token, add_special_tokens=False)['input_ids']) for r in local]
        total_tokens = torch.tensor(sum(counts), device='cuda:0', dtype=torch.float64)
        dist.all_reduce(total_tokens)
        assert total_tokens > 0
        cursor += len(selected)
        step += 1
        progress = cursor/len(indices)
        warmup = config['warmup_fraction']
        factor = progress/warmup if progress < warmup else .5*(1+math.cos(math.pi*(progress-warmup)/(1-warmup)))
        for group in optimizer.param_groups:
            group['lr'] = group['initial_lr']*factor
        engine.mode(True)
        optimizer.zero_grad(set_to_none=True)
        local_nll = 0.
        for row, count in zip(local, counts):
            x = engine.encode(row)
            expected = engine.tokenizer(row['target']+engine.tokenizer.eos_token, add_special_tokens=False)['input_ids']
            assert x['ids'][0, x['prefix']:].tolist() == expected, 'Assistant-only loss token alignment failed'
            lp = engine.logprobs(x).flatten()
            assert lp.numel() == count
            local_nll -= lp.detach().sum().item()
            (-lp.sum()/total_tokens).backward()
        norm = gradients(engine)
        optimizer.step()
        if step == 1 or step % 16 == 0:
            numerator = torch.tensor(local_nll, device='cuda:0', dtype=torch.float64)
            dist.all_reduce(numerator)
            emit(folder, 'sft_step', step=step, examples_seen=cursor, total_examples=len(indices),
                 token_nll=(numerator/total_tokens).item(), gradient_norm=norm, seconds=time.time()-started)
        if step == 1 or cursor % config['checkpoint_examples'] == 0 or cursor == len(indices):
            metric = development(engine, dev, rank, world, folder, f'dev_{cursor:07d}')
            checkpoint = folder/'checkpoints'/f'examples_{cursor:07d}'
            if metric['assistant_token_nll'] < best:
                best, bestpath = metric['assistant_token_nll'], str(checkpoint)
            save(folder, engine, optimizer, {'cursor': cursor, 'step': step, 'order_hash': order_hash,
                 'best': best, 'bestpath': bestpath}, rank, world)
            if rank == 0:
                write(folder/'selected.json', {'checkpoint': bestpath, 'selection': 'minimum development assistant-token NLL',
                                               'development_nll': best, 'test_used': False})
            emit(folder, 'sft_development', examples_seen=cursor, **metric)
    engine.load(bestpath)
    metric = development(engine, dataset.indices(split='val'), rank, world, folder, 'full_validation', generate=True)
    if rank == 0:
        write(folder/'complete.json', {'status': 'complete', 'examples_seen': cursor, 'selected_checkpoint': bestpath,
              'validation': metric, 'world_size': world, 'weak_supervision_experiment': True})


def sample_batch(engine, dataset, indices, rank, world, config):
    """Reference probabilities are computed before any update graph exists."""
    trajectories, rewards, varying = [], [], 0
    group_size = config['group_size']
    for idx in indices[rank::world]:
        row = dataset.get(idx)
        prompt = engine.encode(row, target=False)
        completions = engine.rollouts(prompt, count=group_size, microbatch=config['rollout_microbatch'])
        outputs = [engine.tokenizer.decode(tokens, skip_special_tokens=True) for tokens in completions]
        scores = [reward(row, text) for text in outputs]
        special = set(engine.tokenizer.all_special_ids)-{engine.tokenizer.eos_token_id}
        for score, tokens in zip(scores, completions):
            if any(t in special for t in tokens):
                score.update(total=-.06, annotation_accuracy_reward=-.1, exact_answer=False,
                             format_reward=False, invalid_special_token=True)
        values = torch.tensor([s['total'] for s in scores], device='cuda:0')
        std = values.std(unbiased=False)
        varying += int(std.item() > 0)
        advantages = (values-values.mean())/(std+1e-4)
        for tokens, text, score, advantage in zip(completions, outputs, scores, advantages):
            x = engine.append_completion(prompt, tokens)
            with torch.no_grad():
                old = engine.logprobs(x).flatten().detach()
            positions = torch.randperm(len(tokens), device='cuda:0')[:min(8, len(tokens))].sort().values
            stream = random.randrange(len(x['stream_slices']))
            a, b = x['stream_slices'][stream]
            noise = torch.randn_like(x['features'][a:b])
            trajectories.append({'x': x, 'old': old, 'advantage': advantage.detach(),
                 'groups': token_segments(engine.tokenizer, tokens), 'positions': positions,
                 'stream': stream, 'noise': noise, 'id': row['id'], 'tokens': tokens, 'text': text, 'reward': score})
            rewards.append(score['total'])
    return trajectories, rewards, varying


def grpo_loss(engine, trajectory, config, parity=False):
    x = trajectory['x']
    lp, intact = engine.policy(x, trajectory['positions'])
    if parity:
        assert torch.allclose(lp.detach(), trajectory['old'], atol=.08, rtol=.01), 'Old/policy logprob parity failed'
    ratio = (lp-trajectory['old']).exp()
    advantage = trajectory['advantage']
    lo, hi = config['clip_bounds']
    surrogate = torch.minimum(ratio*advantage, ratio.clamp(lo, hi)*advantage)
    delta = trajectory['reference']-lp
    kl = delta.exp()-delta-1
    policy_loss = segment_mean(-surrogate+config['kl_beta']*kl, trajectory['groups'])
    noisy = dict(x)
    noisy['features'] = x['features'].clone()
    a, b = x['stream_slices'][trajectory['stream']]
    noisy['features'][a:b] = trajectory['noise']
    perturbed = engine.noisy_logdist(noisy, trajectory['positions'])
    # Positive KL is MAXIMIZED in the paper objective; gradient loss subtracts it.
    grounding = (intact.exp()*(intact-perturbed)).sum(-1).mean()
    loss = policy_loss-config['grounding_gamma']*grounding
    assert torch.isfinite(loss) and torch.isfinite(grounding)
    return loss, grounding.detach(), segment_mean(kl.detach(), trajectory['groups'])


def grpo(engine, dataset, plan, rank, world, dev):
    folder = ROOT/'grpo'
    config = plan['grpo']
    selected_sft = Path(read(ROOT/'sft/selected.json')['checkpoint'])
    engine.load(selected_sft)
    reference = {n: p.detach().clone() for n, p in engine.parameters}
    indices = dataset.indices(stage='grpo')
    random.Random(SEED+1).shuffle(indices)
    order_hash = key(indices)
    optimizer = make_optimizer(engine, config)
    previous = resume(folder, engine, optimizer, order_hash, rank, world)
    cursor = previous['cursor'] if previous else 0
    step = previous['step'] if previous else 0
    if previous:
        best, bestpath = previous['best'], previous['bestpath']
    else:
        baseline = development(engine, dev, rank, world, folder, 'sft_reference_development', generate=True)
        best, bestpath = baseline['macro_exact_accuracy'], str(selected_sft)
    started = time.time()
    while cursor < len(indices):
        batch = indices[cursor:cursor+config['global_prompts']]
        trajectories, rewards, varying = sample_batch(engine, dataset, batch, rank, world, config)
        engine.mode(False)
        with torch.no_grad(), engine.reference(reference):
            for tr in trajectories:
                tr['reference'] = engine.logprobs(tr['x']).flatten().detach()
        cursor += len(batch)
        count = len(batch)*config['group_size']
        for iteration in range(config['updates_per_group']):
            step += 1
            for group in optimizer.param_groups:
                group['lr'] = group['initial_lr']*min(1., step/config['warmup_steps'])
            optimizer.zero_grad(set_to_none=True)
            engine.mode(True, dropout=False)
            grounding_sum = kl_sum = 0.
            for tr in trajectories:
                loss, grounding, kl = grpo_loss(engine, tr, config, parity=iteration == 0)
                grounding_sum += grounding.item()
                kl_sum += kl.item()
                (loss/count).backward()
            norm = gradients(engine)
            optimizer.step()
        summary = torch.tensor([sum(rewards), len(rewards), varying, grounding_sum, kl_sum], device='cuda:0', dtype=torch.float64)
        dist.all_reduce(summary)
        if rank == 0:
            with (folder/'rollouts_rank0.jsonl').open('a') as f:
                import json
                for tr in trajectories:
                    f.write(json.dumps({'prompts_seen': cursor, 'id': tr['id'], 'token_ids': tr['tokens'],
                         'response': tr['text'], 'reward': tr['reward'], 'perturbed_stream': tr['stream'],
                         'grounding_positions': tr['positions'].tolist()}, ensure_ascii=False)+'\n')
        emit(folder, 'grpo_step', optimizer_step=step, prompts_seen=cursor, total_prompts=len(indices),
             reward_mean=(summary[0]/summary[1]).item(), varying_groups=int(summary[2].item()),
             zero_variance_groups=len(batch)-int(summary[2].item()),
             grounding_kl=(summary[3]/summary[1]).item(), reference_kl=(summary[4]/summary[1]).item(),
             gradient_norm=norm, seconds=time.time()-started, dynamic_advantage_mask=True,
             physical_cross_sensor_reward=0, weak_supervision_experiment=True)
        del trajectories
        due = cursor % config['validation_prompts'] == 0 or cursor == len(indices)
        if due:
            metric = development(engine, dev, rank, world, folder, f'dev_{cursor:07d}', generate=True)
            if metric['macro_exact_accuracy'] > best:
                best = metric['macro_exact_accuracy']
                bestpath = str(folder/'checkpoints'/f'examples_{cursor:07d}')
        if due or cursor % config['checkpoint_prompts'] == 0:
            save(folder, engine, optimizer, {'cursor': cursor, 'step': step, 'order_hash': order_hash,
                'best': best, 'bestpath': bestpath, 'fixed_reference': str(selected_sft)}, rank, world)
            if rank == 0:
                write(folder/'selected.json', {'checkpoint': bestpath, 'development_macro_exact_accuracy': best,
                     'selection': 'best fixed development subset; SFT retained if GRPO does not improve',
                     'is_sft_reference': bestpath == str(selected_sft), 'test_used': False})
    lastpath = read(folder/'latest.json')['checkpoint']
    # Always retain and evaluate the actual three-stage model, even if SFT wins dev.
    final_metric = development(engine, dataset.indices(split='val'), rank, world, folder, 'last_grpo_full_validation', generate=True)
    engine.load(bestpath)
    metric = development(engine, dataset.indices(split='val'), rank, world, folder, 'selected_full_validation', generate=True)
    if rank == 0:
        write(folder/'complete.json', {'status': 'complete', 'unique_prompts_seen': cursor, 'rollouts': cursor*config['group_size'],
              'selected_checkpoint': bestpath, 'actual_three_stage_checkpoint': lastpath,
              'selected_validation': metric, 'last_grpo_validation': final_metric,
              'weak_supervision_experiment': True, 'test_used_for_selection': False})


def main(stage):
    from training_policy import require_full_training_allowed
    require_full_training_allowed(stage)
    rank, world = initialize()
    folder = ROOT/stage
    folder.mkdir(parents=True, exist_ok=True)
    engine = Engine()
    for _, p in engine.parameters:
        dist.broadcast(p.data, src=0)
    seed_rank(rank)
    plan = read(ROOT/'plan.json')
    dataset = Dataset()
    dev = pin_dev(rank)
    (sft if stage == 'sft' else grpo)(engine, dataset, plan, rank, world, dev)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['sft', 'grpo'], required=True)
    args = parser.parse_args()
    try:
        main(args.stage)
    except BaseException as exc:
        write(ROOT/args.stage/f"failure_{os.environ.get('RANK', 'unknown')}.json", {'error': str(exc), 'type': type(exc).__name__})
        raise
