"""Sixteen-rank Touch-R1 GRPO from a frozen public continual-SFT reference."""
import datetime
import gc
import json
import math
import os
from pathlib import Path
import random
import time
from common import ROOT,Dataset,read,write,digest,identity,emit,make_engine,save_model
from worker import initial_seed,rng_state,restore_rng,optimizer_for,allreduce_gradients
from fast_worker import FeatureCache
from rewarding import reward,token_segments
from grpo_ops import policy_batch,rollout_batch,objective_batch
from evaluation_helpers import distributed_validation

def main():
    import torch
    import torch.distributed as dist
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);assert world==16
    torch.cuda.set_device(0)
    dist.init_process_group('nccl',init_method=os.environ['DIST_URL'],rank=rank,world_size=world,
                            timeout=datetime.timedelta(minutes=60))
    device_name=torch.cuda.get_device_name(0)
    assert 'A800' in device_name, ('Expected A800', device_name)
    topology=[None]*world
    import socket
    dist.all_gather_object(topology, dict(rank=rank,host=socket.gethostname(),gpu=device_name))
    assert len({item['host'] for item in topology})==2, topology
    probe=torch.tensor(float(rank+1),device='cuda:0');dist.all_reduce(probe)
    assert probe.item()==136.0, probe.item()
    if rank==0:write(ROOT/'distributed_preflight.json',dict(passed=True,world_size=world,
        nodes=2,gpus_per_node=8,nccl_sum=probe.item(),topology=topology))
    dist.barrier()
    plan=read(ROOT/'plan.json');cfg=plan['config'];cap=int(os.environ['JOB_CAP_MIB'])
    assert read(ROOT/'training_policy.json')['grpo_enabled']
    initial_seed(plan['seed']);engine=make_engine(cfg,plan['sft_reference'])
    # Only small adaptation tensors are duplicated; frozen 7B and vision stay shared.
    reference={n:p.detach().clone() for n,p in engine.parameters}
    for _,p in engine.parameters:dist.broadcast(p.data,src=0)
    ds=Dataset();split=read(ROOT/'splits.json')
    orders=list(split['native_grpo'])+list(split['public_train'])
    random.Random(plan['seed']).shuffle(orders);order_hash=identity(orders)
    optimizer=optimizer_for(engine,cfg);cursor=step=0
    initial_seed(plan['seed']+rank)
    if (ROOT/'latest.json').exists():
        latest=read(ROOT/'latest.json');engine.load(latest['checkpoint'])
        state=torch.load(Path(latest['checkpoint'])/'trainer_state.pt',map_location='cpu',weights_only=False)
        assert state['plan_hash']==digest(ROOT/'plan.json') and state['order_hash']==order_hash
        optimizer.load_state_dict(state['optimizer'])
        for values in optimizer.state.values():
            for k,v in values.items():
                if torch.is_tensor(v) and k!='step':values[k]=v.to('cuda:0')
        cursor,step=state['cursor'],state['step'];restore_rng(state['rng'][rank]);del state
    cache=FeatureCache(engine,cap);engine.feature=cache
    micro=int(os.environ.get('TRAIN_MICROBATCH','2'));rollout_micro=int(os.environ.get('ROLLOUT_MICROBATCH','32'))
    safe_micro=plan['max_train_microbatch'];last_oom=-100;started=time.monotonic();initial_cursor=cursor
    if rank==0:emit(ROOT,'grpo_started',cursor=cursor,optimizer_step=step,world_size=world,
        trainable_parameters=sum(p.numel() for _,p in engine.parameters),reference=plan['sft_reference'],config=cfg)
    last_checkpoint=time.monotonic();next_checkpoint=(cursor//plan['checkpoint_prompts']+1)*plan['checkpoint_prompts']

    def checkpoint():
        states=[None]*world;dist.all_gather_object(states,rng_state())
        if rank==0:
            path=ROOT/'checkpoints'/f'prompts_{cursor:07d}_step_{step:07d}';partial=path.with_name(path.name+'.partial')
            assert not path.exists() and not partial.exists()
            save_model(engine,partial,cfg,dict(cursor=cursor,step=step,order_hash=order_hash,
                plan_hash=digest(ROOT/'plan.json'),optimizer=optimizer.state_dict(),rng=states,world_size=world,
                fixed_reference=plan['sft_reference']))
            partial.rename(path);write(ROOT/'latest.json',dict(checkpoint=str(path),cursor=cursor,step=step))
            emit(ROOT,'checkpoint_saved',checkpoint=str(path),cursor=cursor,optimizer_step=step)
            import shutil
            best_path=Path(read(ROOT/'best_validation.json')['checkpoint']) if (ROOT/'best_validation.json').exists() else None
            for old in sorted((ROOT/'checkpoints').glob('prompts_*_step_*'))[:-3]:
                if old==best_path:continue
                if old.is_dir() and not old.name.endswith('.partial'):shutil.rmtree(old)
        dist.barrier()

    while cursor<len(orders):
        prompts_per_batch=plan['global_prompts'] if step<4 else plan['steady_global_prompts']
        selected=orders[cursor:min(cursor+prompts_per_batch,len(orders))]
        local=[ds.get(i) for i in selected[rank::world]]
        assert all(r['split']=='train' and r['stage']=='grpo' for r in local)
        engine.mode(False)
        with torch.no_grad():prompts=[engine.encode(r,target=False) for r in local]
        for attempt in range(7):
            saved=rng_state()
            try:
                groups=rollout_batch(engine,prompts,plan['group_size'],rollout_micro,plan['max_new_tokens']);break
            except torch.cuda.OutOfMemoryError:
                cache.trim(0);gc.collect();torch.cuda.empty_cache();restore_rng(saved)
                if rollout_micro==1:raise
                rollout_micro=max(1,rollout_micro//2)
                emit(ROOT,'rollout_oom_backoff',rank=rank,microbatch=rollout_micro,cursor=cursor)
        else:raise RuntimeError('Rollout OOM retries exhausted')
        emit(ROOT,'grpo_rollout_complete',rank=rank,cursor=cursor,
             prompts=len(local),completions=sum(len(g) for g in groups),rollout_microbatch=rollout_micro)
        trajectories=[];all_rewards=[];all_exact=[];varying=0;rollout_log=[]
        forbidden=set(engine.tokenizer.all_special_ids)-{engine.tokenizer.eos_token_id}
        for row,prompt,group in zip(local,prompts,groups):
            scores=[]
            for item in group:
                text=engine.tokenizer.decode(item['tokens'],skip_special_tokens=True)
                score=reward(row,text)
                if any(t in forbidden for t in item['tokens']):
                    score.update(total=-.06,annotation_accuracy_reward=-.1,exact_answer=False,format_reward=False,invalid_special_token=True)
                item.update(text=text,reward=score);scores.append(score['total'])
                all_rewards.append(score['total']);all_exact.append(score['exact_answer'])
                rollout_log.append(dict(cursor=cursor,id=row['id'],response=text,tokens=item['tokens'],
                    sampled_logprobs=item['sampled_logprobs'],reward=score,truncated=item['truncated']))
            values=torch.tensor(scores,device='cuda:0');std=values.std(unbiased=False)
            varying+=int(std.item()>1e-8)
            # Constant reward groups have zero GRPO advantages. Keep the
            # independent tactile-grounding term, as in language.grpo_loss.
            advantages=(values-values.mean())/(std+1e-4)
            for item,advantage in zip(group,advantages):
                x=engine.append_completion(prompt,item['tokens']);positions=torch.randperm(len(item['tokens']),device='cuda:0')[:min(8,len(item['tokens']))].sort().values
                stream=random.randrange(len(x['stream_slices']));a,b=x['stream_slices'][stream]
                trajectories.append(dict(x=x,positions=positions,stream=stream,noise=torch.randn_like(x['features'][a:b]),
                    advantage=advantage.detach(),segments=token_segments(engine.tokenizer,item['tokens']),
                    sampled_lp=torch.tensor(item['sampled_logprobs'],device='cuda:0'),id=row['id']))
        # Sort by context length to reduce padding without changing prompt groups.
        trajectories.sort(key=lambda tr:tr['x']['ids'].shape[1])
        with (ROOT/f'rollouts_rank_{rank}.jsonl').open('a') as out:
            for item in rollout_log:out.write(json.dumps(item,ensure_ascii=False)+'\n')
        total=torch.tensor(len(trajectories),device='cuda:0',dtype=torch.float64);dist.all_reduce(total)
        # Reward-constant groups contribute no surrogate policy gradient.
        # The label-free tactile grounding objective still applies to them.
        if total.item():
            engine.mode(False)
            with torch.no_grad():
                reference_micro=min(micro,4)
                for start in range(0,len(trajectories),reference_micro):
                    group=trajectories[start:start+reference_micro];old,_=policy_batch(engine,[tr['x'] for tr in group])
                    for tr,lp in zip(group,old):tr['old']=lp.detach()
                if step==0 and trajectories:
                    error=torch.cat([(tr['old']-tr['sampled_lp']).abs() for tr in trajectories])
                    probability=torch.cat([(tr['old'].exp()-tr['sampled_lp'].exp()).abs() for tr in trajectories])
                    assert error.mean().item()<=.04 and probability.max().item()<=.04,('KV/sampled probability mismatch',error.mean().item(),probability.max().item())
                    write(ROOT/f'rollout_parity_rank_{rank}.json',dict(passed=True,mean_logprob_error=error.mean().item(),max_probability_error=probability.max().item(),n_tokens=error.numel()))
                with engine.reference(reference):
                    for start in range(0,len(trajectories),reference_micro):
                        group=trajectories[start:start+reference_micro];refs,_=policy_batch(engine,[tr['x'] for tr in group])
                        for tr,lp in zip(group,refs):tr['reference']=lp.detach()
            for iteration in range(plan['updates_per_group']):
                warm=min(1.,(step+1)/plan['warmup_steps'])
                decay=max(0.,1-(cursor/max(1,len(orders))-plan['warmup_fraction'])/(1-plan['warmup_fraction']))
                for group in optimizer.param_groups:group['lr']=group['initial_lr']*warm*min(1.,decay)
                for retry in range(7):
                    state_rng=rng_state();optimizer.zero_grad(set_to_none=True);engine.mode(True,dropout=False)
                    torch.cuda.reset_peak_memory_stats();bad=0;stats=[];loss_sum=0.
                    try:
                        for start in range(0,len(trajectories),micro):
                            loss,chunk_stats=objective_batch(engine,trajectories[start:start+micro],plan,parity=iteration==0)
                            loss_sum+=loss.detach().item();stats.extend(chunk_stats);(loss/total).backward();del loss
                    except torch.cuda.OutOfMemoryError:
                        bad=1
                        if 'loss' in locals():del loss
                    failed=torch.tensor(bad,device='cuda:0');dist.all_reduce(failed,op=dist.ReduceOp.MAX)
                    if not failed.item():break
                    optimizer.zero_grad(set_to_none=True);cache.trim(0);gc.collect();torch.cuda.empty_cache();restore_rng(state_rng)
                    if bad:
                        if micro==1:raise RuntimeError('GRPO single-trajectory OOM under checkpointing')
                        micro=max(1,micro//2);safe_micro=micro;last_oom=step
                        emit(ROOT,'update_oom_backoff',rank=rank,microbatch=micro,cursor=cursor)
                    dist.barrier()
                else:raise RuntimeError('GRPO update OOM retries exhausted')
                norm=allreduce_gradients(engine,plan['gradient_clip']);optimizer.step();step+=1
                if rank==0 and step==1:
                    write(ROOT/'training_accepted.json',dict(passed=True,optimizer_step=step,
                        gradient_norm=norm,world_size=world,actual_gpu=device_name,
                        rollout_probability_check_required=True,test_used=False))
                if rank==0:emit(ROOT,'grpo_optimizer_update',optimizer_step=step,
                    cursor_before_group=cursor,gradient_norm=norm,lr=optimizer.param_groups[0]['lr'])
                peak=torch.cuda.max_memory_allocated()/2**20
                memory=dict(rank=rank,cursor=cursor+len(selected),optimizer_step=step,
                    train_microbatch=micro,rollout_microbatch=rollout_micro,peak_allocated_mib=peak,
                    allocated_mib=torch.cuda.memory_allocated()/2**20,reserved_mib=torch.cuda.memory_reserved()/2**20,cap_mib=cap,
                    feature_cache_mib=cache.bytes/2**20,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                write(ROOT/f'memory_rank_{rank}.json',memory)
                if step-last_oom>=32:safe_micro=plan['max_train_microbatch']
                if step%2==0 and micro<safe_micro and peak+6144<cap:
                    cache.trim(0);micro=min(safe_micro,micro*2)
            summary=torch.tensor([sum(all_rewards),len(all_rewards),sum(all_exact),varying,
                sum(s[0] for s in stats),sum(s[1] for s in stats),loss_sum],device='cuda:0',dtype=torch.float64)
        else:
            summary=torch.tensor([sum(all_rewards),len(all_rewards),sum(all_exact),varying,0,0,0],device='cuda:0',dtype=torch.float64)
        dist.all_reduce(summary);cursor+=len(selected)
        if rank==0:
            progress=dict(cursor=cursor,budget=len(orders),optimizer_step=step,fraction=cursor/len(orders),
                sampled_reward=(summary[0]/summary[1]).item(),sampled_annotation_accuracy=(summary[2]/summary[1]).item(),
                varying_groups=int(summary[3].item()),zero_variance_groups=len(selected)-int(summary[3].item()),
                optimized_rollouts=int(total.item()),grounding_kl=(summary[4]/max(1,total.item())).item(),
                reference_kl=(summary[5]/max(1,total.item())).item(),gradient_norm=norm if total.item() else None,
                prompts_per_second=(cursor-initial_cursor)/max(.001,time.monotonic()-started),
                updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),test_used=False,
                cross_sensor_reward_active=False,weak_supervision_experiment=True)
            write(ROOT/'progress.json',progress);emit(ROOT,'grpo_step',**progress)
        del trajectories,groups,prompts
        # Useful cached future tactile features fill capacity left by train graphs.
        if total.item():
            transient=torch.cuda.max_memory_allocated()-cache.bytes
            live_cap=min(cap,torch.cuda.mem_get_info()[0]/2**20+torch.cuda.memory_reserved()/2**20-4096)
            cache.trim(max(0,live_cap*2**20-transient-4*2**30))
            cache.prewarm(ds,orders,cursor+rank,rank,world,max_seconds=2)
        due=torch.tensor(int(rank==0 and ((step and not (ROOT/'latest.json').exists()) or cursor>=next_checkpoint or
            cursor==len(orders) or time.monotonic()-last_checkpoint>600)),device='cuda:0')
        dist.broadcast(due,src=0)
        if due.item():
            checkpoint();last_checkpoint=time.monotonic();next_checkpoint=(cursor//plan['checkpoint_prompts']+1)*plan['checkpoint_prompts']
            cache.trim(0);torch.cuda.empty_cache()
            # Compare GRPO against the fixed SFT weights using the SAME decoder
            # precision. The older BF16 validation is kept as historical data.
            baseline_due=torch.tensor(int(rank==0 and not (ROOT/'reference_validation_current_precision/metrics.json').exists()),device='cuda:0')
            dist.broadcast(baseline_due,src=0)
            if baseline_due.item():
                with engine.reference(reference):
                    baseline=distributed_validation(engine,ds,split['screen_validation'],
                        ROOT/'reference_validation_current_precision',rank,world)
                if rank==0:
                    write(ROOT/'sft_reference_validation_current_precision.json',dict(
                        checkpoint=plan['sft_reference'],config=cfg,metrics=baseline,test_used=False))
                    best=read(ROOT/'best_validation.json')
                    if best.get('is_sft_reference'):
                        best.update(local_development_score=baseline['real']['local_development_score'],
                            decoder_precision=cfg.get('decoder_dtype','bfloat16'))
                        write(ROOT/'best_validation.json',best)
            metrics=distributed_validation(engine,ds,split['screen_validation'],
                ROOT/'validation_checkpoints'/Path(read(ROOT/'latest.json')['checkpoint']).name,rank,world)
            if rank==0:
                score=metrics['real']['local_development_score'];best=read(ROOT/'best_validation.json')
                if score>best['local_development_score']:
                    write(ROOT/'best_validation.json',dict(checkpoint=read(ROOT/'latest.json')['checkpoint'],
                        local_development_score=score,selected_on='fixed independent validation; never test'))
    if not (ROOT/'latest.json').exists() or read(ROOT/'latest.json')['cursor']!=cursor:
        checkpoint()
    cache.trim(0);torch.cuda.empty_cache()
    full=distributed_validation(engine,ds,split['full_validation'],ROOT/'final_validation',rank,world)
    if rank==0:write(ROOT/'complete.json',dict(status='complete',latest=read(ROOT/'latest.json'),
        validation=full,best_validation=read(ROOT/'best_validation.json'),test_used=False))
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        write(ROOT/f"failure_rank_{os.environ.get('RANK',0)}.json",dict(error=str(exc),type=type(exc).__name__,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()));raise
