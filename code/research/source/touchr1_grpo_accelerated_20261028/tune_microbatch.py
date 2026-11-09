"""Calibrate real training throughput within each GPU's actual memory budget."""
import gc,time
from pathlib import Path

def calibrate(engine,optimizer,trajectories,plan,cache,cap,rank,root):
    import torch
    from common import write,emit
    from worker import rng_state,restore_rng
    from grpo_ops import objective_batch
    previous=Path(root)/f'microbatch_calibration_rank_{rank}.json'
    candidates=[4,8,10,12,16,24,32,48]
    if previous.exists():
        record=__import__('json').loads(previous.read_text())
        if record.get('gpu')==torch.cuda.get_device_name(0) and record.get('cap_mib',cap)<=cap+1024:
            if record.get('max_sequence_length'):return record['chosen']
            candidates=[record['chosen'],max(1,record['chosen']//2),2]
    saved=rng_state();cache.trim(0);results=[]
    # Each trial uses the longest current training sequences and the same objective.
    # No optimizer step is taken; discard all trial gradients and restore RNG.
    for micro in candidates:
        if micro>len(trajectories):continue
        optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
        engine.mode(True,dropout=False);restore_rng(saved)
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.monotonic()
        loss=None
        try:
            loss,_=objective_batch(engine,trajectories[-micro:],plan,parity=True)
            (loss/(micro*int(__import__('os').environ['WORLD_SIZE']))).backward();torch.cuda.synchronize()
            seconds=time.monotonic()-started;peak=torch.cuda.max_memory_allocated()/2**20
            result=dict(microbatch=micro,seconds=seconds,peak_mib=peak,
                        trajectories_per_second=micro/seconds,accepted=peak+3072<cap)
            results.append(result)
            if previous.exists() and result['accepted']:break
            if not result['accepted']:break
        except torch.cuda.OutOfMemoryError:
            results.append(dict(microbatch=micro,oom=True))
            if not previous.exists():break
        finally:
            del loss
            optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
    restore_rng(saved)
    valid=[v for v in results if v.get('accepted')]
    chosen=max(valid,key=lambda v:v['trajectories_per_second'])['microbatch'] if valid else 2
    write(Path(root)/f'microbatch_calibration_rank_{rank}.json',dict(rank=rank,chosen=chosen,
          cap_mib=cap,gpu=torch.cuda.get_device_name(0),trials=results,optimizer_steps_taken=0,
          max_sequence_length=max(tr['x']['ids'].shape[1] for tr in trajectories)))
    emit(Path(root),'microbatch_calibrated',rank=rank,chosen=chosen,trials=results)
    return chosen
