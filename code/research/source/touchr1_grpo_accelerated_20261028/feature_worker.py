"""Precompute frozen, parity-checked training visual features on isolated H20s."""
import collections,concurrent.futures,datetime,gc,json,os,time,types
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image,ImageOps
from common import ROOT,V12,PARENT,read,write,digest,emit
from experiment import resolved_episodes
from vision import VisionEncoder,attach
from model import Engine

@torch.no_grad()
def main():
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.9)
    assert 'H20' in torch.cuda.get_device_name(0)
    import cv2
    cv2.setNumThreads(1);torch.set_num_threads(4)
    plan=read(V12/'plan.json');reference=read(Path(read(ROOT/'plan.json')['sft_reference'])/'manifest.json')
    assert digest(Path(plan['stage1_checkpoint'])/'vision_adapter.pt')==reference['stage1_vision_sha256']
    visual=VisionEncoder()
    state=torch.load(Path(plan['stage1_checkpoint'])/'vision_adapter.pt',map_location='cpu',weights_only=True)
    attach(visual.model,state,trainable=False);visual.model.eval();visual.model.requires_grad_(False)
    engine=Engine.__new__(Engine);engine.base=types.SimpleNamespace(visual=visual.model)
    engine.processor=visual.processor;engine.episodes=resolved_episodes();engine.mismatch={}
    aliases=read(ROOT/'verified_grpo_source_aliases.json')
    for old,alias in aliases.items():
        ep=engine.episodes[alias['episode_id']];assert ep['source']['path']==old
        ep['source']=dict(ep['source'],path=alias['path'])
        if ep['source']['kind']=='png_sequence':ep['verified_alias_frame_sha256']=alias['frame_sha256']
        else:ep['verified_alias_video_sha256']=alias['video_sha256']
    engine.feature_key=reference['feature_key'];engine.feature_memory=collections.OrderedDict()
    dest=ROOT/'features'/engine.feature_key;dest.mkdir(parents=True,exist_ok=True)
    old=[V12/'features'/engine.feature_key,PARENT/'features'/engine.feature_key]
    tasks=[json.loads(s)['media'] for s in (ROOT/'feature_tasks.jsonl').read_text().splitlines()]
    checked=set()
    def images(media):
        if media['kind']=='raw':
            ep=engine.episodes[media['episode_id']];source=ep['source'];path=Path(source['path'])
            if 'verified_alias_video_sha256' in ep and str(path) not in checked:
                assert digest(path)==ep['verified_alias_video_sha256'];checked.add(str(path))
            if 'verified_alias_frame_sha256' in ep:
                for index in media['frame_indices']:
                    name=source['filenames'][index];p=path/name
                    if str(p) not in checked:assert digest(p)==ep['verified_alias_frame_sha256'][name];checked.add(str(p))
        return [ImageOps.pad(im,(448,448),method=Image.Resampling.BICUBIC,color=(127,127,127)) for im in engine.images(media)]
    def compute(ims):
        batch=engine.processor.image_processor(images=ims,return_tensors='pt')
        assert batch['image_grid_thw'].tolist()==[[1,32,32]]*len(ims)
        z=engine.base.visual(batch['pixel_values'].to('cuda:0',dtype=torch.bfloat16),grid_thw=batch['image_grid_thw'].to('cuda:0'))
        z=z.reshape(len(ims),16,16,3584).permute(0,3,1,2).float()
        z=F.avg_pool2d(z,2).permute(0,2,3,1).reshape(len(ims),64,3584).to(torch.bfloat16).cpu()
        assert torch.isfinite(z).all();return z
    # Acceptance compares actual H20 output against an existing identical encoder cache.
    parity=None
    for media in tasks:
        source=next((p/(media['key']+'.pt') for p in old if (p/(media['key']+'.pt')).exists()),None)
        if source:
            expected=torch.load(source,map_location='cpu',weights_only=True)
            ims=images(media);actual=torch.cat([compute(ims[i:i+8]) for i in range(0,len(ims),8)])
            assert actual.shape==expected['features'].shape and expected['feature_key']==engine.feature_key
            diff=(actual.float()-expected['features'].float()).abs()
            relative=(actual.float()-expected['features'].float()).norm()/expected['features'].float().norm().clamp_min(1e-8)
            assert relative.item()<.01,('Visual cache parity',relative.item())
            parity=dict(passed=True,media_key=media['key'],relative_l2=relative.item(),mean_abs=diff.mean().item(),max_abs=diff.max().item())
            break
    assert parity is not None,'No audited cached feature available for numerical comparison'
    write(ROOT/f'feature_parity_rank_{rank}.json',parity)
    selected=tasks[rank::world];completed=skipped=0;started=time.monotonic();budget=128
    def exists(media):return (dest/(media['key']+'.pt')).exists() or any((p/(media['key']+'.pt')).exists() for p in old)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        pending=[];frames=0
        def flush():
            nonlocal completed
            if not pending:return
            decoded=[(m,f.result()) for m,f in pending];all_images=[im for _,ims in decoded for im in ims]
            pieces=[]
            for start in range(0,len(all_images),budget):pieces.append(compute(all_images[start:start+budget]))
            values=torch.cat(pieces);offset=0
            for media,ims in decoded:
                path=dest/(media['key']+'.pt');temp=path.with_suffix(f'.{os.getpid()}.{rank}.tmp')
                payload=dict(feature_key=engine.feature_key,features=values[offset:offset+len(ims)].clone(),media_key=media['key'],representation=media.get('representation','recorded tactile input'))
                torch.save(payload,temp);temp.replace(path);offset+=len(ims);completed+=1
            write(ROOT/f'feature_progress_rank_{rank}.json',dict(rank=rank,completed=completed,skipped=skipped,total=len(selected),seconds=time.monotonic()-started,peak_mib=torch.cuda.max_memory_allocated()/2**20,gpu=torch.cuda.get_device_name(0),updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
            pending.clear()
        for media in selected:
            if exists(media):skipped+=1;continue
            pending.append((media,executor.submit(images,media)));frames+=8
            if frames>=budget:flush();frames=0
        flush()
    write(ROOT/f'feature_complete_rank_{rank}.json',dict(completed=completed,skipped=skipped,total=len(selected),seconds=time.monotonic()-started))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        write(ROOT/f'feature_failure_rank_{os.environ.get("RANK",0)}.json',dict(type=type(e).__name__,error=str(e)));raise
