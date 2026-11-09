"""Frozen validation-selected GRPO full inference; seal all shards before scoring."""
import datetime, fcntl, hashlib, json, os, sys, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
TRAIN=Path('/mnt/workspace/lyx/touchr1_grpo_accelerated_20261028')
FOLLOW=Path('/mnt/workspace/lyx/touchr1_followup_20261030')
DATA=FOLLOW/'evaluation_data'
def read(p): return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(4*2**20),b''): h.update(chunk)
 return h.hexdigest()
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def inputs():
 manifest=read(ROOT/'frozen_run.json'); data={}
 for name in ['touchreason','vtv']:
  p=DATA/f'{name}_inputs.jsonl';assert sha(p)==manifest['data'][name]['inputs_sha256']
  rows=[json.loads(s) for s in p.read_text().splitlines()]
  assert len(rows)==manifest['data'][name]['n'] and len({r['id'] for r in rows})==len(rows)
  assert all(not {'target','answer','gold','vtv_gold'}&set(r) for r in rows)
  data[name]=rows
 return manifest,data
def finalize(meta,data):
 with (ROOT/'finalize.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if (ROOT/'complete.json').exists():return
  label=meta['model']['label'];world=meta['world']
  if not all((ROOT/label/name/f'rank_{r}_sealed.json').exists() for name in data for r in range(world)):return
  sys.path[:0]=[str(FOLLOW),str(TRAIN)]
  import evaluate_7b as ev
  for name,rows in data.items(): ev.combine(ROOT/label/name,rows,world)
  for name in data:assert sha(DATA/f'{name}_gold.jsonl')==meta['data'][name]['gold_sha256']
  ev.score_all(ROOT,[meta['model']],data)
  write(ROOT/'complete.json',dict(utc=stamp(),status='complete',world=world,counts={n:len(r) for n,r in data.items()},checkpoint=meta['model']['checkpoint'],test_used_for_selection=False))
def main():
 import torch
 rank=int(os.environ.get('RANK','0'));meta,data=inputs();assert 0<=rank<meta['world']
 checkpoint=Path(meta['model']['checkpoint'])
 for name,digest in meta['model']['files_sha256'].items():assert sha(checkpoint/name)==digest
 sys.path.insert(0,'/mnt/workspace/lyx/touchr1_public_sft_grpo_16a800_20261028')
 from common import make_engine
 torch.manual_seed(meta['seed']);torch.cuda.set_device(0)
 engine=make_engine(meta['config'],str(checkpoint));assert engine.feature_key==meta['feature_key']
 # Add the independently verified VTV frozen feature cache to the existing cache loader.
 original=engine.feature
 def feature(media):
  p=Path('/mnt/workspace/lyx/touchr1_partial_test_amount_search_20261028/vtv_feature_cache')/engine.feature_key/(media['key']+'.pt')
  if p.exists():
   v=torch.load(p,map_location='cpu',weights_only=True)
   assert v['feature_key']==engine.feature_key and v['media_key']==media['key'] and v['frozen_vision_sha256']==engine.vision_sha
   return v
  return original(media)
 engine.feature=feature;engine.peft.gradient_checkpointing_disable();engine.mode(False)
 assert next(engine.base.model.parameters()).dtype==torch.float32
 write(ROOT/f'ready_rank_{rank}.json',dict(utc=stamp(),rank=rank,gpu=torch.cuda.get_device_name(0),checkpoint=str(checkpoint),decoder_dtype='float32',tf32=torch.backends.cuda.matmul.allow_tf32))
 for name,rows in data.items():
  folder=ROOT/meta['model']['label']/name;folder.mkdir(parents=True,exist_ok=True)
  selected=rows[rank::meta['world']];p=folder/f'rank_{rank}.jsonl'
  done=[json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
  assert [r['id'] for r in done]==[r['id'] for r in selected[:len(done)]]
  with p.open('a') as out,torch.no_grad():
   for i,row in enumerate(selected[len(done):],len(done)+1):
    response,tokens=engine.generate(row,sample=False,max_new_tokens=meta['max_new_tokens'])
    out.write(json.dumps(dict(id=row['id'],response=response,token_ids=tokens),ensure_ascii=False)+'\n');out.flush()
    write(ROOT/f'progress_rank_{rank}.json',dict(utc=stamp(),dataset=name,completed=i,total=len(selected)))
  write(folder/f'rank_{rank}_sealed.json',dict(utc=stamp(),n=len(selected),sha256=sha(p),gold_opened=False))
 finalize(meta,data)
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  write(ROOT/f"failure_rank_{os.environ.get('RANK','0')}.json",dict(utc=stamp(),error=str(exc),traceback=traceback.format_exc()));raise
