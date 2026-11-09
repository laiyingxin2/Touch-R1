"""Public tactile QA: frozen visual features, residual adapter, language LoRA.

No source filenames, annotation metadata, object IDs or gold labels enter prompts.
Features are computed from the eight uniformly sampled real frames of each clip.
"""
import contextlib
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import sys

CODE_ROOT = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('TACTILE_RUN_ROOT', str(CODE_ROOT)))
ROOT.mkdir(parents=True, exist_ok=True)
DATA = Path('/mnt/workspace/lyx/public_tactile_qa_20261022')
MODEL = '/mnt/workspace/zfc/Qwen/Qwen2.5-VL-7B-Instruct'
MODEL_LOAD_PATH = os.environ.get('TACTILE_MODEL_LOAD_PATH', MODEL)
sys.path.insert(0, str(CODE_ROOT / 'deps'))
sys.path.insert(0, str(DATA))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
sys.modules['deepspeed'] = None
import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

SEED = 20261023
POOL_SIDE = int(os.environ.get('TACTILE_POOL_SIDE', '4'))
IMAGE_SIDE = int(os.environ.get('TACTILE_IMAGE_SIDE', '224'))
assert IMAGE_SIDE in [224,448]
MERGED_SIDE = IMAGE_SIDE//28
LORA_R = int(os.environ.get('TACTILE_LORA_R', '8'))
LORA_TARGETS = os.environ.get('TACTILE_LORA_TARGETS', 'q_proj,v_proj').split(',')
assert POOL_SIDE in [4,8]
RAW_SPEC = {'version': 2, 'model': MODEL, 'frames': 8, 'image_side': IMAGE_SIDE,
                'letterbox': 'gray_127', 'merged_side': MERGED_SIDE, 'dtype': 'bfloat16',
                'sampling': 'round(linspace(0,n-1,min(n,8)))'}
RAW_KEY = hashlib.sha256(json.dumps(RAW_SPEC, sort_keys=True).encode()).hexdigest()[:16]
FEATURE_SPEC = dict(RAW_SPEC,pool_side=POOL_SIDE,image_grid=[1,POOL_SIDE*2,POOL_SIDE*2])
FEATURE_KEY = hashlib.sha256(json.dumps(FEATURE_SPEC, sort_keys=True).encode()).hexdigest()[:16]
CACHE = CODE_ROOT / 'features' / RAW_KEY

def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(x) for x in f if x.strip()]

def records(split='train', view='quality_v3'):
    return read_jsonl(DATA / 'organized/annotations' / view / f'{split}.jsonl')

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)

def emit(event, **kwargs):
    item = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'event': event, **kwargs}
    print(json.dumps(item, ensure_ascii=False), flush=True)
    with (ROOT / 'events.jsonl').open('a') as f:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

def setup():
    torch.set_num_threads(4)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    assert torch.cuda.is_available(), 'Use the documented CUDA Python, not /usr/bin/python'
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('TACTILE_MEMORY_FRACTION','0.36')))

def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()

class ResidualAdapter(torch.nn.Module):
    def __init__(self, width=3584, rank=64):
        super().__init__()
        self.norm = torch.nn.LayerNorm(width)
        self.down = torch.nn.Linear(width, rank)
        self.up = torch.nn.Linear(rank, width)
        torch.nn.init.zeros_(self.up.weight); torch.nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(F.gelu(self.down(self.norm(x))))

class Engine:
    def __init__(self):
        setup()
        if MODEL_LOAD_PATH != MODEL:
            identity=json.loads((Path(MODEL_LOAD_PATH)/'source_identity.json').read_text())
            assert identity['source_model']==MODEL and identity['verified_local_sha256']
            for name,info in identity['files'].items():
                assert (Path(MODEL_LOAD_PATH)/name).stat().st_size==info['bytes']
        self.processor = AutoProcessor.from_pretrained(MODEL_LOAD_PATH, local_files_only=True,
                                                       min_pixels=IMAGE_SIDE**2, max_pixels=IMAGE_SIDE**2)
        self.tokenizer = self.processor.tokenizer
        self.base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_LOAD_PATH, torch_dtype=torch.bfloat16, device_map={'': 'cuda:0'},
            attn_implementation='sdpa', local_files_only=True)
        self.peft = get_peft_model(self.base, LoraConfig(r=LORA_R, lora_alpha=2*LORA_R,
            lora_dropout=0.05, target_modules=LORA_TARGETS, bias='none', task_type='CAUSAL_LM'))
        self.adapter = ResidualAdapter().to('cuda:0', dtype=torch.bfloat16)
        self.base.config.use_cache = False
        self.peft.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.peft.enable_input_require_grads()
        self.media = {x['id']: x for x in read_jsonl(DATA / 'organized/media_index.jsonl')}
        self.memory = {}
        CACHE.mkdir(parents=True, exist_ok=True)
        self.parameters = [(n, p) for n, p in self.peft.named_parameters() if p.requires_grad]
        assert self.parameters and all('lora_' in n and 'visual' not in n for n, _ in self.parameters)
        self.parameters += [('tactile_adapter.' + n, p) for n, p in self.adapter.named_parameters()]
        emit('model_loaded', trainable=sum(p.numel() for _, p in self.parameters),
             total=sum(p.numel() for p in self.peft.parameters()) + sum(p.numel() for p in self.adapter.parameters()),
             cache_key=FEATURE_KEY, gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
             allocated_gib=torch.cuda.memory_allocated()/2**30,model_load_path=MODEL_LOAD_PATH)

    def mode(self, training, dropout=True):
        self.peft.train(training); self.adapter.train(training)
        self.base.visual.eval()
        if training and not dropout:
            # Gradient checkpointing follows the decoder's training flag. RL
            # and parity checks need checkpointing with deterministic dropout.
            for module in self.peft.modules():
                if isinstance(module,torch.nn.Dropout):module.eval()

    def frames(self, mid):
        m = self.media[mid]; path = DATA / 'organized' / m['path']
        if m['kind'] == 'frames':
            n = len(m['frames'])
        else:
            probe = cv2.VideoCapture(str(path))
            try: n = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
            finally: probe.release()
        assert n > 0, f'Empty media: {mid}'
        indices = np.linspace(0, n-1, min(n, 8)).round().astype(int).tolist()
        assert indices == sorted(set(indices))
        frames = []
        if m['kind'] == 'frames':
            for i in indices:
                with Image.open(path / m['frames'][i]) as im: frames.append(im.convert('RGB'))
        else:
            cap = cv2.VideoCapture(str(path))
            try:
                # Decode sequentially; inaccurate codec seeking cannot reorder frames.
                for i in range(indices[-1]+1):
                    ok, frame = cap.read()
                    if not ok: raise RuntimeError(f'Frame decode failed: {mid}, {i}')
                    if i in indices: frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            finally: cap.release()
        assert len(frames) == len(indices)
        return [ImageOps.pad(x, (IMAGE_SIDE,IMAGE_SIDE), method=Image.Resampling.BICUBIC, color=(127,127,127)) for x in frames], indices

    @torch.no_grad()
    def feature(self, mid):
        if mid in self.memory: return self.memory[mid]
        path = CACHE / f'{mid}.pt'
        if path.exists():
            obj = torch.load(path, map_location='cpu', weights_only=True)
        else:
            frames, indices = self.frames(mid)
            x = self.processor.image_processor(images=frames, return_tensors='pt')
            grids = x['image_grid_thw']
            assert grids.tolist() == [[1,IMAGE_SIDE//14,IMAGE_SIDE//14]] * len(frames)
            self.base.visual.eval()
            z = self.base.visual(x['pixel_values'].to('cuda:0', dtype=torch.bfloat16),
                                 grid_thw=grids.to('cuda:0'))
            assert z.shape == (len(frames)*MERGED_SIDE**2, 3584)
            # Qwen returns merged spatial blocks in row-major order after undoing window permutation.
            obj = {'features': z.to(torch.bfloat16).cpu(), 'indices': indices, 'cache_key': RAW_KEY}
            tmp = path.with_suffix(f'.{os.getpid()}.tmp'); torch.save(obj, tmp); tmp.replace(path)
        assert obj['cache_key'] == RAW_KEY
        assert obj['features'].shape == (len(obj['indices'])*MERGED_SIDE**2, 3584)
        if POOL_SIDE!=MERGED_SIDE:
            assert MERGED_SIDE%POOL_SIDE==0
            z=obj['features'].reshape(-1,MERGED_SIDE,MERGED_SIDE,3584).permute(0,3,1,2).float()
            z=F.avg_pool2d(z,MERGED_SIDE//POOL_SIDE).permute(0,2,3,1).reshape(-1,3584).to(torch.bfloat16)
            obj=dict(obj,features=z)
        assert torch.isfinite(obj['features']).all()
        self.memory[mid] = obj
        return obj

    def encode(self, row, target=True, control='real'):
        # Explicit allowlist. Metadata is used by evaluation only, never by this prompt builder.
        msgs = row['messages'][:-1]
        assert len(msgs) == 1 and msgs[0]['role'] == 'user'
        mids = msgs[0]['media_ids']; assert mids == row['media_ids']
        parts = msgs[0]['content'].split('<tactile>'); assert len(parts) == len(mids)+1
        contents = []; feats = []; grids = []
        for j, mid in enumerate(mids):
            contents.append({'type':'text', 'text':parts[j] + '\nOrdered tactile frames:\n'})
            real = self.feature(mid)
            obj = self.feature('__blank__') if control == 'blank' else real
            z = obj['features']
            if control == 'mismatch':
                # Deterministic cyclic replacement within the current split; set by evaluator.
                z = self.feature(self.mismatch[mid])['features']
            nf = len(real['indices'])
            count=POOL_SIDE**2
            if len(z) != nf*count:
                z = z.reshape(-1,count,3584)[torch.linspace(0,len(z)//count-1,nf).round().long()].reshape(-1,3584)
            feats.append(z)
            for i in range(nf):
                contents.extend([{'type':'text','text':f'Frame {i+1}: '}, {'type':'image'}])
                grids.append([1,POOL_SIDE*2,POOL_SIDE*2])
        contents.append({'type':'text','text':parts[-1]})
        messages = [{'role':'system','content':'Infer the requested categories and relations from the ordered tactile frames. Follow the question and output only the requested JSON.'},
                    {'role':'user','content':contents}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt = prompt.replace('<|image_pad|>', '<|image_pad|>'*(POOL_SIDE**2))
        prefix = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False)['input_ids']
        answer = row['messages'][-1]['content'] if target else ''
        full = self.tokenizer(prompt + answer + (self.tokenizer.eos_token if target else ''),
                              return_tensors='pt', add_special_tokens=False)['input_ids']
        assert torch.equal(prefix, full[:,:prefix.shape[1]]), 'Prompt is not a token prefix'
        assert full.shape[1] < 4096
        f = torch.cat(feats).to('cuda:0'); ids = full.to('cuda:0')
        grid = torch.tensor(grids, dtype=torch.long, device='cuda:0')
        assert (ids == self.base.config.image_token_id).sum().item() == f.shape[0]
        pos, delta = self.base.get_rope_index(ids, image_grid_thw=grid, attention_mask=torch.ones_like(ids))
        return {'ids':ids, 'features':f, 'grid':grid, 'positions':pos, 'delta':delta,
                'prefix':prefix.shape[1]}

    def embeds(self, x, use_adapter=True):
        embeds = self.base.model.embed_tokens(x['ids'])
        f = self.adapter(x['features']) if use_adapter else x['features']
        is_prompt=torch.arange(x['ids'].shape[1],device=x['ids'].device)<x['prefix']
        mask = ((x['ids'] == self.base.config.image_token_id)&is_prompt).unsqueeze(-1).expand_as(embeds)
        return embeds.masked_scatter(mask, f.to(embeds.dtype))

    def append_completion(self, prompt, tokens):
        """Keep actual sampled IDs and the same linear text M-RoPE as KV decoding.

        Generated special IDs must not be reinterpreted as additional images.
        Their low format reward is handled by the trainer, not a sampling filter.
        """
        assert prompt['ids'].shape[1]==prompt['prefix'] and len(tokens)>0
        x=dict(prompt)
        suffix=torch.tensor([tokens],dtype=torch.long,device=prompt['ids'].device)
        x['ids']=torch.cat([prompt['ids'],suffix],dim=1)
        positions=torch.arange(len(tokens),device=suffix.device).view(1,1,-1)
        positions=positions+prompt['prefix']+prompt['delta'].item()
        x['positions']=torch.cat([prompt['positions'],positions.expand(3,1,-1)],dim=-1)
        return x

    def logprobs(self, x, use_adapter=True):
        hidden = self.base.model(inputs_embeds=self.embeds(x, use_adapter),
            position_ids=x['positions'], attention_mask=torch.ones_like(x['ids']),
            use_cache=False, return_dict=True).last_hidden_state
        # Only assistant-target predictions need the very large vocabulary projection.
        logits = self.base.lm_head(hidden[:, x['prefix']-1:-1]).float()
        labels = x['ids'][:, x['prefix']:]
        assert logits.shape[1] == labels.shape[1] and labels.numel() > 0
        return -F.cross_entropy(logits.flatten(0,1), labels.flatten(), reduction='none').reshape_as(labels)

    @torch.no_grad()
    def generate(self, row, control='real', baseline=False, sample=False, max_new_tokens=320):
        self.mode(False)
        x = self.encode(row, target=False, control=control)
        # Explicit KV decoding: no feature reinjection, and use Qwen's M-RoPE delta.
        emb = self.embeds(x, use_adapter=not baseline)
        out = self.base.model(inputs_embeds=emb, position_ids=x['positions'], use_cache=True, return_dict=True)
        generated = []
        for i in range(max_new_tokens):
            logits = self.base.lm_head(out.last_hidden_state[:,-1]).float()
            token = torch.multinomial(logits.softmax(-1), 1) if sample else logits.argmax(-1, keepdim=True)
            generated.append(token.item())
            if token.item() in {self.tokenizer.eos_token_id, self.base.config.eos_token_id if isinstance(self.base.config.eos_token_id,int) else -1}: break
            # First generated token occupies sequence position len(prompt).
            p = x['ids'].shape[1] + i + x['delta'].item()
            position = torch.full((3,1,1), p, dtype=torch.long, device='cuda:0')
            out = self.base.model(input_ids=token, position_ids=position, past_key_values=out.past_key_values,
                                  use_cache=True, return_dict=True)
        return self.tokenizer.decode(generated, skip_special_tokens=True), generated

    def save(self, path, state=None):
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        self.peft.save_pretrained(path)
        torch.save(self.adapter.state_dict(), path/'tactile_adapter.pt')
        if state is not None:
            temp = path/'trainer_state.tmp'; torch.save(state, temp); temp.replace(path/'trainer_state.pt')
        write_json(path/'manifest.json', {'base_model':MODEL, 'feature_spec':FEATURE_SPEC,
                   'feature_key':FEATURE_KEY, 'seed':SEED, 'complete':True})

    def load(self, path):
        from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
        path = Path(path)
        assert json.loads((path/'manifest.json').read_text())['feature_key'] == FEATURE_KEY
        set_peft_model_state_dict(self.peft, load_peft_weights(str(path), device='cuda:0'))
        self.adapter.load_state_dict(torch.load(path/'tactile_adapter.pt', map_location='cuda:0', weights_only=True))

    @torch.no_grad()
    def blank_feature(self):
        path = CACHE/'__blank__.pt'
        if not path.exists():
            x = self.processor.image_processor(images=[Image.new('RGB',(IMAGE_SIDE,IMAGE_SIDE),(127,127,127))]*8,return_tensors='pt')
            z = self.base.visual(x['pixel_values'].to('cuda:0',dtype=torch.bfloat16),grid_thw=x['image_grid_thw'].to('cuda:0'))
            z = z.to(torch.bfloat16).cpu()
            tmp=path.with_suffix(f'.{os.getpid()}.tmp'); torch.save({'features':z,'indices':list(range(8)),'cache_key':RAW_KEY},tmp);tmp.replace(path)

    @contextlib.contextmanager
    def reference(self, values):
        current = {n:p.detach().clone() for n,p in self.parameters}
        with torch.no_grad():
            for n,p in self.parameters: p.copy_(values[n])
        try: yield
        finally:
            with torch.no_grad():
                for n,p in self.parameters: p.copy_(current[n])
