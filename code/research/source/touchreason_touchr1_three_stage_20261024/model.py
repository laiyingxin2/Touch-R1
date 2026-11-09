"""One trained visual tower, cached real evidence, and a PEFT Qwen QA policy."""
import collections
import hashlib
import json
import os
import re
from pathlib import Path
import sys

from data import ROOT, PUBLIC, SEED, read, write, digest, key
from vision import attach, decode, MODEL, LOAD_MODEL, efficient_visual_attention

OLD = Path('/mnt/workspace/lyx/public_tactile_touchr1_peft_20261023')
sys.path.insert(0, str(ROOT/'deps'))
sys.path.insert(0, str(OLD))
os.environ['TACTILE_IMAGE_SIDE'] = '448'
os.environ['TACTILE_POOL_SIDE'] = '8'
os.environ.setdefault('TACTILE_MODEL_LOAD_PATH', LOAD_MODEL)
os.environ.setdefault('TACTILE_MEMORY_FRACTION', '.36')
import runtime as rt
rt.SEED = SEED
from method_runtime import MethodEngine
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps


SPEC = {'name': 'touchreason_three_stage_r8', 'rank': 8, 'alpha': 16, 'dropout': .05,
        'pool_side': 8, 'adapter_rank': 64, 'targets': ['q_proj', 'v_proj']}
SYSTEM = ('Infer the requested answer from the supplied tactile observations. '
          'Respond with exactly <perceive>brief directly visible evidence</perceive>'
          '<compare>brief comparison if needed</compare><conclude>the requested answer only</conclude>. '
          'Do not invent physical measurements, sensor calibration, or unavailable evidence. '
          'Keep the requested native answer format inside conclude, including JSON when requested.')


class Engine(MethodEngine):
    def __init__(self, vision_checkpoint=None):
        self.vision_checkpoint = Path(vision_checkpoint or read(ROOT/'stage1/selected.json')['checkpoint'])
        self.vision_sha = digest(self.vision_checkpoint/'vision_adapter.pt')
        load_spec = SPEC
        if any(name in SPEC['targets'] for name in ('gate_proj', 'up_proj', 'down_proj')):
            # The visual tower also has an MLP. Anchor expanded language LoRA
            # to decoder layers so the frozen, cached visual encoder stays fixed.
            targets = '|'.join(re.escape(name) for name in SPEC['targets'])
            load_spec = dict(SPEC, targets=rf'model\.layers\.\d+\.(?:self_attn|mlp)\.(?:{targets})')
        super().__init__(load_spec)
        self.spec = SPEC
        state = torch.load(self.vision_checkpoint/'vision_adapter.pt', map_location='cpu', weights_only=True)
        self.visual_modules = attach(self.base.visual, state, trainable=False)
        efficient_visual_attention(self.base.visual)
        # FP32 optimizer updates prevent tiny PEFT steps being rounded away.
        self.adapter.float()
        self.feature_identity = {'base': MODEL, 'stage1_vision_sha256': self.vision_sha,
            'image_side': 448, 'letterbox': 'gray127', 'pooled_tokens': 64, 'version': 2,
            'vision_attention': 'equivalent block-diagonal segmented SDPA'}
        self.feature_key = key(self.feature_identity)[:24]
        self.cache = ROOT/'features'/self.feature_key
        self.cache.mkdir(parents=True, exist_ok=True)
        self.episodes = read(ROOT/'episodes.json')
        self.feature_memory = collections.OrderedDict()
        self.mismatch = {}

    def images(self, media):
        kind = media['kind']
        if kind == 'images':
            images = []
            for i, name in enumerate(media['paths']):
                if media.get('sha256'):
                    assert digest(name) == media['sha256'][i], 'Reviewed evidence image changed'
                with Image.open(name) as im:
                    images.append(im.convert('RGB'))
            return images
        if kind == 'raw':
            decoded = decode(self.episodes[media['episode_id']], media['frame_indices'])
            return [decoded[i] for i in media['frame_indices']]
        if kind == 'blank':
            return [Image.new('RGB', (448, 448), (127, 127, 127))]
        assert kind == 'public'
        source = media['source']
        path = PUBLIC/'organized'/source['path']
        if source['kind'] == 'frames':
            count = len(source['frames'])
            indices = torch.linspace(0, count-1, min(8, count)).round().long().tolist()
            images = []
            for i in indices:
                with Image.open(path/source['frames'][i]) as im:
                    images.append(im.convert('RGB'))
            return images
        import cv2
        cap = cv2.VideoCapture(str(path))
        try:
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        assert count > 0
        indices = torch.linspace(0, count-1, min(8, count)).round().long().tolist()
        ep = {'source': {'kind': 'video', 'path': str(path)}}
        decoded = decode(ep, indices)
        return [decoded[i] for i in indices]

    @torch.no_grad()
    def feature(self, media):
        mid = media['key']
        if mid in self.feature_memory:
            self.feature_memory.move_to_end(mid)
            return self.feature_memory[mid]
        path = self.cache/f'{mid}.pt'
        if path.exists():
            value = torch.load(path, map_location='cpu', weights_only=True)
            assert value['feature_key'] == self.feature_key
        else:
            images = self.images(media)
            images = [ImageOps.pad(im, (448, 448), method=Image.Resampling.BICUBIC, color=(127, 127, 127)) for im in images]
            assert images
            self.base.visual.eval()
            parts = []
            for start in range(0, len(images), 8):
                ims = images[start:start+8]
                batch = self.processor.image_processor(images=ims, return_tensors='pt')
                assert batch['image_grid_thw'].tolist() == [[1, 32, 32]]*len(ims)
                z = self.base.visual(batch['pixel_values'].to('cuda:0', dtype=torch.bfloat16),
                                     grid_thw=batch['image_grid_thw'].to('cuda:0'))
                z = z.reshape(len(ims), 16, 16, 3584).permute(0, 3, 1, 2).float()
                z = F.avg_pool2d(z, 2).permute(0, 2, 3, 1).reshape(len(ims), 64, 3584).to(torch.bfloat16).cpu()
                parts.append(z)
            value = {'feature_key': self.feature_key, 'features': torch.cat(parts),
                     'media_key': mid, 'representation': media.get('representation', 'recorded tactile input')}
            assert torch.isfinite(value['features']).all()
            temp = path.with_suffix(f'.{os.getpid()}.tmp')
            torch.save(value, temp)
            temp.replace(path)
        self.feature_memory[mid] = value
        while len(self.feature_memory) > 96:
            self.feature_memory.popitem(last=False)
        return value

    def blank_feature(self):
        return self.feature({'kind': 'blank', 'key': 'blank_frame'})

    def encode(self, row, target=True, control='real'):
        assert control in ('real', 'blank', 'mismatch')
        real = [self.feature(m)['features'] for m in row['media']]
        nimages = sum(len(z) for z in real)
        pool = 8 if nimages <= 32 else 4
        contents, features, grids, streams = [], [], [], []
        offset = 0
        for i, (media, actual) in enumerate(zip(row['media'], real)):
            z = actual
            if control == 'blank':
                z = self.blank_feature()['features'].repeat(len(actual), 1, 1)
            elif control == 'mismatch':
                donor = self.mismatch[media['key']]
                z = self.feature(donor)['features']
                if len(z) != len(actual):
                    z = z[torch.linspace(0, len(z)-1, len(actual)).round().long()]
            if pool == 4:
                z = z.reshape(len(z), 8, 8, 3584).permute(0, 3, 1, 2).float()
                z = F.avg_pool2d(z, 2).permute(0, 2, 3, 1).reshape(len(z), 16, 3584).to(torch.bfloat16)
            public_definitions = row['input'].get('input_definitions')
            public = public_definitions[i] if public_definitions else row['input']
            representation = public.get('representation', media.get('representation', 'recorded tactile observations'))
            contents.append({'type': 'text', 'text': f'Input {i+1} ({"abcd"[i] if i<4 else i+1}): {representation}\n'})
            declared = public.get('frames')
            for j in range(len(z)):
                frame = declared[j] if declared and j < len(declared) else f'Image {j+1}'
                contents.extend([{'type': 'text', 'text': str(frame)+': '}, {'type': 'image'}])
                grids.append([1, pool*2, pool*2])
            flat = z.reshape(-1, 3584)
            streams.append((offset, offset+len(flat)))
            offset += len(flat)
            features.append(flat)
        # Explicit public input allowlist: no IDs, private paths, source label,
        # reviewer output, gold answer, or dataset name is serialized here.
        prompt_data = {k: row['input'][k] for k in ('question', 'options', 'frames', 'representation',
            'input_definitions', 'recorded_frame_indices') if k in row['input']}
        assert '/mnt/' not in json.dumps(prompt_data) and '/tmp/' not in json.dumps(prompt_data)
        contents.append({'type': 'text', 'text': json.dumps(prompt_data, ensure_ascii=False)+
                         '\nUse the three tags; put only the requested native answer inside <conclude>.'})
        prompt = self.processor.apply_chat_template([{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': contents}], tokenize=False, add_generation_prompt=True)
        prompt = prompt.replace('<|image_pad|>', '<|image_pad|>'*(pool**2))
        prefix = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False)['input_ids']
        if target:
            ids = self.tokenizer(prompt+row['target']+self.tokenizer.eos_token,
                                 return_tensors='pt', add_special_tokens=False)['input_ids']
            assert torch.equal(ids[:, :prefix.shape[1]], prefix), 'Prompt must be a true target-token prefix'
        else:
            ids = prefix
        assert ids.shape[1] <= 4096, f"Context exceeds 4096: {row['id']} {ids.shape[1]}"
        ids = ids.to('cuda:0')
        f = torch.cat(features).to('cuda:0')
        grid = torch.tensor(grids, dtype=torch.long, device='cuda:0')
        assert (ids[:, :prefix.shape[1]] == self.base.config.image_token_id).sum().item() == len(f)
        positions, delta = self.base.get_rope_index(ids, image_grid_thw=grid, attention_mask=torch.ones_like(ids))
        return {'ids': ids, 'features': f, 'grid': grid, 'positions': positions, 'delta': delta,
                'prefix': prefix.shape[1], 'stream_slices': streams}

    def embeds(self, x, use_adapter=True):
        embeds = self.base.model.embed_tokens(x['ids'])
        f = self.adapter(x['features'].float()) if use_adapter else x['features']
        prompt_mask = torch.arange(x['ids'].shape[1], device=x['ids'].device) < x['prefix']
        mask = ((x['ids'] == self.base.config.image_token_id) & prompt_mask).unsqueeze(-1).expand_as(embeds)
        return embeds.masked_scatter(mask, f.to(embeds.dtype))

    def policy(self, x, selected_positions=None):
        hidden = self.base.model(inputs_embeds=self.embeds(x), position_ids=x['positions'],
                                 attention_mask=torch.ones_like(x['ids']), use_cache=False, return_dict=True).last_hidden_state
        logits = self.base.lm_head(hidden[:, x['prefix']-1:-1]).float()
        labels = x['ids'][:, x['prefix']:]
        lp = -F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction='none')
        logdist = F.log_softmax(logits[0, selected_positions], dim=-1) if selected_positions is not None else None
        return lp, logdist

    def noisy_logdist(self, x, selected_positions):
        hidden = self.base.model(inputs_embeds=self.embeds(x), position_ids=x['positions'],
                                 attention_mask=torch.ones_like(x['ids']), use_cache=False, return_dict=True).last_hidden_state
        z = hidden[0, x['prefix']-1+selected_positions]
        return F.log_softmax(self.base.lm_head(z).float(), dim=-1)

    @torch.no_grad()
    def rollouts(self, prompt, count=8, microbatch=8, max_new_tokens=384, return_logprobs=False):
        """Batched KV decoding, preserving the exact IDs sampled at temperature 1."""
        self.mode(False)
        results = []
        sampled_logprobs = []
        limit = min(max_new_tokens, 4096-prompt['prefix'])
        assert limit > 0
        eos = self.tokenizer.eos_token_id
        for start in range(0, count, microbatch):
            size = min(microbatch, count-start)
            embeddings = self.embeds(prompt).expand(size, -1, -1)
            positions = prompt['positions'].expand(-1, size, -1)
            out = self.base.model(inputs_embeds=embeddings, position_ids=positions, use_cache=True, return_dict=True)
            sequences = [[] for _ in range(size)]
            probabilities = [[] for _ in range(size)]
            finished = [False]*size
            for t in range(limit):
                logits = self.base.lm_head(out.last_hidden_state[:, -1]).float()
                tokens = torch.multinomial(logits.softmax(-1), 1)
                lp = F.log_softmax(logits, -1).gather(-1, tokens).flatten().tolist() if return_logprobs else None
                for j, token in enumerate(tokens.flatten().tolist()):
                    if not finished[j]:
                        sequences[j].append(token)
                        if return_logprobs:
                            probabilities[j].append(lp[j])
                        finished[j] = token == eos
                if all(finished):
                    break
                position = torch.full((3, size, 1), prompt['prefix']+t+prompt['delta'].item(),
                                      device='cuda:0', dtype=torch.long)
                out = self.base.model(input_ids=tokens, position_ids=position,
                                      past_key_values=out.past_key_values, use_cache=True, return_dict=True)
            results.extend(sequences)
            sampled_logprobs.extend(probabilities)
        return (results, sampled_logprobs) if return_logprobs else results

    def save(self, path, state=None):
        path = Path(path)
        super().save(path, state)
        torch.save(torch.load(self.vision_checkpoint/'vision_adapter.pt', map_location='cpu', weights_only=True), path/'vision_adapter.pt')
        manifest = read(path/'manifest.json')
        manifest.update(format='touchreason-three-stage-v1', stage1_vision_sha256=self.vision_sha,feature_key=self.feature_key,
                        stage1_checkpoint=str(self.vision_checkpoint), feature_identity=self.feature_identity,
                        test_used_for_selection=False, seed=rt.SEED)
        write(path/'manifest.json', manifest)

    def load(self, path):
        path=Path(path)
        manifest = read(path/'manifest.json')
        assert manifest['stage1_vision_sha256'] == self.vision_sha
        assert manifest['feature_key']==self.feature_key and manifest['method_spec']==SPEC
        self.restore(torch.load(path/'adaptation.pt',map_location='cpu',weights_only=True))
