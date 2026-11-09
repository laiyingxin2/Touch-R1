"""Parameter-efficient ViT tactile dynamics encoder, transferred into QA.

Only Qwen's visual tower is loaded for Stage 1. Low-rank updates on its last
four attention blocks learn from future-token prediction. The frozen original
visual tower supplies the future target, preventing a moving-target collapse.
"""
import contextlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'deps'))
sys.modules['deepspeed'] = None
import cv2
import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
from torch import nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
from efficient_attention import enable as efficient_visual_attention

MODEL = '/mnt/workspace/zfc/Qwen/Qwen2.5-VL-7B-Instruct'
RAM_MODEL = Path('/dev/shm/lyx_tactile_qwen25vl7b_20261023')
LOAD_MODEL = str(RAM_MODEL) if (RAM_MODEL/'source_identity.json').exists() else MODEL
VISION_SPEC = {'rank': 4, 'alpha': 8, 'last_blocks': 4, 'targets': ['qkv', 'proj'],
               'resolution': 448, 'tokens_per_frame': 64,
               'rotary_buffer_precision': 'native float32', 'pool_accumulation': 'float32'}


class LoRALinear(nn.Module):
    def __init__(self, base, rank=4, alpha=8):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha/rank
        self.enabled = True

    def forward(self, x):
        result = self.base(x)
        if self.enabled:
            result = result+(F.linear(F.linear(x.float(), self.A), self.B)*self.scale).to(result.dtype)
        return result


def attach(model, state=None, trainable=False):
    blocks = model.blocks
    modules = {}
    for i in range(len(blocks)-VISION_SPEC['last_blocks'], len(blocks)):
        for name in VISION_SPEC['targets']:
            layer = getattr(blocks[i].attn, name)
            if not isinstance(layer, LoRALinear):
                layer = LoRALinear(layer, VISION_SPEC['rank'], VISION_SPEC['alpha'])
                setattr(blocks[i].attn, name, layer)
            modules[f'blocks.{i}.attn.{name}'] = layer
    if state is not None:
        expected = {f'{n}.{p}' for n in modules for p in ('A', 'B')}
        assert set(state) == expected
        with torch.no_grad():
            for n, module in modules.items():
                module.A.copy_(state[n+'.A'].to(module.A.device))
                module.B.copy_(state[n+'.B'].to(module.B.device))
    for module in modules.values():
        module.A.requires_grad_(trainable)
        module.B.requires_grad_(trainable)
    return modules


def adapter_state(modules):
    return {f'{n}.{p}': getattr(m, p).detach().cpu().clone() for n, m in modules.items() for p in ('A', 'B')}


def decode(ep, indices, history_scale_indices=None):
    source = ep['source']
    indices = sorted(set(indices))
    result = {}
    if source['kind'] == 'png_sequence':
        for i in indices:
            with Image.open(Path(source['path'])/source['filenames'][i]) as im:
                result[i] = im.convert('RGB')
        return result
    if source['kind'] == 'video':
        cap = cv2.VideoCapture(source['path'])
        wanted = set(indices)
        try:
            for i in range(max(indices)+1):
                if not cap.grab():
                    raise ValueError(f"Frame decode failed: {source['path']} {i}")
                if i in wanted:
                    ok, frame = cap.retrieve()
                    if not ok:
                        raise ValueError('Frame retrieve failed')
                    result[i] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        return result
    assert source['kind'] == 'tac3d_matrix'
    with h5py.File(source['path']) as f:
        all_dis = np.asarray(f['displacements'])
        pos = np.asarray(f['positions'][0])
    assert np.isfinite(all_dis).all() and np.isfinite(pos).all()
    xy = pos[:, :2]
    xy = (xy-xy.min(0))/np.maximum(xy.max(0)-xy.min(0), 1e-9)*350+49
    # Stage 1 fits the display scale on PAST frames only; a future value must
    # never alter its context representation. QA keeps the audited release's
    # whole-episode rendering contract when no explicit images are supplied.
    scale_frames = all_dis[history_scale_indices] if history_scale_indices is not None else all_dis
    scale = max(float(np.percentile(np.linalg.norm(scale_frames, axis=-1), 99)), 1e-9)
    for i in indices:
        image = Image.new('RGB', (448, 448), (245, 245, 245))
        draw = ImageDraw.Draw(image)
        for p, v in zip(xy, all_dis[i]):
            end = p+v[:2]/scale*20
            z = float(np.clip(v[2]/scale, -1, 1))
            color = (int(120+120*max(z, 0)), 80, int(120+120*max(-z, 0)))
            draw.line([tuple(p), tuple(end)], fill=color, width=2)
            draw.ellipse((p[0]-2, p[1]-2, p[0]+2, p[1]+2), fill=color)
        result[i] = image
    return result


class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        config = AutoConfig.from_pretrained(LOAD_MODEL, local_files_only=True).vision_config
        config._attn_implementation = 'sdpa'
        index = json.loads((Path(LOAD_MODEL)/'model.safetensors.index.json').read_text())['weight_map']
        groups = {}
        for name, shard in index.items():
            if name.startswith('visual.'):
                groups.setdefault(shard, []).append(name)
        state = {}
        for shard, names in groups.items():
            with safe_open(str(Path(LOAD_MODEL)/shard), framework='pt', device='cpu') as f:
                state.update({n[len('visual.'):]: f.get_tensor(n) for n in names})
        # Match Qwen's full-model loader. Casting the entire tower with .to(BF16)
        # also rounds its nonpersistent rotary frequencies. Native from_pretrained
        # creates those frequencies in FP32 on CPU and preserves them on transfer.
        self.model, info = Qwen2_5_VisionTransformerPretrainedModel.from_pretrained(
            None, config=config, state_dict=state, torch_dtype=torch.bfloat16,
            attn_implementation='sdpa', output_loading_info=True)
        assert not info['missing_keys'] and not info['unexpected_keys'] and not info.get('mismatched_keys')
        self.model.to('cuda:0')
        assert self.model.rotary_pos_emb.inv_freq.dtype == torch.float32
        del state
        self.model.requires_grad_(False)
        efficient_visual_attention(self.model)
        self.modules_lora = attach(self.model, trainable=True)
        self.processor = AutoProcessor.from_pretrained(LOAD_MODEL, local_files_only=True,
                                                       min_pixels=448**2, max_pixels=448**2)
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    @contextlib.contextmanager
    def teacher(self):
        flags = [m.enabled for m in self.modules_lora.values()]
        training = self.model.training
        for m in self.modules_lora.values():
            m.enabled = False
        self.model.eval()
        try:
            with torch.no_grad():
                yield
        finally:
            for module, flag in zip(self.modules_lora.values(), flags):
                module.enabled = flag
            self.model.train(training)

    def forward(self, images):
        images = [ImageOps.pad(im, (448, 448), method=Image.Resampling.BICUBIC, color=(127, 127, 127)) for im in images]
        x = self.processor.image_processor(images=images, return_tensors='pt')
        grids = x['image_grid_thw'].to('cuda:0')
        assert grids.tolist() == [[1, 32, 32]]*len(images)
        z = self.model(x['pixel_values'].to('cuda:0', dtype=torch.bfloat16), grid_thw=grids)
        assert z.shape == (len(images)*256, 3584)
        z = z.reshape(len(images), 16, 16, 3584).permute(0, 3, 1, 2).float()
        return F.avg_pool2d(z, 2).permute(0, 2, 3, 1).reshape(len(images), 64, 3584)


class DynamicsPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(3584), nn.Linear(3584, 256), nn.GELU())
        self.temporal = nn.GRU(256, 256, num_layers=2, batch_first=True)
        self.predict = nn.Linear(256, 3584)
        nn.init.zeros_(self.predict.weight)
        nn.init.zeros_(self.predict.bias)

    def forward(self, context):
        last = F.layer_norm(context[-1].float(), (3584,))
        z = self.project(context.float()).transpose(0, 1)
        z, _ = self.temporal(z)
        return last+self.predict(z[:, -1])
