"""Exact block-diagonal visual SDPA without a quadratic cross-frame mask."""
import torch
import torch.nn.functional as F
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLVisionSdpaAttention, apply_rotary_pos_emb_vision,
)

ORIGINAL = Qwen2_5_VLVisionSdpaAttention.forward


def segmented(self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None):
    length = hidden_states.shape[0]
    q, k, v = self.qkv(hidden_states).reshape(length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    if position_embeddings is None:
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), -1)
        cos, sin = emb.cos(), emb.sin()
    else:
        cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
    # At 448px every image has the same grid, hence equal window/image lengths.
    # Cache the verified layout, not a mask. Variable grids remain supported by
    # separate exact attention calls; no padding or cross-segment visibility.
    cache = getattr(self, '_tactile_segment_layouts', None)
    if cache is None:
        cache = self._tactile_segment_layouts = {}
    layout_key = (length, cu_seqlens.numel())
    # Reusing a shape alone would be unsafe for arbitrary variable image grids.
    # The caller explicitly opts in only for fixed 448px grids; general layouts
    # inspect their actual boundaries on each call.
    fixed = getattr(self, '_tactile_fixed_448', False)
    boundaries = cache.get(layout_key) if fixed else None
    if boundaries is None:
        boundaries = tuple(cu_seqlens.tolist())
        assert boundaries[0] == 0 and boundaries[-1] == length
        if fixed:
            cache[layout_key] = boundaries
    lengths = [b-a for a, b in zip(boundaries, boundaries[1:])]
    assert lengths and min(lengths) > 0
    if len(set(lengths)) == 1:
        size = lengths[0]
        def batch(z):
            return z.reshape(len(lengths), size, self.num_heads, -1).transpose(1, 2)
        out = F.scaled_dot_product_attention(batch(q), batch(k), batch(v), dropout_p=0., is_causal=False)
        out = out.transpose(1, 2).reshape(length, -1)
    else:
        outputs = []
        for a, b in zip(boundaries, boundaries[1:]):
            out = F.scaled_dot_product_attention(q[a:b].transpose(0, 1).unsqueeze(0),
                k[a:b].transpose(0, 1).unsqueeze(0), v[a:b].transpose(0, 1).unsqueeze(0),
                dropout_p=0., is_causal=False)
            outputs.append(out.squeeze(0).transpose(0, 1).reshape(b-a, -1))
        out = torch.cat(outputs)
    return self.proj(out)


def enable(model):
    import types
    count = 0
    for module in model.modules():
        if isinstance(module, Qwen2_5_VLVisionSdpaAttention):
            module.forward = types.MethodType(segmented, module)
            module._tactile_fixed_448 = True
            count += 1
    assert count == 32, f'Expected the 32-block Qwen visual tower, got {count}'
