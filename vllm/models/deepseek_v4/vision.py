# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 Flash Vision-Exp vision tower and aligner.

Faithful port of the official reference ``inference/vision.py`` (DeepSeek
ViT with full bidirectional attention and 2D RoPE, then spatial r x r
downsampling + projection into the LM hidden size). Deliberately free of
vLLM imports so the modules can be instantiated and shape-checked
standalone on CPU.

The tower and aligner are replicated on every tensor-parallel rank. The
input is one variable-size image at a time (``patches`` plus its grid
dimensions), matching the reference; batching across images happens
outside this module.
"""

import torch
import torch.nn.functional as F
from torch import nn


def get_vision_cos_sin(
    n_h: int, n_w: int, dim: int, theta: float, device=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """2D RoPE cos/sin tables for an (n_h, n_w) patch grid."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    cos = freqs.cos().unsqueeze(1)
    sin = freqs.sin().unsqueeze(1)
    if device is not None:
        cos = cos.to(device)
        sin = sin.to(device)
    return cos, sin


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


class DeepseekV4VisionRMSNorm(nn.Module):
    """RMSNorm exactly as in the reference: fp32 weight and compute."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class DeepseekV4PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, vision_dim: int):
        super().__init__()
        self.proj = nn.Linear(3 * patch_size**2, vision_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(1))


class DeepseekV4VisionAttention(nn.Module):
    def __init__(self, vision_dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = vision_dim // n_heads
        self.wqkv = nn.Linear(vision_dim, 3 * vision_dim)
        self.wo = nn.Linear(vision_dim, vision_dim)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        n = x.size(0)
        q, k, v = (
            t.view(n, self.n_heads, self.head_dim)
            for t in self.wqkv(x).chunk(3, dim=-1)
        )
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        o = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        )
        return self.wo(o.transpose(0, 1).reshape(n, -1))


class DeepseekV4VisionMLP(nn.Module):
    def __init__(self, vision_dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(vision_dim, 2 * inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, vision_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class DeepseekV4VisionBlock(nn.Module):
    def __init__(self, vision_dim: int, n_heads: int, inter_dim: int):
        super().__init__()
        self.norm1 = DeepseekV4VisionRMSNorm(vision_dim)
        self.attn = DeepseekV4VisionAttention(vision_dim, n_heads)
        self.norm2 = DeepseekV4VisionRMSNorm(vision_dim)
        self.mlp = DeepseekV4VisionMLP(vision_dim, inter_dim)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class DeepseekV4ViT(nn.Module):
    """DeepSeek ViT: full bidirectional attention over one image with 2D RoPE."""

    def __init__(self, config):
        super().__init__()
        self.n_heads = config.vision_n_heads
        self.patch_size = config.vision_patch_size
        self.rope_dim = config.vision_dim // config.vision_n_heads // 2
        self.rope_theta = config.vision_rope_theta
        self.patch_embed = DeepseekV4PatchEmbed(
            config.vision_patch_size, config.vision_dim
        )
        self.blocks = nn.ModuleList(
            [
                DeepseekV4VisionBlock(
                    config.vision_dim,
                    config.vision_n_heads,
                    config.vision_inter_dim,
                )
                for _ in range(config.vision_n_layers)
            ]
        )
        self.norm = DeepseekV4VisionRMSNorm(config.vision_dim)
        self._rope_cache: dict[tuple, tuple] = {}

    def _get_cos_sin(
        self, n_h: int, n_w: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (int(n_h), int(n_w), str(device))
        entry = self._rope_cache.get(key)
        if entry is None:
            entry = get_vision_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta)
            self._rope_cache[key] = entry
        cos, sin = entry
        if cos.device != device:
            cos = cos.to(device)
            sin = sin.to(device)
            self._rope_cache[key] = (cos, sin)
        return cos, sin

    def forward(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        x = self.patch_embed(patches)
        cos, sin = self._get_cos_sin(n_h, n_w, x.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)


class DeepseekV4Aligner(nn.Module):
    """Spatial r x r downsampling plus projection to the LM hidden size."""

    def __init__(self, vision_dim: int, hidden_size: int, downsample_ratio: int):
        super().__init__()
        self.downsample_ratio = downsample_ratio
        in_dim = vision_dim * downsample_ratio**2
        self.w1 = nn.Linear(in_dim, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        r = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % r, 0, -n_h % r))
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(x)))
