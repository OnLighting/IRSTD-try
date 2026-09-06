"""G0 Swin-UNet style backbone.

Design constraints from docs/irstd_research_direction_decision_record.md:
- Keep high-resolution shallow branch (avoid repeated mean/max pooling that destroys
  small targets) -- decision §pain point 2.
- Use local window attention (Swin-style) for linear cost w.r.t. image size.
- I-only input; output P(Y|I) at original resolution.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Window attention (Swin-style, with relative position bias)
# -----------------------------------------------------------------------------

class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        # relative position bias table
        self.rel_bias = nn.Parameter(torch.zeros((2 * window_size - 1) ** 2, num_heads))
        nn.init.trunc_normal_(self.rel_bias, std=0.02)
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flat = coords.flatten(1)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[..., 0] += window_size - 1
        rel[..., 1] += window_size - 1
        rel_index = rel.sum(-1)
        self.register_buffer("rel_index", rel_index, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*num_windows, N, C)
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        bias = self.rel_bias[self.rel_index.view(-1)].view(N, N, -1).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(out)


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    # (B, H, W, C) -> (B*num_windows, window_size*window_size, C)
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class SwinBlock(nn.Module):
    """One Swin transformer block: window attn (optionally shifted) + MLP."""

    def __init__(self, dim: int, num_heads: int, window_size: int = 8, shift: int = 0, mlp_ratio: float = 2.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift = shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)
        if self.shift:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))
        x_windows = window_partition(x, self.window_size)
        attn_windows = self.attn(x_windows)
        x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(1, 2))
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


# -----------------------------------------------------------------------------
# Stem / Down / Up
# -----------------------------------------------------------------------------

class Stem(nn.Module):
    """Keep high resolution: one conv (no stride) on the input."""

    def __init__(self, in_ch: int, dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)  # (B, C, H, W)
        x = x.permute(0, 2, 3, 1)
        return self.norm(x)


class PatchDown(nn.Module):
    """Stride-2 patch merging that halves H,W and doubles channels (Swin style)."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduce = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        # pad if odd
        if H % 2 or W % 2:
            x = F.pad(x.permute(0, 3, 1, 2), (0, W % 2, 0, H % 2)).permute(0, 2, 3, 1)
            H, W = x.shape[1], x.shape[2]
        x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H // 2, W // 2, 4 * C)
        return self.reduce(self.norm(x))


class PatchUp(nn.Module):
    """2x upsample + reduce channels."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(dim_in, dim_out, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C)
        x = x.permute(0, 3, 1, 2)
        x = self.up(x)
        return self.norm(x.permute(0, 2, 3, 1))


# -----------------------------------------------------------------------------
# Stage = sequence of Swin blocks
# -----------------------------------------------------------------------------

class Stage(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                SwinBlock(dim, num_heads, window_size=window_size, shift=0 if i % 2 == 0 else window_size // 2)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


# -----------------------------------------------------------------------------
# G0 Swin-UNet
# -----------------------------------------------------------------------------

class G0SwinUNet(nn.Module):
    """Minimal Swin-UNet:

    stem (1/1, dim=C1) -> stage1
                       -> down (1/2, dim=C2) -> stage2
                                             -> down (1/4, dim=C3) -> stage3 (bottleneck)
                                                                        -> up (1/2, dim=C2) + skip stage2
                                                                                       -> up (1/1, dim=C1) + skip stage1
                                                                                                              -> head 1x1 -> sigmoid at inference

    Default dims keep params < 30M at 512x512.
    """

    def __init__(
        self,
        in_ch: int = 1,
        dims: Sequence[int] = (48, 96, 192),
        depths: Sequence[int] = (2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12),
        window_size: int = 8,
    ):
        super().__init__()
        c1, c2, c3 = dims
        d1, d2, d3 = depths
        h1, h2, h3 = num_heads

        self.stem = Stem(in_ch, c1)
        self.stage1 = Stage(c1, d1, h1, window_size)
        self.down1 = PatchDown(c1)  # -> c2
        self.stage2 = Stage(c2, d2, h2, window_size)
        self.down2 = PatchDown(c2)  # -> c3
        self.stage3 = Stage(c3, d3, h3, window_size)
        self.up1 = PatchUp(c3, c2)
        self.fuse1 = nn.Linear(c2 * 2, c2)
        self.stage4 = Stage(c2, d2, h2, window_size)
        self.up2 = PatchUp(c2, c1)
        self.fuse2 = nn.Linear(c1 * 2, c1)
        self.stage5 = Stage(c1, d1, h1, window_size)
        self.head = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W) -- H,W must be divisible by 4 and stage1 window_size.
        B, _, H, W = x.shape
        s1 = self.stage1(self.stem(x))            # (B, H,   W,   c1)
        s2 = self.stage2(self.down1(s1))          # (B, H/2, W/2, c2)
        s3 = self.stage3(self.down2(s2))          # (B, H/4, W/4, c3)

        u1 = self.up1(s3)                         # (B, H/2, W/2, c2)
        u1 = torch.cat([u1, s2], dim=-1)
        u1 = self.stage4(self.fuse1(u1))           # (B, H/2, W/2, c2)

        u2 = self.up2(u1)                         # (B, H, W, c1)
        u2 = torch.cat([u2, s1], dim=-1)
        u2 = self.stage5(self.fuse2(u2))          # (B, H, W, c1)

        out = self.head(u2.permute(0, 3, 1, 2))    # (B, 1, H, W)
        return out


def build_g0_model(in_ch: int = 1, dims=(48, 96, 192), depths=(2, 2, 2), num_heads=(3, 6, 12), window_size: int = 8) -> G0SwinUNet:
    return G0SwinUNet(in_ch=in_ch, dims=dims, depths=depths, num_heads=num_heads, window_size=window_size)


# -----------------------------------------------------------------------------
# Param/FLOPs probe (lazy import so model.py is importable without thop/fvcore)
# -----------------------------------------------------------------------------

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model: nn.Module, input_size=(1, 1, 512, 512)) -> float:
    """FLOPs via thop if installed; otherwise raise ImportError with install hint."""
    try:
        from thop import profile
    except ImportError as e:
        raise ImportError("Install `thop` for FLOPs counting: pip install thop") from e
    model.eval()
    device = next(model.parameters()).device
    dummy = torch.zeros(*input_size, device=device)
    flops, _ = profile(model, inputs=(dummy,), verbose=False)
    return float(flops)
