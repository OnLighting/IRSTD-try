from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _context_crops(features: torch.Tensor, proposals: torch.Tensor) -> torch.Tensor:
    """Sample 64x64 input-pixel support as fixed 32x32 half-resolution crops."""
    batch, channels, _, _ = features.shape
    proposal_count = proposals.shape[1]
    half = F.interpolate(features, scale_factor=0.5, mode="bilinear", align_corners=False)
    height, width = half.shape[-2:]
    local = torch.arange(32, device=features.device, dtype=features.dtype) - 15.5
    local_y, local_x = torch.meshgrid(local, local, indexing="ij")
    center_x = (proposals[..., 1] - 0.5) / 2
    center_y = (proposals[..., 2] - 0.5) / 2
    sample_x = center_x[..., None, None] + local_x
    sample_y = center_y[..., None, None] + local_y
    normalized_x = 2 * sample_x / max(1, width - 1) - 1 if width > 1 else torch.zeros_like(sample_x)
    normalized_y = 2 * sample_y / max(1, height - 1) - 1 if height > 1 else torch.zeros_like(sample_y)
    grid = torch.stack((normalized_x, normalized_y), dim=-1).reshape(batch * proposal_count, 32, 32, 2)
    source = half[:, None].expand(-1, proposal_count, -1, -1, -1).reshape(
        batch * proposal_count, channels, height, width
    )
    return F.grid_sample(source, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


class _ResidualDepthwiseBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(1, channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ContextRefiner(nn.Module):
    """Correct K1 Gaussian proposals from shared half-resolution context crops."""

    def __init__(self, in_ch: int = 9, width: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, width, 1, bias=False),
            nn.GroupNorm(1, width),
            nn.SiLU(inplace=True),
            _ResidualDepthwiseBlock(width),
            _ResidualDepthwiseBlock(width),
            nn.AdaptiveAvgPool2d(4),
        )
        self.head = nn.Linear(width * 4 * 4, 6)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, features: torch.Tensor, proposals: torch.Tensor) -> torch.Tensor:
        batch, proposal_count, fields = proposals.shape
        if fields != 6 or features.shape[0] != batch:
            raise ValueError("expected features (B,C,H,W) and proposals (B,K,6)")
        correction = self.head(self.encoder(_context_crops(features, proposals)).flatten(1))
        correction = correction.reshape(batch, proposal_count, 6)
        score_logit = torch.logit(proposals[..., 0].clamp(1e-6, 1 - 1e-6)) + correction[..., 0]
        center = proposals[..., 1:3] + 4 * torch.tanh(correction[..., 1:3])
        log_sigma = proposals[..., 3:5].clamp_min(1e-6).log()
        sigma = (log_sigma + math.log(2) * torch.tanh(correction[..., 3:5])).exp().clamp(0.5, 8.0)
        uncertainty_logit = torch.logit(proposals[..., 5].clamp(1e-6, 1 - 1e-6)) + correction[..., 5]
        return torch.cat(
            (
                torch.sigmoid(score_logit)[..., None],
                center,
                sigma,
                torch.sigmoid(uncertainty_logit)[..., None],
            ),
            dim=-1,
        )


def _greedy_matches(proposals: torch.Tensor, instances: torch.Tensor, max_distance: float = 12.0):
    if not len(instances):
        empty = torch.empty(0, dtype=torch.long, device=proposals.device)
        return empty, empty
    with torch.no_grad():
        distance = torch.cdist(proposals[:, 1:3], instances[:, :2])
        used_proposals, used_targets, pairs = set(), set(), []
        for flat_index in distance.flatten().argsort().tolist():
            proposal_index, target_index = divmod(flat_index, len(instances))
            if distance[proposal_index, target_index] > max_distance:
                break
            if proposal_index not in used_proposals and target_index not in used_targets:
                used_proposals.add(proposal_index)
                used_targets.add(target_index)
                pairs.append((proposal_index, target_index))
    if not pairs:
        empty = torch.empty(0, dtype=torch.long, device=proposals.device)
        return empty, empty
    proposal_index, target_index = zip(*pairs)
    return (
        torch.tensor(proposal_index, dtype=torch.long, device=proposals.device),
        torch.tensor(target_index, dtype=torch.long, device=proposals.device),
    )


def context_refiner_loss(
    proposals: torch.Tensor,
    instances: torch.Tensor,
    max_distance: float = 12.0,
    match_from: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """One-to-one proposal classification and Gaussian-parameter losses."""
    if proposals.shape[0] != 1:
        raise ValueError("context probe uses batch size 1")
    proposals = proposals[0]
    instances = instances.to(proposals.device)
    matching_proposals = proposals if match_from is None else match_from[0]
    proposal_index, target_index = _greedy_matches(matching_proposals.detach(), instances, max_distance)
    target_objectness = torch.zeros(len(proposals), device=proposals.device, dtype=proposals.dtype)
    target_objectness[proposal_index] = 1
    logits = torch.logit(proposals[:, 0].clamp(1e-6, 1 - 1e-6))
    positive_count = len(proposal_index)
    pos_weight = (
        proposals.new_tensor((len(proposals) - positive_count) / positive_count)
        if positive_count else proposals.new_tensor(1.0)
    )
    classification = F.binary_cross_entropy_with_logits(
        logits, target_objectness, pos_weight=pos_weight
    )
    zero = proposals.sum() * 0
    center = sigma = isotropy = zero
    if len(proposal_index):
        selected = proposals[proposal_index]
        target = instances[target_index]
        center = F.smooth_l1_loss(selected[:, 1:3] / 8, target[:, :2] / 8)
        sigma = F.smooth_l1_loss(selected[:, 3:5].log(), target[:, 2:4].clamp_min(1e-6).log())
        isotropy = (selected[:, 3].log() - selected[:, 4].log()).abs().mean()
    total = 0.5 * classification + center + 0.25 * sigma + 0.05 * isotropy
    return {
        "total": total,
        "classification": classification,
        "center": center,
        "sigma": sigma,
        "isotropy": isotropy,
    }


def select_detail_proposals(proposals: torch.Tensor, k: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Select exactly K2 proposals by the V1 probability/uncertainty priority."""
    if proposals.ndim != 3 or proposals.shape[-1] != 6 or proposals.shape[1] < k:
        raise ValueError("expected proposals (B,K1,6) with K1 >= K2")
    priority = proposals[..., 0] * (1 + 0.5 * proposals[..., 5])
    indices = priority.topk(k, dim=1).indices
    selected = proposals.gather(1, indices[..., None].expand(-1, -1, 6))
    return selected, indices


def detail_crops(source: torch.Tensor, proposals: torch.Tensor, crop_size: int = 48) -> torch.Tensor:
    """Extract fixed full-resolution crops aligned with sparse composition."""
    batch, channels, height, width = source.shape
    proposal_count = proposals.shape[1]
    local = torch.arange(crop_size, device=source.device, dtype=source.dtype)
    local_y, local_x = torch.meshgrid(local, local, indexing="ij")
    origin = torch.floor(proposals[..., 1:3] - crop_size / 2)
    sample_x = origin[..., 0, None, None] + local_x
    sample_y = origin[..., 1, None, None] + local_y
    normalized_x = 2 * sample_x / max(1, width - 1) - 1 if width > 1 else torch.zeros_like(sample_x)
    normalized_y = 2 * sample_y / max(1, height - 1) - 1 if height > 1 else torch.zeros_like(sample_y)
    grid = torch.stack((normalized_x, normalized_y), dim=-1).reshape(
        batch * proposal_count, crop_size, crop_size, 2
    )
    expanded = source[:, None].expand(-1, proposal_count, -1, -1, -1).reshape(
        batch * proposal_count, channels, height, width
    )
    crops = F.grid_sample(expanded, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return crops.reshape(batch, proposal_count, channels, crop_size, crop_size)


def gaussian_patch_logits(
    proposals: torch.Tensor,
    crop_size: int = 48,
    background_logit: float = -12.0,
    residual_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return local Gaussian logits in the same coordinate frame as detail crops."""
    local = torch.arange(crop_size, device=proposals.device, dtype=proposals.dtype)
    local_y, local_x = torch.meshgrid(local, local, indexing="ij")
    origin = torch.floor(proposals[..., 1:3] - crop_size / 2)
    x = origin[..., 0, None, None] + local_x
    y = origin[..., 1, None, None] + local_y
    sigma = proposals[..., 3:5].clamp(0.5, 8.0)
    dx = (x - proposals[..., 1, None, None]) / sigma[..., 0, None, None]
    dy = (y - proposals[..., 2, None, None]) / sigma[..., 1, None, None]
    distance = dx.square() + dy.square()
    objectness = torch.logit(proposals[..., 0].clamp(1e-6, 1 - 1e-6))[..., None, None]
    inside = objectness - 0.5 * distance
    if residual_logits is not None:
        inside = inside + residual_logits[:, :, 0]
    logits = torch.where(
        distance <= 9,
        inside,
        torch.full_like(distance, background_logit),
    )
    return logits[:, :, None]


class _ConvBlock(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(1, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(1, out_ch),
            nn.SiLU(inplace=True),
        )


class DetailRefiner(nn.Module):
    """Shared K2 full-resolution residual U-Net for the staged feasibility probe."""

    def __init__(self, in_ch: int = 9, widths=(24, 48, 72)):
        super().__init__()
        self.enc1 = _ConvBlock(in_ch, widths[0])
        self.enc2 = _ConvBlock(widths[0], widths[1])
        self.bottleneck = _ConvBlock(widths[1], widths[2])
        self.dec2 = _ConvBlock(widths[2] + widths[1], widths[1])
        self.dec1 = _ConvBlock(widths[1] + widths[0], widths[0])
        self.output = nn.Conv2d(widths[0], 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: torch.Tensor, proposals: torch.Tensor) -> torch.Tensor:
        batch, proposal_count, _ = proposals.shape
        crops = detail_crops(features, proposals)
        x = crops.reshape(batch * proposal_count, features.shape[1], 48, 48)
        skip1 = self.enc1(x)
        skip2 = self.enc2(F.avg_pool2d(skip1, 2))
        x = self.bottleneck(F.avg_pool2d(skip2, 2))
        x = self.dec2(torch.cat((F.interpolate(x, size=skip2.shape[-2:], mode="bilinear", align_corners=False), skip2), dim=1))
        x = self.dec1(torch.cat((F.interpolate(x, size=skip1.shape[-2:], mode="bilinear", align_corners=False), skip1), dim=1))
        return self.output(x).reshape(batch, proposal_count, 1, 48, 48)
