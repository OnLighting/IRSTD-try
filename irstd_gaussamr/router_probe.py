from __future__ import annotations

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage


def _gaussian_kernel1d(sigma: float) -> torch.Tensor:
    radius = math.ceil(3 * sigma)
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-(x * x) / (2 * sigma * sigma))
    return kernel / kernel.sum()


def fixed_subset(ids: list[str], size: int, seed: int) -> list[str]:
    ids = list(ids)
    random.Random(seed).shuffle(ids)
    return ids[:size]


def augment_pair(
    image: torch.Tensor,
    mask: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """G0's paired flips and 90-degree rotation, expressed on tensors."""
    if torch.rand((), generator=generator) < 0.5:
        image, mask = image.flip(-1), mask.flip(-1)
    if torch.rand((), generator=generator) < 0.5:
        image, mask = image.flip(-2), mask.flip(-2)
    turns = int(torch.randint(0, 4, (1,), generator=generator).item())
    return torch.rot90(image, turns, (-2, -1)), torch.rot90(mask, turns, (-2, -1))


class GaussianFeatureBank(nn.Module):
    """Fixed raw/Gaussian/standardized-DoG features from the V1 design."""

    def __init__(self, sigmas=(0.8, 1.2, 1.8, 2.6), kappa: float = 1.6, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.n_scales = len(sigmas)
        for i, sigma in enumerate(sigmas):
            self.register_buffer(f"smooth_{i}", _gaussian_kernel1d(sigma))
            self.register_buffer(f"surround_{i}", _gaussian_kernel1d(kappa * sigma))

    @staticmethod
    def _blur(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        radius = kernel.numel() // 2
        horizontal = kernel.view(1, 1, 1, -1)
        vertical = kernel.view(1, 1, -1, 1)
        image = F.conv2d(F.pad(image, (radius, radius, 0, 0), mode="replicate"), horizontal)
        return F.conv2d(F.pad(image, (0, 0, radius, radius), mode="replicate"), vertical)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError("GaussianFeatureBank expects (B,1,H,W)")
        smooth, dogs = [], []
        for i in range(self.n_scales):
            center = self._blur(image, getattr(self, f"smooth_{i}"))
            surround_kernel = getattr(self, f"surround_{i}")
            surround = self._blur(image, surround_kernel)
            variance = (self._blur(image.square(), surround_kernel) - surround.square()).clamp_min(self.eps)
            smooth.append(center)
            dogs.append((center - surround) / variance.sqrt())
        return torch.cat([image, *smooth, *dogs], dim=1)


class _DepthwiseSeparableBlock(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(1, out_ch),
            nn.SiLU(inplace=True),
        )


class GaussianRouter(nn.Module):
    def __init__(self, in_ch: int = 9, widths=(16, 24, 32)):
        super().__init__()
        layers = []
        for out_ch in widths:
            layers.append(_DepthwiseSeparableBlock(in_ch, out_ch))
            in_ch = out_ch
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(in_ch, 6, 1)
        with torch.no_grad():
            self.head.bias[0] = -4.595  # ~1% foreground prior.

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        head = self.head(self.body(features))
        return {
            "objectness_logit": head[:, 0:1],
            "offset_xy": 0.5 * torch.tanh(head[:, 1:3]),
            "log_sigma_xy": head[:, 3:5],
            "uncertainty_logit": head[:, 5:6],
        }


def extract_instances(mask: torch.Tensor) -> torch.Tensor:
    """Return 8-connected mask components as [mu_x,mu_y,sigma_x,sigma_y]."""
    array = mask.detach().cpu().numpy().squeeze()
    if array.ndim != 2:
        raise ValueError("mask must have shape (H,W) or (1,H,W)")
    labels, count = ndimage.label(array > 0.5, structure=torch.ones(3, 3).numpy())
    instances = []
    for label_id in range(1, count + 1):
        ys, xs = (labels == label_id).nonzero()
        mu_x, mu_y = float(xs.mean()), float(ys.mean())
        if xs.size == 1:
            sigma_x = sigma_y = 0.75
        else:
            sigma_x = min(8.0, max(0.5, float(((xs - mu_x) ** 2).mean() ** 0.5)))
            sigma_y = min(8.0, max(0.5, float(((ys - mu_y) ** 2).mean() ** 0.5)))
        instances.append((mu_x, mu_y, sigma_x, sigma_y))
    return torch.tensor(instances, dtype=torch.float32).reshape(-1, 4)


def router_loss(
    maps: dict[str, torch.Tensor],
    instances: torch.Tensor,
    stride: int = 8,
    match_radius: float = 12.0,
) -> dict[str, torch.Tensor]:
    """Router-only V1 losses for one image."""
    logits = maps["objectness_logit"]
    if logits.shape[0] != 1:
        raise ValueError("router probe uses batch size 1 for variable image sizes")
    _, _, height, width = logits.shape
    instances = instances.to(logits.device)
    target = torch.zeros_like(logits)
    valid = torch.ones_like(logits, dtype=torch.bool)

    if len(instances):
        gx = torch.floor(instances[:, 0] / stride).long().clamp(0, width - 1)
        gy = torch.floor(instances[:, 1] / stride).long().clamp(0, height - 1)
        for x, y in zip(gx.tolist(), gy.tolist()):
            valid[:, :, max(0, y - 1):min(height, y + 2), max(0, x - 1):min(width, x + 2)] = False
        target[0, 0, gy, gx] = 1
        valid[0, 0, gy, gx] = True

    probability = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = probability * target + (1 - probability) * (1 - target)
    alpha_t = 0.75 * target + 0.25 * (1 - target)
    focal = (alpha_t * (1 - p_t).square() * ce)[valid].sum() / max(1, len(instances))

    zero = logits.sum() * 0
    center = sigma = coverage = zero
    if len(instances):
        offset = maps["offset_xy"][0, :, gy, gx].transpose(0, 1)
        cells = torch.stack((gx, gy), dim=1).to(logits.dtype)
        predicted_center = stride * (cells + 0.5 + offset)
        center = F.smooth_l1_loss(predicted_center / stride, instances[:, :2] / stride)

        predicted_sigma = (F.softplus(maps["log_sigma_xy"][0, :, gy, gx].transpose(0, 1)) + 0.5).clamp(0.5, 8.0)
        sigma = F.smooth_l1_loss(predicted_sigma.log(), instances[:, 2:].log())

        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=logits.device, dtype=logits.dtype),
            torch.arange(width, device=logits.device, dtype=logits.dtype),
            indexing="ij",
        )
        grid = stride * torch.stack((grid_x.flatten() + 0.5, grid_y.flatten() + 0.5), dim=1)
        near = torch.cdist(instances[:, :2], grid) <= match_radius
        nearby_probability = probability.flatten().expand(len(instances), -1).masked_fill(~near, -1)
        coverage = -nearby_probability.max(dim=1).values.clamp_min(1e-6).log().mean()

    total = focal + center + 0.25 * sigma + coverage
    return {"total": total, "focal": focal, "center": center, "sigma": sigma, "coverage": coverage}


def proposal_diagnostics(
    proposals: torch.Tensor,
    instances: torch.Tensor,
    match_radius: float = 12.0,
    active_threshold: float = 0.5,
) -> dict[str, float | int]:
    """Return additive counters for dataset-level router diagnostics."""
    if proposals.shape[0] != 1:
        raise ValueError("router probe diagnostics expect batch size 1")
    proposals = proposals[0]
    instances = instances.to(proposals.device)
    active = proposals[:, 0] >= active_threshold
    stats: dict[str, float | int] = {
        "targets": len(instances),
        "matched": 0,
        "center_error_sum": 0.0,
        "sigma_log_error_sum": 0.0,
        "active_positive": 0,
        "hard_negative": 0,
    }
    if not len(instances):
        stats["hard_negative"] = int(active.sum().item())
        return stats

    for k in sorted({min(8, len(proposals)), min(16, len(proposals)), len(proposals)}):
        distance = torch.cdist(instances[:, :2], proposals[:k, 1:3])
        stats[f"hits_at_{k}"] = int((distance.min(dim=1).values <= match_radius).sum().item())

    distance = torch.cdist(instances[:, :2], proposals[:, 1:3])
    nearest_distance, nearest_index = distance.min(dim=1)
    matched = nearest_distance <= match_radius
    stats["matched"] = int(matched.sum().item())
    stats["center_error_sum"] = float(nearest_distance[matched].sum().item())
    if matched.any():
        predicted_sigma = proposals[:, 3:5][nearest_index[matched]].clamp_min(1e-6)
        target_sigma = instances[matched, 2:].clamp_min(1e-6)
        stats["sigma_log_error_sum"] = float((predicted_sigma.log() - target_sigma.log()).abs().mean(dim=1).sum().item())

    proposal_is_positive = distance.min(dim=0).values <= match_radius
    stats["active_positive"] = int((active & proposal_is_positive).sum().item())
    stats["hard_negative"] = int((active & ~proposal_is_positive).sum().item())
    return stats


def decode_proposals(maps: dict[str, torch.Tensor], k: int = 16, stride: int = 8) -> torch.Tensor:
    """Decode exactly k locally maximal router cells as [score,x,y,sx,sy,u]."""
    logits = maps["objectness_logit"]
    batch, _, height, width = logits.shape
    selected = min(k, height * width)
    local_max = F.max_pool2d(logits, 3, stride=1, padding=1)
    suppressed = torch.where(logits >= local_max, logits, torch.full_like(logits, -1e9))
    values, indices = suppressed.flatten(1).topk(selected, dim=1)

    def gather(name: str) -> torch.Tensor:
        tensor = maps[name].flatten(2)
        return tensor.gather(2, indices[:, None].expand(-1, tensor.shape[1], -1)).transpose(1, 2)

    offset = gather("offset_xy")
    sigma = (F.softplus(gather("log_sigma_xy")) + 0.5).clamp(0.5, 8.0)
    uncertainty = torch.sigmoid(gather("uncertainty_logit")).squeeze(-1)
    grid_x = (indices % width).to(logits.dtype)
    grid_y = torch.div(indices, width, rounding_mode="floor").to(logits.dtype)
    mu_x = stride * (grid_x + 0.5 + offset[..., 0])
    mu_y = stride * (grid_y + 0.5 + offset[..., 1])
    proposals = torch.stack((torch.sigmoid(values), mu_x, mu_y, sigma[..., 0], sigma[..., 1], uncertainty), dim=-1)
    if selected < k:
        proposals = torch.cat((proposals, proposals.new_zeros(batch, k - selected, 6)), dim=1)
    return proposals
