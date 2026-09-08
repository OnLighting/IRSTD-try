from __future__ import annotations

import math

import torch
import torch.nn as nn


class SparseGaussianComposer(nn.Module):
    """Compose fixed-budget local Gaussian patches without dense convolution."""

    def __init__(self, crop_size: int = 48, background_logit: float = -12.0):
        super().__init__()
        self.crop_size = crop_size
        self.background_logit = background_logit

    def forward(self, proposals: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        height, width = output_size
        local = torch.arange(self.crop_size, device=proposals.device)
        local_y, local_x = torch.meshgrid(local, local, indexing="ij")
        outputs = []
        for image_proposals in proposals:
            mu = image_proposals[:, 1:3]
            sigma = image_proposals[:, 3:5].clamp(0.5, 8.0)
            origin = torch.floor(mu - self.crop_size / 2).long()
            x = origin[:, 0, None, None] + local_x
            y = origin[:, 1, None, None] + local_y
            dx = (x - mu[:, 0, None, None]) / sigma[:, 0, None, None]
            dy = (y - mu[:, 1, None, None]) / sigma[:, 1, None, None]
            mahalanobis_sq = dx.square() + dy.square()
            objectness_logit = torch.logit(image_proposals[:, 0].clamp(1e-6, 1 - 1e-6))[:, None, None]
            patch = torch.where(
                mahalanobis_sq <= 9,
                objectness_logit - 0.5 * mahalanobis_sq,
                torch.full_like(mahalanobis_sq, self.background_logit),
            )
            valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            flat_index = (y * width + x).flatten()
            canvas = proposals.new_full((height * width,), math.exp(self.background_logit))
            canvas = canvas.scatter_add(0, flat_index[valid.flatten()], patch.exp().flatten()[valid.flatten()])
            outputs.append(canvas.log().reshape(1, height, width))
        return torch.stack(outputs)
