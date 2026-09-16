from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .composer import SparseGaussianComposer
from .refiners import DetailRefiner, gaussian_patch_logits, select_detail_proposals
from .router_probe import GaussianFeatureBank, GaussianRouter, decode_proposals


class GaussAMRV1(nn.Module):
    """Packaged GaussAMR V1 inference path."""

    def __init__(self):
        super().__init__()
        self.feature_bank = GaussianFeatureBank()
        self.router = GaussianRouter()
        self.detail_refiner = DetailRefiner()
        self.composer = SparseGaussianComposer()

    @classmethod
    def from_probe_checkpoint(cls, path, map_location="cpu") -> "GaussAMRV1":
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        model = cls()
        model.router.load_state_dict(checkpoint["router"])
        model.detail_refiner.load_state_dict(checkpoint["detail_refiner"])
        return model

    def forward(
        self,
        image: torch.Tensor,
        routing_mode: str = "predicted",
        targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        if routing_mode != "predicted":
            raise NotImplementedError(f"routing_mode={routing_mode!r} is not implemented")
        if targets is not None:
            raise ValueError("targets must not be passed in predicted routing mode")

        height, width = image.shape[-2:]
        pad_height = (-height) % 8
        pad_width = (-width) % 8
        padded = F.pad(image, (0, pad_width, 0, pad_height))

        features = self.feature_bank(padded)
        router_maps = self.router(features)
        proposals_l1 = decode_proposals(router_maps, k=24)
        proposals_l2, indices_l2 = select_detail_proposals(proposals_l1, k=8)
        residual_logits = self.detail_refiner(features, proposals_l2)
        local_logits = gaussian_patch_logits(
            proposals_l2, residual_logits=residual_logits
        )

        full_residual = residual_logits.new_zeros(
            residual_logits.shape[0], proposals_l1.shape[1], 1, 48, 48
        )
        scatter_index = indices_l2[..., None, None, None].expand_as(residual_logits)
        full_residual = full_residual.scatter(1, scatter_index, residual_logits)

        padded_size = padded.shape[-2:]
        gaussian_logits = self.composer(proposals_l1, padded_size)
        logits = self.composer(proposals_l1, padded_size, full_residual)
        return {
            "logits": logits[..., :height, :width],
            "gaussian_logits": gaussian_logits[..., :height, :width],
            "router_maps": router_maps,
            "proposals_l1": proposals_l1,
            "proposals_l2": proposals_l2,
            "local_logits": local_logits,
        }
