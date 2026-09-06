"""Losses that constrain the otherwise non-identifiable PSF decomposition."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


LOSS_NAMES = (
    "rec",
    "bg",
    "presence",
    "amplitude",
    "sparse",
    "target",
    "residual",
    "independence",
    "psf_diversity",
    "uncertainty",
)


def _masked_mean(value: Tensor, mask: Tensor, eps: float = 1e-6) -> Tensor:
    return (value * mask).sum() / mask.sum().clamp_min(eps)


def _edge_aware_tv(value: Tensor, reference: Tensor) -> Tensor:
    delta_x = value[..., :, 1:] - value[..., :, :-1]
    delta_y = value[..., 1:, :] - value[..., :-1, :]
    ref_x = reference[..., :, 1:] - reference[..., :, :-1]
    ref_y = reference[..., 1:, :] - reference[..., :-1, :]
    loss_x = (delta_x.abs() * torch.exp(-10.0 * ref_x.abs())).mean()
    loss_y = (delta_y.abs() * torch.exp(-10.0 * ref_y.abs())).mean()
    return loss_x + loss_y


def _high_frequency(value: Tensor) -> Tensor:
    return value - F.avg_pool2d(value, kernel_size=5, stride=1, padding=2)


class APSFUnmixingLoss(nn.Module):
    """Weighted reconstruction, weak-supervision, and anti-collapse objective."""

    def __init__(
        self,
        weights: Mapping[str, float],
    ) -> None:
        super().__init__()
        missing = set(LOSS_NAMES) - set(weights)
        extra = set(weights) - set(LOSS_NAMES)
        if missing or extra:
            raise ValueError(f"loss weights mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
        if any(float(weights[name]) < 0 for name in LOSS_NAMES):
            raise ValueError("loss weights must be non-negative")
        self.weights = {name: float(weights[name]) for name in LOSS_NAMES}

    @staticmethod
    def _require_keys(container: Mapping, names: set[str], label: str) -> None:
        missing = names - set(container)
        if missing:
            raise KeyError(f"{label} missing required keys: {sorted(missing)}")

    @staticmethod
    def _psf_diversity(params: Mapping[str, Tensor]) -> Tensor:
        sigma_x = params["sigma_x"]
        sigma_y = params["sigma_y"]
        theta = params["theta"]
        if sigma_x.numel() < 2:
            return sigma_x.new_zeros(())
        vectors = torch.stack(
            (
                (sigma_x - 0.6) / 3.4,
                (sigma_y - 0.6) / 3.4,
                torch.sin(theta),
                torch.cos(theta),
            ),
            dim=1,
        )
        distances = torch.pdist(vectors, p=2)
        return F.relu(0.15 - distances).mean()

    def forward(
        self,
        image: Tensor,
        mask: Tensor,
        prediction: Mapping[str, Tensor | Mapping[str, Tensor]],
        targets: Mapping[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        self._require_keys(
            prediction,
            {
                "B",
                "S",
                "presence_logits",
                "amplitude_logits",
                "T_psf_raw",
                "R",
                "u_rec",
                "psf_params",
                "reconstruction_raw",
            },
            "prediction",
        )
        self._require_keys(
            targets,
            {
                "center",
                "support",
                "psf_support",
                "local_background",
                "target_proxy",
                "source_proxy",
            },
            "targets",
        )
        if image.shape != mask.shape:
            raise ValueError("image and mask must have the same shape")

        background = prediction["B"]
        source = prediction["S"]
        presence_logits = prediction["presence_logits"]
        amplitude_logits = prediction["amplitude_logits"]
        psf = prediction["T_psf_raw"]
        residual = prediction["R"]
        uncertainty = prediction["u_rec"]
        reconstruction = prediction["reconstruction_raw"]
        params = prediction["psf_params"]
        if not all(isinstance(value, Tensor) for value in (background, source, psf, residual, uncertainty, reconstruction)):
            raise TypeError("spatial prediction values must be tensors")
        for name, logits in (("presence_logits", presence_logits), ("amplitude_logits", amplitude_logits)):
            if not isinstance(logits, Tensor) or logits.shape != source.shape:
                raise ValueError(f"prediction {name} must be a tensor matching S")
        if not isinstance(params, Mapping):
            raise TypeError("psf_params must be a mapping")
        self._require_keys(params, {"sigma_x", "sigma_y", "theta"}, "psf_params")

        support = targets["support"]
        outside = 1.0 - support
        psf_outside = 1.0 - targets["psf_support"]
        center = targets["center"]
        source_proxy = targets["source_proxy"]
        local_background = targets["local_background"]
        target_proxy = targets["target_proxy"]
        error = image - reconstruction
        scale = 0.01 + 0.49 * uncertainty.clamp(0.0, 1.0)

        rec = (error.abs() / scale + torch.log(scale)).mean()
        bg_outside = _masked_mean((background - image).abs(), outside)
        bg_inside = _masked_mean((background - local_background).abs(), support)
        bg = bg_outside + bg_inside + 0.1 * _edge_aware_tv(background, image)
        source_float = source.float()
        presence_loss = _masked_mean(
            F.softplus(-presence_logits.float()),
            center.float(),
        ) + _masked_mean(
            F.softplus(presence_logits.float()),
            1.0 - center.float(),
        )
        amplitude_loss = _masked_mean(
            F.binary_cross_entropy_with_logits(
                amplitude_logits.float(),
                source_proxy.float(),
                reduction="none",
            ),
            center.float(),
        )
        noncenter_mass = (source_float * (1.0 - center.float())).sum(dim=(-1, -2, -3))
        expected_source_mass = source_proxy.float().sum(dim=(-1, -2, -3))
        # The energy ratio is physically meaningful but can be hundreds at
        # sigmoid initialization.  log1p preserves ordering and a non-zero
        # drive toward sparsity without overwhelming rare center gradients.
        sparse = torch.log1p(
            noncenter_mass / expected_source_mass.clamp_min(1e-3)
        ).mean()

        target_output = psf + residual
        spatial_dims = (-1, -2, -3)
        proxy_energy = target_proxy.float().sum(dim=spatial_dims)
        safe_proxy_energy = proxy_energy.clamp_min(1e-6)
        target_fit = (
            (target_output.float() - target_proxy.float()).abs() * support.float()
        ).sum(dim=spatial_dims) / safe_proxy_energy
        target_energy = (target_output.float() * support.float()).sum(dim=spatial_dims)
        target_energy_error = (target_energy - proxy_energy).abs() / safe_proxy_energy
        target_leakage = torch.log1p(
            (target_output.float().abs() * psf_outside.float()).sum(dim=spatial_dims)
            / safe_proxy_energy
        )
        present_target_loss = target_fit + target_energy_error + 2.0 * target_leakage
        empty_target_loss = target_output.float().abs().mean(dim=spatial_dims)
        target = torch.where(
            proxy_energy > 1e-6,
            present_target_loss,
            empty_target_loss,
        ).mean()

        residual_loss = (
            2.0 * _masked_mean(residual.abs(), outside)
            + 0.1 * residual.abs().mean()
            + 0.2 * _edge_aware_tv(residual, image)
        )
        overlap = (psf.abs() * residual.abs()).sum() / torch.sqrt(
            psf.square().sum() * residual.square().sum()
        ).clamp_min(1e-6)
        background_detail = _high_frequency(background).abs()
        target_detail = target_output.abs()
        detail_overlap = (background_detail * target_detail).sum() / torch.sqrt(
            background_detail.square().sum() * target_detail.square().sum()
        ).clamp_min(1e-6)
        independence = overlap + 0.25 * detail_overlap
        psf_diversity = self._psf_diversity(params)

        detached_error = error.detach().abs()
        error_scale = detached_error / detached_error.amax(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
        uncertainty_loss = F.l1_loss(uncertainty, error_scale) + 0.1 * uncertainty.mean()

        raw_terms = {
            "rec": rec,
            "bg": bg,
            "presence": presence_loss,
            "amplitude": amplitude_loss,
            "sparse": sparse,
            "target": target,
            "residual": residual_loss,
            "independence": independence,
            "psf_diversity": psf_diversity,
            "uncertainty": uncertainty_loss,
        }
        for name, value in raw_terms.items():
            if not torch.isfinite(value):
                scalars = {
                    key: float(term.detach().cpu()) if torch.isfinite(term) else str(term.detach().cpu())
                    for key, term in raw_terms.items()
                }
                raise FloatingPointError(f"non-finite loss term {name}: {scalars}")
        total = sum(self.weights[name] * raw_terms[name] for name in LOSS_NAMES)
        if not torch.isfinite(total):
            raise FloatingPointError("non-finite weighted total loss")
        return total, {**raw_terms, "total": total}
