"""Losses that constrain the otherwise non-identifiable PSF decomposition.

v6 design (2026-09-07): eight physics-named addends with weight 1.0 except
for two engineering constants (rec support_weight=5.0 and psf usage
entropy coefficient 0.1). The earlier v5 ten-term loss was reduced to
this set after v5 in-domain results were clean but the public ``U`` was
negative-Spearman against reconstruction error, ``S`` failed the
horizontal-flip correlation gate, and five IRSTD-1K samples hit
``centroid_recall=0`` with ``presence_at_centroid=1.0`` because the
``presence`` BCE had no geometric signal.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


LOSS_NAMES = ("rec", "bg", "sp", "ctr", "psf", "ind", "flip", "amp")


def _require_keys(container: Mapping, names: set[str], label: str) -> None:
    missing = names - set(container)
    if missing:
        raise KeyError(f"{label} missing required keys: {sorted(missing)}")


class APSFUnmixingLoss(nn.Module):
    """Eight-term decomposition objective. All weights are 1.0."""

    def __init__(
        self,
        weights: Mapping[str, float],
        sigma_min: float = 0.6,
        sigma_max: float = 4.0,
    ) -> None:
        super().__init__()
        missing = set(LOSS_NAMES) - set(weights)
        extra = set(weights) - set(LOSS_NAMES)
        if missing or extra:
            raise ValueError(
                f"loss weights mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
            )
        if any(float(weights[name]) < 0 for name in LOSS_NAMES):
            raise ValueError("loss weights must be non-negative")
        if not 0 < float(sigma_min) < float(sigma_max):
            raise ValueError("sigma bounds must satisfy 0 < sigma_min < sigma_max")
        self.weights = {name: float(weights[name]) for name in LOSS_NAMES}
        self._sigma_min = float(sigma_min)
        self._sigma_max = float(sigma_max)

    # ------------------------------------------------------------------
    # rec — area-weighted reconstruction (target support upweighted 5x)
    # ------------------------------------------------------------------
    @staticmethod
    def _rec(image: Tensor, prediction: Mapping[str, Tensor], support: Tensor) -> Tensor:
        support_weight = 5.0
        recon = prediction["reconstruction_raw"]
        inside_err = (image - recon).abs() * support
        outside_err = (image - recon).abs() * (1.0 - support)
        pixel_count = float(image.shape[-1] * image.shape[-2])
        return (
            inside_err.sum(dim=(-1, -2, -3)) * support_weight
            + outside_err.sum(dim=(-1, -2, -3))
        ).sum() / pixel_count

    # ------------------------------------------------------------------
    # bg — outside-target smoothness (TV outside support only)
    # ------------------------------------------------------------------
    @staticmethod
    def _bg(background: Tensor, support: Tensor) -> Tensor:
        dx = (background[..., :, 1:] - background[..., :, :-1]).abs()
        dy = (background[..., 1:, :] - background[..., :-1, :]).abs()
        dx_mask = 1.0 - support[..., :, 1:]
        dy_mask = 1.0 - support[..., 1:, :]
        return (dx * dx_mask).mean() + (dy * dy_mask).mean()

    # ------------------------------------------------------------------
    # sp — log-mass sparsity
    # ------------------------------------------------------------------
    @staticmethod
    def _sp(source: Tensor, center: Tensor) -> Tensor:
        target_mass = (source * center).sum(dim=(-1, -2, -3))
        outside_mass = (source * (1.0 - center)).sum(dim=(-1, -2, -3))
        return torch.log1p(outside_mass / target_mass.clamp_min(1e-8)).mean()

    # ------------------------------------------------------------------
    # ctr — geometric localisation (BCE + worst-centre local maximum)
    # ------------------------------------------------------------------
    @staticmethod
    def _ctr_bce(presence_logits: Tensor, center: Tensor) -> Tensor:
        return (
            center * F.softplus(-presence_logits)
            + (1.0 - center) * F.softplus(presence_logits)
        ).mean()

    @staticmethod
    def _ctr_local(presence_logits: Tensor, center: Tensor) -> Tensor:
        P = torch.sigmoid(presence_logits)
        # 11x11 window centred on output pixel (stride=1, padding=5).
        local_max = F.max_pool2d(P, kernel_size=11, stride=1, padding=5)
        penalty = F.relu(0.5 - local_max)
        # Worst-centre penalty, restricted to true centres via `center`.
        return (penalty * center).max()

    @classmethod
    def _ctr(cls, presence_logits: Tensor, center: Tensor) -> Tensor:
        return cls._ctr_bce(presence_logits, center) + cls._ctr_local(presence_logits, center)

    # ------------------------------------------------------------------
    # psf — boundary penalty + usage entropy
    # ------------------------------------------------------------------
    @staticmethod
    def _psf(params: Mapping[str, Tensor], psf_weights: Tensor, sigma_min: float, sigma_max: float) -> Tensor:
        sigma_x = params["sigma_x"]
        sigma_y = params["sigma_y"]
        span = sigma_max - sigma_min
        # Boundary penalty: relu on both sides of the [sigma_min, sigma_max] band.
        boundary = (
            F.relu(sigma_min - sigma_x)
            + F.relu(sigma_x - sigma_max)
            + F.relu(sigma_min - sigma_y)
            + F.relu(sigma_y - sigma_max)
        ).mean()
        # Usage entropy across the kernel axis, normalised to [0, 1].
        num_kernels = float(psf_weights.shape[1])
        usage = psf_weights.mean(dim=(0, 2, 3))
        usage_h = -(usage * torch.log(usage.clamp_min(1e-8))).sum() / math.log(num_kernels)
        # Entropy term is bounded in [0, 1]; boundary is typically < 0.1,
        # so a 0.1 coefficient keeps both on comparable scale.
        return boundary - 0.1 * usage_h

    # ------------------------------------------------------------------
    # ind — component non-collapse
    # ------------------------------------------------------------------
    @staticmethod
    def _ind(
        background: Tensor,
        psf: Tensor,
        residual: Tensor,
        support: Tensor,
    ) -> Tensor:
        overlap_bt = (background * psf).mean()
        overlap_rt = (residual * psf).mean()
        outside_R = (residual * (1.0 - support)).mean()
        return overlap_bt + overlap_rt + outside_R

    # ------------------------------------------------------------------
    # flip — input-flip equivariance on P and A
    # ------------------------------------------------------------------
    @staticmethod
    def _flip(prediction: Mapping[str, Tensor], flipped_prediction: Mapping[str, Tensor]) -> Tensor:
        if "presence_logits" not in flipped_prediction or "amplitude_logits" not in flipped_prediction:
            raise KeyError(
                "flipped_prediction must contain 'presence_logits' and 'amplitude_logits'; "
                "run model on flip(image, dims=[-1]) and pass the aux output"
            )
        P_orig = torch.sigmoid(prediction["presence_logits"])
        A_orig = torch.sigmoid(prediction["amplitude_logits"])
        P_flip = torch.sigmoid(flipped_prediction["presence_logits"])
        A_flip = torch.sigmoid(flipped_prediction["amplitude_logits"])
        p_diff = (P_flip - torch.flip(P_orig, dims=(-1,))).abs().mean()
        a_diff = (A_flip - torch.flip(A_orig, dims=(-1,))).abs().mean()
        return p_diff + a_diff

    # ------------------------------------------------------------------
    # amp — physical amplitude alignment (L1 to source_proxy)
    # ------------------------------------------------------------------
    @staticmethod
    def _amp(amplitude_logits: Tensor, source_proxy: Tensor, center: Tensor) -> Tensor:
        A = torch.sigmoid(amplitude_logits)
        return (center * (A - source_proxy).abs()).mean()

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        image: Tensor,
        mask: Tensor,
        prediction: Mapping[str, Tensor | Mapping[str, Tensor]],
        targets: Mapping[str, Tensor],
        flipped_prediction: Mapping[str, Tensor | Mapping[str, Tensor]] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if image.shape != mask.shape:
            raise ValueError("image and mask must have the same shape")

        _require_keys(
            prediction,
            {
                "B",
                "S",
                "presence_logits",
                "amplitude_logits",
                "T_psf_raw",
                "R",
                "psf_weights",
                "psf_params",
                "reconstruction_raw",
            },
            "prediction",
        )
        _require_keys(
            targets,
            {
                "center",
                "support",
                "psf_support",
                "source_proxy",
            },
            "targets",
        )

        background = prediction["B"]
        source = prediction["S"]
        presence_logits = prediction["presence_logits"]
        amplitude_logits = prediction["amplitude_logits"]
        psf = prediction["T_psf_raw"]
        residual = prediction["R"]
        psf_weights = prediction["psf_weights"]
        params = prediction["psf_params"]
        reconstruction = prediction["reconstruction_raw"]
        if not isinstance(params, Mapping):
            raise TypeError("psf_params must be a mapping")
        _require_keys(params, {"sigma_x", "sigma_y"}, "psf_params")

        support = targets["support"]
        center = targets["center"]
        source_proxy = targets["source_proxy"]

        rec = self._rec(image, prediction, support)
        bg = self._bg(background, support)
        sp = self._sp(source, center)
        ctr = self._ctr(presence_logits, center)
        psf_term = self._psf(params, psf_weights, self._sigma_min, self._sigma_max)
        ind = self._ind(background, psf, residual, support)
        amp = self._amp(amplitude_logits, source_proxy, center)
        if flipped_prediction is None:
            flip = torch.zeros((), device=image.device, dtype=image.dtype)
        else:
            flip = self._flip(prediction, flipped_prediction)

        raw_terms: dict[str, Tensor] = {
            "rec": rec,
            "bg": bg,
            "sp": sp,
            "ctr": ctr,
            "psf": psf_term,
            "ind": ind,
            "flip": flip,
            "amp": amp,
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