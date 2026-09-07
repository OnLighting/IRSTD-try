"""Constrained image-only network for background-aware sparse PSF unmixing."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvGNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class DepthwiseResidualBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.pre = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(inplace=True),
        )
        self.mix = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return F.silu(inputs + self.mix(self.pre(inputs)))


class AnisotropicGaussianPSFBank(nn.Module):
    """Differentiable positive, unit-energy anisotropic Gaussian PSFs."""

    def __init__(
        self,
        num_kernels: int = 6,
        kernel_size: int = 15,
        sigma_min: float = 0.6,
        sigma_max: float = 4.0,
    ) -> None:
        super().__init__()
        if num_kernels <= 0:
            raise ValueError("num_kernels must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if not 0 < sigma_min < sigma_max:
            raise ValueError("sigma bounds must satisfy 0 < sigma_min < sigma_max")
        self.num_kernels = int(num_kernels)
        self.kernel_size = int(kernel_size)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

        fractions = torch.linspace(0.08, 0.92, self.num_kernels)
        reverse = torch.flip(fractions, dims=(0,))
        self.raw_sigma_x = nn.Parameter(torch.logit(fractions))
        self.raw_sigma_y = nn.Parameter(torch.logit(reverse))
        theta_fraction = torch.linspace(-0.75, 0.75, self.num_kernels)
        self.raw_theta = nn.Parameter(torch.atanh(theta_fraction))

        half = kernel_size // 2
        coordinates = torch.arange(-half, half + 1, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
        self.register_buffer("grid_x", grid_x, persistent=False)
        self.register_buffer("grid_y", grid_y, persistent=False)

    def kernels_and_params(self) -> tuple[Tensor, dict[str, Tensor]]:
        span = self.sigma_max - self.sigma_min
        sigma_x = self.sigma_min + span * torch.sigmoid(self.raw_sigma_x)
        sigma_y = self.sigma_min + span * torch.sigmoid(self.raw_sigma_y)
        theta = (math.pi / 2.0) * torch.tanh(self.raw_theta)

        cos_theta = torch.cos(theta)[:, None, None]
        sin_theta = torch.sin(theta)[:, None, None]
        x = self.grid_x[None]
        y = self.grid_y[None]
        rotated_x = cos_theta * x + sin_theta * y
        rotated_y = -sin_theta * x + cos_theta * y
        exponent = -0.5 * (
            (rotated_x / sigma_x[:, None, None]).square()
            + (rotated_y / sigma_y[:, None, None]).square()
        )
        kernels = torch.exp(exponent)
        kernels = kernels / kernels.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-12)
        return kernels[:, None], {
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "theta": theta,
        }

    def forward(
        self,
        source: Tensor,
        mixture_weights: Tensor,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        if source.ndim != 4 or source.shape[1] != 1:
            raise ValueError("source must have shape (N,1,H,W)")
        expected = (source.shape[0], self.num_kernels, *source.shape[-2:])
        if tuple(mixture_weights.shape) != expected:
            raise ValueError(f"mixture_weights must have shape {expected}")
        kernels, params = self.kernels_and_params()
        padding = self.kernel_size // 2
        responses = [
            F.conv2d(
                source * mixture_weights[:, index : index + 1],
                kernels[index : index + 1],
                padding=padding,
            )
            for index in range(self.num_kernels)
        ]
        return torch.stack(responses, dim=0).sum(dim=0), kernels, params


class APSFUnmixingNet(nn.Module):
    """Map one infrared image to explicit background/PSF/residual components."""

    def __init__(
        self,
        in_ch: int = 1,
        dims: Sequence[int] = (32, 64, 128),
        num_psf: int = 6,
        kernel_size: int = 15,
        sigma_min: float = 0.6,
        sigma_max: float = 4.0,
        source_flux_scale: float = 16.0,
    ) -> None:
        super().__init__()
        if in_ch != 1:
            raise ValueError("A supports exactly one input channel")
        if len(dims) != 3 or any(int(value) <= 0 for value in dims):
            raise ValueError("dims must contain three positive channel widths")
        c1, c2, c3 = (int(value) for value in dims)
        self.stem = nn.Sequential(ConvGNAct(in_ch, c1), DepthwiseResidualBlock(c1))
        self.down1 = nn.Sequential(
            ConvGNAct(c1, c2, stride=2),
            DepthwiseResidualBlock(c2),
        )
        self.down2 = nn.Sequential(
            ConvGNAct(c2, c3, stride=2),
            DepthwiseResidualBlock(c3),
            DepthwiseResidualBlock(c3),
        )
        self.decode2 = nn.Sequential(
            ConvGNAct(c3 + c2, c2),
            DepthwiseResidualBlock(c2),
        )
        self.decode1 = nn.Sequential(
            ConvGNAct(c2 + c1, c1),
            DepthwiseResidualBlock(c1),
        )
        self.background_head = nn.Conv2d(c1, 1, 1)
        self.presence_head = nn.Conv2d(c1, 1, 1)
        self.amplitude_head = nn.Conv2d(c1, 1, 1)
        self.residual_head = nn.Conv2d(c1, 1, 1)
        self.mixing_head = nn.Conv2d(c1, num_psf, 1)
        self.uncertainty_head = nn.Conv2d(c1, 1, 1)
        self.psf_bank = AnisotropicGaussianPSFBank(
            num_kernels=num_psf,
            kernel_size=kernel_size,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
        self.num_psf = int(num_psf)
        if source_flux_scale <= 0:
            raise ValueError("source_flux_scale must be positive")
        self.source_flux_scale = float(source_flux_scale)
        self._initialize_heads()

    def _initialize_heads(self) -> None:
        nn.init.zeros_(self.mixing_head.weight)
        nn.init.zeros_(self.mixing_head.bias)
        for head, bias in (
            (self.presence_head, -6.0),
            (self.amplitude_head, 0.0),
            (self.residual_head, -6.0),
            (self.uncertainty_head, -2.0),
        ):
            nn.init.normal_(head.weight, mean=0.0, std=0.01)
            nn.init.constant_(head.bias, bias)

    @staticmethod
    def _validate_image(image: Tensor) -> None:
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError("image must be a four-dimensional single-channel tensor")
        if not image.is_floating_point():
            raise ValueError("image must use a floating-point dtype")
        if not torch.isfinite(image).all():
            raise ValueError("image must contain only finite values")

    def forward(self, image: Tensor, return_aux: bool = False) -> dict[str, Tensor | dict[str, Tensor]]:
        self._validate_image(image)
        original_height, original_width = image.shape[-2:]
        pad_height = (-original_height) % 4
        pad_width = (-original_width) % 4
        padded = F.pad(image, (0, pad_width, 0, pad_height), mode="replicate")

        enc1 = self.stem(padded)
        enc2 = self.down1(enc1)
        bottleneck = self.down2(enc2)
        up2 = F.interpolate(bottleneck, size=enc2.shape[-2:], mode="bilinear", align_corners=False)
        dec2 = self.decode2(torch.cat((up2, enc2), dim=1))
        up1 = F.interpolate(dec2, size=enc1.shape[-2:], mode="bilinear", align_corners=False)
        features = self.decode1(torch.cat((up1, enc1), dim=1))

        background = torch.sigmoid(self.background_head(features))
        presence_logits = self.presence_head(features)
        amplitude_logits = self.amplitude_head(features)
        source_presence = torch.sigmoid(presence_logits)
        source_amplitude = torch.sigmoid(amplitude_logits)
        source = source_presence * source_amplitude
        source_flux = source * self.source_flux_scale
        residual = torch.sigmoid(self.residual_head(features))
        psf_weights = torch.softmax(self.mixing_head(features), dim=1)
        uncertainty_reconstruction = torch.sigmoid(self.uncertainty_head(features))
        psf_raw, kernels, params = self.psf_bank(source_flux, psf_weights)
        entropy = -(
            psf_weights * torch.log(psf_weights.clamp_min(1e-8))
        ).sum(dim=1, keepdim=True) / math.log(self.num_psf)
        # v6: public U carries exactly one construct -- the reconstruction
        # uncertainty head. PSF mixture entropy is exposed only as an aux
        # diagnostic. Mixing them in v5 made the loss unable to fix the
        # public U (Spearman = -0.21) because entropy contributed without
        # supervision.
        reconstruction_raw = background + psf_raw + residual

        def crop(value: Tensor) -> Tensor:
            return value[..., :original_height, :original_width]

        public: dict[str, Tensor | dict[str, Tensor]] = {
            "B": crop(background),
            "S": crop(source),
            "T_psf": crop(psf_raw.clamp(0.0, 1.0)),
            "R": crop(residual),
            "U": crop(uncertainty_reconstruction),
            "reconstruction": crop(reconstruction_raw.clamp(0.0, 1.0)),
        }
        if return_aux:
            public.update(
                {
                    "psf_weights": crop(psf_weights),
                    "presence_logits": crop(presence_logits),
                    "amplitude_logits": crop(amplitude_logits),
                    "source_presence": crop(source_presence),
                    "source_amplitude": crop(source_amplitude),
                    "S_flux": crop(source_flux),
                    "psf_kernels": kernels,
                    "psf_params": params,
                    "u_rec": crop(uncertainty_reconstruction),
                    "psf_entropy": crop(entropy),
                    "T_psf_raw": crop(psf_raw),
                    "reconstruction_raw": crop(reconstruction_raw),
                }
            )
        return public


def build_a_model(
    in_ch: int = 1,
    dims: Sequence[int] = (32, 64, 128),
    num_psf: int = 6,
    kernel_size: int = 15,
    sigma_min: float = 0.6,
    sigma_max: float = 4.0,
    source_flux_scale: float = 16.0,
) -> APSFUnmixingNet:
    return APSFUnmixingNet(
        in_ch=in_ch,
        dims=dims,
        num_psf=num_psf,
        kernel_size=kernel_size,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        source_flux_scale=source_flux_scale,
    )
