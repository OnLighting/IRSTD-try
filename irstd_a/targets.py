"""Weak decomposition targets built from images and training-only masks."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch import Tensor


def split_ids(ids: Sequence[str], val_count: int, seed: int) -> tuple[list[str], list[str]]:
    """Return a deterministic, disjoint train/validation partition."""
    clean = [str(sample_id) for sample_id in ids]
    if len(clean) != len(set(clean)):
        raise ValueError("ids contain duplicate sample identifiers")
    if not 0 < val_count < len(clean):
        raise ValueError("val_count must leave non-empty train and validation splits")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    order = torch.randperm(len(clean), generator=generator).tolist()
    val_indices = order[:val_count]
    train_indices = order[val_count:]
    return [clean[i] for i in train_indices], [clean[i] for i in val_indices]


def _validate_spatial_tensor(tensor: Tensor, name: str) -> None:
    if tensor.ndim != 4 or tensor.shape[1] != 1:
        raise ValueError(f"{name} must be a four-dimensional single-channel tensor")
    if not tensor.is_floating_point():
        raise ValueError(f"{name} must use a floating-point dtype")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")


def _validate_mask(mask: Tensor) -> None:
    _validate_spatial_tensor(mask, "mask")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("mask must be binary with values 0 or 1")


def dilate_mask(mask: Tensor, radius: int) -> Tensor:
    """Dilate a binary mask with a square (Chebyshev) neighborhood."""
    _validate_mask(mask)
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if radius == 0:
        return mask.clone()
    size = 2 * radius + 1
    return F.max_pool2d(mask, kernel_size=size, stride=1, padding=radius)


def centroid_map(mask: Tensor, radius: int = 1) -> Tensor:
    """Mark rounded centroids of 8-connected components in each batch item."""
    _validate_mask(mask)
    if radius < 0:
        raise ValueError("radius must be non-negative")

    result = torch.zeros_like(mask)
    masks = mask.detach().to(device="cpu", dtype=torch.uint8).numpy()
    structure = np.ones((3, 3), dtype=np.uint8)
    height, width = mask.shape[-2:]
    for batch_index in range(mask.shape[0]):
        labelled, count = ndimage.label(masks[batch_index, 0], structure=structure)
        if count == 0:
            continue
        centers = ndimage.center_of_mass(
            masks[batch_index, 0],
            labelled,
            range(1, count + 1),
        )
        for y_coord, x_coord in centers:
            y = min(height - 1, max(0, int(round(float(y_coord)))))
            x = min(width - 1, max(0, int(round(float(x_coord)))))
            result[batch_index, 0, y, x] = 1

    return dilate_mask(result, radius) if radius else result


def build_weak_targets(
    image: Tensor,
    mask: Tensor,
    dilation_radius: int,
    ring_radius: int,
    source_flux_scale: float = 16.0,
    psf_radius: int = 7,
) -> dict[str, Tensor]:
    """Create centroid, support, local-background, and target-contrast maps."""
    _validate_spatial_tensor(image, "image")
    _validate_mask(mask)
    if image.shape != mask.shape:
        raise ValueError("image and mask must have the same shape")
    if dilation_radius < 0:
        raise ValueError("dilation_radius must be non-negative")
    if ring_radius <= dilation_radius:
        raise ValueError("ring_radius must be greater than dilation_radius")
    if psf_radius < 0:
        raise ValueError("psf_radius must be non-negative")
    if not np.isfinite(source_flux_scale) or source_flux_scale <= 0:
        raise ValueError("source_flux_scale must be positive and finite")

    support = dilate_mask(mask, dilation_radius)
    psf_support = dilate_mask(mask, psf_radius)
    center = centroid_map(mask, radius=0)
    kernel_size = 2 * ring_radius + 1
    kernel = torch.ones(
        1,
        1,
        kernel_size,
        kernel_size,
        device=image.device,
        dtype=image.dtype,
    )
    valid_background = 1.0 - support
    numerator = F.conv2d(
        image * valid_background,
        kernel,
        padding=ring_radius,
    )
    denominator = F.conv2d(
        valid_background,
        kernel,
        padding=ring_radius,
    ).clamp_min(1.0)
    ring_background = numerator / denominator
    local_background = torch.where(support.bool(), ring_background, image)
    target_proxy = F.relu(image - local_background) * support
    local_target_flux = F.conv2d(target_proxy, kernel, padding=ring_radius)
    source_proxy = center * (local_target_flux / float(source_flux_scale)).clamp(0.0, 1.0)

    return {
        "center": center,
        "support": support,
        "psf_support": psf_support,
        "local_background": local_background,
        "target_proxy": target_proxy,
        "source_proxy": source_proxy,
    }
