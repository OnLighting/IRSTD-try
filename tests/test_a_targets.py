from __future__ import annotations

import pytest
import torch

from irstd_a.targets import (
    build_weak_targets,
    centroid_map,
    dilate_mask,
    split_ids,
)


def test_split_ids_is_disjoint_complete_and_deterministic() -> None:
    """Catches nondeterministic, overlapping, or incomplete train/val splits."""
    ids = [f"XDU{i}" for i in range(800)]

    train1, val1 = split_ids(ids, val_count=80, seed=42)
    train2, val2 = split_ids(ids, val_count=80, seed=42)

    assert (train1, val1) == (train2, val2)
    assert len(train1) == 720
    assert len(val1) == 80
    assert set(train1).isdisjoint(val1)
    assert set(train1) | set(val1) == set(ids)


def test_split_ids_rejects_duplicate_ids() -> None:
    """Catches split leakage hidden by duplicate sample identifiers."""
    with pytest.raises(ValueError, match="duplicate"):
        split_ids(["XDU1", "XDU1", "XDU2"], val_count=1, seed=42)


def test_split_ids_rejects_invalid_validation_size() -> None:
    """Catches empty train/validation partitions."""
    with pytest.raises(ValueError, match="val_count"):
        split_ids(["XDU1", "XDU2"], val_count=2, seed=42)


def test_centroid_map_marks_one_center_per_eight_connected_component() -> None:
    """Catches component counting or centroid placement errors."""
    mask = torch.zeros(1, 1, 16, 16)
    mask[..., 2:4, 2:4] = 1
    mask[..., 10:13, 11:14] = 1

    center = centroid_map(mask, radius=0)

    assert center.shape == mask.shape
    assert int(center.sum().item()) == 2
    assert center[0, 0, 2, 2].item() == 1
    assert center[0, 0, 11, 12].item() == 1


def test_centroid_map_uses_eight_connectivity() -> None:
    """Catches accidental four-connectivity for diagonally touching targets."""
    mask = torch.zeros(1, 1, 8, 8)
    mask[..., 2, 2] = 1
    mask[..., 3, 3] = 1

    center = centroid_map(mask, radius=0)

    assert int(center.sum().item()) == 1


def test_dilate_mask_expands_by_requested_chebyshev_radius() -> None:
    """Catches incorrect support dilation size."""
    mask = torch.zeros(1, 1, 9, 9)
    mask[..., 4, 4] = 1

    support = dilate_mask(mask, radius=2)

    assert int(support.sum().item()) == 25
    assert support[0, 0, 2, 2].item() == 1
    assert support[0, 0, 6, 6].item() == 1
    assert support[0, 0, 1, 4].item() == 0


def test_weak_target_recovers_positive_local_contrast() -> None:
    """Catches a target proxy that loses a bright point over flat background."""
    image = torch.full((1, 1, 17, 17), 0.2)
    image[..., 8, 8] = 0.9
    mask = torch.zeros_like(image)
    mask[..., 8, 8] = 1

    out = build_weak_targets(
        image,
        mask,
        dilation_radius=2,
        ring_radius=5,
    )

    assert set(out) == {
        "center",
        "support",
        "psf_support",
        "local_background",
        "target_proxy",
        "source_proxy",
    }
    assert all(value.shape == image.shape for value in out.values())
    assert out["center"].sum().item() == 1
    assert out["target_proxy"][0, 0, 8, 8].item() > 0.6
    assert torch.all(out["target_proxy"] >= 0)
    assert out["support"][0, 0, 8, 8].item() == 1
    assert out["target_proxy"][0, 0, 0, 0].item() == 0


def test_source_proxy_conserves_local_target_flux_at_the_centroid() -> None:
    """Catches treating physical source amplitude as a binary class label."""
    image = torch.full((1, 1, 17, 17), 0.2)
    image[..., 8, 8] = 0.9
    mask = torch.zeros_like(image)
    mask[..., 8, 8] = 1

    out = build_weak_targets(
        image,
        mask,
        dilation_radius=2,
        ring_radius=5,
        source_flux_scale=16.0,
    )

    assert out["source_proxy"].count_nonzero().item() == 1
    assert out["source_proxy"][0, 0, 8, 8].item() == pytest.approx(0.7 / 16.0)


def test_psf_support_covers_the_configured_kernel_footprint() -> None:
    """Catches scoring valid PSF tails as target-external leakage."""
    image = torch.full((1, 1, 21, 21), 0.2)
    mask = torch.zeros_like(image)
    mask[..., 10, 10] = 1

    out = build_weak_targets(
        image,
        mask,
        dilation_radius=2,
        ring_radius=5,
        psf_radius=7,
    )

    assert out["support"].sum().item() == 25
    assert out["psf_support"].sum().item() == 225
    assert out["psf_support"][0, 0, 10, 17].item() == 1


def test_weak_target_is_zero_for_dark_target_candidate() -> None:
    """Catches negative contrast leaking into the non-negative target proxy."""
    image = torch.full((1, 1, 17, 17), 0.8)
    image[..., 8, 8] = 0.1
    mask = torch.zeros_like(image)
    mask[..., 8, 8] = 1

    out = build_weak_targets(image, mask, dilation_radius=1, ring_radius=4)

    assert out["target_proxy"][0, 0, 8, 8].item() == 0


@pytest.mark.parametrize(
    ("image", "mask", "message"),
    [
        (torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 7, 8), "same shape"),
        (torch.zeros(1, 3, 8, 8), torch.zeros(1, 3, 8, 8), "single-channel"),
        (torch.full((1, 1, 8, 8), float("nan")), torch.zeros(1, 1, 8, 8), "finite"),
    ],
)
def test_weak_targets_reject_invalid_inputs(
    image: torch.Tensor,
    mask: torch.Tensor,
    message: str,
) -> None:
    """Catches malformed tensors before they contaminate decomposition losses."""
    with pytest.raises(ValueError, match=message):
        build_weak_targets(image, mask, dilation_radius=1, ring_radius=3)
