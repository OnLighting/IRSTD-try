from __future__ import annotations

import copy

import pytest
import torch

from irstd_a.losses import APSFUnmixingLoss
from irstd_a.model import build_a_model
from irstd_a.targets import build_weak_targets


def _weights() -> dict[str, float]:
    return {
        "rec": 1.0,
        "bg": 1.0,
        "presence": 0.5,
        "amplitude": 0.5,
        "sparse": 0.01,
        "target": 0.5,
        "residual": 0.05,
        "independence": 0.05,
        "psf_diversity": 0.01,
        "uncertainty": 0.01,
    }


def _synthetic_batch() -> tuple[torch.Tensor, torch.Tensor]:
    image = torch.full((2, 1, 32, 32), 0.2)
    mask = torch.zeros_like(image)
    image[0, 0, 15:17, 15:17] = 0.9
    mask[0, 0, 15:17, 15:17] = 1
    image[1, 0, 8, 23] = 0.75
    mask[1, 0, 8, 23] = 1
    return image, mask


def _controlled_prediction(image: torch.Tensor) -> dict:
    batch, _, height, width = image.shape
    kernels = torch.zeros(3, 1, 7, 7)
    kernels[:, 0, 3, 3] = 1
    return {
        "B": image.clone(),
        "S": torch.zeros_like(image),
        "presence_logits": torch.full_like(image, -20.0),
        "amplitude_logits": torch.zeros_like(image),
        "source_presence": torch.zeros_like(image),
        "source_amplitude": torch.full_like(image, 0.5),
        "T_psf": torch.zeros_like(image),
        "T_psf_raw": torch.zeros_like(image),
        "R": torch.zeros_like(image),
        "U": torch.full_like(image, 0.1),
        "u_rec": torch.full_like(image, 0.1),
        "psf_entropy": torch.zeros_like(image),
        "psf_weights": torch.full((batch, 3, height, width), 1 / 3),
        "psf_kernels": kernels,
        "psf_params": {
            "sigma_x": torch.tensor([0.8, 1.8, 3.2]),
            "sigma_y": torch.tensor([3.2, 1.8, 0.8]),
            "theta": torch.tensor([-0.8, 0.0, 0.8]),
        },
        "reconstruction": image.clone(),
        "reconstruction_raw": image.clone(),
    }


def test_total_loss_is_finite_and_backpropagates_to_every_head() -> None:
    model = build_a_model(dims=(8, 16, 32), num_psf=3, kernel_size=7)
    image, mask = _synthetic_batch()
    prediction = model(image, return_aux=True)
    targets = build_weak_targets(image, mask, dilation_radius=2, ring_radius=5)

    total, terms = APSFUnmixingLoss(_weights())(image, mask, prediction, targets)

    assert set(terms) == {
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
        "total",
    }
    assert torch.isfinite(total)
    total.backward()
    for name in (
        "background_head",
        "presence_head",
        "amplitude_head",
        "residual_head",
        "mixing_head",
        "uncertainty_head",
    ):
        module = getattr(model, name)
        assert any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in module.parameters()
        ), name


def test_dense_source_increases_sparse_penalty() -> None:
    image, mask = _synthetic_batch()
    targets = build_weak_targets(image, mask, dilation_radius=2, ring_radius=5)
    sparse = _controlled_prediction(image)
    dense = copy.deepcopy(sparse)
    dense["S"] = torch.ones_like(image)

    loss = APSFUnmixingLoss(_weights())
    _, sparse_terms = loss(image, mask, sparse, targets)
    _, dense_terms = loss(image, mask, dense, targets)

    assert dense_terms["sparse"] > sparse_terms["sparse"]


def test_amplitude_supervision_is_not_diluted_by_image_area() -> None:
    """Catches averaging rare positive centers over every background pixel."""
    losses = []
    for size in (32, 256):
        image = torch.full((1, 1, size, size), 0.2)
        mask = torch.zeros_like(image)
        midpoint = size // 2
        image[..., midpoint, midpoint] = 0.9
        mask[..., midpoint, midpoint] = 1
        targets = build_weak_targets(image, mask, dilation_radius=2, ring_radius=5)
        prediction = _controlled_prediction(image)
        prediction["amplitude_logits"].fill_(torch.logit(torch.tensor(0.1)))

        _, terms = APSFUnmixingLoss(_weights())(image, mask, prediction, targets)
        losses.append(float(terms["amplitude"]))

    assert losses[1] == pytest.approx(losses[0], rel=0.05)


def test_amplitude_loss_prefers_flux_matched_value_over_binary_one() -> None:
    """Catches forcing a physical source-strength map toward a binary label."""
    image = torch.full((1, 1, 32, 32), 0.2)
    mask = torch.zeros_like(image)
    image[..., 16, 16] = 0.9
    mask[..., 16, 16] = 1
    targets = build_weak_targets(image, mask, dilation_radius=2, ring_radius=5)
    matched = _controlled_prediction(image)
    matched["amplitude_logits"] = torch.logit(targets["source_proxy"].clamp(1e-6, 1 - 1e-6))
    binary = copy.deepcopy(matched)
    binary["amplitude_logits"][targets["center"].bool()] = 14.0

    loss = APSFUnmixingLoss(_weights())
    _, matched_terms = loss(image, mask, matched, targets)
    _, binary_terms = loss(image, mask, binary, targets)

    assert matched_terms["amplitude"] < binary_terms["amplitude"]


def test_presence_and_amplitude_have_independent_nonvanishing_gradients() -> None:
    """Catches coupling binary localization and fractional flux in one logit."""
    image = torch.full((1, 1, 32, 32), 0.2)
    mask = torch.zeros_like(image)
    image[..., 16, 16] = 0.9
    mask[..., 16, 16] = 1
    targets = build_weak_targets(image, mask, dilation_radius=2, ring_radius=5)
    prediction = _controlled_prediction(image)
    presence_logits = torch.full_like(image, -6.0, requires_grad=True)
    amplitude_logits = torch.zeros_like(image, requires_grad=True)
    prediction["presence_logits"] = presence_logits
    prediction["amplitude_logits"] = amplitude_logits
    prediction["source_presence"] = torch.sigmoid(presence_logits)
    prediction["source_amplitude"] = torch.sigmoid(amplitude_logits)
    prediction["S"] = prediction["source_presence"] * prediction["source_amplitude"]

    _, terms = APSFUnmixingLoss(_weights())(image, mask, prediction, targets)
    presence_gradient = torch.autograd.grad(terms["presence"], presence_logits, retain_graph=True)[0]
    amplitude_gradient = torch.autograd.grad(terms["amplitude"], amplitude_logits)[0]

    at_center = targets["center"].bool()
    assert presence_gradient[at_center].mean() < -0.9
    assert amplitude_gradient[at_center].mean() > 0.4


def test_presence_loss_pushes_noncentroid_false_activation_down() -> None:
    image = torch.full((1, 1, 32, 32), 0.2)
    mask = torch.zeros_like(image)
    image[..., 16, 16] = 0.9
    mask[..., 16, 16] = 1
    targets = build_weak_targets(image, mask, dilation_radius=2, ring_radius=5)
    prediction = _controlled_prediction(image)
    presence_logits = torch.full_like(image, -6.0, requires_grad=True)
    presence_logits = presence_logits + torch.zeros_like(image)
    prediction["presence_logits"] = presence_logits

    _, terms = APSFUnmixingLoss(_weights())(image, mask, prediction, targets)
    gradient = torch.autograd.grad(terms["presence"], presence_logits)[0]

    assert gradient[~targets["center"].bool()].mean() > 0


def test_sparse_penalty_tracks_noncentroid_source_energy_fraction() -> None:
    """Catches a tiny dense sigmoid floor looking sparse under a pixel mean."""
    image = torch.full((1, 1, 256, 256), 0.2)
    mask = torch.zeros_like(image)
    image[..., 128, 128] = 0.9
    mask[..., 128, 128] = 1
    targets = build_weak_targets(image, mask, dilation_radius=2, ring_radius=5)
    prediction = _controlled_prediction(image)
    prediction["S"].fill_(1e-4)
    prediction["S"][targets["center"].bool()] = 0.1

    _, terms = APSFUnmixingLoss(_weights())(image, mask, prediction, targets)

    assert terms["sparse"] > 0.95


def test_sparse_penalty_reduces_when_dense_sigmoid_floor_is_reduced() -> None:
    """Catches a scale-invariant mass ratio giving no incentive to lower a dense floor."""
    image = torch.full((1, 1, 32, 32), 0.2)
    mask = torch.zeros_like(image)
    image[..., 16, 16] = 0.9
    mask[..., 16, 16] = 1
    targets = build_weak_targets(image, mask, dilation_radius=2, ring_radius=5)
    faint = _controlled_prediction(image)
    faint["S"].fill_(1e-4)
    dense = copy.deepcopy(faint)
    dense["S"].fill_(1e-2)

    loss = APSFUnmixingLoss(_weights())
    _, faint_terms = loss(image, mask, faint, targets)
    _, dense_terms = loss(image, mask, dense, targets)

    assert dense_terms["sparse"] > 1.5 * faint_terms["sparse"]


def test_sparse_regularizer_does_not_overwhelm_dim_center_supervision() -> None:
    """Catches aggregate background mass pinning the shared source bias at -6."""
    image = torch.full((1, 1, 32, 32), 0.2)
    mask = torch.zeros_like(image)
    image[..., 16, 16] = 0.9
    mask[..., 16, 16] = 1
    targets = build_weak_targets(image, mask, dilation_radius=2, ring_radius=5)
    source_bias = torch.tensor(-6.0, requires_grad=True)
    prediction = _controlled_prediction(image)
    prediction["presence_logits"] = source_bias.expand_as(image)
    prediction["source_presence"] = torch.sigmoid(prediction["presence_logits"])
    prediction["S"] = prediction["source_presence"] * prediction["source_amplitude"]
    weights = _weights()

    _, terms = APSFUnmixingLoss(weights)(image, mask, prediction, targets)
    center_gradient = torch.autograd.grad(terms["presence"], source_bias, retain_graph=True)[0]
    sparse_gradient = torch.autograd.grad(terms["sparse"], source_bias)[0]

    assert weights["sparse"] * sparse_gradient.abs() < weights["presence"] * center_gradient.abs()


def test_full_image_residual_increases_residual_penalty() -> None:
    image, mask = _synthetic_batch()
    targets = build_weak_targets(image, mask, dilation_radius=2, ring_radius=5)
    empty = _controlled_prediction(image)
    copied = copy.deepcopy(empty)
    copied["R"] = image.clone()

    loss = APSFUnmixingLoss(_weights())
    _, empty_terms = loss(image, mask, empty, targets)
    _, copied_terms = loss(image, mask, copied, targets)

    assert copied_terms["residual"] > empty_terms["residual"]


def test_residual_is_free_to_explain_supported_non_psf_target_energy() -> None:
    """Catches directly suppressing R where an extended target may require it."""
    image = torch.full((1, 1, 32, 32), 0.2)
    mask = torch.zeros_like(image)
    image[..., 16, 16] = 0.9
    mask[..., 16, 16] = 1
    targets = build_weak_targets(image, mask, dilation_radius=2, ring_radius=5)
    inside = _controlled_prediction(image)
    inside["R"][..., 16, 16] = 0.2
    outside = _controlled_prediction(image)
    outside["R"][..., 0, 0] = 0.2
    loss = APSFUnmixingLoss(_weights())

    _, inside_terms = loss(image, mask, inside, targets)
    _, outside_terms = loss(image, mask, outside, targets)

    assert inside_terms["residual"] < 0.02
    assert outside_terms["residual"] > inside_terms["residual"]


def test_target_energy_outside_support_increases_target_penalty() -> None:
    image, mask = _synthetic_batch()
    targets = build_weak_targets(image, mask, dilation_radius=2, ring_radius=5)
    clean = _controlled_prediction(image)
    leaked = copy.deepcopy(clean)
    leaked["T_psf_raw"][..., 0, 0] = 1

    loss = APSFUnmixingLoss(_weights())
    _, clean_terms = loss(image, mask, clean, targets)
    _, leaked_terms = loss(image, mask, leaked, targets)

    assert leaked_terms["target"] > clean_terms["target"]


def test_target_leakage_is_not_diluted_by_unrelated_image_area() -> None:
    losses = []
    for size in (32, 256):
        image = torch.full((1, 1, size, size), 0.2)
        mask = torch.zeros_like(image)
        midpoint = size // 2
        image[..., midpoint, midpoint] = 0.9
        mask[..., midpoint, midpoint] = 1
        targets = build_weak_targets(image, mask, dilation_radius=2, ring_radius=5)
        prediction = _controlled_prediction(image)
        prediction["T_psf_raw"] = targets["target_proxy"].clone()
        prediction["T_psf_raw"][..., 0, 0] = 0.2
        _, terms = APSFUnmixingLoss(_weights())(image, mask, prediction, targets)
        losses.append(float(terms["target"]))

    assert losses[1] == pytest.approx(losses[0], rel=0.05)


def test_target_loss_directly_penalizes_energy_under_and_overshoot() -> None:
    image = torch.full((1, 1, 32, 32), 0.2)
    mask = torch.zeros_like(image)
    image[..., 16, 16] = 0.9
    mask[..., 16, 16] = 1
    targets = build_weak_targets(image, mask, dilation_radius=2, ring_radius=5)
    matched = _controlled_prediction(image)
    matched["T_psf_raw"] = targets["target_proxy"].clone()
    under = copy.deepcopy(matched)
    under["T_psf_raw"].zero_()
    over = copy.deepcopy(matched)
    over["T_psf_raw"] = 2.0 * targets["target_proxy"]
    loss = APSFUnmixingLoss(_weights())

    _, matched_terms = loss(image, mask, matched, targets)
    _, under_terms = loss(image, mask, under, targets)
    _, over_terms = loss(image, mask, over, targets)

    assert under_terms["target"] > matched_terms["target"] + 0.5
    assert over_terms["target"] > matched_terms["target"] + 0.5


def test_valid_psf_tail_inside_kernel_footprint_is_not_leakage() -> None:
    """Catches penalizing a valid 15x15 PSF tail using a 5x5 target support."""
    image = torch.full((1, 1, 32, 32), 0.2)
    mask = torch.zeros_like(image)
    image[..., 16, 16] = 0.9
    mask[..., 16, 16] = 1
    targets = build_weak_targets(
        image,
        mask,
        dilation_radius=2,
        ring_radius=5,
        psf_radius=7,
    )
    clean = _controlled_prediction(image)
    tailed = copy.deepcopy(clean)
    tailed["T_psf_raw"][..., 16, 22] = 0.2

    loss = APSFUnmixingLoss(_weights())
    _, clean_terms = loss(image, mask, clean, targets)
    _, tailed_terms = loss(image, mask, tailed, targets)

    assert tailed_terms["target"] == pytest.approx(clean_terms["target"])


def test_identical_psfs_increase_diversity_penalty() -> None:
    image, mask = _synthetic_batch()
    targets = build_weak_targets(image, mask, dilation_radius=2, ring_radius=5)
    diverse = _controlled_prediction(image)
    identical = copy.deepcopy(diverse)
    identical["psf_params"] = {
        "sigma_x": torch.ones(3),
        "sigma_y": torch.ones(3),
        "theta": torch.zeros(3),
    }

    loss = APSFUnmixingLoss(_weights())
    _, diverse_terms = loss(image, mask, diverse, targets)
    _, identical_terms = loss(image, mask, identical, targets)

    assert identical_terms["psf_diversity"] > diverse_terms["psf_diversity"]


def test_independence_overlap_is_not_amplified_by_image_area() -> None:
    image, mask = _synthetic_batch()
    targets = build_weak_targets(image, mask, dilation_radius=2, ring_radius=5)
    prediction = _controlled_prediction(image)
    prediction["T_psf_raw"][..., 15, 15] = 1
    prediction["R"][..., 15, 15] = 1

    _, terms = APSFUnmixingLoss(_weights())(image, mask, prediction, targets)

    assert 0.9 <= terms["independence"].item() <= 1.25


def test_uniformly_large_uncertainty_is_penalized() -> None:
    image, mask = _synthetic_batch()
    targets = build_weak_targets(image, mask, dilation_radius=2, ring_radius=5)
    small = _controlled_prediction(image)
    large = copy.deepcopy(small)
    large["u_rec"] = torch.full_like(image, 0.95)

    loss = APSFUnmixingLoss(_weights())
    _, small_terms = loss(image, mask, small, targets)
    _, large_terms = loss(image, mask, large, targets)

    assert large_terms["uncertainty"] > small_terms["uncertainty"]


def test_loss_rejects_missing_keys_and_nonfinite_values() -> None:
    image, mask = _synthetic_batch()
    targets = build_weak_targets(image, mask, dilation_radius=2, ring_radius=5)
    missing = _controlled_prediction(image)
    del missing["S"]
    with pytest.raises(KeyError, match="S"):
        APSFUnmixingLoss(_weights())(image, mask, missing, targets)

    bad = _controlled_prediction(image)
    bad["B"][..., 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="non-finite"):
        APSFUnmixingLoss(_weights())(image, mask, bad, targets)
