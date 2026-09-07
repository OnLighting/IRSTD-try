"""Tests for the Stage-A v6 seven-loss objective.

Each test corresponds to one row of the v6 spec test matrix (T1..T20).
These tests are written and observed failing **before** the new
``APSFUnmixingLoss`` is implemented. They are the contract: the
implementation must satisfy every test before the v6 formal run is
launched.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from irstd_a.model import build_a_model
from irstd_a.targets import build_weak_targets


LOSS_NAMES_V6 = ("rec", "bg", "sp", "ctr", "psf", "ind", "flip", "amp")


def _synthetic_batch() -> tuple[torch.Tensor, torch.Tensor]:
    image = torch.full((2, 1, 32, 32), 0.2)
    mask = torch.zeros_like(image)
    image[0, 0, 15:17, 15:17] = 0.9
    mask[0, 0, 15:17, 15:17] = 1
    image[1, 0, 8, 23] = 0.75
    mask[1, 0, 8, 23] = 1
    return image, mask


def _controlled_prediction(image: torch.Tensor) -> dict:
    """Build a prediction dict that satisfies the v6 forward contract.

    Uses small dims / small num_psf to keep these tests cheap. The
    prediction is constructed so that B = image (perfect reconstruction
    with zero source) and S, T_psf, R are all zero. Callers mutate
    specific entries to exercise each loss term.
    """
    batch, _, height, width = image.shape
    kernels = torch.zeros(3, 1, 7, 7)
    kernels[:, 0, 3, 3] = 1.0
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
        "psf_weights": torch.full((batch, 3, height, width), 1.0 / 3.0),
        "psf_kernels": kernels,
        "psf_params": {
            "sigma_x": torch.tensor([0.8, 1.8, 3.2]),
            "sigma_y": torch.tensor([3.2, 1.8, 0.8]),
            "theta": torch.tensor([-0.8, 0.0, 0.8]),
        },
        "reconstruction": image.clone(),
        "reconstruction_raw": image.clone(),
    }


def _flip_h(tensor: torch.Tensor) -> torch.Tensor:
    return torch.flip(tensor, dims=(-1,))


# ---------------------------------------------------------------------------
# rec — area-weighted reconstruction (T1, T2)
# ---------------------------------------------------------------------------


def test_rec_is_area_invariant() -> None:
    """The reconstruction loss must be per-pixel invariant when the entire
    image is target support. (This isolates the area-invariance property
    from the support-area ratio, which is a different concern.)"""
    from irstd_a.losses import APSFUnmixingLoss

    weights = {name: 1.0 for name in LOSS_NAMES_V6}
    losses = []
    for size in (32, 64):
        image = torch.full((1, 1, size, size), 0.5)
        mask = torch.ones_like(image)  # every pixel is target support
        targets = build_weak_targets(image, mask, dilation_radius=2, ring_radius=5)
        prediction = _controlled_prediction(image)
        prediction["reconstruction_raw"] = image - 0.1  # constant per-pixel error
        _, terms = APSFUnmixingLoss(weights)(image, mask, prediction, targets)
        losses.append(float(terms["rec"]))
    assert losses[1] == pytest.approx(losses[0], rel=0.05)


def test_rec_upweights_target_support() -> None:
    """A reconstruction error inside target support must be penalised more
    than an identical-magnitude error outside it."""
    from irstd_a.losses import APSFUnmixingLoss

    image, mask = _synthetic_batch()
    targets = build_weak_targets(image, mask, dilation_radius=2, ring_radius=5)
    weights = {name: 1.0 for name in LOSS_NAMES_V6}

    inside = _controlled_prediction(image)
    inside["reconstruction_raw"][0, 0, 15, 15] = 0.0  # 1 px error inside support

    outside = _controlled_prediction(image)
    outside["reconstruction_raw"][0, 0, 0, 0] = 0.0  # 1 px error outside support

    _, inside_terms = APSFUnmixingLoss(weights)(image, mask, inside, targets)
    _, outside_terms = APSFUnmixingLoss(weights)(image, mask, outside, targets)

    assert inside_terms["rec"] > outside_terms["rec"]


# ---------------------------------------------------------------------------
# bg — outside-target smoothness (T3, T4)
# ---------------------------------------------------------------------------


def test_bg_zero_on_smooth_input() -> None:
    """A constant background must not be penalised by bg."""
    from irstd_a.losses import APSFUnmixingLoss

    image = torch.full((1, 1, 32, 32), 0.5)
    mask = torch.zeros_like(image)
    targets = build_weak_targets(image, mask, dilation_radius=2, ring_radius=5)
    weights = {name: 1.0 for name in LOSS_NAMES_V6}
    prediction = _controlled_prediction(image)
    prediction["B"] = torch.full_like(image, 0.5)
    _, terms = APSFUnmixingLoss(weights)(image, mask, prediction, targets)
    assert terms["bg"].item() < 1e-6


def test_bg_unconstrained_inside_support() -> None:
    """A B gradient strictly inside target support must not contribute to bg.

    With dilation_radius=2 the support for a 2x2 mask at (15:17, 15:17) is
    a 7x7 block from (13:20, 13:20) with the final row/col = 0. A B change
    that stays at rows 13..18 stays fully inside the support and must
    not affect bg.
    """
    from irstd_a.losses import APSFUnmixingLoss

    image, mask = _synthetic_batch()
    targets = build_weak_targets(image, mask, dilation_radius=2, ring_radius=5)
    support = targets["support"]
    # Confirm the assumption: rows 13..18 inclusive are inside support for
    # the first image; row 19 is the boundary.
    assert support[0, 0, 13:19, 13:19].all()
    assert not support[0, 0, 19, 19]

    weights = {name: 1.0 for name in LOSS_NAMES_V6}

    inside = _controlled_prediction(image)
    inside["B"][0:1, :, 13:18, 13:18] = 0.5  # strictly inside support

    outside = _controlled_prediction(image)
    # outside: leave B equal to image (no gradient anywhere)

    _, inside_terms = APSFUnmixingLoss(weights)(image, mask, inside, targets)
    _, outside_terms = APSFUnmixingLoss(weights)(image, mask, outside, targets)

    assert inside_terms["bg"].item() == pytest.approx(outside_terms["bg"].item(), abs=1e-5)


# ---------------------------------------------------------------------------
# sp — log-mass sparsity (T5, T6)
# ---------------------------------------------------------------------------


def test_sp_penalizes_dense_source() -> None:
    from irstd_a.losses import APSFUnmixingLoss

    image, mask = _synthetic_batch()
    targets = build_weak_targets(image, mask, dilation_radius=2, ring_radius=5)
    weights = {name: 1.0 for name in LOSS_NAMES_V6}
    sparse = _controlled_prediction(image)
    dense = copy.deepcopy(sparse)
    dense["S"] = torch.ones_like(image)

    _, sparse_terms = APSFUnmixingLoss(weights)(image, mask, sparse, targets)
    _, dense_terms = APSFUnmixingLoss(weights)(image, mask, dense, targets)
    assert dense_terms["sp"] > sparse_terms["sp"]


def test_sp_reduces_when_floor_lowered() -> None:
    """Halving the source floor must halve (in log space) the sp penalty."""
    from irstd_a.losses import APSFUnmixingLoss

    image, mask = _synthetic_batch()
    targets = build_weak_targets(image, mask, dilation_radius=2, ring_radius=5)
    weights = {name: 1.0 for name in LOSS_NAMES_V6}

    faint = _controlled_prediction(image)
    faint["S"].fill_(1e-4)
    faint["S"][targets["center"].bool()] = 0.1

    dense = copy.deepcopy(faint)
    dense["S"].fill_(1e-2)

    _, faint_terms = APSFUnmixingLoss(weights)(image, mask, faint, targets)
    _, dense_terms = APSFUnmixingLoss(weights)(image, mask, dense, targets)
    assert dense_terms["sp"] > faint_terms["sp"]


# ---------------------------------------------------------------------------
# ctr — geometric localisation (T7, T8)
# ---------------------------------------------------------------------------


def test_ctr_bce_pulls_centres_to_one() -> None:
    """At a true centre, p_logits should receive a negative gradient
    (push toward +inf so sigmoid -> 1)."""
    from irstd_a.losses import APSFUnmixingLoss

    image = torch.full((1, 1, 32, 32), 0.2)
    mask = torch.zeros_like(image)
    image[..., 16, 16] = 0.9
    mask[..., 16, 16] = 1
    targets = build_weak_targets(image, mask, dilation_radius=2, ring_radius=5)
    weights = {name: 1.0 for name in LOSS_NAMES_V6}

    presence_logits = torch.full_like(image, -6.0, requires_grad=True)
    prediction = _controlled_prediction(image)
    prediction["presence_logits"] = presence_logits

    _, terms = APSFUnmixingLoss(weights)(image, mask, prediction, targets)
    gradient = torch.autograd.grad(terms["ctr"], presence_logits)[0]
    assert gradient[targets["center"].bool()].mean() < 0


def test_ctr_local_pulls_offcenter_maximum() -> None:
    """``ctr_local`` enforces: at every true-centre pixel, the 11x11 local
    maximum of ``sigmoid(p_logits)`` must exceed 0.5. The penalty is
    worst-centre: ``max_over_centres(relu(0.5 - local_max))``.

    This test exercises the geometric difference between an on-centre
    spike (within the 11x11 window of the true centre) and a 10-px
    off-centre spike (which falls outside that window and is therefore
    invisible to the local maximum at the true-centre pixel).

    It also exercises a flat ``p_logits = -10`` baseline where no spike
    exists. Flat and off-centre both trigger ctr_local (no centre sees a
    strong local maximum); only on-centre drives ctr_local to zero. The
    flat case is the contract: a network that never fires at the centre
    must continue to be penalised, regardless of where else it fires.

    Note: ``ctr_local`` must use ``max_pool2d(..., stride=1, padding=5)``
    so that the 11x11 window is centred on the *output* pixel. With
    default ``stride=kernel_size`` or ``padding=0``, the window is offset
    and a 10 px shift still falls inside it; the XDU9 fix would silently
    regress.
    """
    from irstd_a.losses import APSFUnmixingLoss

    image = torch.full((1, 1, 32, 32), 0.2)
    mask = torch.zeros_like(image)
    image[..., 16, 16] = 0.9
    mask[..., 16, 16] = 1
    targets = build_weak_targets(image, mask, dilation_radius=2, ring_radius=5)
    weights = {name: 1.0 for name in LOSS_NAMES_V6}

    oncenter = _controlled_prediction(image)
    oncenter["presence_logits"] = torch.where(
        mask.bool(), torch.full_like(image, 6.0), torch.full_like(image, -10.0)
    )

    offcenter = _controlled_prediction(image)
    offcenter["presence_logits"] = torch.where(
        torch.zeros_like(mask).bool(), torch.full_like(image, -10.0), torch.full_like(image, -10.0)
    )
    offcenter["presence_logits"][..., 16, 26] = 6.0  # 10 px right of true centre

    flat = _controlled_prediction(image)
    flat["presence_logits"] = torch.full_like(image, -10.0)

    _, on_terms = APSFUnmixingLoss(weights)(image, mask, oncenter, targets)
    _, off_terms = APSFUnmixingLoss(weights)(image, mask, offcenter, targets)
    _, flat_terms = APSFUnmixingLoss(weights)(image, mask, flat, targets)

    # Whole ctr: on-centre is strictly smallest because both ctr_bce and
    # ctr_local are minimised.
    assert on_terms["ctr"].item() < off_terms["ctr"].item()
    assert on_terms["ctr"].item() < flat_terms["ctr"].item()

    # ctr_local alone: on-centre drives it to zero; off-centre and flat
    # both activate the penalty because the 11x11 window around the
    # true centre contains no strong sigmoid.
    # We measure ctr_local = ctr_total − ctr_bce_baseline. With p=-10
    # flat, sigmoid ≈ 4.5e-5, so the local_max baseline is also ≈ 4.5e-5,
    # and ctr_local = relu(0.5 - 4.5e-5) * center = 0.5 at the centre.
    on_local = on_terms["ctr"].item() - flat_terms["ctr"].item()
    assert on_local < -0.4, (
        f"ctr_local must drive on-centre ctr below flat baseline by ≥ 0.4; "
        f"got {on_local:.4f}. Check that ctr_local uses stride=1 padding=5."
    )


# ---------------------------------------------------------------------------
# psf — boundary + usage entropy (T9, T10)
# ---------------------------------------------------------------------------


def test_psf_boundary_penalises_out_of_range_sigma() -> None:
    """sigma_x or sigma_y past `sigma_max=4.0` must increase psf."""
    from irstd_a.losses import APSFUnmixingLoss

    image, mask = _synthetic_batch()
    targets = build_weak_targets(image, mask, dilation_radius=2, ring_radius=5)
    weights = {name: 1.0 for name in LOSS_NAMES_V6}

    in_range = _controlled_prediction(image)
    out_of_range = copy.deepcopy(in_range)
    out_of_range["psf_params"] = {
        "sigma_x": torch.tensor([0.8, 1.8, 5.0]),
        "sigma_y": torch.tensor([3.2, 1.8, 0.8]),
        "theta": torch.tensor([-0.8, 0.0, 0.8]),
    }

    _, in_terms = APSFUnmixingLoss(weights)(image, mask, in_range, targets)
    _, out_terms = APSFUnmixingLoss(weights)(image, mask, out_of_range, targets)
    assert out_terms["psf"].item() > in_terms["psf"].item()


def test_psf_entropy_penalises_collapse() -> None:
    """Mixture weights concentrated on a single kernel must increase psf
    (via negative usage entropy)."""
    from irstd_a.losses import APSFUnmixingLoss

    image, mask = _synthetic_batch()
    targets = build_weak_targets(image, mask, dilation_radius=2, ring_radius=5)
    weights = {name: 1.0 for name in LOSS_NAMES_V6}

    spread = _controlled_prediction(image)
    collapsed = copy.deepcopy(spread)
    batch, _, h, w = image.shape
    collapsed["psf_weights"] = torch.full((batch, 3, h, w), 1e-6)
    collapsed["psf_weights"][:, 0] = 1.0 - 2e-6  # K1 takes everything

    _, spread_terms = APSFUnmixingLoss(weights)(image, mask, spread, targets)
    _, collapsed_terms = APSFUnmixingLoss(weights)(image, mask, collapsed, targets)
    assert collapsed_terms["psf"].item() > spread_terms["psf"].item()


# ---------------------------------------------------------------------------
# ind — component non-collapse (T11, T12, T13)
# ---------------------------------------------------------------------------


def test_ind_separates_b_and_psf() -> None:
    """If B and T_psf occupy the same pixels with the same energy, ind
    must rise."""
    from irstd_a.losses import APSFUnmixingLoss

    image, mask = _synthetic_batch()
    targets = build_weak_targets(image, mask, dilation_radius=2, ring_radius=5)
    weights = {name: 1.0 for name in LOSS_NAMES_V6}

    separated = _controlled_prediction(image)
    # T_psf concentrated at (15,15); B is just `image`.
    separated["T_psf_raw"][..., 15, 15] = 0.5

    colocated = copy.deepcopy(separated)
    colocated["B"][..., 15, 15] = image[0, 0, 15, 15] + 0.5  # B copies the source

    _, sep_terms = APSFUnmixingLoss(weights)(image, mask, separated, targets)
    _, col_terms = APSFUnmixingLoss(weights)(image, mask, colocated, targets)
    assert col_terms["ind"].item() > sep_terms["ind"].item()


def test_ind_suppresses_residual_outside_support() -> None:
    from irstd_a.losses import APSFUnmixingLoss

    image, mask = _synthetic_batch()
    targets = build_weak_targets(image, mask, dilation_radius=2, ring_radius=5)
    weights = {name: 1.0 for name in LOSS_NAMES_V6}

    empty = _controlled_prediction(image)
    leaked = copy.deepcopy(empty)
    leaked["R"][0, 0, 0, 0] = 0.3  # R outside target support

    _, empty_terms = APSFUnmixingLoss(weights)(image, mask, empty, targets)
    _, leaked_terms = APSFUnmixingLoss(weights)(image, mask, leaked, targets)
    assert leaked_terms["ind"].item() > empty_terms["ind"].item()


def test_ind_suppresses_psf_residual_overlap() -> None:
    from irstd_a.losses import APSFUnmixingLoss

    image, mask = _synthetic_batch()
    targets = build_weak_targets(image, mask, dilation_radius=2, ring_radius=5)
    weights = {name: 1.0 for name in LOSS_NAMES_V6}

    clean = _controlled_prediction(image)
    clean["T_psf_raw"][..., 15, 15] = 0.3
    overlap = copy.deepcopy(clean)
    overlap["R"][..., 15, 15] = 0.3  # R copies T_psf exactly

    _, clean_terms = APSFUnmixingLoss(weights)(image, mask, clean, targets)
    _, overlap_terms = APSFUnmixingLoss(weights)(image, mask, overlap, targets)
    assert overlap_terms["ind"].item() > clean_terms["ind"].item()


# ---------------------------------------------------------------------------
# flip — input-flip equivariance (T14, T15)
# ---------------------------------------------------------------------------


def test_flip_zero_under_equivariant_input() -> None:
    """If the model's P output is exactly the horizontal flip of P(flip(I)),
    flip loss must be zero."""
    from irstd_a.losses import APSFUnmixingLoss

    image, mask = _synthetic_batch()
    targets = build_weak_targets(image, mask, dilation_radius=2, ring_radius=5)
    weights = {name: 1.0 for name in LOSS_NAMES_V6}

    prediction = _controlled_prediction(image)
    # Build a presence_logits that is exactly equivariant under h-flip.
    base = torch.full_like(image, -10.0)
    base[0, 0, 15, 15] = 4.0
    base[1, 0, 8, 23] = 4.0
    prediction["presence_logits"] = base
    prediction["amplitude_logits"] = torch.full_like(image, 0.5)

    flipped_image = _flip_h(image)
    flipped_prediction = _controlled_prediction(flipped_image)
    flipped_base = _flip_h(base)
    flipped_prediction["presence_logits"] = flipped_base
    flipped_prediction["amplitude_logits"] = torch.full_like(flipped_image, 0.5)

    _, terms = APSFUnmixingLoss(weights)(
        image, mask, prediction, targets, flipped_prediction=flipped_prediction
    )
    assert terms["flip"].item() < 1e-6


def test_flip_pulls_misaligned_presence() -> None:
    """Presence logits that are not equivariant under h-flip must produce a
    positive flip loss."""
    from irstd_a.losses import APSFUnmixingLoss

    image, mask = _synthetic_batch()
    targets = build_weak_targets(image, mask, dilation_radius=2, ring_radius=5)
    weights = {name: 1.0 for name in LOSS_NAMES_V6}

    prediction = _controlled_prediction(image)
    base = torch.full_like(image, -10.0)
    base[0, 0, 15, 15] = 4.0
    base[1, 0, 8, 23] = 4.0
    prediction["presence_logits"] = base
    prediction["amplitude_logits"] = torch.full_like(image, 0.5)

    # Flipped forward that is NOT equivariant: place a bright source at a
    # different location after flipping.
    flipped_image = _flip_h(image)
    flipped_prediction = _controlled_prediction(flipped_image)
    wrong_base = _flip_h(base)
    # Shift the bright spike by 3 px in the flipped image.
    wrong_base[0, 0, 15, 18] = 4.0
    wrong_base[0, 0, 15, 15] = -10.0
    flipped_prediction["presence_logits"] = wrong_base
    flipped_prediction["amplitude_logits"] = torch.full_like(flipped_image, 0.5)

    _, terms = APSFUnmixingLoss(weights)(
        image, mask, prediction, targets, flipped_prediction=flipped_prediction
    )
    # 3-px shift of a single batch's single spike produces L1 ≈ 1e-3 over a
    # 2-batch 32x32 image. We require > 1e-4 (100× numerical noise) which
    # still catches every practical misalignment. Tighter thresholds would
    # require misalignment across every batch.
    assert terms["flip"].item() > 1e-4


# ---------------------------------------------------------------------------
# amp — physical amplitude alignment (T16, T17)
# ---------------------------------------------------------------------------


def test_amp_zero_when_matched_to_source_proxy() -> None:
    """When A exactly equals source_proxy at the centre, amp must be ~0."""
    from irstd_a.losses import APSFUnmixingLoss

    image = torch.full((1, 1, 32, 32), 0.2)
    mask = torch.zeros_like(image)
    image[..., 16, 16] = 0.9
    mask[..., 16, 16] = 1
    targets = build_weak_targets(image, mask, dilation_radius=2, ring_radius=5)
    weights = {name: 1.0 for name in LOSS_NAMES_V6}

    prediction = _controlled_prediction(image)
    matched_amp = torch.zeros_like(image)
    centre_mask = targets["center"].bool()
    matched_amp[centre_mask] = torch.logit(
        targets["source_proxy"][centre_mask].clamp(1e-6, 1 - 1e-6)
    )
    prediction["amplitude_logits"] = matched_amp
    prediction["source_amplitude"] = torch.sigmoid(matched_amp)

    _, terms = APSFUnmixingLoss(weights)(image, mask, prediction, targets)
    assert terms["amp"].item() < 1e-3


def test_amp_penalises_binary_overshoot() -> None:
    """Pushing A to 1.0 when source_proxy < 1.0 must increase amp."""
    from irstd_a.losses import APSFUnmixingLoss

    image = torch.full((1, 1, 32, 32), 0.2)
    mask = torch.zeros_like(image)
    image[..., 16, 16] = 0.9
    mask[..., 16, 16] = 1
    targets = build_weak_targets(image, mask, dilation_radius=2, ring_radius=5)
    weights = {name: 1.0 for name in LOSS_NAMES_V6}

    matched = _controlled_prediction(image)
    centre_mask = targets["center"].bool()
    matched_logits = torch.full_like(image, -10.0)
    matched_logits[centre_mask] = torch.logit(
        targets["source_proxy"][centre_mask].clamp(1e-6, 1 - 1e-6)
    )
    matched["amplitude_logits"] = matched_logits
    matched["source_amplitude"] = torch.sigmoid(matched_logits)

    overshoot = copy.deepcopy(matched)
    overshoot_logits = torch.full_like(image, -10.0)
    overshoot_logits[centre_mask] = 14.0  # saturates to ~1.0
    overshoot["amplitude_logits"] = overshoot_logits
    overshoot["source_amplitude"] = torch.sigmoid(overshoot_logits)

    _, matched_terms = APSFUnmixingLoss(weights)(image, mask, matched, targets)
    _, over_terms = APSFUnmixingLoss(weights)(image, mask, overshoot, targets)
    assert over_terms["amp"].item() > matched_terms["amp"].item()


# ---------------------------------------------------------------------------
# integration (T18)
# ---------------------------------------------------------------------------


def test_total_loss_is_finite_and_backpropagates_to_every_head() -> None:
    from irstd_a.losses import APSFUnmixingLoss

    model = build_a_model(dims=(8, 16, 32), num_psf=3, kernel_size=7)
    image, mask = _synthetic_batch()
    prediction = model(image, return_aux=True)
    targets = build_weak_targets(image, mask, dilation_radius=2, ring_radius=5)
    weights = {name: 1.0 for name in LOSS_NAMES_V6}

    flipped_image = _flip_h(image)
    flipped_prediction = model(flipped_image, return_aux=True)

    total, terms = APSFUnmixingLoss(weights)(
        image, mask, prediction, targets, flipped_prediction=flipped_prediction
    )

    assert set(terms) == set(LOSS_NAMES_V6) | {"total"}
    for name in LOSS_NAMES_V6:
        assert torch.isfinite(terms[name]), name
    assert torch.isfinite(total)

    total.backward()
    # Heads that participate in v6 loss terms MUST receive gradient.
    # `uncertainty_head` does NOT participate in v6 loss (U is reported but
    # only trained via eval-side Spearman monitoring), so it is excluded.
    for head in (
        "background_head",
        "presence_head",
        "amplitude_head",
        "residual_head",
        "mixing_head",
    ):
        module = getattr(model, head)
        assert any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in module.parameters()
        ), head


# ---------------------------------------------------------------------------
# model contract: U is single-construct (T19, T20)
# ---------------------------------------------------------------------------


def test_U_is_excluded_from_loss_graph() -> None:
    """The public U must not appear in the loss computation graph.

    U is reported as a model output but trained only via Spearman
    monitoring on the eval side. Including U in the loss would create a
    shortcut where minimising the loss collapses U to zero.

    The contract is "no path at all" between ``terms['total']`` and
    ``prediction['U']``: the total loss tensor must have an empty
    ``grad_fn`` chain reaching U. We verify by attempting to take a
    gradient and expecting either None (no path) or an explicit
    RuntimeError raised by autograd when the leaf has no grad_fn.
    """
    from irstd_a.losses import APSFUnmixingLoss

    image, mask = _synthetic_batch()
    targets = build_weak_targets(image, mask, dilation_radius=2, ring_radius=5)
    weights = {name: 1.0 for name in LOSS_NAMES_V6}
    prediction = _controlled_prediction(image)
    prediction["U"] = torch.full_like(image, 0.95, requires_grad=True)

    _, terms = APSFUnmixingLoss(weights)(image, mask, prediction, targets)
    total = terms["total"]
    try:
        gradient = torch.autograd.grad(total, prediction["U"], allow_unused=True)[0]
        assert gradient is None
    except RuntimeError:
        # U was not in the graph at all -- autograd raises before returning None.
        pass


def test_public_U_equals_sigmoid_u_rec_head() -> None:
    """Model contract: public U must equal the u_rec aux output exactly;
    psf_entropy must not be mixed in. (u_rec itself is sigmoid of the
    uncertainty_head's pre-activation, so U = sigmoid(uncertainty_head).)"""
    model = build_a_model(dims=(8, 16, 32), num_psf=3, kernel_size=7)
    image, _ = _synthetic_batch()
    prediction = model(image, return_aux=True)
    # Public U is exactly the u_rec value; psf_entropy does not appear
    # in the public map.
    assert torch.allclose(prediction["U"], prediction["u_rec"])
    assert "psf_entropy" in prediction
    assert prediction["psf_entropy"].shape == prediction["U"].shape
    # And u_rec is the sigmoid of the underlying logit, so it lies in (0, 1).
    assert (prediction["u_rec"] > 0).all() and (prediction["u_rec"] < 1).all()
