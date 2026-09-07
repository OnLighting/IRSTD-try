from __future__ import annotations

import math

import pytest
import torch

from irstd_a.diagnostics import (
    aggregate_diagnostics,
    compare_component_maps,
    component_diagnostics,
    degeneration_flags,
    _coverage_at_5px,
)


def _perfect_case() -> tuple[torch.Tensor, torch.Tensor, dict, dict]:
    image = torch.full((1, 1, 9, 9), 0.2)
    image[..., 4, 4] = 0.8
    mask = torch.zeros_like(image)
    mask[..., 4, 4] = 1
    background = torch.full_like(image, 0.2)
    source = torch.zeros_like(image)
    source[..., 4, 4] = 1
    psf = torch.zeros_like(image)
    psf[..., 4, 4] = 0.6
    residual = torch.zeros_like(image)
    uncertainty = torch.zeros_like(image)
    prediction = {
        "B": background,
        "S": source,
        "source_presence": source.clone(),
        "source_amplitude": source.clone(),
        "T_psf": psf,
        "R": residual,
        "U": uncertainty,
        "reconstruction": background + psf,
        "psf_weights": torch.ones(1, 1, 9, 9),
        "psf_entropy": torch.zeros_like(image),
        "psf_params": {
            "sigma_x": torch.tensor([1.0]),
            "sigma_y": torch.tensor([1.5]),
            "theta": torch.tensor([0.0]),
        },
    }
    targets = {
        "center": source.clone(),
        "support": mask.clone(),
        "psf_support": mask.clone(),
        "local_background": background.clone(),
        "target_proxy": psf.clone(),
        "source_proxy": source.clone(),
    }
    return image, mask, prediction, targets


def test_perfect_decomposition_has_exact_energy_metrics() -> None:
    image, mask, prediction, targets = _perfect_case()

    record = component_diagnostics(image, mask, prediction, targets, ["sample"])[0]

    assert record["sample_id"] == "sample"
    assert record["has_target"] is True
    assert record["gt_pixels"] == 1
    assert abs(record["target_proxy_energy"] - 0.6) < 1e-6
    assert record["reconstruction_mae"] < 1e-7
    assert math.isinf(record["reconstruction_psnr"])
    assert abs(record["target_energy_precision"] - 1.0) < 1e-6
    assert abs(record["target_contrast_recall"] - 1.0) < 1e-6
    assert abs(record["source_false_activation"] - 0.0) < 1e-6
    assert abs(record["centroid_recall_5px"] - 1.0) < 1e-6
    assert abs(record["background_target_leakage"] - 0.0) < 1e-6
    assert abs(record["residual_energy_ratio"] - 0.0) < 1e-6
    assert record["presence_at_centroid"] == pytest.approx(1.0)
    assert record["presence_false_activation"] == pytest.approx(0.0)
    assert record["amplitude_centroid_mae"] == pytest.approx(0.0)
    assert record["source_noncenter_mass_ratio"] == pytest.approx(0.0)
    assert record["target_outside_energy_ratio"] == pytest.approx(0.0)
    assert record["residual_meaningful"] is False


def test_dual_source_and_energy_diagnostics_have_hand_checked_ratios() -> None:
    image, mask, prediction, targets = _perfect_case()
    prediction["S"].zero_()
    prediction["S"][..., 4, 4] = 0.4
    prediction["S"][..., 0, 0] = 0.1
    prediction["source_presence"].zero_()
    prediction["source_presence"][..., 4, 4] = 0.8
    prediction["source_presence"][..., 0, 0] = 0.2
    prediction["source_amplitude"].zero_()
    prediction["source_amplitude"][..., 4, 4] = 0.5
    targets["source_proxy"].zero_()
    targets["source_proxy"][..., 4, 4] = 0.5
    prediction["T_psf"][..., 0, 0] = 0.3
    prediction["R"][..., 4, 4] = 0.006

    record = component_diagnostics(image, mask, prediction, targets, ["ratios"])[0]

    assert record["presence_at_centroid"] == pytest.approx(0.8)
    assert record["presence_false_activation"] == pytest.approx(0.2)
    assert record["amplitude_centroid_mae"] == pytest.approx(0.0)
    assert record["source_noncenter_mass_ratio"] == pytest.approx(0.2)
    assert record["target_outside_energy_ratio"] == pytest.approx(0.5)
    assert record["residual_meaningful"] is True


def test_centroid_recall_accepts_a_localized_subunit_source_peak() -> None:
    """Catches a fixed 0.5 threshold that hides valid dim source impulses."""
    image, mask, prediction, targets = _perfect_case()
    prediction["S"][..., 4, 4] = 0.2

    record = component_diagnostics(image, mask, prediction, targets, ["dim"])[0]

    assert record["centroid_recall_5px"] == 1.0


def test_residual_target_fraction_measures_the_target_support_shortcut() -> None:
    """Catches a global residual ratio hiding residual energy at the target."""
    image, mask, prediction, targets = _perfect_case()
    prediction["R"][..., 4, 4] = 0.2

    record = component_diagnostics(image, mask, prediction, targets, ["routed"])[0]

    assert record["residual_target_fraction"] == pytest.approx(0.25)


def test_target_precision_includes_valid_psf_kernel_tails() -> None:
    """Catches measuring PSF precision with the narrower source support."""
    image, mask, prediction, targets = _perfect_case()
    targets["psf_support"] = torch.ones_like(mask)
    prediction["T_psf"][..., 0, 0] = 0.2

    record = component_diagnostics(image, mask, prediction, targets, ["tail"])[0]

    assert record["target_energy_precision"] == pytest.approx(1.0)


def test_empty_target_uses_null_target_only_metrics() -> None:
    image, _, prediction, targets = _perfect_case()
    mask = torch.zeros_like(image)
    targets = {key: torch.zeros_like(value) for key, value in targets.items()}
    targets["local_background"] = image.clone()

    record = component_diagnostics(image, mask, prediction, targets, ["empty"])[0]

    assert record["has_target"] is False
    assert record["target_contrast_recall"] is None
    assert record["centroid_recall_5px"] is None
    assert record["background_target_leakage"] is None


def test_aggregate_reports_valid_counts_and_medians() -> None:
    records = [
        {"has_target": True, "reconstruction_mae": 0.1, "target_contrast_recall": 0.4, "residual_meaningful": True},
        {"has_target": False, "reconstruction_mae": 0.3, "target_contrast_recall": None, "residual_meaningful": False},
    ]

    summary = aggregate_diagnostics(records)

    assert summary["n_images"] == 2
    assert summary["n_target_images"] == 1
    assert summary["reconstruction_mae_mean"] == 0.2
    assert summary["reconstruction_mae_median"] == 0.2
    assert summary["target_contrast_recall_valid_count"] == 1
    assert summary["target_contrast_recall_mean"] == 0.4
    assert summary["residual_meaningful_fraction"] == 1.0


def test_degeneration_flags_detect_known_collapses() -> None:
    summary = {
        "source_false_activation_median": 0.91,
        "residual_energy_ratio_median": 0.97,
        "residual_target_fraction_median": 0.85,
        "target_contrast_recall_median": 8.0,
        "target_psf_zero_fraction": 0.80,
        "background_input_correlation_median": 0.9999,
    }
    gates = {
        "source_false_activation_max": 0.20,
        "residual_energy_ratio_max": 0.90,
        "residual_target_fraction_max": 0.50,
        "target_contrast_recall_min": 0.50,
        "target_contrast_recall_max": 2.00,
        "target_psf_zero_fraction_max": 0.25,
        "background_input_correlation_max": 0.999,
    }

    flags = degeneration_flags(summary, gates)

    assert flags == {
        "source_dense": True,
        "residual_copies_input": True,
        "residual_takes_target": True,
        "target_energy_miscalibrated": True,
        "psf_zero": True,
        "background_copies_input": True,
    }


def test_component_comparison_detects_identity_and_change() -> None:
    reference = {
        "B": torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4),
        "S": torch.eye(4).view(1, 1, 4, 4),
    }
    identical = compare_component_maps(reference, reference)
    changed = compare_component_maps(reference, {"B": reference["B"] * 0, "S": 1 - reference["S"]})

    assert identical["B"]["normalized_l1"] == 0.0
    assert identical["B"]["pearson"] > 0.999
    assert identical["B"]["similarity"] > 0.999
    assert changed["B"]["normalized_l1"] > 0.9
    assert changed["S"]["similarity"] < identical["S"]["similarity"]


def test_coverage_at_5px_marks_centres_inside_radius() -> None:
    import numpy as np
    presence = np.zeros((16, 16), dtype=np.float64)
    center = np.zeros((16, 16), dtype=np.float64)
    center[8, 8] = 1.0
    # No strong presence anywhere -> 0.0.
    assert _coverage_at_5px(presence, center) == 0.0
    # On centre.
    presence[8, 8] = 0.9
    assert _coverage_at_5px(presence, center) == 1.0
    # 4 px away -> still inside.
    presence.fill(0.0)
    presence[8, 12] = 0.9
    assert _coverage_at_5px(presence, center) == 1.0
    # 6 px away -> outside the 5px radius.
    presence.fill(0.0)
    presence[8, 14] = 0.9
    assert _coverage_at_5px(presence, center) == 0.0


def test_coverage_at_5px_is_per_centre() -> None:
    import numpy as np
    presence = np.zeros((16, 16), dtype=np.float64)
    center = np.zeros((16, 16), dtype=np.float64)
    center[8, 8] = 1.0
    center[4, 12] = 1.0  # >5 px from (8,8)
    presence[8, 8] = 0.9  # covers only first centre.
    assert _coverage_at_5px(presence, center) == 0.5
    presence[4, 12] = 0.9  # covers both.
    assert _coverage_at_5px(presence, center) == 1.0

