from __future__ import annotations

from pathlib import Path

import pytest
import torch

from visualize_a import (
    _diagnostic_subtitle,
    _display_vmax,
    save_component_panel,
    save_psf_kernel_sheet,
)


def test_v5_diagnostic_subtitle_reports_presence_and_amplitude() -> None:
    subtitle = _diagnostic_subtitle(
        {"presence_at_centroid": "0.8123", "amplitude_centroid_mae": "0.0476"}
    )

    assert "P@centroid=0.812" in subtitle
    assert "A_MAE=0.048" in subtitle


def test_sparse_component_panels_use_their_observed_dynamic_range() -> None:
    values = torch.tensor([[0.0, 0.02]], dtype=torch.float32).numpy()

    assert _display_vmax("S", values) == pytest.approx(0.02)
    assert _display_vmax("I", values) == 1.0


def test_component_panel_and_kernel_sheet_are_nonempty_pngs(tmp_path: Path) -> None:
    image = torch.linspace(0, 1, 256).view(1, 1, 16, 16)
    mask = torch.zeros_like(image)
    mask[..., 7:9, 7:9] = 1
    prediction = {
        "B": image * 0.8,
        "S": mask,
        "T_psf": mask * 0.15,
        "R": mask * 0.05,
        "U": torch.abs(image - 0.5),
        "reconstruction": image,
    }
    original = image.clone()
    panel_path = tmp_path / "panel.png"
    kernel_path = tmp_path / "kernels.png"

    save_component_panel(image, mask, prediction, "sample", "irstd1k", panel_path)
    kernels = torch.rand(3, 1, 7, 7)
    kernels = kernels / kernels.sum(dim=(-1, -2), keepdim=True)
    params = {
        "sigma_x": torch.tensor([0.8, 1.4, 2.2]),
        "sigma_y": torch.tensor([2.2, 1.4, 0.8]),
        "theta": torch.tensor([-0.5, 0.0, 0.5]),
    }
    save_psf_kernel_sheet(kernels, params, [0.2, 0.5, 0.3], kernel_path)

    assert panel_path.stat().st_size > 1_000
    assert kernel_path.stat().st_size > 1_000
    assert torch.equal(image, original)
