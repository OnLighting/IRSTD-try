from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from irstd_a.model import build_a_model
from irstd_a.objectives import V5_1_OBJECTIVE, V5_2A_OBJECTIVE
from irstd_a.diagnostics import component_diagnostics
from irstd_a.targets import build_weak_targets
from irstd_g0.data import IRSTD1KDataset
from train_a import validation_score


def _small_model(objective_version: str) -> torch.nn.Module:
    return build_a_model(
        dims=(8, 16, 32),
        num_psf=2,
        kernel_size=5,
        objective_version=objective_version,
    )


def test_v51_uses_nonnegative_residual_and_direct_uncertainty() -> None:
    model = _small_model(V5_1_OBJECTIVE).eval()

    with torch.no_grad():
        output = model(torch.rand(1, 1, 32, 32), return_aux=True)

    assert torch.all(output["R"] >= 0)
    torch.testing.assert_close(output["U"], output["u_rec"])


def test_v51_and_v52a_residual_initialization_is_distinct() -> None:
    v51 = _small_model(V5_1_OBJECTIVE)
    v52 = _small_model(V5_2A_OBJECTIVE)

    assert torch.all(v51.residual_head.bias < 0)
    assert torch.count_nonzero(v52.residual_head.bias) == 0


def test_v51_diagnostics_do_not_require_signed_residual_teachers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("irstd_a.diagnostics._spearman", lambda _a, _b: 0.0)
    image = torch.rand(1, 1, 32, 32)
    mask = torch.zeros_like(image)
    mask[..., 15:17, 15:17] = 1
    targets = build_weak_targets(
        image,
        mask,
        dilation_radius=3,
        ring_radius=9,
        source_flux_scale=16.0,
        psf_radius=2,
        objective_version=V5_1_OBJECTIVE,
    )
    model = _small_model(V5_1_OBJECTIVE).eval()

    with torch.no_grad():
        prediction = model(image, return_aux=True)
    records = component_diagnostics(image, mask, prediction, targets, ["sample"])

    assert len(records) == 1
    assert "residual_teacher_nmae" not in records[0]


def test_historical_v51_checkpoint_loads_and_reconstructs() -> None:
    checkpoint_path = Path("runs/a_psf/irstd1k_seed42_v5_1_ur/a_best.pt")
    if not checkpoint_path.exists():
        pytest.skip("historical v5.1 checkpoint is not present")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = build_a_model(
        **config["model"],
        objective_version=config["loss"]["objective_version"],
    ).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    sample = IRSTD1KDataset(config["data"]["root"], split="test", augment=False)[0]

    with torch.no_grad():
        output = model(sample["image"].unsqueeze(0), return_aux=True)

    assert output["R"].min() >= 0
    assert output["R"].max() < 1e-3
    assert torch.count_nonzero(output["reconstruction"]) > 0
    metrics_path = checkpoint_path.parent / "per_image_metrics.csv"
    with metrics_path.open(encoding="utf-8", newline="") as stream:
        expected = next(
            row for row in csv.DictReader(stream) if row["sample_id"] == sample["id"]
        )
    for component in ("B", "S", "T_psf", "R", "U"):
        values = output[component][0, 0].double()
        actual = {
            "mean": float(values.mean()),
            "std": float(values.std(unbiased=False)),
            "min": float(values.min()),
            "max": float(values.max()),
            "energy": float(values.abs().sum()),
        }
        for statistic, value in actual.items():
            assert np.isclose(
                value,
                float(expected[f"{component}_{statistic}"]),
                rtol=5e-3,
                atol=1e-6,
            ), f"historical {component}_{statistic} mismatch"


@pytest.mark.parametrize(
    ("epoch", "expected"),
    ((46, 0.2625969584949904), (49, 0.20267867070305798)),
)
def test_v51_validation_score_reproduces_historical_log(epoch: int, expected: float) -> None:
    log_path = Path("runs/a_psf/irstd1k_seed42_v5_1_ur/train.jsonl")
    if not log_path.exists():
        pytest.skip("historical v5.1 log is not present")
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    record = next(item for item in records if item["epoch"] == epoch)

    actual = validation_score(record["validation"], objective_version=V5_1_OBJECTIVE)

    assert actual == pytest.approx(expected, abs=1e-12)
