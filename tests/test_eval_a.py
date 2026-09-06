from __future__ import annotations

import torch

from configs import a_psf_irstd1k as config_module
from eval_a import build_eval_datasets, predict_perturbed


class _IdentityModel(torch.nn.Module):
    def forward(self, image: torch.Tensor, return_aux: bool = False) -> dict:
        zero = torch.zeros_like(image)
        return {
            "B": image,
            "S": zero,
            "T_psf": zero,
            "R": zero,
            "U": zero,
            "reconstruction": image,
        }


def test_dataset_routing_separates_sirst4_overlap() -> None:
    datasets = build_eval_datasets(config_module.data, ["irstd1k", "sirst_uavb", "sirst4"])
    sizes = {name: len(dataset) for name, dataset in datasets}

    assert sizes == {
        "irstd1k": 201,
        "sirst_uavb": 600,
        "sirst4_xdu": 201,
        "sirst4_non_xdu": 866,
    }
    assert all(sample_id.startswith("XDU") for sample_id in dict(datasets)["sirst4_xdu"].ids)
    assert all(not sample_id.startswith("XDU") for sample_id in dict(datasets)["sirst4_non_xdu"].ids)


def test_flip_perturbations_are_inverted_before_comparison() -> None:
    image = torch.arange(24, dtype=torch.float32).view(1, 1, 4, 6) / 23
    model = _IdentityModel()

    horizontal = predict_perturbed(model, image, "hflip", seed=7)
    vertical = predict_perturbed(model, image, "vflip", seed=7)

    assert torch.equal(horizontal["B"], image)
    assert torch.equal(vertical["B"], image)


def test_noise_perturbation_uses_local_deterministic_generator() -> None:
    image = torch.full((1, 1, 4, 6), 0.5)
    model = _IdentityModel()

    first = predict_perturbed(model, image, "noise", seed=19, noise_std=0.01)
    second = predict_perturbed(model, image, "noise", seed=19, noise_std=0.01)

    assert torch.equal(first["B"], second["B"])
    assert not torch.equal(first["B"], image)
    assert torch.all((first["B"] >= 0) & (first["B"] <= 1))


def test_intensity_perturbation_is_bounded() -> None:
    image = torch.tensor([[[[0.0, 1.0]]]])
    output = predict_perturbed(
        _IdentityModel(),
        image,
        "intensity",
        seed=1,
        intensity_scale=2.0,
        intensity_bias=0.2,
    )
    assert output["B"].min() >= 0
    assert output["B"].max() <= 1
