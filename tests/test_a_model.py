from __future__ import annotations

import inspect

import pytest
import torch

from irstd_a.model import (
    APSFUnmixingNet,
    AnisotropicGaussianPSFBank,
    build_a_model,
)


def test_psf_kernels_are_positive_normalized_and_bounded() -> None:
    bank = AnisotropicGaussianPSFBank(num_kernels=6, kernel_size=15)

    kernels, params = bank.kernels_and_params()

    assert kernels.shape == (6, 1, 15, 15)
    assert torch.all(kernels >= 0)
    assert torch.allclose(
        kernels.sum(dim=(-1, -2)),
        torch.ones(6, 1),
        atol=1e-5,
    )
    for name in ("sigma_x", "sigma_y"):
        assert torch.all(params[name] >= 0.6)
        assert torch.all(params[name] <= 4.0)
    assert torch.all(params["theta"] >= -torch.pi / 2)
    assert torch.all(params["theta"] <= torch.pi / 2)


def test_psf_bank_rejects_even_kernel_or_too_few_components() -> None:
    with pytest.raises(ValueError, match="odd"):
        AnisotropicGaussianPSFBank(num_kernels=3, kernel_size=8)
    with pytest.raises(ValueError, match="positive"):
        AnisotropicGaussianPSFBank(num_kernels=0, kernel_size=7)


def test_a_forward_returns_bounded_full_resolution_components() -> None:
    model = build_a_model(dims=(8, 16, 32), num_psf=3, kernel_size=7)
    image = torch.rand(2, 1, 65, 67)

    out = model(image, return_aux=True)

    for key in ("B", "S", "T_psf", "R", "U", "reconstruction"):
        assert out[key].shape == image.shape
        assert torch.isfinite(out[key]).all()
        assert torch.all((out[key] >= 0) & (out[key] <= 1))
    assert out["psf_weights"].shape == (2, 3, 65, 67)
    assert torch.allclose(
        out["psf_weights"].sum(dim=1),
        torch.ones(2, 65, 67),
        atol=1e-5,
    )
    assert out["psf_kernels"].shape == (3, 1, 7, 7)
    assert out["psf_entropy"].shape == image.shape
    assert out["u_rec"].shape == image.shape
    assert out["reconstruction_raw"].shape == image.shape


def test_a_forward_public_mode_hides_training_auxiliaries() -> None:
    model = build_a_model(dims=(8, 16, 32), num_psf=3, kernel_size=7)

    out = model(torch.rand(1, 1, 32, 36))

    assert set(out) == {"B", "S", "T_psf", "R", "U", "reconstruction"}


def test_internal_source_flux_scales_normalized_public_source() -> None:
    model = build_a_model(
        dims=(8, 16, 32),
        num_psf=3,
        kernel_size=7,
        source_flux_scale=16.0,
    )

    out = model(torch.rand(1, 1, 32, 32), return_aux=True)

    assert torch.allclose(out["S_flux"], 16.0 * out["S"], atol=1e-6)


def test_factorized_source_combines_presence_and_amplitude() -> None:
    """Catches reverting to one logit that conflates localization and flux."""
    model = build_a_model(dims=(8, 16, 32), num_psf=3, kernel_size=7)

    out = model(torch.rand(1, 1, 31, 33), return_aux=True)

    assert out["presence_logits"].shape == out["S"].shape
    assert out["amplitude_logits"].shape == out["S"].shape
    assert torch.allclose(torch.sigmoid(out["presence_logits"]), out["source_presence"], atol=1e-6)
    assert torch.allclose(torch.sigmoid(out["amplitude_logits"]), out["source_amplitude"], atol=1e-6)
    assert torch.allclose(out["S"], out["source_presence"] * out["source_amplitude"], atol=1e-6)


def test_forward_api_does_not_accept_mask() -> None:
    signature = inspect.signature(APSFUnmixingNet.forward)
    assert set(signature.parameters) == {"self", "image", "return_aux"}


def test_model_rejects_malformed_or_nonfinite_input() -> None:
    model = build_a_model(dims=(8, 16, 32), num_psf=3, kernel_size=7)
    with pytest.raises(ValueError, match="single-channel"):
        model(torch.rand(1, 3, 32, 32))
    bad = torch.rand(1, 1, 32, 32)
    bad[..., 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(bad)


def test_explicit_psf_operator_preserves_a_unit_impulse_energy() -> None:
    bank = AnisotropicGaussianPSFBank(num_kernels=3, kernel_size=7)
    source = torch.zeros(1, 1, 31, 31)
    source[..., 15, 15] = 1
    weights = torch.zeros(1, 3, 31, 31)
    weights[:, 1] = 1

    response, kernels, _ = bank(source, weights)

    assert torch.allclose(response.sum(), torch.tensor(1.0), atol=1e-5)
    expected = kernels[1, 0]
    actual = response[0, 0, 12:19, 12:19]
    assert torch.allclose(actual, expected, atol=1e-6)
