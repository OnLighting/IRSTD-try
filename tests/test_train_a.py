from __future__ import annotations

from pathlib import Path

import pytest
import torch

from irstd_a.losses import LOSS_NAMES
from train_a import (
    EarlyStopper,
    assert_finite_batch,
    build_checkpoint_payload,
    load_config,
    validate_resume_config,
    validate_run_args,
    validation_score,
)


def test_load_config_exposes_only_public_configuration() -> None:
    config = load_config("configs/a_psf_irstd1k.py")
    assert set(config) == {"data", "model", "loss", "optim", "diagnostics", "run"}
    assert config["run"]["seed"] == 42


def test_v6_config_has_seven_loss_terms_with_uniform_weights() -> None:
    """v6 spec: eight physics-named terms, all weights 1.0 except two
    engineering constants in the implementation that don't surface as
    weights (rec support_weight and psf usage_entropy_coef)."""
    config = load_config("configs/a_psf_irstd1k.py")

    assert set(config["loss"]["weights"]) == set(LOSS_NAMES)
    for name in LOSS_NAMES:
        assert config["loss"]["weights"][name] == pytest.approx(1.0), name


def test_validation_score_rewards_interpretable_decomposition() -> None:
    good = {
        "reconstruction_mae_mean": 0.02,
        "target_energy_precision_median": 0.9,
        "target_contrast_recall_median": 0.9,
        "source_false_activation_median": 0.1,
        "background_target_leakage_median": 0.1,
        "psf_residual_overlap_median": 0.2,
    }
    bad = dict(good)
    bad.update(
        reconstruction_mae_mean=0.2,
        target_energy_precision_median=0.2,
        source_false_activation_median=0.8,
    )
    assert validation_score(good) < validation_score(bad)


def test_validation_score_penalizes_residual_taking_the_target() -> None:
    summary = {
        "reconstruction_mae_mean": 0.02,
        "target_energy_precision_median": 0.9,
        "target_contrast_recall_median": 0.9,
        "source_false_activation_median": 0.1,
        "background_target_leakage_median": 0.1,
        "psf_residual_overlap_median": 0.2,
        "residual_target_fraction_median": 0.1,
    }
    shortcut = dict(summary, residual_target_fraction_median=0.9)

    assert validation_score(summary) < validation_score(shortcut)


def test_validation_score_penalizes_target_energy_overshoot() -> None:
    summary = {
        "reconstruction_mae_mean": 0.02,
        "target_energy_precision_median": 0.9,
        "target_contrast_recall_median": 1.0,
        "source_false_activation_median": 0.1,
        "background_target_leakage_median": 0.1,
        "psf_residual_overlap_median": 0.2,
        "residual_target_fraction_median": 0.1,
    }
    overshot = dict(summary, target_contrast_recall_median=10.0)

    assert validation_score(summary) < validation_score(overshot)


def test_validation_score_penalizes_unlocalized_source_map() -> None:
    """Catches checkpoint selection ignoring whether S finds target centroids."""
    localized = {
        "reconstruction_mae_mean": 0.02,
        "target_energy_precision_median": 0.9,
        "target_contrast_recall_median": 1.0,
        "source_false_activation_median": 0.1,
        "centroid_recall_5px_median": 1.0,
        "background_target_leakage_median": 0.1,
        "psf_residual_overlap_median": 0.2,
        "residual_target_fraction_median": 0.1,
    }
    unlocalized = dict(localized, centroid_recall_5px_median=0.0)

    assert validation_score(localized) < validation_score(unlocalized)


def test_assert_finite_batch_reports_sample_ids() -> None:
    with pytest.raises(FloatingPointError, match="XDU9"):
        assert_finite_batch(
            {"total": torch.tensor(float("nan"))},
            sample_ids=["XDU9"],
            stage="train",
        )


def test_debug_limits_cannot_write_to_formal_run() -> None:
    with pytest.raises(ValueError, match="--debug"):
        validate_run_args(Path("runs/a_psf/formal"), debug=False, limit_train=8, limit_val=4)
    validate_run_args(Path("runs/a_psf/debug/smoke"), debug=True, limit_train=8, limit_val=4)


def test_early_stopper_tracks_improvement_and_patience() -> None:
    stopper = EarlyStopper(patience=2)
    assert stopper.update(1.0) is True
    assert stopper.best_score == 1.0 and stopper.bad_epochs == 0
    assert stopper.update(1.1) is False
    assert stopper.should_stop is False
    assert stopper.update(1.2) is False
    assert stopper.should_stop is True
    assert stopper.update(0.9) is True
    assert stopper.best_score == 0.9 and stopper.bad_epochs == 0


def test_checkpoint_payload_contains_resume_and_provenance_state() -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    payload = build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scaler=None,
        epoch=3,
        global_step=17,
        best_score=0.4,
        bad_epochs=2,
        config={"model": {"dims": (8, 16, 32)}},
        train_ids=["a", "b"],
        val_ids=["c"],
    )
    assert set(payload) == {
        "model",
        "optimizer",
        "scaler",
        "epoch",
        "global_step",
        "best_score",
        "bad_epochs",
        "config",
        "train_ids",
        "val_ids",
        "rng_state",
    }
    assert payload["epoch"] == 3
    assert payload["train_ids"] == ["a", "b"]


def test_resume_rejects_changed_training_objective() -> None:
    """Catches resuming optimizer state after the loss semantics changed."""
    current = {"model": {"dims": (8, 16, 32)}, "loss": {"weights": {"sparse": 0.01}}}
    stale = {"model": {"dims": (8, 16, 32)}, "loss": {"weights": {"sparse": 0.05}}}

    with pytest.raises(ValueError, match="training config"):
        validate_resume_config(stale, current)


def test_resume_rejects_stale_loss_term_configuration() -> None:
    """Catches resuming after the loss-name set changed between experiments
    (e.g. v5 -> v6 dropped ``target`` and ``residual`` in favour of
    ``ind``). The configs are not interchangeable even if their numerical
    values happen to match."""
    current = load_config("configs/a_psf_irstd1k.py")
    stale = dict(current)
    stale["loss"] = {"weights": dict(current["loss"]["weights"])}
    # Drop a v6 term and reintroduce a v5 one; the key set must match.
    stale["loss"]["weights"].pop("ind", None)
    stale["loss"]["weights"]["target"] = 1.0

    with pytest.raises(ValueError, match="training config"):
        validate_resume_config(stale, current)
