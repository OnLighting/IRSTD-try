from __future__ import annotations

from irstd_a.acceptance import evaluate_v6_acceptance


def _passing_inputs() -> tuple[dict, dict, list[dict]]:
    metrics = {
        "datasets": {
            "irstd1k": {
                "target_energy_precision_median": 0.90,
                "target_contrast_recall_median": 1.00,
                "source_false_activation_median": 0.10,
                "background_target_leakage_median": 0.10,
                "psf_residual_overlap_median": 0.10,
                "residual_target_fraction_median": 0.10,
                "uncertainty_error_spearman_median": 0.40,
                "coverage_at_5px_mean": 0.90,
            }
        }
    }
    stability = {
        "perturbation": {"hflip": {"S": {"pearson": {"median": 0.95}}}}
    }
    records = [{"centroid_recall_5px": 0.75} for _ in range(20)]
    return metrics, stability, records


def test_v6_acceptance_passes_only_when_every_required_gate_passes() -> None:
    metrics, stability, records = _passing_inputs()

    assert evaluate_v6_acceptance(metrics, stability, records) == []


def test_v6_acceptance_reports_scientific_failures() -> None:
    metrics, stability, records = _passing_inputs()
    in_domain = metrics["datasets"]["irstd1k"]
    in_domain["target_energy_precision_median"] = 0.21
    in_domain["target_contrast_recall_median"] = 3.63
    stability["perturbation"]["hflip"]["S"]["pearson"]["median"] = 0.79
    records[0]["centroid_recall_5px"] = 0.0

    failures = evaluate_v6_acceptance(metrics, stability, records)

    assert {failure.name for failure in failures} == {
        "target_energy_precision_median",
        "target_contrast_recall_median",
        "hflip_S_pearson_median",
        "centroid_recall_5px_failure_tail",
    }


def test_v6_acceptance_treats_missing_measurements_as_failures() -> None:
    metrics, stability, records = _passing_inputs()
    del metrics["datasets"]["irstd1k"]["uncertainty_error_spearman_median"]

    failures = evaluate_v6_acceptance(metrics, stability, records)

    assert any(
        failure.name == "uncertainty_error_spearman_median"
        and failure.value is None
        for failure in failures
    )
