"""Acceptance checks for the formal Stage-A v6 experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class GateFailure:
    name: str
    value: float | None
    requirement: str

    def __str__(self) -> str:
        rendered = "missing" if self.value is None else f"{self.value:.6g}"
        return f"{self.name}={rendered}, required {self.requirement}"


def _nested(container: Mapping[str, Any], *keys: str) -> Any:
    value: Any = container
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _check_min(
    failures: list[GateFailure], name: str, value: Any, threshold: float
) -> None:
    number = _finite_float(value)
    if number is None or number < threshold:
        failures.append(GateFailure(name, number, f">= {threshold:g}"))


def _check_max(
    failures: list[GateFailure], name: str, value: Any, threshold: float
) -> None:
    number = _finite_float(value)
    if number is None or number > threshold:
        failures.append(GateFailure(name, number, f"<= {threshold:g}"))


def evaluate_v6_acceptance(
    metrics: Mapping[str, Any],
    stability: Mapping[str, Any],
    per_image_records: Sequence[Mapping[str, Any]],
) -> list[GateFailure]:
    """Return every failed v6 scientific gate; missing data fails closed."""
    failures: list[GateFailure] = []
    in_domain = _nested(metrics, "datasets", "irstd1k")
    if not isinstance(in_domain, Mapping):
        in_domain = {}

    _check_min(
        failures,
        "target_energy_precision_median",
        in_domain.get("target_energy_precision_median"),
        0.80,
    )
    contrast = _finite_float(in_domain.get("target_contrast_recall_median"))
    if contrast is None or not 0.50 <= contrast <= 2.00:
        failures.append(
            GateFailure("target_contrast_recall_median", contrast, "in [0.5, 2]")
        )
    _check_max(
        failures,
        "source_false_activation_median",
        in_domain.get("source_false_activation_median"),
        0.20,
    )
    _check_max(
        failures,
        "background_target_leakage_median",
        in_domain.get("background_target_leakage_median"),
        0.30,
    )
    _check_max(
        failures,
        "psf_residual_overlap_median",
        in_domain.get("psf_residual_overlap_median"),
        0.50,
    )
    _check_max(
        failures,
        "residual_target_fraction_median",
        in_domain.get("residual_target_fraction_median"),
        0.50,
    )
    _check_min(
        failures,
        "uncertainty_error_spearman_median",
        in_domain.get("uncertainty_error_spearman_median"),
        0.30,
    )
    _check_min(
        failures,
        "coverage_at_5px_mean",
        in_domain.get("coverage_at_5px_mean"),
        0.85,
    )
    _check_min(
        failures,
        "hflip_S_pearson_median",
        _nested(stability, "perturbation", "hflip", "S", "pearson", "median"),
        0.90,
    )

    recalls = [
        number
        for record in per_image_records
        if (number := _finite_float(record.get("centroid_recall_5px"))) is not None
    ]
    worst_count = max(1, math.ceil(0.10 * len(recalls))) if recalls else 0
    worst_tail_min = min(sorted(recalls)[:worst_count]) if worst_count else None
    _check_min(
        failures,
        "centroid_recall_5px_failure_tail",
        worst_tail_min,
        0.50,
    )
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.run_dir
    with (root / "metrics.json").open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    with (root / "stability.json").open(encoding="utf-8") as handle:
        stability = json.load(handle)
    with (root / "per_image_metrics.csv").open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))

    failures = evaluate_v6_acceptance(metrics, stability, records)
    if failures:
        print("[A-v6] acceptance FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("[A-v6] all acceptance gates PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
