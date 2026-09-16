"""Build stable JSON/CSV summaries for the Stage-A v5.1 3x3 matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from irstd_a.objectives import V5_1_OBJECTIVE
from irstd_a.runtime import atomic_json_dump


SOURCE_RUNS = {
    "irstd1k": "train_irstd1k_seed42",
    "sirst_uavb": "train_sirst_uavb_seed42",
    "sirst4": "train_sirst4_seed42",
}
OFFICIAL_TARGETS = ("irstd1k", "sirst_uavb", "sirst4_all")
PRIMARY_METRICS = (
    "target_energy_precision_median",
    "target_contrast_recall_median",
    "centroid_recall_5px_mean",
    "source_false_activation_median",
    "background_target_leakage_median",
    "reconstruction_mae_mean",
    "reconstruction_psnr_mean",
    "uncertainty_error_spearman_median",
    "model_latency_ms_per_image",
    "end_to_end_imgs_per_s",
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing matrix input: {path}")
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"matrix input must be a JSON object: {path}")
    return payload


def _cell(
    source: str,
    target: str,
    subset_kind: str,
    payload: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    cell = {
        "train_dataset": source,
        "test_dataset": target,
        "subset_kind": subset_kind,
        "n_images": int(metrics["n_images"]),
        "checkpoint_sha256": str(payload["checkpoint_sha256"]),
        "objective_version": str(payload["objective_version"]),
        "seed": int(payload["seed"]),
    }
    for name in PRIMARY_METRICS:
        if name not in metrics:
            raise KeyError(f"dataset {source}->{target} missing metric {name}")
        cell[name] = metrics[name]
    return cell


def build_matrix_summary(run_root: Path) -> dict[str, Any]:
    root = Path(run_root)
    official: list[dict[str, Any]] = []
    clean: list[dict[str, Any]] = []
    overlap: list[dict[str, Any]] = []
    sources: dict[str, dict[str, Any]] = {}
    for source, run_name in SOURCE_RUNS.items():
        payload = _load_json(root / run_name / "metrics.json")
        if payload.get("source_dataset") != source:
            raise ValueError(
                f"source_dataset mismatch in {run_name}: "
                f"expected {source!r}, got {payload.get('source_dataset')!r}"
            )
        if payload.get("objective_version") != V5_1_OBJECTIVE:
            raise ValueError(
                f"objective_version mismatch in {run_name}: "
                f"expected {V5_1_OBJECTIVE!r}, got {payload.get('objective_version')!r}"
            )
        if int(payload.get("seed", -1)) != 42:
            raise ValueError(f"seed mismatch in {run_name}: expected 42")
        datasets = payload.get("datasets")
        if not isinstance(datasets, dict):
            raise TypeError(f"datasets must be an object in {run_name}/metrics.json")
        for target in (*OFFICIAL_TARGETS, "sirst4_xdu", "sirst4_non_xdu"):
            if target not in datasets:
                raise KeyError(f"missing dataset {target} in {run_name}/metrics.json")
        official.extend(
            _cell(source, target, "official", payload, datasets[target])
            for target in OFFICIAL_TARGETS
        )
        clean.append(
            _cell(source, "sirst4_non_xdu", "clean", payload, datasets["sirst4_non_xdu"])
        )
        overlap.append(
            _cell(source, "sirst4_xdu", "overlap", payload, datasets["sirst4_xdu"])
        )
        sources[source] = {
            "run_dir": run_name,
            "checkpoint_sha256": payload["checkpoint_sha256"],
        }
    return {
        "experiment": "a_v5_1_cross_dataset_matrix",
        "objective_version": V5_1_OBJECTIVE,
        "seed": 42,
        "sources": sources,
        "official_cells": official,
        "sirst4_clean_companions": clean,
        "sirst4_overlap_companions": overlap,
        "sirst4_overlap_note": (
            "sirst4_xdu duplicates the 201-image IRSTD-1K test subset; "
            "use sirst4_non_xdu as clean cross-domain evidence"
        ),
    }


def _write_summary_csv(summary: dict[str, Any], path: Path) -> None:
    rows = [
        *summary["official_cells"],
        *summary["sirst4_clean_companions"],
        *summary["sirst4_overlap_companions"],
    ]
    fields = (
        "train_dataset",
        "test_dataset",
        "subset_kind",
        "n_images",
        "checkpoint_sha256",
        "objective_version",
        "seed",
        *PRIMARY_METRICS,
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _combine_per_image(run_root: Path, path: Path) -> None:
    rows: list[dict[str, str]] = []
    fields: list[str] = []
    for source, run_name in SOURCE_RUNS.items():
        source_path = run_root / run_name / "per_image_metrics.csv"
        if not source_path.is_file():
            raise FileNotFoundError(f"missing matrix input: {source_path}")
        with source_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                row["train_dataset"] = source
                for key in row:
                    if key not in fields:
                        fields.append(key)
                rows.append(row)
    if "train_dataset" in fields:
        fields.remove("train_dataset")
    fields.insert(0, "train_dataset")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_matrix_artifacts(run_root: Path) -> dict[str, Path]:
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    summary = build_matrix_summary(root)
    json_path = root / "matrix_summary.json"
    csv_path = root / "matrix_summary.csv"
    per_image_path = root / "per_image_metrics.csv"
    atomic_json_dump(summary, json_path)
    _write_summary_csv(summary, csv_path)
    _combine_per_image(root, per_image_path)
    return {"json": json_path, "csv": csv_path, "per_image": per_image_path}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("runs/a_v5_1_matrix"))
    args = parser.parse_args()
    outputs = write_matrix_artifacts(args.run_root)
    print(f"[A-v5.1-matrix] wrote {outputs['json']} and {outputs['csv']}", flush=True)


if __name__ == "__main__":
    main()
