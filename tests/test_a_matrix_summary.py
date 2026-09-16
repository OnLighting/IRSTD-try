from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from summarize_a_matrix import (
    OFFICIAL_TARGETS,
    PRIMARY_METRICS,
    SOURCE_RUNS,
    build_matrix_summary,
    write_matrix_artifacts,
)


def _metric(n_images: int, offset: float) -> dict[str, float | int]:
    result: dict[str, float | int] = {"n_images": n_images}
    for index, name in enumerate(PRIMARY_METRICS):
        result[name] = offset + index / 100.0
    return result


def _write_source_run(root: Path, source: str, run_name: str, offset: float) -> None:
    run_dir = root / run_name
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "a_best.pt"
    checkpoint.write_bytes(f"checkpoint:{source}".encode())
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    metrics = {
        "source_dataset": source,
        "objective_version": "v5.1-ur",
        "seed": 42,
        "checkpoint_sha256": checkpoint_hash,
        "datasets": {
            "irstd1k": _metric(201, offset),
            "sirst_uavb": _metric(600, offset + 1),
            "sirst4_all": _metric(1067, offset + 2),
            "sirst4_xdu": _metric(201, offset + 3),
            "sirst4_non_xdu": _metric(866, offset + 4),
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    with (run_dir / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("dataset", "sample_id"))
        writer.writeheader()
        writer.writerow({"dataset": "irstd1k", "sample_id": f"{source}-1"})


def test_build_matrix_summary_maps_nine_official_cells_and_clean_companions(
    tmp_path: Path,
) -> None:
    for index, (source, run_name) in enumerate(SOURCE_RUNS.items()):
        _write_source_run(tmp_path, source, run_name, float(index * 10))

    summary = build_matrix_summary(tmp_path)

    assert len(summary["official_cells"]) == 9
    assert len(summary["sirst4_clean_companions"]) == 3
    assert len(summary["sirst4_overlap_companions"]) == 3
    assert {cell["test_dataset"] for cell in summary["official_cells"]} == set(OFFICIAL_TARGETS)
    assert {cell["train_dataset"] for cell in summary["official_cells"]} == set(SOURCE_RUNS)
    assert all(cell["objective_version"] == "v5.1-ur" for cell in summary["official_cells"])


def test_summary_rejects_wrong_source_identity(tmp_path: Path) -> None:
    for index, (source, run_name) in enumerate(SOURCE_RUNS.items()):
        _write_source_run(tmp_path, source, run_name, float(index))
    path = tmp_path / SOURCE_RUNS["sirst4"] / "metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["source_dataset"] = "irstd1k"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="source_dataset"):
        build_matrix_summary(tmp_path)


def test_summary_rejects_checkpoint_hash_mismatch(tmp_path: Path) -> None:
    for index, (source, run_name) in enumerate(SOURCE_RUNS.items()):
        _write_source_run(tmp_path, source, run_name, float(index))
    (tmp_path / SOURCE_RUNS["irstd1k"] / "a_best.pt").write_bytes(b"replaced")

    with pytest.raises(ValueError, match="checkpoint hash"):
        build_matrix_summary(tmp_path)


def test_write_matrix_artifacts_emits_json_csv_and_combined_rows(tmp_path: Path) -> None:
    for index, (source, run_name) in enumerate(SOURCE_RUNS.items()):
        _write_source_run(tmp_path, source, run_name, float(index))

    outputs = write_matrix_artifacts(tmp_path)

    assert outputs["json"].is_file()
    assert outputs["csv"].is_file()
    assert outputs["per_image"].is_file()
    with outputs["csv"].open(newline="", encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 15
    with outputs["per_image"].open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert {row["train_dataset"] for row in rows} == set(SOURCE_RUNS)
