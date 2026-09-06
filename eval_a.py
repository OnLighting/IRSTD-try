"""Interpretability, degeneration, and stability evaluation for module A."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from irstd_g0.data import IRSTD1KDataset, SIRST4Dataset, SIRSTUAVBDataset
from irstd_a.diagnostics import (
    COMPONENT_KEYS,
    aggregate_diagnostics,
    compare_component_maps,
    component_diagnostics,
    degeneration_flags,
)
from irstd_a.model import build_a_model
from irstd_a.runtime import atomic_json_dump, file_sha256
from irstd_a.targets import build_weak_targets


def build_eval_datasets(
    data_config: Mapping[str, Any],
    names: Sequence[str],
) -> list[tuple[str, torch.utils.data.Dataset]]:
    """Build named datasets, separating SIRST4's duplicated XDU subset."""
    result: list[tuple[str, torch.utils.data.Dataset]] = []
    for name in names:
        if name == "irstd1k":
            dataset = IRSTD1KDataset(
                data_config["root"],
                split=data_config.get("test_split", "test"),
                augment=False,
            )
            result.append(("irstd1k", dataset))
        elif name == "sirst_uavb":
            result.append(
                ("sirst_uavb", SIRSTUAVBDataset(data_config["sirst_uavb_root"], split="test"))
            )
        elif name == "sirst4":
            all_samples = SIRST4Dataset(data_config["sirst4_root"], split="test")
            xdu = SIRST4Dataset(data_config["sirst4_root"], split="test")
            non_xdu = SIRST4Dataset(data_config["sirst4_root"], split="test")
            xdu.ids = [sample_id for sample_id in all_samples.ids if sample_id.startswith("XDU")]
            non_xdu.ids = [sample_id for sample_id in all_samples.ids if not sample_id.startswith("XDU")]
            result.extend((("sirst4_xdu", xdu), ("sirst4_non_xdu", non_xdu)))
        else:
            raise ValueError(f"unknown evaluation dataset: {name}")
    return result


def _public_maps(prediction: Mapping[str, Any]) -> dict[str, Tensor]:
    return {key: prediction[key] for key in COMPONENT_KEYS}


def predict_perturbed(
    model: torch.nn.Module,
    image: Tensor,
    kind: str,
    *,
    seed: int,
    noise_std: float = 0.01,
    intensity_scale: float = 0.9,
    intensity_bias: float = 0.05,
) -> dict[str, Tensor]:
    """Predict after a controlled perturbation and undo spatial transforms."""
    if kind == "hflip":
        transformed = torch.flip(image, dims=(-1,))
    elif kind == "vflip":
        transformed = torch.flip(image, dims=(-2,))
    elif kind == "noise":
        generator = torch.Generator(device=image.device)
        generator.manual_seed(int(seed))
        noise = torch.randn(
            image.shape,
            generator=generator,
            device=image.device,
            dtype=image.dtype,
        )
        transformed = (image + noise_std * noise).clamp(0.0, 1.0)
    elif kind == "intensity":
        transformed = (intensity_scale * image + intensity_bias).clamp(0.0, 1.0)
    else:
        raise ValueError(f"unknown perturbation: {kind}")

    with torch.no_grad():
        prediction = _public_maps(model(transformed, return_aux=False))
    if kind == "hflip":
        return {key: torch.flip(value, dims=(-1,)) for key, value in prediction.items()}
    if kind == "vflip":
        return {key: torch.flip(value, dims=(-2,)) for key, value in prediction.items()}
    return prediction


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_csv(records: Sequence[Mapping[str, Any]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for record in records for key in record})
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for record in records:
                row = {}
                for key in fields:
                    value = _json_safe(record.get(key))
                    row[key] = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _aggregate_comparisons(entries: Sequence[Mapping[str, Mapping[str, float | None]]]) -> dict:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        for component, measurements in entry.items():
            for metric, value in measurements.items():
                if value is not None and math.isfinite(float(value)):
                    values[component][metric].append(float(value))
    return {
        component: {
            metric: {
                "mean": float(np.mean(items)),
                "median": float(np.median(items)),
                "count": len(items),
            }
            for metric, items in metrics.items()
        }
        for component, metrics in values.items()
    }


def perturbation_stability(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    config: Mapping[str, Any],
    probe_count: int,
    seed: int,
) -> dict:
    settings = config["diagnostics"]["perturbations"]
    comparisons: dict[str, list[dict]] = {kind: [] for kind in ("hflip", "vflip", "noise", "intensity")}
    model.eval()
    for index in range(min(probe_count, len(dataset))):
        sample = dataset[index]
        image = sample["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            reference = _public_maps(model(image, return_aux=False))
        for kind in comparisons:
            candidate = predict_perturbed(
                model,
                image,
                kind,
                seed=seed + index,
                noise_std=float(settings["noise_std"]),
                intensity_scale=float(settings["intensity_scale"]),
                intensity_bias=float(settings["intensity_bias"]),
            )
            comparisons[kind].append(compare_component_maps(reference, candidate))
    return {kind: _aggregate_comparisons(entries) for kind, entries in comparisons.items()}


def checkpoint_stability(
    reference_model: torch.nn.Module,
    reference_checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    probe_count: int,
) -> dict:
    candidates = [path for path in checkpoint_path.parent.glob("a_*.pt") if path != checkpoint_path]
    if not candidates:
        return {"available": False, "reason": "no last or milestone checkpoints found"}
    reference_model.eval()
    images = [dataset[index]["image"].unsqueeze(0).to(device) for index in range(min(probe_count, len(dataset)))]
    with torch.no_grad():
        references = [_public_maps(reference_model(image, return_aux=False)) for image in images]
    result: dict[str, Any] = {"available": True, "reference_epoch": reference_checkpoint.get("epoch"), "candidates": {}}
    for candidate_path in sorted(candidates):
        checkpoint = torch.load(candidate_path, map_location=device, weights_only=False)
        candidate_model = build_a_model(**checkpoint["config"]["model"]).to(device)
        candidate_model.load_state_dict(checkpoint["model"])
        candidate_model.eval()
        entries = []
        with torch.no_grad():
            for image, reference in zip(images, references):
                prediction = _public_maps(candidate_model(image, return_aux=False))
                entries.append(compare_component_maps(reference, prediction))
        result["candidates"][candidate_path.name] = {
            "epoch": checkpoint.get("epoch"),
            "comparison": _aggregate_comparisons(entries),
        }
    return result


def _evaluate_dataset(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    dataset_name: str,
    device: torch.device,
    config: Mapping[str, Any],
    batch_size: int,
) -> tuple[list[dict], dict]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    records: list[dict] = []
    model_times: list[float] = []
    end_to_end_start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            sample_ids = [str(value) for value in batch["id"]]
            targets = build_weak_targets(
                image,
                mask,
                dilation_radius=int(config["data"]["dilation_radius"]),
                ring_radius=int(config["data"]["ring_radius"]),
                source_flux_scale=float(config["model"]["source_flux_scale"]),
                psf_radius=int(config["model"]["kernel_size"]) // 2,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            prediction = model(image, return_aux=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            model_times.append(time.perf_counter() - started)
            batch_records = component_diagnostics(image, mask, prediction, targets, sample_ids)
            for record in batch_records:
                record["dataset"] = dataset_name
            records.extend(batch_records)
    elapsed = time.perf_counter() - end_to_end_start
    summary = aggregate_diagnostics(records)
    summary["model_latency_ms_per_image"] = 1000.0 * sum(model_times) / max(1, len(dataset))
    summary["end_to_end_imgs_per_s"] = len(dataset) / max(elapsed, 1e-8)
    try:
        summary["degeneration_flags"] = degeneration_flags(summary, config["diagnostics"]["gates"])
    except (KeyError, TypeError) as error:
        summary["degeneration_flags"] = {"not_evaluable": True, "reason": str(error)}
    return records, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--datasets", nargs="+", default=["irstd1k", "sirst_uavb", "sirst4"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--probe-count", type=int, default=8)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_a_model(**config["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    datasets = build_eval_datasets(config["data"], args.datasets)
    if args.limit:
        for _, dataset in datasets:
            dataset.ids = dataset.ids[: args.limit]

    all_records: list[dict] = []
    summaries: dict[str, dict] = {}
    for name, dataset in datasets:
        print(f"[A-eval] {name}: {len(dataset)} images", flush=True)
        records, summary = _evaluate_dataset(
            model, dataset, name, device, config, args.batch_size
        )
        all_records.extend(records)
        summaries[name] = summary

    primary = next((dataset for name, dataset in datasets if name == "irstd1k"), datasets[0][1])
    stability = {
        "initialization_stability_tested": False,
        "initialization_stability_reason": "exploratory stage uses one seed",
        "perturbation": perturbation_stability(
            model,
            primary,
            device,
            config,
            probe_count=args.probe_count,
            seed=int(config["run"]["seed"]),
        ),
        "checkpoint": checkpoint_stability(
            model,
            checkpoint,
            checkpoint_path,
            primary,
            device,
            probe_count=args.probe_count,
        ),
    }
    output = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "device": str(device),
        "datasets": summaries,
        "sirst4_overlap_note": "sirst4_xdu duplicates the 201-image IRSTD-1K test subset",
    }
    _write_csv(all_records, run_dir / "per_image_metrics.csv")
    atomic_json_dump(_json_safe(output), run_dir / "metrics.json")
    atomic_json_dump(_json_safe(stability), run_dir / "stability.json")
    print(f"[A-eval] wrote metrics and stability to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
