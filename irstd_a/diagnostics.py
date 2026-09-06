"""Quantitative interpretability and stability diagnostics for module A."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from scipy import ndimage, stats
from torch import Tensor


EPS = 1e-8
COMPONENT_KEYS = ("B", "S", "T_psf", "R", "U")


def _to_numpy(value: Tensor) -> np.ndarray:
    return value.detach().to(device="cpu", dtype=torch.float64).numpy()


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > EPS else None


def _pearson(first: np.ndarray, second: np.ndarray) -> float | None:
    a = first.reshape(-1).astype(np.float64)
    b = second.reshape(-1).astype(np.float64)
    if a.size != b.size or a.size < 2:
        return None
    centered_a = a - float(a.mean())
    centered_b = b - float(b.mean())
    energy_a = float(np.square(centered_a).sum())
    energy_b = float(np.square(centered_b).sum())
    denominator = math.sqrt(energy_a * energy_b)
    if denominator <= EPS:
        return None
    return float((centered_a * centered_b).sum() / denominator)


def _spearman(first: np.ndarray, second: np.ndarray) -> float | None:
    a = first.reshape(-1).astype(np.float64)
    b = second.reshape(-1).astype(np.float64)
    if a.size != b.size or a.size < 2:
        return None
    if float(a.std()) <= EPS or float(b.std()) <= EPS:
        return None
    result = stats.spearmanr(a, b)
    return float(result.statistic) if np.isfinite(result.statistic) else None


def _centroids(binary: np.ndarray) -> np.ndarray:
    labelled, count = ndimage.label(binary.astype(np.uint8), structure=np.ones((3, 3), np.uint8))
    if count == 0:
        return np.zeros((0, 2), dtype=np.float64)
    values = ndimage.center_of_mass(binary, labelled, range(1, count + 1))
    return np.asarray(values, dtype=np.float64)


def _centroid_recall(source: np.ndarray, mask: np.ndarray, radius: float = 5.0) -> float | None:
    ground_truth = _centroids(mask > 0.5)
    if len(ground_truth) == 0:
        return None
    # S is a normalized impulse-strength map.  Dim targets can legitimately
    # peak below 0.5, while tiny sigmoid-floor activations must not become a
    # full-image component.  Combine an absolute floor with a relative peak
    # threshold so centroid recall measures localization rather than amplitude.
    threshold = max(0.05, 0.5 * float(source.max()))
    predicted = _centroids(source >= threshold)
    if len(predicted) == 0:
        return 0.0
    distance = np.sqrt(((ground_truth[:, None, :] - predicted[None, :, :]) ** 2).sum(axis=-1))
    return float((distance.min(axis=1) <= radius).mean())


def _validate_spatial_maps(
    image: Tensor,
    mask: Tensor,
    prediction: Mapping,
    targets: Mapping,
    sample_ids: Sequence[str],
) -> None:
    if image.ndim != 4 or image.shape[1] != 1 or image.shape != mask.shape:
        raise ValueError("image and mask must have equal shape (N,1,H,W)")
    if len(sample_ids) != image.shape[0]:
        raise ValueError("sample_ids length must match batch size")
    for key in (*COMPONENT_KEYS, "reconstruction"):
        if key not in prediction or not isinstance(prediction[key], Tensor):
            raise KeyError(f"prediction missing tensor {key}")
        if prediction[key].shape != image.shape:
            raise ValueError(f"prediction {key} shape does not match image")
    for key in ("center", "support", "psf_support", "local_background", "target_proxy", "source_proxy"):
        if key not in targets or targets[key].shape != image.shape:
            raise KeyError(f"targets missing matching tensor {key}")
    auxiliary_source_keys = ("source_presence", "source_amplitude")
    if any(key in prediction for key in auxiliary_source_keys):
        for key in auxiliary_source_keys:
            if key not in prediction or not isinstance(prediction[key], Tensor):
                raise KeyError(f"prediction missing tensor {key}")
            if prediction[key].shape != image.shape:
                raise ValueError(f"prediction {key} shape does not match image")


def component_diagnostics(
    image: Tensor,
    mask: Tensor,
    prediction: Mapping,
    targets: Mapping[str, Tensor],
    sample_ids: Sequence[str],
) -> list[dict]:
    """Compute auditable per-image component and degeneration measurements."""
    _validate_spatial_maps(image, mask, prediction, targets, sample_ids)
    records: list[dict] = []
    for index, sample_id in enumerate(sample_ids):
        img = _to_numpy(image[index, 0])
        gt = _to_numpy(mask[index, 0])
        support = _to_numpy(targets["support"][index, 0])
        center = _to_numpy(targets["center"][index, 0])
        psf_support = _to_numpy(targets["psf_support"][index, 0])
        local_background = _to_numpy(targets["local_background"][index, 0])
        proxy = _to_numpy(targets["target_proxy"][index, 0])
        source_proxy = _to_numpy(targets["source_proxy"][index, 0])
        maps = {key: _to_numpy(prediction[key][index, 0]) for key in COMPONENT_KEYS}
        reconstruction = _to_numpy(prediction["reconstruction"][index, 0])
        target_output = maps["T_psf"] + maps["R"]
        absolute_error = np.abs(img - reconstruction)
        mse = float(np.mean((img - reconstruction) ** 2))
        has_target = bool(gt.sum() > 0)
        target_energy = float((target_output * support).sum())
        psf_support_energy = float((target_output * psf_support).sum())
        total_target_output = float(target_output.sum())
        proxy_energy = float(proxy.sum())
        source_energy = float(maps["S"].sum())
        residual_energy = float(maps["R"].sum())
        residual_target_energy = float((maps["R"] * support).sum())
        expected_source_mass = float(source_proxy.sum())
        target_outside_energy = float((np.abs(target_output) * (1.0 - psf_support)).sum())
        background_contrast = np.maximum(maps["B"] - local_background, 0.0) * support

        record: dict[str, object] = {
            "sample_id": str(sample_id),
            "has_target": has_target,
            "gt_pixels": int((gt > 0.5).sum()),
            "target_proxy_energy": proxy_energy,
            "reconstruction_mae": float(absolute_error.mean()),
            "reconstruction_psnr": math.inf if mse <= EPS else float(10.0 * math.log10(1.0 / mse)),
            "target_energy_precision": _safe_ratio(psf_support_energy, total_target_output),
            "target_contrast_recall": _safe_ratio(target_energy, proxy_energy) if has_target else None,
            "source_false_activation": _safe_ratio(float((maps["S"] * (1.0 - support)).sum()), source_energy),
            "source_noncenter_mass_ratio": _safe_ratio(
                float((maps["S"] * (1.0 - center)).sum()), expected_source_mass
            ),
            "target_outside_energy_ratio": (
                _safe_ratio(target_outside_energy, proxy_energy) if has_target else None
            ),
            "centroid_recall_5px": _centroid_recall(maps["S"], gt),
            "background_target_leakage": (
                _safe_ratio(float(background_contrast.sum()), proxy_energy) if has_target else None
            ),
            "residual_energy_ratio": _safe_ratio(residual_energy, total_target_output),
            "residual_target_fraction": _safe_ratio(residual_target_energy, target_energy),
            "residual_outside_fraction": _safe_ratio(
                float((maps["R"] * (1.0 - support)).sum()), residual_energy
            ),
            "psf_residual_overlap": _safe_ratio(
                float((maps["T_psf"] * maps["R"]).sum()),
                float(np.sqrt((maps["T_psf"] ** 2).sum() * (maps["R"] ** 2).sum())),
            ),
            "uncertainty_error_spearman": _spearman(maps["U"], absolute_error),
            "background_input_correlation": _pearson(maps["B"], img),
            "target_psf_zero": bool(
                has_target and float((maps["T_psf"] * psf_support).sum()) <= EPS
            ),
            "residual_meaningful": bool(
                has_target and residual_target_energy >= 0.01 * proxy_energy - EPS
            ),
        }
        if "source_presence" in prediction:
            presence = _to_numpy(prediction["source_presence"][index, 0])
            amplitude = _to_numpy(prediction["source_amplitude"][index, 0])
            record.update(
                {
                    "presence_at_centroid": _safe_ratio(
                        float((presence * center).sum()), float(center.sum())
                    ),
                    "presence_false_activation": _safe_ratio(
                        float((presence * (1.0 - support)).sum()), float(presence.sum())
                    ),
                    "amplitude_centroid_mae": _safe_ratio(
                        float((np.abs(amplitude - source_proxy) * center).sum()),
                        float(center.sum()),
                    ),
                }
            )
        for name, values in maps.items():
            record[f"{name}_mean"] = float(values.mean())
            record[f"{name}_std"] = float(values.std())
            record[f"{name}_min"] = float(values.min())
            record[f"{name}_max"] = float(values.max())
            record[f"{name}_energy"] = float(np.abs(values).sum())

        if "psf_weights" in prediction:
            weights = _to_numpy(prediction["psf_weights"][index])
            record["psf_usage"] = weights.mean(axis=(1, 2)).tolist()
            entropy = prediction.get("psf_entropy")
            if isinstance(entropy, Tensor):
                record["psf_entropy_mean"] = float(_to_numpy(entropy[index]).mean())
        if "psf_params" in prediction:
            record["psf_params"] = {
                name: _to_numpy(value).tolist()
                for name, value in prediction["psf_params"].items()
            }
        records.append(record)
    return records


def aggregate_diagnostics(records: Sequence[Mapping]) -> dict:
    """Aggregate numeric fields while preserving undefined-value counts."""
    if not records:
        return {"n_images": 0, "n_target_images": 0}
    summary: dict[str, object] = {
        "n_images": len(records),
        "n_target_images": sum(bool(record.get("has_target")) for record in records),
    }
    keys = sorted({key for record in records for key in record})
    excluded = {
        "sample_id",
        "has_target",
        "psf_usage",
        "psf_params",
        "target_psf_zero",
        "residual_meaningful",
    }
    for key in keys:
        if key in excluded:
            continue
        values = [record.get(key) for record in records]
        numeric = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)]
        if not numeric:
            continue
        summary[f"{key}_valid_count"] = len(numeric)
        summary[f"{key}_mean"] = float(np.mean(numeric))
        summary[f"{key}_median"] = float(np.median(numeric))
        summary[f"{key}_std"] = float(np.std(numeric))
    target_records = [record for record in records if bool(record.get("has_target"))]
    summary["target_psf_zero_fraction"] = (
        float(np.mean([bool(record.get("target_psf_zero")) for record in target_records]))
        if target_records
        else None
    )
    summary["residual_meaningful_fraction"] = (
        float(np.mean([bool(record.get("residual_meaningful")) for record in target_records]))
        if target_records
        else None
    )
    return summary


def degeneration_flags(summary: Mapping[str, float], gates: Mapping[str, float]) -> dict[str, bool]:
    """Convert selected aggregate thresholds into explicit collapse flags."""
    requirements = {
        "source_false_activation_median": "source_false_activation_max",
        "residual_energy_ratio_median": "residual_energy_ratio_max",
        "residual_target_fraction_median": "residual_target_fraction_max",
        "target_psf_zero_fraction": "target_psf_zero_fraction_max",
        "background_input_correlation_median": "background_input_correlation_max",
    }
    missing = (set(requirements) - set(summary)) | (set(requirements.values()) - set(gates))
    missing |= {"target_contrast_recall_median"} - set(summary)
    missing |= {"target_contrast_recall_min", "target_contrast_recall_max"} - set(gates)
    if missing:
        raise KeyError(f"missing degeneration fields: {sorted(missing)}")
    return {
        "source_dense": summary["source_false_activation_median"] > gates["source_false_activation_max"],
        "residual_copies_input": summary["residual_energy_ratio_median"] > gates["residual_energy_ratio_max"],
        "residual_takes_target": summary["residual_target_fraction_median"]
        > gates["residual_target_fraction_max"],
        "target_energy_miscalibrated": not (
            gates["target_contrast_recall_min"]
            <= summary["target_contrast_recall_median"]
            <= gates["target_contrast_recall_max"]
        ),
        "psf_zero": summary["target_psf_zero_fraction"] > gates["target_psf_zero_fraction_max"],
        "background_copies_input": summary["background_input_correlation_median"] > gates["background_input_correlation_max"],
    }


def _global_similarity(first: np.ndarray, second: np.ndarray) -> float:
    a = first.reshape(-1).astype(np.float64)
    b = second.reshape(-1).astype(np.float64)
    mean_a, mean_b = float(a.mean()), float(b.mean())
    variance_a, variance_b = float(a.var()), float(b.var())
    covariance = float(((a - mean_a) * (b - mean_b)).mean())
    c1, c2 = 0.01**2, 0.03**2
    numerator = (2 * mean_a * mean_b + c1) * (2 * covariance + c2)
    denominator = (mean_a**2 + mean_b**2 + c1) * (variance_a + variance_b + c2)
    return float(numerator / denominator) if denominator > EPS else 1.0


def compare_component_maps(reference: Mapping[str, Tensor], candidate: Mapping[str, Tensor]) -> dict:
    """Compare matching component maps for checkpoint or perturbation stability."""
    if set(reference) != set(candidate):
        raise KeyError("reference and candidate must contain identical component keys")
    result: dict[str, dict[str, float | None]] = {}
    for key in reference:
        if reference[key].shape != candidate[key].shape:
            raise ValueError(f"component {key} shapes do not match")
        first = _to_numpy(reference[key])
        second = _to_numpy(candidate[key])
        result[key] = {
            "pearson": _pearson(first, second),
            "normalized_l1": float(np.abs(first - second).sum() / max(np.abs(first).sum(), EPS)),
            "similarity": _global_similarity(first, second),
        }
    return result
