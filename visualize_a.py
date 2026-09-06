"""Deterministic visual audit panels for module-A decompositions."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from eval_a import build_eval_datasets
from irstd_a.model import build_a_model


def _image_array(value: Tensor) -> np.ndarray:
    array = value.detach().to(device="cpu", dtype=torch.float32).numpy()
    if array.ndim == 4:
        if array.shape[0] != 1 or array.shape[1] != 1:
            raise ValueError("visualization expects a single one-channel image")
        return array[0, 0]
    if array.ndim == 3 and array.shape[0] == 1:
        return array[0]
    if array.ndim == 2:
        return array
    raise ValueError("visualization expects shape (1,1,H,W), (1,H,W), or (H,W)")


def _display_vmax(title: str, values: np.ndarray) -> float:
    """Keep intensity-like maps absolute while revealing sparse components."""
    if title in {"I", "Y", "B", "U", "reconstruction"}:
        return 1.0
    finite = values[np.isfinite(values)]
    return max(float(finite.max()), 1e-6) if finite.size else 1.0


def save_component_panel(
    image: Tensor,
    mask: Tensor,
    prediction: Mapping[str, Tensor],
    sample_id: str,
    dataset: str,
    destination: Path,
    diagnostics: Mapping[str, Any] | None = None,
) -> None:
    """Save the fixed nine-panel decomposition audit for one image."""
    required = {"B", "S", "T_psf", "R", "U", "reconstruction"}
    missing = required - set(prediction)
    if missing:
        raise KeyError(f"prediction missing panel maps: {sorted(missing)}")
    input_array = _image_array(image)
    mask_array = _image_array(mask)
    maps = {key: _image_array(prediction[key]) for key in required}
    error = np.abs(input_array - maps["reconstruction"])
    panels = [
        ("I", input_array, "gray"),
        ("Y", mask_array, "gray"),
        ("B", maps["B"], "gray"),
        ("S", maps["S"], "magma"),
        ("T_psf", maps["T_psf"], "magma"),
        ("R", maps["R"], "magma"),
        ("U", maps["U"], "viridis"),
        ("reconstruction", maps["reconstruction"], "gray"),
        ("absolute error", error, "magma"),
    ]
    figure, axes = plt.subplots(3, 3, figsize=(13, 12), constrained_layout=True)
    for axis, (title, values, color_map) in zip(axes.flat, panels):
        display_max = _display_vmax(title, values)
        shown = axis.imshow(values, cmap=color_map, vmin=0.0, vmax=display_max)
        axis.set_title(
            f"{title}\nmean={values.mean():.4f}, energy={np.abs(values).sum():.2f}, "
            f"display_max={display_max:.4f}",
            fontsize=9,
        )
        axis.axis("off")
        figure.colorbar(shown, ax=axis, fraction=0.046, pad=0.02)
    subtitle = _diagnostic_subtitle(diagnostics or {})
    title = f"{dataset} / {sample_id}"
    if subtitle:
        title += f"\n{subtitle}"
    figure.suptitle(title, fontsize=14)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def save_psf_kernel_sheet(
    kernels: Tensor,
    params: Mapping[str, Tensor],
    usage: Sequence[float],
    destination: Path,
) -> None:
    """Save learned PSFs with bounded parameters and mean usage."""
    kernel_array = kernels.detach().to(device="cpu", dtype=torch.float32).numpy()
    count = kernel_array.shape[0]
    if len(usage) != count:
        raise ValueError("usage length must match the number of kernels")
    columns = min(3, count)
    rows = math.ceil(count / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows), squeeze=False)
    values = {
        key: tensor.detach().to(device="cpu", dtype=torch.float32).numpy()
        for key, tensor in params.items()
    }
    for index, axis in enumerate(axes.flat):
        if index >= count:
            axis.axis("off")
            continue
        shown = axis.imshow(kernel_array[index, 0], cmap="magma")
        axis.set_title(
            f"K{index}: sx={values['sigma_x'][index]:.2f}, "
            f"sy={values['sigma_y'][index]:.2f}\n"
            f"theta={values['theta'][index]:.2f}, usage={float(usage[index]):.3f}"
        )
        axis.axis("off")
        figure.colorbar(shown, ax=axis, fraction=0.046, pad=0.02)
    figure.suptitle("Learned anisotropic PSF bank")
    figure.tight_layout()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _as_float(row: Mapping[str, str], key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _diagnostic_subtitle(row: Mapping[str, Any]) -> str:
    """Format the v5 latent and energy checks without adding public panels."""
    fields = (
        ("presence_at_centroid", "P@centroid"),
        ("amplitude_centroid_mae", "A_MAE"),
        ("source_noncenter_mass_ratio", "S_noncenter/Esrc"),
        ("target_outside_energy_ratio", "T_out/Etarget"),
    )
    values: list[str] = []
    for key, label in fields:
        value = _as_float(row, key)
        if value is not None:
            values.append(f"{label}={value:.3f}")
    return " | ".join(values)


def _select_rows(rows: list[dict[str, str]], maximum: int = 12) -> list[dict[str, str]]:
    """Select deterministic best, failure, scale, and uncertainty cases."""
    selected: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(row: dict[str, str]) -> None:
        key = (row["dataset"], row["sample_id"])
        if key not in seen and len(selected) < maximum:
            seen.add(key)
            selected.append(row)

    criteria = [
        ("target_proxy_energy", False),
        ("target_proxy_energy", True),
        ("gt_pixels", False),
        ("gt_pixels", True),
        ("source_false_activation", True),
        ("U_mean", True),
        ("residual_energy_ratio", True),
        ("reconstruction_mae", True),
    ]
    for field, descending in criteria:
        valid = [(value, row) for row in rows if (value := _as_float(row, field)) is not None]
        if valid:
            valid.sort(key=lambda item: (item[0], item[1]["dataset"], item[1]["sample_id"]), reverse=descending)
            add(valid[0][1])
    for row in rows:
        if row.get("has_target", "").lower() in {"false", "0"}:
            add(row)
            break
    for row in rows:
        add(row)
    return selected


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-panels", type=int, default=12)
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_a_model(**config["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with Path(args.metrics).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = _select_rows(rows, maximum=args.max_panels)
    datasets = dict(build_eval_datasets(config["data"], ["irstd1k", "sirst_uavb", "sirst4"]))
    lookup = {
        name: {sample_id: index for index, sample_id in enumerate(dataset.ids)}
        for name, dataset in datasets.items()
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    usage_values: list[np.ndarray] = []
    last_aux: dict[str, Any] | None = None
    for row in selected:
        dataset_name, sample_id = row["dataset"], row["sample_id"]
        dataset = datasets[dataset_name]
        sample = dataset[lookup[dataset_name][sample_id]]
        image = sample["image"].unsqueeze(0).to(device)
        mask = sample["mask"].unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = model(image, return_aux=True)
        usage_values.append(
            prediction["psf_weights"].mean(dim=(0, 2, 3)).detach().cpu().numpy()
        )
        last_aux = prediction
        filename = f"{_safe_filename(dataset_name)}__{_safe_filename(sample_id)}.png"
        save_component_panel(
            image,
            mask,
            prediction,
            sample_id,
            dataset_name,
            output_dir / filename,
            diagnostics=row,
        )

    if last_aux is None:
        raise ValueError("metrics CSV contains no selectable rows")
    mean_usage = np.stack(usage_values).mean(axis=0).tolist()
    save_psf_kernel_sheet(
        last_aux["psf_kernels"],
        last_aux["psf_params"],
        mean_usage,
        output_dir / "psf_kernels.png",
    )
    print(f"[A-vis] wrote {len(selected)} panels and PSF sheet to {output_dir}")


if __name__ == "__main__":
    main()
