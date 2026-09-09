from __future__ import annotations

import os
import random
from dataclasses import dataclass

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from irstd_g0.data import SIRST4Dataset
from irstd_g0.losses import BCEDiceLoss
from irstd_g0.metrics import MetricAccumulator
from irstd_gaussamr.composer import SparseGaussianComposer
from irstd_gaussamr.refiners import (
    DetailRefiner,
    detail_crops,
    gaussian_patch_logits,
    select_detail_proposals,
)
from irstd_gaussamr.router_probe import (
    GaussianFeatureBank,
    GaussianRouter,
    decode_proposals,
    extract_instances,
    fixed_subset,
    proposal_diagnostics,
    router_loss,
)


@dataclass(frozen=True)
class GateBThresholds:
    n_iou: float = 0.90
    coverage_at_24: float = 1.00
    coverage_at_8: float = 0.95


def router_gate_passes(
    metrics: dict[str, float], thresholds: GateBThresholds = GateBThresholds()
) -> bool:
    return (
        metrics["coverage_at_24"] >= thresholds.coverage_at_24
        and metrics["coverage_at_8"] >= thresholds.coverage_at_8
    )


def detail_gate_passes(
    metrics: dict[str, float], thresholds: GateBThresholds = GateBThresholds()
) -> bool:
    return (
        router_gate_passes(metrics, thresholds)
        and metrics["detail_n_iou"] >= thresholds.n_iou
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False


def _pad_to_stride(image: torch.Tensor, stride: int = 8) -> torch.Tensor:
    height, width = image.shape[-2:]
    return F.pad(image, (0, (-width) % stride, 0, (-height) % stride))


def scatter_detail_residual(
    residual: torch.Tensor,
    indices: torch.Tensor,
    proposal_count: int,
) -> torch.Tensor:
    output = residual.new_zeros(
        residual.shape[0],
        proposal_count,
        residual.shape[2],
        residual.shape[3],
        residual.shape[4],
    )
    scatter_index = indices[..., None, None, None].expand_as(residual)
    return output.scatter(1, scatter_index, residual)


def router_stage_loss(
    bank: GaussianFeatureBank,
    router: GaussianRouter,
    composer: SparseGaussianComposer,
    mask_loss_fn: BCEDiceLoss,
    image: torch.Tensor,
    mask: torch.Tensor,
    k1: int = 24,
) -> dict[str, torch.Tensor]:
    mask = mask.to(image.device)
    features = bank(_pad_to_stride(image))
    maps = router(features)
    instances = extract_instances(mask[0])
    router_losses = router_loss(maps, instances)
    proposals = decode_proposals(maps, k=k1)
    full_mask = mask_loss_fn(composer(proposals, mask.shape[-2:]), mask)
    return {
        "total": router_losses["total"] + full_mask,
        **{
            f"router_{name}": value
            for name, value in router_losses.items()
            if name != "total"
        },
        "full_mask": full_mask,
    }


def detail_stage_loss(
    bank: GaussianFeatureBank,
    router: GaussianRouter,
    detail_refiner: DetailRefiner,
    composer: SparseGaussianComposer,
    mask_loss_fn: BCEDiceLoss,
    image: torch.Tensor,
    mask: torch.Tensor,
    k1: int = 24,
    k2: int = 8,
) -> dict[str, torch.Tensor]:
    mask = mask.to(image.device)
    with torch.no_grad():
        features = bank(_pad_to_stride(image))
        maps = router(features)
        proposals = decode_proposals(maps, k=k1)
        selected, indices = select_detail_proposals(proposals, k=k2)

    residual = detail_refiner(features, selected)
    local_target = detail_crops(mask, selected)
    local_logits = gaussian_patch_logits(selected, residual_logits=residual)
    local_loss = mask_loss_fn(
        local_logits.flatten(0, 1), local_target.flatten(0, 1)
    )
    full_residual = scatter_detail_residual(residual, indices, proposals.shape[1])
    full_logits = composer(proposals, mask.shape[-2:], full_residual)
    full_loss = mask_loss_fn(full_logits, mask)
    return {
        "total": 2 * local_loss + full_loss,
        "local": local_loss,
        "full": full_loss,
    }


@torch.no_grad()
def evaluate_gate_b(
    bank: GaussianFeatureBank,
    router: GaussianRouter,
    loader: DataLoader,
    device: str | torch.device,
    composer: SparseGaussianComposer,
    detail_refiner: DetailRefiner | None = None,
    k1: int = 24,
    k2: int = 8,
) -> dict[str, float | int | None]:
    bank.eval()
    router.eval()
    if detail_refiner is not None:
        detail_refiner.eval()

    gaussian_metrics = MetricAccumulator()
    detail_metrics = MetricAccumulator() if detail_refiner is not None else None
    counters: dict[str, float] = {
        name: 0.0
        for name in (
            "targets",
            "hits_at_8",
            "hits_at_16",
            "hits_at_24",
            "matched",
            "center_error_sum",
            "sigma_log_error_sum",
            "active_positive",
            "hard_negative",
        )
    }
    router_loss_sum = 0.0
    image_count = 0

    for batch in loader:
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        features = bank(_pad_to_stride(image))
        maps = router(features)
        instances = extract_instances(mask[0])
        router_loss_sum += float(router_loss(maps, instances)["total"].item())
        proposals = decode_proposals(maps, k=k1)
        diagnostics = proposal_diagnostics(proposals, instances)
        for name in counters:
            counters[name] += float(diagnostics.get(name, 0))

        gaussian_logits = composer(proposals, mask.shape[-2:])
        gaussian_metrics.update_batch(torch.sigmoid(gaussian_logits), mask)
        if detail_refiner is not None:
            selected, indices = select_detail_proposals(proposals, k=k2)
            residual = detail_refiner(features, selected)
            full_residual = scatter_detail_residual(
                residual, indices, proposal_count=proposals.shape[1]
            )
            detail_logits = composer(proposals, mask.shape[-2:], full_residual)
            detail_metrics.update_batch(torch.sigmoid(detail_logits), mask)
        image_count += mask.shape[0]

    targets = int(counters["targets"])
    matched = int(counters["matched"])
    summary: dict[str, float | int | None] = {
        "images": image_count,
        "targets": targets,
        "hits_at_8": int(counters["hits_at_8"]),
        "hits_at_16": int(counters["hits_at_16"]),
        "hits_at_24": int(counters["hits_at_24"]),
        "coverage_at_8": counters["hits_at_8"] / targets if targets else 1.0,
        "coverage_at_16": counters["hits_at_16"] / targets if targets else 1.0,
        "coverage_at_24": counters["hits_at_24"] / targets if targets else 1.0,
        "matched": matched,
        "matched_center_error_px": (
            counters["center_error_sum"] / matched if matched else None
        ),
        "matched_sigma_log_error": (
            counters["sigma_log_error_sum"] / matched if matched else None
        ),
        "active_positive": int(counters["active_positive"]),
        "hard_negative": int(counters["hard_negative"]),
        "router_loss": router_loss_sum / image_count if image_count else 0.0,
    }

    def add_mask_metrics(prefix: str, accumulator: MetricAccumulator) -> None:
        metrics = accumulator.summary()
        summary.update(
            {
                f"{prefix}_images": metrics.get("n_images", 0),
                f"{prefix}_iou": metrics.get("iou_mean", 0.0),
                f"{prefix}_n_iou": metrics.get("n_iou_mean", 0.0),
                f"{prefix}_pd": metrics.get("pd", 0.0),
                f"{prefix}_fa_per_image": metrics.get("fa_per_image", 0.0),
            }
        )

    add_mask_metrics("gaussian", gaussian_metrics)
    if detail_metrics is not None:
        add_mask_metrics("detail", detail_metrics)
    return summary


def build_overfit_loaders(
    data_root: str | os.PathLike[str],
    subset_size: int = 16,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, list[str]]:
    if subset_size <= 0:
        raise ValueError("subset_size must be positive")
    train_dataset = SIRST4Dataset(str(data_root), split="train")
    if len(train_dataset.ids) < subset_size:
        raise ValueError(
            f"requested {subset_size} images, but train split contains "
            f"{len(train_dataset.ids)}"
        )
    eval_dataset = SIRST4Dataset(str(data_root), split="train")
    selected_ids = fixed_subset(train_dataset.ids, subset_size, seed)
    train_dataset.ids = list(selected_ids)
    eval_dataset.ids = list(selected_ids)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, eval_loader, selected_ids
