"""Stage-1 GaussAMR probe: can a Gaussian-aware fixed-budget router cover targets?"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # required by torch.use_deterministic_algorithms(True) + cuBLAS

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from irstd_g0.data import SIRST4Dataset
from irstd_g0.losses import BCEDiceLoss
from irstd_g0.metrics import MetricAccumulator
from irstd_gaussamr.composer import SparseGaussianComposer
from irstd_gaussamr.refiners import (
    ContextRefiner,
    DetailRefiner,
    context_refiner_loss,
    detail_crops,
    gaussian_patch_logits,
    select_detail_proposals,
)
from irstd_gaussamr.router_probe import (
    GaussianFeatureBank,
    GaussianRouter,
    augment_pair,
    decode_proposals,
    extract_instances,
    fixed_subset,
    proposal_diagnostics,
    router_loss,
)
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False


def pad_to_stride(image: torch.Tensor, stride: int = 8) -> torch.Tensor:
    height, width = image.shape[-2:]
    return F.pad(image, (0, (-width) % stride, 0, (-height) % stride))


def summarize(counters: dict[str, float], images: int, mean_loss: float) -> dict[str, float | int | None]:
    targets = int(counters["targets"])
    matched = int(counters["matched"])
    probability_mean = counters["probability_sum"] / counters["router_cells"]
    probability_variance = counters["probability_square_sum"] / counters["router_cells"] - probability_mean ** 2
    return {
        "images": images,
        "targets": targets,
        "loss": mean_loss,
        "coverage_at_8": counters["hits_at_8"] / targets if targets else 1.0,
        "coverage_at_16": counters["hits_at_16"] / targets if targets else 1.0,
        "coverage_at_24": counters["hits_at_24"] / targets if targets else 1.0,
        "matched_center_error_px": counters["center_error_sum"] / matched if matched else None,
        "matched_sigma_log_error": counters["sigma_log_error_sum"] / matched if matched else None,
        "active_positive_per_image": counters["active_positive"] / images,
        "hard_negative_per_image": counters["hard_negative"] / images,
        "router_probability_mean": probability_mean,
        "router_probability_std": max(0.0, probability_variance) ** 0.5,
    }


def _proposal_counters() -> dict[str, float]:
    return {
        key: 0.0 for key in (
            "targets", "hits_at_8", "hits_at_16", "hits_at_24", "matched",
            "center_error_sum", "sigma_log_error_sum", "active_positive", "hard_negative",
        )
    }


def _scatter_detail_residual(
    residual: torch.Tensor,
    indices: torch.Tensor,
    proposal_count: int,
) -> torch.Tensor:
    output = residual.new_zeros(
        residual.shape[0], proposal_count, 1, residual.shape[-2], residual.shape[-1]
    )
    scatter_index = indices[..., None, None, None].expand_as(residual)
    return output.scatter(1, scatter_index, residual)


@torch.no_grad()
def evaluate(
    bank,
    router,
    loader,
    device,
    composer=None,
    context_refiner=None,
    detail_refiner=None,
) -> dict[str, float | int | None]:
    router.eval()
    if context_refiner:
        context_refiner.eval()
    if detail_refiner:
        detail_refiner.eval()
    mask_metrics = MetricAccumulator() if composer else None
    detail_metrics = MetricAccumulator() if detail_refiner else None
    counters = {
        key: 0.0 for key in (
            "targets", "hits_at_8", "hits_at_16", "hits_at_24", "matched",
            "center_error_sum", "sigma_log_error_sum", "active_positive", "hard_negative",
            "probability_sum", "probability_square_sum", "router_cells",
        )
    }
    context_counters = _proposal_counters() if context_refiner else None
    detail_counters = _proposal_counters() if detail_refiner else None
    loss_sum = 0.0
    context_loss_sum = 0.0
    for batch in loader:
        image = pad_to_stride(batch["image"].to(device))
        instances = extract_instances(batch["mask"][0])
        features = bank(image)
        maps = router(features)
        probability = torch.sigmoid(maps["objectness_logit"])
        counters["probability_sum"] += float(probability.sum().item())
        counters["probability_square_sum"] += float(probability.square().sum().item())
        counters["router_cells"] += probability.numel()
        losses = router_loss(maps, instances)
        loss_sum += float(losses["total"].item())
        proposals = decode_proposals(maps, k=24)
        stats = proposal_diagnostics(proposals, instances)
        for key, value in stats.items():
            counters[key] += value
        mask_proposals = proposals
        if context_refiner:
            mask_proposals = context_refiner(features, proposals)
            context_loss_sum += float(
                context_refiner_loss(mask_proposals, instances, match_from=proposals)["total"].item()
            )
            for key, value in proposal_diagnostics(mask_proposals, instances).items():
                context_counters[key] += value
        if composer:
            height, width = batch["mask"].shape[-2:]
            mask_metrics.update_batch(torch.sigmoid(composer(mask_proposals, (height, width))).cpu(), batch["mask"])
            if detail_refiner:
                selected, indices = select_detail_proposals(mask_proposals, k=8)
                residual = detail_refiner(features, selected)
                full_residual = _scatter_detail_residual(residual, indices, mask_proposals.shape[1])
                detail_metrics.update_batch(
                    torch.sigmoid(composer(mask_proposals, (height, width), full_residual)).cpu(),
                    batch["mask"],
                )
                for key, value in proposal_diagnostics(selected, instances).items():
                    detail_counters[key] += value
    summary = summarize(counters, len(loader.dataset), loss_sum / max(1, len(loader)))
    if context_refiner:
        targets = int(context_counters["targets"])
        matched = int(context_counters["matched"])
        summary.update(
            context_loss=context_loss_sum / max(1, len(loader)),
            context_coverage_at_8=context_counters["hits_at_8"] / targets if targets else 1.0,
            context_coverage_at_16=context_counters["hits_at_16"] / targets if targets else 1.0,
            context_coverage_at_24=context_counters["hits_at_24"] / targets if targets else 1.0,
            context_matched_center_error_px=(
                context_counters["center_error_sum"] / matched if matched else None
            ),
            context_matched_sigma_log_error=(
                context_counters["sigma_log_error_sum"] / matched if matched else None
            ),
            context_active_positive_per_image=context_counters["active_positive"] / len(loader.dataset),
            context_hard_negative_per_image=context_counters["hard_negative"] / len(loader.dataset),
        )
    if mask_metrics:
        gaussian = mask_metrics.summary()
        summary.update(
            gaussian_iou=gaussian["iou_mean"],
            gaussian_n_iou=gaussian["n_iou_mean"],
            gaussian_pd=gaussian["pd"],
            gaussian_fa_per_image=gaussian["fa_per_image"],
        )
    if detail_metrics:
        detail = detail_metrics.summary()
        targets = int(detail_counters["targets"])
        summary.update(
            detail_coverage_at_8=detail_counters["hits_at_8"] / targets if targets else 1.0,
            detail_iou=detail["iou_mean"],
            detail_n_iou=detail["n_iou_mean"],
            detail_pd=detail["pd"],
            detail_fa_per_image=detail["fa_per_image"],
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/SIRST4-ForLiTE")
    parser.add_argument("--run-dir", default="runs/gaussamr_router_probe_seed42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=64)
    parser.add_argument("--val-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--full-mask-loss", action="store_true")
    parser.add_argument("--context-refiner", action="store_true")
    parser.add_argument("--detail-refiner", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if min(args.train_size, args.val_size, args.epochs, args.max_steps) <= 0:
        parser.error("train-size, val-size, epochs, and max-steps must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.context_refiner and (not args.init_checkpoint or not args.full_mask_loss):
        parser.error("context-refiner requires init-checkpoint and full-mask-loss")
    if args.detail_refiner and (not args.init_checkpoint or not args.full_mask_loss):
        parser.error("detail-refiner requires init-checkpoint and full-mask-loss")
    if args.context_refiner and args.detail_refiner:
        parser.error("context-refiner and detail-refiner are separate staged probes")

    seed_everything(args.seed)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = SIRST4Dataset(args.data_root, split="train")
    val_ds = SIRST4Dataset(args.data_root, split="test")
    train_ds.ids = fixed_subset(train_ds.ids, args.train_size, args.seed)
    val_ds.ids = fixed_subset(val_ds.ids, args.val_size, args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    augmentation_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    bank = GaussianFeatureBank().to(args.device).eval()
    router = GaussianRouter().to(args.device)
    context_refiner = ContextRefiner().to(args.device) if args.context_refiner else None
    detail_refiner = DetailRefiner().to(args.device) if args.detail_refiner else None
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=args.device, weights_only=False)
        router.load_state_dict(checkpoint["router"])
        if context_refiner and "context_refiner" in checkpoint:
            context_refiner.load_state_dict(checkpoint["context_refiner"])
        if detail_refiner and "detail_refiner" in checkpoint:
            detail_refiner.load_state_dict(checkpoint["detail_refiner"])
    if context_refiner or detail_refiner:
        router.eval()
        router.requires_grad_(False)
    composer = SparseGaussianComposer().to(args.device) if args.full_mask_loss else None
    mask_loss_fn = BCEDiceLoss().to(args.device) if args.full_mask_loss else None
    trainable_parameters = (
        context_refiner.parameters() if context_refiner
        else detail_refiner.parameters() if detail_refiner
        else router.parameters()
    )
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=0.05)

    config = {
        "dataset": "sirst4",
        "data_root": args.data_root,
        "seed": args.seed,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "batch_size": 1,
        "lr": args.lr,
        "lr_schedule": "constant",
        "weight_decay": 0.05,
        "k1": 24,
        "k2_diagnostic": 8,
        "match_radius_px": 12,
        "augmentation": "g0_hflip_vflip_rot90",
        "init_checkpoint": args.init_checkpoint,
        "full_mask_loss": args.full_mask_loss,
        "context_refiner": args.context_refiner,
        "detail_refiner": args.detail_refiner,
        "router_frozen": bool(context_refiner or detail_refiner),
    }
    history = []
    best = None
    step = 0
    for epoch in range(1, args.epochs + 1):
        if context_refiner:
            context_refiner.train()
        elif detail_refiner:
            detail_refiner.train()
        else:
            router.train()
        train_loss = 0.0
        batches = 0
        for batch in train_loader:
            image, mask = augment_pair(batch["image"], batch["mask"], augmentation_generator)
            image = pad_to_stride(image.to(args.device))
            instances = extract_instances(mask[0])
            with torch.no_grad():
                features = bank(image)
                maps = router(features) if context_refiner or detail_refiner else None
            if context_refiner:
                router_proposals = decode_proposals(maps, k=24)
                proposals = context_refiner(features, router_proposals)
                losses = context_refiner_loss(proposals, instances, match_from=router_proposals)
            elif detail_refiner:
                proposals = decode_proposals(maps, k=24)
                selected, indices = select_detail_proposals(proposals, k=8)
                residual = detail_refiner(features, selected)
                target_crops = detail_crops(mask.to(args.device), selected)
                local_logits = gaussian_patch_logits(
                    selected, residual_logits=residual
                )
                local_loss = mask_loss_fn(
                    local_logits.flatten(0, 1), target_crops.flatten(0, 1)
                )
                full_residual = _scatter_detail_residual(residual, indices, proposals.shape[1])
                full_logits = composer(proposals, mask.shape[-2:], full_residual)
                full_loss = mask_loss_fn(full_logits, mask.to(args.device))
                losses = {"total": 2 * local_loss + full_loss}
            else:
                maps = router(features)
                proposals = decode_proposals(maps, k=24)
                losses = router_loss(maps, instances)
            total_loss = losses["total"]
            if composer and not detail_refiner:
                gaussian_logits = composer(proposals, mask.shape[-2:])
                total_loss = total_loss + mask_loss_fn(gaussian_logits, mask.to(args.device))
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"non-finite router loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
            optimizer.step()
            train_loss += float(total_loss.item())
            batches += 1
            step += 1
            if step >= args.max_steps:
                break

        metrics = evaluate(
            bank, router, val_loader, args.device, composer, context_refiner, detail_refiner
        )
        metrics.update(epoch=epoch, step=step, train_loss=train_loss / max(1, batches))
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
        metric_name = "detail_n_iou" if detail_refiner else "gaussian_n_iou"
        score = (metrics[metric_name], metrics["coverage_at_24"]) if composer else (metrics["coverage_at_24"], metrics["coverage_at_8"])
        best_score = None if best is None else ((best[metric_name], best["coverage_at_24"]) if composer else (best["coverage_at_24"], best["coverage_at_8"]))
        if best is None or score > best_score:
            best = dict(metrics)
            checkpoint = {"router": router.state_dict(), "config": config, "best": best}
            if context_refiner:
                checkpoint["context_refiner"] = context_refiner.state_dict()
            if detail_refiner:
                checkpoint["detail_refiner"] = detail_refiner.state_dict()
            best_name = (
                "context_best.pt" if context_refiner
                else "detail_best.pt" if detail_refiner
                else "router_best.pt"
            )
            torch.save(checkpoint, run_dir / best_name)
        output = {
            "config": config,
            "gate": {"coverage_at_24_min": 0.95, "router_probability_std_min": 1e-4},
            "best": best,
            "passes_router_gate": bool(
                best["coverage_at_24"] >= 0.95
                and best["router_probability_std"] >= 1e-4
                and math.isfinite(best["loss"])
            ),
            "epochs": history,
        }
        if detail_refiner:
            output["gate"]["detail_must_improve_gaussian_n_iou"] = True
            output["passes_detail_gate"] = bool(
                best["detail_n_iou"] > best["gaussian_n_iou"]
                and math.isfinite(best["detail_n_iou"])
            )
        (run_dir / "metrics.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        if step >= args.max_steps:
            break

    checkpoint = {"router": router.state_dict(), "config": config, "last": history[-1]}
    if context_refiner:
        checkpoint["context_refiner"] = context_refiner.state_dict()
    if detail_refiner:
        checkpoint["detail_refiner"] = detail_refiner.state_dict()
    last_name = (
        "context_last.pt" if context_refiner
        else "detail_last.pt" if detail_refiner
        else "router_last.pt"
    )
    torch.save(checkpoint, run_dir / last_name)
    print(f"wrote {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
