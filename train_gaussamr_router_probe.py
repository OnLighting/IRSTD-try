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


@torch.no_grad()
def evaluate(bank, router, loader, device, composer=None) -> dict[str, float | int | None]:
    router.eval()
    mask_metrics = MetricAccumulator() if composer else None
    counters = {
        key: 0.0 for key in (
            "targets", "hits_at_8", "hits_at_16", "hits_at_24", "matched",
            "center_error_sum", "sigma_log_error_sum", "active_positive", "hard_negative",
            "probability_sum", "probability_square_sum", "router_cells",
        )
    }
    loss_sum = 0.0
    for batch in loader:
        image = pad_to_stride(batch["image"].to(device))
        instances = extract_instances(batch["mask"][0])
        maps = router(bank(image))
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
        if composer:
            height, width = batch["mask"].shape[-2:]
            mask_metrics.update_batch(torch.sigmoid(composer(proposals, (height, width))).cpu(), batch["mask"])
    summary = summarize(counters, len(loader.dataset), loss_sum / max(1, len(loader)))
    if mask_metrics:
        gaussian = mask_metrics.summary()
        summary.update(
            gaussian_iou=gaussian["iou_mean"],
            gaussian_n_iou=gaussian["n_iou_mean"],
            gaussian_pd=gaussian["pd"],
            gaussian_fa_per_image=gaussian["fa_per_image"],
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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if min(args.train_size, args.val_size, args.epochs, args.max_steps) <= 0:
        parser.error("train-size, val-size, epochs, and max-steps must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

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
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=args.device, weights_only=False)
        router.load_state_dict(checkpoint["router"])
    composer = SparseGaussianComposer().to(args.device) if args.full_mask_loss else None
    mask_loss_fn = BCEDiceLoss().to(args.device) if args.full_mask_loss else None
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.lr, weight_decay=0.05)

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
    }
    history = []
    best = None
    step = 0
    for epoch in range(1, args.epochs + 1):
        router.train()
        train_loss = 0.0
        batches = 0
        for batch in train_loader:
            image, mask = augment_pair(batch["image"], batch["mask"], augmentation_generator)
            image = pad_to_stride(image.to(args.device))
            instances = extract_instances(mask[0])
            with torch.no_grad():
                features = bank(image)
            maps = router(features)
            losses = router_loss(maps, instances)
            total_loss = losses["total"]
            if composer:
                gaussian_logits = composer(decode_proposals(maps, k=24), mask.shape[-2:])
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

        metrics = evaluate(bank, router, val_loader, args.device, composer)
        metrics.update(epoch=epoch, step=step, train_loss=train_loss / max(1, batches))
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
        score = (metrics["gaussian_n_iou"], metrics["coverage_at_24"]) if composer else (metrics["coverage_at_24"], metrics["coverage_at_8"])
        best_score = None if best is None else ((best["gaussian_n_iou"], best["coverage_at_24"]) if composer else (best["coverage_at_24"], best["coverage_at_8"]))
        if best is None or score > best_score:
            best = dict(metrics)
            torch.save({"router": router.state_dict(), "config": config, "best": best}, run_dir / "router_best.pt")
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
        (run_dir / "metrics.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        if step >= args.max_steps:
            break

    torch.save({"router": router.state_dict(), "config": config, "last": history[-1]}, run_dir / "router_last.pt")
    print(f"wrote {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
