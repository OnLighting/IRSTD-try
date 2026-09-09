"""Deterministic two-stage GaussAMR V1 Gate B trainer and verifier."""
from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from irstd_g0.losses import BCEDiceLoss
from irstd_gaussamr.composer import SparseGaussianComposer
from irstd_gaussamr.gate_b import (
    GateBThresholds,
    build_overfit_loaders,
    detail_gate_passes,
    detail_stage_loss,
    evaluate_gate_b,
    router_gate_passes,
    router_stage_loss,
    seed_everything,
)
from irstd_gaussamr.model import GaussAMRV1
from irstd_gaussamr.refiners import DetailRefiner
from irstd_gaussamr.router_probe import (
    GaussianFeatureBank,
    GaussianRouter,
    augment_pair,
)


REAL_THRESHOLDS = GateBThresholds()
SMOKE_THRESHOLDS = GateBThresholds(
    n_iou=0.0,
    coverage_at_24=0.0,
    coverage_at_8=0.0,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/SIRST4-ForLiTE")
    parser.add_argument("--run-dir", default="runs/gaussamr_gate_b_seed42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=3200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--router-checkpoint")
    parser.add_argument("--detail-checkpoint")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    numeric = {
        "subset size": args.subset_size,
        "epochs": args.epochs,
        "max steps": args.max_steps,
        "learning rate": args.lr,
        "batch size": args.batch_size,
    }
    if any(value <= 0 for value in numeric.values()):
        raise ValueError("subset size, epochs, max steps, learning rate, and batch size must be positive")
    if args.batch_size != 1:
        raise ValueError("Gate B requires batch size 1")
    train_split = Path(args.data_root) / "img_idx" / "train.txt"
    if not train_split.is_file():
        raise ValueError(f"dataset train split does not exist: {train_split}")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if args.detail_checkpoint and not args.router_checkpoint:
        raise ValueError("--detail-checkpoint requires --router-checkpoint")
    if args.verify_only and not args.checkpoint:
        raise ValueError("--verify-only requires --checkpoint")
    if args.verify_only and args.smoke_test:
        raise ValueError("--verify-only and --smoke-test cannot be combined")
    if not args.verify_only and args.checkpoint:
        raise ValueError("--checkpoint is only valid with --verify-only")
    for label, value in (
        ("router checkpoint", args.router_checkpoint),
        ("detail checkpoint", args.detail_checkpoint),
        ("checkpoint", args.checkpoint),
    ):
        if value and not Path(value).is_file():
            raise ValueError(f"{label} does not exist: {value}")


def _json_write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_checkpoint(path: str | os.PathLike[str], device: str) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint is not a dictionary: {path}")
    return checkpoint


def _state_dict_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(
        left[name].shape == right[name].shape
        and torch.equal(left[name].detach().cpu(), right[name].detach().cpu())
        for name in left
    )


def validate_resume_compatibility(
    router_checkpoint: str | os.PathLike[str],
    detail_checkpoint: str | os.PathLike[str],
    device: str,
) -> None:
    router_payload = _load_checkpoint(router_checkpoint, device)
    detail_payload = _load_checkpoint(detail_checkpoint, device)
    if "router" not in router_payload:
        raise ValueError("router checkpoint is missing 'router'")
    if "router" not in detail_payload or "detail_refiner" not in detail_payload:
        raise ValueError("detail checkpoint must contain 'router' and 'detail_refiner'")
    if not _state_dict_equal(router_payload["router"], detail_payload["router"]):
        raise ValueError("detail checkpoint router is incompatible with router checkpoint")


def _finite_diagnostics(
    stage: str,
    losses: dict[str, torch.Tensor],
    module: torch.nn.Module,
    epoch: int,
    step: int,
    sample_id: str,
) -> dict[str, float | int]:
    for name, value in losses.items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(
                f"non-finite {stage} loss {name} at epoch={epoch} step={step} sample={sample_id}"
            )
    max_gradient = 0.0
    parameter_count = 0
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        parameter_count += 1
        if parameter.grad is None:
            raise FloatingPointError(
                f"missing {stage} gradient {name} at epoch={epoch} step={step} sample={sample_id}"
            )
        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(
                f"non-finite {stage} gradient {name} at epoch={epoch} step={step} sample={sample_id}"
            )
        max_gradient = max(max_gradient, float(parameter.grad.detach().abs().max().item()))
    return {
        "checks": 1,
        "trainable_parameter_tensors": parameter_count,
        "max_abs_loss": max(float(value.detach().abs().item()) for value in losses.values()),
        "max_abs_gradient": max_gradient,
    }


_STAGE_SEED_OFFSET = {"router": 10_000_019, "detail": 20_000_033}


def _epoch_order(size: int, seed: int, stage: str, epoch: int) -> list[int]:
    generator = torch.Generator().manual_seed(
        int(seed) * 1_000_003 + _STAGE_SEED_OFFSET[stage] + int(epoch) * 10_007
    )
    return torch.randperm(size, generator=generator).tolist()


def _augmentation_generator(
    seed: int, stage: str, epoch: int, batch_offset: int
) -> torch.Generator:
    return torch.Generator().manual_seed(
        int(seed) * 1_000_003
        + _STAGE_SEED_OFFSET[stage]
        + int(epoch) * 10_007
        + int(batch_offset) * 101
    )


def _training_batch(
    dataset: Any,
    seed: int,
    stage: str,
    epoch: int,
    batch_offset: int,
) -> dict[str, Any]:
    order = _epoch_order(len(dataset), seed, stage, epoch)
    item = dataset[order[batch_offset]]
    return {
        "image": item["image"].unsqueeze(0),
        "mask": item["mask"].unsqueeze(0),
        "id": [item["id"]],
    }


def _resume_cursor(resume: dict[str, Any] | None) -> tuple[int, int]:
    if not resume:
        return 1, 0
    epoch = int(resume.get("epoch", 0))
    if epoch <= 0:
        return 1, 0
    if bool(resume.get("epoch_complete", True)):
        return epoch + 1, 0
    return epoch, int(resume.get("batch_offset", 0))


def _stage_gate_passes(
    stage: str,
    metrics: dict[str, Any],
    thresholds: GateBThresholds,
    require_targets: bool,
) -> bool:
    target_pass = not require_targets or int(metrics.get("targets", 0)) > 0
    metric_pass = (
        router_gate_passes(metrics, thresholds)
        if stage == "router"
        else detail_gate_passes(metrics, thresholds)
    )
    return target_pass and metric_pass


def _checkpoint_payload(
    router: GaussianRouter,
    config: dict[str, Any],
    selected_ids: list[str],
    history: list[dict[str, Any]],
    best: dict[str, Any],
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    step: int,
    batch_offset: int = 0,
    epoch_complete: bool = True,
    stopping_reason: str | None = None,
    detail_refiner: DetailRefiner | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "router": copy.deepcopy(router.state_dict()),
        "config": config,
        "selected_ids": selected_ids,
        "history": history,
        "best": best,
        "epoch": epoch,
        "step": step,
        "batch_offset": batch_offset,
        "epoch_complete": epoch_complete,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if stopping_reason is not None:
        payload["stopping_reason"] = stopping_reason
    if detail_refiner is not None:
        payload["detail_refiner"] = copy.deepcopy(detail_refiner.state_dict())
    return payload


def _router_score(metrics: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(metrics["coverage_at_24"]),
        float(metrics["coverage_at_8"]),
        float(metrics["gaussian_n_iou"]),
    )


def _evaluate_router(
    bank: GaussianFeatureBank,
    router: GaussianRouter,
    eval_loader: Any,
    device: str,
    composer: SparseGaussianComposer,
) -> dict[str, Any]:
    return evaluate_gate_b(bank, router, eval_loader, device, composer)


def _run_router_stage(
    args: argparse.Namespace,
    config: dict[str, Any],
    selected_ids: list[str],
    train_loader: Any,
    eval_loader: Any,
    bank: GaussianFeatureBank,
    composer: SparseGaussianComposer,
    thresholds: GateBThresholds,
    run_dir: Path,
    require_targets: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]], str, int]:
    router = GaussianRouter().to(args.device)
    resume = None
    if args.router_checkpoint:
        resume = _load_checkpoint(args.router_checkpoint, args.device)
        if "router" not in resume:
            raise ValueError("router checkpoint is missing 'router'")
        router.load_state_dict(resume["router"])
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.lr, weight_decay=0.05)
    if resume and "optimizer" in resume:
        optimizer.load_state_dict(resume["optimizer"])

    history: list[dict[str, Any]] = []
    step = int(resume.get("step", 0)) if resume else 0
    saved_epoch = int(resume.get("epoch", 0)) if resume else 0
    current_epoch, current_offset = _resume_cursor(resume)
    last_epoch = saved_epoch
    last_batch_offset = int(resume.get("batch_offset", 0)) if resume else 0
    last_epoch_complete = bool(resume.get("epoch_complete", True)) if resume else True
    initial = _evaluate_router(bank, router, eval_loader, args.device, composer)
    initial.update(epoch=saved_epoch, step=step, phase="initial", train_loss=None)
    history.append(initial)
    best = dict(initial)

    stopping_reason: str | None = None
    successful_finite_checks = 0
    can_advance = step < args.max_steps and current_epoch <= args.epochs
    initial_pass = _stage_gate_passes(
        "router", initial, thresholds, require_targets
    )
    if resume and (initial_pass or not can_advance):
        router.train()
        probe_epoch = min(max(1, current_epoch), args.epochs)
        probe_offset = current_offset if current_epoch == probe_epoch else 0
        if probe_offset >= len(train_loader.dataset):
            probe_offset = 0
        batch = _training_batch(
            train_loader.dataset, args.seed, "router", probe_epoch, probe_offset
        )
        image, mask = augment_pair(
            batch["image"],
            batch["mask"],
            _augmentation_generator(args.seed, "router", probe_epoch, probe_offset),
        )
        image = image.to(args.device)
        mask = mask.to(args.device)
        sample_id = str(batch["id"][0])
        optimizer.zero_grad(set_to_none=True)
        probe_losses = router_stage_loss(
            bank, router, composer, BCEDiceLoss(), image, mask
        )
        probe_losses["total"].backward()
        diagnostics = _finite_diagnostics(
            "router", probe_losses, router, probe_epoch, step, sample_id
        )
        optimizer.zero_grad(set_to_none=True)
        successful_finite_checks = int(diagnostics["checks"])
        initial["finite_checks"] = {
            "count": successful_finite_checks,
            "max_abs_loss": diagnostics["max_abs_loss"],
            "max_abs_gradient": diagnostics["max_abs_gradient"],
            "probe_without_optimizer_step": True,
            "sample_id": sample_id,
        }
        if initial_pass:
            stopping_reason = "threshold_met_initial_after_finite_probe"
        elif step >= args.max_steps:
            stopping_reason = "max_steps_reached"
        else:
            stopping_reason = "max_epochs_reached"

    torch.save(
        _checkpoint_payload(
            router,
            config,
            selected_ids,
            history,
            best,
            optimizer,
            saved_epoch,
            step,
            last_batch_offset,
            last_epoch_complete,
        ),
        run_dir / "router_best.pt",
    )

    while (
        stopping_reason is None
        and current_epoch <= args.epochs
        and step < args.max_steps
    ):
        router.train()
        total_loss = 0.0
        batches = 0
        finite_checks = 0
        max_abs_gradient = 0.0
        max_abs_loss = 0.0
        sample_ids: list[str] = []
        order = _epoch_order(
            len(train_loader.dataset), args.seed, "router", current_epoch
        )
        for batch_offset in range(current_offset, len(order)):
            batch = _training_batch(
                train_loader.dataset,
                args.seed,
                "router",
                current_epoch,
                batch_offset,
            )
            image, mask = augment_pair(
                batch["image"],
                batch["mask"],
                _augmentation_generator(
                    args.seed, "router", current_epoch, batch_offset
                ),
            )
            image = image.to(args.device)
            mask = mask.to(args.device)
            sample_id = str(batch["id"][0])
            optimizer.zero_grad(set_to_none=True)
            losses = router_stage_loss(
                bank, router, composer, BCEDiceLoss(), image, mask
            )
            losses["total"].backward()
            diagnostics = _finite_diagnostics(
                "router", losses, router, current_epoch, step + 1, sample_id
            )
            torch.nn.utils.clip_grad_norm_(
                router.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            total_loss += float(losses["total"].detach().item())
            batches += 1
            step += 1
            finite_checks += int(diagnostics["checks"])
            successful_finite_checks += int(diagnostics["checks"])
            max_abs_gradient = max(max_abs_gradient, float(diagnostics["max_abs_gradient"]))
            max_abs_loss = max(max_abs_loss, float(diagnostics["max_abs_loss"]))
            sample_ids.append(sample_id)
            last_epoch = current_epoch
            last_batch_offset = batch_offset + 1
            last_epoch_complete = last_batch_offset == len(order)
            if step >= args.max_steps:
                break

        metrics = _evaluate_router(bank, router, eval_loader, args.device, composer)
        metrics.update(
            epoch=current_epoch,
            step=step,
            phase="train",
            train_loss=total_loss / max(1, batches),
            sample_ids=sample_ids,
            finite_checks={
                "count": finite_checks,
                "max_abs_loss": max_abs_loss,
                "max_abs_gradient": max_abs_gradient,
            },
        )
        history.append(metrics)
        print(json.dumps({"stage": "router", **metrics}, ensure_ascii=False), flush=True)
        if _router_score(metrics) > _router_score(best):
            best = dict(metrics)
            torch.save(
                _checkpoint_payload(
                    router,
                    config,
                    selected_ids,
                    history,
                    best,
                    optimizer,
                    last_epoch,
                    step,
                    last_batch_offset,
                    last_epoch_complete,
                ),
                run_dir / "router_best.pt",
            )
        if successful_finite_checks > 0 and _stage_gate_passes(
            "router", metrics, thresholds, require_targets
        ):
            stopping_reason = "threshold_met"
        elif step >= args.max_steps:
            stopping_reason = "max_steps_reached"
        torch.save(
            _checkpoint_payload(
                router,
                config,
                selected_ids,
                history,
                best,
                optimizer,
                last_epoch,
                step,
                last_batch_offset,
                last_epoch_complete,
                stopping_reason=stopping_reason,
            ),
            run_dir / "router_last.pt",
        )
        if stopping_reason is None:
            if last_epoch_complete:
                current_epoch += 1
                current_offset = 0
            else:
                break

    if stopping_reason is None:
        stopping_reason = "max_epochs_reached"
    torch.save(
        _checkpoint_payload(
            router,
            config,
            selected_ids,
            history,
            best,
            optimizer,
            last_epoch,
            step,
            last_batch_offset,
            last_epoch_complete,
            stopping_reason=stopping_reason,
        ),
        run_dir / "router_last.pt",
    )
    return best, history, stopping_reason, step


def _run_detail_stage(
    args: argparse.Namespace,
    config: dict[str, Any],
    selected_ids: list[str],
    train_loader: Any,
    eval_loader: Any,
    bank: GaussianFeatureBank,
    composer: SparseGaussianComposer,
    thresholds: GateBThresholds,
    run_dir: Path,
    require_targets: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]], str, int]:
    router_checkpoint = _load_checkpoint(run_dir / "router_best.pt", args.device)
    router = GaussianRouter().to(args.device)
    router.load_state_dict(router_checkpoint["router"])
    router.eval().requires_grad_(False)
    detail_refiner = DetailRefiner().to(args.device)
    resume = None
    if args.detail_checkpoint:
        validate_resume_compatibility(
            run_dir / "router_best.pt", args.detail_checkpoint, args.device
        )
        resume = _load_checkpoint(args.detail_checkpoint, args.device)
        if "detail_refiner" not in resume or "router" not in resume:
            raise ValueError("detail checkpoint must contain 'router' and 'detail_refiner'")
        detail_refiner.load_state_dict(resume["detail_refiner"])
    optimizer = torch.optim.AdamW(detail_refiner.parameters(), lr=args.lr, weight_decay=0.05)
    if resume and "optimizer" in resume:
        optimizer.load_state_dict(resume["optimizer"])

    history: list[dict[str, Any]] = []
    step = int(resume.get("step", 0)) if resume else 0
    saved_epoch = int(resume.get("epoch", 0)) if resume else 0
    current_epoch, current_offset = _resume_cursor(resume)
    last_epoch = saved_epoch
    last_batch_offset = int(resume.get("batch_offset", 0)) if resume else 0
    last_epoch_complete = bool(resume.get("epoch_complete", True)) if resume else True
    initial = evaluate_gate_b(
        bank, router, eval_loader, args.device, composer, detail_refiner
    )
    initial.update(epoch=saved_epoch, step=step, phase="initial", train_loss=None)
    history.append(initial)
    best = dict(initial)

    stopping_reason: str | None = None
    successful_finite_checks = 0
    can_advance = step < args.max_steps and current_epoch <= args.epochs
    initial_pass = _stage_gate_passes(
        "detail", initial, thresholds, require_targets
    )
    if resume and (initial_pass or not can_advance):
        detail_refiner.train()
        probe_epoch = min(max(1, current_epoch), args.epochs)
        probe_offset = current_offset if current_epoch == probe_epoch else 0
        if probe_offset >= len(train_loader.dataset):
            probe_offset = 0
        batch = _training_batch(
            train_loader.dataset, args.seed, "detail", probe_epoch, probe_offset
        )
        image, mask = augment_pair(
            batch["image"],
            batch["mask"],
            _augmentation_generator(args.seed, "detail", probe_epoch, probe_offset),
        )
        image = image.to(args.device)
        mask = mask.to(args.device)
        sample_id = str(batch["id"][0])
        optimizer.zero_grad(set_to_none=True)
        probe_losses = detail_stage_loss(
            bank,
            router,
            detail_refiner,
            composer,
            BCEDiceLoss(),
            image,
            mask,
        )
        probe_losses["total"].backward()
        diagnostics = _finite_diagnostics(
            "detail", probe_losses, detail_refiner, probe_epoch, step, sample_id
        )
        optimizer.zero_grad(set_to_none=True)
        successful_finite_checks = int(diagnostics["checks"])
        initial["finite_checks"] = {
            "count": successful_finite_checks,
            "max_abs_loss": diagnostics["max_abs_loss"],
            "max_abs_gradient": diagnostics["max_abs_gradient"],
            "probe_without_optimizer_step": True,
            "sample_id": sample_id,
        }
        if initial_pass:
            stopping_reason = "threshold_met_initial_after_finite_probe"
        elif step >= args.max_steps:
            stopping_reason = "max_steps_reached"
        else:
            stopping_reason = "max_epochs_reached"

    torch.save(
        _checkpoint_payload(
            router,
            config,
            selected_ids,
            history,
            best,
            optimizer,
            saved_epoch,
            step,
            last_batch_offset,
            last_epoch_complete,
            detail_refiner=detail_refiner,
        ),
        run_dir / "detail_best.pt",
    )

    while (
        stopping_reason is None
        and current_epoch <= args.epochs
        and step < args.max_steps
    ):
        detail_refiner.train()
        total_loss = 0.0
        batches = 0
        finite_checks = 0
        max_abs_gradient = 0.0
        max_abs_loss = 0.0
        sample_ids: list[str] = []
        order = _epoch_order(
            len(train_loader.dataset), args.seed, "detail", current_epoch
        )
        for batch_offset in range(current_offset, len(order)):
            batch = _training_batch(
                train_loader.dataset,
                args.seed,
                "detail",
                current_epoch,
                batch_offset,
            )
            image, mask = augment_pair(
                batch["image"],
                batch["mask"],
                _augmentation_generator(
                    args.seed, "detail", current_epoch, batch_offset
                ),
            )
            image = image.to(args.device)
            mask = mask.to(args.device)
            sample_id = str(batch["id"][0])
            optimizer.zero_grad(set_to_none=True)
            losses = detail_stage_loss(
                bank,
                router,
                detail_refiner,
                composer,
                BCEDiceLoss(),
                image,
                mask,
            )
            losses["total"].backward()
            diagnostics = _finite_diagnostics(
                "detail", losses, detail_refiner, current_epoch, step + 1, sample_id
            )
            torch.nn.utils.clip_grad_norm_(
                detail_refiner.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            total_loss += float(losses["total"].detach().item())
            batches += 1
            step += 1
            finite_checks += int(diagnostics["checks"])
            successful_finite_checks += int(diagnostics["checks"])
            max_abs_gradient = max(max_abs_gradient, float(diagnostics["max_abs_gradient"]))
            max_abs_loss = max(max_abs_loss, float(diagnostics["max_abs_loss"]))
            sample_ids.append(sample_id)
            last_epoch = current_epoch
            last_batch_offset = batch_offset + 1
            last_epoch_complete = last_batch_offset == len(order)
            if step >= args.max_steps:
                break

        metrics = evaluate_gate_b(
            bank, router, eval_loader, args.device, composer, detail_refiner
        )
        metrics.update(
            epoch=current_epoch,
            step=step,
            phase="train",
            train_loss=total_loss / max(1, batches),
            sample_ids=sample_ids,
            finite_checks={
                "count": finite_checks,
                "max_abs_loss": max_abs_loss,
                "max_abs_gradient": max_abs_gradient,
            },
        )
        history.append(metrics)
        print(json.dumps({"stage": "detail", **metrics}, ensure_ascii=False), flush=True)
        if float(metrics["detail_n_iou"]) > float(best["detail_n_iou"]):
            best = dict(metrics)
            torch.save(
                _checkpoint_payload(
                    router,
                    config,
                    selected_ids,
                    history,
                    best,
                    optimizer,
                    last_epoch,
                    step,
                    last_batch_offset,
                    last_epoch_complete,
                    detail_refiner=detail_refiner,
                ),
                run_dir / "detail_best.pt",
            )
        if successful_finite_checks > 0 and _stage_gate_passes(
            "detail", metrics, thresholds, require_targets
        ):
            stopping_reason = "threshold_met"
        elif step >= args.max_steps:
            stopping_reason = "max_steps_reached"
        torch.save(
            _checkpoint_payload(
                router,
                config,
                selected_ids,
                history,
                best,
                optimizer,
                last_epoch,
                step,
                last_batch_offset,
                last_epoch_complete,
                stopping_reason=stopping_reason,
                detail_refiner=detail_refiner,
            ),
            run_dir / "detail_last.pt",
        )
        if stopping_reason is None:
            if last_epoch_complete:
                current_epoch += 1
                current_offset = 0
            else:
                break

    if stopping_reason is None:
        stopping_reason = "max_epochs_reached"
    torch.save(
        _checkpoint_payload(
            router,
            config,
            selected_ids,
            history,
            best,
            optimizer,
            last_epoch,
            step,
            last_batch_offset,
            last_epoch_complete,
            stopping_reason=stopping_reason,
            detail_refiner=detail_refiner,
        ),
        run_dir / "detail_last.pt",
    )
    return best, history, stopping_reason, step


def _config(args: argparse.Namespace, selected_ids: list[str]) -> dict[str, Any]:
    return {
        "dataset": "SIRST4",
        "data_root": str(Path(args.data_root)),
        "split": "train",
        "seed": args.seed,
        "subset_size": args.subset_size,
        "selected_ids": selected_ids,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "lr": args.lr,
        "batch_size": 1,
        "weight_decay": 0.05,
        "k1": 24,
        "k2": 8,
        "augmentation": "seeded_hflip_vflip_rot90",
    }


def _finite_check_count(history: list[dict[str, Any]]) -> int:
    return sum(
        int(record.get("finite_checks", {}).get("count", 0))
        for record in history
    )


def run_gate_b(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    if args.detail_checkpoint:
        validate_resume_compatibility(
            args.router_checkpoint, args.detail_checkpoint, args.device
        )
    seed_everything(args.seed)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    train_loader, eval_loader, selected_ids = build_overfit_loaders(
        args.data_root, args.subset_size, args.seed
    )
    _json_write(run_dir / "selected_ids.json", selected_ids)
    config = _config(args, selected_ids)
    bank = GaussianFeatureBank().to(args.device).eval().requires_grad_(False)
    composer = SparseGaussianComposer().to(args.device)
    execution_thresholds = SMOKE_THRESHOLDS if args.smoke_test else REAL_THRESHOLDS
    resume_provenance = {
        "router": str(Path(args.router_checkpoint).resolve()) if args.router_checkpoint else None,
        "detail": str(Path(args.detail_checkpoint).resolve()) if args.detail_checkpoint else None,
        "router_initialization": "checkpoint" if args.router_checkpoint else "fresh",
        "detail_initialization": "checkpoint" if args.detail_checkpoint else "fresh",
    }

    router_best, router_history, router_reason, router_steps = _run_router_stage(
        args,
        config,
        selected_ids,
        train_loader,
        eval_loader,
        bank,
        composer,
        execution_thresholds,
        run_dir,
        require_targets=not args.smoke_test,
    )
    router_finite_check_count = _finite_check_count(router_history)
    router_finite_check_pass = router_finite_check_count > 0
    target_presence_pass = int(router_best.get("targets", 0)) > 0
    router_execution_pass = bool(
        router_finite_check_pass
        and _stage_gate_passes(
            "router",
            router_best,
            execution_thresholds,
            require_targets=not args.smoke_test,
        )
    )
    router_real_pass = bool(
        target_presence_pass and router_gate_passes(router_best, REAL_THRESHOLDS)
    )
    detail_best: dict[str, Any] | None = None
    detail_history: list[dict[str, Any]] = []
    detail_reason = "router_threshold_not_met"
    detail_steps = 0
    if router_execution_pass:
        detail_best, detail_history, detail_reason, detail_steps = _run_detail_stage(
            args,
            config,
            selected_ids,
            train_loader,
            eval_loader,
            bank,
            composer,
            execution_thresholds,
            run_dir,
            require_targets=not args.smoke_test,
        )

    detail_finite_check_count = _finite_check_count(detail_history)
    detail_finite_check_pass = detail_finite_check_count > 0
    detail_execution_pass = bool(
        detail_finite_check_pass
        and detail_best is not None
        and detail_gate_passes(detail_best, execution_thresholds)
    )
    detail_real_pass = bool(
        target_presence_pass
        and detail_best is not None
        and detail_gate_passes(detail_best, REAL_THRESHOLDS)
    )
    overall_pass = bool(
        not args.smoke_test
        and router_finite_check_pass
        and detail_finite_check_pass
        and router_real_pass
        and detail_real_pass
    )
    summary: dict[str, Any] = {
        "mode": "smoke-test" if args.smoke_test else "gate-b",
        "config": config,
        "selected_ids": selected_ids,
        "selected_target_count": int(router_best.get("targets", 0)),
        "target_presence_pass": target_presence_pass,
        "thresholds": asdict(REAL_THRESHOLDS),
        "execution_thresholds": asdict(execution_thresholds),
        "router_history": router_history,
        "detail_history": detail_history,
        "router_best_metrics": router_best,
        "detail_best_metrics": detail_best,
        "router_stopping_reason": router_reason,
        "detail_stopping_reason": detail_reason,
        "resume_provenance": resume_provenance,
        "finite_checks": {
            "router": {
                "count": router_finite_check_count,
                "passed": router_finite_check_pass,
            },
            "detail": {
                "count": detail_finite_check_count,
                "passed": detail_finite_check_pass,
            },
        },
        "router_steps": router_steps,
        "detail_steps": detail_steps,
        "router_threshold_pass": router_real_pass,
        "detail_threshold_pass": detail_real_pass,
        "router_execution_pass": router_execution_pass,
        "detail_execution_pass": detail_execution_pass,
        "overall_pass": overall_pass,
    }

    if detail_best is not None:
        detail_checkpoint = _load_checkpoint(run_dir / "detail_best.pt", args.device)
        consolidated = {
            "router": detail_checkpoint["router"],
            "detail_refiner": detail_checkpoint["detail_refiner"],
            "config": config,
            "best": {
                "router": router_best,
                "detail": detail_best,
                "overall_pass": overall_pass,
            },
        }
        torch.save(consolidated, run_dir / "gaussamr_v1_gate_b.pt")
    _json_write(run_dir / "gate_b_summary.json", summary)
    return summary


def verify_gate_b_checkpoint(
    checkpoint: str | os.PathLike[str],
    data_root: str | os.PathLike[str],
    subset_size: int = 16,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, Any]:
    seed_everything(seed)
    _, eval_loader, selected_ids = build_overfit_loaders(
        data_root, subset_size, seed
    )
    model = GaussAMRV1.from_probe_checkpoint(checkpoint, map_location=device).to(device)
    metrics = evaluate_gate_b(
        model.feature_bank,
        model.router,
        eval_loader,
        device,
        model.composer,
        model.detail_refiner,
    )
    target_presence_pass = int(metrics.get("targets", 0)) > 0
    return {
        "mode": "verify-only",
        "checkpoint": str(Path(checkpoint).resolve()),
        "data_root": str(Path(data_root)),
        "seed": seed,
        "subset_size": subset_size,
        "selected_ids": selected_ids,
        "thresholds": asdict(REAL_THRESHOLDS),
        "metrics": metrics,
        "target_presence_pass": target_presence_pass,
        "overall_pass": bool(
            target_presence_pass and detail_gate_passes(metrics, REAL_THRESHOLDS)
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        if args.verify_only:
            summary = verify_gate_b_checkpoint(
                args.checkpoint,
                args.data_root,
                args.subset_size,
                args.seed,
                args.device,
            )
        else:
            summary = run_gate_b(args)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)
    if args.smoke_test:
        return 0
    return 0 if summary["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
