"""Standalone training entry for background-aware sparse PSF unmixing."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from irstd_g0.data import IRSTD1KDataset
from irstd_a.diagnostics import aggregate_diagnostics, component_diagnostics
from irstd_a.losses import APSFUnmixingLoss
from irstd_a.model import build_a_model
from irstd_a.runtime import (
    atomic_json_dump,
    capture_rng_state,
    environment_manifest,
    prepare_new_run,
    restore_rng_state,
    save_checkpoint,
)
from irstd_a.targets import build_weak_targets, split_ids


def load_config(path: str) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("a_cfg", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = {"data", "model", "loss", "optim", "diagnostics", "run"}
    config = {name: getattr(module, name) for name in required if hasattr(module, name)}
    if set(config) != required:
        raise KeyError(f"config missing sections: {sorted(required - set(config))}")
    return config


def validation_score(summary: Mapping[str, Any]) -> float:
    """Lower is better; missing interpretability metrics receive a full penalty."""
    def value(name: str, default: float) -> float:
        item = summary.get(name)
        return float(item) if isinstance(item, (int, float)) and math.isfinite(float(item)) else default

    reconstruction = value("reconstruction_mae_mean", 1.0)
    precision = value("target_energy_precision_median", 0.0)
    recall = value("target_contrast_recall_median", 0.0)
    source_false = value("source_false_activation_median", 1.0)
    centroid_recall = value("centroid_recall_5px_median", 0.0)
    background_leakage = value("background_target_leakage_median", 1.0)
    overlap = value("psf_residual_overlap_median", 1.0)
    residual_target = value("residual_target_fraction_median", 1.0)
    recall_error = abs(math.log(min(max(recall, 1e-3), 1e3)))
    return (
        reconstruction
        + 0.5 * (1.0 - min(precision, 1.0))
        + 0.25 * recall_error
        + 0.5 * source_false
        + 0.5 * (1.0 - min(max(centroid_recall, 0.0), 1.0))
        + 0.5 * background_leakage
        + 0.1 * overlap
        + 0.25 * residual_target
    )


def assert_finite_batch(
    values: Mapping[str, Tensor | float],
    sample_ids: list[str],
    stage: str,
) -> None:
    bad: list[str] = []
    for name, value in values.items():
        tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
        if not torch.isfinite(tensor).all():
            bad.append(name)
    if bad:
        raise FloatingPointError(
            f"{stage} encountered non-finite values {bad} for sample_ids={sample_ids}"
        )


def validate_run_args(
    run_dir: Path,
    debug: bool,
    limit_train: int,
    limit_val: int,
) -> None:
    if limit_train < 0 or limit_val < 0:
        raise ValueError("sample limits must be non-negative")
    if (limit_train or limit_val) and not debug:
        raise ValueError("sample limits require --debug")
    if debug and "debug" not in {part.lower() for part in Path(run_dir).parts}:
        raise ValueError("--debug run directories must contain a 'debug' path component")


def validate_resume_config(
    checkpoint_config: Mapping[str, Any],
    current_config: Mapping[str, Any],
) -> None:
    if checkpoint_config != current_config:
        raise ValueError("resume checkpoint training config does not match current config")


@dataclass
class EarlyStopper:
    patience: int
    best_score: float = math.inf
    bad_epochs: int = 0

    def update(self, score: float) -> bool:
        if score < self.best_score:
            self.best_score = float(score)
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.patience > 0 and self.bad_epochs >= self.patience


def build_checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    epoch: int,
    global_step: int,
    best_score: float,
    bad_epochs: int,
    config: dict,
    train_ids: list[str],
    val_ids: list[str],
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_score": float(best_score),
        "bad_epochs": int(bad_epochs),
        "config": config,
        "train_ids": list(train_ids),
        "val_ids": list(val_ids),
        "rng_state": capture_rng_state(),
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _learning_rate(step: int, warmup: int, total: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _write_log(path: Path, message: str) -> None:
    print(message, flush=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(message + "\n")


def _write_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def _move_targets(targets: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in targets.items()}


def _make_loader(
    dataset: IRSTD1KDataset,
    batch_size: int,
    shuffle: bool,
    workers: int,
    device: torch.device,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=shuffle,
        worker_init_fn=_seed_worker,
        generator=generator,
        persistent_workers=workers > 0,
    )


def _validate_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    data_config: Mapping[str, Any],
    source_flux_scale: float,
    psf_radius: int,
    amp_enabled: bool,
) -> dict[str, Any]:
    model.eval()
    records: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            sample_ids = [str(value) for value in batch["id"]]
            image = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            targets = build_weak_targets(
                image,
                mask,
                dilation_radius=int(data_config["dilation_radius"]),
                ring_radius=int(data_config["ring_radius"]),
                source_flux_scale=source_flux_scale,
                psf_radius=psf_radius,
            )
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                prediction = model(image, return_aux=True)
            assert_finite_batch(
                {name: prediction[name] for name in ("B", "S", "T_psf", "R", "U", "reconstruction")},
                sample_ids,
                "validation",
            )
            records.extend(component_diagnostics(image, mask, prediction, targets, sample_ids))
    return aggregate_diagnostics(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    validate_run_args(run_dir, args.debug, args.limit_train, args.limit_val)
    config = load_config(args.config)
    seed = int(args.seed if args.seed is not None else config["run"]["seed"])
    epochs = int(args.epochs if args.epochs is not None else config["optim"]["epochs"])
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    if args.resume:
        if not run_dir.is_dir():
            raise FileNotFoundError(f"resume run directory does not exist: {run_dir}")
    else:
        prepare_new_run(run_dir)
    log_path = run_dir / "log.txt"
    jsonl_path = run_dir / "train.jsonl"
    _seed_everything(seed)

    full_train = IRSTD1KDataset(config["data"]["root"], split=config["data"]["train_split"], augment=True)
    train_ids, val_ids = split_ids(
        full_train.ids,
        val_count=int(config["data"]["val_count"]),
        seed=int(config["data"]["split_seed"]),
    )
    if args.limit_train:
        train_ids = train_ids[: args.limit_train]
    if args.limit_val:
        val_ids = val_ids[: args.limit_val]
    train_dataset = IRSTD1KDataset(config["data"]["root"], split=config["data"]["train_split"], augment=True)
    val_dataset = IRSTD1KDataset(config["data"]["root"], split=config["data"]["train_split"], augment=False)
    train_dataset.ids = train_ids
    val_dataset.ids = val_ids

    workers = int(config["optim"].get("num_workers", 4))
    batch_size = int(config["optim"]["batch_size"])
    train_loader = _make_loader(train_dataset, batch_size, True, workers, device, seed)
    val_loader = _make_loader(val_dataset, 1, False, workers, device, seed + 1)

    model = build_a_model(**config["model"]).to(device)
    criterion = APSFUnmixingLoss(**config["loss"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optim"]["lr"]),
        weight_decay=float(config["optim"]["weight_decay"]),
    )
    amp_enabled = bool(config["optim"].get("amp", True) and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    stopper = EarlyStopper(int(config["optim"].get("early_stop_patience", 0)))
    global_step = 0
    start_epoch = 1

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        validate_resume_config(checkpoint["config"], config)
        if checkpoint["train_ids"] != train_ids or checkpoint["val_ids"] != val_ids:
            raise ValueError("resume checkpoint split IDs do not match current split")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        global_step = int(checkpoint["global_step"])
        start_epoch = int(checkpoint["epoch"]) + 1
        stopper.best_score = float(checkpoint["best_score"])
        stopper.bad_epochs = int(checkpoint["bad_epochs"])
        restore_rng_state(checkpoint["rng_state"])
    else:
        atomic_json_dump(config, run_dir / "config_snapshot.json")
        atomic_json_dump(environment_manifest(), run_dir / "environment.json")
        atomic_json_dump({"ids": train_ids}, run_dir / "train_ids.json")
        atomic_json_dump({"ids": val_ids}, run_dir / "val_ids.json")

    accumulation = int(config["optim"].get("grad_accum_steps", 1))
    total_optimizer_steps = epochs * max(1, math.ceil(len(train_loader) / accumulation))
    warmup_steps = int(config["optim"].get("warmup_steps", 0))
    base_lr = float(config["optim"]["lr"])
    grad_clip = float(config["optim"].get("grad_clip", 0.0))
    checkpoint_interval = int(config["run"].get("checkpoint_interval", 10))
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    _write_log(log_path, f"[A] device={device} seed={seed} params={parameter_count:,} train={len(train_ids)} val={len(val_ids)}")

    started = time.time()
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        sums = {name: 0.0 for name in (*criterion.weights, "total")}
        batches = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader, start=1):
            sample_ids = [str(value) for value in batch["id"]]
            image = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            targets = build_weak_targets(
                image,
                mask,
                dilation_radius=int(config["data"]["dilation_radius"]),
                ring_radius=int(config["data"]["ring_radius"]),
                source_flux_scale=float(config["model"]["source_flux_scale"]),
                psf_radius=int(config["model"]["kernel_size"]) // 2,
            )
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                prediction = model(image, return_aux=True)
                total, terms = criterion(image, mask, prediction, targets)
                scaled_total = total / accumulation
            assert_finite_batch(terms, sample_ids, "training")
            scaler.scale(scaled_total).backward()
            should_step = batch_index % accumulation == 0 or batch_index == len(train_loader)
            if should_step:
                scaler.unscale_(optimizer)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                lr = _learning_rate(global_step, warmup_steps, total_optimizer_steps, base_lr)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            for name, value in terms.items():
                sums[name] += float(value.detach().cpu())
            batches += 1

        validation = _validate_epoch(
            model,
            val_loader,
            device,
            config["data"],
            float(config["model"]["source_flux_scale"]),
            int(config["model"]["kernel_size"]) // 2,
            amp_enabled,
        )
        score = validation_score(validation)
        improved = stopper.update(score)
        averages = {name: value / max(1, batches) for name, value in sums.items()}
        peak_memory_mb = (
            torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
        )
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_s": time.time() - started,
            "peak_memory_mb": peak_memory_mb,
            "train_loss": averages,
            "validation_score": score,
            "validation": validation,
            "best_score": stopper.best_score,
            "bad_epochs": stopper.bad_epochs,
        }
        _write_jsonl(jsonl_path, record)
        _write_log(
            log_path,
            f"[A] epoch {epoch}/{epochs} loss={averages['total']:.5f} "
            f"presence={averages['presence']:.4f} amplitude={averages['amplitude']:.4f} "
            f"sparse={averages['sparse']:.4f} "
            f"val_score={score:.5f} best={stopper.best_score:.5f} "
            f"centroid={float(validation.get('centroid_recall_5px_median', 0.0)):.3f} "
            f"falseS={float(validation.get('source_false_activation_median', 1.0)):.3f} "
            f"massX={float(validation.get('source_noncenter_mass_ratio_median', 0.0)):.3f} "
            f"outsideT={float(validation.get('target_outside_energy_ratio_median', 0.0)):.3f} "
            f"recall={float(validation.get('target_contrast_recall_median', 0.0)):.3f} "
            f"precision={float(validation.get('target_energy_precision_median', 0.0)):.3f} "
            f"bad={stopper.bad_epochs} lr={optimizer.param_groups[0]['lr']:.2e} "
            f"peak_mb={peak_memory_mb:.0f} elapsed={record['elapsed_s']:.1f}s",
        )
        payload = build_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            best_score=stopper.best_score,
            bad_epochs=stopper.bad_epochs,
            config=config,
            train_ids=train_ids,
            val_ids=val_ids,
        )
        if improved:
            save_checkpoint(run_dir / "a_best.pt", payload)
        if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
            save_checkpoint(run_dir / f"a_epoch_{epoch:03d}.pt", payload)
        save_checkpoint(run_dir / "a_last.pt", payload)
        if stopper.should_stop:
            _write_log(log_path, f"[A] early stop at epoch {epoch} after {stopper.bad_epochs} non-improving epochs")
            break

    _write_log(log_path, f"[A] complete best_validation_score={stopper.best_score:.6f}")


if __name__ == "__main__":
    main()
