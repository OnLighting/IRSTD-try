"""G0 training entry.

Trains I-only Swin-UNet on IRSTD-1K. Saves checkpoint to runs/g0/.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent))

from irstd_g0.data import build_dataset, DatasetSpec
from irstd_g0.model import build_g0_model, count_params
from irstd_g0.losses import BCEDiceLoss
from irstd_g0.metrics import MetricAccumulator


def load_config(path: str) -> dict:
    spec = importlib.util.spec_from_file_location("g0_cfg", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = {k: v for k, v in vars(mod).items() if not k.startswith("_")}
    return cfg


def cosine_warmup(step: int, warmup: int, total: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/g0_irstd1k.py")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int, default=0, help="Limit train samples (debug only). 0 = no limit.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(args.seed)

    run_dir = Path(cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    # Data
    train_spec = DatasetSpec(name="irstd1k", root=cfg["data_root"], split=cfg["train_split"], augment=True)
    train_ds = build_dataset(train_spec)
    if args.limit_train:
        train_ds.ids = train_ds.ids[: args.limit_train]
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["optim"]["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=(args.device == "cuda"),
        drop_last=True,
    )

    # Model
    model = build_g0_model(**cfg["model"]).to(args.device)
    print(f"[G0] trainable params: {count_params(model):,}")

    # Loss
    loss_fn = BCEDiceLoss(**cfg["loss"]).to(args.device)

    # Optim
    base_lr = cfg["optim"]["lr"]
    wd = cfg["optim"]["weight_decay"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=wd)
    total_steps = cfg["optim"]["epochs"] * max(1, len(train_loader))
    warmup = cfg["optim"]["warmup_steps"]
    grad_clip = cfg["optim"]["grad_clip"]

    # Val loader: IRSTD convention (DNANet/UIUNet) evaluates on the test split
    # each epoch and keeps the best-IoU checkpoint. Mild test-set leakage, but
    # matches the literature baseline protocol.
    val_spec = DatasetSpec(name="irstd1k", root=cfg["data_root"], split=cfg["test_split"], augment=False)
    val_ds = build_dataset(val_spec)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    val_acc = MetricAccumulator()

    best_iou = -1.0
    best_epoch = 0
    bad_epochs = 0
    early_stop = cfg["optim"].get("early_stop_patience", 0)  # 0 = disabled
    step = 0
    stopped = False
    t0 = time.time()
    for epoch in range(cfg["optim"]["epochs"]):
        model.train()
        ep_loss = 0.0
        n = 0
        for batch in train_loader:
            img = batch["image"].to(args.device, non_blocking=True)
            msk = batch["mask"].to(args.device, non_blocking=True)
            lr_now = cosine_warmup(step, warmup, total_steps, base_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now
            logits = model(img)
            loss = loss_fn(logits, msk)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            ep_loss += float(loss.item())
            n += 1
            step += 1
        avg = ep_loss / max(1, n)
        elapsed = time.time() - t0

        # Per-epoch val IoU -> best checkpoint + early stop
        model.eval()
        val_acc.records.clear()
        with torch.no_grad():
            for batch in val_loader:
                prob = torch.sigmoid(model(batch["image"].to(args.device))).cpu()
                val_acc.update_batch(prob, batch["mask"])
        iou = val_acc.summary()["iou_mean"]
        if iou > best_iou:
            best_iou, best_epoch, bad_epochs = iou, epoch + 1, 0
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch + 1, "iou": iou},
                       run_dir / "g0_best.pt")
        else:
            bad_epochs += 1

        print(f"[G0] epoch {epoch+1}/{cfg['optim']['epochs']} loss={avg:.4f} iou={iou:.4f} "
              f"best={best_iou:.4f}@{best_epoch} lr={lr_now:.2e} elapsed={elapsed:.1f}s")

        if early_stop and bad_epochs >= early_stop:
            print(f"[G0] early stop at epoch {epoch+1} (no improvement for {early_stop} epochs)")
            stopped = True
            break

    # Save last checkpoint + config snapshot (best already saved during loop)
    ckpt_path = run_dir / "g0_last.pt"
    torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch + 1}, ckpt_path)
    with open(run_dir / "config_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"[G0] saved checkpoint -> {ckpt_path}" + (" (early stopped)" if stopped else ""))
    print(f"[G0] best iou={best_iou:.4f} @ epoch {best_epoch} -> g0_best.pt")


if __name__ == "__main__":
    main()
