"""IRSTD datasets for G0 baseline.

Contract:
- Train input: single-frame infrared image I + binary mask Y (supervision only).
- Eval input: I only; Y is loaded only for metric computation.
- No label leakage: Y is never used as a model input feature.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _read_split(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _resolve_sirst(idx: str, root: str) -> str:
    # SIRST splits are bare integers or names; images are <idx>.png.
    return os.path.join(root, "images", f"{idx}.png")


# -----------------------------------------------------------------------------
# IRSTD-1K (512x512, 80/20 split files in data/IRSTD-1K/80_20/)
# -----------------------------------------------------------------------------

class IRSTD1KDataset(Dataset):
    def __init__(self, root: str, split: str = "train", augment: bool = True):
        self.root = root
        self.split = split
        self.augment = augment and (split == "train")
        self.ids = _read_split(os.path.join(root, "80_20", f"{split}.txt"))

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> dict:
        from PIL import Image
        sid = self.ids[i]
        img = np.asarray(Image.open(os.path.join(self.root, "images", f"{sid}.png")).convert("L"), dtype=np.float32)
        msk = np.asarray(Image.open(os.path.join(self.root, "masks", f"{sid}.png")).convert("L"), dtype=np.float32)
        msk = (msk > 127).astype(np.float32)
        if self.augment:
            img, msk = _augment(img, msk)
        img = _normalize(img)
        return {
            "image": torch.from_numpy(img).unsqueeze(0),  # (1,H,W)
            "mask": torch.from_numpy(msk).unsqueeze(0),   # (1,H,W)
            "id": sid,
        }


# -----------------------------------------------------------------------------
# SIRST-UAVB / SIRST4 (cross-dataset eval only)
# -----------------------------------------------------------------------------

class SIRSTUAVBDataset(Dataset):
    """Cross-dataset eval only. Splits in data/SIRST-UAVB_OnlyUAV_Form/img_idx/."""

    def __init__(self, root: str, split: str = "test"):
        self.root = root
        ids = _read_split(os.path.join(root, "img_idx", f"{split}.txt"))
        self.ids = ids

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> dict:
        from PIL import Image
        sid = self.ids[i]
        img = np.asarray(Image.open(os.path.join(self.root, "images", f"{sid}.png")).convert("L"), dtype=np.float32)
        # SIRST-UAVB masks carry a "_mask" suffix (e.g. 2401_mask.png).
        msk = np.asarray(Image.open(os.path.join(self.root, "masks", f"{sid}_mask.png")).convert("L"), dtype=np.float32)
        msk = (msk > 127).astype(np.float32)
        img = _normalize(img)
        return {
            "image": torch.from_numpy(img).unsqueeze(0),
            "mask": torch.from_numpy(msk).unsqueeze(0),
            "id": sid,
        }


class SIRST4Dataset(Dataset):
    """Cross-dataset eval only. Splits in data/SIRST4-ForLiTE/img_idx/."""

    def __init__(self, root: str, split: str = "test"):
        self.root = root
        # Use the original test split names; for SIRST4 "test.txt" == "test_SIRST4.txt".
        ids = _read_split(os.path.join(root, "img_idx", f"{split}.txt"))
        self.ids = ids

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> dict:
        from PIL import Image
        sid = self.ids[i]
        img = np.asarray(Image.open(os.path.join(self.root, "images", f"{sid}.png")).convert("L"), dtype=np.float32)
        msk = np.asarray(Image.open(os.path.join(self.root, "masks", f"{sid}.png")).convert("L"), dtype=np.float32)
        msk = (msk > 127).astype(np.float32)
        img = _normalize(img)
        return {
            "image": torch.from_numpy(img).unsqueeze(0),
            "mask": torch.from_numpy(msk).unsqueeze(0),
            "id": sid,
        }


# -----------------------------------------------------------------------------
# Augment / Normalize
# -----------------------------------------------------------------------------

def _normalize(img: np.ndarray) -> np.ndarray:
    # Min-max to [0,1]. Per-image normalization keeps relative contrast.
    lo, hi = float(img.min()), float(img.max())
    if hi - lo < 1e-6:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - lo) / (hi - lo)).astype(np.float32)


def _augment(img: np.ndarray, msk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Random h-flip / v-flip / 90° rotate. No photometric changes for IR."""
    # h-flip
    if np.random.rand() < 0.5:
        img = img[:, ::-1].copy()
        msk = msk[:, ::-1].copy()
    # v-flip
    if np.random.rand() < 0.5:
        img = img[::-1, :].copy()
        msk = msk[::-1, :].copy()
    # 90° rotate K=4
    k = np.random.randint(0, 4)
    if k:
        img = np.rot90(img, k=k).copy()
        msk = np.rot90(msk, k=k).copy()
    return img, msk


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------

@dataclass
class DatasetSpec:
    name: str
    root: str
    split: str
    augment: bool


def build_dataset(spec: DatasetSpec) -> Dataset:
    if spec.name == "irstd1k":
        return IRSTD1KDataset(spec.root, split=spec.split, augment=spec.augment)
    if spec.name == "sirst_uavb":
        assert not spec.augment, "SIRST-UAVB is eval-only"
        return SIRSTUAVBDataset(spec.root, split=spec.split)
    if spec.name == "sirst4":
        assert not spec.augment, "SIRST4 is eval-only"
        return SIRST4Dataset(spec.root, split=spec.split)
    raise ValueError(f"Unknown dataset: {spec.name}")
