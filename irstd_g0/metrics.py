"""G0 metrics: pixel-level IoU / nIoU + target-level Pd / Fa.

Decision record §evaluation:
- Pixel: IoU, nIoU.
- Target: detection probability Pd, false-alarm rate Fa. Connected-component
  matching rule must be stated explicitly.
- Connectivity rule: 8-connectivity. Predicted component is a hit if its centroid
  lies within MATCH_RADIUS_PX pixels of any GT component centroid.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
import torch
from scipy import ndimage


MATCH_RADIUS_PX = 5


# -----------------------------------------------------------------------------
# Per-image metric accumulation
# -----------------------------------------------------------------------------

@dataclass
class ImageMetrics:
    iou: float
    n_iou: float
    pd: float           # 1 if any predicted component hit any GT component else 0 (per-image Pd)
    fa: float           # #unmatched predicted components per image (per-image Fa count)
    n_gt: int
    n_pred: int


def _binary(mask: np.ndarray, thr: float = 0.5) -> np.ndarray:
    return (mask > thr).astype(np.uint8)


def _centroids(mask: np.ndarray) -> np.ndarray:
    """Return (K, 2) array of centroids (y, x) of 8-connected components."""
    if mask.sum() == 0:
        return np.zeros((0, 2), dtype=np.float32)
    structure = np.ones((3, 3), dtype=np.uint8)
    lab, n = ndimage.label(mask, structure=structure)
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32)
    com = ndimage.center_of_mass(mask, lab, range(1, n + 1))
    return np.asarray(com, dtype=np.float32)


def _match(pred: np.ndarray, gt: np.ndarray, radius: float = MATCH_RADIUS_PX) -> tuple[int, int, int]:
    """Greedy nearest-centroid matching. Returns (hits, unmatched_pred, unmatched_gt)."""
    p, g = _centroids(pred), _centroids(gt)
    if len(p) == 0 and len(g) == 0:
        return 0, 0, 0
    if len(p) == 0:
        return 0, 0, len(g)
    if len(g) == 0:
        return 0, len(p), 0
    # pairwise distances
    diff = p[:, None, :] - g[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    hit_pred = np.zeros(len(p), dtype=bool)
    hit_gt = np.zeros(len(g), dtype=bool)
    # greedily match the closest pair first
    order = np.argsort(dist, axis=None)
    for idx in order:
        i, j = divmod(int(idx), dist.shape[1])
        if hit_pred[i] or hit_gt[j]:
            continue
        if dist[i, j] <= radius:
            hit_pred[i] = True
            hit_gt[j] = True
    return int(hit_pred.sum()), int((~hit_pred).sum()), int((~hit_gt).sum())


def _per_image(prob: np.ndarray, gt: np.ndarray, thr: float = 0.5) -> ImageMetrics:
    pred = _binary(prob, thr)
    # IoU (image-level)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = float(inter) / float(union) if union > 0 else 1.0
    # nIoU = mean IoU over {target-present, target-absent} normalized categories
    # IRSTD convention: nIoU = 0.5*(IoU_present + IoU_absent) where absent IoU = TN/(TN+FP)
    if gt.sum() > 0:
        iou_present = iou
        tn = float(((pred == 0) & (gt == 0)).sum())
        fp = float(((pred == 1) & (gt == 0)).sum())
        denom_abs = tn + fp
        iou_absent = tn / denom_abs if denom_abs > 0 else 1.0
        n_iou = 0.5 * (iou_present + iou_absent)
    else:
        # no GT targets: Pd is undefined; report nIoU as background-only IoU.
        tn = float(((pred == 0) & (gt == 0)).sum())
        fp = float(((pred == 1) & (gt == 0)).sum())
        denom_abs = tn + fp
        iou_absent = tn / denom_abs if denom_abs > 0 else 1.0
        iou = iou_absent
        n_iou = iou_absent
    hits, fa, miss = _match(pred, gt)
    pd = 1.0 if hits > 0 else 0.0
    return ImageMetrics(iou=iou, n_iou=n_iou, pd=pd, fa=float(fa),
                        n_gt=int(_centroids(gt).shape[0]), n_pred=int(_centroids(pred).shape[0]))


# -----------------------------------------------------------------------------
# Aggregators
# -----------------------------------------------------------------------------

class MetricAccumulator:
    def __init__(self):
        self.records: list[ImageMetrics] = []

    def update(self, prob: np.ndarray, gt: np.ndarray) -> None:
        self.records.append(_per_image(prob, gt))

    def update_batch(self, prob: torch.Tensor, gt: torch.Tensor) -> None:
        # prob: (B,1,H,W) after sigmoid; gt: (B,1,H,W) binary float.
        prob = prob.detach().cpu().numpy()
        gt = gt.detach().cpu().numpy()
        for i in range(prob.shape[0]):
            self.update(prob[i, 0], gt[i, 0])

    def summary(self) -> dict:
        if not self.records:
            return {}
        ious = np.array([r.iou for r in self.records])
        n_ious = np.array([r.n_iou for r in self.records])
        pds = np.array([r.pd for r in self.records])
        fas = np.array([r.fa for r in self.records])
        return {
            "n_images": int(len(self.records)),
            "iou_mean": float(ious.mean()),
            "n_iou_mean": float(n_ious.mean()),
            "pd": float(pds.mean()),  # image-level recall
            "fa_per_image": float(fas.mean()),
            "match_rule": {
                "connectivity": "8-neighborhood",
                "radius_px": MATCH_RADIUS_PX,
                "matching": "greedy nearest centroid",
            },
        }


def to_jsonable(summary: dict) -> dict:
    """Recursively convert to JSON-friendly types."""
    out = {}
    for k, v in summary.items():
        if isinstance(v, dict):
            out[k] = to_jsonable(v)
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out
