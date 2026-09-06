"""G0 loss: BCE + Dice. Equal weight, no target-level loss yet (added post-G0)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def _dice(self, logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        num = 2 * (prob * target).sum(dim=(1, 2, 3))
        den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
        return 1 - (num / den)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits, target: (B,1,H,W)
        bce = self.bce(logits, target)
        dice = self._dice(logits, target).mean()
        return self.bce_weight * bce + self.dice_weight * dice
