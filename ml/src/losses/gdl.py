from __future__ import annotations

import torch
import torch.nn as nn


class GeneralizedDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        w = 1.0 / (targets.sum(dim=(2, 3)) ** 2 + self.smooth)
        num = (w * (probs * targets).sum(dim=(2, 3))).sum(dim=1)
        den = (w * (probs + targets).sum(dim=(2, 3))).sum(dim=1)
        dice = 1 - (2 * num + self.smooth) / (den + self.smooth)
        return dice.mean()
