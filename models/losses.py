"""
losses.py — Physics-informed loss functions for reef segmentation on CPU.

PhysicsInformedLoss = BCE + λ_depth * depth_consistency + λ_smooth * smoothness

depth_consistency: predictions in shallow water should tend toward reef (positive),
    predictions in deep water should tend toward no-reef (negative).
    Penalises the mismatch between predicted probability and the
    physics-derived likelihood of reef presence at that depth.

smoothness: total-variation penalty on the predicted mask — reduces
    salt-and-pepper noise common in CPU-trained models on small datasets.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthConsistencyLoss(nn.Module):
    """
    Penalises predictions inconsistent with the depth physics.

    Shallow pixels (normalised depth near 1.0) should have HIGH reef probability.
    Deep pixels (normalised depth near 0.0) should have LOW reef probability.

    The penalty is the L2 distance between sigmoid(logits) and the
    depth-derived prior p_reef = depth_norm (linear heuristic).
    """

    def forward(self, logits: torch.Tensor, depth_norm: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        # depth_norm ∈ [0,1]: 1 = at surface (shallow), 0 = at optical limit (deep)
        prior = depth_norm.clamp(0.0, 1.0)
        return F.mse_loss(probs, prior)


class SmoothnessLoss(nn.Module):
    """Total-variation smoothness penalty on the predicted logit map."""

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        diff_h = (logits[:, :, 1:, :] - logits[:, :, :-1, :]).abs()
        diff_w = (logits[:, :, :, 1:] - logits[:, :, :, :-1]).abs()
        return diff_h.mean() + diff_w.mean()


class PhysicsInformedLoss(nn.Module):
    """
    Combined loss for reef segmentation with bathymetry physics constraints.

    Args:
        lambda_depth:  Weight for the depth-consistency term (default 0.3).
        lambda_smooth: Weight for the smoothness/TV term (default 0.1).

    Forward signature:
        loss = criterion(logits, masks, depth_norm=None)

    If depth_norm is None the depth_consistency term is skipped (inference
    datasets without a depth input still work, just without the physics term).
    """

    def __init__(self, lambda_depth: float = 0.3, lambda_smooth: float = 0.1) -> None:
        super().__init__()
        self.lambda_depth = lambda_depth
        self.lambda_smooth = lambda_smooth
        self._bce = nn.BCEWithLogitsLoss()
        self._dice = _DiceLoss()
        self._depth = DepthConsistencyLoss()
        self._smooth = SmoothnessLoss()

    def forward(
        self,
        logits: torch.Tensor,
        masks: torch.Tensor,
        depth_norm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = self._bce(logits, masks) + self._dice(logits, masks)

        if depth_norm is not None and self.lambda_depth > 0:
            loss = loss + self.lambda_depth * self._depth(logits, depth_norm)

        if self.lambda_smooth > 0:
            loss = loss + self.lambda_smooth * self._smooth(logits)

        return loss


class _DiceLoss(nn.Module):
    """Soft Dice loss — works well for imbalanced reef/no-reef pixels."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2.0 * intersection + eps) / (union + eps)
        return 1.0 - dice.mean()
