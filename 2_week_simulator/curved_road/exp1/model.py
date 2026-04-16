"""
Experiment 1 model: Vision CNN (3-conv) + MLP head.

Assembles two shared modules:
    CNNBackbone  (3-conv variant) — extracts image features
    MLPHead                       — maps features to action logits

No transformer, no CVAE, no language conditioning.
This is the simplest possible VLA model: see an image, predict actions.

# ---------------------------------------------------------------------------
# WHAT PROBLEM DOES THIS SOLVE?
# ---------------------------------------------------------------------------
# Exp 1 asks the most basic question in this curriculum:
#   "Can a simple CNN + MLP follow a curved road using only camera images?"
#
# There is no language, no temporal reasoning, no uncertainty modelling.
# Just raw pixels → features → actions. This sets the performance baseline
# that every subsequent experiment is measured against.

# ---------------------------------------------------------------------------
# STRUCTURE
# ---------------------------------------------------------------------------
#   Input:  (B, 3, 128, 128)   RGB image batch
#
#   CNNBackbone (3-conv):
#     Conv(3→16, stride=2) → ReLU   →  (B, 16, 64, 64)
#     Conv(16→32, stride=2) → ReLU  →  (B, 32, 32, 32)
#     Conv(32→64, stride=2) → ReLU  →  (B, 64, 16, 16)
#     AdaptiveAvgPool2d(1)           →  (B, 64,  1,  1)
#     Flatten                        →  (B, 64)
#
#   MLPHead:
#     Linear(64 → 64) → ReLU
#     Linear(64 → 4)
#
#   Output: (B, 4)  raw logits — [forward, backward, left, right]

# ---------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------
#   from curved_road.exp1.model import Exp1Model
#   model = Exp1Model(feature_dim=64, mlp_hidden_dim=64, num_actions=4)
#   logits = model(images)   # images: (B, 3, 128, 128) → logits: (B, 4)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from shared.models.cnn_backbone import CNNBackbone
from shared.models.mlp_head import MLPHead


class Exp1Model(nn.Module):
    """CNN (3-conv) + MLP — vision-only baseline for curved road.

    Parameters
    ----------
    feature_dim     : CNN output size (must be 64 for the 3-conv backbone)
    mlp_hidden_dim  : hidden size inside the MLP head
    num_actions     : number of output actions (always 4)
    """

    def __init__(
        self,
        feature_dim: int = 64,
        mlp_hidden_dim: int = 64,
        num_actions: int = 4,
    ) -> None:
        super().__init__()

        # 3-conv backbone: (B, 3, 128, 128) → (B, 64)
        self.backbone = CNNBackbone(variant="3conv")

        # MLP head: (B, 64) → (B, 4)
        self.head = MLPHead(
            input_dim=feature_dim,
            hidden_dim=mlp_hidden_dim,
            num_actions=num_actions,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images shape: (B, 3, 128, 128)
        features = self.backbone(images)   # (B, 64)
        logits = self.head(features)       # (B, 4)
        return logits                      # (B, 4)
