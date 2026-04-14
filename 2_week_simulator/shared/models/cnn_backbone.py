"""
CNN feature extractor — shared across all 9 experiments.

# ---------------------------------------------------------------------------
# WHAT PROBLEM DOES THIS SOLVE?
# ---------------------------------------------------------------------------
# The model receives a 128×128 RGB image — that's 128×128×3 = 49,152 numbers.
# You cannot feed 49,152 numbers directly into a linear layer and expect it
# to learn anything useful about road structure. The numbers are raw pixels;
# what the model needs to understand is shapes, edges, and curves.
#
# The CNN backbone solves this by progressively:
#   1. Detecting local patterns (edges, colour boundaries) with convolutions
#   2. Shrinking the spatial size at each step (downsampling)
#   3. Increasing the number of feature channels (richer representation)
# Until the entire 128×128 image is compressed into a small vector (64 or 128
# numbers) that captures the road structure — not individual pixels.

# ---------------------------------------------------------------------------
# KEY CONCEPTS YOU WILL LEARN HERE
# ---------------------------------------------------------------------------
# 1. nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
#    A convolution slides a small filter (e.g. 3×3) across the entire image,
#    computing a dot product at every position. Each filter learns to detect
#    ONE type of pattern (e.g. vertical edge, left curve).
#    out_channels = how many different filters (patterns) to learn.
#    A layer with 16 filters produces 16 feature maps — 16 different pattern
#    detectors applied to the same image simultaneously.
#
# 2. stride=2 — moves the filter 2 pixels at a time instead of 1.
#    This halves the spatial size of the output:
#      128×128 → stride=2 → 64×64
#    It does the job of MaxPool (downsampling) built into the convolution
#    itself, saving parameters vs Conv + separate MaxPool.
#
# 3. padding=1 — adds a 1-pixel border of zeros around the input before
#    the convolution. Without padding, a 3×3 filter on a 128×128 image
#    produces 126×126 (loses 1 pixel on each side). padding=1 preserves
#    the intended output size when using stride=1, and gives clean halving
#    when using stride=2.
#
# 4. Feature maps — the output of each conv layer.
#    After Conv1 (16 filters): shape is (B, 16, 64, 64)
#    16 different "views" of the image, each 64×64.
#    Each view has learned to detect something different.
#
# 5. Spatial dimensions shrink, channels grow — this is the universal
#    pattern in CNNs:
#      Input:  (B,  3, 128, 128)   3 colour channels, full resolution
#      Conv1:  (B, 16,  64,  64)   16 patterns, half resolution
#      Conv2:  (B, 32,  32,  32)   32 patterns, quarter resolution
#      Conv3:  (B, 64,  16,  16)   64 patterns, eighth resolution
#    Fewer spatial pixels but richer description of what's in each position.
#
# 6. AdaptiveAvgPool2d(1) — global average pooling.
#    Takes a (B, 64, 16, 16) tensor and averages each of the 64 feature
#    maps down to a single number → (B, 64, 1, 1).
#    This collapses "where in the image" entirely, leaving only "what was
#    detected". It also makes the backbone input-size agnostic.
#
# 7. nn.Flatten() — reshapes (B, 64, 1, 1) → (B, 64).
#    Converts the spatial tensor into a flat vector the MLP head can consume.
#
# 8. WHY TWO VARIANTS?
#    3-conv backbone (feature_dim=64): curved road is a single path,
#    simpler visual structure, fewer features needed.
#    4-conv backbone (feature_dim=128): forked road has a junction —
#    the model needs to distinguish two visually similar paths, so we
#    give it a deeper, wider network to extract richer features.

# ---------------------------------------------------------------------------
# STRUCTURE
# ---------------------------------------------------------------------------
# 3-conv variant (Exp 1–3, curved road):
#   (B, 3, 128, 128)
#     → Conv(3→16, stride=2) → ReLU   →  (B, 16, 64, 64)
#     → Conv(16→32, stride=2) → ReLU  →  (B, 32, 32, 32)
#     → Conv(32→64, stride=2) → ReLU  →  (B, 64, 16, 16)
#     → AdaptiveAvgPool2d(1)          →  (B, 64,  1,  1)
#     → Flatten                       →  (B, 64)
#
# 4-conv variant (Exp 4–9, forked road):
#   Same as above, plus:
#     → Conv(64→128, stride=2) → ReLU →  (B, 128, 8, 8)
#     → AdaptiveAvgPool2d(1)          →  (B, 128, 1, 1)
#     → Flatten                       →  (B, 128)

# ---------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------
#   from shared.models.cnn_backbone import CNNBackbone
#   backbone = CNNBackbone(variant="3conv")   # or "4conv"
#   features = backbone(images)              # (B, 3, 128, 128) → (B, 64)
"""

from __future__ import annotations

import torch.nn as nn


class CNNBackbone(nn.Module):
    """
    Convolutional feature extractor.
    variant="3conv" → output dim 64   (curved road, Exp 1–3)
    variant="4conv" → output dim 128  (forked road, Exp 4–9)
    """

    def __init__(self, variant: str = "3conv") -> None:
        super().__init__()

        if variant not in ("3conv", "4conv"):
            raise ValueError(f"variant must be '3conv' or '4conv', got '{variant}'")

        # Each conv block: Conv2d → ReLU
        # stride=2 halves spatial resolution at each step.
        # padding=1 ensures the halving is clean (no off-by-one pixel loss).

        # Block 1: RGB (3 channels) → 16 feature maps, 128×128 → 64×64
        block1 = [nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), nn.ReLU()]

        # Block 2: 16 → 32 feature maps, 64×64 → 32×32
        block2 = [nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU()]

        # Block 3: 32 → 64 feature maps, 32×32 → 16×16
        block3 = [nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU()]

        if variant == "3conv":
            # 3 blocks then pool and flatten → output: (B, 64)
            conv_blocks = block1 + block2 + block3

        else:  # 4conv
            # Block 4: 64 → 128 feature maps, 16×16 → 8×8
            # Wider output (128) gives the model more capacity for the harder
            # forked-road task where two visually similar paths must be distinguished.
            block4 = [nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU()]
            conv_blocks = block1 + block2 + block3 + block4

        # AdaptiveAvgPool2d(1) collapses any spatial size (H×W) to (1×1)
        # by averaging across the entire feature map.
        # "Adaptive" means you specify the OUTPUT size (1×1), not the kernel
        # size — PyTorch figures out the kernel automatically.
        # This is more flexible than fixed MaxPool and handles any input resolution.
        conv_blocks.append(nn.AdaptiveAvgPool2d(1))

        # Flatten turns (B, C, 1, 1) → (B, C)
        # start_dim=1 means flatten everything EXCEPT the batch dimension.
        conv_blocks.append(nn.Flatten(start_dim=1))

        # Wrap the list of layers into a Sequential so we can call it as one unit.
        # The * unpacks the list: Sequential(*[a, b, c]) = Sequential(a, b, c)
        self.features = nn.Sequential(*conv_blocks)

        # Store output dimension so the model assembler can query it
        # without having to hard-code 64 or 128 elsewhere.
        self.output_dim = 64 if variant == "3conv" else 128

    def forward(self, x):
        # x shape: (B, 3, 128, 128)
        # self.features applies all conv blocks + pool + flatten in sequence
        # output shape: (B, 64) for 3conv  or  (B, 128) for 4conv
        return self.features(x)
