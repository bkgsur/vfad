# CNN — Convolutional Neural Network

## What it is
A neural network designed to process images by sliding small filters across the input,
detecting local patterns (edges, curves, colour regions) at each position.
The output is a compact feature vector summarising what the image contains.

## Why it matters in this project
Every experiment (1–9) starts with a CNN backbone. It is the "eyes" of the model —
the part that reads the camera image and converts it into a form the rest of the
network can reason about.

Without the CNN, the model would have to work with 49,152 raw pixel values
(128×128×3). That's too many numbers, arranged in a way that has no structure
a linear layer can exploit. The CNN compresses this into 64 or 128 meaningful numbers.

## How it works

Each convolutional layer slides a small filter (3×3 pixels) across the image.
At every position, it computes a dot product between the filter weights and the
local patch of pixels. This produces one number per position — a score for how
much that pattern appears there.

**Shape walkthrough (Exp 1, 3-conv backbone, B=2 images):**

```
Input:          (2, 3, 128, 128)   — 2 RGB images, 128×128 pixels

Conv1 (16 filters, stride=2):
  → (2, 16, 64, 64)   — 16 pattern detectors, spatial size halved

Conv2 (32 filters, stride=2):
  → (2, 32, 32, 32)   — 32 pattern detectors, halved again

Conv3 (64 filters, stride=2):
  → (2, 64, 16, 16)   — 64 pattern detectors, halved again

AdaptiveAvgPool2d(1):
  → (2, 64, 1, 1)     — collapse each feature map to one number (its average)

Flatten:
  → (2, 64)           — final feature vector, one per image
```

**stride=2** does double duty: it moves the filter 2 pixels at a time, halving
the spatial size while also acting as downsampling — no separate pooling layer needed.

**Why channels grow (3→16→32→64):** Early layers detect simple patterns (edges).
Later layers combine simple patterns into complex ones (curves, junctions). More
complex patterns need more channels to represent their variety.

## Key intuition
The CNN doesn't see "a left curve" directly. It learns filter weights that detect
edge orientations, then combines those detections into curve detectors, then combines
those into road-structure detectors. The final 64-number vector is a compressed
description of the road geometry — not pixels, but meaning.

## Where in the code
`2_week_simulator/shared/models/cnn_backbone.py`
- `CNNBackbone(variant="3conv")` → output dim 64 (Exp 1–3)
- `CNNBackbone(variant="4conv")` → output dim 128 (Exp 4–9)

## Experiments that use this
All 9 experiments. The 3-conv variant is used in Exp 1–3 (curved road);
the 4-conv variant in Exp 4–9 (forked road, harder task needing richer features).

## See also
- [MLP Head](mlp_head.md) — what processes the CNN output
- [Transforms](transforms.md) — how images are preprocessed before the CNN sees them
