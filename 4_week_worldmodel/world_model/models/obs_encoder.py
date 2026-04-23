"""
Observation Encoder — obs_encoder.py
=====================================

WHAT PROBLEM DOES THIS SOLVE?
    The world model receives raw camera frames (pixels) as input.
    Pixels are high-dimensional (3 × 128 × 128 = 49,152 numbers per image)
    and most of that information is redundant or irrelevant.
    The ObsEncoder compresses each frame into a small, dense latent vector
    (256 numbers) that captures only the meaningful visual features.
    All other parts of the world model work in this compact latent space,
    not in pixel space — making learning faster and more stable.

KEY CONCEPTS YOU WILL LEARN HERE
    - Convolutional layers: how to extract spatial features from images
    - stride=2: how to downsample spatial size without a separate pooling layer
    - Flattening: converting a 3D feature map to a 1D vector per sample
    - Linear projection: mapping a flat vector to a fixed-size latent space
    - Shape tracking: how tensor dimensions change at each layer

STRUCTURE
    Input:   (batch, 3, 128, 128)  — a batch of RGB images
    Conv 1:  (batch, 32, 64, 64)   — 32 feature maps, spatial halved (stride=2)
    Conv 2:  (batch, 64, 32, 32)   — 64 feature maps, spatial halved (stride=2)
    Conv 3:  (batch, 128, 16, 16)  — 128 feature maps, spatial halved (stride=2)
    Conv 4:  (batch, 128, 16, 16)  — 128 feature maps, spatial UNCHANGED (stride=1, refines features)
    Flatten: (batch, 32768)        — 128 × 16 × 16 = 32,768
    FC:      (batch, 256)          — final latent vector (~9.1M params total)

PARAMETER COUNT  (~9.1M total)
    Formula for a Conv2d layer:
        params = (in_channels × kernel_h × kernel_w × out_channels) + out_channels
                  ↑ weights                                            ↑ bias (one per output channel)

    Formula for a Linear layer:
        params = (in_features × out_features) + out_features
                  ↑ weights                     ↑ bias

    Layer-by-layer breakdown:
        Conv1:  (3  × 4 × 4 × 32)  + 32  =  1,536 + 32  =      1,568
        Conv2:  (32 × 4 × 4 × 64)  + 64  = 32,768 + 64  =     32,832
        Conv3:  (64 × 4 × 4 × 128) + 128 = 131,072 + 128 =    131,200
        Conv4:  (128 × 3 × 3 × 128)+ 128 = 147,456 + 128 =    147,584
        FC:     (32768 × 256)       + 256 = 8,388,608+256 =  8,388,864
                                            ─────────────────────────
        Total:                                              ≈ 8,702,048  (~8.7M)

    Note: the FC layer alone accounts for 96% of all parameters.
    The 4 conv layers combined are only ~313K — tiny in comparison.
    This is why the spatial size before flattening matters so much:
    32768 inputs × 256 outputs = 8.4M weights in a single layer.

USAGE
    encoder = ObsEncoder(latent_dim=256)
    z = encoder(images)   # images: (B, 3, 128, 128) → z: (B, 256)
"""

import torch
import torch.nn as nn


class ObsEncoder(nn.Module):
    """
    CNN encoder that maps a raw RGB image to a fixed-size latent vector.

    Each conv layer:
      - Increases the number of channels (learns more feature types)
      - Halves the spatial size via stride=2 (no separate MaxPool needed)
      - Applies ReLU to introduce non-linearity (without it, stacking layers
        would collapse to a single linear transformation — useless for vision)

    After the conv stack, the spatial feature map is flattened and projected
    to the latent dimension via a single fully-connected layer.
    """

    def __init__(self, latent_dim: int = 256) -> None:
        """
        Args:
            latent_dim: size of the output latent vector.
                        256 is a common choice — large enough to capture
                        scene structure, small enough to stay manageable.
        """
        super().__init__()

        # --- Convolutional stack ---
        # nn.Sequential chains layers so we can call them with a single forward pass.
        # Each Conv2d is followed by ReLU — this is the standard pattern.
        self.conv = nn.Sequential(

            # Layer 1: 3 input channels (RGB) → 32 feature maps
            # kernel_size=4, stride=2, padding=1 is a common recipe:
            #   output_size = (input_size + 2*padding - kernel_size) / stride + 1
            #               = (128 + 2 - 4) / 2 + 1 = 64
            # Spatial size: 128 → 64
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),  # negatives → 0, positives unchanged; introduces non-linearity

            # Layer 2: 32 → 64 feature maps, spatial: 64 → 32
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            # Layer 3: 64 → 128 feature maps, spatial: 32 → 16
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            # Layer 4: 128 → 128 feature maps, spatial: 16 → 16 (stride=1, no downsampling)
            # This layer refines features without shrinking the spatial map further.
            # It adds depth (more non-linear processing) cheaply — conv params are small
            # compared to the FC layer. This is what pushes total params to ~9.1M.
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # After 4 layers, tensor shape is still (batch, 128, 16, 16)
        )

        # --- Fully-connected projection ---
        # 128 channels × 16 height × 16 width = 32,768 values per image.
        # We project this flat vector down to latent_dim (256).
        # This layer learns which combinations of spatial features matter most.
        self.fc = nn.Linear(128 * 16 * 16, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of images into latent vectors.

        Args:
            x: raw images, shape (batch, 3, 128, 128)
               values expected in [0, 1] after normalisation

        Returns:
            z: latent vectors, shape (batch, latent_dim)
        """
        # Pass through conv stack.
        # Shape: (batch, 3, 128, 128) → (batch, 128, 16, 16)
        x = self.conv(x)

        # Flatten spatial + channel dimensions into one long vector per sample.
        # x.shape[0] preserves the batch dimension.
        # -1 tells PyTorch to calculate the remaining size automatically:
        #   128 × 16 × 16 = 32,768
        # Shape: (batch, 128, 16, 16) → (batch, 32768)
        x = x.view(x.shape[0], -1)

        # Project to latent space.
        # Shape: (batch, 32768) → (batch, 256)
        return self.fc(x)
