"""
Image preprocessing and augmentation pipeline.

# ---------------------------------------------------------------------------
# WHAT PROBLEM DOES THIS SOLVE?
# ---------------------------------------------------------------------------
# Raw images from the simulator are JPEG-encoded pixels with values 0–255.
# Neural networks train very poorly on raw pixel values because:
#   - Large input values (0–255) cause large activations and unstable gradients
#   - The scale of inputs affects how big gradient updates are
#
# We need to transform raw images into clean tensors the model can learn from.
# This module defines exactly how that transformation happens — consistently
# for every image in every experiment.

# ---------------------------------------------------------------------------
# KEY CONCEPTS YOU WILL LEARN HERE
# ---------------------------------------------------------------------------
# 1. NORMALISATION — dividing pixel values by 255 to scale them to [0, 1].
#    This makes gradient updates stable. Without it, early training often
#    diverges (loss jumps around rather than decreasing steadily).
#
# 2. AUGMENTATION — randomly modifying training images to simulate variety.
#    The simulator always produces clean, well-lit images. In training we
#    artificially vary brightness and add tiny noise so the model doesn't
#    memorise exact pixel values but learns the road structure itself.
#    CRITICAL: augmentation is applied to TRAINING images only, never val.
#
# 3. WHY NOT AUGMENT VALIDATION?
#    Validation exists to measure real performance on real data.
#    If we augmented val images too, our accuracy numbers would be measuring
#    "how well does the model handle noisy images" — not the same question.
#
# 4. WHY NOT FLIP IMAGES HORIZONTALLY?
#    A common augmentation for general image models — but wrong here.
#    Flipping a road image reverses left and right, which would corrupt
#    the action labels (left becomes right). Never flip driving images.
#
# 5. torchvision.transforms — PyTorch's standard image transform library.
#    transforms.Compose([t1, t2, t3]) chains transforms: each output
#    becomes the next transform's input.
#    transforms.ToTensor() converts a PIL Image (H, W, C) uint8
#    into a float tensor (C, H, W) with values in [0, 1] in one step.
#
# 6. PIL Image — the format images are in after base64 decoding.
#    PIL = Python Imaging Library (now maintained as Pillow).
#    ToTensor() knows how to convert it directly to a PyTorch tensor.

# ---------------------------------------------------------------------------
# STRUCTURE
# ---------------------------------------------------------------------------
# get_train_transform() → Compose([ToTensor, ColorJitter, GaussianNoise])
# get_val_transform()   → Compose([ToTensor])
#
# Both return a callable. Call it on a PIL Image to get a tensor:
#   transform = get_train_transform()
#   tensor = transform(pil_image)   # shape: (3, 128, 128), values in [0, 1]

# ---------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------
#   from shared.data.transforms import get_train_transform, get_val_transform
#
#   train_transform = get_train_transform()
#   val_transform   = get_val_transform()
#
#   tensor = train_transform(pil_image)  # (3, 128, 128) float tensor
"""

from __future__ import annotations

# torch is needed for the custom GaussianNoise transform below
import torch

# torchvision.transforms provides standard image transforms used across
# almost all PyTorch vision projects. It is part of torchvision,
# which is installed alongside torch.
from torchvision import transforms


# ---------------------------------------------------------------------------
# Custom transform: Gaussian Noise
# ---------------------------------------------------------------------------
# torchvision doesn't include a gaussian noise transform (it was removed),
# so we write our own. A transform is just a callable class with __call__.
# torchvision's Compose works with any callable — not just built-in transforms.

class GaussianNoise:
    """Adds small random gaussian noise to a tensor image.

    Simulates sensor noise — the camera in the simulator is perfect,
    but real cameras and varied lighting introduce noise. Training with
    noise makes the model slightly more robust.
    """

    def __init__(self, std: float = 0.01) -> None:
        # std controls the strength of the noise.
        # 0.01 is very small — just enough to break pixel-perfect memorisation.
        # Too large (e.g. 0.1) would damage the road structure in the image.
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # torch.randn_like(tensor) creates a tensor of the same shape,
        # filled with values drawn from a normal distribution (mean=0, std=1).
        # Multiplying by self.std scales the noise down to the desired level.
        noise = torch.randn_like(tensor) * self.std

        # clamp(0, 1) ensures pixel values stay in the valid range after adding noise.
        # Without this, noise could push values slightly above 1 or below 0,
        # which would be outside the normalised range the model expects.
        return (tensor + noise).clamp(0, 1)


# ---------------------------------------------------------------------------
# Transform factories
# ---------------------------------------------------------------------------

def get_train_transform() -> transforms.Compose:
    """
    Returns the transform applied to training images.
    Order matters: ToTensor must come first (converts PIL → tensor),
    then augmentations operate on the tensor.
    """
    return transforms.Compose([
        # ToTensor does two things in one step:
        #   1. Converts PIL Image (H, W, C) with uint8 values [0, 255]
        #      to a float32 tensor (C, H, W) — note the axis reorder
        #   2. Divides by 255, scaling values to [0.0, 1.0]
        transforms.ToTensor(),

        # ColorJitter randomly changes brightness.
        # brightness=0.15 means the brightness factor is drawn uniformly
        # from [1-0.15, 1+0.15] = [0.85, 1.15].
        # This simulates the car driving through patches of light and shadow.
        transforms.ColorJitter(brightness=0.15),

        # Our custom noise transform — adds tiny gaussian noise.
        # Applied after ColorJitter so both augmentations are independent.
        GaussianNoise(std=0.01),
    ])


def get_val_transform() -> transforms.Compose:
    """
    Returns the transform applied to validation images.
    No augmentation — validation must reflect real conditions exactly.
    """
    return transforms.Compose([
        # Identical to the train transform's first step:
        # convert PIL Image → normalised float tensor.
        # That's all validation needs.
        transforms.ToTensor(),
    ])
