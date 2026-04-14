"""
Image preprocessing and augmentation pipeline.

Responsibilities:
- Resize to target dimensions (128×128)
- Normalise pixel values to [0, 1]
- Training augmentations: brightness jitter, gaussian noise
- Validation transform: normalise only (no augmentation)
"""
