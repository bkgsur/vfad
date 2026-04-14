"""
VLADataset: loads training-data JSON files for any experiment.

# ---------------------------------------------------------------------------
# WHAT PROBLEM DOES THIS SOLVE?
# ---------------------------------------------------------------------------
# The training data lives in JSON files on disk. Each file contains base64-
# encoded JPEG images and binary action labels. The model cannot consume
# JSON — it needs batches of float tensors delivered efficiently.
#
# This module bridges that gap:
#   JSON files on disk → PIL Images → tensors → batches for the model
#
# It also handles the train/val split so training and evaluation always
# see different samples, measured fairly.

# ---------------------------------------------------------------------------
# KEY CONCEPTS YOU WILL LEARN HERE
# ---------------------------------------------------------------------------
# 1. torch.utils.data.Dataset — the base class for all PyTorch datasets.
#    You must implement two methods:
#      __len__()         returns the total number of samples
#      __getitem__(idx)  returns one (input, label) pair at index idx
#    PyTorch's DataLoader calls these to assemble batches automatically.
#
# 2. torch.utils.data.DataLoader — wraps a Dataset and handles:
#      - Batching: groups N samples into one tensor of shape (N, ...)
#      - Shuffling: randomises order each epoch so the model doesn't
#        memorise sequence patterns
#      - num_workers: loads data in parallel background processes so
#        the GPU is never idle waiting for data
#
# 3. base64 encoding — the simulator stores images as base64 strings
#    inside the JSON. base64 is a way to embed binary data (image bytes)
#    inside a text file. We decode it back to raw bytes, then open as
#    a PIL Image.
#
# 4. EAGER vs LAZY loading:
#      Eager: load ALL images into RAM at startup
#      Lazy:  load each image from disk when __getitem__ is called
#    We use EAGER loading. With ~14,820 images at 128×128×3 ≈ 58 MB
#    as PIL objects — easily fits in RAM. Eager loading means zero
#    disk I/O during training, which is much faster.
#
# 5. WHY store PIL Images, not tensors?
#    The transform (including random augmentation) is applied inside
#    __getitem__ — meaning it runs fresh on every access.
#    Each epoch the same image gets a DIFFERENT random brightness/noise.
#    If we stored pre-converted tensors, augmentation would be fixed
#    and the model would see identical augmented images every epoch.
#
# 6. Random split — we shuffle all indices with a seeded RNG, then
#    take the first 80% as train and the remaining 20% as val.
#    The seed makes the split reproducible: same seed = same split
#    every time you run, so results are comparable across runs.

# ---------------------------------------------------------------------------
# STRUCTURE
# ---------------------------------------------------------------------------
# VLADataset         — Dataset subclass holding PIL images + action tensors
# load_datasets()    — reads JSON files, decodes images, splits train/val,
#                      returns (train_dataset, val_dataset)
# make_dataloaders() — wraps datasets in DataLoaders with the right settings

# ---------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------
#   from shared.data.dataset import load_datasets, make_dataloaders
#   from shared.data.transforms import get_train_transform, get_val_transform
#   from shared.utils.config import load_config
#
#   cfg = load_config("curved_road/exp1/config.yaml")
#   train_ds, val_ds = load_datasets(cfg.data, get_train_transform(), get_val_transform())
#   train_dl, val_dl = make_dataloaders(train_ds, val_ds, cfg.training)
#
#   for images, actions in train_dl:
#       # images: (batch_size, 3, 128, 128)  float32
#       # actions: (batch_size, 4)            float32  [fwd, bwd, left, right]
"""

from __future__ import annotations

import base64
import io
import json

# glob finds all files matching a pattern — used to collect all JSON files
# in the training-data directory
from glob import glob
from pathlib import Path
from typing import Callable

import torch
from PIL import Image

# Dataset is the abstract base class we inherit from.
# DataLoader wraps a dataset and handles batching + shuffling.
from torch.utils.data import DataLoader, Dataset

from shared.utils.config import DataConfig, TrainingConfig


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class VLADataset(Dataset):
    """
    Holds a list of PIL images and their corresponding action labels.
    Applies a transform when a sample is accessed.
    """

    def __init__(
        self,
        images: list[Image.Image],   # list of PIL Images (loaded eagerly)
        actions: torch.Tensor,        # shape (N, 4) float32, binary labels
        transform: Callable,          # transform applied on each __getitem__ call
    ) -> None:
        self.images = images
        self.actions = actions
        self.transform = transform

    def __len__(self) -> int:
        # DataLoader calls this to know how many samples exist.
        # len(self.images) and len(self.actions) are always equal.
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Called by DataLoader once per sample when building a batch.
        #
        # The transform is applied HERE (not at load time) so that random
        # augmentations (brightness, noise) are different every time this
        # sample is accessed — i.e. different each epoch during training.
        image = self.transform(self.images[idx])

        # actions[idx] is a 1D tensor of shape (4,): [fwd, bwd, left, right]
        return image, self.actions[idx]


# ---------------------------------------------------------------------------
# Helper: decode one base64 image string → PIL Image
# ---------------------------------------------------------------------------

def _decode_image(b64_string: str) -> Image.Image:
    """Convert a base64-encoded JPEG string into a PIL Image."""

    # The simulator sometimes prepends a data URI prefix like:
    #   "data:image/jpeg;base64,/9j/4AAQ..."
    # We only want the actual base64 data after the comma.
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]

    # base64.b64decode() converts the base64 string back to raw bytes.
    # io.BytesIO wraps those bytes as a file-like object in memory —
    # no temporary file on disk needed.
    # Image.open() reads the JPEG bytes and returns a PIL Image.
    raw_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(raw_bytes)).convert("RGB")

    # .convert("RGB") ensures the image always has 3 channels.
    # Some JPEGs can be greyscale or have an alpha channel (RGBA).
    # RGB guarantees the shape the model expects: (3, H, W) after ToTensor.


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_datasets(
    cfg: DataConfig,
    train_transform: Callable,
    val_transform: Callable,
) -> tuple[VLADataset, VLADataset]:
    """
    Read all JSON files from cfg.training_data_dir, decode every image,
    extract action labels, split into train/val, and return two datasets.
    """

    # glob("path/*.json") returns a list of all .json file paths in the folder.
    # sorted() makes the file order deterministic — important for reproducibility.
    json_files = sorted(glob(str(Path(cfg.training_data_dir) / "*.json")))

    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found in {cfg.training_data_dir}. "
            "Check that training_data_dir in config.yaml is correct."
        )

    # We collect all samples across all files into two flat lists.
    all_images: list[Image.Image] = []
    all_actions: list[list[float]] = []

    for file_path in json_files:
        data = json.loads(Path(file_path).read_text())

        # Each JSON file has the structure: {"metadata": {...}, "samples": [...]}
        for sample in data["samples"]:
            all_images.append(_decode_image(sample["image"]))

            # Actions are a dict: {"forward": 1, "backward": 0, "left": 1, "right": 0}
            # We always extract them in the same fixed order so the model's
            # output index 0 always means "forward", index 2 always means "left", etc.
            a = sample["actions"]
            all_actions.append([
                float(a["forward"]),
                float(a["backward"]),
                float(a["left"]),
                float(a["right"]),
            ])

    n = len(all_images)
    print(f"Loaded {n} samples from {len(json_files)} files")

    # Stack all action rows into a single tensor of shape (N, 4).
    # float32 is required because BCEWithLogitsLoss expects float targets.
    actions_tensor = torch.tensor(all_actions, dtype=torch.float32)

    # --- Random train/val split ---
    #
    # torch.Generator with a fixed seed makes the shuffle reproducible.
    # Same seed always produces the same train/val split, so you can
    # compare two runs and know they saw exactly the same data.
    generator = torch.Generator().manual_seed(cfg.seed)

    # torch.randperm(n) returns a random permutation of [0, 1, ..., n-1].
    # Using the seeded generator makes this permutation deterministic.
    indices = torch.randperm(n, generator=generator).tolist()

    # Calculate the number of training samples (e.g. 80% of total).
    # int() truncates — e.g. int(0.8 * 14820) = 11856
    split = int(n * (1 - cfg.val_split))

    train_indices = indices[:split]   # first 80%
    val_indices   = indices[split:]   # remaining 20%

    print(f"Train: {len(train_indices)} samples | Val: {len(val_indices)} samples")

    # Build the two datasets.
    # We index into all_images/actions with the split indices to create
    # two separate lists — train and val never share a sample.
    train_dataset = VLADataset(
        images=[all_images[i] for i in train_indices],
        actions=actions_tensor[train_indices],
        transform=train_transform,   # includes augmentation
    )
    val_dataset = VLADataset(
        images=[all_images[i] for i in val_indices],
        actions=actions_tensor[val_indices],
        transform=val_transform,     # normalisation only
    )

    return train_dataset, val_dataset


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloaders(
    train_dataset: VLADataset,
    val_dataset: VLADataset,
    cfg: TrainingConfig,
) -> tuple[DataLoader, DataLoader]:
    """Wrap train and val datasets in DataLoaders."""

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,       # reshuffle training data every epoch
                            # so the model doesn't memorise batch order
        num_workers=2,      # 2 background processes pre-load the next batch
                            # while the GPU is processing the current one
        pin_memory=True,    # keeps loaded batches in pinned (page-locked) RAM
                            # which allows faster transfer to GPU memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size * 2,  # val doesn't need gradients so we can
                                         # fit twice as many samples per batch
        shuffle=False,      # val order doesn't matter — no point shuffling
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader
