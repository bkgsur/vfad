"""
Config loader: reads an experiment's config.yaml and returns a frozen dataclass.

# ---------------------------------------------------------------------------
# WHAT PROBLEM DOES THIS SOLVE?
# ---------------------------------------------------------------------------
# Every experiment has settings like lr, epochs, batch_size.
# Without a config loader you have two bad options:
#   1. Hardcode values in training code → must edit code to change them
#   2. Pass them all as command-line args → tedious and error-prone
#
# A config loader lets you change an experiment by editing only config.yaml.
# The code never changes — only the config does.

# ---------------------------------------------------------------------------
# KEY CONCEPTS YOU WILL LEARN HERE
# ---------------------------------------------------------------------------
# 1. YAML — a human-friendly format for structured data. Python's pyyaml
#    library reads a .yaml file into a plain Python dict.
#
# 2. Frozen dataclass — a Python class that holds data with typed fields
#    that CANNOT be changed after creation (frozen = immutable).
#    Instead of cfg["training"]["lr"] (fragile dict access), you get
#    cfg.training.lr (typed, autocompleted, crashes on typos).
#
# 3. WHY FROZEN? If config could be mutated mid-training, a bug could
#    silently change your learning rate halfway through a run. Frozen
#    makes config a contract: set once at startup, never touched again.

# ---------------------------------------------------------------------------
# STRUCTURE
# ---------------------------------------------------------------------------
# The config.yaml has 4 sections. Each becomes its own frozen dataclass:
#
#   config.yaml          →    Python
#   ─────────────────────────────────────────
#   data:                →    DataConfig
#   model:               →    ModelConfig
#   training:            →    TrainingConfig
#   (top level)          →    ExperimentConfig  (contains the 3 above)

# ---------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------
#   cfg = load_config("curved_road/exp1/config.yaml")
#   cfg.training.lr       # float
#   cfg.training.epochs   # int
#   cfg.data.val_split    # float
"""

# Allow writing type hints like `str | Path` without errors on Python 3.11.
# Safe to always include at the top of every file.
from __future__ import annotations

# dataclass: a decorator that auto-generates __init__, __repr__, __eq__ for a class.
# You define the fields; Python writes the boilerplate for you.
from dataclasses import dataclass

# Path is Python's modern way to work with file paths.
# Path("foo/bar.yaml").read_text() reads the whole file in one call — cleaner than open().
from pathlib import Path

# pyyaml: reads a .yaml file into a plain Python dict.
# We always use safe_load (not load) — load can execute arbitrary Python embedded
# in a YAML file, which is a security risk. safe_load only produces dicts/lists/strings.
import yaml


# ---------------------------------------------------------------------------
# Section dataclasses
#
# Each dataclass maps to one top-level section in config.yaml.
#
# frozen=True makes every field READ-ONLY after the object is created.
# If you try cfg.training.lr = 0.1 later in the code, Python raises FrozenInstanceError.
# This is intentional — config is a contract set once at startup, never mutated.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataConfig:
    training_data_dir: str   # path to the folder containing training JSON files
    val_split: float         # fraction of samples held out for validation (e.g. 0.2 = 20%)
    seed: int                # random seed so the train/val split is reproducible


@dataclass(frozen=True)
class ModelConfig:
    cnn_variant: str         # "3conv" for curved road, "4conv" for forked road
    feature_dim: int         # number of values output by the CNN backbone (e.g. 64)
    num_actions: int         # number of action outputs — always 4 (fwd/bwd/left/right)

    # The fields below have DEFAULT VALUES.
    # This means they are optional in config.yaml — if a key is absent, the default is used.
    # Exp 1 doesn't need lang_dim or transformer settings, so they just fall back to defaults.
    mlp_hidden_dim: int = 64             # hidden size inside the MLP action head
    lang_dim: int = 32                   # language embedding vector size (used in Exp 5, 8, 9)
    transformer_d_model: int = 64        # internal width of the transformer (Exp 2, 3, 6–9)
    transformer_nhead: int = 4           # number of parallel attention heads in transformer
    transformer_num_layers: int = 2      # how many encoder + decoder layers in the transformer
    chunk_size: int = 10                 # how many future actions the transformer predicts at once
    latent_dim: int = 32                 # size of the CVAE latent vector z (Exp 3, 7, 9)
    cvae_beta: float = 1.0               # weight on KL divergence in CVAE loss (loss = BCE + β·KL)
    num_intents: int = 1                 # number of language instructions: 1 curved, 2 forked


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int              # total number of passes through the training data
    batch_size: int          # how many samples are processed together in one gradient update
    lr: float                # learning rate — how large each gradient step is
    weight_decay: float      # L2 regularisation: penalises large weights to prevent overfitting
    scheduler: str           # how lr changes over training: "cosine" decays smoothly to zero


@dataclass(frozen=True)
class ExperimentConfig:
    # Top-level fields from config.yaml
    experiment: str          # experiment ID, e.g. "exp1"
    description: str         # human-readable label shown in logs
    results_dir: str         # folder where checkpoints and plots are saved

    # Nested configs — each is a frozen dataclass of its own
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> ExperimentConfig:
    """Read a config.yaml file and return a fully typed, immutable config."""

    # Path(path).read_text() opens the file and returns its contents as a string.
    # yaml.safe_load() parses that string into a nested Python dict.
    raw = yaml.safe_load(Path(path).read_text())

    # raw["data"] is a plain dict: {"training_data_dir": "...", "val_split": 0.2, "seed": 42}
    # The ** operator unpacks a dict as keyword arguments.
    # So DataConfig(**raw["data"]) becomes DataConfig(training_data_dir=..., val_split=..., seed=...)
    return ExperimentConfig(
        experiment=raw["experiment"],
        description=raw["description"],
        results_dir=raw["results_dir"],
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
    )
