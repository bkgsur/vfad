"""
Training entry point for Experiment 1: Vision CNN (3-conv) + MLP.

Run from 2_week_simulator/ directory:
    uv run python -m curved_road.exp1

What this script does:
    1. Load config from config.yaml
    2. Build Dataset and DataLoaders from training-data/
    3. Build the Exp1Model
    4. Train with Trainer (BCEWithLogitsLoss, Adam, CosineAnnealingLR)
    5. Save metrics.json + training curves to results/
    6. Export trained weights to model.json
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from curved_road.exp1.model import Exp1Model
from shared.data.dataset import load_datasets, make_dataloaders
from shared.data.transforms import get_train_transform, get_val_transform
from shared.training.callbacks import CheckpointCallback, EarlyStoppingCallback
from shared.training.trainer import Trainer
from shared.utils.config import load_config
from shared.utils.exporter import export_model
from shared.utils.visualize import plot_curves


def main() -> None:
    # ── 1. Load config ────────────────────────────────────────────────
    # config.yaml lives next to this file (curved_road/exp1/config.yaml).
    # Path(__file__).parent resolves to the exp1/ directory regardless of
    # where the script is invoked from.
    cfg = load_config(Path(__file__).parent / "config.yaml")
    print(f"Starting {cfg.experiment}: {cfg.description}")

    # ── 2. Device ─────────────────────────────────────────────────────
    # Use GPU if available, otherwise CPU.
    # The string "cuda" refers to any NVIDIA GPU. "cpu" is always available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 3. Data ───────────────────────────────────────────────────────
    # training_data_dir in config.yaml is relative to 2_week_simulator/.
    # Path(__file__).parent.parent.parent goes:
    #   exp1/ → curved_road/ → 2_week_simulator/
    # then we append the config path to get the absolute training data path.
    from shared.utils.config import DataConfig, TrainingConfig
    data_cfg_abs = DataConfig(
        training_data_dir=str(
            Path(__file__).parent.parent.parent / cfg.data.training_data_dir
        ),
        val_split=cfg.data.val_split,
        seed=cfg.data.seed,
    )

    # get_train_transform: normalise + random brightness jitter + gaussian noise
    # get_val_transform:   normalise only (no augmentation — we want stable metrics)
    train_dataset, val_dataset = load_datasets(
        cfg=data_cfg_abs,
        train_transform=get_train_transform(),
        val_transform=get_val_transform(),
    )

    train_loader, val_loader = make_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        cfg=cfg.training,
    )

    # ── 4. Model ──────────────────────────────────────────────────────
    model = Exp1Model(
        feature_dim=cfg.model.feature_dim,
        mlp_hidden_dim=cfg.model.mlp_hidden_dim,
        num_actions=cfg.model.num_actions,
    ).to(device)   # move all parameters to device

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # ── 5. Loss, optimiser, scheduler ────────────────────────────────
    # BCEWithLogitsLoss: fuses sigmoid + binary cross-entropy.
    # Multi-label: each of the 4 actions is an independent binary prediction.
    # NEVER apply sigmoid in the model when using this loss.
    criterion = nn.BCEWithLogitsLoss()

    # Adam: adaptive learning rate optimiser — adjusts lr per parameter.
    # weight_decay adds L2 regularisation: loss += wd * sum(w²)
    # This penalises large weights, reducing overfitting.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    # CosineAnnealingLR: decays lr smoothly from cfg.lr → ~0 over T_max epochs.
    # The cosine shape decays slowly at first (big steps while still learning)
    # and very slowly at the end (fine-tuning near convergence).
    # T_max = total epochs = one full cosine half-cycle.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.epochs,
    )

    # ── 6. Callbacks ─────────────────────────────────────────────────
    results_dir = Path(__file__).parent.parent.parent / cfg.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        # Writes best_model.pt to results_dir whenever val loss improves.
        CheckpointCallback(save_path=results_dir),
        # Stops training after 15 epochs without val loss improvement.
        EarlyStoppingCallback(patience=15),
    ]

    # ── 7. Train ──────────────────────────────────────────────────────
    # log_dir: TensorBoard writes event files here.
    # "runs/exp1" keeps each experiment's logs separate.
    # Launch TensorBoard with: tensorboard --logdir runs/
    log_dir = Path(__file__).parent.parent.parent / "runs" / "exp1"

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        callbacks=callbacks,
        device=device,
        log_dir=log_dir,
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.training.epochs,
    )

    # Close the TensorBoard writer to flush all buffered events to disk.
    trainer.close()

    # ── 8. Save metrics.json ──────────────────────────────────────────
    # visualize.plot_curves() reads metrics.json from results_dir.
    # Trainer.history has the per-epoch lists we need.
    metrics_path = results_dir / "metrics.json"
    metrics_path.write_text(json.dumps(history, indent=2))
    print(f"Metrics saved → {metrics_path}")

    # ── 9. Plot training curves ───────────────────────────────────────
    plot_curves(results_dir=results_dir)
    print(f"Training curves saved → {results_dir}/curves.png")

    # ── 10. Load best weights before export ───────────────────────────
    # The CheckpointCallback saved the best weights mid-training.
    # Load them back so the exported model.json has the best model,
    # not necessarily the final-epoch model (which may have overfit slightly).
    best_ckpt = results_dir / "best_model.pt"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        print(f"Loaded best checkpoint from {best_ckpt}")

    # ── 11. Export to model.json ──────────────────────────────────────
    # state_dict() returns all learnable parameters by their PyTorch path names.
    sd = model.state_dict()

    # To find key names, mentally trace the attribute path from the model root:
    #   model.backbone   → CNNBackbone
    #   .features        → the nn.Sequential holding all conv blocks
    #   .0               → index 0 in Sequential = first Conv2d
    #   .weight / .bias  → the parameter
    #
    # Sequential indices for 3-conv + pool + flatten:
    #   0 = Conv2d(3→16)    ← has weight and bias
    #   1 = ReLU            ← no parameters
    #   2 = Conv2d(16→32)   ← has weight and bias
    #   3 = ReLU            ← no parameters
    #   4 = Conv2d(32→64)   ← has weight and bias
    #   5 = ReLU            ← no parameters
    #   6 = AdaptiveAvgPool2d ← no parameters
    #   7 = Flatten         ← no parameters
    #
    # MLPHead Sequential:
    #   0 = Linear(64→64)   ← has weight and bias
    #   1 = ReLU            ← no parameters
    #   2 = Linear(64→4)    ← has weight and bias
    weight_mapping = {
        "conv1_weight": [sd["backbone.features.0.weight"]],   # (16, 3, 3, 3) = 432
        "conv1_bias":   [sd["backbone.features.0.bias"]],     # (16,)
        "conv2_weight": [sd["backbone.features.2.weight"]],   # (32, 16, 3, 3) = 4608
        "conv2_bias":   [sd["backbone.features.2.bias"]],     # (32,)
        "conv3_weight": [sd["backbone.features.4.weight"]],   # (64, 32, 3, 3) = 18432
        "conv3_bias":   [sd["backbone.features.4.bias"]],     # (64,)
        "fc1_weight":   [sd["head.layers.0.weight"]],         # (64, 64) = 4096
        "fc1_bias":     [sd["head.layers.0.bias"]],           # (64,)
        "fc2_weight":   [sd["head.layers.2.weight"]],         # (4, 64) = 256
        "fc2_bias":     [sd["head.layers.2.bias"]],           # (4,)
    }

    model_json_path = Path(__file__).parent / "model.json"
    export_model(
        model=model,
        weight_mapping=weight_mapping,
        model_json_path=model_json_path,
    )

    print("\nDone. Fill in curved_road/exp1/notes.md with your observations.")


if __name__ == "__main__":
    main()
