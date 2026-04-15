"""
Training callbacks: CheckpointCallback and EarlyStoppingCallback.

# ---------------------------------------------------------------------------
# WHAT PROBLEM DOES THIS SOLVE?
# ---------------------------------------------------------------------------
# A simple training loop runs for N epochs and saves the final weights.
# This has two problems:
#
# Problem 1 — the final model is rarely the best model.
#   Val loss typically falls for a while, then rises again as the model starts
#   memorising the training data (overfitting). The best weights existed at
#   some earlier epoch. If we only save at the end, we throw them away.
#   CheckpointCallback watches val loss every epoch and saves whenever it
#   improves, so we always have the best snapshot on disk.
#
# Problem 2 — training wastes time after learning stops.
#   If val loss hasn't improved for 10 epochs in a row, it almost certainly
#   won't improve later either — we're just burning compute. EarlyStoppingCallback
#   counts how many consecutive epochs have passed without improvement and
#   signals the training loop to stop when that count hits a threshold.

# ---------------------------------------------------------------------------
# KEY CONCEPTS YOU WILL LEARN HERE
# ---------------------------------------------------------------------------
# 1. STATEFUL OBJECTS IN A TRAINING LOOP
#    A callback is a plain Python object that the training loop calls at the
#    end of each epoch. It maintains its own state (best loss seen, patience
#    counter) across those calls. This is the standard way to add optional
#    behaviours to a loop without cluttering the loop itself.
#
# 2. model.state_dict() — how PyTorch saves weights.
#    state_dict() returns an OrderedDict mapping parameter names → tensors.
#    Example:
#      { 'layers.0.weight': tensor(...), 'layers.0.bias': tensor(...), ... }
#    torch.save() serialises this dict to a .pt file.
#    torch.load() + model.load_state_dict() restores it later.
#    We save state_dict(), not the whole model object, because it is
#    smaller, portable, and does not depend on the class definition path.
#
# 3. copy.deepcopy — why we copy before saving in memory.
#    Tensors in state_dict() are views into the model's live parameters.
#    If we stored the dict reference directly, it would mutate as training
#    continues and our "best" snapshot would silently change.
#    deepcopy creates a fully independent clone of every tensor.
#    (torch.save to disk avoids this issue; in-memory best_weights needs it.)
#
# 4. PATIENCE — the early stopping hyperparameter.
#    patience=10 means: stop if val loss hasn't improved for 10 epochs.
#    Too small → stops too soon, underfitting.
#    Too large → wastes time, might as well not use it.
#    Typical values: 5–20, depending on how noisy the val loss curve is.
#
# 5. WHY TRACK val loss, NOT train loss?
#    Train loss always decreases — the model is directly optimised on it.
#    Val loss measures generalisation: how well the model performs on data
#    it has never trained on. That is the loss we actually care about.

# ---------------------------------------------------------------------------
# STRUCTURE
# ---------------------------------------------------------------------------
#   CheckpointCallback
#     __init__(save_path)   where to write best_model.pt
#     on_epoch_end(epoch, val_loss, model)
#       → if val_loss < best_val_loss: save model, update best_val_loss
#
#   EarlyStoppingCallback
#     __init__(patience)    how many epochs without improvement before stopping
#     on_epoch_end(epoch, val_loss)
#       → if improved: reset counter
#       → else: increment counter; set self.should_stop = True when patience hit
#
# The training loop checks early_stopping.should_stop after each epoch
# and breaks out of the loop if True.

# ---------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------
#   from shared.training.callbacks import CheckpointCallback, EarlyStoppingCallback
#   from pathlib import Path
#
#   checkpoint = CheckpointCallback(save_path=Path("results/curved_road/exp1"))
#   early_stop = EarlyStoppingCallback(patience=10)
#
#   for epoch in range(cfg.epochs):
#       train_loss = train_one_epoch(...)
#       val_loss   = validate(...)
#
#       checkpoint.on_epoch_end(epoch, val_loss, model)
#       early_stop.on_epoch_end(epoch, val_loss)
#
#       if early_stop.should_stop:
#           print(f"Early stopping at epoch {epoch}")
#           break
"""

from __future__ import annotations

import copy          # for deepcopy — needed to snapshot tensors safely in memory
from pathlib import Path

import torch         # for torch.save, which serialises state_dict to disk


class CheckpointCallback:
    """Saves the model weights to disk whenever val loss reaches a new minimum.

    Parameters
    ----------
    save_path : Path
        Directory where best_model.pt will be written.
        Created automatically if it does not exist.
    """

    def __init__(self, save_path: Path) -> None:
        # Convert to Path so callers can pass either str or Path.
        self.save_path = Path(save_path)

        # We start with infinity so that ANY real loss in epoch 0 counts
        # as an improvement and triggers the first checkpoint save.
        self.best_val_loss: float = float("inf")

        # Track which epoch produced the best model — useful for logging.
        self.best_epoch: int = -1

    def on_epoch_end(self, epoch: int, val_loss: float, model: torch.nn.Module) -> bool:
        """Called by the training loop at the end of every epoch.

        Parameters
        ----------
        epoch     : current epoch index (0-based)
        val_loss  : validation loss for this epoch
        model     : the model being trained

        Returns
        -------
        bool : True if a checkpoint was saved this epoch, False otherwise.
        """
        if val_loss < self.best_val_loss:
            # New best — record the improvement.
            self.best_val_loss = val_loss
            self.best_epoch = epoch

            # Ensure the output directory exists.
            # parents=True creates any missing parent directories.
            # exist_ok=True is a no-op if the directory already exists.
            self.save_path.mkdir(parents=True, exist_ok=True)

            # state_dict() returns the model's parameters as an OrderedDict.
            # We deepcopy it before saving so that subsequent training steps
            # cannot mutate this snapshot (tensors are normally live views).
            weights = copy.deepcopy(model.state_dict())

            # torch.save serialises the dict to a binary .pt file.
            # To restore: model.load_state_dict(torch.load(path))
            torch.save(weights, self.save_path / "best_model.pt")

            return True  # checkpoint was saved

        return False  # no improvement this epoch


class EarlyStoppingCallback:
    """Stops training when val loss has not improved for `patience` epochs.

    The training loop should check `self.should_stop` after each epoch
    and break if it is True.

    Parameters
    ----------
    patience : int
        Number of consecutive epochs without improvement before stopping.
        Typical range: 5–20.
    min_delta : float
        Minimum change in val loss to count as an improvement.
        Default 0.0 means any decrease, however tiny, resets the counter.
        Increase this (e.g. 1e-4) to ignore negligible fluctuations.
    """

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta

        # Start at inf so the first real val loss always counts as an improvement.
        self.best_val_loss: float = float("inf")

        # How many consecutive epochs have passed without a meaningful improvement.
        self.epochs_without_improvement: int = 0

        # The training loop polls this flag after every epoch.
        # It starts False and flips to True when patience is exhausted.
        self.should_stop: bool = False

    def on_epoch_end(self, epoch: int, val_loss: float) -> None:
        """Called by the training loop at the end of every epoch.

        Parameters
        ----------
        epoch    : current epoch index (0-based), used only for logging
        val_loss : validation loss for this epoch
        """
        # An improvement is defined as: new loss < best loss - min_delta.
        # Subtracting min_delta means tiny improvements (e.g. 0.00001 drop)
        # don't reset the counter unless they exceed the threshold.
        if val_loss < self.best_val_loss - self.min_delta:
            # Genuine improvement — reset the patience counter.
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
        else:
            # No meaningful improvement — increment the stall counter.
            self.epochs_without_improvement += 1

            # If we have stalled for `patience` consecutive epochs, signal stop.
            if self.epochs_without_improvement >= self.patience:
                self.should_stop = True
