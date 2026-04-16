"""
Base Trainer used by all 9 experiments.

# ---------------------------------------------------------------------------
# WHAT PROBLEM DOES THIS SOLVE?
# ---------------------------------------------------------------------------
# Every experiment needs the same outer loop:
#   for each epoch:
#       for each batch: forward → loss → backward → update weights
#       validate: forward → loss (no weight update)
#       save best model, maybe stop early, log metrics
#
# Writing this loop from scratch in every experiment would be repetitive and
# error-prone. The Trainer centralises the loop so each experiment only needs
# to provide: a model, a loss function, DataLoaders, and a config.

# ---------------------------------------------------------------------------
# KEY CONCEPTS YOU WILL LEARN HERE
# ---------------------------------------------------------------------------
# 1. THE FOUR-STEP TRAINING UPDATE — every batch, always in this order:
#
#      optimizer.zero_grad()   ← CLEAR old gradients
#      logits = model(x)       ← FORWARD pass (builds computation graph)
#      loss = criterion(logits, labels)
#      loss.backward()         ← BACKWARD pass (fills .grad on every parameter)
#      optimizer.step()        ← UPDATE parameters using .grad
#
#    Why zero_grad() first? PyTorch ACCUMULATES gradients by default.
#    If you skip it, batch N's gradients pile on top of batch N-1's and
#    your updates become wrong. Always clear before each batch.
#
# 2. COMPUTATION GRAPH (autograd)
#    During the forward pass, PyTorch silently records every operation
#    (add, matmul, relu, ...) in a directed acyclic graph.
#    loss.backward() walks this graph in reverse, applying the chain rule
#    to fill each parameter's .grad tensor with ∂loss/∂parameter.
#    torch.no_grad() tells PyTorch NOT to build this graph — used during
#    validation where we only want loss numbers, not gradients.
#    Benefit: smaller memory footprint and faster execution.
#
# 3. DEVICE MANAGEMENT
#    PyTorch tensors live on a specific device: CPU or a GPU (cuda:0, etc.).
#    The model and all input tensors must be on the SAME device —
#    mixing them raises a RuntimeError immediately.
#    Convention:
#      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#      model.to(device)          ← move all model parameters once at start
#      x = x.to(device)          ← move each batch at training time
#
# 4. BCEWithLogitsLoss
#    Our 4 actions (forward/backward/left/right) are each an independent
#    binary decision. BCEWithLogitsLoss fuses sigmoid + binary cross-entropy
#    into one numerically stable operation.
#    Input shape:  (B, 4) raw logits from the model
#    Target shape: (B, 4) float labels — 0.0 or 1.0
#    Output:       scalar loss (mean over all B×4 values)
#
# 5. LEARNING RATE SCHEDULER
#    The learning rate controls how large each weight update is.
#    A fixed learning rate often leads to:
#      - Too large: loss oscillates / diverges
#      - Too small: training is painfully slow
#    A scheduler automatically reduces the lr during training.
#    We use ReduceLROnPlateau: if val loss hasn't improved for `patience`
#    epochs, multiply lr by `factor` (e.g. 0.5 → halve it).
#    This lets us start with a generous lr and fine-tune later.
#
# 6. model.train() vs model.eval()
#    Some layers behave differently during training vs inference:
#      Dropout: randomly zeroes neurons during train, passes all during eval
#      BatchNorm: uses batch statistics during train, running stats during eval
#    We don't use these in the current experiments, but the convention is
#    mandatory — always switch modes correctly. Forgetting model.eval()
#    during validation is a silent bug that causes slightly wrong results.

# ---------------------------------------------------------------------------
# STRUCTURE
# ---------------------------------------------------------------------------
#   Trainer.__init__(model, criterion, optimizer, scheduler, callbacks, device)
#   Trainer.fit(train_loader, val_loader, epochs)
#     → for epoch in range(epochs):
#           train_loss, train_acc = _train_epoch(train_loader)
#           val_loss,   val_acc   = _val_epoch(val_loader)
#           call callbacks
#           log metrics to dict
#           check early stopping → break
#   Trainer._train_epoch(loader) → (loss, accuracy_dict)
#   Trainer._val_epoch(loader)   → (loss, accuracy_dict)

# ---------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------
#   from shared.training.trainer import Trainer
#   from shared.training.callbacks import CheckpointCallback, EarlyStoppingCallback
#   import torch, torch.nn as nn
#
#   device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#   model     = MyModel().to(device)
#   criterion = nn.BCEWithLogitsLoss()
#   optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
#   callbacks = [
#       CheckpointCallback(save_path=Path("results/exp1")),
#       EarlyStoppingCallback(patience=10),
#   ]
#
#   trainer = Trainer(model, criterion, optimizer, scheduler, callbacks, device)
#   history = trainer.fit(train_loader, val_loader, epochs=100)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from shared.training.metrics import compute_metrics


class Trainer:
    """Device-agnostic training loop shared across all 9 experiments.

    Parameters
    ----------
    model       : the nn.Module to train — any architecture works
    criterion   : loss function, e.g. nn.BCEWithLogitsLoss()
    optimizer   : e.g. torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler   : learning rate scheduler; None = constant lr
    callbacks   : list of CheckpointCallback / EarlyStoppingCallback objects
    device      : torch.device("cuda") or torch.device("cpu")
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None,
        callbacks: list,
        device: torch.device,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.callbacks = callbacks
        self.device = device

        # history stores per-epoch metrics so callers can plot curves later.
        # Structure: { "train_loss": [...], "val_loss": [...], "train_acc": [...], ... }
        self.history: dict[str, list[float]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int,
    ) -> dict[str, list[float]]:
        """Run the full training loop.

        Parameters
        ----------
        train_loader : DataLoader for training data
        val_loader   : DataLoader for validation data
        epochs       : maximum number of epochs to run

        Returns
        -------
        dict mapping metric name → list of per-epoch values
        (same as self.history)
        """
        for epoch in range(epochs):
            # ── Train ──────────────────────────────────────────────────
            # model.train() enables dropout / batchnorm training behaviour.
            # Even though our models don't use them yet, this is mandatory
            # convention — always switch correctly.
            train_loss, train_acc = self._train_epoch(train_loader)

            # ── Validate ───────────────────────────────────────────────
            # model.eval() disables dropout / batchnorm training behaviour.
            val_loss, val_acc = self._val_epoch(val_loader)

            # ── Scheduler ─────────────────────────────────────────────
            # ReduceLROnPlateau needs the current val loss to decide
            # whether to reduce the learning rate.
            # We step it after validation, not after training, because
            # we want to react to generalisation loss, not training loss.
            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            # ── Log metrics ────────────────────────────────────────────
            self._log(epoch, train_loss, val_loss, train_acc, val_acc)

            # ── Callbacks ─────────────────────────────────────────────
            # Each callback receives the epoch index, val loss, and model.
            # Callbacks are called in order; early stopping checks should
            # come last so checkpointing always runs first.
            should_stop = self._run_callbacks(epoch, val_loss)

            if should_stop:
                # One or more callbacks signalled that training should stop.
                print(f"  Early stopping triggered at epoch {epoch + 1}.")
                break

        return self.history

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_epoch(
        self, loader: torch.utils.data.DataLoader
    ) -> tuple[float, dict[str, float]]:
        """Run one full pass over the training data.

        Returns
        -------
        (mean_loss, accuracy_dict)
            accuracy_dict has keys: forward, backward, left, right, aggregate
        """
        # Switch model to training mode.
        # This matters for Dropout (randomly zeros activations) and
        # BatchNorm (uses batch statistics). We don't use these yet,
        # but the call is mandatory for correctness in general.
        self.model.train()

        total_loss = 0.0
        # Accumulate raw predictions and labels across all batches so we
        # can compute per-action accuracy over the whole epoch at once.
        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        for batch in loader:
            # Each batch is a (images, actions) tuple collated by DataLoader.
            labels = batch[1].to(self.device)   # (B, 4) float 0/1

            # ── Step 1: zero_grad ─────────────────────────────────────
            # PyTorch accumulates gradients across calls to .backward().
            # If we don't clear them here, the gradients from the PREVIOUS
            # batch add to the current batch's gradients — wrong updates.
            self.optimizer.zero_grad()

            # ── Step 2: forward pass ──────────────────────────────────
            # Pass inputs through the model. PyTorch builds the computation
            # graph silently as each operation runs.
            # logits shape: (B, 4) — raw scores, no sigmoid yet
            logits = self._forward(batch)              # (B, 4)

            # ── Step 3: compute loss ──────────────────────────────────
            # BCEWithLogitsLoss applies sigmoid internally then computes
            # binary cross-entropy for each of the 4 actions independently.
            # It returns a scalar: the mean loss over all B×4 values.
            loss = self.criterion(logits, labels)      # scalar

            # ── Step 4: backward pass ─────────────────────────────────
            # Walk the computation graph backwards.
            # Every parameter's .grad tensor is filled with ∂loss/∂parameter.
            loss.backward()

            # ── Step 5: optimizer step ────────────────────────────────
            # Read each parameter's .grad and update the parameter:
            #   Adam:  param -= lr × (bias-corrected moment estimate)
            #   SGD:   param -= lr × grad
            self.optimizer.step()

            # Accumulate loss for epoch-level averaging.
            # .item() extracts the Python float from a scalar tensor.
            total_loss += loss.item()

            # Detach from the computation graph before storing.
            # We only need the values for accuracy — no need to keep
            # the graph alive in memory.
            all_logits.append(logits.detach())
            all_labels.append(labels.detach())

        mean_loss = total_loss / len(loader)

        # Stack all batches into a single tensor for accuracy computation.
        # torch.cat along dim=0 stacks (B1,4), (B2,4), ... → (N, 4)
        epoch_logits = torch.cat(all_logits, dim=0)   # (N, 4)
        epoch_labels = torch.cat(all_labels, dim=0)   # (N, 4)

        accuracy = compute_metrics(epoch_logits, epoch_labels)

        return mean_loss, accuracy

    def _val_epoch(
        self, loader: torch.utils.data.DataLoader
    ) -> tuple[float, dict[str, float]]:
        """Run one full pass over the validation data (no weight updates).

        Returns
        -------
        (mean_loss, accuracy_dict)
        """
        # Switch model to eval mode.
        self.model.eval()

        total_loss = 0.0
        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        # torch.no_grad() tells PyTorch not to build a computation graph.
        # We don't call .backward() during validation, so the graph is
        # wasteful. no_grad saves memory and makes validation faster.
        with torch.no_grad():
            for batch in loader:
                labels = batch[1].to(self.device)   # (B, 4)

                logits = self._forward(batch)        # (B, 4)
                loss = self.criterion(logits, labels)  # scalar

                total_loss += loss.item()
                all_logits.append(logits)
                all_labels.append(labels)

        mean_loss = total_loss / len(loader)

        epoch_logits = torch.cat(all_logits, dim=0)   # (N, 4)
        epoch_labels = torch.cat(all_labels, dim=0)   # (N, 4)

        accuracy = compute_metrics(epoch_logits, epoch_labels)

        return mean_loss, accuracy

    def _forward(self, batch: tuple) -> torch.Tensor:
        """Route a batch through the model.

        batch is a (images, labels) tuple from the DataLoader.
        Language experiments (Exp 5, 8, 9) extend this to a 3-tuple
        (images, labels, instruction_ids) — the model handles both cases.

        Returns
        -------
        logits : (B, 4) raw action logits
        """
        images = batch[0].to(self.device)   # (B, 3, 128, 128)

        if len(batch) == 3:
            # Language experiments: (images, labels, instruction_ids)
            instruction_ids = batch[2].to(self.device)   # (B,)
            return self.model(images, instruction_ids)   # (B, 4)
        else:
            # Vision-only experiments (Exp 1, 2, 3, 4, 6, 7)
            return self.model(images)                    # (B, 4)

    def _run_callbacks(self, epoch: int, val_loss: float) -> bool:
        """Call all callbacks and return True if any requests early stopping."""
        stop = False
        for cb in self.callbacks:
            # CheckpointCallback has on_epoch_end(epoch, val_loss, model)
            # EarlyStoppingCallback has on_epoch_end(epoch, val_loss)
            # We detect which by checking the attribute that early stopping has.
            if hasattr(cb, "should_stop"):
                # EarlyStoppingCallback
                cb.on_epoch_end(epoch, val_loss)
                if cb.should_stop:
                    stop = True
            else:
                # CheckpointCallback (or any future callback with model)
                cb.on_epoch_end(epoch, val_loss, self.model)
        return stop

    def _log(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_acc: dict[str, float],
        val_acc: dict[str, float],
    ) -> None:
        """Append this epoch's metrics to self.history and print a summary."""
        # First epoch: initialise the history lists.
        if not self.history:
            self.history = {
                "train_loss": [],
                "val_loss": [],
                "train_acc": [],
                "val_acc": [],
            }

        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        # Store aggregate accuracy as the headline number.
        self.history["train_acc"].append(train_acc["acc"])
        self.history["val_acc"].append(val_acc["acc"])

        # Print one line per epoch so the user can watch training progress.
        # Format: epoch | train_loss | val_loss | train_acc | val_acc
        print(
            f"Epoch {epoch + 1:>4d} | "
            f"train_loss {train_loss:.4f} | "
            f"val_loss {val_loss:.4f} | "
            f"train_acc {train_acc['acc']:.3f} | "
            f"val_acc {val_acc['acc']:.3f}"
        )
