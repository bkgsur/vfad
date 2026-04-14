"""
Evaluation metrics for action prediction.

# ---------------------------------------------------------------------------
# WHAT PROBLEM DOES THIS SOLVE?
# ---------------------------------------------------------------------------
# The loss function (BCEWithLogitsLoss) tells us how wrong the model is
# in mathematical terms — a single number like 0.34. That's useful for
# training but hard to interpret as a human.
#
# Accuracy is easier to understand: "the model predicted the correct
# action 89% of the time." But for this problem, a single accuracy
# number hides an important truth about class imbalance:
#
#   forward  is 1 in nearly every sample → trivially predicted correctly
#   backward is 0 in every sample        → trivially predicted correctly
#   left     is 1 in ~42% of samples     → actually needs to be learned
#   right    is 1 in ~10% of samples     → hardest to learn
#
# A model that always predicts forward=1, backward=0, left=0, right=0
# would score ~75% aggregate accuracy while being completely useless
# for steering. Per-action accuracy exposes this — left and right would
# show ~58% and ~90% respectively, clearly below a useful threshold.
#
# This module computes both so we can see the full picture.

# ---------------------------------------------------------------------------
# KEY CONCEPTS YOU WILL LEARN HERE
# ---------------------------------------------------------------------------
# 1. sigmoid — converts a raw logit into a probability between 0 and 1.
#    sigmoid(x) = 1 / (1 + e^(-x))
#    logit=0    → 0.5   (model is uncertain)
#    logit=+2   → 0.88  (model leans strongly toward 1)
#    logit=-2   → 0.12  (model leans strongly toward 0)
#
# 2. Threshold — we convert probabilities to binary predictions.
#    probability > 0.5  →  predicted label = 1
#    probability ≤ 0.5  →  predicted label = 0
#    0.5 is the natural midpoint. We could tune this threshold later
#    if we wanted to favour recall over precision for specific actions.
#
# 3. Element-wise comparison — (preds == targets) produces a boolean
#    tensor of the same shape. True where prediction matches label.
#    .float() converts True→1.0, False→0.0 so we can average it.
#
# 4. .mean(dim=0) vs .mean() — shape matters:
#    logits shape: (B, 4)   B=batch size, 4=actions
#    .mean(dim=0) averages across the batch dimension → shape (4,)
#    giving one accuracy number per action across all samples.
#    .mean() averages everything → one scalar for aggregate accuracy.

# ---------------------------------------------------------------------------
# STRUCTURE
# ---------------------------------------------------------------------------
# compute_metrics(logits, targets) → dict with keys:
#   "acc"          float  aggregate accuracy across all 4 actions
#   "acc_forward"  float  accuracy for forward action only
#   "acc_backward" float
#   "acc_left"     float
#   "acc_right"    float

# ---------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------
#   from shared.training.metrics import compute_metrics
#
#   logits  = model(images)           # (B, 4) raw logits
#   metrics = compute_metrics(logits, actions)
#   print(metrics["acc_left"])        # e.g. 0.813
"""

from __future__ import annotations

import torch

# ACTION_NAMES defines the fixed order of actions in the output tensor.
# Index 0 = forward, 1 = backward, 2 = left, 3 = right.
# Defined once here so every part of the codebase uses the same order.
ACTION_NAMES = ["forward", "backward", "left", "right"]


def compute_metrics(
    logits: torch.Tensor,   # shape (B, 4), raw model outputs — no sigmoid yet
    targets: torch.Tensor,  # shape (B, 4), binary labels 0.0 or 1.0
) -> dict[str, float]:
    """
    Compute per-action and aggregate accuracy.
    Returns a dict of plain Python floats (not tensors) for easy logging.
    """

    # We never want gradients during metric computation — it would waste
    # memory and time. torch.no_grad() is a context manager that disables
    # gradient tracking for everything inside the block.
    with torch.no_grad():

        # Step 1: logits → probabilities
        # sigmoid maps any real number to (0, 1).
        # Shape stays (B, 4).
        probs = torch.sigmoid(logits)

        # Step 2: probabilities → binary predictions
        # (probs > 0.5) returns a boolean tensor: True where prob > 0.5.
        # .float() converts True→1.0, False→0.0.
        # Shape stays (B, 4).
        preds = (probs > 0.5).float()

        # Step 3: compare predictions to ground-truth targets
        # (preds == targets) is True wherever prediction matches label.
        # .float() again converts booleans to 0.0/1.0.
        # Shape stays (B, 4).
        correct = (preds == targets).float()

        # Step 4: per-action accuracy
        # .mean(dim=0) averages across the batch (dim=0), keeping the 4 actions.
        # Result shape: (4,) — one accuracy value per action.
        # e.g. tensor([0.99, 1.00, 0.83, 0.91])
        per_action = correct.mean(dim=0)

        # Step 5: aggregate accuracy
        # .mean() averages across everything — all samples, all actions.
        # Result: a single scalar tensor.
        aggregate = correct.mean()

    # Build the result dict.
    # .item() converts a single-element tensor to a plain Python float.
    # We return floats (not tensors) because they're easier to log, print,
    # and store in metrics.json without serialisation issues.
    metrics: dict[str, float] = {"acc": aggregate.item()}

    for i, name in enumerate(ACTION_NAMES):
        # per_action[i] is the accuracy for action i (e.g. per_action[2] = left)
        metrics[f"acc_{name}"] = per_action[i].item()

    return metrics
