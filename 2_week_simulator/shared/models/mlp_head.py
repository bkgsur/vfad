"""
MLP action head: maps a feature vector to action logits.

# ---------------------------------------------------------------------------
# WHAT PROBLEM DOES THIS SOLVE?
# ---------------------------------------------------------------------------
# The CNN backbone takes an image and produces a compact feature vector
# (e.g. 64 numbers). That vector captures WHAT the image looks like —
# "there is a left curve ahead" — but doesn't yet produce actions.
#
# The MLP head takes that feature vector and maps it to 4 action logits:
# [forward, backward, left, right]. It is the decision-making layer
# that answers: "given what I see, what should I do?"

# ---------------------------------------------------------------------------
# KEY CONCEPTS YOU WILL LEARN HERE
# ---------------------------------------------------------------------------
# 1. nn.Module — the base class for EVERY neural network in PyTorch.
#    All models, layers, and building blocks inherit from it.
#    It provides parameter tracking, .to(device), .train()/.eval(), etc.
#    You must implement two methods:
#      __init__()       define the layers (called once at creation)
#      forward(x)       define how data flows through them (called every batch)
#
# 2. nn.Linear(in, out) — a fully connected layer.
#    Internally it holds a weight matrix W of shape (out, in) and a
#    bias vector b of shape (out,).
#    forward: output = input @ W.T + b
#    This is just a matrix multiplication — the fundamental operation
#    in all neural networks.
#
# 3. nn.ReLU() — Rectified Linear Unit. The most common activation function.
#    formula: ReLU(x) = max(0, x)
#    It sets all negative values to zero and keeps positive ones unchanged.
#    WHY do we need it? Without activations, stacking Linear layers is
#    mathematically equivalent to ONE linear layer — no matter how many
#    layers you add. Activations introduce non-linearity so the network
#    can learn curved decision boundaries.
#
# 4. nn.Sequential — chains layers into a pipeline.
#    Sequential([layer1, layer2, layer3])(x) is identical to:
#      x = layer1(x)
#      x = layer2(x)
#      x = layer3(x)
#    It's just cleaner syntax for simple feed-forward paths.
#
# 5. WHY output raw logits, not probabilities?
#    We use BCEWithLogitsLoss which internally applies sigmoid + BCE.
#    Doing sigmoid inside the loss is numerically more stable than
#    calling sigmoid first and then computing BCE separately.
#    Rule: the model outputs raw logits; loss and metrics apply sigmoid.

# ---------------------------------------------------------------------------
# STRUCTURE
# ---------------------------------------------------------------------------
#   Input:  (B, input_dim)     feature vector, B = batch size
#     → Linear(input_dim, hidden_dim)
#     → ReLU
#     → Linear(hidden_dim, num_actions)
#   Output: (B, num_actions)   raw logits — one per action
#
# input_dim is flexible because later experiments concatenate extra features:
#   Exp 1:   CNN output only          → input_dim = 64
#   Exp 4:   4-conv CNN output        → input_dim = 128
#   Exp 5:   CNN + language embed     → input_dim = 128 + 32 = 160

# ---------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------
#   from shared.models.mlp_head import MLPHead
#   head = MLPHead(input_dim=64, hidden_dim=64, num_actions=4)
#   logits = head(features)   # features: (B, 64) → logits: (B, 4)
"""

from __future__ import annotations

# nn is the neural network module — contains all layer types (Linear, Conv2d, etc.)
# and the Module base class. Convention: always imported as nn.
import torch.nn as nn


class MLPHead(nn.Module):
    """Two-layer MLP that maps a feature vector to action logits."""

    def __init__(self, input_dim: int, hidden_dim: int, num_actions: int) -> None:
        # ALWAYS call super().__init__() first in nn.Module subclasses.
        # This initialises PyTorch's internal parameter tracking machinery.
        # Without it, none of the parameters (weights, biases) would be
        # registered and the model would not train correctly.
        super().__init__()

        # nn.Sequential chains these three operations into one callable.
        # Data will flow left to right through them in forward().
        self.layers = nn.Sequential(
            # Layer 1: project input_dim → hidden_dim
            # e.g. 64 input features → 64 hidden features
            # This layer has input_dim × hidden_dim weights + hidden_dim biases
            nn.Linear(input_dim, hidden_dim),

            # ReLU non-linearity between the two linear layers.
            # Without this, two Linear layers would collapse into one —
            # the whole point of depth is the non-linearity between layers.
            nn.ReLU(),

            # Layer 2: project hidden_dim → num_actions
            # e.g. 64 hidden features → 4 action logits
            # Output has NO activation — raw logits are what BCEWithLogitsLoss needs.
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x):
        # x shape: (B, input_dim)  where B is the batch size
        # self.layers(x) passes x through Linear → ReLU → Linear in sequence
        # output shape: (B, num_actions)
        return self.layers(x)
