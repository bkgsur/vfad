"""
Action Encoder — action_encoder.py
====================================

WHAT PROBLEM DOES THIS SOLVE?
    The world model needs to understand *what the agent did* between two frames.
    Raw action values (v_x, v_y, ω) are just 3 numbers — very low dimensional.
    But their *combined meaning* is non-linear:
        "high v_x + high ω" means a fast sweeping turn
        "zero v_x + high ω" means spinning in place
    A single linear layer can't capture these combinations.
    The ActionEncoder uses a two-layer MLP to project the raw action into a
    64-dim embedding that the Transition Model can combine with the image latent.

KEY CONCEPTS YOU WILL LEARN HERE
    - MLP (Multi-Layer Perceptron): stacked Linear + activation layers
    - Why two layers: one layer learns individual features, the second learns
      combinations of those features — essential for non-linear relationships
    - Why no ReLU after the final layer: the output flows directly into the
      Transition Model, which should receive unclamped values
    - Parameter count: ~4K total (tiny compared to the 9.1M encoder)

INPUTS
    a[0] = v_x — linear velocity in X direction (forward / backward)
    a[1] = v_y — linear velocity in Y direction (lateral / sideways)
    a[2] = ω   — rotational velocity (yaw rate — how fast the agent turns)

STRUCTURE
    Input:   (batch, 3)   — raw action vector [v_x, v_y, ω]
    Linear1: (batch, 64)  — first projection, learns individual features
    ReLU:    (batch, 64)  — non-linearity; allows combinations in next layer
    Linear2: (batch, 64)  — second projection, learns feature combinations
    Output:  (batch, 64)  — action embedding (no activation — unclamped)

PARAMETER COUNT  (~4K total)
    Formula for a Linear layer:
        params = (in_features × out_features) + out_features
                  ↑ weights                     ↑ bias (one per output neuron)

    Layer-by-layer breakdown:
        Linear1:  (3  × 64) + 64  =   192 + 64  =    256
        Linear2:  (64 × 64) + 64  = 4,096 + 64  =  4,160
                                     ────────────────────
        Total:                                    =  4,416  (~4K)

    Compare this to the ObsEncoder FC layer alone: 8,388,864 params.
    The action vector carries so little information (just 3 numbers)
    that the entire ActionEncoder is ~2000× smaller than one FC layer
    in the image encoder. Actions are cheap to encode.

USAGE
    action_encoder = ActionEncoder(action_dim=3, embed_dim=64)
    a_emb = action_encoder(actions)   # actions: (B, 3) → a_emb: (B, 64)
"""

import torch
import torch.nn as nn


class ActionEncoder(nn.Module):
    """
    Two-layer MLP that maps a raw action vector to a fixed-size embedding.

    The embedding is designed to be concatenated with the image latent z_t
    inside the Transition Model, so its size (64) is intentionally smaller
    than the image latent (256) — actions carry far less information than images.
    """

    def __init__(self, action_dim: int = 3, embed_dim: int = 64) -> None:
        """
        Args:
            action_dim: number of action components — 3 for (v_x, v_y, ω)
            embed_dim:  size of the output embedding — 64 to match PARAMS.png
        """
        super().__init__()

        # Two-layer MLP wrapped in nn.Sequential so forward() is one call.
        # Layer 1: projects the 3 raw values into a 64-dim space.
        #          Each of the 64 neurons learns a weighted combination of
        #          v_x, v_y, and ω — picking up on individual signals.
        # ReLU:    kills negative activations, forcing the next layer to only
        #          build on positive evidence. This is what makes the network
        #          non-linear — without it, Linear + Linear = one Linear.
        # Layer 2: projects the 64 intermediate features into the final embed_dim.
        #          Each output neuron now learns combinations of the combinations
        #          from layer 1 — e.g. "fast forward AND turning" as one unit.
        # No final activation: the embedding goes straight into the Transition
        #          Model's input. Clamping with ReLU here would destroy negative
        #          directional signals (e.g. v_y < 0 means moving left).
        self.net = nn.Sequential(
            nn.Linear(action_dim, 64),   # (batch, 3)  → (batch, 64)
            nn.ReLU(),                   # non-linearity between layers
            nn.Linear(64, embed_dim),    # (batch, 64) → (batch, embed_dim)
        )

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of action vectors into embeddings.

        Args:
            a: action vectors, shape (batch, 3)
               a[:, 0] = v_x  — forward/backward velocity
               a[:, 1] = v_y  — lateral velocity
               a[:, 2] = ω    — rotational velocity (yaw rate)

        Returns:
            embedding of shape (batch, embed_dim) — ready to be concatenated
            with the image latent z_t in the Transition Model
        """
        # Pass through both Linear layers and the ReLU between them.
        # Shape: (batch, 3) → (batch, 64)
        return self.net(a)
