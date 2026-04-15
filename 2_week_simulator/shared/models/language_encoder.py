"""
Language encoder used in Exp 5, 8, 9.

Converts a language instruction ID (e.g. 0="turn left", 1="turn right")
into a fixed-size embedding vector that conditions the action prediction.

Input:  (B,) integer language IDs
Output: (B, lang_dim) language embedding vectors

# ---------------------------------------------------------------------------
# WHAT PROBLEM DOES THIS SOLVE?
# ---------------------------------------------------------------------------
# In Exp 4 the car sees a forked road but receives no instruction — it has to
# guess, which means it can't reliably pick the correct branch.
#
# In Exp 5 we add a language instruction: "turn left" or "turn right".
# The model must combine what it SEES (CNN features) with what it's TOLD
# (language embedding) to pick the right path.
#
# This module handles the "told" half: it turns an integer instruction ID
# into a dense vector the model can use.

# ---------------------------------------------------------------------------
# KEY CONCEPTS YOU WILL LEARN HERE
# ---------------------------------------------------------------------------
# 1. WHY NOT ONE-HOT ENCODING?
#    You might think: 2 instructions → encode as [1,0] and [0,1].
#    Problems:
#      a) Scales badly — 1000 instructions → 1000-dim sparse vector.
#      b) All instructions are equally "far" from each other; the encoding
#         carries no semantic information at all.
#      c) The model can't generalise across related instructions.
#
# 2. nn.Embedding — the solution.
#    An Embedding layer is simply a LOOKUP TABLE: a matrix of shape
#    (num_embeddings, embedding_dim) whose rows are learnable parameters.
#
#    When you pass integer ID k, it returns row k of the matrix.
#    That's it — there is no matrix multiplication, just an index lookup.
#
#    The values in that row start random and are updated by backprop,
#    exactly like any other weight in the network.
#
#    Lookup table visualised (num_embeddings=2, embedding_dim=4):
#
#      ID 0 ("turn left")  → [  0.12, -0.45,  0.88,  0.03 ]
#      ID 1 ("turn right") → [ -0.71,  0.99, -0.12,  0.55 ]
#
#    After training, these rows will have drifted to values that make the
#    downstream action predictor work well for each instruction.
#
# 3. How language conditions action prediction (Exp 5, 8, 9).
#    The embedding vector is CONCATENATED with the CNN feature vector
#    before the MLP head:
#
#      image → CNN → (B, cnn_dim)          e.g. (B, 128)
#      instruction ID → Embedding → (B, lang_dim)   e.g. (B,  32)
#      concat → (B, cnn_dim + lang_dim)    e.g. (B, 160)
#      → MLP head → (B, 4) action logits
#
#    The MLP then jointly learns: "given BOTH what I see AND what I'm told,
#    what should I do?" The two streams are blended inside the MLP's weights.
#
# 4. Vocabulary size vs. embedding dim.
#    num_embeddings = how many distinct instructions exist (vocabulary size).
#    embedding_dim  = how many numbers represent each instruction.
#    For this project: 2 instructions, 32-dim embeddings — very small.
#    Real language models (GPT) have 50 000+ tokens and 768–12 288-dim embeddings.
#
# 5. Integer IDs must be torch.long (int64).
#    nn.Embedding uses the integer as an array index, so it requires
#    torch.long dtype. Float IDs will raise a RuntimeError.

# ---------------------------------------------------------------------------
# STRUCTURE
# ---------------------------------------------------------------------------
#   Input:  (B,)          integer instruction IDs, dtype=torch.long
#     → nn.Embedding lookup
#   Output: (B, lang_dim) float embedding vectors, ready to concatenate
#                         with CNN features

# ---------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------
#   from shared.models.language_encoder import LanguageEncoder
#
#   encoder = LanguageEncoder(num_instructions=2, lang_dim=32)
#
#   import torch
#   ids = torch.tensor([0, 1, 0])          # batch of 3 IDs
#   embeddings = encoder(ids)              # → (3, 32)
#
#   # In a full model (Exp 5):
#   cnn_feats = backbone(images)           # (B, 128)
#   lang_feats = encoder(instruction_ids)  # (B, 32)
#   combined = torch.cat([cnn_feats, lang_feats], dim=1)  # (B, 160)
#   logits = mlp_head(combined)            # (B, 4)
"""

from __future__ import annotations

# nn is imported as the standard alias for torch.nn throughout this project.
import torch.nn as nn


class LanguageEncoder(nn.Module):
    """Maps an integer instruction ID to a dense embedding vector.

    Parameters
    ----------
    num_instructions : int
        How many distinct language instructions exist.
        Each one maps to a unique row in the embedding table.
        For this project: 2 (turn left, turn right).
    lang_dim : int
        Dimension of the output embedding vector.
        For this project: 32 — small enough to be cheap, large enough
        to carry useful signal when concatenated with CNN features.
    """

    def __init__(self, num_instructions: int, lang_dim: int) -> None:
        # Always call super().__init__() before anything else.
        # This registers the layer with PyTorch's parameter tracking.
        super().__init__()

        # nn.Embedding(num_embeddings, embedding_dim) creates the lookup table.
        # The table has shape (num_instructions, lang_dim) — one row per instruction.
        # All values are initialised from N(0, 1) and updated during training.
        #
        # There is NO matrix multiplication here: given an integer index k,
        # Embedding simply returns self.weight[k, :] — pure array indexing.
        self.embedding = nn.Embedding(
            num_embeddings=num_instructions,  # rows in the lookup table
            embedding_dim=lang_dim,           # columns — size of each embedding
        )

    def forward(self, instruction_ids):
        # instruction_ids shape: (B,)  dtype: torch.long (int64)
        # Each element is an integer in [0, num_instructions).
        #
        # self.embedding(instruction_ids) looks up one row per element in the batch.
        # Output shape: (B, lang_dim)
        #
        # This is differentiable: gradients flow back into self.embedding.weight,
        # so the lookup table rows are updated by backprop just like Linear weights.
        return self.embedding(instruction_ids)
