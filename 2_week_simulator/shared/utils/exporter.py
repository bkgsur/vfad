"""
Exports trained model weights to the model.json format the simulator reads.

# ---------------------------------------------------------------------------
# WHAT PROBLEM DOES THIS SOLVE?
# ---------------------------------------------------------------------------
# After training, weights live inside a PyTorch model as tensors with
# PyTorch-specific names like "backbone.conv_blocks.0.0.weight".
#
# The simulator cannot load PyTorch files — it reads model.json, which has
# a fixed key schema and flat float arrays. The exporter bridges this gap:
#
#   PyTorch state_dict                model.json
#   ────────────────────              ──────────
#   backbone.conv_blocks.0.0.weight → conv1_weight: [0.12, -0.45, ...]
#   backbone.conv_blocks.0.0.bias   → conv1_bias:   [0.03, ...]
#   (all transformer enc tensors)   → transformer_enc_weight: [...]
#   (all transformer enc biases)    → transformer_enc_bias:   [...]
#
# KEY DESIGN DECISION:
# The exporter itself is architecture-agnostic. It knows nothing about
# specific layer names. Instead, each experiment provides a `weight_mapping`
# dict that says: "for model.json key X, use these tensors from state_dict."
# The exporter just flattens and writes.

# ---------------------------------------------------------------------------
# KEY CONCEPTS YOU WILL LEARN HERE
# ---------------------------------------------------------------------------
# 1. model.state_dict()
#    Returns an OrderedDict mapping parameter name → tensor for every
#    learnable parameter in the model.
#    Example (Exp 1 model):
#      {
#        'backbone.conv_blocks.0.0.weight': tensor(16, 3, 3, 3),
#        'backbone.conv_blocks.0.0.bias':   tensor(16,),
#        'head.layers.0.weight':            tensor(64, 64),
#        'head.layers.0.bias':              tensor(64,),
#        'head.layers.2.weight':            tensor(4, 64),
#        'head.layers.2.bias':              tensor(4,),
#      }
#    This is the standard PyTorch format for saving/loading weights.
#    It is portable: the keys depend on how you named your nn.Module attributes,
#    NOT on the Python class path — so you can load weights into a differently-
#    named module as long as the shapes match.
#
# 2. tensor.detach().cpu().numpy().flatten().tolist()
#    This chain converts a tensor to a Python list of floats:
#      .detach()  — detach from the autograd computation graph
#                   (required if the tensor still has grad_fn)
#      .cpu()     — move to CPU (tensors on GPU can't be converted to numpy directly)
#      .numpy()   — convert to a NumPy array (zero-copy when on CPU)
#      .flatten() — collapse all dimensions to a 1-D array
#      .tolist()  — convert to a Python list of native floats
#    JSON serialisation requires Python native types, not numpy floats.
#
# 3. Why flat arrays?
#    The simulator is written in JavaScript/TypeScript and reads model.json
#    directly in the browser. Flat arrays are the simplest, most portable
#    format — no nested structure, no dtype metadata. The simulator knows
#    the expected shapes from the format field and reconstructs tensors itself.
#
# 4. The template contract
#    Each experiment ships a pre-built model.json template with:
#      - The correct format string (e.g. "vision_only_vla")
#      - The correct weight keys (e.g. "conv1_weight", "fc1_bias")
#      - Empty arrays [] as placeholder values
#    The exporter reads this template, fills in the arrays, and writes it back.
#    This ensures the exporter can NEVER produce a file with wrong keys —
#    it can only fill existing keys, not invent new ones.
#
# 5. num_params
#    Total number of scalar parameters across all weight arrays.
#    Computed by summing the length of every filled array.
#    Useful for sanity-checking that the export captured all parameters.

# ---------------------------------------------------------------------------
# STRUCTURE
# ---------------------------------------------------------------------------
#   export_model(model, weight_mapping, model_json_path)
#     → reads template from model_json_path
#     → for each key in template["weights"]:
#           flatten tensors listed in weight_mapping[key]
#     → counts total params
#     → writes updated JSON back to model_json_path
#
#   _flatten_tensors(tensors) → list[float]
#     → concatenates multiple tensors into one flat list of Python floats

# ---------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------
#   from shared.utils.exporter import export_model
#   from pathlib import Path
#
#   # In each experiment's __main__.py, after training:
#   sd = model.state_dict()   # get all parameter tensors by name
#
#   weight_mapping = {
#       "conv1_weight": [sd["backbone.conv_blocks.0.0.weight"]],
#       "conv1_bias":   [sd["backbone.conv_blocks.0.0.bias"]],
#       "fc1_weight":   [sd["head.layers.0.weight"]],
#       "fc1_bias":     [sd["head.layers.0.bias"]],
#       "fc2_weight":   [sd["head.layers.2.weight"]],
#       "fc2_bias":     [sd["head.layers.2.bias"]],
#   }
#
#   export_model(
#       model=model,
#       weight_mapping=weight_mapping,
#       model_json_path=Path("curved_road/exp1/model.json"),
#   )
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn


def _flatten_tensors(tensors: list[torch.Tensor]) -> list[float]:
    """Concatenate a list of tensors into a single flat list of Python floats.

    Parameters
    ----------
    tensors : list of torch.Tensor
        Can be any shapes — they are all flattened and concatenated in order.

    Returns
    -------
    list[float]
        All values from all tensors, in row-major (C) order, as Python floats.

    Example
    -------
    >>> t1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])   # shape (2, 2)
    >>> t2 = torch.tensor([5.0, 6.0])                  # shape (2,)
    >>> _flatten_tensors([t1, t2])
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    """
    result: list[float] = []
    for t in tensors:
        # .detach()  — remove from autograd graph (safe if it has grad_fn)
        # .cpu()     — move to CPU for numpy conversion
        # .numpy()   — zero-copy conversion to NumPy array
        # .flatten() — collapse all dims into 1-D
        # .tolist()  — Python list of native floats (JSON-serialisable)
        result.extend(t.detach().cpu().numpy().flatten().tolist())
    return result


def export_model(
    model: nn.Module,
    weight_mapping: dict[str, list[torch.Tensor]],
    model_json_path: Path,
) -> None:
    """Fill the model.json template with trained weights and write it to disk.

    Parameters
    ----------
    model : nn.Module
        The trained model. Used only to verify that weight_mapping covers
        all its parameters (optional sanity check).
    weight_mapping : dict[str, list[torch.Tensor]]
        Maps each model.json weight key to a list of tensors to concatenate.
        Built by the experiment's __main__.py using model.state_dict().
        Example:
          {
            "conv1_weight": [sd["backbone.conv_blocks.0.0.weight"]],
            "conv1_bias":   [sd["backbone.conv_blocks.0.0.bias"]],
            "transformer_enc_weight": [
                sd["transformer.encoder.layers.0.self_attn.in_proj_weight"],
                sd["transformer.encoder.layers.0.linear1.weight"],
                ...
            ],
          }
    model_json_path : Path
        Path to the model.json template file for this experiment.
        The file is read, updated in memory, and written back.

    Raises
    ------
    KeyError
        If weight_mapping contains a key not present in the template.
        This catches mismatches between the experiment mapping and the
        simulator contract early — before the simulator loads a bad file.
    """
    model_json_path = Path(model_json_path)

    # ── Step 1: Read the template ─────────────────────────────────────
    # The template has the correct format string and key structure.
    # Weight arrays are empty [] — we fill them below.
    with open(model_json_path, "r") as f:
        template = json.load(f)
    # template structure:
    #   { "format": "vision_only_vla", "num_params": 0, "weights": { "conv1_weight": [], ... } }

    # ── Step 2: Fill each weight key ──────────────────────────────────
    for json_key, tensors in weight_mapping.items():
        # Validate: the key must exist in the template.
        # If it doesn't, the experiment mapping is wrong — fail loudly.
        if json_key not in template["weights"]:
            raise KeyError(
                f"Key '{json_key}' not found in template {model_json_path}. "
                f"Available keys: {list(template['weights'].keys())}"
            )

        # Flatten the list of tensors for this key into a Python list of floats.
        # Multiple tensors are concatenated in the order provided.
        template["weights"][json_key] = _flatten_tensors(tensors)

    # ── Step 3: Count total parameters ────────────────────────────────
    # Sum the length of every filled array.
    # This gives the total number of scalar parameters exported.
    total_params = sum(len(v) for v in template["weights"].values())
    template["num_params"] = total_params

    # ── Step 4: Sanity check ──────────────────────────────────────────
    # Warn if any template key was left as an empty array.
    # This usually means the weight_mapping is incomplete.
    missing = [k for k, v in template["weights"].items() if len(v) == 0]
    if missing:
        print(
            f"WARNING: the following keys were not filled by weight_mapping "
            f"and remain empty: {missing}"
        )

    # ── Step 5: Write updated JSON back to disk ───────────────────────
    # indent=2 produces human-readable output (large but inspectable).
    # The simulator reads this file directly.
    with open(model_json_path, "w") as f:
        json.dump(template, f, indent=2)

    print(
        f"Exported {total_params:,} parameters → {model_json_path}"
    )
