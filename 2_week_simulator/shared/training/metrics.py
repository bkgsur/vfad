"""
Evaluation metrics for action prediction.

Computes per-action accuracy for each of:
  forward, backward, left, right

And aggregate accuracy (mean across all 4 actions).
Threshold: sigmoid(logit) > 0.5 → predicted 1
"""
