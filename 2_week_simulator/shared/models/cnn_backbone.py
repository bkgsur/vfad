"""
CNN feature extractor shared across all 9 experiments.

Variants:
- 3-conv: used in curved-road experiments (Exp 1–3)
- 4-conv: used in forked-road experiments (Exp 4–9)

Input:  (B, 3, 128, 128)  RGB image batch
Output: (B, feature_dim)  flattened feature vector
"""
