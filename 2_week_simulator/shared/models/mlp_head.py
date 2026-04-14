"""
MLP action head used in Exp 1, 4, 5.

Input:  (B, input_dim)  feature vector from CNN backbone
Output: (B, num_actions) raw logits (no sigmoid — BCEWithLogitsLoss expects logits)

Architecture: Linear → ReLU → Linear
"""
