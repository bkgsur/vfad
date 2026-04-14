"""
Transformer Encoder/Decoder block (ACT-style) used in Exp 2, 3, 6, 7, 8, 9.

Implements action chunking: predicts a sequence of future actions
rather than a single action, producing smoother driving behaviour.

Input:  (B, seq_len, d_model) encoded features
Output: (B, chunk_size, num_actions) action sequence logits
"""
