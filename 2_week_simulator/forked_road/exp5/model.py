"""
Experiment 5 model: Vision CNN + Language Embedding — forked road.

Assembles:
    shared.models.cnn_backbone      (4-conv variant)
    shared.models.language_encoder
    shared.models.mlp_head

Language embedding is concatenated with CNN features before MLP.
"""
