"""
Experiment 8 model: ACT + Language — forked road.

Assembles:
    shared.models.cnn_backbone      (4-conv variant)
    shared.models.language_encoder
    shared.models.transformer
    shared.models.mlp_head

Language embedding conditions the transformer, not just the MLP.
Best-performing combo for instruction-following at a junction.
"""
