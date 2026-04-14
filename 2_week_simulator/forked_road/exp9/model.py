"""
Experiment 9 model: ACT + Language + CVAE — forked road.

Assembles:
    shared.models.cnn_backbone      (4-conv variant)
    shared.models.language_encoder
    shared.models.transformer
    shared.models.cvae
    shared.models.mlp_head

Full architecture. CVAE adds marginal complexity when language
already resolves path ambiguity — this experiment tests whether it hurts.
"""
