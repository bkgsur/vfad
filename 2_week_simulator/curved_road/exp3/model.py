"""
Experiment 3 model: ACT + CVAE.

Assembles:
    shared.models.cnn_backbone  (3-conv variant)
    shared.models.transformer
    shared.models.cvae
    shared.models.mlp_head

Adds latent variable modelling over Exp 2.
Loss = BCE + beta * KL
"""
