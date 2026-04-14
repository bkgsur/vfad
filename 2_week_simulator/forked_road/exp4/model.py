"""
Experiment 4 model: Vision CNN (4-conv) + MLP head — forked road.

Assembles:
    shared.models.cnn_backbone  (4-conv variant)
    shared.models.mlp_head

No transformer, no CVAE, no language conditioning.
Deeper backbone than Exp 1 for harder forked-road task.
"""
