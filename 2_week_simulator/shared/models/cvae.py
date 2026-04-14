"""
Conditional Variational Autoencoder (CVAE) used in Exp 3, 7, 9.

Models the distribution over possible action sequences,
allowing the model to handle ambiguous situations (e.g. two valid paths).

Loss adds a KL divergence term to BCE: total_loss = BCE + beta * KL
"""
