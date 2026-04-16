# MLP Head — Multi-Layer Perceptron Action Head

## What it is
A two-layer fully connected network that takes a feature vector and maps it to
action logits. It is the decision-making layer: "given what I see, what should I do?"

## Why it matters in this project
The CNN backbone produces a feature vector describing the image.
The MLP head converts that description into 4 action scores —
one for each of forward, backward, left, right.

It is used in every experiment as the final output layer, even when a Transformer
or CVAE sits between the CNN and the MLP.

## How it works

Two linear layers with a ReLU non-linearity between them:

```
Input:  (B, input_dim)      e.g. (B, 64) for Exp 1

Linear(input_dim → hidden_dim)    e.g. 64 → 64
ReLU                              sets negatives to zero
Linear(hidden_dim → num_actions)  e.g. 64 → 4

Output: (B, 4)   raw logits — one score per action
```

**Why two layers and not one?**
A single linear layer can only learn a linear decision boundary.
Adding a second layer with a ReLU in between lets the model learn curved,
non-linear boundaries — necessary for tasks where the relationship between
image features and actions is not a straight line.

**Why raw logits, not probabilities?**
`BCEWithLogitsLoss` applies sigmoid internally. Fusing sigmoid into the loss
is numerically more stable than calling sigmoid first then computing BCE.
Rule: model outputs logits; loss and metrics apply sigmoid.

**input_dim grows across experiments:**
- Exp 1: CNN output only → input_dim = 64
- Exp 4: 4-conv CNN → input_dim = 128
- Exp 5: CNN + language embedding → input_dim = 128 + 32 = 160

The MLP head doesn't change — only its input_dim changes.

## Key intuition
The MLP is the part of the network that "decides". Everything before it is
perception (what do I see?) or context (what am I told?). The MLP takes all
of that and outputs a score for each possible action.

## Where in the code
`2_week_simulator/shared/models/mlp_head.py` — `MLPHead(input_dim, hidden_dim, num_actions)`

## Experiments that use this
All 9 experiments. Input dim varies; the architecture is the same.

## See also
- [CNN](cnn.md) — produces the input to the MLP in vision-only experiments
- [BCELoss](bce_loss.md) — the loss applied to MLP outputs
- [LanguageEncoder](language_encoder.md) — concatenated with CNN features in Exp 5, 8, 9
