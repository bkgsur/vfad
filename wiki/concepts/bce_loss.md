# BCEWithLogitsLoss — Binary Cross-Entropy Loss

## What it is
The loss function used in all 9 experiments. It measures how wrong the model's
action predictions are, and produces a single number the optimizer uses to
update the weights.

## Why it matters in this project
Each of the 4 actions (forward, backward, left, right) is an **independent binary
decision** — the car can go forward AND left at the same time. This is multi-label
classification, not single-label.

`BCEWithLogitsLoss` treats each action as its own binary problem:
"Is forward happening? Yes or no. Is left happening? Yes or no." independently.
CrossEntropyLoss would force a single choice — wrong for this task.

## How it works

For one action, binary cross-entropy is:

```
BCE = -[ y * log(p) + (1-y) * log(1-p) ]

where:
  y = true label (0 or 1)
  p = predicted probability (after sigmoid)
```

- If y=1 and p≈1: loss ≈ 0 (correct, confident)
- If y=1 and p≈0: loss → ∞ (correct label, wrong prediction — heavily penalised)
- If y=0 and p≈0: loss ≈ 0 (correct)
- If y=0 and p≈1: loss → ∞ (heavily penalised)

`BCEWithLogitsLoss` fuses sigmoid + BCE into one operation:
```
p = sigmoid(logit) = 1 / (1 + exp(-logit))
loss = BCE(p, y)
```
Fusing them is more numerically stable than computing sigmoid then BCE separately
(avoids log(0) when probabilities are very close to 0 or 1).

**Shape walkthrough (B=2 samples, 4 actions):**
```
logits:  (2, 4)   raw model output
labels:  (2, 4)   float 0.0 or 1.0

BCEWithLogitsLoss applies sigmoid to logits → (2, 4) probabilities
computes BCE per element → (2, 4) per-element losses
takes mean over all 2×4=8 values → scalar loss
```

## Key intuition
The loss asks: "For every action in every sample, how surprised were you?"
Confident correct predictions contribute near zero. Confident wrong predictions
contribute a large penalty. The mean over all actions and all samples is what
the optimizer minimises.

## Where in the code
Used in all `__main__.py` files:
```python
criterion = nn.BCEWithLogitsLoss()
loss = criterion(logits, labels)   # logits: (B,4), labels: (B,4)
```

## Experiments that use this
All 9. CVAE experiments (3, 7, 9) add a KL term on top:
`total_loss = BCE + beta * KL`. See [CVAE](cvae.md).

## See also
- [MLP Head](mlp_head.md) — produces the logits this loss consumes
- [Metrics](metrics.md) — how accuracy is computed from the same logits
- [CVAE](cvae.md) — adds KL divergence to this loss
