# Exp 1 — CNN + MLP (Curved Road)

## What changed from previous experiment
Baseline — no previous experiment. This is the simplest possible VLA model.

## Architecture

```
Input: (B, 3, 128, 128)   RGB camera image

CNN Backbone (3-conv):
  Conv(3→16, stride=2) → ReLU   →  (B, 16, 64, 64)
  Conv(16→32, stride=2) → ReLU  →  (B, 32, 32, 32)
  Conv(32→64, stride=2) → ReLU  →  (B, 64, 16, 16)
  AdaptiveAvgPool2d(1)           →  (B, 64, 1, 1)
  Flatten                        →  (B, 64)

MLP Head:
  Linear(64→64) → ReLU
  Linear(64→4)

Output: (B, 4)   logits — [forward, backward, left, right]
```

**Total parameters: 28,004**

## Results

| Metric | Value |
|--------|-------|
| Val accuracy (aggregate) | **91.4%** |
| Reference (experiment_progression.md) | 90.0% |
| Best epoch | 71 / 80 |
| Train loss (epoch 80) | 0.1999 |
| Val loss (epoch 80) | 0.2018 |

**Per-action breakdown:**

| Action   | Val Accuracy | Note |
|----------|-------------|------|
| forward  | 98.0% | Near-trivial — present in ~100% of samples |
| backward | 100.0% | Trivial — never occurs, model always predicts 0 |
| left     | 77.5% | **The real test** — 42% frequency, genuinely learned |
| right    | 90.2% | Rare (10%) but visually distinct, well learned |

## What the results tell us

The 91.4% aggregate is inflated by backward (100%) and forward (98%) —
both are trivially predictable from data distribution alone.

**Left at 77.5% is the honest score.** The model gets 1 in 4 left turns wrong.
This makes sense: left-turn images look similar to straight-ahead images, and
the model has only 64 features to distinguish them.

The loss curve was clean — smooth exponential decay, train and val tracking closely
(gap of 0.002 at epoch 80), no overfitting. The task is learnable with this
architecture but not fully solved.

## Key takeaway
A 28,004-parameter CNN + MLP achieves 91.4% aggregate accuracy on curved road,
but the honest measure (left turns) is only 77.5%. This is the baseline every
subsequent experiment must beat on the metric that matters.

## Concepts used
- [CNN](../concepts/cnn.md)
- [MLP Head](../concepts/mlp_head.md)
- [BCELoss](../concepts/bce_loss.md)
- [Training Loop](../concepts/training_loop.md)
- [Callbacks](../concepts/callbacks.md)
- [Metrics](../concepts/metrics.md)

## Code
- `2_week_simulator/curved_road/exp1/model.py`
- `2_week_simulator/curved_road/exp1/__main__.py`
- `2_week_simulator/curved_road/exp1/config.yaml`
- `2_week_simulator/curved_road/exp1/notes.md` — full observations
