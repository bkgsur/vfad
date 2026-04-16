# Metrics — Per-Action Accuracy

## What it is
A function that computes accuracy for each of the 4 actions independently,
plus an aggregate (mean) accuracy across all actions.

## Why it matters in this project
Aggregate accuracy is misleading here. The data distribution is heavily skewed:
- forward: ~100% of samples
- backward: 0% of samples
- left: ~42% of samples
- right: ~10% of samples

A model that predicts `forward=1, backward=0` for every sample regardless of
the image would score ~75% aggregate accuracy without learning anything.
Per-action accuracy exposes this: backward would be 100%, forward 100%,
left and right 0% — immediately revealing the model learned nothing real.

## How it works

```
logits → sigmoid → probabilities
probabilities > 0.5 → predicted labels (0 or 1)
predicted == true labels → correct (1) or wrong (0)
mean over batch (dim=0) → per-action accuracy: (4,)
mean over everything → aggregate accuracy: scalar
```

**Shape walkthrough (B=4 samples, 4 actions):**
```
logits:  (4, 4)
sigmoid  → probs:   (4, 4)   values in (0, 1)
> 0.5    → preds:   (4, 4)   0.0 or 1.0
== labels → correct: (4, 4)   0.0 (wrong) or 1.0 (right)
mean(dim=0) → per_action: (4,)  one accuracy per action column
mean()      → aggregate:  scalar
```

## Exp 1 results (curved road baseline)

| Action   | Val Accuracy |
|----------|-------------|
| forward  | 98.0%        |
| backward | 100.0%       |
| left     | 77.5%        |
| right    | 90.2%        |
| aggregate | 91.4%       |

**Interpretation:** backward (100%) and forward (98%) are trivial — the model
exploits data imbalance. The real measure of learning is **left at 77.5%**.
That is the number to watch across experiments.

## Key intuition
Aggregate accuracy hides behind easy classes. Always look at the hardest class.
In this project, that is `left` — the minority positive class on curved road.

## Where in the code
`2_week_simulator/shared/training/metrics.py` — `compute_metrics(logits, targets)`
Returns dict with keys: `acc`, `acc_forward`, `acc_backward`, `acc_left`, `acc_right`

## See also
- [BCELoss](bce_loss.md) — loss computed from the same logits
- [Training Loop](training_loop.md) — where metrics are computed each epoch
