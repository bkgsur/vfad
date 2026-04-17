# TensorBoard

## What it is
A browser-based dashboard for visualising what happens inside a neural network
during training. You point it at a folder of log files and it draws live charts
of loss, accuracy, weight distributions, and gradients.

## Why it matters in this project
The terminal prints one line per epoch — loss and accuracy. TensorBoard shows
the *shape* of learning: which layers are updating, which are stuck, whether
weights are growing or dying, and exactly when the learning rate drops.
It turns training from a black box into something you can watch.

## How it works

**Three types of data are logged each epoch:**

### 1. Scalars — the headline numbers
```
Loss/train        smooth exponential decay → healthy
Loss/val          tracks train loss → no overfitting
Accuracy/acc      aggregate across all 4 actions
Accuracy/acc_left the honest score — hardest action
LearningRate      see exactly when the scheduler fires
```

### 2. Weight histograms — are the weights learning?
Each layer's weight tensor is visualised as a distribution over time.

```
Epoch 1:   narrow spike near 0   (random initialisation: weights start small)
Epoch 10:  slightly wider         (weights beginning to spread)
Epoch 80:  bell-shaped, centred   (healthy convergence)
```

A layer stuck as a spike near 0 at epoch 80 means it isn't contributing.
A layer with a distribution skewed far from 0 may be exploding.

Key layers to watch:
- `backbone.features.0.weight` (conv1) — learns edge detectors first
- `head.layers.2.weight` (fc2) — the action layer; its 4 rows diverge as
  the model learns which features predict each action

### 3. Gradient histograms — which layers are actively learning?
Gradients measure how much each weight changed this epoch: `∂loss/∂weight`.

```
Large gradients  → layer learning fast (early training, or adapting to signal)
Near-zero grads  → layer barely updating (converged, or vanishing gradient)
```

In Exp 1 you'll typically see:
- `fc2` gradients are largest — the final decision layer adapts fastest
- `conv1` gradients are smaller — the backbone learns more slowly
- `backbone.features.4.weight` (conv3) sits in between

This is expected and healthy. If conv1 had zero gradients throughout,
the backbone wouldn't be learning at all.

## How to launch

```bash
# From 2_week_simulator/
uv run tensorboard --logdir runs/
# Open http://localhost:6006
```

Each experiment writes to its own subfolder: `runs/exp1/`, `runs/exp2/`, etc.
Pointing `--logdir runs/` shows all experiments — you can overlay them to
compare how training dynamics change as the architecture grows.

## Key intuition
Loss tells you *whether* the model is learning. Weight and gradient histograms
tell you *where* inside the model the learning is happening. You need both to
diagnose training problems.

## Where in the code
`2_week_simulator/shared/training/trainer.py` — `Trainer._tb_log()`

Log files written to: `2_week_simulator/runs/<exp_name>/`

## See also
- [Training Loop](training_loop.md) — the loop that generates the logged data
- [Callbacks](callbacks.md) — checkpointing and early stopping that also run each epoch
