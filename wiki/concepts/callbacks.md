# Callbacks — Checkpointing and Early Stopping

## What it is
Stateful objects the training loop calls at the end of each epoch to:
1. Save the best model seen so far (`CheckpointCallback`)
2. Stop training when the model stops improving (`EarlyStoppingCallback`)

## Why it matters in this project
Without callbacks, training runs for a fixed number of epochs and saves the
final weights. Two problems:
- The final model is rarely the best — val loss typically rises after a peak
- Training wastes time after learning stalls

Callbacks solve both. They are the reason `best_model.pt` contains the
best weights, not the last-epoch weights.

## How it works

**CheckpointCallback:**
```
each epoch:
  if val_loss < best_val_loss:
      save model.state_dict() → best_model.pt
      update best_val_loss
```
`model.state_dict()` is deepcopied before saving — tensors are live views
and would silently mutate if not copied.

**EarlyStoppingCallback:**
```
each epoch:
  if val_loss improved:
      reset counter to 0
  else:
      counter += 1
      if counter >= patience:
          set should_stop = True
```
The training loop polls `should_stop` after each epoch and breaks.

**Exp 1 result:** Best val accuracy 91.45% was reached at epoch 71 out of 80.
The checkpoint saved those weights; the final epoch (80) was slightly worse.

## Key intuition
The best model lives somewhere in the middle of training — after the model
has learned but before it starts memorising training data. Callbacks freeze
that moment on disk.

## Where in the code
`2_week_simulator/shared/training/callbacks.py`

## See also
- [Training Loop](training_loop.md) — where callbacks are called
