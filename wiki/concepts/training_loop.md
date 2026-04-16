# Training Loop

## What it is
The repeated cycle that teaches a neural network. Each iteration shows the model
a batch of data, measures how wrong it is, and nudges its weights in the direction
that reduces the error.

## Why it matters in this project
Every experiment runs the same training loop. Understanding it once means you
understand how all 9 experiments learn.

## How it works

**One batch, always in this order:**

```
1. optimizer.zero_grad()          clear old gradients
2. logits = model(images)         forward pass — build computation graph
3. loss = criterion(logits, labels)  measure error
4. loss.backward()                backward pass — fill .grad on every parameter
5. optimizer.step()               update parameters using .grad
```

**Why zero_grad() first?**
PyTorch accumulates gradients by default. Without clearing, batch N's gradients
pile on top of batch N-1's and updates become wrong.

**What is the computation graph?**
During the forward pass, PyTorch silently records every operation (add, matmul,
relu...) in a directed graph. `loss.backward()` walks this graph in reverse,
applying the chain rule to compute `∂loss/∂parameter` for every weight.

**What does optimizer.step() actually do?**
- SGD: `param -= lr × grad`
- Adam: `param -= lr × (bias-corrected moment estimate of grad)`
Adam adapts the learning rate per parameter — parameters with historically
large gradients get smaller steps, and vice versa.

**Validation loop — the same but simpler:**
```python
with torch.no_grad():   # don't build computation graph — saves memory
    logits = model(images)
    loss = criterion(logits, labels)
    # no backward(), no step()
```
`torch.no_grad()` tells PyTorch not to record operations. Since we never call
`.backward()` during validation, the graph is wasteful — skipping it makes
validation faster and uses less memory.

**model.train() vs model.eval():**
Some layers behave differently during training vs inference:
- Dropout: zeros random neurons during train, passes all during eval
- BatchNorm: uses batch statistics during train, running stats during eval
Always switch correctly — forgetting `model.eval()` during validation is a
silent bug that gives slightly wrong metrics.

## Key intuition
Training is gradient descent: repeatedly measure "how wrong am I?" and take
a small step downhill. The chain rule (backprop) tells you which direction is
downhill for every single weight simultaneously.

## Where in the code
`2_week_simulator/shared/training/trainer.py` — `Trainer.fit()`, `_train_epoch()`, `_val_epoch()`

## See also
- [BCELoss](bce_loss.md) — the loss being minimised
- [Callbacks](callbacks.md) — checkpointing and early stopping that wrap the loop
- [Metrics](metrics.md) — accuracy computed after each epoch
