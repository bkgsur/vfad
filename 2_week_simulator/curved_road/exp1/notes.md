# Exp 1 — CNN + MLP (Curved Road) · Notes

Fill this in after the experiment trains successfully.

---

## 1. Loss Curve

*What did the training and validation loss curves look like?*

- Did loss drop smoothly, oscillate, or flatline?
- Did train and val loss track each other, or did they diverge (sign of overfitting)?
- Roughly which epoch did val loss stop improving?

```
[paste your observations here, or attach curves.png]
```

---

## 2. Accuracy

*Final val accuracy:*  _____ %  (reference: 90.0% from experiment_progression.md)

Per-action breakdown:

| Action   | Val Accuracy |
|----------|-------------|
| forward  |             |
| backward |             |
| left     |             |
| right    |             |

*Which actions did the model struggle with most? Why might that be?*
(Hint: look at the data distribution — 100% forward, ~42% left, ~10% right, 0% backward)

---

## 3. Deliberate Break

*Pick one thing to break intentionally, run training, observe what happens.*

Suggested options:
- Remove `optimizer.zero_grad()` — what happens to the loss?
- Skip `model.eval()` during validation — does accuracy change?
- Pass logits through `sigmoid()` before `BCEWithLogitsLoss` — what does the loss curve look like?
- Use a learning rate 10× too high — what does the loss curve do?

What I broke:

```
[describe the change]
```

What happened:

```
[describe the result — loss curve, accuracy, error message]
```

Why it happened:

```
[your explanation]
```

---

## 4. System Trace

*Pick one sample and manually trace it through the model in a notebook.*

```python
import torch
# load one image from training data, run it through each module,
# print shape and value after each step

# e.g.
# x = load_sample()               # (1, 3, 128, 128)
# feats = backbone(x)             # (1, 64)
# logits = head(feats)            # (1, 4)
# probs = torch.sigmoid(logits)   # (1, 4)  — what probabilities did the model assign?
```

What surprised you?

```
[write here]
```

---

## 5. Architecture Question

*One question this experiment raised that you want to answer in a future experiment.*

```
[e.g. "The model gets right turns wrong — would a deeper backbone help?"
      "What if the road had a fork — could this model handle it with no language?"]
```

---

## 6. Key Takeaway

*One sentence: what is the most important thing you learned from this experiment?*

```
[write here]
```
