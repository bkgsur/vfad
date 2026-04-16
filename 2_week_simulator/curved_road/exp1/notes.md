# Exp 1 — CNN + MLP (Curved Road) · Notes

---

## 1. Loss Curve

- Loss dropped smoothly and consistently — no oscillation, no flatline
- Train and val loss tracked each other closely throughout — no sign of overfitting
- The gap at epoch 80 is tiny: train 0.1999 vs val 0.2018 (difference of 0.002)
- Most of the learning happened in the first 20 epochs; after that it was slow, steady improvement

| Epoch | Train Loss | Val Loss | Val Acc |
|-------|-----------|----------|---------|
| 1     | 0.3114    | 0.3032   | 85.2%   |
| 10    | 0.2416    | 0.2351   | 89.7%   |
| 20    | 0.2238    | 0.2188   | 90.4%   |
| 40    | 0.2123    | 0.2094   | 91.0%   |
| 80    | 0.1999    | 0.2018   | 91.4%   |

See: `results/curved_road/exp1/curves.png`

---

## 2. Accuracy

**Final val accuracy: 91.4%** (reference: 90.0%) — exceeded the reference.

**Best val accuracy: 91.45% at epoch 71.**

Per-action breakdown:

| Action   | Val Accuracy |
|----------|-------------|
| forward  | 98.0%        |
| backward | 100.0%       |
| left     | 77.5%        |
| right    | 90.2%        |

**Analysis:**
- forward (98.0%) — near-perfect, as expected; it appears in almost every sample
- backward (100.0%) — perfect; it never appears so the model always predicts 0 and is always right
- left (77.5%) — the weakest action and the most informative one; 42% frequency means the model
  has to genuinely learn when to turn, and it still gets 1 in 4 wrong
- right (90.2%) — better than left despite being rarer (10%); right turns on a curved road may
  be visually more distinct than left turns, making them easier to learn

The aggregate 91.4% is heavily inflated by backward (100%) and forward (98%).
The real measure of learning is **left at 77.5%** — that's where the model earns its accuracy.
This is the number to watch in Exp 2: does action chunking improve left-turn prediction?

---

## 3. Deliberate Break

*TODO — try one of these after reviewing the results:*
- Remove `optimizer.zero_grad()` — what happens to the loss?
- Skip `model.eval()` during validation — does accuracy change?
- Pass logits through `sigmoid()` before `BCEWithLogitsLoss` — what does the loss do?
- Use a learning rate 10× too high (1e-2) — does loss explode or oscillate?

---

## 4. System Trace

*TODO — run this in a notebook to trace one sample end-to-end:*

```python
import torch, sys
sys.path.insert(0, '.')  # from 2_week_simulator/

from curved_road.exp1.model import Exp1Model
from shared.utils.config import load_config
from pathlib import Path

cfg = load_config('curved_road/exp1/config.yaml')
model = Exp1Model(cfg.model.feature_dim, cfg.model.mlp_hidden_dim, cfg.model.num_actions)
model.load_state_dict(torch.load('results/curved_road/exp1/best_model.pt', map_location='cpu'))
model.eval()

# Trace one image through each module
x = torch.randn(1, 3, 128, 128)          # fake image — replace with a real one
print("input:   ", x.shape)              # (1, 3, 128, 128)

feats = model.backbone(x)
print("backbone:", feats.shape)           # (1, 64)

logits = model.head(feats)
print("logits:  ", logits)               # (1, 4) raw scores

probs = torch.sigmoid(logits)
print("probs:   ", probs)                # (1, 4) probabilities
# [forward, backward, left, right]
```

---

## 5. Architecture Question

Does action chunking (predicting a sequence of future actions rather than one at a time)
produce smoother driving on the curved road? → answered by Exp 2.

---

## 6. Key Takeaway

A simple 3-layer CNN + 2-layer MLP with only 28,004 parameters can follow a curved road
at 91.4% accuracy — proving that even minimal visual models work when the task has a
single, learnable pattern. The baseline is stronger than expected.
