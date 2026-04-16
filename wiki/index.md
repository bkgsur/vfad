# Wiki Index

Every page in this wiki, one line each. Read this first when looking for anything.

---

## Concepts

| Page | Summary |
|------|---------|
| [CNN](concepts/cnn.md) | Convolutional feature extractor — converts 128×128 images to 64/128-dim vectors |
| [MLP Head](concepts/mlp_head.md) | Two-layer fully connected network mapping features to 4 action logits |
| [BCELoss](concepts/bce_loss.md) | Binary cross-entropy with sigmoid — loss function for multi-label action prediction |
| [Training Loop](concepts/training_loop.md) | zero_grad → forward → loss → backward → step — the core 5-step update cycle |
| [Callbacks](concepts/callbacks.md) | CheckpointCallback saves best weights; EarlyStoppingCallback halts stalled training |
| [Metrics](concepts/metrics.md) | Per-action accuracy — why left@77.5% matters more than aggregate@91.4% |

---

## Experiments

| Page | Summary |
|------|---------|
| [Exp 1](experiments/exp1.md) | CNN+MLP curved road baseline — 91.4% aggregate, 77.5% left turns |

---

## Architecture

| Page | Summary |
|------|---------|
| [vision_only_vla](architecture/vision_only_vla.md) | CNN + MLP — simplest architecture, used in Exp 1 and 4 |

---

## Reference

| Page | Summary |
|------|---------|
| [Glossary](glossary.md) | One-line definitions for all technical terms |
| [SCHEMA.md](SCHEMA.md) | Rules for how this wiki is structured and maintained |
