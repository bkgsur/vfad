# Shared Modules: Learning Plan

Write and learn each shared module in order — simplest first, most complex last.
Each module is written one at a time with full explanation before moving on.

---

## Progress Tracker

| # | File | Status | Concepts |
|---|------|--------|----------|
| 1 | `shared/utils/config.py` | done | YAML loading, frozen dataclasses, type annotations |
| 2 | `shared/utils/visualize.py` | done | matplotlib, plotting loss/accuracy curves |
| 3 | `shared/data/transforms.py` | done | tensor normalisation, image augmentation |
| 4 | `shared/data/dataset.py` | done | `torch.utils.data.Dataset`, base64 decode, random split, `DataLoader` |
| 5 | `shared/models/mlp_head.py` | done | `nn.Module`, `nn.Linear`, `nn.ReLU`, forward pass |
| 6 | `shared/models/cnn_backbone.py` | done | `nn.Conv2d`, `nn.AdaptiveAvgPool2d`, spatial → vector, stride vs pooling |
| 7 | `shared/models/language_encoder.py` | done | `nn.Embedding`, lookup tables, integer IDs → dense vectors |
| 8 | `shared/training/metrics.py` | done | sigmoid threshold, per-action accuracy |
| 9 | `shared/training/callbacks.py` | done | stateful objects, best-model tracking, early stopping |
| 10 | `shared/training/trainer.py` | done | training loop, `loss.backward()`, optimizer, scheduler, device |
| 11 | `shared/models/transformer.py` | pending | attention, positional encoding, action chunking |
| 12 | `shared/models/cvae.py` | pending | latent space, reparameterisation trick, KL divergence, ELBO loss |
| 13 | `shared/utils/exporter.py` | pending | `state_dict()`, flattening tensors, writing model.json |

Update status to `done` as each module is completed.

---

## Layer 0 — Pure Python utilities (no ML)

### 1. `shared/utils/config.py`
- Reads a `config.yaml` file and returns a **frozen dataclass**
- No PyTorch involved — pure Python
- Teaches: YAML parsing, `dataclasses`, type annotations, immutability

### 2. `shared/utils/visualize.py`
- Reads saved metrics from `results/` and plots training curves
- Teaches: matplotlib basics, reading files, per-action bar charts

---

## Layer 1 — Data pipeline

### 3. `shared/data/transforms.py`
- Defines train transform (normalise + augment) and val transform (normalise only)
- Teaches: tensor operations, why augmentation helps, brightness jitter, gaussian noise

### 4. `shared/data/dataset.py`
- Parses the JSON training files, decodes base64 JPEG images, extracts binary action labels
- Implements `torch.utils.data.Dataset`, performs random 80/20 split, wraps in `DataLoader`
- Teaches: how PyTorch loads data, what a Dataset and DataLoader are, batching

---

## Layer 2 — Model building blocks

### 5. `shared/models/mlp_head.py`
- Two linear layers with ReLU in between: `Linear → ReLU → Linear`
- Outputs raw logits (no sigmoid — BCEWithLogitsLoss handles that)
- Teaches: `nn.Module`, `__init__` vs `forward`, what a linear layer does

### 6. `shared/models/cnn_backbone.py`
- Two variants: 3-conv (curved road) and 4-conv (forked road)
- Each conv block: `Conv2d → ReLU → MaxPool`
- Teaches: convolution intuition, pooling, how spatial dimensions shrink, feature maps

### 7. `shared/models/language_encoder.py`
- Converts an integer instruction ID into a dense embedding vector
- Teaches: `nn.Embedding`, why we don't use one-hot vectors, how language conditions behaviour

---

## Layer 3 — Training infrastructure

### 8. `shared/training/metrics.py`
- Computes accuracy per action: forward, backward, left, right + aggregate
- Threshold: `sigmoid(logit) > 0.5 → predicted 1`
- Teaches: why we track per-action not just aggregate, class imbalance visibility

### 9. `shared/training/callbacks.py`
- `CheckpointCallback`: saves `best_model.pt` when val loss improves
- `EarlyStoppingCallback`: stops training when val loss stops improving for N epochs
- Teaches: stateful objects in a training loop, why we save the best not the last

### 10. `shared/training/trainer.py`
- `fit()` method: epoch loop → train epoch → val epoch → callbacks → log metrics
- Device-agnostic (CPU or CUDA)
- Teaches: full PyTorch training loop, `loss.backward()`, `optimizer.step()`, `zero_grad()`, scheduler

---

## Layer 4 — Advanced model modules (Exp 2+ only)

> Not needed for Exp 1. Written after the training loop is understood.

### 11. `shared/models/transformer.py`
- Transformer Encoder/Decoder for ACT-style action chunking
- Predicts a sequence of future actions rather than one at a time
- Teaches: self-attention, cross-attention, positional encoding, why chunking smooths driving

### 12. `shared/models/cvae.py`
- Conditional VAE: encoder produces `mu` + `logvar`, decoder reconstructs from sampled `z`
- Loss = BCE + β × KL divergence
- Teaches: latent variable models, reparameterisation trick, why ELBO works, when CVAE helps vs hurts

---

## Layer 5 — Export

### 13. `shared/utils/exporter.py`
- After training, reads the experiment's `model.json` template for the expected key structure
- Extracts weights from `model.state_dict()`, flattens to lists, writes into the template
- Teaches: `state_dict()`, how PyTorch stores weights, the simulator contract

---

## Dependency Order

```
config → visualize → transforms → dataset
    → mlp_head → cnn_backbone → language_encoder
        → metrics → callbacks → trainer
            → transformer → cvae
                → exporter
```

Each module depends only on modules written before it.
