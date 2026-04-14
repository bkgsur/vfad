# VFAD Project — Claude Context

## What This Project Is

A step-by-step learning project covering **everything needed to understand VLA
(Vision-Language-Action) models** — from basic PyTorch building blocks through
Transformers, CVAEs, and language conditioning — built around a real car simulator.

The car navigates roads automatically using only camera images as input.
The 9 experiments form a deliberate curriculum: each one adds exactly one new idea
so the user can isolate and understand what each component contributes.

**Learning rules:**
- Explain every concept before writing the code for it
- Every source file must contain inline comments explaining each line and concept — not just what, but why
- The file itself is the learning material — explanations must live in the code, not only in chat
- Never skip ahead — understanding matters more than speed
- This project IS the VLA curriculum, not just a coding exercise

---

## Active Work: `2_week_simulator/`

### Always read at session start
1. `2_week_simulator/learning_plan.md` — which shared module to write next, progress tracker
2. `2_week_simulator/experiment_progression.md` — all 9 experiments, architectures, goals

---

## The 9 Experiments

### Set 1: Curved Road (Exp 1–3)
Goal: car follows a curved road automatically. Single path, no choices.

| Exp | Architecture | Key concept introduced |
|-----|-------------|----------------------|
| 1 | CNN (3-conv) + MLP | Baseline vision model |
| 2 | CNN + Transformer (ACT) | Action chunking |
| 3 | CNN + Transformer + CVAE | Handling uncertainty |

### Set 2: Forked Road (Exp 4–9)
Goal: car navigates a junction and chooses the correct path from a language instruction.

| Exp | Architecture | Key concept introduced |
|-----|-------------|----------------------|
| 4 | CNN (4-conv) + MLP | Deeper backbone for harder task |
| 5 | CNN + Language embedding | Language conditioning |
| 6 | CNN + Transformer (ACT) | Action chunking at junction |
| 7 | CNN + Transformer + CVAE | Path ambiguity modelling |
| 8 | CNN + Transformer + Language | Best combo: instruction-aware chunking |
| 9 | CNN + Transformer + Language + CVAE | Full architecture ablation |

---

## Project Structure

```
2_week_simulator/
├── shared/                         # Reusable modules across all 9 experiments
│   ├── data/
│   │   ├── dataset.py              # JSON loading, base64 decode, DataLoader
│   │   └── transforms.py           # Image normalisation, augmentation
│   ├── models/
│   │   ├── cnn_backbone.py         # 3-conv (Exp 1–3) or 4-conv (Exp 4–9)
│   │   ├── mlp_head.py             # Action MLP head
│   │   ├── transformer.py          # ACT-style Transformer Enc/Dec (Exp 2,3,6–9)
│   │   ├── cvae.py                 # CVAE (Exp 3,7,9)
│   │   └── language_encoder.py    # Language embedding (Exp 5,8,9)
│   ├── training/
│   │   ├── trainer.py              # Base training loop
│   │   ├── metrics.py              # Per-action accuracy
│   │   └── callbacks.py            # Checkpointing, early stopping
│   └── utils/
│       ├── config.py               # Loads config.yaml → frozen dataclass
│       ├── exporter.py             # Writes trained weights → model.json
│       └── visualize.py            # Loss/accuracy plots
├── curved_road/
│   ├── exp1/                       # training-data/ + model.json + config.yaml + model.py
│   ├── exp2/
│   └── exp3/
├── forked_road/
│   ├── exp4/ … exp9/
├── results/
│   ├── curved_road/exp1–3/         # best_model.pt, model.json, curves.png
│   └── forked_road/exp4–9/
├── learning_plan.md
├── experiment_progression.md
└── pyproject.toml                  # uv/hatchling, Python >=3.11, PyTorch
```

---

## Key Technical Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Framework | PyTorch | — |
| Action labels | Binary (0/1), multi-label | Each of 4 actions is independent |
| Loss | BCEWithLogitsLoss | Multi-label binary, logits not sigmoids |
| Val split | Random 80/20 across all samples | — |
| Accuracy | Per-action + aggregate | Forward/backward dominate — per-action reveals real learning |
| Run command | `uv run python -m curved_road.exp1` | From `2_week_simulator/` root |

---

## Data Format

Each experiment's training data lives in `<exp>/training-data/*.json`.

```json
{
  "metadata": { "image_width": 128, "image_height": 128, "num_samples": 390 },
  "samples": [
    {
      "image": "<base64 JPEG>",
      "actions": { "forward": 1, "backward": 0, "left": 1, "right": 0 },
      "speed": 1.0,
      "steering": 0.206
    }
  ]
}
```

Speed and steering are **ignored** — only the 4 binary actions are predicted.

---

## model.json Contract

Each experiment folder has a `model.json`. Before training it is a **template**
(empty weight arrays). After training, `exporter.py` fills it with real weights.
The simulator reads this file directly.

| Exp | `format` field |
|-----|---------------|
| 1 | `vision_only_vla` |
| 2 | `act_vla` |
| 3 | `act_cvae_vla` |
| 4 | `vision_only_vla` |
| 5 | `vision_language_vla` |
| 6 | `act_vla` |
| 7 | `act_cvae_vla` |
| 8 | `act_language_vla` |
| 9 | `act_language_cvae_vla` |

The simulator source code is not available — the model.json templates are the contract.

---

## Shared Module Learning Order

See `2_week_simulator/learning_plan.md` for full detail and progress tracking.

```
config → visualize → transforms → dataset
    → mlp_head → cnn_backbone → language_encoder
        → metrics → callbacks → trainer
            → transformer → cvae → exporter
```
