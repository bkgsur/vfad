# Wiki Log

Append-only chronological record of all wiki operations.
Format: `## [YYYY-MM-DD] operation | description`

---

## [2026-04-16] ingest | Exp 1 results + initial concept batch

**Trigger:** Exp 1 trained successfully (91.4% val accuracy, 77.5% left turns).

**Pages created:**
- `concepts/cnn.md` — CNN backbone, 3-conv and 4-conv variants, shape walkthrough
- `concepts/mlp_head.md` — MLP action head, why raw logits, input_dim across experiments
- `concepts/bce_loss.md` — BCEWithLogitsLoss, multi-label vs single-label, shape walkthrough
- `concepts/training_loop.md` — 5-step update cycle, autograd, model.train()/eval()
- `concepts/callbacks.md` — CheckpointCallback, EarlyStoppingCallback, patience
- `concepts/metrics.md` — per-action accuracy, why left@77.5% is the honest score
- `experiments/exp1.md` — full results, per-action breakdown, interpretation
- `architecture/vision_only_vla.md` — CNN+MLP architecture, Exp 1 and 4
- `glossary.md` — initial term definitions
- `index.md` — catalog of all pages
- `SCHEMA.md` — wiki maintenance rules

**Notes:** Wiki scaffolded from scratch at vfad root. First ingest covers all
concepts introduced in Exp 1. Concepts for Transformer, CVAE, LanguageEncoder,
and Dataset/Transforms not yet written — to be added when relevant experiments train.
