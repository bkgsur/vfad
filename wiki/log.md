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

---

## [2026-04-16] update | TensorBoard added to trainer; system understanding session

**Trigger:** TensorBoard logging added to `shared/training/trainer.py`; session
focused on system understanding — data flow, weight meaning, architecture-as-diagnosis.

**Pages created:**
- `concepts/tensorboard.md` — scalars, weight histograms, gradient histograms, how to launch

**Pages updated:**
- `concepts/training_loop.md` — added See also link to TensorBoard

**Code changes (not wiki):**
- `shared/training/trainer.py` — added `SummaryWriter`, `_tb_log()`, `close()`; logs loss,
  per-action accuracy, learning rate, weight histograms, gradient histograms each epoch
- `curved_road/exp1/__main__.py` — passes `log_dir="runs/exp1"` to Trainer, calls `trainer.close()`
- `2_week_simulator/.gitignore` — added `runs/`

**Key insight recorded this session:** Each experiment's architecture is a diagnosis.
Jerkiness → add Transformer. Junction ambiguity → add CVAE. Can't choose path → add language.
The architecture choices are not arbitrary — they are fixes to observable failures.
