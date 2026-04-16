---
description: Guide for building system understanding, debugging intuition, domain expertise, and architecture thinking through the VFAD project experiments.
---

The user wants guidance on deepening one or more of the four learning dimensions in this project. Read the current state of `2_week_simulator/learning_plan.md` and `2_week_simulator/experiment_progression.md` to understand where they are, then provide targeted, experiment-specific advice for whichever dimension(s) they ask about.

If no specific dimension is mentioned, give an overview of all four and ask which they want to go deeper on.

---

## The Four Learning Dimensions

### 1. System Understanding
*Goal: hold the whole pipeline in your head — raw JSON → model → simulator.*

Practices to suggest based on where they are:
- **Manual end-to-end trace**: pick one sample from the training JSON, run it through every module in a notebook, print shape and value after each step. Do this for the current experiment's model.
- **Removal test**: ask "what breaks if I remove this module?" Then actually remove it and observe.
- **Draw the pipeline**: raw JSON → Dataset → DataLoader → Model → Loss → Callbacks → model.json → Simulator. One box per component, one arrow per data flow. Keep it physical (paper or whiteboard).
- **Cross-module questions to pose**: Why does `dataset.py` return float labels instead of int? Why does `exporter.py` need the model.json template? Why does `trainer.py` call `model.eval()` before validation?

### 2. Debugging Intuition
*Goal: form a hypothesis within 30 seconds of seeing a problem and know exactly where to look.*

Practices to suggest:
- **Deliberate breakage**: after each experiment trains, introduce one bug at a time from this list and observe the result:
  - Remove `optimizer.zero_grad()` — loss grows or oscillates
  - Skip `model.eval()` during validation — val accuracy is slightly wrong
  - Apply `sigmoid()` before `BCEWithLogitsLoss` — loss starts near 0, model doesn't learn
  - Set learning rate 10× too high — loss explodes or oscillates wildly
  - Forget `.to(device)` on a batch tensor — RuntimeError: device mismatch
  - Use `loss.item()` inside the batch loop but forget to divide by `len(loader)` — loss scale is wrong
- **Shape asserts**: suggest adding `assert tensor.shape == expected, f"got {tensor.shape}"` at every module boundary. When shapes break, you get the exact location.
- **Loss curve reading**: teach them to narrate what they see:
  - Smooth exponential drop → healthy learning
  - Immediate flatline → broken gradient (check zero_grad, check loss)
  - Oscillation → lr too high
  - Train loss drops, val loss rises → overfitting
  - Both losses plateau early → model too small or lr too low

### 3. Domain Expertise (VLA / Robot Control)
*Goal: understand WHY the architecture choices matter for robot control, not just that they work.*

Practices to suggest:
- **Read the ACT paper** (Action Chunking with Transformers) after Exp 2 trains. They've already implemented it — reading it after is 10× more effective.
- **Analyse the data distribution**: curved road is 100% forward, ~42% left, ~10% right, 0% backward. Ask: what does a model that minimises average loss predict when uncertain? Which action does it default to?
- **Connect experiment deltas to robot behaviour**: Exp 1→2 adds chunking. Ask: what does jerky vs smooth driving look like physically? Why does predicting 10 steps at once force consistency?
- **After all 9 experiments**: read RT-1 and OpenVLA papers. They will make complete sense because this project is a miniature version.
- **Suggested question to pose after each experiment**: "In a real robot arm task (not a car), would this architectural change matter more or less? Why?"

### 4. Architecture Thinking
*Goal: look at a problem and choose the right model — and explain the tradeoffs.*

Practices to suggest:
- **Treat the 9 experiments as a controlled ablation study**: each experiment changes exactly one thing. After each one, write one paragraph: "Adding X improved accuracy by Y%. The reason is Z. In a task with higher ambiguity / richer language, I expect this to matter more/less because..."
- **Design a 10th experiment**: after Exp 9, ask: what would YOU try next? Larger chunk size? Attention over multiple past frames? A deeper language encoder? Write the hypothesis and expected result — don't need to run it.
- **Failure case analysis**: don't just read aggregate accuracy. Ask which specific actions the model gets wrong. Does the Transformer help more on left/right than forward? Why?
- **Architecture tradeoff questions to pose**:
  - Why do we use BCEWithLogitsLoss instead of CrossEntropyLoss? (multi-label vs single-label)
  - Why is the CNN backbone frozen across experiments instead of being jointly trained with the Transformer?
  - When does CVAE hurt rather than help? (Exp 8→9 shows this)
  - Why chunk_size=10 and not 1 or 100?

---

## Per-Experiment Checklist

After each experiment trains, prompt the user to:

1. Fill in `notes.md` inside the experiment folder (template at `2_week_simulator/notes_template.md`)
2. Compare val accuracy to the reference in `experiment_progression.md`
3. Run one deliberate break from the debugging list above
4. Trace one sample end-to-end in a notebook
5. Write one architecture question raised by this experiment

---

## Reading List (in order)

Suggest these after the corresponding experiment trains:

| After | Read |
|-------|------|
| Exp 2 | ACT paper: "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (Zhao et al. 2023) |
| Exp 3 | VAE tutorial by Kingma & Welling (original CVAE paper is also short) |
| Exp 5 | Any intro to word embeddings (word2vec blog post is enough) |
| Exp 9 | RT-1 paper (Google, 2022) — they'll recognise every component |
| After all 9 | OpenVLA paper (2024) — the state of the art version of what they built |
