# 9 Experiments: VLA Architecture Progression

Source: Experiments.png in 2_week_simulator/
All models trained from scratch on 128×128 images. No pre-training. Same training data per map type.

---

## Set 1: Curved Road (Experiments 1–3)

**Goal:** Teach the car to follow a curved road automatically.
**Challenge:** Single path, no choices — just smooth navigation.

| Exp | Model | Transformer | Language | CVAE | Val Accuracy |
|-----|-------|-------------|----------|------|--------------|
| 1   | Vision CNN (3-conv + MLP)       | No  | No  | No  | 90.0% |
| 2   | ACT (CNN + Transformer Enc/Dec) | Yes | No  | No  | 92.4% |
| 3   | ACT + CVAE                      | Yes | No  | Yes | 91.1% |

### Progression
```
Exp 1  →  Exp 2              →  Exp 3
CNN+MLP   CNN+Transformer(ACT)  CNN+Transformer+CVAE
```

| Exp | What you add | Why |
|-----|--------------|-----|
| 1   | Baseline CNN + MLP          | Can a simple model follow a curve? |
| 2   | Transformer Enc/Dec (ACT)   | Does action chunking smooth the driving? |
| 3   | CVAE                        | Can the model handle uncertainty in curved paths? |

---

## Set 2: Forked Road (Experiments 4–9)

**Goal:** Teach the car to navigate a junction and choose the correct path.
**Challenge:** Two valid paths exist — the car must understand which way to go (from language instructions).

| Exp | Model | Transformer | Language | CVAE | Val Accuracy |
|-----|-------|-------------|----------|------|--------------|
| 4   | Vision CNN (4-conv + MLP)       | No  | No  | No  | 91.7% |
| 5   | Vision CNN + Language Embedding | No  | Yes | No  | 91.7% |
| 6   | ACT                             | Yes | No  | No  | 93.5% |
| 7   | ACT + CVAE                      | Yes | No  | Yes | 93.3% |
| 8   | ACT + Language                  | Yes | Yes | No  | 94.0% |
| 9   | ACT + Language + CVAE           | Yes | Yes | Yes | 92.1% |

### Progression
```
Exp 4  →  Exp 5        →  Exp 6  →  Exp 7      →  Exp 8           →  Exp 9
4-conv     CNN+Language   ACT       ACT+CVAE       ACT+Language       ACT+Language+CVAE
CNN+MLP
```

| Exp | What you add | Why |
|-----|--------------|-----|
| 4   | Deeper CNN + MLP (4 conv)   | Does a 4th conv layer help on a harder map? |
| 5   | Language embedding          | Can "turn left" change the car's behaviour? |
| 6   | Transformer (ACT)           | Does action chunking help at a junction? |
| 7   | CVAE                        | Can the model handle the ambiguity of two paths? |
| 8   | ACT + Language              | Best combo — instruction-aware chunked actions |
| 9   | ACT + Language + CVAE       | Does CVAE still help when language already guides? |

---

## Key Insights (from Experiments.png)

1. **Transformers consistently help (+2–3%)**
   - Exp 1→2 and Exp 4→6 both improve by adding Transformer Enc/Dec
   - Action chunking (predicting a sequence of future actions) is why

2. **Language conditioning is cheap but valuable**
   - Exp 4→5 and Exp 6→8: language adds minimal parameters
   - Critical on forked road where the car must choose a direction from an instruction

3. **CVAE helps with ambiguity, but not always**
   - Forked road (Exp 6→7): CVAE helps because two valid paths exist
   - When language already resolves ambiguity (Exp 8→9), CVAE slightly hurts

---

## The Bridge Between Sets

- Exp 1–3 teach the **core building blocks** on a forgiving single-path track
- Exp 4–9 reuse those exact blocks on a harder problem where language and uncertainty handling become necessary
- The jump is not new ML theory — the **task gets harder**, revealing limits of simpler models

---

## Model Architecture: Experiment 1 (current focus)

File: `curved_road/1_cnn_mlp/model.json`
- Format: `vision_only_vla`
- Total parameters: **28,004**
- Input: 128×128 RGB image
- Output: 4 actions (forward, backward, left, right)

### Layer shapes
```
conv1:  (16, 3, 3, 3)   — 16 filters, 3-channel RGB input, 3×3 kernel
conv2:  (32, 16, 3, 3)  — 32 filters
conv3:  (64, 32, 3, 3)  — 64 filters
fc1:    (64, 64)         — 64 hidden neurons
fc2:    (64, 4)          — 4 output actions
```

### Data flow
```
128×128×3 image
    → Conv1 (16 filters) → ReLU → MaxPool
    → Conv2 (32 filters) → ReLU → MaxPool
    → Conv3 (64 filters) → ReLU → MaxPool
    → Flatten → 64 numbers
    → FC1 (64 neurons) → ReLU
    → FC2 (4 outputs) → Sigmoid
    → [forward, backward, left, right]
```

---

## Training Data: Curved Road

Location: `curved_road/1_cnn_mlp/training-data/`
- 38 JSON files, each ~390 samples (~14,820 total samples)
- Each sample: timestamp, base64 JPEG image, actions dict, speed float, steering float
- Single intent: "Drive"
- Action distribution: 100% forward, ~42% left, ~10% right, 0% backward
- Speed: near max (avg ~1.0)
- Steering: continuous float [-1, +1]
