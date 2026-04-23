# World Model — Mathematical Foundations

---

## The 4 Pillars

---

## Pillar 1 — The Problem Statement (State Space Model)

The world at time `t` has some true state `s_t`. You never see `s_t` directly — you only see a noisy, high-dimensional **observation** `o_t` (a camera image).

You take an action `a_t`. The world transitions to a new state `s_{t+1}`.

Formally this is a **Markov Decision Process**:

```
p(s_{t+1} | s_t, a_t)     ← transition: next state depends only on current state + action
p(o_t | s_t)               ← observation: image is generated from state
p(r_t | s_t, a_t)          ← reward: depends on state and action
```

**The problem:** `s_t` is unknown. We observe `o_t` (128×128 pixels = 49,152 numbers). Working directly in pixel space is intractable.

**The solution:** Learn a compressed latent variable `z_t` that approximates `s_t`.

---

## Pillar 2 — Latent Variables and Gaussian Distributions

We model `z_t` as a **random variable** drawn from a probability distribution:

```
z_t ~ N(μ, σ²)
```

Where `N(μ, σ²)` is a Gaussian with mean `μ` and variance `σ²`.

**Why Gaussian?**
- Mathematically tractable
- KL divergence has a closed form (see Pillar 4)
- Smooth latent space — nearby `z` values produce similar observations

The encoder doesn't output a single point `z`. It outputs the **parameters of a distribution**:

```
Encoder(o_t) → (μ, log σ²)     shape: (256,) each

z_t = μ + σ · ε,    ε ~ N(0, 1)    ← reparameterization trick
```

**The reparameterization trick** is critical — it lets gradients flow through the sampling step during backpropagation. Without it, sampling is a non-differentiable operation and the encoder cannot be trained.

---

## Pillar 3 — The Transition Model (Core of the World Model)

Given current latent `z_t` and action embedding `h_a`, predict next latent:

```
z_{t+1} = f_θ(z_t, h_a)
```

Where `f_θ` is a neural network (MLP or GRU). This is a **deterministic** prediction in the simplest case.

More precisely, it predicts the **parameters of the next latent distribution**:

```
(μ_{t+1}, σ_{t+1}) = f_θ(z_t, h_a)

z_{t+1} ~ N(μ_{t+1}, σ_{t+1}²)
```

This captures **uncertainty** — the model says "I think the next state is around here, but I'm not totally sure." Uncertainty grows the further you predict into the future.

---

## Pillar 4 — The Loss Function (Evidence Lower Bound)

We want to maximize the **likelihood** of observing `o_{t+1}` given `o_t` and `a_t`:

```
maximize:  log p(o_{t+1} | o_t, a_t)
```

This integral is intractable (you'd have to sum over all possible `z`). So we maximize a **lower bound** called the **ELBO** (Evidence Lower Bound):

```
log p(o_{t+1}) ≥ E[log p(o_{t+1} | z_{t+1})]  -  KL( q(z_{t+1}) ∥ p(z_{t+1}) )
                 \_______________________/          \________________________________/
                      reconstruction term                  regularization term
```

### Term 1 — Reconstruction Loss

```
L_recon = MSE(ô_{t+1}, o_{t+1})
```

"How well does decoding `z_{t+1}` reproduce the real next frame?"

### Term 2 — KL Divergence

```
L_KL = KL( N(μ, σ²) ∥ N(0, 1) )

     = -½ Σ (1 + log σ² - μ² - σ²)
```

"How far is our learned latent distribution from a standard Normal?"

This keeps the latent space structured — prevents it from collapsing (all z become identical) or spreading randomly (z becomes meaningless). β controls how strongly this is enforced.

### Term 3 — Transition Consistency

```
L_trans = MSE( z_{t+1}^predicted,  z_{t+1}^encoded )
```

"Does the transition model's prediction match what the encoder finds when it actually sees `o_{t+1}`?"

This is the self-supervised signal that teaches the transition model to simulate dynamics.

### Term 4 — Reward Prediction

```
L_reward = MSE( r̂_{t+1},  r_{t+1} )
```

Weighted at 0.1 so it doesn't overpower the image reconstruction signal.

### Total Loss

```
L_total = L_recon + L_trans + 0.1 · L_reward + β · L_KL
```

---

## Full Pipeline Summary

```
o_t (image, shape: 3×128×128)
    ↓ CNN Encoder
(μ_t, σ_t)  shape: (256,) each
    ↓ reparameterize: z_t = μ + σ·ε
z_t  shape: (256,)

a_t (action, shape: 4,)
    ↓ Action Encoder MLP
h_a  shape: (64,)

[z_t, h_a]  shape: (320,)
    ↓ Transition Model f_θ
(μ_{t+1}, σ_{t+1})
    ↓ reparameterize
z_{t+1}  shape: (256,)

z_{t+1}
    ↓ CNN Decoder → ô_{t+1}   (L_recon)
    ↓ Reward MLP  → r̂_{t+1}  (L_reward)

z_{t+1}^predicted vs z_{t+1}^encoded  (L_trans)
KL( q(z) ∥ N(0,1) )                   (L_KL)
```

---

## Key Concepts to Understand Before Moving On

1. **Why a distribution over `z` rather than a single vector** — a single vector has no uncertainty. A distribution lets the model express "I'm not sure exactly where the state is."

2. **What the reparameterization trick does** — moves the randomness outside the gradient path: `z = μ + σ·ε` lets gradients flow through `μ` and `σ` while `ε` is just noise.

3. **What KL divergence measures geometrically** — the "distance" between two probability distributions. KL = 0 means they are identical. KL penalizes the encoder for drifting too far from a standard Normal prior.

4. **Why the ELBO is a lower bound** — because Jensen's inequality: `log E[x] ≥ E[log x]`. Maximizing the ELBO pushes up the true log-likelihood without computing the intractable integral.
