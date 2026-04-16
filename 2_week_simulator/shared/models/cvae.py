"""
Conditional Variational Autoencoder (CVAE) used in Exp 3, 7, 9.

Models the distribution over possible action sequences, allowing the model
to handle ambiguous situations where multiple valid action sequences exist
(e.g. two plausible trajectories through a forked road junction).

Loss: total_loss = BCE + beta * KL

# ---------------------------------------------------------------------------
# WHAT PROBLEM DOES THIS SOLVE?
# ---------------------------------------------------------------------------
# A deterministic model (CNN + MLP or CNN + Transformer) always produces the
# same output for the same input. On a forked road with two valid paths,
# the model must commit to one. If the training data contains both left-path
# and right-path trajectories for similar images, the model averages them —
# producing an action sequence that matches neither path well.
#
# A CVAE instead learns the DISTRIBUTION over valid action sequences.
# A small latent vector z is sampled from this distribution at inference time,
# allowing the model to produce different plausible outputs on different runs.
#
# In practice this means: given "turn left", the CVAE can model the range of
# valid left-turn trajectories rather than collapsing them to a single mean.

# ---------------------------------------------------------------------------
# KEY CONCEPTS YOU WILL LEARN HERE
# ---------------------------------------------------------------------------
# ── CONCEPT 1: THE LATENT VARIABLE z ─────────────────────────────────────
# z is a small vector (e.g. 16 numbers) that represents WHICH specific
# valid action sequence the model should produce.
#
# Think of it as a "style" knob: same input image + instruction, but
# different z → different plausible trajectories through the junction.
#
# The model learns:
#   Encoder: given (image, true_actions) → what z produced these actions?
#   Decoder: given (image, z) → what actions does this z imply?
#
# ── CONCEPT 2: TRAINING vs INFERENCE ─────────────────────────────────────
# TRAINING (both image and true actions are available):
#   Encoder(image, true_actions) → mu, logvar   ← "posterior"
#   z = reparameterise(mu, logvar)              ← sample from posterior
#   Decoder(image, z) → predicted_actions
#   Loss = BCE(predicted, true) + beta * KL(posterior || N(0,1))
#
# INFERENCE (only image is available — no true actions):
#   z ~ N(0, I)                    ← sample from PRIOR (standard normal)
#   Decoder(image, z) → actions    ← same decoder as training
#
# The KL term in training pushes the posterior toward N(0,I) so that
# sampling from N(0,I) at inference gives valid, useful z values.
#
# ── CONCEPT 3: THE REPARAMETERISATION TRICK ──────────────────────────────
# We cannot backpropagate through a random sample — sampling breaks the
# computation graph.
#
# WRONG (no gradient):
#   z ~ N(mu, exp(logvar))         ← z is a random draw, not a function of mu/logvar
#
# RIGHT (reparameterisation):
#   epsilon ~ N(0, 1)              ← random, but NOT a parameter — no grad needed
#   z = mu + epsilon * exp(0.5 * logvar)  ← now z IS a function of mu and logvar
#
# Now gradients flow back through z → mu and logvar → encoder weights.
# epsilon is just external noise injected into a deterministic computation.
#
# Shape walkthrough (B=2, latent_dim=16):
#   mu      : (2, 16)   ← encoder output, mean of posterior
#   logvar  : (2, 16)   ← encoder output, log-variance of posterior
#   epsilon : (2, 16)   ← sampled from N(0,1), same shape
#   z       : (2, 16)   ← mu + epsilon * exp(0.5 * logvar)
#
# ── CONCEPT 4: THE ELBO LOSS (BCE + β×KL) ────────────────────────────────
# Two terms, pulled in opposite directions:
#
# 1. RECONSTRUCTION LOSS (BCE):
#    "Given z, can you decode back to the correct actions?"
#    This pushes the encoder to produce z values that are useful for decoding.
#    Shape: criterion(logits, labels) → scalar
#
# 2. KL DIVERGENCE:
#    "How far is the posterior N(mu, sigma²) from the prior N(0,1)?"
#    This pushes the posterior toward the prior so inference-time sampling works.
#    Closed-form formula (no sampling needed):
#
#      KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
#
#    Derivation intuition: if mu=0 and logvar=0 (i.e. sigma=1), this is exactly 0.
#    Any deviation from the standard normal increases KL.
#    We sum over the latent_dim dimension and mean over the batch:
#      KL.shape during computation: (B, latent_dim) → reduced to scalar
#
# 3. β COEFFICIENT:
#    total_loss = BCE + beta * KL
#    beta=1.0  — standard VAE, equal weight
#    beta<1.0  — favours reconstruction; z may encode arbitrary structure
#    beta>1.0  — favours a clean latent space; reconstruction may suffer
#    In this project beta=1.0 is a reasonable default.
#
# ── CONCEPT 5: WHEN DOES CVAE HELP vs HURT? ──────────────────────────────
# Looking at experiment_progression.md:
#   Exp 6 (ACT, no CVAE):          93.5%
#   Exp 7 (ACT + CVAE):            93.3%  ← slight drop
#   Exp 8 (ACT + Language):        94.0%
#   Exp 9 (ACT + Language + CVAE): 92.1%  ← larger drop
#
# Insight: when language ALREADY resolves the ambiguity ("turn left" leaves
# no uncertainty about which path to take), the CVAE's latent variable z
# has nothing useful to model. It adds parameters and noise without benefit.
# CVAE helps when genuine ambiguity exists and language can't resolve it.

# ---------------------------------------------------------------------------
# STRUCTURE
# ---------------------------------------------------------------------------
#   CVAE
#     __init__(condition_dim, action_dim, latent_dim, hidden_dim, beta)
#     encode(condition, actions) → (mu, logvar)   ← used during TRAINING only
#     reparameterise(mu, logvar) → z              ← sample z from posterior
#     decode(condition, z)       → logits          ← used during TRAINING and INFERENCE
#     forward(condition, actions=None)
#       training:   encode → reparameterise → decode → return (logits, mu, logvar)
#       inference:  sample z from N(0,1) → decode → return (logits, None, None)
#
#   cvae_loss(logits, labels, mu, logvar, criterion, beta) → total_loss
#     BCE = criterion(logits, labels)
#     KL  = -0.5 * mean(sum(1 + logvar - mu^2 - exp(logvar), dim=1))
#     return BCE + beta * KL

# ---------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------
#   from shared.models.cvae import CVAE, cvae_loss
#   import torch, torch.nn as nn
#
#   # condition_dim = CNN output dim (+ language dim if applicable)
#   # action_dim    = chunk_size * num_actions (flattened action sequence)
#   cvae = CVAE(condition_dim=64, action_dim=40, latent_dim=16, hidden_dim=128)
#
#   # Training forward pass:
#   logits, mu, logvar = cvae(condition, actions)   # actions: (B, chunk_size, 4)
#   loss = cvae_loss(logits, labels, mu, logvar, criterion, beta=1.0)
#
#   # Inference forward pass (no actions provided):
#   logits, _, _ = cvae(condition)                  # z sampled from N(0,1)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CVAE(nn.Module):
    """Conditional Variational Autoencoder for action sequence modelling.

    Parameters
    ----------
    condition_dim : int
        Size of the conditioning vector (CNN features, optionally + language).
        This is the "given" context — the image (and instruction) the car sees.
    action_dim : int
        Flattened size of the action sequence.
        For chunk_size=10, num_actions=4: action_dim = 10 * 4 = 40.
    latent_dim : int
        Size of the latent variable z.
        Small (8–32) is typical. Larger = more capacity but harder to train.
    hidden_dim : int
        Hidden size used in both the encoder and decoder MLPs.
    beta : float
        Weight on the KL term in the loss. 1.0 = standard VAE.
    """

    def __init__(
        self,
        condition_dim: int,
        action_dim: int,
        latent_dim: int,
        hidden_dim: int,
        beta: float = 1.0,
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.beta = beta

        # ── Encoder ───────────────────────────────────────────────────
        # Input: condition concatenated with flattened true actions.
        # (B, condition_dim + action_dim) → (B, hidden_dim) → mu and logvar
        #
        # Why concatenate condition WITH actions?
        # The encoder's job is: "given what the car sees AND what it actually
        # did, what latent z best explains this behaviour?"
        # Both pieces of information are needed to infer z.
        #
        # Why output logvar instead of sigma directly?
        # logvar = log(sigma²). Taking log keeps the value unconstrained
        # (logvar can be any real number, while sigma must be positive).
        # This makes optimisation easier — no clamping or softplus needed.
        self.encoder_net = nn.Sequential(
            nn.Linear(condition_dim + action_dim, hidden_dim),   # (B, condition+action) → (B, hidden)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),                   # (B, hidden) → (B, hidden)
            nn.ReLU(),
        )
        # Two separate heads — one for mu, one for logvar.
        # They share the encoder_net trunk but have independent final layers.
        self.mu_head     = nn.Linear(hidden_dim, latent_dim)   # (B, hidden) → (B, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)   # (B, hidden) → (B, latent_dim)

        # ── Decoder ───────────────────────────────────────────────────
        # Input: condition concatenated with sampled z.
        # (B, condition_dim + latent_dim) → (B, action_dim) action logits
        #
        # Why concatenate condition WITH z?
        # The decoder must know both WHAT the car sees (condition) and
        # WHICH specific trajectory is intended (z) to reconstruct actions.
        self.decoder_net = nn.Sequential(
            nn.Linear(condition_dim + latent_dim, hidden_dim),   # (B, condition+latent) → (B, hidden)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),                   # (B, hidden) → (B, hidden)
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),                   # (B, hidden) → (B, action_dim)
            # No activation here — output is raw logits for BCEWithLogitsLoss
        )

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------

    def encode(
        self, condition: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode (condition, actions) into posterior parameters mu and logvar.

        Parameters
        ----------
        condition : (B, condition_dim)
        actions   : (B, action_dim)   — flattened true action sequence

        Returns
        -------
        mu     : (B, latent_dim)   mean of the posterior distribution
        logvar : (B, latent_dim)   log-variance of the posterior distribution
        """
        # Concatenate condition and actions along the feature dimension.
        # (B, condition_dim) + (B, action_dim) → (B, condition_dim + action_dim)
        x = torch.cat([condition, actions], dim=1)   # (B, condition_dim + action_dim)

        # Shared trunk: compress to hidden representation
        h = self.encoder_net(x)   # (B, hidden_dim)

        # Two independent heads produce mu and logvar from the same h
        mu     = self.mu_head(h)      # (B, latent_dim)
        logvar = self.logvar_head(h)  # (B, latent_dim)

        return mu, logvar

    # ------------------------------------------------------------------
    # Reparameterisation
    # ------------------------------------------------------------------

    def reparameterise(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Sample z from N(mu, exp(logvar)) using the reparameterisation trick.

        Parameters
        ----------
        mu     : (B, latent_dim)
        logvar : (B, latent_dim)

        Returns
        -------
        z : (B, latent_dim)
        """
        if self.training:
            # During training: sample z from the posterior.
            # The trick: z = mu + epsilon * sigma
            #   where epsilon ~ N(0,1) is independent noise (not a parameter).
            # This keeps z differentiable w.r.t. mu and logvar.
            #
            # logvar = log(sigma²)
            # exp(0.5 * logvar) = exp(0.5 * log(sigma²)) = exp(log(sigma)) = sigma
            std = torch.exp(0.5 * logvar)           # (B, latent_dim)  sigma = exp(0.5 * logvar)
            epsilon = torch.randn_like(std)          # (B, latent_dim)  sampled from N(0,1)
            # torch.randn_like creates a tensor of the same shape and device as std,
            # filled with samples from the standard normal distribution.
            z = mu + epsilon * std                  # (B, latent_dim)
        else:
            # During inference: no actions available, so no encoder output.
            # Use the mean of the posterior as the z value — deterministic,
            # zero-noise decoding. In a generative setting you'd sample here,
            # but for driving we want the most likely trajectory.
            z = mu                                  # (B, latent_dim)

        return z

    # ------------------------------------------------------------------
    # Decoder
    # ------------------------------------------------------------------

    def decode(
        self, condition: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Decode (condition, z) into action logits.

        Parameters
        ----------
        condition : (B, condition_dim)
        z         : (B, latent_dim)

        Returns
        -------
        logits : (B, action_dim)   raw action logits (no sigmoid)
        """
        # Concatenate condition and z along the feature dimension.
        # (B, condition_dim) + (B, latent_dim) → (B, condition_dim + latent_dim)
        x = torch.cat([condition, z], dim=1)   # (B, condition_dim + latent_dim)

        return self.decoder_net(x)   # (B, action_dim)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        condition: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Full CVAE forward pass.

        Training mode (actions provided):
            encode(condition, actions) → mu, logvar
            z = reparameterise(mu, logvar)
            logits = decode(condition, z)
            returns (logits, mu, logvar)   ← mu/logvar needed to compute KL loss

        Inference mode (actions=None):
            z ~ N(0, I)                    ← sample from prior
            logits = decode(condition, z)
            returns (logits, None, None)   ← no KL term needed at inference

        Parameters
        ----------
        condition : (B, condition_dim)   CNN features (+ language if applicable)
        actions   : (B, action_dim) or None

        Returns
        -------
        logits : (B, action_dim)
        mu     : (B, latent_dim) or None
        logvar : (B, latent_dim) or None
        """
        if actions is not None:
            # ── Training path ────────────────────────────────────────
            # We have the true actions — use the encoder to infer z.
            mu, logvar = self.encode(condition, actions)   # (B, latent_dim) each
            z = self.reparameterise(mu, logvar)            # (B, latent_dim)
            logits = self.decode(condition, z)             # (B, action_dim)
            return logits, mu, logvar

        else:
            # ── Inference path ───────────────────────────────────────
            # No true actions available — sample z from the prior N(0,I).
            # The KL training term has taught the encoder to stay close to
            # N(0,I), so samples from it should decode to valid actions.
            B = condition.size(0)
            z = torch.randn(B, self.latent_dim, device=condition.device)  # (B, latent_dim)
            logits = self.decode(condition, z)             # (B, action_dim)
            return logits, None, None


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def cvae_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    criterion: nn.Module,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute CVAE loss = BCE + beta * KL.

    Parameters
    ----------
    logits    : (B, action_dim)   model output — raw logits
    labels    : (B, action_dim)   ground truth — float 0/1
    mu        : (B, latent_dim)   encoder posterior mean
    logvar    : (B, latent_dim)   encoder posterior log-variance
    criterion : BCEWithLogitsLoss instance
    beta      : KL weight (1.0 = standard VAE)

    Returns
    -------
    total_loss : scalar  BCE + beta * KL
    bce        : scalar  reconstruction term alone (for logging)
    kl         : scalar  KL term alone (for logging)
    """
    # ── Reconstruction loss (BCE) ─────────────────────────────────────
    # How well does the decoded z reconstruct the true actions?
    # criterion = BCEWithLogitsLoss(), which applies sigmoid internally.
    # logits: (B, action_dim),  labels: (B, action_dim)  → scalar
    bce = criterion(logits, labels)   # scalar

    # ── KL divergence ─────────────────────────────────────────────────
    # KL(N(mu, sigma²) || N(0,1))
    #
    # Closed-form derivation:
    #   KL = -0.5 * sum_over_dims(1 + logvar - mu² - exp(logvar))
    #
    # Breaking it down term by term (per latent dimension):
    #   +1            → constant offset from the entropy of the unit Gaussian
    #   +logvar       → entropy of the posterior: high variance = high entropy
    #   -mu²          → penalises posterior mean drifting away from 0
    #   -exp(logvar)  → penalises posterior variance exceeding 1
    #
    # If mu=0 and logvar=0 (sigma=1): KL = -0.5*(1+0-0-1) = 0. ✓
    # Any deviation from N(0,1) increases KL.
    #
    # Shape walkthrough:
    #   1 + logvar - mu.pow(2) - logvar.exp()  →  (B, latent_dim)
    #   .sum(dim=1)                             →  (B,)   sum over latent dims
    #   .mean()                                 →  scalar  mean over batch
    #   * -0.5                                  →  scalar  KL value (always ≥ 0)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
    # kl shape: scalar

    # ── Total loss ────────────────────────────────────────────────────
    total_loss = bce + beta * kl   # scalar

    # Return all three so the experiment can log BCE and KL separately.
    # Watching them individually reveals if KL is collapsing (going to 0)
    # or if BCE is dominating and the latent space isn't being regularised.
    return total_loss, bce, kl
