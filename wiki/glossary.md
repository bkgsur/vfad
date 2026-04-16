# Glossary

Quick-lookup definitions for terms used across the VFAD project.
For full explanations follow the links to concept pages.

---

| Term | Definition |
|------|-----------|
| **action chunking** | Predicting a sequence of N future actions at once instead of one action per frame. Produces smoother, temporally consistent behaviour. See [Transformer](concepts/transformer.md). |
| **BCEWithLogitsLoss** | Binary Cross-Entropy loss with sigmoid fused in. Used for multi-label binary classification. See [BCELoss](concepts/bce_loss.md). |
| **CVAE** | Conditional Variational Autoencoder. Models a distribution over outputs rather than a single deterministic output. See [CVAE](concepts/cvae.md). |
| **chunk_size** | Number of future action steps predicted in one forward pass by the Transformer decoder. |
| **condition** | The input that the CVAE decoder is conditioned on — in this project, CNN features (+ language embedding). |
| **cross-attention** | Attention where Q comes from one sequence and K/V from another. Used in Transformer decoder. See [Transformer](concepts/transformer.md). |
| **d_model** | Internal dimension of the Transformer — the width of every token embedding. |
| **feature map** | The output of a single Conv2d filter applied to an image — a 2D grid of activations detecting one pattern. |
| **KL divergence** | Measures how far one probability distribution is from another. Used in CVAE loss. See [CVAE](concepts/cvae.md). |
| **language conditioning** | Using a language instruction to change the model's output. Implemented via nn.Embedding. See [LanguageEncoder](concepts/language_encoder.md). |
| **latent variable z** | A small sampled vector that encodes which specific valid output to produce. Core to CVAE. |
| **logit** | Raw model output before sigmoid/softmax. BCEWithLogitsLoss expects logits, not probabilities. |
| **multi-head attention** | Running attention in parallel across h subspaces (heads), then concatenating. See [Transformer](concepts/transformer.md). |
| **nn.Embedding** | A learnable lookup table mapping integer IDs to dense vectors. See [LanguageEncoder](concepts/language_encoder.md). |
| **positional encoding** | Sinusoidal vectors added to token embeddings to give the Transformer a sense of sequence position. |
| **reparameterisation trick** | Sampling z = mu + epsilon × sigma so gradients flow through mu and sigma. See [CVAE](concepts/cvae.md). |
| **self-attention** | Each token in a sequence attends to every other token. Core Transformer operation. |
| **state_dict** | PyTorch's format for saving model weights — an OrderedDict of parameter name → tensor. |
| **VLA** | Vision-Language-Action model. Takes image + language instruction → action. The goal architecture of this project. |
| **stride** | Step size of a Conv2d filter. stride=2 halves the spatial dimensions of the output. |
