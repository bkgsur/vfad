"""
Base Trainer used by all 9 experiments.

Responsibilities:
- Run train epoch: forward pass, loss, backward, optimizer step
- Run val epoch: forward pass, loss (no grad)
- fit(): outer loop over epochs, call callbacks, log metrics
- Accepts any model + DataLoader pair — experiment-agnostic
"""
