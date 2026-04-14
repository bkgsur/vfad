"""
Training visualisation utilities.

# ---------------------------------------------------------------------------
# WHAT PROBLEM DOES THIS SOLVE?
# ---------------------------------------------------------------------------
# After training you have numbers — loss going down, accuracy going up —
# but numbers alone are hard to interpret quickly.
#
# Plots let you instantly see:
#   - Is the model still learning, or has it plateaued?
#   - Is val loss rising while train loss falls? (= overfitting)
#   - Which specific actions is the model struggling with?
#
# This module is completely decoupled from training — it only reads a
# metrics.json file. You can re-plot any past run without retraining.

# ---------------------------------------------------------------------------
# KEY CONCEPTS YOU WILL LEARN HERE
# ---------------------------------------------------------------------------
# 1. matplotlib — Python's standard plotting library.
#    plt.plot(x, y) draws a line. plt.bar(labels, heights) draws bars.
#    plt.savefig() writes the figure to disk as an image file.
#
# 2. Subplots — multiple panels inside one figure.
#    fig, (ax1, ax2) = plt.subplots(1, 2) creates two side-by-side panels.
#    Each ax is an independent plot you control separately.
#
# 3. WHY savefig, NOT plt.show()?
#    Training runs on a server with no screen. plt.show() would crash or
#    block execution waiting for a window. Always save to disk instead.
#
# 4. WHY close the figure after saving?
#    matplotlib keeps every figure in memory until closed. Across many
#    training runs, unclosed figures would exhaust RAM silently.

# ---------------------------------------------------------------------------
# WHAT THIS MODULE PRODUCES
# ---------------------------------------------------------------------------
# curves.png     — train loss + val loss + aggregate val accuracy over epochs.
#                  Both loss lines should go down together.
#                  Val loss rising while train loss falls = overfitting.
#
# per_action.png — one bar per action (forward, backward, left, right).
#                  This reveals class imbalance: forward and backward are
#                  trivially near 100%. left and right show real learning.

# ---------------------------------------------------------------------------
# WHERE DOES THE DATA COME FROM?
# ---------------------------------------------------------------------------
# The trainer (shared/training/trainer.py) writes a metrics.json file into
# results_dir after each epoch. This module reads that file.
#
# metrics.json structure:
# {
#   "train_loss":       [0.5, 0.4, ...],   one value per epoch
#   "val_loss":         [0.6, 0.5, ...],
#   "val_acc":          [0.7, 0.8, ...],   aggregate across all 4 actions
#   "val_acc_forward":  [...],
#   "val_acc_backward": [...],
#   "val_acc_left":     [...],
#   "val_acc_right":    [...]
# }

# ---------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------
#   from shared.utils.visualize import plot_curves, plot_per_action_accuracy
#   plot_curves("results/curved_road/exp1")
#   plot_per_action_accuracy("results/curved_road/exp1")
"""

from __future__ import annotations

import json

# Path for all file operations — avoids messy string joins like "dir" + "/" + "file.png"
from pathlib import Path

# matplotlib.pyplot is the main plotting interface.
# Convention: always imported as `plt`.
import matplotlib.pyplot as plt


# The trainer saves metrics in this filename inside results_dir.
# Defining it as a constant here means if we ever rename it, we change one line.
METRICS_FILE = "metrics.json"


def _load_metrics(results_dir: str | Path) -> dict:
    """Read metrics.json from results_dir and return as a dict."""
    # Path(x) / "file" is the clean way to join paths — works on Windows and Linux.
    path = Path(results_dir) / METRICS_FILE

    if not path.exists():
        raise FileNotFoundError(
            f"No metrics file found at {path}. Has training been run yet?"
        )

    # json.loads parses a JSON string into a Python dict.
    # path.read_text() reads the whole file as a string in one call.
    return json.loads(path.read_text())


def plot_curves(results_dir: str | Path) -> None:
    """
    Plot train loss, val loss, and aggregate val accuracy over epochs.
    Saves curves.png into results_dir.
    """
    metrics = _load_metrics(results_dir)

    # range(1, n+1) gives epoch numbers starting at 1, which is more readable than 0-indexed.
    epochs = range(1, len(metrics["train_loss"]) + 1)

    # plt.subplots(1, 2) creates ONE figure with TWO side-by-side panels (axes).
    # figsize=(12, 4) sets the width and height in inches.
    # The two axes are unpacked directly into ax_loss and ax_acc.
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    # --- Left panel: loss curves ---
    # ax.plot(x, y, label="...") draws a line. The label appears in the legend.
    ax_loss.plot(epochs, metrics["train_loss"], label="train loss")
    ax_loss.plot(epochs, metrics["val_loss"],   label="val loss")

    # A good model shows both lines going down together.
    # If val loss rises while train loss falls → overfitting.
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("BCE Loss")

    # ax.legend() draws the box showing which colour = which line.
    ax_loss.legend()

    # ax.grid() adds faint background gridlines — makes it easier to read values.
    ax_loss.grid(True)

    # --- Right panel: accuracy curve ---
    ax_acc.plot(epochs, metrics["val_acc"], label="val accuracy", color="green")
    ax_acc.set_title("Validation Accuracy (aggregate)")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")

    # ylim clamps the y-axis to [0, 1] so the scale is always consistent across runs.
    ax_acc.set_ylim(0, 1)
    ax_acc.legend()
    ax_acc.grid(True)

    # tight_layout() adjusts spacing so panel titles and labels don't overlap.
    fig.tight_layout()

    # Save to disk — never use plt.show() in training code.
    # plt.show() blocks execution and requires a display (crashes on servers).
    out = Path(results_dir) / "curves.png"
    fig.savefig(out, dpi=120)

    # Always close the figure after saving.
    # If you don't, matplotlib keeps it in memory — running many experiments would
    # eventually exhaust RAM from accumulated open figures.
    plt.close(fig)

    print(f"Saved {out}")


def plot_per_action_accuracy(results_dir: str | Path) -> None:
    """
    Plot final validation accuracy as a bar chart, one bar per action.
    Saves per_action.png into results_dir.

    This plot matters because aggregate accuracy hides class imbalance:
    'forward' is almost always 1, 'backward' almost always 0, so the model
    gets those right trivially. 'left' and 'right' reveal whether the model
    is actually learning to steer.
    """
    metrics = _load_metrics(results_dir)

    # The four action names — order matches the model's output: [fwd, bwd, left, right]
    actions = ["forward", "backward", "left", "right"]

    # Take the LAST epoch's per-action accuracy values (index -1 = final epoch).
    # These keys are written by shared/training/metrics.py.
    final_acc = [metrics[f"val_acc_{a}"][-1] for a in actions]

    fig, ax = plt.subplots(figsize=(6, 4))

    # ax.bar(x_labels, heights) draws a bar chart.
    bars = ax.bar(actions, final_acc, color=["steelblue", "salmon", "orange", "green"])

    ax.set_title("Final Val Accuracy per Action")
    ax.set_ylabel("Accuracy")

    # Fix y-axis at [0, 1] so bars are always on the same scale.
    ax.set_ylim(0, 1)

    # Draw a dashed reference line at 0.9 — our target accuracy for Exp 1.
    # This makes it immediately obvious whether each action has reached the goal.
    ax.axhline(0.9, linestyle="--", color="grey", linewidth=0.8, label="target 0.90")
    ax.legend()

    # Add the numeric value above each bar so you don't have to estimate from the axis.
    for bar, acc in zip(bars, final_acc):
        # bar.get_x() + bar.get_width() / 2 finds the horizontal centre of the bar.
        # bar.get_height() is the top of the bar (the accuracy value).
        # ha="center", va="bottom" aligns the text centred above the bar.
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()

    out = Path(results_dir) / "per_action.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)

    print(f"Saved {out}")
