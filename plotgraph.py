"""
plot_history.py

Plots training_history.csv (produced by main.py) as a set of curves:
loss (train vs val), accuracy, and learning rate over epochs.

Saves a PNG instead of calling plt.show() -- this box is headless
(no display server), so .show() would just hang.

Usage:
    python3 plot_history.py
    -> writes training_curves.png in the current directory
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe backend, must be set before pyplot import
import matplotlib.pyplot as plt

HISTORY_PATH = "final_history_training.csv"
OUTPUT_PATH = "training_curves.png"


def main():
    df = pd.read_csv(HISTORY_PATH)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # --- Loss ---
    ax = axes[0]
    ax.plot(df["epoch"], df["train_loss"], marker="o", label="Train Loss")
    ax.plot(df["epoch"], df["val_loss"], marker="o", label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Accuracy ---
    ax = axes[1]
    ax.plot(df["epoch"], df["accuracy"], marker="o", color="green")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Token Accuracy")
    ax.grid(True, alpha=0.3)

    # --- Learning rate ---
    ax = axes[2]
    ax.plot(df["epoch"], df["lr"], marker="o", color="orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("LR Schedule")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Training History", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
