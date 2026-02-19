"""Plotting helpers for training/evaluation artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_training_curves(history_path: str | Path, output_path: str | Path, run_name: str) -> None:
    """Plot train/val loss and accuracy curves from history CSV."""
    history = pd.read_csv(history_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = history["epoch"]

    axes[0].plot(epochs, history["train_loss"], label="train_loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="val_loss", linewidth=2)
    axes[0].set_title(f"{run_name}: Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train_acc", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], label="val_acc", linewidth=2)
    axes[1].set_title(f"{run_name}: Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_confusion_matrix(
    matrix,
    class_names: list[str],
    output_path: str | Path,
    run_name: str,
) -> None:
    """Plot confusion matrix heatmap."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix,
        cmap="Blues",
        annot=False,
        square=True,
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title(f"{run_name}: Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

