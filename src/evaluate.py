"""Model evaluation entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn

from src.config import load_config
from src.data import create_cifar10_loaders
from src.models import create_model
from src.plotting import plot_confusion_matrix, plot_training_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model on CIFAR-10 test set.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--weights", required=True, help="Path to checkpoint weights.")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower().strip()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


@torch.no_grad()
def evaluate_test(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    all_preds = []
    all_targets = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean().item()

        running_loss += loss.item()
        running_acc += acc
        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(labels.cpu().numpy().tolist())

    total_batches = max(len(loader), 1)
    return (
        running_loss / total_batches,
        running_acc / total_batches,
        np.asarray(all_targets, dtype=np.int64),
        np.asarray(all_preds, dtype=np.int64),
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    checkpoint_path = Path(args.weights)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    run_name = str(config.get("run_name", checkpoint.get("run_name", "run")))
    device = resolve_device(str(config.get("device", "cuda")))

    _, _, test_loader, classes = create_cifar10_loaders(config)
    model_name = str(checkpoint.get("model_name", config.get("model", {}).get("name", "baseline")))

    model_cfg = config.get("model", {})
    model_kwargs = {key: value for key, value in model_cfg.items() if key != "name"}
    model = create_model(model_name, num_classes=len(classes), **model_kwargs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, targets, preds = evaluate_test(model, test_loader, criterion, device)
    matrix = confusion_matrix(targets, preds, labels=list(range(len(classes))))

    plots_dir = Path("artifacts/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = Path(config.get("paths", {}).get("runs_dir", "artifacts/runs"))

    curves_path = plots_dir / f"{run_name}_curves.png"
    confusion_path = plots_dir / f"confusion_{run_name}.png"
    history_path = runs_dir / run_name / "history.csv"

    if history_path.exists():
        plot_training_curves(history_path, curves_path, run_name)
    else:
        print(f"History file not found, skipping curves: {history_path}")
    plot_confusion_matrix(matrix, classes, confusion_path, run_name)

    eval_summary = {
        "run_name": run_name,
        "model_name": model_name,
        "weights": str(checkpoint_path),
        "test_loss": round(float(test_loss), 6),
        "test_acc": round(float(test_acc), 6),
        "curves_path": str(curves_path),
        "confusion_path": str(confusion_path),
    }
    eval_summary_path = runs_dir / run_name / "eval.json"
    eval_summary_path.parent.mkdir(parents=True, exist_ok=True)
    eval_summary_path.write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")

    print(f"Run: {run_name}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test acc: {test_acc:.4f}")
    print(f"Saved eval summary: {eval_summary_path}")
    print(f"Saved confusion matrix: {confusion_path}")
    if history_path.exists():
        print(f"Saved curves: {curves_path}")


if __name__ == "__main__":
    main()

