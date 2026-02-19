"""Training entrypoint."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from src.config import load_config
from src.data import create_cifar10_loaders
from src.models import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train model on CIFAR-10.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower().strip()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = torch.argmax(logits, dim=1)
    return (predictions == targets).float().mean().item()


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += accuracy_from_logits(logits, labels)

    total_batches = max(len(loader), 1)
    return running_loss / total_batches, running_acc / total_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_acc = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        running_loss += loss.item()
        running_acc += accuracy_from_logits(logits, labels)

    total_batches = max(len(loader), 1)
    return running_loss / total_batches, running_acc / total_batches


def save_history(history_rows: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"],
        )
        writer.writeheader()
        writer.writerows(history_rows)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    run_name = str(config.get("run_name", "baseline"))
    device = resolve_device(str(config.get("device", "cuda")))
    print(f"Training run: {run_name}")
    print(f"Using device: {device}")

    train_loader, val_loader, _, classes = create_cifar10_loaders(config)
    num_classes = int(config["data"].get("num_classes", len(classes)))

    model_cfg = config.get("model", {})
    model = create_model(
        model_cfg.get("name", "baseline"),
        num_classes=num_classes,
        dropout=model_cfg.get("dropout", 0.25),
    ).to(device)

    train_cfg = config.get("train", {})
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    epochs = int(train_cfg.get("epochs", 5))
    runs_dir = Path(config.get("paths", {}).get("runs_dir", "artifacts/runs"))
    checkpoints_dir = Path(config.get("paths", {}).get("checkpoints_dir", "checkpoints"))
    runs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    history_path = runs_dir / run_name / "history.csv"
    summary_path = runs_dir / run_name / "metrics.json"
    checkpoint_path = checkpoints_dir / f"{run_name}_best.pt"
    (runs_dir / run_name).mkdir(parents=True, exist_ok=True)

    best_val_acc = -1.0
    history_rows: List[Dict] = []
    started_at = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "lr": round(float(lr), 8),
        }
        history_rows.append(row)
        print(
            f"[{epoch}/{epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "run_name": run_name,
                    "model_name": model_cfg.get("name", "baseline"),
                    "num_classes": num_classes,
                    "class_names": classes,
                    "best_val_acc": best_val_acc,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                },
                checkpoint_path,
            )

    duration = time.time() - started_at
    save_history(history_rows, history_path)
    summary = {
        "run_name": run_name,
        "model_name": model_cfg.get("name", "baseline"),
        "device": str(device),
        "epochs": epochs,
        "best_val_acc": round(best_val_acc, 6),
        "training_seconds": round(duration, 3),
        "history_path": str(history_path),
        "checkpoint_path": str(checkpoint_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved history to {history_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved best checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()

