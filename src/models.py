"""Model definitions."""

from __future__ import annotations

import torch.nn as nn


class BaselineCNN(nn.Module):
    """Baseline CNN for CIFAR-10."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.25) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def create_model(name: str, num_classes: int, **kwargs):
    """Create model by name."""
    model_name = name.lower().strip()
    if model_name == "baseline":
        return BaselineCNN(num_classes=num_classes, dropout=float(kwargs.get("dropout", 0.25)))
    raise ValueError(f"Unsupported model name: {name}")

