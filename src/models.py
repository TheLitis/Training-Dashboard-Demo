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


class ImprovedCNN(nn.Module):
    """Improved CNN with batch normalization and deeper feature extractor."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.35, base_width: int = 64) -> None:
        super().__init__()
        w1 = base_width
        w2 = base_width * 2
        w3 = base_width * 4

        self.features = nn.Sequential(
            nn.Conv2d(3, w1, kernel_size=3, padding=1),
            nn.BatchNorm2d(w1),
            nn.ReLU(inplace=True),
            nn.Conv2d(w1, w1, kernel_size=3, padding=1),
            nn.BatchNorm2d(w1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(w1, w2, kernel_size=3, padding=1),
            nn.BatchNorm2d(w2),
            nn.ReLU(inplace=True),
            nn.Conv2d(w2, w2, kernel_size=3, padding=1),
            nn.BatchNorm2d(w2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(w2, w3, kernel_size=3, padding=1),
            nn.BatchNorm2d(w3),
            nn.ReLU(inplace=True),
            nn.Conv2d(w3, w3, kernel_size=3, padding=1),
            nn.BatchNorm2d(w3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(w3, 256),
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
    if model_name == "improved":
        return ImprovedCNN(
            num_classes=num_classes,
            dropout=float(kwargs.get("dropout", 0.35)),
            base_width=int(kwargs.get("base_width", 64)),
        )
    raise ValueError(f"Unsupported model name: {name}")
