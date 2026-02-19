"""Model forward-pass smoke tests."""

from __future__ import annotations

import torch

from src.models import create_model


def test_baseline_forward_shape():
    model = create_model("baseline", num_classes=10, dropout=0.25)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 10)


def test_improved_forward_shape():
    model = create_model("improved", num_classes=10, dropout=0.35, base_width=32)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 10)

