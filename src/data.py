"""Dataset and dataloader utilities."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


def _to_int(value: int | None) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        return None
    return parsed


def _clip_subset(dataset, max_samples: int | None):
    if max_samples is None:
        return dataset
    max_samples = min(max_samples, len(dataset))
    return Subset(dataset, list(range(max_samples)))


def create_cifar10_loaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Create train/val/test loaders for CIFAR-10."""
    data_cfg = config["data"]
    seed = int(config.get("seed", 42))

    mean = data_cfg.get("normalize_mean", [0.4914, 0.4822, 0.4465])
    std = data_cfg.get("normalize_std", [0.2470, 0.2435, 0.2616])
    root = data_cfg.get("root", "data")
    batch_size = int(data_cfg.get("batch_size", 128))
    val_split = float(data_cfg.get("val_split", 0.1))
    num_workers = int(config.get("num_workers", 2))
    pin_memory = bool(config.get("pin_memory", True))

    common_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    train_full = datasets.CIFAR10(root=root, train=True, download=True, transform=common_transform)
    test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=common_transform)
    classes = list(train_full.classes)

    val_size = int(len(train_full) * val_split)
    train_size = len(train_full) - val_size
    split_generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(train_full, [train_size, val_size], generator=split_generator)

    train_dataset = _clip_subset(train_dataset, _to_int(data_cfg.get("max_train_samples")))
    val_dataset = _clip_subset(val_dataset, _to_int(data_cfg.get("max_val_samples")))
    test_dataset = _clip_subset(test_dataset, _to_int(data_cfg.get("max_test_samples")))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader, classes

