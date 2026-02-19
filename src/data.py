"""Dataset and dataloader utilities."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Subset
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
    if isinstance(dataset, Subset):
        clipped_indices = list(dataset.indices)[:max_samples]
        return Subset(dataset.dataset, clipped_indices)
    return Subset(dataset, list(range(max_samples)))


def _build_transforms(mean, std, train_augmentation: bool):
    train_ops = []
    if train_augmentation:
        train_ops.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )

    train_ops.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    eval_ops = [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


def create_cifar10_loaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Create train/val/test loaders for CIFAR-10."""
    data_cfg = config["data"]
    seed = int(config.get("seed", 42))

    mean = data_cfg.get("normalize_mean", [0.4914, 0.4822, 0.4465])
    std = data_cfg.get("normalize_std", [0.2470, 0.2435, 0.2616])
    root = data_cfg.get("root", "data")
    batch_size = int(data_cfg.get("batch_size", 128))
    val_split = float(data_cfg.get("val_split", 0.1))
    train_augmentation = bool(data_cfg.get("train_augmentation", False))

    num_workers = int(config.get("num_workers", 2))
    pin_memory = bool(config.get("pin_memory", True))
    persistent_workers = num_workers > 0

    train_transform, eval_transform = _build_transforms(mean, std, train_augmentation)

    classes = list(datasets.CIFAR10(root=root, train=True, download=True).classes)
    train_dataset_for_fit = datasets.CIFAR10(root=root, train=True, download=True, transform=train_transform)
    train_dataset_for_eval = datasets.CIFAR10(root=root, train=True, download=True, transform=eval_transform)
    test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=eval_transform)

    total_train = len(train_dataset_for_fit)
    val_size = int(total_train * val_split)
    train_size = total_train - val_size

    split_generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_train, generator=split_generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(train_dataset_for_fit, train_indices)
    val_dataset = Subset(train_dataset_for_eval, val_indices)

    train_dataset = _clip_subset(train_dataset, _to_int(data_cfg.get("max_train_samples")))
    val_dataset = _clip_subset(val_dataset, _to_int(data_cfg.get("max_val_samples")))
    test_dataset = _clip_subset(test_dataset, _to_int(data_cfg.get("max_test_samples")))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader, test_loader, classes
