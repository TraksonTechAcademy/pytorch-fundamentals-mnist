from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass(frozen=True)
class MNISTDataConfig:
    data_dir: str
    batch_size: int = 128
    num_workers: int = 2


def make_loaders(cfg: MNISTDataConfig) -> Tuple[DataLoader, DataLoader]:
    """Create train/val loaders for MNIST.

    Note: torchvision will download MNIST on first run.
    """
    os.makedirs(cfg.data_dir, exist_ok=True)

    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_ds = datasets.MNIST(root=cfg.data_dir, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root=cfg.data_dir, train=False, download=True, transform=tfm)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
