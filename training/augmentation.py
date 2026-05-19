"""Voxel augmentation utilities for feature recognition training."""

from __future__ import annotations

import random

import torch


class RandomRotate90(object):
    """Randomly rotate a voxel tensor by 0, 90, 180, or 270 degrees around Z."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        k = random.randint(0, 3)
        if k == 0:
            return x
        return torch.rot90(x, k=k, dims=[1, 2])


class RandomFlip(object):
    """Randomly flip a voxel tensor along each spatial axis with probability p."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for dim in [1, 2, 3]:
            if random.random() < self.p:
                x = torch.flip(x, dims=[dim])
        return x


class RandomVoxelNoise(object):
    """Randomly flip a small fraction of voxel values."""

    def __init__(self, p: float = 0.01):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0:
            return x
        noise_mask = torch.rand_like(x) < self.p
        return torch.where(noise_mask, 1.0 - x, x)


class Compose(object):
    """Apply a list of transforms in sequence."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            x = transform(x)
        return x


def get_train_transforms() -> Compose:
    """Standard augmentation pipeline for training."""
    return Compose(
        [
            RandomRotate90(),
            RandomFlip(p=0.5),
            RandomVoxelNoise(p=0.005),
        ]
    )


def get_val_transforms() -> None:
    """No augmentation at validation or inference time."""
    return None
