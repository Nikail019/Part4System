"""PyTorch dataset for voxelised machining feature recognition."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, random_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.feature_net import FEATURE_NAMES, FEATURE_TO_IDX, NUM_CLASSES
from phase1_voxeliser import process_step_file


class MachiningFeatureDataset(Dataset):
    """Loads voxel tensors and multi-hot feature labels."""

    def __init__(self, root, resolution: int = 64, transform=None, cache: bool = True):
        self.root = Path(root)
        self.resolution = resolution
        self.transform = transform
        self.cache = cache
        self.samples = self._scan_root()

    def _scan_root(self) -> list[dict]:
        if not self.root.exists():
            return []

        samples = []
        for part_dir in sorted(path for path in self.root.iterdir() if path.is_dir()):
            step_path = part_dir / "part.stp"
            labels_path = part_dir / "labels.json"
            if not step_path.exists() or not labels_path.exists():
                continue
            try:
                with labels_path.open(encoding="utf-8") as f:
                    labels = json.load(f).get("labels", [])
            except (OSError, json.JSONDecodeError):
                continue
            valid_labels = [label for label in labels if label in FEATURE_TO_IDX]
            if valid_labels:
                samples.append(
                    {"step_path": step_path, "labels": valid_labels, "part_dir": part_dir}
                )
        return samples

    def _get_voxel(self, sample: dict) -> np.ndarray:
        voxel_path = sample["part_dir"] / f"voxel_{self.resolution}.npy"
        if self.cache and voxel_path.exists():
            return np.load(voxel_path)

        out_dir = sample["part_dir"] if self.cache else sample["part_dir"] / "_voxel_tmp"
        metadata = process_step_file(str(sample["step_path"]), str(out_dir), resolution=self.resolution)
        return np.load(metadata["voxel_file"])

    def _labels_to_multihot(self, labels: list[str]) -> np.ndarray:
        y = np.zeros(NUM_CLASSES, dtype=np.float32)
        for label in labels:
            if label in FEATURE_TO_IDX:
                y[FEATURE_TO_IDX[label]] = 1.0
        return y

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        voxel = self._get_voxel(sample).astype(np.float32)
        x = torch.from_numpy(voxel).unsqueeze(0)
        if self.transform is not None:
            x = self.transform(x)
        y = torch.from_numpy(self._labels_to_multihot(sample["labels"]))
        return x, y


def random_split_dataset(dataset, train_frac: float = 0.80, val_frac: float = 0.10, seed: int = 42):
    """Return train, validation, and test subsets."""
    total = len(dataset)
    train_size = int(total * train_frac)
    val_size = int(total * val_frac)
    test_size = total - train_size - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size, test_size], generator=generator)


def compute_class_weights(dataset) -> torch.Tensor:
    """Compute normalized inverse-frequency positive weights per class."""
    counts = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    total = len(dataset)
    if total == 0:
        return torch.ones(NUM_CLASSES, dtype=torch.float32)

    if hasattr(dataset, "samples"):
        for sample in dataset.samples:
            for label in sample.get("labels", []):
                if label in FEATURE_TO_IDX:
                    counts[FEATURE_TO_IDX[label]] += 1.0
    else:
        for _, y in dataset:
            counts += y.float()

    pos_freq = counts / float(total)
    weights = 1.0 / (pos_freq + 1e-6)
    weights = weights / weights.mean()
    return weights.float()
