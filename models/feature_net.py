"""3D CNN model for machining feature recognition."""

from __future__ import annotations

import os

import torch
from torch import nn


NUM_CLASSES = 12
FEATURE_NAMES = [
    "through_hole",
    "blind_hole",
    "rectangular_pocket",
    "circular_pocket",
    "rectangular_slot",
    "circular_slot",
    "rectangular_step",
    "chamfer",
    "fillet",
    "boss",
    "flat_face",
    "triangular_pocket",
]
FEATURE_TO_IDX = {name: idx for idx, name in enumerate(FEATURE_NAMES)}


class FeatureNet3D(nn.Module):
    """3D CNN for multi-label machining feature recognition."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.5,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((4, 4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4 * 4, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def load_model(
    checkpoint_path: str,
    num_classes: int = NUM_CLASSES,
    device: str | torch.device = "cpu",
) -> FeatureNet3D:
    """Load a FeatureNet3D checkpoint and return an eval-mode model."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    config = checkpoint.get("training_config", {})
    hidden_dim = int(config.get("hidden_dim", state_dict["classifier.1.weight"].shape[0]))
    model = FeatureNet3D(num_classes=num_classes, hidden_dim=hidden_dim)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
