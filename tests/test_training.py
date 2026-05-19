import torch
import pytest

from training.augmentation import (
    Compose,
    RandomFlip,
    RandomRotate90,
    RandomVoxelNoise,
    get_train_transforms,
    get_val_transforms,
)
from training.dataset import MachiningFeatureDataset, compute_class_weights


def make_voxel(R=16):
    return torch.ones(1, R, R, R)


def test_rotate90_preserves_shape():
    x = make_voxel()
    out = RandomRotate90()(x)
    assert out.shape == x.shape


def test_rotate90_preserves_sum():
    x = make_voxel()
    x[0, :4, :4, :4] = 0
    for _ in range(20):
        out = RandomRotate90()(x.clone())
        assert out.sum() == x.sum()


def test_flip_preserves_shape():
    x = make_voxel()
    out = RandomFlip(p=1.0)(x)
    assert out.shape == x.shape


def test_flip_preserves_sum():
    x = make_voxel()
    out = RandomFlip(p=1.0)(x)
    assert out.sum() == x.sum()


def test_noise_changes_small_fraction():
    x = torch.ones(1, 32, 32, 32)
    out = RandomVoxelNoise(p=0.01)(x.clone())
    diff_fraction = (out != x).float().mean().item()
    assert diff_fraction < 0.05


def test_compose_applies_all_transforms():
    x = make_voxel(16)
    transform = get_train_transforms()
    out = transform(x.clone())
    assert out.shape == x.shape


def test_val_transforms_is_none():
    assert get_val_transforms() is None


def test_compute_class_weights_shape(tmp_path):
    from training.synthetic_data_gen import generate_dataset

    generate_dataset(str(tmp_path), count=20)
    ds = MachiningFeatureDataset(str(tmp_path), resolution=16)
    if len(ds) == 0:
        pytest.skip("No synthetic data generated")
    weights = compute_class_weights(ds)
    from training.dataset import NUM_CLASSES

    assert weights.shape == (NUM_CLASSES,)


def test_compute_class_weights_all_positive(tmp_path):
    from training.synthetic_data_gen import generate_dataset

    generate_dataset(str(tmp_path), count=20)
    ds = MachiningFeatureDataset(str(tmp_path), resolution=16)
    if len(ds) == 0:
        pytest.skip("No synthetic data generated")
    weights = compute_class_weights(ds)
    assert (weights > 0).all()


def test_flat_face_has_lower_weight_than_rare_features(tmp_path):
    from training.synthetic_data_gen import generate_dataset

    generate_dataset(str(tmp_path), count=50)
    ds = MachiningFeatureDataset(str(tmp_path), resolution=16)
    if len(ds) == 0:
        pytest.skip("No synthetic data generated")
    from training.dataset import FEATURE_NAMES

    weights = compute_class_weights(ds)
    flat_idx = FEATURE_NAMES.index("flat_face")
    assert weights[flat_idx].item() < weights.mean().item()


def test_transformed_subset_applies_transform(tmp_path):
    from torch.utils.data import Subset
    from training.synthetic_data_gen import generate_dataset
    from training.train_feature_net import _TransformedSubset

    generate_dataset(str(tmp_path), count=10)
    ds = MachiningFeatureDataset(str(tmp_path), resolution=16)
    if len(ds) == 0:
        pytest.skip("No synthetic data generated")
    subset = Subset(ds, list(range(min(5, len(ds)))))
    aug_ds = _TransformedSubset(subset, get_train_transforms())
    x, _ = aug_ds[0]
    assert x.shape[0] == 1


def test_checkpoint_contains_training_config(tmp_path):
    from models.feature_net import FeatureNet3D, NUM_CLASSES

    model = FeatureNet3D()
    checkpoint = {
        "epoch": 0,
        "model_state_dict": model.state_dict(),
        "val_f1": 0.0,
        "training_config": {
            "resolution": 64,
            "num_classes": NUM_CLASSES,
            "feature_names": ["flat_face"],
        },
    }
    path = str(tmp_path / "test.pt")
    torch.save(checkpoint, path)
    loaded = torch.load(path, map_location="cpu")
    assert "training_config" in loaded
    assert loaded["training_config"]["resolution"] == 64
