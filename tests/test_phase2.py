import json
import os

import numpy as np
import pytest
import torch

from models.feature_net import FEATURE_NAMES, FEATURE_TO_IDX, NUM_CLASSES, FeatureNet3D, load_model
from phase2_feature_recognition import recognise_features
from training.dataset import MachiningFeatureDataset
from training.synthetic_data_gen import generate_dataset, generate_part


FIXTURE_VOXEL = "data/processed/simple_block_cli/voxel_64.npy"


@pytest.fixture(scope="session")
def random_checkpoint(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("ckpt") / "random.pt")
    model = FeatureNet3D(num_classes=NUM_CLASSES)
    model.eval()
    torch.save({"model_state_dict": model.state_dict()}, path)
    return path


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    generate_dataset(str(root), count=2)
    return root


def test_model_output_shape_64():
    model = FeatureNet3D(num_classes=NUM_CLASSES)
    model.eval()
    with torch.no_grad():
        y = model(torch.zeros(2, 1, 64, 64, 64))
    assert y.shape == (2, 12)


def test_model_output_shape_32():
    model = FeatureNet3D(num_classes=NUM_CLASSES)
    model.eval()
    with torch.no_grad():
        y = model(torch.zeros(1, 1, 32, 32, 32))
    assert y.shape == (1, 12)


def test_model_sigmoid_in_range():
    model = FeatureNet3D(num_classes=NUM_CLASSES)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.zeros(1, 1, 32, 32, 32)))
    assert torch.all(probs >= 0)
    assert torch.all(probs <= 1)


def test_load_model_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_model("does_not_exist.pt")


def test_generate_part_returns_labels():
    _, labels = generate_part()
    assert isinstance(labels, list)
    assert labels


def test_generate_part_always_has_flat_face():
    _, labels = generate_part()
    assert "flat_face" in labels


def test_generate_part_labels_are_valid():
    _, labels = generate_part()
    assert all(label in FEATURE_NAMES for label in labels)


def test_generate_dataset_creates_files(tmp_path):
    generate_dataset(str(tmp_path), count=1)
    assert (tmp_path / "00000" / "part.stp").exists()
    assert (tmp_path / "00000" / "labels.json").exists()
    with open(tmp_path / "00000" / "labels.json", encoding="utf-8") as f:
        labels = json.load(f)["labels"]
    assert all(label in FEATURE_TO_IDX for label in labels)


def test_dataset_loads(synthetic_root):
    ds = MachiningFeatureDataset(synthetic_root, resolution=32)
    assert len(ds) > 0


def test_dataset_item_shapes(synthetic_root):
    ds = MachiningFeatureDataset(synthetic_root, resolution=32)
    x, y = ds[0]
    assert x.shape == (1, 32, 32, 32)
    assert y.shape == (12,)


def test_dataset_label_is_multihot(synthetic_root):
    ds = MachiningFeatureDataset(synthetic_root, resolution=32)
    _, y = ds[0]
    assert set(y.numpy().tolist()).issubset({0.0, 1.0})


def test_dataset_voxel_cache(synthetic_root):
    ds = MachiningFeatureDataset(synthetic_root, resolution=32)
    _ = ds[0]
    assert (ds.samples[0]["part_dir"] / "voxel_32.npy").exists()


@pytest.mark.skipif(not os.path.exists(FIXTURE_VOXEL), reason="Phase 1 CLI output not available")
def test_recognise_features_returns_dict(random_checkpoint):
    result = recognise_features(FIXTURE_VOXEL, random_checkpoint)
    assert isinstance(result, dict)


@pytest.mark.skipif(not os.path.exists(FIXTURE_VOXEL), reason="Phase 1 CLI output not available")
def test_recognise_features_schema(random_checkpoint):
    result = recognise_features(FIXTURE_VOXEL, random_checkpoint)
    for key in ("features", "feature_count", "all_scores", "threshold", "voxel_file", "model_path"):
        assert key in result


@pytest.mark.skipif(not os.path.exists(FIXTURE_VOXEL), reason="Phase 1 CLI output not available")
def test_all_scores_has_all_classes(random_checkpoint):
    result = recognise_features(FIXTURE_VOXEL, random_checkpoint)
    assert set(result["all_scores"]) == set(FEATURE_NAMES)
    assert all(0.0 <= score <= 1.0 for score in result["all_scores"].values())


@pytest.mark.skipif(not os.path.exists(FIXTURE_VOXEL), reason="Phase 1 CLI output not available")
def test_feature_count_matches_list(random_checkpoint):
    result = recognise_features(FIXTURE_VOXEL, random_checkpoint)
    assert result["feature_count"] == len(result["features"])


@pytest.mark.skipif(not os.path.exists(FIXTURE_VOXEL), reason="Phase 1 CLI output not available")
def test_threshold_zero_returns_all(random_checkpoint):
    result = recognise_features(FIXTURE_VOXEL, random_checkpoint, threshold=0.0)
    assert result["feature_count"] == 12


@pytest.mark.skipif(not os.path.exists(FIXTURE_VOXEL), reason="Phase 1 CLI output not available")
def test_threshold_one_returns_none(random_checkpoint):
    result = recognise_features(FIXTURE_VOXEL, random_checkpoint, threshold=1.0)
    assert result["feature_count"] == 0


@pytest.mark.skipif(not os.path.exists(FIXTURE_VOXEL), reason="Phase 1 CLI output not available")
def test_features_sorted_by_confidence(random_checkpoint):
    result = recognise_features(FIXTURE_VOXEL, random_checkpoint, threshold=0.0)
    confidences = [feature["confidence"] for feature in result["features"]]
    assert confidences == sorted(confidences, reverse=True)


def test_voxel_not_found_raises(random_checkpoint):
    with pytest.raises(FileNotFoundError):
        recognise_features("missing.npy", random_checkpoint)


def test_model_not_found_raises(tmp_path):
    voxel = tmp_path / "voxel.npy"
    np.save(voxel, np.zeros((4, 4, 4), dtype=bool))
    with pytest.raises(FileNotFoundError):
        recognise_features(str(voxel), "missing.pt")


@pytest.mark.skipif(not os.path.exists(FIXTURE_VOXEL), reason="Phase 1 CLI output not available")
def test_output_paths_are_absolute(random_checkpoint):
    result = recognise_features(FIXTURE_VOXEL, random_checkpoint)
    assert os.path.isabs(result["voxel_file"])
    assert os.path.isabs(result["model_path"])
