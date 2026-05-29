import json
import os

import numpy as np
import pytest
import torch

from models.feature_net import FEATURE_NAMES, FEATURE_TO_IDX, NUM_CLASSES, FeatureNet3D, load_model
from phase2_feature_recognition import (
    clean_feature_predictions,
    fuse_features_with_voxel_geometry,
    recognise_features,
    reconcile_features_with_brep,
)
from training.dataset import MachiningFeatureDataset
from training.synthetic_data_gen import EXCLUDED_TRAINING_FEATURES, generate_dataset, generate_part


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


def test_generate_part_can_target_flat_only_negative():
    _, labels = generate_part(target_feature="flat_face")
    assert labels == ["flat_face"]


@pytest.mark.parametrize("feature", sorted(EXCLUDED_TRAINING_FEATURES))
def test_generate_part_excludes_edge_detail_training_features(feature):
    _, labels = generate_part(target_feature=feature)
    assert feature not in labels
    assert labels == ["flat_face"]


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


def test_balanced_dataset_includes_flat_only_samples(tmp_path):
    generate_dataset(str(tmp_path), count=len(FEATURE_NAMES), balanced=True)
    flat_only = 0
    for label_path in tmp_path.glob("*/labels.json"):
        with open(label_path, encoding="utf-8") as f:
            labels = json.load(f)["labels"]
        if labels == ["flat_face"]:
            flat_only += 1
    assert flat_only >= 1


def test_generated_dataset_excludes_edge_detail_labels(tmp_path):
    generate_dataset(str(tmp_path), count=len(FEATURE_NAMES) * 2, balanced=True)
    labels = []
    for label_path in tmp_path.glob("*/labels.json"):
        with open(label_path, encoding="utf-8") as f:
            labels.extend(json.load(f)["labels"])

    assert EXCLUDED_TRAINING_FEATURES.isdisjoint(labels)

    with open(tmp_path / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    assert set(manifest["excluded_features"]) == EXCLUDED_TRAINING_FEATURES


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
def test_checkpoint_thresholds_override_argument(tmp_path):
    checkpoint_path = tmp_path / "thresholded.pt"
    model = FeatureNet3D(num_classes=NUM_CLASSES)
    thresholds = {name: 1.0 for name in FEATURE_NAMES}
    thresholds["flat_face"] = 0.0
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_thresholds": thresholds,
            "training_config": {
                "hidden_dim": model.hidden_dim,
                "class_thresholds": thresholds,
            },
        },
        checkpoint_path,
    )

    result = recognise_features(FIXTURE_VOXEL, str(checkpoint_path), threshold=0.0)

    assert result["threshold_source"] == "checkpoint"
    assert result["thresholds"]["flat_face"] == 0.0
    assert all(
        result["thresholds"][name] == 1.0
        for name in FEATURE_NAMES
        if name != "flat_face"
    )
    assert [feature["type"] for feature in result["features"]] == ["flat_face"]


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


def test_clean_feature_predictions_marks_uncertain_and_rejected():
    scores = {name: 0.0 for name in FEATURE_NAMES}
    scores["through_hole"] = 0.62
    scores["blind_hole"] = 0.57
    scores["rectangular_pocket"] = 0.40
    thresholds = {name: 0.60 for name in FEATURE_NAMES}

    result = clean_feature_predictions(scores, thresholds, uncertainty_margin=0.05)

    feature_types = {feature["type"] for feature in result["features"]}
    uncertain_types = {feature["type"] for feature in result["uncertain_features"]}
    assert "through_hole" in feature_types
    assert "blind_hole" in uncertain_types
    assert "rectangular_pocket" not in feature_types
    assert result["prediction_summary"]["detected"] == 1
    assert result["prediction_summary"]["uncertain"] == 1


def test_clean_feature_predictions_marks_training_excluded_edge_features():
    scores = {name: 0.0 for name in FEATURE_NAMES}
    scores["chamfer"] = 0.80
    scores["fillet"] = 0.70
    thresholds = {name: 0.50 for name in FEATURE_NAMES}

    result = clean_feature_predictions(scores, thresholds)

    statuses = {feature["type"]: feature["status"] for feature in result["features"]}
    assert statuses["chamfer"] == "training_excluded"
    assert statuses["fillet"] == "training_excluded"
    assert result["prediction_summary"]["training_excluded"] == 2
    assert result["warnings"]


@pytest.mark.skipif(not os.path.exists(FIXTURE_VOXEL), reason="Phase 1 CLI output not available")
def test_recognise_features_includes_cleanup_metadata(random_checkpoint):
    result = recognise_features(FIXTURE_VOXEL, random_checkpoint)
    for key in (
        "candidate_features",
        "uncertain_features",
        "uncertain_count",
        "prediction_summary",
        "active_features",
        "excluded_features",
    ):
        assert key in result


def test_reconcile_suppresses_simple_block_edge_false_positives():
    recognition = {
        "features": [
            {"type": "flat_face", "confidence": 0.99},
            {"type": "fillet", "confidence": 0.56},
            {"type": "chamfer", "confidence": 0.55},
        ],
        "all_scores": {"flat_face": 0.99, "fillet": 0.56, "chamfer": 0.55},
    }
    brep = {"holes": [], "planar_recesses": [], "bounding_box_mm": {"z": 40.0}}
    result = reconcile_features_with_brep(recognition, brep)
    assert [feature["type"] for feature in result["features"]] == ["flat_face"]
    assert result["warnings"]


def test_reconcile_removes_blind_hole_when_only_through_holes_measured():
    recognition = {
        "features": [
            {"type": "flat_face", "confidence": 0.99},
            {"type": "through_hole", "confidence": 0.95},
            {"type": "blind_hole", "confidence": 0.68},
        ],
        "all_scores": {"flat_face": 0.99, "through_hole": 0.95, "blind_hole": 0.68},
    }
    brep = {
        "holes": [{"diameter_mm": 10.0, "depth_mm": 40.0}],
        "planar_recesses": [],
        "bounding_box_mm": {"z": 40.0},
    }
    result = reconcile_features_with_brep(recognition, brep)
    feature_types = {feature["type"] for feature in result["features"]}
    assert "through_hole" in feature_types
    assert "blind_hole" not in feature_types


def test_voxel_geometry_fusion_supports_through_hole(tmp_path):
    grid = np.ones((32, 32, 32), dtype=bool)
    grid[14:18, 14:18, :] = False
    voxel_path = tmp_path / "voxel.npy"
    np.save(voxel_path, grid)
    recognition = {
        "features": [{"type": "through_hole", "confidence": 0.82}],
        "warnings": [],
    }

    result = fuse_features_with_voxel_geometry(recognition, str(voxel_path))

    assert result["feature_count"] == 1
    assert result["features"][0]["geometry_support"] == "supported"
    assert result["geometry_fusion"]["evidence"]["through_z_components"] == 1


def test_voxel_geometry_fusion_suppresses_low_confidence_unsupported_feature(tmp_path):
    voxel_path = tmp_path / "solid.npy"
    np.save(voxel_path, np.ones((24, 24, 24), dtype=bool))
    recognition = {
        "features": [{"type": "blind_hole", "confidence": 0.61}],
        "warnings": [],
    }

    result = fuse_features_with_voxel_geometry(recognition, str(voxel_path))

    assert result["feature_count"] == 0
    assert result["geometry_fusion"]["suppressed_count"] == 1
    assert result["warnings"]


def test_voxel_geometry_fusion_keeps_high_confidence_unsupported_for_review(tmp_path):
    voxel_path = tmp_path / "solid.npy"
    np.save(voxel_path, np.ones((24, 24, 24), dtype=bool))
    recognition = {
        "features": [{"type": "circular_slot", "confidence": 0.97}],
        "warnings": [],
    }

    result = fuse_features_with_voxel_geometry(recognition, str(voxel_path))

    assert result["feature_count"] == 1
    assert result["features"][0]["status"] == "unverified_high_confidence"
    assert result["features"][0]["geometry_support"] == "unsupported"
