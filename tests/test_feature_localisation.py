import json
import os

import numpy as np
import pytest

from phase2c_feature_localisation import localise_feature_instances


def make_block_with_blind_pocket(r: int = 32) -> np.ndarray:
    grid = np.ones((r, r, r), dtype=bool)
    grid[10:22, 10:22, 22:32] = False
    return grid


def make_block_with_through_hole(r: int = 32) -> np.ndarray:
    grid = np.ones((r, r, r), dtype=bool)
    cx, cy = r // 2, r // 2
    radius = r // 8
    for x in range(r):
        for y in range(r):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius**2:
                grid[x, y, :] = False
    return grid


def write_inputs(tmp_path, grid, features):
    voxel_path = tmp_path / "voxel_32.npy"
    features_path = tmp_path / "features.json"
    np.save(voxel_path, grid)
    features_path.write_text(
        json.dumps({"features": features, "feature_count": len(features)}),
        encoding="utf-8",
    )
    return str(voxel_path), str(features_path)


def test_localise_creates_file_and_schema(tmp_path):
    voxel_path, features_path = write_inputs(
        tmp_path,
        make_block_with_blind_pocket(),
        [{"type": "rectangular_pocket", "confidence": 0.9}],
    )
    result = localise_feature_instances(voxel_path, features_path, str(tmp_path))
    assert os.path.exists(tmp_path / "feature_instances.json")
    assert result["instance_count"] == 1
    instance = result["instances"][0]
    for key in [
        "type",
        "instance_id",
        "confidence",
        "centroid_voxel",
        "bbox_voxel",
        "volume_voxels",
        "primary_direction",
        "access_directions",
        "localisation_status",
    ]:
        assert key in instance


def test_localise_blind_pocket_access_from_plus_z(tmp_path):
    voxel_path, features_path = write_inputs(
        tmp_path,
        make_block_with_blind_pocket(),
        [{"type": "rectangular_pocket", "confidence": 0.9}],
    )
    result = localise_feature_instances(voxel_path, features_path, str(tmp_path))
    instance = result["instances"][0]
    assert instance["localisation_status"] == "localised"
    assert instance["primary_direction"] == "+Z"
    assert instance["volume_voxels"] > 0
    assert instance["top_accessible"] is True
    assert instance["access_class"] in ("top", "top_and_side")
    assert instance["depth_voxels"] > 0
    assert instance["opening_span_voxels"] > 0


def test_localise_through_hole_has_two_z_access_directions(tmp_path):
    voxel_path, features_path = write_inputs(
        tmp_path,
        make_block_with_through_hole(),
        [{"type": "through_hole", "confidence": 0.9}],
    )
    result = localise_feature_instances(voxel_path, features_path, str(tmp_path))
    dirs = set(result["instances"][0]["access_directions"])
    assert "+Z" in dirs
    assert "-Z" in dirs
    assert result["instances"][0]["access_class"] == "through_z"


def test_localise_flat_face_fallback(tmp_path):
    voxel_path, features_path = write_inputs(
        tmp_path,
        np.ones((16, 16, 16), dtype=bool),
        [{"type": "flat_face", "confidence": 0.99}],
    )
    result = localise_feature_instances(voxel_path, features_path, str(tmp_path))
    instance = result["instances"][0]
    assert instance["type"] == "flat_face"
    assert instance["localisation_status"] == "estimated"
    assert any("flat_face" in warning for warning in result["warnings"])


def test_localise_expands_instances_from_pmi(tmp_path):
    voxel_path, features_path = write_inputs(
        tmp_path,
        make_block_with_through_hole(),
        [{"type": "through_hole", "confidence": 0.99}],
    )
    pmi_path = tmp_path / "pmi_data.json"
    pmi_path.write_text(
        json.dumps(
            {
                "features": [
                    {"type": "through_hole", "instance_id": 0, "diameter_mm": 10.0, "depth_mm": 40.0},
                    {"type": "through_hole", "instance_id": 1, "diameter_mm": 10.0, "depth_mm": 40.0},
                    {"type": "through_hole", "instance_id": 2, "diameter_mm": 10.0, "depth_mm": 40.0},
                ]
            }
        )
    )
    result = localise_feature_instances(
        voxel_path,
        features_path,
        str(tmp_path),
        pmi_data_path=str(pmi_path),
    )
    holes = [instance for instance in result["instances"] if instance["type"] == "through_hole"]
    assert len(holes) == 3
    assert [instance["instance_id"] for instance in holes] == [0, 1, 2]
    assert all(instance["diameter_mm"] == 10.0 for instance in holes)


def test_missing_inputs_raise(tmp_path):
    voxel_path, features_path = write_inputs(
        tmp_path,
        np.ones((8, 8, 8), dtype=bool),
        [{"type": "flat_face", "confidence": 0.99}],
    )
    with pytest.raises(FileNotFoundError):
        localise_feature_instances("no_voxel.npy", features_path, str(tmp_path))
    with pytest.raises(FileNotFoundError):
        localise_feature_instances(voxel_path, "no_features.json", str(tmp_path))
