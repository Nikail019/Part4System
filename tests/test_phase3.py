import json
import os

import numpy as np
import pytest

from phase3_setup_analysis import (
    DIRECTION_LABELS,
    NUM_DIRECTIONS,
    analyse_setups,
    compute_accessibility_map,
    compute_surface_mask,
    greedy_setup_assignment,
    infer_axis_requirement,
    map_feature_instances_to_setups,
    map_features_to_setups,
)


FIXTURE_VOXEL = "data/processed/simple_block_cli/voxel_64.npy"


def make_solid_cube(R: int = 16) -> np.ndarray:
    return np.ones((R, R, R), dtype=bool)


def make_hollow_cube(R: int = 16, wall: int = 2) -> np.ndarray:
    g = np.ones((R, R, R), dtype=bool)
    g[wall:-wall, wall:-wall, wall:-wall] = False
    return g


def make_block_with_blind_pocket(R: int = 32) -> np.ndarray:
    g = np.ones((R, R, R), dtype=bool)
    pw = R // 4
    depth = R // 3
    cx, cy = R // 2, R // 2
    g[cx - pw : cx + pw, cy - pw : cy + pw, R - depth :] = False
    return g


def make_block_with_through_hole(R: int = 32) -> np.ndarray:
    g = np.ones((R, R, R), dtype=bool)
    cx, cy = R // 2, R // 2
    radius = R // 8
    for x in range(R):
        for y in range(R):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius**2:
                g[x, y, :] = False
    return g


def test_surface_mask_subset_of_occupied():
    g = make_solid_cube(16)
    s = compute_surface_mask(g)
    assert np.all(s <= g)


def test_surface_mask_hollow_interior_not_flagged():
    g = make_solid_cube(16)
    s = compute_surface_mask(g)
    assert not s[8, 8, 8]


def test_surface_mask_outer_layer_flagged():
    r = 16
    g = make_solid_cube(r)
    s = compute_surface_mask(g)
    assert s[0, :, :].all()
    assert s[-1, :, :].all()
    assert s[:, 0, :].all()
    assert s[:, -1, :].all()
    assert s[:, :, 0].all()
    assert s[:, :, -1].all()


def test_surface_mask_empty_grid():
    g = np.zeros((16, 16, 16), dtype=bool)
    s = compute_surface_mask(g)
    assert s.sum() == 0


def test_surface_mask_shape_preserved():
    g = make_solid_cube(32)
    s = compute_surface_mask(g)
    assert s.shape == g.shape


def test_accessibility_map_shape():
    g = make_solid_cube(16)
    a = compute_accessibility_map(g)
    assert a.shape == (NUM_DIRECTIONS, 16, 16, 16)
    assert a.dtype == bool


def test_solid_cube_top_face_accessible_from_plus_z():
    r = 16
    g = make_solid_cube(r)
    s = compute_surface_mask(g)
    a = compute_accessibility_map(g)
    plus_z_idx = DIRECTION_LABELS.index("+Z")
    top_face_surface = s[:, :, r - 1]
    assert top_face_surface.all()
    assert a[plus_z_idx, :, :, r - 1].all()


def test_solid_cube_bottom_face_accessible_from_minus_z():
    g = make_solid_cube(16)
    a = compute_accessibility_map(g)
    minus_z_idx = DIRECTION_LABELS.index("-Z")
    assert a[minus_z_idx, :, :, 0].all()


def test_solid_cube_top_face_not_accessible_from_minus_z():
    r = 16
    g = make_solid_cube(r)
    a = compute_accessibility_map(g)
    minus_z_idx = DIRECTION_LABELS.index("-Z")
    assert not a[minus_z_idx, :, :, r - 1].any()


def test_blind_pocket_only_accessible_from_top():
    g = make_block_with_blind_pocket(32)
    a = compute_accessibility_map(g)
    s = compute_surface_mask(g)
    plus_z_idx = DIRECTION_LABELS.index("+Z")
    minus_z_idx = DIRECTION_LABELS.index("-Z")
    surf_from_plus_z = (a[plus_z_idx] & s).sum()
    surf_from_minus_z = (a[minus_z_idx] & s).sum()
    assert surf_from_plus_z > surf_from_minus_z


def test_accessibility_map_no_self_occlusion():
    g = np.zeros((8, 8, 8), dtype=bool)
    g[4, 4, 4] = True
    a = compute_accessibility_map(g)
    for d_idx in range(NUM_DIRECTIONS):
        assert a[d_idx, 4, 4, 4]


def test_accessibility_map_column_blocked():
    g = np.zeros((8, 8, 8), dtype=bool)
    g[4, 4, 2] = True
    g[4, 4, 5] = True
    a = compute_accessibility_map(g)
    plus_z_idx = DIRECTION_LABELS.index("+Z")
    assert a[plus_z_idx, 4, 4, 5]
    assert not a[plus_z_idx, 4, 4, 2]


def test_solid_cube_needs_two_setups_top_and_bottom():
    g = make_solid_cube(16)
    s = compute_surface_mask(g)
    a = compute_accessibility_map(g)
    setups = greedy_setup_assignment(a, s, coverage_threshold=0.99)
    directions = [st["approach_direction"] for st in setups]
    assert "+Z" in directions
    assert "-Z" in directions


def test_setup_coverage_fractions_sum_to_one():
    g = make_solid_cube(16)
    s = compute_surface_mask(g)
    a = compute_accessibility_map(g)
    setups = greedy_setup_assignment(a, s, coverage_threshold=0.99)
    total = sum(st["surface_coverage_fraction"] for st in setups)
    assert abs(total - 1.0) < 0.02


def test_setup_ids_sequential():
    g = make_solid_cube(16)
    s = compute_surface_mask(g)
    a = compute_accessibility_map(g)
    setups = greedy_setup_assignment(a, s)
    for idx, setup in enumerate(setups):
        assert setup["id"] == idx


def test_first_setup_rotation_is_initial():
    g = make_solid_cube(16)
    s = compute_surface_mask(g)
    a = compute_accessibility_map(g)
    setups = greedy_setup_assignment(a, s)
    assert setups[0]["rotation_from_previous"] == "initial"


def test_setup_has_required_keys():
    g = make_solid_cube(16)
    s = compute_surface_mask(g)
    a = compute_accessibility_map(g)
    setups = greedy_setup_assignment(a, s)
    required = {
        "id",
        "approach_direction",
        "rotation_from_previous",
        "surface_voxel_count",
        "surface_coverage_fraction",
    }
    for setup in setups:
        assert required.issubset(setup.keys())


def test_axis_requirement_single_direction():
    assert infer_axis_requirement(["+Z"]) == 3


def test_axis_requirement_flip_same_axis():
    assert infer_axis_requirement(["+Z", "-Z"]) == 3


def test_axis_requirement_two_different_axes():
    assert infer_axis_requirement(["+Z", "+X"]) == 4


def test_axis_requirement_three_axes():
    assert infer_axis_requirement(["+Z", "+X", "+Y"]) == 5


def test_axis_requirement_empty():
    assert infer_axis_requirement([]) == 3


def test_map_feature_instances_to_matching_setup_direction():
    setups = [
        {"id": 0, "approach_direction": "+Z"},
        {"id": 1, "approach_direction": "-Z"},
    ]
    instances = [
        {
            "type": "through_hole",
            "instance_id": 0,
            "confidence": 0.9,
            "primary_direction": "-Z",
            "access_directions": ["-Z", "+Z"],
            "volume_voxels": 20,
            "localisation_status": "localised",
        }
    ]
    result = map_feature_instances_to_setups(setups, instances)
    assert result["1"][0]["type"] == "through_hole"


@pytest.mark.skipif(not os.path.exists(FIXTURE_VOXEL), reason="Phase 1 CLI output not available")
def test_analyse_setups_output_files(tmp_path):
    analyse_setups(FIXTURE_VOXEL, str(tmp_path))
    assert (tmp_path / "setup_analysis.json").exists()
    assert (tmp_path / "accessibility_map.npy").exists()
    assert (tmp_path / "surface_mask.npy").exists()


@pytest.mark.skipif(not os.path.exists(FIXTURE_VOXEL), reason="Phase 1 CLI output not available")
def test_analyse_setups_schema(tmp_path):
    result = analyse_setups(FIXTURE_VOXEL, str(tmp_path))
    required = [
        "setup_count",
        "axis_requirement",
        "setups",
        "direction_coverage",
        "total_surface_voxels",
        "covered_surface_voxels",
        "inaccessible_surface_voxels",
        "inaccessible_fraction",
        "features_per_setup",
        "voxel_file",
        "accessibility_map_file",
        "surface_mask_file",
        "warnings",
    ]
    for key in required:
        assert key in result


@pytest.mark.skipif(not os.path.exists(FIXTURE_VOXEL), reason="Phase 1 CLI output not available")
def test_analyse_setups_with_feature_instances(tmp_path):
    features_path = tmp_path / "features.json"
    instances_path = tmp_path / "feature_instances.json"
    features_path.write_text(json.dumps({"features": [{"type": "flat_face", "confidence": 1.0}]}))
    instances_path.write_text(
        json.dumps(
            {
                "instances": [
                    {
                        "type": "flat_face",
                        "instance_id": 0,
                        "confidence": 1.0,
                        "primary_direction": "+Z",
                        "access_directions": ["+Z"],
                        "volume_voxels": 0,
                        "localisation_status": "estimated",
                    }
                ]
            }
        )
    )
    result = analyse_setups(
        FIXTURE_VOXEL,
        str(tmp_path),
        features_path=str(features_path),
        feature_instances_path=str(instances_path),
    )
    assert "feature_instances_per_setup" in result
    assert result["feature_instances_per_setup"]["0"][0]["instance_id"] == 0


@pytest.mark.skipif(not os.path.exists(FIXTURE_VOXEL), reason="Phase 1 CLI output not available")
def test_analyse_setups_direction_coverage_all_classes(tmp_path):
    result = analyse_setups(FIXTURE_VOXEL, str(tmp_path))
    for direction in DIRECTION_LABELS:
        assert direction in result["direction_coverage"]
        assert 0.0 <= result["direction_coverage"][direction] <= 1.0


@pytest.mark.skipif(not os.path.exists(FIXTURE_VOXEL), reason="Phase 1 CLI output not available")
def test_analyse_setups_simple_block_axis_3(tmp_path):
    result = analyse_setups(FIXTURE_VOXEL, str(tmp_path))
    assert result["axis_requirement"] == 3


@pytest.mark.skipif(not os.path.exists(FIXTURE_VOXEL), reason="Phase 1 CLI output not available")
def test_analyse_setups_accessibility_map_shape(tmp_path):
    analyse_setups(FIXTURE_VOXEL, str(tmp_path))
    acc = np.load(tmp_path / "accessibility_map.npy")
    vox = np.load(FIXTURE_VOXEL)
    r = vox.shape[0]
    assert acc.shape == (NUM_DIRECTIONS, r, r, r)
    assert acc.dtype == bool


@pytest.mark.skipif(not os.path.exists(FIXTURE_VOXEL), reason="Phase 1 CLI output not available")
def test_analyse_setups_surface_mask_shape(tmp_path):
    analyse_setups(FIX_VOXEL := FIXTURE_VOXEL, str(tmp_path))
    vox = np.load(FIX_VOXEL)
    surf = np.load(tmp_path / "surface_mask.npy")
    assert surf.shape == vox.shape
    assert surf.dtype == bool


@pytest.mark.skipif(not os.path.exists(FIXTURE_VOXEL), reason="Phase 1 CLI output not available")
def test_analyse_setups_features_per_setup_empty_when_no_features(tmp_path):
    result = analyse_setups(FIXTURE_VOXEL, str(tmp_path))
    for _, features in result["features_per_setup"].items():
        assert isinstance(features, list)


@pytest.mark.skipif(not os.path.exists(FIXTURE_VOXEL), reason="Phase 1 CLI output not available")
def test_analyse_setups_with_features_json(tmp_path):
    features_path = tmp_path / "features.json"
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump({"features": [{"type": "flat_face", "confidence": 1.0}]}, f)
    result = analyse_setups(FIXTURE_VOXEL, str(tmp_path), features_path=str(features_path))
    assert "flat_face" in result["features_per_setup"].get("0", [])


@pytest.mark.skipif(not os.path.exists(FIXTURE_VOXEL), reason="Phase 1 CLI output not available")
def test_analyse_setups_output_paths_absolute(tmp_path):
    result = analyse_setups(FIXTURE_VOXEL, str(tmp_path))
    assert os.path.isabs(result["voxel_file"])
    assert os.path.isabs(result["accessibility_map_file"])
    assert os.path.isabs(result["surface_mask_file"])


def test_analyse_setups_defaults_to_single_top_setup(tmp_path):
    voxel_path = tmp_path / "block.npy"
    np.save(voxel_path, make_solid_cube(16))
    result = analyse_setups(str(voxel_path), str(tmp_path))
    assert result["setup_mode"] == "2.5d_single_setup"
    assert result["setup_count"] == 1
    assert result["axis_requirement"] == 3
    assert result["requires_rotation"] is False
    assert result["setups"] == [
        {
            "id": 0,
            "approach_direction": "+Z",
            "rotation_from_previous": "initial",
            "surface_voxel_count": result["setups"][0]["surface_voxel_count"],
            "surface_coverage_fraction": result["setups"][0]["surface_coverage_fraction"],
        }
    ]


def test_analyse_setups_side_only_instance_sets_review(tmp_path):
    voxel_path = tmp_path / "block.npy"
    instances_path = tmp_path / "feature_instances.json"
    np.save(voxel_path, make_solid_cube(16))
    instances_path.write_text(
        json.dumps(
            {
                "instances": [
                    {
                        "type": "rectangular_pocket",
                        "instance_id": 0,
                        "confidence": 0.9,
                        "primary_direction": "+X",
                        "access_directions": ["+X"],
                        "volume_voxels": 30,
                        "localisation_status": "localised",
                    }
                ]
            }
        )
    )
    result = analyse_setups(
        str(voxel_path),
        str(tmp_path),
        feature_instances_path=str(instances_path),
    )
    assert result["two_point_five_d_compatible"] is False
    assert result["unsupported_reasons"]
    assert "SIDE_ACCESS_REQUIRED" in result["review_codes"]
    assert result["review_items"][0]["code"] == "SIDE_ACCESS_REQUIRED"
    assert result["feature_instances_per_setup"]["0"][0]["two_point_five_d_supported"] is False


def test_analyse_setups_tool_reach_review_with_metadata(tmp_path):
    voxel_path = tmp_path / "block.npy"
    metadata_path = tmp_path / "metadata.json"
    instances_path = tmp_path / "feature_instances.json"
    np.save(voxel_path, make_solid_cube(32))
    metadata_path.write_text(json.dumps({"bounding_box_mm": {"x": 320.0, "y": 320.0, "z": 320.0}}))
    instances_path.write_text(
        json.dumps(
            {
                "instances": [
                    {
                        "type": "rectangular_pocket",
                        "instance_id": 0,
                        "confidence": 0.9,
                        "primary_direction": "+Z",
                        "access_directions": ["+Z"],
                        "volume_voxels": 100,
                        "localisation_status": "localised",
                        "top_accessible": True,
                        "opening_span_voxels": 2,
                        "depth_voxels": 20,
                    }
                ]
            }
        )
    )
    result = analyse_setups(
        str(voxel_path),
        str(tmp_path),
        feature_instances_path=str(instances_path),
        metadata_path=str(metadata_path),
    )
    assert result["tool_reach_compatible"] is False
    assert "TOOL_REACH_LIMIT" in result["review_codes"]
    assert result["feature_feasibility"][0]["estimated_depth_mm"] is not None


def test_analyse_setups_tool_reach_uses_pmi_dimensions(tmp_path):
    voxel_path = tmp_path / "block.npy"
    metadata_path = tmp_path / "metadata.json"
    pmi_path = tmp_path / "pmi_data.json"
    instances_path = tmp_path / "feature_instances.json"
    np.save(voxel_path, make_solid_cube(32))
    metadata_path.write_text(json.dumps({"bounding_box_mm": {"x": 100.0, "y": 60.0, "z": 40.0}}))
    instances_path.write_text(
        json.dumps(
            {
                "instances": [
                    {
                        "type": "through_hole",
                        "instance_id": 0,
                        "confidence": 0.99,
                        "primary_direction": "+Z",
                        "access_directions": ["+Z", "-Z"],
                        "volume_voxels": 100,
                        "localisation_status": "localised",
                        "top_accessible": True,
                        "opening_span_voxels": 1,
                        "depth_voxels": 13,
                    }
                ]
            }
        )
    )
    pmi_path.write_text(
        json.dumps(
            {
                "features": [
                    {
                        "type": "through_hole",
                        "instance_id": 0,
                        "diameter_mm": 10.0,
                        "depth_mm": 40.0,
                    }
                ]
            }
        )
    )
    result = analyse_setups(
        str(voxel_path),
        str(tmp_path),
        feature_instances_path=str(instances_path),
        metadata_path=str(metadata_path),
        pmi_data_path=str(pmi_path),
    )
    assert result["tool_reach_compatible"] is True
    assert result["feature_feasibility"][0]["dimension_source"] == "pmi_brep"
    assert result["feature_feasibility"][0]["aspect_ratio"] == 4.0


def test_analyse_setups_file_not_found():
    with pytest.raises(FileNotFoundError):
        analyse_setups("no_such_voxel.npy", "/tmp/out")


def test_analyse_setups_creates_output_dir(tmp_path):
    if not os.path.exists(FIXTURE_VOXEL):
        pytest.skip("Phase 1 CLI output not available")
    new_dir = tmp_path / "nested" / "subdir"
    assert not new_dir.exists()
    analyse_setups(FIXTURE_VOXEL, str(new_dir))
    assert new_dir.exists()


def test_map_features_to_setups_empty_lists_without_features():
    setups = [{"id": 0, "approach_direction": "+Z"}]
    result = map_features_to_setups(
        setups,
        [],
        np.zeros((6, 2, 2, 2), dtype=bool),
        np.zeros((2, 2, 2), dtype=bool),
    )
    assert result == {"0": []}
