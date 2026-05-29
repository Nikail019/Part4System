import copy
import json
import os

import pytest

from phase4_process_plan import (
    OPERATION_MAP,
    OPERATION_NOTES,
    ROUGHING_PRIORITY,
    _build_operations,
    _resolve_features_per_setup,
    generate_process_plan,
)


SIMPLE_METADATA = {
    "bounding_box_mm": {"x": 100.0, "y": 60.0, "z": 40.0},
    "volume_mm3": 240000.0,
    "surface_area_mm2": 28800.0,
}

SIMPLE_FEATURES = {
    "features": [
        {"type": "flat_face", "confidence": 0.99},
        {"type": "rectangular_pocket", "confidence": 0.87},
        {"type": "through_hole", "confidence": 0.81},
    ],
    "feature_count": 3,
    "threshold": 0.5,
}

SIMPLE_SETUP = {
    "setup_count": 2,
    "axis_requirement": 3,
    "setups": [
        {
            "id": 0,
            "approach_direction": "+Z",
            "rotation_from_previous": "initial",
            "surface_voxel_count": 1820,
            "surface_coverage_fraction": 0.74,
        },
        {
            "id": 1,
            "approach_direction": "-Z",
            "rotation_from_previous": "flip_around_X_180",
            "surface_voxel_count": 640,
            "surface_coverage_fraction": 0.26,
        },
    ],
    "features_per_setup": {
        "0": ["flat_face", "rectangular_pocket", "through_hole"],
        "1": ["flat_face"],
    },
    "warnings": [],
}

CLI_DIR = "data/processed/simple_block_cli"


@pytest.fixture
def input_files(tmp_path):
    meta_path = tmp_path / "metadata.json"
    feat_path = tmp_path / "features.json"
    setup_path = tmp_path / "setup_analysis.json"
    meta_path.write_text(json.dumps(SIMPLE_METADATA))
    feat_path.write_text(json.dumps(SIMPLE_FEATURES))
    setup_path.write_text(json.dumps(SIMPLE_SETUP))
    return {
        "metadata": str(meta_path),
        "features": str(feat_path),
        "setup": str(setup_path),
        "out": str(tmp_path),
    }


def test_operation_map_all_feature_classes_present():
    from models.feature_net import FEATURE_NAMES

    for name in FEATURE_NAMES:
        assert name in OPERATION_MAP


def test_operation_map_phases_valid():
    for _, ops in OPERATION_MAP.items():
        for op in ops:
            assert op["phase"] in ("roughing", "finishing")


def test_operation_map_required_keys():
    for _, ops in OPERATION_MAP.items():
        for op in ops:
            assert "type" in op
            assert "tool" in op
            assert "phase" in op


def test_roughing_priority_all_features_present():
    from models.feature_net import FEATURE_NAMES

    for name in FEATURE_NAMES:
        assert name in ROUGHING_PRIORITY


def test_flat_face_priority_is_zero():
    assert ROUGHING_PRIORITY["flat_face"] == 0


def test_flat_face_is_first_operation():
    features_per_setup = {"0": ["through_hole", "flat_face", "rectangular_pocket"]}
    setup_list = [{"id": 0, "approach_direction": "+Z"}]
    ops, _ = _build_operations(features_per_setup, setup_list)
    assert ops[0]["feature_type"] == "flat_face"


def test_roughing_before_finishing():
    features_per_setup = {"0": ["rectangular_pocket"]}
    setup_list = [{"id": 0, "approach_direction": "+Z"}]
    ops, _ = _build_operations(features_per_setup, setup_list)
    phases = [op["phase"] for op in ops]
    seen_finishing = False
    for phase in phases:
        if phase == "finishing":
            seen_finishing = True
        if seen_finishing:
            assert phase == "finishing"


def test_setup_0_before_setup_1():
    features_per_setup = {"0": ["flat_face"], "1": ["flat_face"]}
    setup_list = [
        {"id": 0, "approach_direction": "+Z"},
        {"id": 1, "approach_direction": "-Z"},
    ]
    ops, _ = _build_operations(features_per_setup, setup_list)
    setup_ids = [op["setup_id"] for op in ops]
    last_0 = max(i for i, setup_id in enumerate(setup_ids) if setup_id == 0)
    first_1 = min(i for i, setup_id in enumerate(setup_ids) if setup_id == 1)
    assert last_0 < first_1


def test_duplicate_features_deduplicated():
    features_per_setup = {"0": ["flat_face", "flat_face", "rectangular_pocket"]}
    setup_list = [{"id": 0, "approach_direction": "+Z"}]
    ops, _ = _build_operations(features_per_setup, setup_list)
    flat_ops = [op for op in ops if op["feature_type"] == "flat_face"]
    assert len(flat_ops) == len(OPERATION_MAP["flat_face"])


def test_unknown_feature_generates_warning():
    features_per_setup = {"0": ["unknown_mystery_feature"]}
    setup_list = [{"id": 0, "approach_direction": "+Z"}]
    _, warnings = _build_operations(features_per_setup, setup_list)
    assert any("unknown_mystery_feature" in warning for warning in warnings)


def test_chamfer_is_finishing_only():
    features_per_setup = {"0": ["chamfer"]}
    setup_list = [{"id": 0, "approach_direction": "+Z"}]
    ops, _ = _build_operations(features_per_setup, setup_list)
    for op in ops:
        if op["feature_type"] == "chamfer":
            assert op["phase"] == "finishing"


def test_resolve_uses_phase3_when_populated():
    setup = copy.deepcopy(SIMPLE_SETUP)
    result = _resolve_features_per_setup(setup, SIMPLE_FEATURES["features"])
    assert result == setup["features_per_setup"]


def test_resolve_falls_back_to_setup0_when_empty():
    setup = copy.deepcopy(SIMPLE_SETUP)
    setup["features_per_setup"] = {"0": [], "1": []}
    features = [{"type": "flat_face", "confidence": 0.99}]
    result = _resolve_features_per_setup(setup, features)
    assert "flat_face" in result["0"]


def test_output_file_created(input_files):
    generate_process_plan(
        input_files["metadata"],
        input_files["features"],
        input_files["setup"],
        input_files["out"],
    )
    assert os.path.exists(os.path.join(input_files["out"], "process_plan.json"))


def test_schema_keys_present(input_files):
    result = generate_process_plan(
        input_files["metadata"],
        input_files["features"],
        input_files["setup"],
        input_files["out"],
    )
    required = [
        "operations",
        "operation_count",
        "setup_count",
        "axis_requirement",
        "tool_list",
        "source_files",
        "process_plan_file",
        "warnings",
    ]
    for key in required:
        assert key in result


def test_operation_steps_sequential(input_files):
    result = generate_process_plan(
        input_files["metadata"],
        input_files["features"],
        input_files["setup"],
        input_files["out"],
    )
    steps = [op["step"] for op in result["operations"]]
    assert steps == list(range(1, len(steps) + 1))


def test_operation_count_matches_list(input_files):
    result = generate_process_plan(
        input_files["metadata"],
        input_files["features"],
        input_files["setup"],
        input_files["out"],
    )
    assert result["operation_count"] == len(result["operations"])


def test_tool_list_is_deduplicated(input_files):
    result = generate_process_plan(
        input_files["metadata"],
        input_files["features"],
        input_files["setup"],
        input_files["out"],
    )
    assert len(result["tool_list"]) == len(set(result["tool_list"]))


def test_operation_required_keys(input_files):
    result = generate_process_plan(
        input_files["metadata"],
        input_files["features"],
        input_files["setup"],
        input_files["out"],
    )
    required = {
        "step",
        "operation_id",
        "setup_id",
        "approach_direction",
        "feature_type",
        "feature_instance_id",
        "operation_type",
        "tool_type",
        "tool_diameter_mm",
        "cut_depth_mm",
        "estimated_removal_volume_mm3",
        "requires_review",
        "phase",
        "notes",
    }
    for op in result["operations"]:
        missing = required - set(op.keys())
        assert not missing, f"Missing operation keys: {missing}"


def test_operation_id_matches_step(input_files):
    result = generate_process_plan(
        input_files["metadata"],
        input_files["features"],
        input_files["setup"],
        input_files["out"],
    )
    assert all(op["operation_id"] == op["step"] for op in result["operations"])


def test_axis_requirement_passed_through(input_files):
    result = generate_process_plan(
        input_files["metadata"],
        input_files["features"],
        input_files["setup"],
        input_files["out"],
    )
    assert result["axis_requirement"] == SIMPLE_SETUP["axis_requirement"]


def test_setup_count_passed_through(input_files):
    result = generate_process_plan(
        input_files["metadata"],
        input_files["features"],
        input_files["setup"],
        input_files["out"],
    )
    assert result["setup_count"] == 1


def test_process_plan_operations_are_top_side_only(input_files):
    result = generate_process_plan(
        input_files["metadata"],
        input_files["features"],
        input_files["setup"],
        input_files["out"],
    )
    assert result["setup_mode"] == "2.5d_single_setup"
    assert result["setups"][0]["approach_direction"] == "+Z"
    assert all(op["setup_id"] == 0 for op in result["operations"])
    assert all(op["approach_direction"] == "+Z" for op in result["operations"])


def test_confidence_threshold_filters_features(input_files, tmp_path):
    features_high_threshold = {
        "features": [
            {"type": "flat_face", "confidence": 0.99},
            {"type": "rectangular_pocket", "confidence": 0.30},
        ],
        "feature_count": 2,
        "threshold": 0.5,
    }
    feat_path = tmp_path / "features_low.json"
    feat_path.write_text(json.dumps(features_high_threshold))

    result = generate_process_plan(
        input_files["metadata"],
        str(feat_path),
        input_files["setup"],
        input_files["out"],
        confidence_threshold=0.5,
    )
    feature_types = {op["feature_type"] for op in result["operations"]}
    assert "rectangular_pocket" not in feature_types
    assert any("rectangular_pocket" in warning for warning in result["warnings"])


def test_missing_metadata_raises(input_files):
    with pytest.raises(FileNotFoundError):
        generate_process_plan(
            "no_metadata.json",
            input_files["features"],
            input_files["setup"],
            input_files["out"],
        )


def test_missing_features_raises(input_files):
    with pytest.raises(FileNotFoundError):
        generate_process_plan(
            input_files["metadata"],
            "no_features.json",
            input_files["setup"],
            input_files["out"],
        )


def test_missing_setup_raises(input_files):
    with pytest.raises(FileNotFoundError):
        generate_process_plan(
            input_files["metadata"],
            input_files["features"],
            "no_setup.json",
            input_files["out"],
        )


def test_output_path_is_absolute(input_files):
    result = generate_process_plan(
        input_files["metadata"],
        input_files["features"],
        input_files["setup"],
        input_files["out"],
    )
    assert os.path.isabs(result["process_plan_file"])


def test_warnings_is_list(input_files):
    result = generate_process_plan(
        input_files["metadata"],
        input_files["features"],
        input_files["setup"],
        input_files["out"],
    )
    assert isinstance(result["warnings"], list)


def test_written_json_is_valid(input_files):
    result = generate_process_plan(
        input_files["metadata"],
        input_files["features"],
        input_files["setup"],
        input_files["out"],
    )
    with open(result["process_plan_file"], encoding="utf-8") as f:
        on_disk = json.load(f)
    assert on_disk["operation_count"] == result["operation_count"]


def test_process_plan_uses_feature_instances_when_available(input_files, tmp_path):
    setup = copy.deepcopy(SIMPLE_SETUP)
    setup["feature_instances_per_setup"] = {
        "0": [
            {
                "type": "rectangular_pocket",
                "instance_id": 2,
                "confidence": 0.9,
                "primary_direction": "+Z",
                "volume_voxels": 120,
                "localisation_status": "localised",
            }
        ],
        "1": [],
    }
    setup_path = tmp_path / "setup_with_instances.json"
    setup_path.write_text(json.dumps(setup))
    result = generate_process_plan(
        input_files["metadata"],
        input_files["features"],
        str(setup_path),
        input_files["out"],
    )
    assert result["operation_count"] > 0
    assert all(op.get("feature_instance_id") == 2 for op in result["operations"])
    assert all(op.get("feature_volume_voxels") == 120 for op in result["operations"])
    assert all(op.get("estimated_removal_volume_mm3") is not None for op in result["operations"])
    assert all(op.get("requires_review") is False for op in result["operations"])


def test_process_plan_generates_operations_for_multiple_hole_instances(input_files, tmp_path):
    setup = copy.deepcopy(SIMPLE_SETUP)
    setup["feature_instances_per_setup"] = {
        "0": [
            {
                "type": "through_hole",
                "instance_id": 0,
                "confidence": 0.99,
                "primary_direction": "+Z",
                "access_directions": ["+Z", "-Z"],
                "volume_voxels": 60,
                "localisation_status": "localised",
                "two_point_five_d_supported": True,
            },
            {
                "type": "through_hole",
                "instance_id": 1,
                "confidence": 0.99,
                "primary_direction": "+Z",
                "access_directions": ["+Z", "-Z"],
                "volume_voxels": 60,
                "localisation_status": "localised",
                "two_point_five_d_supported": True,
            },
            {
                "type": "through_hole",
                "instance_id": 2,
                "confidence": 0.99,
                "primary_direction": "+Z",
                "access_directions": ["+Z", "-Z"],
                "volume_voxels": 60,
                "localisation_status": "localised",
                "two_point_five_d_supported": True,
            },
        ]
    }
    setup_path = tmp_path / "setup_multi_holes.json"
    setup_path.write_text(json.dumps(setup))
    result = generate_process_plan(
        input_files["metadata"],
        input_files["features"],
        str(setup_path),
        input_files["out"],
    )
    drill_ops = [op for op in result["operations"] if op["operation_type"] == "drill"]
    assert len(drill_ops) == 3
    assert {op["feature_instance_id"] for op in drill_ops} == {0, 1, 2}


def test_process_plan_skips_unsupported_25d_instance(input_files, tmp_path):
    setup = copy.deepcopy(SIMPLE_SETUP)
    setup["two_point_five_d_compatible"] = False
    setup["unsupported_reasons"] = ["rectangular_pocket instance 3 requires side access."]
    setup["feature_instances_per_setup"] = {
        "0": [
            {
                "type": "rectangular_pocket",
                "instance_id": 3,
                "confidence": 0.9,
                "primary_direction": "+X",
                "access_directions": ["+X"],
                "volume_voxels": 120,
                "localisation_status": "localised",
                "two_point_five_d_supported": False,
                "unsupported_reason": "rectangular_pocket instance 3 requires side access.",
            }
        ],
        "1": [],
    }
    setup_path = tmp_path / "setup_with_unsupported_instances.json"
    setup_path.write_text(json.dumps(setup))
    result = generate_process_plan(
        input_files["metadata"],
        input_files["features"],
        str(setup_path),
        input_files["out"],
    )
    assert result["two_point_five_d_compatible"] is False
    assert result["unsupported_reasons"]
    assert "UNSUPPORTED_25D_INSTANCE" in result["review_codes"]
    assert result["review_items"]
    assert all(op.get("feature_instance_id") != 3 for op in result["operations"])


def test_process_plan_preserves_setup_review_codes(input_files, tmp_path):
    setup = copy.deepcopy(SIMPLE_SETUP)
    setup["two_point_five_d_compatible"] = False
    setup["tool_reach_compatible"] = False
    setup["review_items"] = [
        {
            "code": "TOOL_REACH_LIMIT",
            "severity": "review",
            "message": "rectangular_pocket instance 0 exceeds tool reach.",
            "source": "phase3_setup_analysis",
        }
    ]
    setup["feature_feasibility"] = [{"instance_id": 0, "tool_reach_ok": False}]
    setup_path = tmp_path / "setup_review.json"
    setup_path.write_text(json.dumps(setup))
    result = generate_process_plan(
        input_files["metadata"],
        input_files["features"],
        str(setup_path),
        input_files["out"],
    )
    assert result["tool_reach_compatible"] is False
    assert "TOOL_REACH_LIMIT" in result["review_codes"]
    assert result["feature_feasibility"] == setup["feature_feasibility"]


@pytest.mark.skipif(
    not all(
        os.path.exists(os.path.join(CLI_DIR, filename))
        for filename in ["metadata.json", "features.json", "setup_analysis.json"]
    ),
    reason="Real Phase 1-3 CLI outputs not available",
)
def test_full_pipeline_on_simple_block(tmp_path):
    result = generate_process_plan(
        os.path.join(CLI_DIR, "metadata.json"),
        os.path.join(CLI_DIR, "features.json"),
        os.path.join(CLI_DIR, "setup_analysis.json"),
        str(tmp_path),
    )
    assert result["operation_count"] > 0
    assert result["operation_count"] >= 2
    feature_types = {op["feature_type"] for op in result["operations"]}
    assert "flat_face" in feature_types
