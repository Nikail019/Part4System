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
        "setup_id",
        "approach_direction",
        "feature_type",
        "operation_type",
        "tool_type",
        "phase",
        "notes",
    }
    for op in result["operations"]:
        assert required.issubset(op.keys())


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
    assert result["setup_count"] == SIMPLE_SETUP["setup_count"]


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
