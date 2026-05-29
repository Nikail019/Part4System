import json
import os

import pytest

from phase5_time_estimate import (
    DEFAULT_MATERIAL,
    FEATURE_VOLUME_WEIGHT,
    MRR_TABLE,
    SETUP_TIME_MIN,
    TOOL_CHANGE_TIME_MIN,
    estimate_operation_time,
    estimate_removal_volumes,
    estimate_time,
)


CLI_DIR = "data/processed/simple_block_cli"

MINIMAL_METADATA = {
    "bounding_box_mm": {"x": 100.0, "y": 60.0, "z": 40.0},
    "volume_mm3": 180000.0,
    "surface_area_mm2": 26800.0,
    "raw_stock_mm": {"x": 115.0, "y": 70.0, "z": 45.0},
}

MINIMAL_PLAN = {
    "operations": [
        {
            "step": 1,
            "setup_id": 0,
            "approach_direction": "+Z",
            "feature_type": "flat_face",
            "operation_type": "face_mill_rough",
            "tool_type": "face_mill",
            "phase": "roughing",
            "notes": "",
        },
        {
            "step": 2,
            "setup_id": 0,
            "approach_direction": "+Z",
            "feature_type": "flat_face",
            "operation_type": "face_mill_finish",
            "tool_type": "face_mill",
            "phase": "finishing",
            "notes": "",
        },
        {
            "step": 3,
            "setup_id": 0,
            "approach_direction": "+Z",
            "feature_type": "rectangular_pocket",
            "operation_type": "endmill_rough",
            "tool_type": "flat_endmill",
            "phase": "roughing",
            "notes": "",
        },
    ],
    "setup_count": 1,
    "axis_requirement": 3,
}


@pytest.fixture
def plan_files(tmp_path):
    meta = tmp_path / "metadata.json"
    plan = tmp_path / "process_plan.json"
    meta.write_text(json.dumps(MINIMAL_METADATA))
    plan.write_text(json.dumps(MINIMAL_PLAN))
    return {"metadata": str(meta), "plan": str(plan), "out": str(tmp_path)}


def test_mrr_table_has_default_material():
    assert DEFAULT_MATERIAL in MRR_TABLE


def test_mrr_table_all_materials_have_same_keys():
    key_sets = [set(values.keys()) for values in MRR_TABLE.values()]
    assert all(keys == key_sets[0] for keys in key_sets)


def test_feature_volume_weight_all_features():
    from models.feature_net import FEATURE_NAMES

    for name in FEATURE_NAMES:
        assert name in FEATURE_VOLUME_WEIGHT


def test_removal_volumes_returns_per_step():
    volumes = estimate_removal_volumes(MINIMAL_PLAN["operations"], MINIMAL_METADATA)
    for op in MINIMAL_PLAN["operations"]:
        assert op["step"] in volumes
        assert volumes[op["step"]] >= 0


def test_roughing_volumes_sum_to_total_removal():
    stock_vol = (
        MINIMAL_METADATA["raw_stock_mm"]["x"]
        * MINIMAL_METADATA["raw_stock_mm"]["y"]
        * MINIMAL_METADATA["raw_stock_mm"]["z"]
    )
    total_removal = max(0.0, stock_vol - MINIMAL_METADATA["volume_mm3"])
    volumes = estimate_removal_volumes(MINIMAL_PLAN["operations"], MINIMAL_METADATA)
    roughing_steps = [
        op["step"] for op in MINIMAL_PLAN["operations"] if op["phase"] == "roughing"
    ]
    roughing_total = sum(volumes[step] for step in roughing_steps)
    assert abs(roughing_total - total_removal) < 1.0


def test_finishing_volume_is_smaller_than_roughing():
    volumes = estimate_removal_volumes(MINIMAL_PLAN["operations"], MINIMAL_METADATA)
    roughing = [volumes[op["step"]] for op in MINIMAL_PLAN["operations"] if op["phase"] == "roughing"]
    finishing = [volumes[op["step"]] for op in MINIMAL_PLAN["operations"] if op["phase"] == "finishing"]
    if roughing and finishing:
        assert max(finishing) < max(roughing)


def test_removal_volume_uses_feature_instance_voxels_when_available():
    metadata = {**MINIMAL_METADATA, "resolution": 32}
    operations = [
        {
            "step": 1,
            "feature_type": "rectangular_pocket",
            "phase": "roughing",
            "feature_volume_voxels": 27,
        }
    ]
    volumes = estimate_removal_volumes(operations, metadata)
    pitch = max(metadata["bounding_box_mm"].values()) / (metadata["resolution"] - 2)
    assert abs(volumes[1] - 27 * pitch**3) < 0.01


def test_operation_time_returns_required_keys():
    result = estimate_operation_time(
        MINIMAL_PLAN["operations"][0], removal_volume_mm3=5000.0, material="aluminium_6061"
    )
    for key in [
        "mrr_mm3_per_min",
        "machining_time_min",
        "tool_change_time_min",
        "operation_time_min",
    ]:
        assert key in result


def test_operation_time_all_positive():
    result = estimate_operation_time(MINIMAL_PLAN["operations"][0], 5000.0, "aluminium_6061")
    assert result["machining_time_min"] >= 0.1
    assert result["operation_time_min"] >= result["machining_time_min"]


def test_zero_volume_gives_minimum_time():
    result = estimate_operation_time(MINIMAL_PLAN["operations"][0], 0.0, "aluminium_6061")
    assert result["machining_time_min"] >= 0.1


def test_output_file_created(plan_files):
    estimate_time(plan_files["plan"], plan_files["metadata"], plan_files["out"])
    assert os.path.exists(os.path.join(plan_files["out"], "time_estimate.json"))


def test_schema_keys_present(plan_files):
    result = estimate_time(plan_files["plan"], plan_files["metadata"], plan_files["out"])
    for key in [
        "total_time_min",
        "machining_time_min",
        "setup_time_min",
        "breakdown",
        "material",
        "setup_count",
        "tool_change_count",
        "assumptions",
        "time_estimate_file",
        "warnings",
    ]:
        assert key in result


def test_total_time_equals_components(plan_files):
    result = estimate_time(plan_files["plan"], plan_files["metadata"], plan_files["out"])
    computed = (
        result["machining_time_min"]
        + result["setup_time_min"]
        + result["tool_change_count"] * TOOL_CHANGE_TIME_MIN
    )
    assert abs(computed - result["total_time_min"]) < 0.1


def test_setup_time_equals_count_times_rate(plan_files):
    result = estimate_time(
        plan_files["plan"], plan_files["metadata"], plan_files["out"], setup_time_min=15.0
    )
    expected = MINIMAL_PLAN["setup_count"] * 15.0
    assert abs(result["setup_time_min"] - expected) < 0.01


def test_breakdown_count_matches_operations(plan_files):
    result = estimate_time(plan_files["plan"], plan_files["metadata"], plan_files["out"])
    assert len(result["breakdown"]) == len(MINIMAL_PLAN["operations"])


def test_invalid_material_raises(plan_files):
    with pytest.raises(ValueError):
        estimate_time(
            plan_files["plan"],
            plan_files["metadata"],
            plan_files["out"],
            material="unobtainium_99",
        )


def test_file_not_found_plan(plan_files):
    with pytest.raises(FileNotFoundError):
        estimate_time("no_plan.json", plan_files["metadata"], plan_files["out"])


def test_file_not_found_metadata(plan_files):
    with pytest.raises(FileNotFoundError):
        estimate_time(plan_files["plan"], "no_meta.json", plan_files["out"])


def test_output_path_absolute(plan_files):
    result = estimate_time(plan_files["plan"], plan_files["metadata"], plan_files["out"])
    assert os.path.isabs(result["time_estimate_file"])


@pytest.mark.skipif(
    not all(os.path.exists(os.path.join(CLI_DIR, filename)) for filename in ["process_plan.json", "metadata.json"]),
    reason="Phase 4 CLI output not available",
)
def test_real_pipeline_time_positive(tmp_path):
    result = estimate_time(
        os.path.join(CLI_DIR, "process_plan.json"),
        os.path.join(CLI_DIR, "metadata.json"),
        str(tmp_path),
    )
    assert result["total_time_min"] > 0
    assert result["machining_time_min"] > 0
