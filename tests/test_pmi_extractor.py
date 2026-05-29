import json
import os

import pytest

from phase4_process_plan import (
    _round_to_standard_drill,
    _select_endmill_size,
    generate_process_plan,
)
from step_pmi_extractor import (
    DEFAULT_RA,
    MATERIAL_NAME_MAP,
    PECK_DEPTH_RATIO,
    assemble_feature_attributes,
    extract_pmi,
    measure_brep_features,
    parse_step_pmi,
)


SIMPLE_BLOCK = "tests/fixtures/simple_block.stp"
BLOCK_HOLES = "tests/fixtures/block_with_holes.stp"
COMPLEX = "tests/fixtures/complex_prismatic.stp"
CLI_DIR = "data/processed/simple_block_cli"
CLI_FEATURES = os.path.join(CLI_DIR, "features.json")


def test_constants_importable():
    assert DEFAULT_RA["flat_face"] > 0
    assert MATERIAL_NAME_MAP["6061"] == "aluminium_6061"
    assert PECK_DEPTH_RATIO == 3.0


def test_parse_step_pmi_returns_schema():
    result = parse_step_pmi(SIMPLE_BLOCK)
    assert set(result) == {"material", "surface_finish", "threads"}


def test_parse_step_pmi_never_raises():
    for path in [SIMPLE_BLOCK, BLOCK_HOLES, COMPLEX, "no_such_file.stp"]:
        result = parse_step_pmi(path)
        assert isinstance(result, dict)


def test_measure_brep_bounding_box_positive():
    result = measure_brep_features(SIMPLE_BLOCK)
    bb = result["bounding_box_mm"]
    assert bb["x"] > 0
    assert bb["y"] > 0
    assert bb["z"] > 0


def test_measure_brep_simple_block_dimensions():
    result = measure_brep_features(SIMPLE_BLOCK)
    dims = sorted(result["bounding_box_mm"].values())
    assert abs(dims[0] - 40) / 40 < 0.05
    assert abs(dims[1] - 60) / 60 < 0.05
    assert abs(dims[2] - 100) / 100 < 0.05


def test_measure_brep_block_with_holes_finds_cylinders():
    result = measure_brep_features(BLOCK_HOLES)
    assert len(result["holes"]) >= 1
    for hole in result["holes"]:
        assert 1.0 <= hole["diameter_mm"] <= 200.0
        assert hole["depth_mm"] > 0


def test_measure_brep_complex_finds_recesses():
    result = measure_brep_features(COMPLEX)
    assert len(result["planar_recesses"]) >= 1


def test_assemble_hole_gets_dimensions_and_peck():
    result = assemble_feature_attributes(
        [{"type": "through_hole", "confidence": 0.85}],
        {"material": None, "surface_finish": [], "threads": []},
        {
            "holes": [{"diameter_mm": 10.0, "depth_mm": 40.0}],
            "planar_recesses": [],
            "bounding_box_mm": {"x": 100, "y": 60, "z": 40},
        },
    )
    feature = result["features"][0]
    assert feature["diameter_mm"] == 10.0
    assert feature["depth_mm"] == 40.0
    assert feature["peck_required"] is True


def test_assemble_repeats_measured_hole_instances():
    result = assemble_feature_attributes(
        [{"type": "through_hole", "confidence": 0.95}],
        {"material": None, "surface_finish": [], "threads": []},
        {
            "holes": [
                {"diameter_mm": 10.0, "depth_mm": 40.0},
                {"diameter_mm": 10.0, "depth_mm": 40.0},
                {"diameter_mm": 10.0, "depth_mm": 40.0},
            ],
            "planar_recesses": [],
            "bounding_box_mm": {"x": 100, "y": 60, "z": 40},
        },
    )
    holes = [feature for feature in result["features"] if feature["type"] == "through_hole"]
    assert len(holes) == 3
    assert [feature["instance_id"] for feature in holes] == [0, 1, 2]


def test_assemble_shallow_hole_no_peck():
    result = assemble_feature_attributes(
        [{"type": "through_hole", "confidence": 0.85}],
        {"material": None, "surface_finish": [], "threads": []},
        {
            "holes": [{"diameter_mm": 10.0, "depth_mm": 10.0}],
            "planar_recesses": [],
            "bounding_box_mm": {"x": 100, "y": 60, "z": 40},
        },
    )
    assert result["features"][0]["peck_required"] is False


def test_assemble_thread_assigned_to_hole():
    result = assemble_feature_attributes(
        [{"type": "through_hole", "confidence": 0.85}],
        {
            "material": None,
            "surface_finish": [],
            "threads": [{"spec": "M10X1.5", "face_ref": "#1"}],
        },
        {
            "holes": [{"diameter_mm": 10.0, "depth_mm": 20.0}],
            "planar_recesses": [],
            "bounding_box_mm": {"x": 100, "y": 60, "z": 40},
        },
    )
    assert result["features"][0]["threaded"] is True
    assert result["features"][0]["thread_spec"] == "M10X1.5"


def test_assemble_material_source():
    result = assemble_feature_attributes(
        [{"type": "flat_face", "confidence": 0.99}],
        {"material": "mild_steel", "surface_finish": [], "threads": []},
        {"holes": [], "planar_recesses": [], "bounding_box_mm": {"x": 100, "y": 60, "z": 40}},
    )
    assert result["material"] == "mild_steel"
    assert result["material_source"] == "pmi"


@pytest.mark.skipif(not os.path.exists(CLI_FEATURES), reason="Phase 2 CLI output not available")
def test_extract_pmi_creates_schema(tmp_path):
    result = extract_pmi(SIMPLE_BLOCK, CLI_FEATURES, str(tmp_path))
    for key in ["source_file", "material", "material_source", "features", "pmi_data_file", "warnings"]:
        assert key in result
    assert os.path.exists(tmp_path / "pmi_data.json")
    assert os.path.isabs(result["pmi_data_file"])


def test_extract_pmi_missing_inputs_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_pmi("no_such.stp", CLI_FEATURES, str(tmp_path))
    with pytest.raises(FileNotFoundError):
        extract_pmi(SIMPLE_BLOCK, "no_features.json", str(tmp_path))


def test_tool_size_helpers():
    assert _round_to_standard_drill(10.2) == 10.0
    assert _select_endmill_size(25.0) == 10


@pytest.mark.skipif(
    not all(
        os.path.exists(os.path.join(CLI_DIR, filename))
        for filename in ["metadata.json", "features.json", "setup_analysis.json"]
    ),
    reason="CLI outputs not available",
)
def test_phase4_pmi_adds_peck_for_deep_hole(tmp_path):
    pmi_path = tmp_path / "pmi_data.json"
    pmi_path.write_text(
        json.dumps(
            {
                "material": "aluminium_6061",
                "material_source": "default",
                "features": [
                    {"type": "flat_face", "instance_id": 0, "Ra_um": 3.2},
                    {
                        "type": "through_hole",
                        "instance_id": 0,
                        "diameter_mm": 10.0,
                        "depth_mm": 40.0,
                        "depth_ratio": 4.0,
                        "Ra_um": 1.6,
                        "threaded": False,
                        "thread_spec": None,
                        "peck_required": True,
                    },
                ],
                "warnings": [],
            }
        )
    )
    result = generate_process_plan(
        os.path.join(CLI_DIR, "metadata.json"),
        os.path.join(CLI_DIR, "features.json"),
        os.path.join(CLI_DIR, "setup_analysis.json"),
        str(tmp_path),
        confidence_threshold=0.0,
        pmi_data_path=str(pmi_path),
    )
    op_types = [op["operation_type"] for op in result["operations"]]
    assert "drill_peck" in op_types


@pytest.mark.skipif(
    not all(
        os.path.exists(os.path.join(CLI_DIR, filename))
        for filename in ["metadata.json", "features.json", "setup_analysis.json"]
    ),
    reason="CLI outputs not available",
)
def test_phase4_pmi_adds_tap_for_threaded_hole(tmp_path):
    pmi_path = tmp_path / "pmi_data.json"
    pmi_path.write_text(
        json.dumps(
            {
                "material": "aluminium_6061",
                "material_source": "default",
                "features": [
                    {"type": "flat_face", "instance_id": 0, "Ra_um": 3.2},
                    {
                        "type": "through_hole",
                        "instance_id": 0,
                        "diameter_mm": 10.0,
                        "depth_mm": 20.0,
                        "depth_ratio": 2.0,
                        "Ra_um": 1.6,
                        "threaded": True,
                        "thread_spec": "M10X1.5",
                        "peck_required": False,
                    },
                ],
                "warnings": [],
            }
        )
    )
    result = generate_process_plan(
        os.path.join(CLI_DIR, "metadata.json"),
        os.path.join(CLI_DIR, "features.json"),
        os.path.join(CLI_DIR, "setup_analysis.json"),
        str(tmp_path),
        confidence_threshold=0.0,
        pmi_data_path=str(pmi_path),
    )
    op_types = [op["operation_type"] for op in result["operations"]]
    assert "tap" in op_types
