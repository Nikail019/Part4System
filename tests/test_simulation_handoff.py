import json
import os

from simulation_handoff import generate_simulation_input


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def base_inputs(tmp_path):
    mesh = tmp_path / "mesh.stl"
    voxel = tmp_path / "voxel_64.npy"
    mesh.write_text("solid test\nendsolid test\n", encoding="utf-8")
    voxel.write_bytes(b"placeholder")

    metadata = {
        "source_file": "/abs/part.stp",
        "resolution": 64,
        "bounding_box_mm": {"x": 100.0, "y": 60.0, "z": 40.0},
        "volume_mm3": 200000.0,
        "surface_area_mm2": 18000.0,
        "raw_stock_mm": {"x": 110.0, "y": 70.0, "z": 45.0},
        "mesh_file": str(mesh),
        "voxel_file": str(voxel),
    }
    features = {
        "features": [
            {"type": "flat_face", "confidence": 1.0},
            {"type": "through_hole", "confidence": 0.92},
        ]
    }
    instances = {
        "instances": [
            {
                "type": "through_hole",
                "instance_id": 0,
                "confidence": 0.92,
                "volume_voxels": 120,
                "feature_volume_voxels": 120,
                "localisation_status": "localised",
                "diameter_mm": 10.0,
                "depth_mm": 40.0,
                "primary_direction": "+Z",
            }
        ],
        "instance_count": 1,
    }
    setup = {
        "setup_mode": "2.5d_single_setup",
        "setup_count": 1,
        "axis_requirement": 3,
        "requires_rotation": False,
        "two_point_five_d_compatible": True,
        "tool_reach_compatible": True,
        "setups": [{"id": 0, "approach_direction": "+Z", "rotation_from_previous": "initial"}],
    }
    plan = {
        "operations": [
            {
                "step": 1,
                "setup_id": 0,
                "approach_direction": "+Z",
                "feature_type": "through_hole",
                "feature_instance_id": 0,
                "operation_type": "drill",
                "tool_type": "twist_drill_10mm",
                "phase": "roughing",
                "notes": "Drill through-hole",
            }
        ],
        "operation_count": 1,
        "setup_count": 1,
        "axis_requirement": 3,
        "tool_list": ["twist_drill_10mm"],
    }
    pmi = {"material": "aluminium_6061", "features": []}

    paths = {}
    for name, data in {
        "metadata": metadata,
        "features": features,
        "feature_instances": instances,
        "setup_analysis": setup,
        "process_plan": plan,
        "pmi_data": pmi,
    }.items():
        path = tmp_path / f"{name}.json"
        write_json(path, data)
        paths[name] = str(path)
    return paths


def test_generate_simulation_input_ready_contract(tmp_path):
    paths = base_inputs(tmp_path)

    result = generate_simulation_input(
        paths["metadata"],
        paths["features"],
        paths["feature_instances"],
        paths["setup_analysis"],
        paths["process_plan"],
        str(tmp_path),
        pmi_data_path=paths["pmi_data"],
    )

    assert os.path.exists(tmp_path / "simulation_input.json")
    assert result["handoff_type"] == "machining_simulation_input"
    assert result["readiness"]["recommendation"] == "READY"
    assert result["readiness"]["ready_for_simulation"] is True
    assert result["operations"][0]["operation_id"] == 1
    assert result["operations"][0]["tool_diameter_mm"] == 10.0
    assert result["operations"][0]["cut_depth_mm"] == 40.0
    assert result["operations"][0]["estimated_removal_volume_mm3"] is not None


def test_generate_simulation_input_reviews_rotation(tmp_path):
    paths = base_inputs(tmp_path)
    setup = json.loads((tmp_path / "setup_analysis.json").read_text())
    setup["requires_rotation"] = True
    setup["two_point_five_d_compatible"] = False
    write_json(tmp_path / "setup_analysis.json", setup)

    result = generate_simulation_input(
        paths["metadata"],
        paths["features"],
        paths["feature_instances"],
        paths["setup_analysis"],
        paths["process_plan"],
        str(tmp_path),
        pmi_data_path=paths["pmi_data"],
    )

    assert result["readiness"]["recommendation"] == "REVIEW"
    assert "SIM_REQUIRES_ROTATION" in result["readiness"]["review_codes"]
    assert "SIM_NOT_25D_COMPATIBLE" in result["readiness"]["review_codes"]


def test_generate_simulation_input_reviews_missing_instance_reference(tmp_path):
    paths = base_inputs(tmp_path)
    plan = json.loads((tmp_path / "process_plan.json").read_text())
    plan["operations"][0]["feature_instance_id"] = 99
    write_json(tmp_path / "process_plan.json", plan)

    result = generate_simulation_input(
        paths["metadata"],
        paths["features"],
        paths["feature_instances"],
        paths["setup_analysis"],
        paths["process_plan"],
        str(tmp_path),
        pmi_data_path=paths["pmi_data"],
    )

    assert result["readiness"]["recommendation"] == "REVIEW"
    assert "OPERATION_INSTANCE_NOT_FOUND" in result["readiness"]["review_codes"]
