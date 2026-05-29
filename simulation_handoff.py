"""Build a compact handoff contract for machining simulation/toolpath modules."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from typing import Any


HANDOFF_SCHEMA_VERSION = "0.1"
READY_RECOMMENDATION = "READY"
REVIEW_RECOMMENDATION = "REVIEW"
BLOCKED_RECOMMENDATION = "BLOCKED"


def _load_json(path: str, label: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{label} must contain a JSON object.")
    return data


def _write_json_atomic(data: dict, path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=directory, delete=False, suffix=".tmp") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
        tmp = f.name
    os.replace(tmp, path)


def _read_optional_json(path: str | None, label: str) -> dict:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")
    return _load_json(path, label)


def _tool_diameter_mm(tool_type: str, instance: dict | None = None) -> float | None:
    if instance:
        diameter = instance.get("diameter_mm")
        if diameter is not None:
            return float(diameter)
        width = instance.get("width_mm")
        if width is not None:
            return max(1.0, float(width) * 0.4)

    match = re.search(r"_(\d+(?:\.\d+)?)mm$", tool_type or "")
    if match:
        return float(match.group(1))
    defaults = {
        "centre_drill": 3.0,
        "twist_drill": None,
        "flat_endmill": 8.0,
        "shoulder_mill": 12.0,
        "face_mill": 50.0,
        "chamfer_mill": 6.0,
        "ball_endmill": 6.0,
        "boring_bar": None,
        "tap": None,
    }
    return defaults.get(tool_type)


def _feature_key(feature_type: str, instance_id: Any) -> tuple[str, int | None]:
    try:
        return feature_type, int(instance_id)
    except (TypeError, ValueError):
        return feature_type, None


def _instance_lookup(feature_instances: dict) -> dict[tuple[str, int | None], dict]:
    lookup: dict[tuple[str, int | None], dict] = {}
    for instance in feature_instances.get("instances", []):
        if not isinstance(instance, dict):
            continue
        feature_type = str(instance.get("type", ""))
        lookup[_feature_key(feature_type, instance.get("instance_id"))] = instance
    return lookup


def _voxel_volume_mm3(metadata: dict, voxel_shape: int | None = None) -> float | None:
    bbox = metadata.get("bounding_box_mm", {})
    dims = [float(bbox.get(axis, 0.0)) for axis in ("x", "y", "z")]
    if min(dims, default=0.0) <= 0:
        return None
    resolution = int(metadata.get("resolution") or voxel_shape or 64)
    return (dims[0] / resolution) * (dims[1] / resolution) * (dims[2] / resolution)


def _operation_depth_mm(operation: dict, instance: dict | None, metadata: dict) -> float | None:
    if instance:
        if instance.get("depth_mm") is not None:
            return float(instance["depth_mm"])
        if instance.get("depth_voxels") is not None:
            bbox = metadata.get("bounding_box_mm", {})
            resolution = int(metadata.get("resolution") or 64)
            z_pitch = float(bbox.get("z", 0.0)) / max(1, resolution)
            return round(float(instance["depth_voxels"]) * z_pitch, 3)
    feature_type = operation.get("feature_type")
    if feature_type == "flat_face":
        return 1.0
    return None


def _estimated_removal_volume_mm3(operation: dict, instance: dict | None, metadata: dict) -> float | None:
    if instance and instance.get("feature_volume_voxels") is not None:
        voxel_volume = _voxel_volume_mm3(metadata)
        if voxel_volume is not None:
            return round(float(instance["feature_volume_voxels"]) * voxel_volume, 3)
    if operation.get("feature_volume_voxels") is not None:
        voxel_volume = _voxel_volume_mm3(metadata)
        if voxel_volume is not None:
            return round(float(operation["feature_volume_voxels"]) * voxel_volume, 3)
    return None


def _normalise_operations(process_plan: dict, feature_instances: dict, metadata: dict) -> tuple[list[dict], list[dict]]:
    instances = _instance_lookup(feature_instances)
    operations = []
    review_items = []
    for raw in process_plan.get("operations", []):
        feature_type = str(raw.get("feature_type", "unknown"))
        instance_id = raw.get("feature_instance_id")
        instance = instances.get(_feature_key(feature_type, instance_id))
        if instance_id is not None and instance is None:
            review_items.append(
                {
                    "code": "OPERATION_INSTANCE_NOT_FOUND",
                    "severity": "review",
                    "message": f"Operation {raw.get('step')} references missing {feature_type} instance {instance_id}.",
                    "operation_step": raw.get("step"),
                    "feature_type": feature_type,
                    "instance_id": instance_id,
                }
            )
        tool_type = str(raw.get("tool_type", "generic_tool"))
        operations.append(
            {
                "operation_id": int(raw.get("operation_id", raw.get("step", len(operations) + 1))),
                "setup_id": int(raw.get("setup_id", 0)),
                "approach_direction": raw.get("approach_direction", "+Z"),
                "feature_type": feature_type,
                "feature_instance_id": instance_id,
                "operation_type": raw.get("operation_type"),
                "phase": raw.get("phase"),
                "tool_type": tool_type,
                "tool_diameter_mm": raw.get("tool_diameter_mm", _tool_diameter_mm(tool_type, instance)),
                "cut_depth_mm": raw.get("cut_depth_mm", _operation_depth_mm(raw, instance, metadata)),
                "estimated_removal_volume_mm3": raw.get(
                    "estimated_removal_volume_mm3",
                    _estimated_removal_volume_mm3(raw, instance, metadata),
                ),
                "requires_review": bool(
                    raw.get("requires_review")
                    or (instance or {}).get("two_point_five_d_supported") is False
                    or (instance or {}).get("localisation_status") in {"estimated", "unknown"}
                ),
                "notes": raw.get("notes", ""),
            }
        )
    return operations, review_items


def _setup_contract(setup_analysis: dict, process_plan: dict) -> dict:
    setups = setup_analysis.get("setups") or process_plan.get("setups") or []
    return {
        "setup_mode": setup_analysis.get("setup_mode", process_plan.get("setup_mode")),
        "setup_count": setup_analysis.get("setup_count", process_plan.get("setup_count")),
        "axis_requirement": setup_analysis.get("axis_requirement", process_plan.get("axis_requirement")),
        "requires_rotation": bool(setup_analysis.get("requires_rotation", process_plan.get("requires_rotation", False))),
        "two_point_five_d_compatible": setup_analysis.get(
            "two_point_five_d_compatible",
            process_plan.get("two_point_five_d_compatible"),
        ),
        "tool_reach_compatible": setup_analysis.get(
            "tool_reach_compatible",
            process_plan.get("tool_reach_compatible", True),
        ),
        "setups": setups,
    }


def _readiness_checks(
    metadata: dict,
    setup: dict,
    process_plan: dict,
    feature_instances: dict,
    operations: list[dict],
    review_items: list[dict],
) -> dict:
    errors = []
    warnings = []

    if not metadata.get("raw_stock_mm"):
        errors.append("Missing raw_stock_mm in metadata.")
    if not metadata.get("mesh_file") or not os.path.exists(str(metadata.get("mesh_file"))):
        warnings.append("mesh_file is missing or does not exist on disk.")
    if not metadata.get("voxel_file") or not os.path.exists(str(metadata.get("voxel_file"))):
        warnings.append("voxel_file is missing or does not exist on disk.")
    if setup.get("setup_count") != 1:
        review_items.append(
            {
                "code": "SIM_SETUP_NOT_SINGLE",
                "severity": "review",
                "message": f"Simulation handoff expected one baseline setup, found {setup.get('setup_count')}.",
            }
        )
    if setup.get("axis_requirement") != 3:
        review_items.append(
            {
                "code": "SIM_AXIS_REQUIREMENT_NOT_3",
                "severity": "review",
                "message": f"Simulation handoff expected 3-axis baseline, found {setup.get('axis_requirement')}.",
            }
        )
    if setup.get("requires_rotation"):
        review_items.append(
            {
                "code": "SIM_REQUIRES_ROTATION",
                "severity": "review",
                "message": "Setup analysis indicates stock rotation is required.",
            }
        )
    if setup.get("two_point_five_d_compatible") is False:
        review_items.append(
            {
                "code": "SIM_NOT_25D_COMPATIBLE",
                "severity": "review",
                "message": "Setup analysis marked the part as outside the single-setup 2.5D baseline.",
            }
        )
    if setup.get("tool_reach_compatible") is False:
        review_items.append(
            {
                "code": "SIM_TOOL_REACH_REVIEW",
                "severity": "review",
                "message": "Tool reach analysis requires review.",
            }
        )
    if not operations:
        errors.append("No process operations available for simulation.")

    instance_count = int(feature_instances.get("instance_count", len(feature_instances.get("instances", [])) or 0))
    operation_instances = [
        op for op in operations if op.get("feature_type") != "flat_face" and op.get("feature_instance_id") is not None
    ]
    if instance_count > 0 and not operation_instances:
        review_items.append(
            {
                "code": "SIM_OPERATIONS_NOT_INSTANCE_MAPPED",
                "severity": "review",
                "message": "Feature instances exist, but non-face operations are not mapped to instance IDs.",
            }
        )

    blocked = bool(errors)
    review = bool(review_items or warnings)
    return {
        "ready_for_simulation": not blocked and not review,
        "recommendation": BLOCKED_RECOMMENDATION if blocked else (REVIEW_RECOMMENDATION if review else READY_RECOMMENDATION),
        "errors": errors,
        "warnings": warnings,
        "review_items": review_items,
        "review_codes": sorted({str(item.get("code")) for item in review_items if item.get("code")}),
    }


def generate_simulation_input(
    metadata_path: str,
    features_path: str,
    feature_instances_path: str,
    setup_analysis_path: str,
    process_plan_path: str,
    output_dir: str,
    pmi_data_path: str | None = None,
) -> dict:
    """Generate simulation_input.json from stable pipeline artifacts."""
    metadata = _load_json(metadata_path, "metadata")
    features = _load_json(features_path, "features")
    feature_instances = _read_optional_json(feature_instances_path, "feature_instances")
    setup_analysis = _load_json(setup_analysis_path, "setup_analysis")
    process_plan = _load_json(process_plan_path, "process_plan")
    pmi_data = _read_optional_json(pmi_data_path, "pmi_data") if pmi_data_path else {}

    setup = _setup_contract(setup_analysis, process_plan)
    operations, operation_review_items = _normalise_operations(process_plan, feature_instances, metadata)
    review_items = []
    for source in (setup_analysis, process_plan):
        review_items.extend(item for item in source.get("review_items", []) if isinstance(item, dict))
    review_items.extend(operation_review_items)
    readiness = _readiness_checks(metadata, setup, process_plan, feature_instances, operations, review_items)

    output_abs = os.path.abspath(output_dir)
    output_path = os.path.join(output_abs, "simulation_input.json")
    result = {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "handoff_type": "machining_simulation_input",
        "source_files": {
            "metadata": os.path.abspath(metadata_path),
            "features": os.path.abspath(features_path),
            "feature_instances": os.path.abspath(feature_instances_path) if feature_instances_path else None,
            "setup_analysis": os.path.abspath(setup_analysis_path),
            "process_plan": os.path.abspath(process_plan_path),
            "pmi_data": os.path.abspath(pmi_data_path) if pmi_data_path else None,
            "mesh": metadata.get("mesh_file"),
            "voxel": metadata.get("voxel_file"),
        },
        "part": {
            "source_file": metadata.get("source_file"),
            "mesh_file": metadata.get("mesh_file"),
            "voxel_file": metadata.get("voxel_file"),
            "resolution": metadata.get("resolution"),
            "bounding_box_mm": metadata.get("bounding_box_mm", {}),
            "volume_mm3": metadata.get("volume_mm3"),
            "surface_area_mm2": metadata.get("surface_area_mm2"),
        },
        "stock": {
            "raw_stock_mm": metadata.get("raw_stock_mm", {}),
            "stock_allowance_source": "phase1_metadata",
        },
        "material": pmi_data.get("material") or process_plan.get("material"),
        "features": features.get("features", []),
        "feature_instances": feature_instances.get("instances", []),
        "setup": setup,
        "operations": operations,
        "tool_list": process_plan.get("tool_list", sorted({op["tool_type"] for op in operations})),
        "readiness": readiness,
        "simulation_input_file": output_path,
    }
    _write_json_atomic(result, output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build simulation_input.json for machining simulation/toolpath modules.")
    parser.add_argument("metadata_path")
    parser.add_argument("features_path")
    parser.add_argument("feature_instances_path")
    parser.add_argument("setup_analysis_path")
    parser.add_argument("process_plan_path")
    parser.add_argument("output_dir")
    parser.add_argument("--pmi-data", default=None, dest="pmi_data")
    args = parser.parse_args()
    result = generate_simulation_input(
        args.metadata_path,
        args.features_path,
        args.feature_instances_path,
        args.setup_analysis_path,
        args.process_plan_path,
        args.output_dir,
        pmi_data_path=args.pmi_data,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
