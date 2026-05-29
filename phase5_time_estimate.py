"""Phase 5 machining time estimation."""

from __future__ import annotations

import argparse
import json
import os
import tempfile


MRR_TABLE = {
    "aluminium_6061": {
        "face_mill_rough": 8000,
        "face_mill_finish": 2000,
        "endmill_rough": 5000,
        "endmill_finish": 800,
        "shoulder_mill_rough": 4000,
        "shoulder_mill_finish": 600,
        "centre_drill": 500,
        "drill": 3000,
        "drill_peck": 2000,
        "chamfer_mill": 300,
        "ball_endmill_finish": 400,
    },
    "mild_steel": {
        "face_mill_rough": 3000,
        "face_mill_finish": 800,
        "endmill_rough": 1500,
        "endmill_finish": 300,
        "shoulder_mill_rough": 1200,
        "shoulder_mill_finish": 200,
        "centre_drill": 200,
        "drill": 1200,
        "drill_peck": 800,
        "chamfer_mill": 100,
        "ball_endmill_finish": 150,
    },
    "stainless_316": {
        "face_mill_rough": 1500,
        "face_mill_finish": 400,
        "endmill_rough": 800,
        "endmill_finish": 150,
        "shoulder_mill_rough": 600,
        "shoulder_mill_finish": 100,
        "centre_drill": 150,
        "drill": 600,
        "drill_peck": 400,
        "chamfer_mill": 80,
        "ball_endmill_finish": 100,
    },
    "titanium_grade5": {
        "face_mill_rough": 500,
        "face_mill_finish": 150,
        "endmill_rough": 300,
        "endmill_finish": 60,
        "shoulder_mill_rough": 250,
        "shoulder_mill_finish": 50,
        "centre_drill": 80,
        "drill": 250,
        "drill_peck": 150,
        "chamfer_mill": 40,
        "ball_endmill_finish": 50,
    },
}

DEFAULT_MATERIAL = "aluminium_6061"
TOOL_CHANGE_TIME_MIN = 2.0
SETUP_TIME_MIN = 15.0
DEFAULT_MRR_FALLBACK = 1000

FEATURE_VOLUME_WEIGHT = {
    "flat_face": 0.05,
    "rectangular_step": 0.30,
    "boss": 0.25,
    "rectangular_pocket": 0.20,
    "circular_pocket": 0.12,
    "triangular_pocket": 0.15,
    "rectangular_slot": 0.18,
    "circular_slot": 0.10,
    "blind_hole": 0.03,
    "through_hole": 0.03,
    "chamfer": 0.01,
    "fillet": 0.01,
}

FEATURE_SURFACE_FRACTION = {
    "flat_face": 0.30,
    "rectangular_step": 0.20,
    "boss": 0.15,
    "rectangular_pocket": 0.20,
    "circular_pocket": 0.12,
    "triangular_pocket": 0.15,
    "rectangular_slot": 0.15,
    "circular_slot": 0.10,
    "blind_hole": 0.05,
    "through_hole": 0.05,
    "chamfer": 0.05,
    "fillet": 0.05,
}

FINISHING_DOC_MM = 0.5


def _load_json(path: str, label: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json_atomic(data: dict, path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=directory, delete=False, suffix=".tmp") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
        tmp = f.name
    os.replace(tmp, path)


def _stock_volume(metadata: dict) -> float:
    raw_stock = metadata.get("raw_stock_mm", {})
    return float(raw_stock.get("x", 0.0) * raw_stock.get("y", 0.0) * raw_stock.get("z", 0.0))


def _voxel_volume_mm3(metadata: dict) -> float:
    bbox = metadata.get("bounding_box_mm", {})
    longest = max(float(bbox.get(axis, 0.0)) for axis in ("x", "y", "z"))
    resolution = int(metadata.get("resolution", 0))
    if longest <= 0 or resolution <= 2:
        return 0.0
    pitch = longest / float(resolution - 2)
    return pitch**3


def estimate_removal_volumes(
    operations: list[dict],
    metadata: dict,
) -> dict[int, float]:
    """Estimate material removal volume in mm3 for each operation step."""
    raw_stock_volume = _stock_volume(metadata)
    part_volume = float(metadata.get("volume_mm3", 0.0))
    total_removal = max(0.0, raw_stock_volume - part_volume)
    surface_area = float(metadata.get("surface_area_mm2", 0.0))
    voxel_volume = _voxel_volume_mm3(metadata)

    roughing_ops = [op for op in operations if op.get("phase") == "roughing"]
    roughing_ops_without_instance_volume = [
        op
        for op in roughing_ops
        if int(op.get("feature_volume_voxels", 0)) <= 0 or voxel_volume <= 0
    ]
    explicit_roughing_volume = sum(
        int(op.get("feature_volume_voxels", 0))
        * voxel_volume
        / max(1, int(op.get("pass_count", 1)))
        for op in roughing_ops
        if int(op.get("feature_volume_voxels", 0)) > 0 and voxel_volume > 0
    )
    remaining_removal = max(0.0, total_removal - explicit_roughing_volume)
    roughing_weight_sum = sum(
        FEATURE_VOLUME_WEIGHT.get(op.get("feature_type"), 0.01)
        for op in roughing_ops_without_instance_volume
    )

    volumes: dict[int, float] = {}
    for op in operations:
        step = int(op["step"])
        feature = op.get("feature_type", "")
        if op.get("phase") == "roughing":
            feature_volume_voxels = int(op.get("feature_volume_voxels", 0))
            if feature_volume_voxels > 0 and voxel_volume > 0:
                volume = feature_volume_voxels * voxel_volume / max(1, int(op.get("pass_count", 1)))
            elif roughing_weight_sum > 0:
                weight = FEATURE_VOLUME_WEIGHT.get(feature, 0.01)
                volume = remaining_removal * weight / roughing_weight_sum
            else:
                volume = 0.0
        else:
            fraction = FEATURE_SURFACE_FRACTION.get(feature, 0.02)
            volume = surface_area * fraction * FINISHING_DOC_MM
        volumes[step] = max(0.0, float(volume))
    return volumes


def estimate_operation_time(
    operation: dict,
    removal_volume_mm3: float,
    material: str,
) -> dict:
    """Estimate machining time for one operation, excluding caller tool context."""
    if material not in MRR_TABLE:
        raise ValueError(f"Unknown material: {material}")
    operation_type = operation.get("operation_type", "")
    mrr = float(MRR_TABLE[material].get(operation_type, DEFAULT_MRR_FALLBACK))
    machining_time = max(0.1, float(removal_volume_mm3) / mrr)
    return {
        "mrr_mm3_per_min": mrr,
        "machining_time_min": machining_time,
        "tool_change_time_min": TOOL_CHANGE_TIME_MIN,
        "operation_time_min": machining_time + TOOL_CHANGE_TIME_MIN,
    }


def estimate_time(
    process_plan_path: str,
    metadata_path: str,
    output_dir: str,
    material: str = DEFAULT_MATERIAL,
    setup_time_min: float = SETUP_TIME_MIN,
    tool_change_time_min: float = TOOL_CHANGE_TIME_MIN,
) -> dict:
    """Estimate total machining time from a Phase 4 process plan."""
    if material not in MRR_TABLE:
        raise ValueError(f"Unknown material: {material}")

    process_plan = _load_json(process_plan_path, "process_plan")
    metadata = _load_json(metadata_path, "metadata")
    operations = process_plan.get("operations", [])
    volumes = estimate_removal_volumes(operations, metadata)

    breakdown = []
    previous_tool = None
    tool_change_count = 0
    machining_time_total = 0.0
    tool_change_total = 0.0
    for op in operations:
        step = int(op["step"])
        volume = volumes.get(step, 0.0)
        time_info = estimate_operation_time(op, volume, material)
        tool_changed = op.get("tool_type") != previous_tool
        if tool_changed:
            time_info["tool_change_time_min"] = float(tool_change_time_min)
            tool_change_count += 1
        else:
            time_info["tool_change_time_min"] = 0.0
        time_info["operation_time_min"] = (
            time_info["machining_time_min"] + time_info["tool_change_time_min"]
        )

        machining_time_total += time_info["machining_time_min"]
        tool_change_total += time_info["tool_change_time_min"]
        previous_tool = op.get("tool_type")

        breakdown.append(
            {
                "step": step,
                "operation_type": op.get("operation_type"),
                "feature_type": op.get("feature_type"),
                "feature_instance_id": op.get("feature_instance_id"),
                "phase": op.get("phase"),
                "estimated_removal_volume_mm3": volume,
                **time_info,
            }
        )

    setup_count = int(process_plan.get("setup_count", 0))
    setup_total = setup_count * float(setup_time_min)
    total_time = machining_time_total + tool_change_total + setup_total
    output_abs = os.path.abspath(output_dir)
    os.makedirs(output_abs, exist_ok=True)
    time_file = os.path.join(output_abs, "time_estimate.json")

    result = {
        "total_time_min": total_time,
        "machining_time_min": machining_time_total,
        "setup_time_min": setup_total,
        "breakdown": breakdown,
        "material": material,
        "setup_count": setup_count,
        "setup_time_per_setup_min": float(setup_time_min),
        "tool_change_count": tool_change_count,
        "tool_change_time_min": float(tool_change_time_min),
        "assumptions": [
            "Roughing volume uses feature instance voxel volume when available",
            "Remaining removal volume distributed by feature type weight heuristic",
            f"Cutting parameters for {material} from standard tables",
            f"{float(setup_time_min)} min per setup for workholding and alignment",
            f"{float(tool_change_time_min)} min per tool change",
        ],
        "metadata_file": os.path.abspath(metadata_path),
        "process_plan_file": os.path.abspath(process_plan_path),
        "time_estimate_file": time_file,
        "warnings": [],
    }
    _write_json_atomic(result, time_file)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate machining time from a process plan.")
    parser.add_argument("process_plan_path")
    parser.add_argument("metadata_path")
    parser.add_argument("output_dir")
    parser.add_argument("--material", default=DEFAULT_MATERIAL)
    args = parser.parse_args()
    result = estimate_time(
        args.process_plan_path,
        args.metadata_path,
        args.output_dir,
        material=args.material,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
