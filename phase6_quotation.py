"""Phase 6 factory capability checks and quotation generation."""

from __future__ import annotations

import argparse
import json
import os
import tempfile


MATERIAL_PROPERTIES = {
    "aluminium_6061": {"density_g_per_mm3": 2.70e-3, "price_per_kg": 4.50},
    "mild_steel": {"density_g_per_mm3": 7.85e-3, "price_per_kg": 2.00},
    "stainless_316": {"density_g_per_mm3": 8.00e-3, "price_per_kg": 8.50},
    "titanium_grade5": {"density_g_per_mm3": 4.43e-3, "price_per_kg": 35.00},
}

FALLBACK_MATERIAL = {"density_g_per_mm3": 7.85e-3, "price_per_kg": 2.00}
REQUIRED_FACTORY_KEYS = {
    "machines",
    "materials_available",
    "weekly_capacity_hours",
    "overhead_factor",
    "currency",
}
DEFAULT_TOOL_LIBRARY = [
    {"tool_type": "face_mill", "max_depth_mm": 3.0, "diameter_mm": 50.0},
    {"tool_type": "centre_drill", "max_depth_mm": 10.0, "diameter_mm": 3.0},
    {"tool_type": "twist_drill", "max_depth_mm": 80.0, "diameter_mm": 10.0},
    {"tool_type": "flat_endmill", "max_depth_mm": 50.0, "diameter_mm": 10.0},
    {"tool_type": "shoulder_mill", "max_depth_mm": 60.0, "diameter_mm": 16.0},
    {"tool_type": "chamfer_mill", "max_depth_mm": 20.0, "diameter_mm": 8.0},
    {"tool_type": "ball_endmill", "max_depth_mm": 40.0, "diameter_mm": 8.0},
    {"tool_type": "tap", "max_depth_mm": 40.0, "diameter_mm": 10.0},
    {"tool_type": "boring_bar", "max_depth_mm": 80.0, "diameter_mm": 10.0},
]


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


def _fits_envelope(part: dict, envelope: dict) -> bool:
    return all(float(part.get(axis, 0.0)) <= float(envelope.get(axis, 0.0)) for axis in ("x", "y", "z"))


def check_axis_capability(
    required_axes: int,
    machines: list[dict],
) -> dict:
    capable = [machine for machine in machines if int(machine.get("axes", 0)) >= required_axes]
    if capable:
        best = min(capable, key=lambda machine: (int(machine.get("axes", 0)), float(machine.get("hourly_rate", 0.0))))
        return {
            "pass": True,
            "required": required_axes,
            "best_machine_id": best.get("id"),
            "best_machine_axes": int(best.get("axes", 0)),
        }
    return {
        "pass": False,
        "required": required_axes,
        "best_machine_id": None,
        "best_machine_axes": None,
        "reason": f"Part requires {required_axes}-axis machining. No suitable machine available.",
    }


def check_work_envelope(
    bounding_box_mm: dict,
    machines: list[dict],
    required_axes: int,
) -> dict:
    candidates = [
        machine
        for machine in machines
        if int(machine.get("axes", 0)) >= required_axes
        and _fits_envelope(bounding_box_mm, machine.get("work_envelope_mm", {}))
    ]
    if candidates:
        best = min(candidates, key=lambda machine: float(machine.get("hourly_rate", 0.0)))
        return {"pass": True, "part_mm": bounding_box_mm, "best_machine_id": best.get("id")}
    dims = f"{bounding_box_mm.get('x')} x {bounding_box_mm.get('y')} x {bounding_box_mm.get('z')} mm"
    return {
        "pass": False,
        "part_mm": bounding_box_mm,
        "best_machine_id": None,
        "reason": f"Part bounding box ({dims}) exceeds all suitable work envelopes.",
    }


def check_material_available(
    material: str,
    factory: dict,
) -> dict:
    available = list(factory.get("materials_available", []))
    passed = material in available
    result = {"pass": passed, "material": material, "available": available}
    if not passed:
        result["reason"] = f"Material {material} is not available in this factory."
    return result


def check_capacity(
    total_time_min: float,
    factory: dict,
) -> dict:
    available_min = float(factory.get("weekly_capacity_hours", 0.0)) * 60.0
    utilisation = float(total_time_min) / available_min if available_min > 0 else float("inf")
    passed = float(total_time_min) <= available_min
    result = {
        "pass": passed,
        "required_min": float(total_time_min),
        "available_min": available_min,
        "utilisation_fraction": utilisation,
    }
    if not passed:
        result["reason"] = "Estimated time exceeds available weekly capacity."
    return result


def _base_tool_type(tool_type: str) -> str:
    if tool_type.startswith("twist_drill_"):
        return "twist_drill"
    if tool_type.startswith("flat_endmill_"):
        return "flat_endmill"
    return tool_type


def _tool_library(factory: dict) -> list[dict]:
    library = factory.get("tool_library", DEFAULT_TOOL_LIBRARY)
    return library if isinstance(library, list) else DEFAULT_TOOL_LIBRARY


def check_tooling_available(process_plan: dict, factory: dict) -> dict:
    required = sorted({_base_tool_type(tool) for tool in process_plan.get("tool_list", [])})
    available = sorted(
        {
            str(tool.get("tool_type"))
            for tool in _tool_library(factory)
            if isinstance(tool, dict) and tool.get("tool_type")
        }
    )
    missing = [tool for tool in required if tool not in available]
    result = {
        "pass": not missing,
        "required": required,
        "available": available,
        "missing": missing,
    }
    if missing:
        result["reason"] = f"Factory tool library is missing required tool types: {', '.join(missing)}."
    return result


def _max_tool_reach_by_type(factory: dict) -> dict[str, float]:
    reach: dict[str, float] = {}
    for tool in _tool_library(factory):
        if not isinstance(tool, dict) or not tool.get("tool_type"):
            continue
        reach[str(tool["tool_type"])] = max(
            reach.get(str(tool["tool_type"]), 0.0),
            float(tool.get("max_depth_mm", 0.0)),
        )
    return reach


def check_tool_reach(process_plan: dict, factory: dict) -> dict:
    reach_by_type = _max_tool_reach_by_type(factory)
    failures = []
    for item in process_plan.get("feature_feasibility", []):
        if not isinstance(item, dict):
            continue
        depth = item.get("estimated_depth_mm")
        if depth is None:
            continue
        feature_type = item.get("type")
        preferred_tool = "twist_drill" if feature_type in ("through_hole", "blind_hole") else "flat_endmill"
        if feature_type == "rectangular_step":
            preferred_tool = "shoulder_mill"
        max_reach = reach_by_type.get(preferred_tool)
        if max_reach is not None and float(depth) > max_reach:
            failures.append(
                {
                    "feature_type": feature_type,
                    "instance_id": int(item.get("instance_id", 0)),
                    "required_depth_mm": float(depth),
                    "tool_type": preferred_tool,
                    "available_reach_mm": max_reach,
                }
            )
    result = {"pass": not failures, "failures": failures}
    if failures:
        result["reason"] = "One or more features exceed available tool reach."
    return result


def select_machine(
    factory: dict,
    required_axes: int,
    bounding_box_mm: dict,
) -> dict | None:
    candidates = [
        machine
        for machine in factory.get("machines", [])
        if int(machine.get("axes", 0)) >= required_axes
        and _fits_envelope(bounding_box_mm, machine.get("work_envelope_mm", {}))
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda machine: float(machine.get("hourly_rate", 0.0)))


def compute_cost(
    time_estimate: dict,
    metadata: dict,
    factory: dict,
    material: str,
    machine: dict | None,
) -> dict:
    hourly_rate = float(machine.get("hourly_rate", 0.0)) if machine else 0.0
    machining_cost = float(time_estimate.get("total_time_min", 0.0)) / 60.0 * hourly_rate

    raw_stock = metadata.get("raw_stock_mm", {})
    stock_volume = float(raw_stock.get("x", 0.0) * raw_stock.get("y", 0.0) * raw_stock.get("z", 0.0))
    material_props = MATERIAL_PROPERTIES.get(material, FALLBACK_MATERIAL)
    material_mass_kg = stock_volume * float(material_props["density_g_per_mm3"]) / 1000.0
    material_cost = material_mass_kg * float(material_props["price_per_kg"])
    subtotal = machining_cost + material_cost
    overhead = float(factory.get("overhead_factor", 1.0))
    total = subtotal * overhead
    return {
        "machining": machining_cost,
        "material": material_cost,
        "subtotal": subtotal,
        "overhead_factor": overhead,
        "total": total,
        "currency": factory.get("currency", ""),
    }


def _validate_factory(factory: dict) -> None:
    missing = sorted(REQUIRED_FACTORY_KEYS - set(factory.keys()))
    if missing:
        raise ValueError(f"Factory profile missing required keys: {', '.join(missing)}")
    if not isinstance(factory.get("machines"), list):
        raise ValueError("Factory profile 'machines' must be a list.")


def generate_quotation(
    process_plan_path: str,
    time_estimate_path: str,
    metadata_path: str,
    factory_profile_path: str,
    output_dir: str,
    material: str = "aluminium_6061",
) -> dict:
    """Generate final job quotation with capability checks."""
    process_plan = _load_json(process_plan_path, "process_plan")
    time_estimate = _load_json(time_estimate_path, "time_estimate")
    metadata = _load_json(metadata_path, "metadata")
    factory = _load_json(factory_profile_path, "factory_profile")
    _validate_factory(factory)

    warnings: list[str] = []
    if material not in MATERIAL_PROPERTIES:
        warnings.append(f"Material properties missing for {material}; using steel fallback.")

    required_axes = int(process_plan.get("axis_requirement", 3))
    bounding_box = metadata.get("bounding_box_mm", {})
    machines = factory.get("machines", [])

    axis_check = check_axis_capability(required_axes, machines)
    envelope_check = check_work_envelope(bounding_box, machines, required_axes)
    material_check = check_material_available(material, factory)
    capacity_check = check_capacity(float(time_estimate.get("total_time_min", 0.0)), factory)
    tooling_check = check_tooling_available(process_plan, factory)
    tool_reach_check = check_tool_reach(process_plan, factory)
    capability_checks = {
        "axis_capability": axis_check,
        "work_envelope": envelope_check,
        "material_available": material_check,
        "capacity": capacity_check,
        "tooling_available": tooling_check,
        "tool_reach": tool_reach_check,
    }

    machine = select_machine(factory, required_axes, bounding_box)
    if machine is None:
        warnings.append("No suitable machine selected; machining cost set to 0.")
    cost = compute_cost(time_estimate, metadata, factory, material, machine)

    failed_checks = [check for check in capability_checks.values() if not check.get("pass")]
    flags = [check.get("reason", "Capability check failed.") for check in failed_checks]
    review_items = [
        item for item in process_plan.get("review_items", []) if isinstance(item, dict)
    ]
    review_flags: list[str] = [str(item.get("message")) for item in review_items if item.get("message")]
    if process_plan.get("two_point_five_d_compatible") is False:
        review_flags.append("Part violates the single-setup +Z 2.5D machining baseline.")
    if process_plan.get("requires_rotation"):
        review_flags.append("Part appears to require stock rotation.")
    if process_plan.get("tool_reach_compatible") is False:
        review_flags.append("Part violates the baseline tool-reach assumptions.")
    review_flags.extend(str(reason) for reason in process_plan.get("unsupported_reasons", []))
    review_flags = list(dict.fromkeys(review_flags))
    review_codes = sorted(
        {
            str(item.get("code"))
            for item in review_items
            if item.get("code")
        }
        | set(str(code) for code in process_plan.get("review_codes", []))
    )
    flags.extend(flag for flag in review_flags if flag not in flags)
    if failed_checks:
        recommendation = "REJECT"
    elif review_flags:
        recommendation = "REVIEW"
    else:
        recommendation = "ACCEPT"

    output_abs = os.path.abspath(output_dir)
    os.makedirs(output_abs, exist_ok=True)
    quotation_file = os.path.join(output_abs, "quotation.json")
    result = {
        "recommendation": recommendation,
        "flags": flags,
        "estimated_cost": cost,
        "time_summary": {
            "total_min": float(time_estimate.get("total_time_min", 0.0)),
            "machining_min": float(time_estimate.get("machining_time_min", 0.0)),
            "setup_min": float(time_estimate.get("setup_time_min", 0.0)),
        },
        "machine_selected": machine.get("id") if machine else None,
        "capability_checks": capability_checks,
        "factory_name": factory.get("factory_name", ""),
        "material": material,
        "axis_required": required_axes,
        "review_required": recommendation == "REVIEW",
        "review_items": review_items,
        "review_codes": review_codes,
        "source_files": {
            "process_plan": os.path.abspath(process_plan_path),
            "time_estimate": os.path.abspath(time_estimate_path),
            "metadata": os.path.abspath(metadata_path),
            "factory_profile": os.path.abspath(factory_profile_path),
        },
        "quotation_file": quotation_file,
        "warnings": warnings,
    }
    _write_json_atomic(result, quotation_file)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate factory quotation.")
    parser.add_argument("process_plan_path")
    parser.add_argument("time_estimate_path")
    parser.add_argument("metadata_path")
    parser.add_argument("factory_profile_path")
    parser.add_argument("output_dir")
    parser.add_argument("--material", default="aluminium_6061")
    args = parser.parse_args()
    result = generate_quotation(
        args.process_plan_path,
        args.time_estimate_path,
        args.metadata_path,
        args.factory_profile_path,
        args.output_dir,
        material=args.material,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
