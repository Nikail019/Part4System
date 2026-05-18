"""Phase 4 rule-based process plan generation."""

from __future__ import annotations

import argparse
import json
import os
import tempfile


OPERATION_MAP = {
    "flat_face": [
        {"type": "face_mill_rough", "tool": "face_mill", "phase": "roughing"},
        {"type": "face_mill_finish", "tool": "face_mill", "phase": "finishing"},
    ],
    "through_hole": [
        {"type": "centre_drill", "tool": "centre_drill", "phase": "roughing"},
        {"type": "drill", "tool": "twist_drill", "phase": "roughing"},
    ],
    "blind_hole": [
        {"type": "centre_drill", "tool": "centre_drill", "phase": "roughing"},
        {"type": "drill_peck", "tool": "twist_drill", "phase": "roughing"},
    ],
    "rectangular_pocket": [
        {"type": "endmill_rough", "tool": "flat_endmill", "phase": "roughing"},
        {"type": "endmill_finish", "tool": "flat_endmill", "phase": "finishing"},
    ],
    "circular_pocket": [
        {"type": "endmill_rough", "tool": "flat_endmill", "phase": "roughing"},
        {"type": "endmill_finish", "tool": "flat_endmill", "phase": "finishing"},
    ],
    "rectangular_slot": [
        {"type": "endmill_rough", "tool": "flat_endmill", "phase": "roughing"},
        {"type": "endmill_finish", "tool": "flat_endmill", "phase": "finishing"},
    ],
    "circular_slot": [
        {"type": "endmill_rough", "tool": "flat_endmill", "phase": "roughing"},
        {"type": "endmill_finish", "tool": "flat_endmill", "phase": "finishing"},
    ],
    "rectangular_step": [
        {"type": "shoulder_mill_rough", "tool": "shoulder_mill", "phase": "roughing"},
        {"type": "shoulder_mill_finish", "tool": "shoulder_mill", "phase": "finishing"},
    ],
    "chamfer": [
        {"type": "chamfer_mill", "tool": "chamfer_mill", "phase": "finishing"},
    ],
    "fillet": [
        {"type": "ball_endmill_finish", "tool": "ball_endmill", "phase": "finishing"},
    ],
    "boss": [
        {"type": "endmill_rough", "tool": "flat_endmill", "phase": "roughing"},
        {"type": "endmill_finish", "tool": "flat_endmill", "phase": "finishing"},
    ],
    "triangular_pocket": [
        {"type": "endmill_rough", "tool": "flat_endmill", "phase": "roughing"},
        {"type": "endmill_finish", "tool": "flat_endmill", "phase": "finishing"},
    ],
}

ROUGHING_PRIORITY = {
    "flat_face": 0,
    "rectangular_step": 1,
    "boss": 2,
    "rectangular_pocket": 3,
    "circular_pocket": 4,
    "triangular_pocket": 5,
    "rectangular_slot": 6,
    "circular_slot": 7,
    "blind_hole": 8,
    "through_hole": 9,
    "chamfer": 10,
    "fillet": 11,
}

OPERATION_NOTES = {
    "face_mill_rough": "Establish datum reference surface",
    "face_mill_finish": "Achieve final face flatness",
    "centre_drill": "Spot drill for hole location accuracy",
    "drill": "Drill through-hole to nominal diameter",
    "drill_peck": "Peck drill blind hole to depth",
    "endmill_rough": "Rough pocket / slot to within 0.5mm of final depth",
    "endmill_finish": "Finish to final profile",
    "shoulder_mill_rough": "Rough shoulder step",
    "shoulder_mill_finish": "Finish step to final dimension",
    "chamfer_mill": "Apply chamfer to edges",
    "ball_endmill_finish": "Blend fillet radius",
}


def _load_json(path: str, label: str) -> dict:
    """Load JSON file, raising explicit file or parse errors."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")
    with open(path, encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {label}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} must contain a JSON object.")
    return data


def _expand_feature(
    feature_type: str,
    setup_id: int,
    approach_direction: str,
) -> list[dict]:
    """Expand a feature type to operation dicts for a setup."""
    operations = []
    for spec in OPERATION_MAP.get(feature_type, []):
        op_type = spec["type"]
        operations.append(
            {
                "setup_id": int(setup_id),
                "approach_direction": approach_direction,
                "feature_type": feature_type,
                "operation_type": op_type,
                "tool_type": spec["tool"],
                "phase": spec["phase"],
                "notes": OPERATION_NOTES.get(op_type, ""),
            }
        )
    return operations


def _dedupe_features(features: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for feature in features:
        if feature in seen:
            continue
        seen.add(feature)
        deduped.append(feature)
    return deduped


def _phase_features(features: list[str], phase: str) -> list[str]:
    result = []
    for feature in features:
        ops = OPERATION_MAP.get(feature, [])
        if any(op["phase"] == phase for op in ops):
            result.append(feature)
    return sorted(result, key=lambda name: ROUGHING_PRIORITY.get(name, 99))


def _build_operations(
    features_per_setup: dict[str, list[str]],
    setup_list: list[dict],
) -> tuple[list[dict], list[str]]:
    """Apply setup, phase, and feature-priority sequencing rules."""
    operations: list[dict] = []
    warnings: list[str] = []

    setup_order = sorted(setup_list, key=lambda setup: int(setup.get("id", 0)))
    for setup in setup_order:
        setup_id = int(setup["id"])
        approach_direction = setup.get("approach_direction", "")
        raw_features = features_per_setup.get(str(setup_id), [])
        features = _dedupe_features(raw_features)

        for feature in features:
            if feature not in OPERATION_MAP:
                warnings.append(f"No operation mapping for feature: {feature}")

        roughing_features = _phase_features(features, "roughing")
        finishing_features = _phase_features(features, "finishing")
        for phase_feature_list, phase in (
            (roughing_features, "roughing"),
            (finishing_features, "finishing"),
        ):
            for feature in phase_feature_list:
                expanded = _expand_feature(feature, setup_id, approach_direction)
                operations.extend(op for op in expanded if op["phase"] == phase)

    return operations, warnings


def _resolve_features_per_setup(
    setup_analysis: dict,
    features: list[dict],
) -> dict[str, list[str]]:
    """Use Phase 3 feature mapping, or fall back to setup 0 if empty."""
    fps = setup_analysis.get("features_per_setup", {})
    if not isinstance(fps, dict):
        fps = {}

    setups = setup_analysis.get("setups", [])
    result = {str(setup["id"]): list(fps.get(str(setup["id"]), [])) for setup in setups}
    all_empty = all(len(values) == 0 for values in result.values())

    if all_empty and features:
        if not result:
            result = {"0": []}
        result["0"] = [feature["type"] for feature in features if "type" in feature]
    return result


def _write_json_atomic(data: dict, path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=directory, delete=False, suffix=".tmp") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
        tmp = f.name
    os.replace(tmp, path)


def _validate_inputs(metadata: dict, features: dict, setup_analysis: dict) -> None:
    if "features" not in features or not isinstance(features["features"], list):
        raise ValueError("features JSON must contain a 'features' list.")
    if "setups" not in setup_analysis or not isinstance(setup_analysis["setups"], list):
        raise ValueError("setup_analysis JSON must contain a 'setups' list.")
    if "setup_count" not in setup_analysis:
        raise ValueError("setup_analysis JSON missing 'setup_count'.")
    if "axis_requirement" not in setup_analysis:
        raise ValueError("setup_analysis JSON missing 'axis_requirement'.")
    if not isinstance(metadata, dict):
        raise ValueError("metadata JSON must contain an object.")


def _filter_features(features: list[dict], threshold: float) -> tuple[list[dict], list[str]]:
    kept = []
    warnings = []
    for feature in features:
        feature_type = feature.get("type")
        confidence = float(feature.get("confidence", 0.0))
        if confidence >= threshold:
            kept.append(feature)
        else:
            warnings.append(
                f"Excluded feature below confidence threshold: {feature_type} ({confidence:.3f})"
            )
    if not kept:
        warnings.append("No features remain after confidence filtering.")
    elif not any(feature.get("type") == "flat_face" for feature in kept):
        kept.insert(0, {"type": "flat_face", "confidence": 1.0})
        warnings.append("Added flat_face baseline operation because every stock setup needs a datum face.")
    return kept, warnings


def _filter_features_per_setup(
    features_per_setup: dict[str, list[str]],
    features: list[dict],
) -> dict[str, list[str]]:
    allowed = {feature["type"] for feature in features if "type" in feature}
    return {
        setup_id: [feature for feature in feature_list if feature in allowed]
        for setup_id, feature_list in features_per_setup.items()
    }


def generate_process_plan(
    metadata_path: str,
    features_path: str,
    setup_analysis_path: str,
    output_dir: str,
    confidence_threshold: float = 0.5,
) -> dict:
    """Generate a sequenced process plan from Phase 1-3 outputs."""
    metadata = _load_json(metadata_path, "metadata")
    features_json = _load_json(features_path, "features")
    setup_analysis = _load_json(setup_analysis_path, "setup_analysis")
    _validate_inputs(metadata, features_json, setup_analysis)

    filtered_features, warnings = _filter_features(
        features_json["features"], confidence_threshold
    )
    features_per_setup = _resolve_features_per_setup(setup_analysis, filtered_features)
    features_per_setup = _filter_features_per_setup(features_per_setup, filtered_features)
    original_fps = setup_analysis.get("features_per_setup", {})
    if filtered_features and all(len(values) == 0 for values in original_fps.values()):
        warnings.append("features_per_setup was empty; assigned detected features to setup 0.")

    raw_operations, build_warnings = _build_operations(
        features_per_setup, setup_analysis["setups"]
    )
    warnings.extend(build_warnings)

    operations = []
    for step, operation in enumerate(raw_operations, start=1):
        operations.append({"step": step, **operation})

    tool_list = sorted({operation["tool_type"] for operation in operations})
    output_abs = os.path.abspath(output_dir)
    os.makedirs(output_abs, exist_ok=True)
    process_plan_file = os.path.join(output_abs, "process_plan.json")

    result = {
        "operations": operations,
        "operation_count": len(operations),
        "setup_count": int(setup_analysis["setup_count"]),
        "axis_requirement": int(setup_analysis["axis_requirement"]),
        "tool_list": tool_list,
        "source_files": {
            "metadata": os.path.abspath(metadata_path),
            "features": os.path.abspath(features_path),
            "setup_analysis": os.path.abspath(setup_analysis_path),
        },
        "process_plan_file": process_plan_file,
        "warnings": warnings,
    }
    _write_json_atomic(result, process_plan_file)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a sequenced process plan from Phase 1-3 outputs."
    )
    parser.add_argument("metadata_path", help="Path to metadata.json from Phase 1")
    parser.add_argument("features_path", help="Path to features.json from Phase 2")
    parser.add_argument("setup_analysis_path", help="Path to setup_analysis.json from Phase 3")
    parser.add_argument("output_dir", help="Directory to write process_plan.json")
    parser.add_argument("--confidence", type=float, default=0.5)
    args = parser.parse_args()

    result = generate_process_plan(
        args.metadata_path,
        args.features_path,
        args.setup_analysis_path,
        args.output_dir,
        confidence_threshold=args.confidence,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
