"""Phase 3 setup and stock rotation analysis from voxel grids."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile

import numpy as np


DIRECTION_LABELS = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
DIRECTION_INDEX = {label: idx for idx, label in enumerate(DIRECTION_LABELS)}
NUM_DIRECTIONS = 6
SETUP_MODE = "2.5d_single_setup"
DEFAULT_APPROACH_DIRECTION = "+Z"
LOCALISATION_REQUIRED_FEATURES = {
    "through_hole",
    "blind_hole",
    "rectangular_pocket",
    "circular_pocket",
    "rectangular_slot",
    "circular_slot",
    "rectangular_step",
    "triangular_pocket",
}
TOOL_REACH_RULES = {
    "through_hole": {"max_depth_to_opening_ratio": 8.0, "max_depth_mm": 80.0},
    "blind_hole": {"max_depth_to_opening_ratio": 6.0, "max_depth_mm": 60.0},
    "rectangular_pocket": {"max_depth_to_opening_ratio": 4.0, "max_depth_mm": 50.0},
    "circular_pocket": {"max_depth_to_opening_ratio": 4.0, "max_depth_mm": 50.0},
    "rectangular_slot": {"max_depth_to_opening_ratio": 5.0, "max_depth_mm": 60.0},
    "circular_slot": {"max_depth_to_opening_ratio": 5.0, "max_depth_mm": 60.0},
    "triangular_pocket": {"max_depth_to_opening_ratio": 4.0, "max_depth_mm": 50.0},
    "rectangular_step": {"max_depth_to_opening_ratio": 5.0, "max_depth_mm": 60.0},
}

ROTATION_DESCRIPTIONS = {
    ("+Z", "-Z"): "flip_around_X_180",
    ("-Z", "+Z"): "flip_around_X_180",
    ("+Z", "+X"): "rotate_around_Y_90",
    ("+Z", "-X"): "rotate_around_Y_minus90",
    ("+Z", "+Y"): "rotate_around_X_minus90",
    ("+Z", "-Y"): "rotate_around_X_90",
    ("+X", "+Y"): "rotate_around_Z_90",
    ("+X", "-Y"): "rotate_around_Z_minus90",
    ("+X", "-X"): "flip_around_Z_180",
    ("+Y", "-Y"): "flip_around_Z_180",
}


def _validate_grid(grid: np.ndarray) -> np.ndarray:
    if grid.ndim != 3 or len(set(grid.shape)) != 1:
        raise ValueError("voxel array must be 3-D and cubic.")
    return grid.astype(bool)


def compute_surface_mask(grid: np.ndarray) -> np.ndarray:
    """Identify occupied voxels with at least one empty face-neighbour."""
    grid = _validate_grid(grid)
    padded = np.pad(grid, pad_width=1, mode="constant", constant_values=False)
    has_empty_neighbour = (
        ~padded[2:, 1:-1, 1:-1]
        | ~padded[:-2, 1:-1, 1:-1]
        | ~padded[1:-1, 2:, 1:-1]
        | ~padded[1:-1, :-2, 1:-1]
        | ~padded[1:-1, 1:-1, 2:]
        | ~padded[1:-1, 1:-1, :-2]
    )
    return grid & has_empty_neighbour


def compute_accessibility_map(grid: np.ndarray) -> np.ndarray:
    """Compute 6-direction line-of-sight accessibility for all voxels."""
    grid = _validate_grid(grid)
    r = grid.shape[0]
    g = grid.astype(np.int32)
    empty = ~grid
    acc = np.ones((NUM_DIRECTIONS, r, r, r), dtype=bool)

    suffix_x = np.zeros_like(g)
    suffix_x[:-1] = np.cumsum(g[::-1], axis=0)[-2::-1]
    acc[DIRECTION_INDEX["+X"]] = suffix_x == 0

    prefix_x = np.zeros_like(g)
    prefix_x[1:] = np.cumsum(g, axis=0)[:-1]
    acc[DIRECTION_INDEX["-X"]] = prefix_x == 0

    suffix_y = np.zeros_like(g)
    suffix_y[:, :-1] = np.cumsum(g[:, ::-1], axis=1)[:, -2::-1]
    acc[DIRECTION_INDEX["+Y"]] = suffix_y == 0

    prefix_y = np.zeros_like(g)
    prefix_y[:, 1:] = np.cumsum(g, axis=1)[:, :-1]
    acc[DIRECTION_INDEX["-Y"]] = prefix_y == 0

    suffix_z = np.zeros_like(g)
    suffix_z[:, :, :-1] = np.cumsum(g[:, :, ::-1], axis=2)[:, :, -2::-1]
    acc[DIRECTION_INDEX["+Z"]] = suffix_z == 0

    prefix_z = np.zeros_like(g)
    prefix_z[:, :, 1:] = np.cumsum(g, axis=2)[:, :, :-1]
    acc[DIRECTION_INDEX["-Z"]] = prefix_z == 0

    # Prismatic machining setups can reach exposed vertical faces from top or
    # bottom when the adjacent empty column is clear in that approach direction.
    has_solid_below = np.zeros_like(grid, dtype=bool)
    has_solid_below[:, :, 1:] = grid[:, :, :-1]
    has_solid_above = np.zeros_like(grid, dtype=bool)
    has_solid_above[:, :, :-1] = grid[:, :, 1:]

    plus_z_side_clear = np.zeros_like(grid, dtype=bool)
    plus_z_side_clear[:-1] |= empty[1:] & (suffix_z[1:] == 0)
    plus_z_side_clear[1:] |= empty[:-1] & (suffix_z[:-1] == 0)
    plus_z_side_clear[:, :-1] |= empty[:, 1:] & (suffix_z[:, 1:] == 0)
    plus_z_side_clear[:, 1:] |= empty[:, :-1] & (suffix_z[:, :-1] == 0)
    plus_z_side_clear[0, :, :] = True
    plus_z_side_clear[-1, :, :] = True
    plus_z_side_clear[:, 0, :] = True
    plus_z_side_clear[:, -1, :] = True
    plus_z_side_clear &= has_solid_below
    acc[DIRECTION_INDEX["+Z"]] |= plus_z_side_clear

    minus_z_side_clear = np.zeros_like(grid, dtype=bool)
    minus_z_side_clear[:-1] |= empty[1:] & (prefix_z[1:] == 0)
    minus_z_side_clear[1:] |= empty[:-1] & (prefix_z[:-1] == 0)
    minus_z_side_clear[:, :-1] |= empty[:, 1:] & (prefix_z[:, 1:] == 0)
    minus_z_side_clear[:, 1:] |= empty[:, :-1] & (prefix_z[:, :-1] == 0)
    minus_z_side_clear[0, :, :] = True
    minus_z_side_clear[-1, :, :] = True
    minus_z_side_clear[:, 0, :] = True
    minus_z_side_clear[:, -1, :] = True
    minus_z_side_clear &= has_solid_above
    acc[DIRECTION_INDEX["-Z"]] |= minus_z_side_clear

    return acc


def _rotation_from_previous(previous: str, current: str) -> str:
    return ROTATION_DESCRIPTIONS.get((previous, current), "refixtured")


def greedy_setup_assignment(
    acc_map: np.ndarray,
    surface_mask: np.ndarray,
    coverage_threshold: float = 0.99,
) -> list[dict]:
    """Greedily select setup directions to cover the surface."""
    total_surface = int(surface_mask.sum())
    if total_surface == 0:
        return []

    covered = np.zeros_like(surface_mask, dtype=bool)
    selected: list[int] = []
    order = [DIRECTION_INDEX["+Z"]] + [
        idx for idx in range(NUM_DIRECTIONS) if idx != DIRECTION_INDEX["+Z"]
    ]

    for direction_idx in order:
        if direction_idx in selected:
            continue
        reachable = acc_map[direction_idx] & surface_mask
        marginal = reachable & ~covered
        marginal_count = int(marginal.sum())
        if marginal_count <= 0 and selected:
            continue

        selected.append(direction_idx)
        covered |= reachable
        if covered.sum() / total_surface >= coverage_threshold:
            break

        while True:
            best_idx = None
            best_count = 0
            for candidate_idx in range(NUM_DIRECTIONS):
                if candidate_idx in selected:
                    continue
                candidate_marginal = acc_map[candidate_idx] & surface_mask & ~covered
                candidate_count = int(candidate_marginal.sum())
                if candidate_count > best_count:
                    best_idx = candidate_idx
                    best_count = candidate_count

            if best_idx is None or best_count == 0:
                break
            selected.append(best_idx)
            covered |= acc_map[best_idx] & surface_mask
            if covered.sum() / total_surface >= coverage_threshold:
                break
        break

    setups = []
    covered_for_marginal = np.zeros_like(surface_mask, dtype=bool)
    previous_direction = None
    for setup_id, direction_idx in enumerate(selected):
        direction = DIRECTION_LABELS[direction_idx]
        reachable = acc_map[direction_idx] & surface_mask
        marginal = reachable & ~covered_for_marginal
        marginal_count = int(marginal.sum())
        rotation = (
            "initial"
            if previous_direction is None
            else _rotation_from_previous(previous_direction, direction)
        )
        setups.append(
            {
                "id": setup_id,
                "approach_direction": direction,
                "rotation_from_previous": rotation,
                "surface_voxel_count": marginal_count,
                "surface_coverage_fraction": float(marginal_count / total_surface),
            }
        )
        covered_for_marginal |= reachable
        previous_direction = direction

    return setups


def infer_axis_requirement(setup_directions: list[str]) -> int:
    """Infer minimum CNC axis count from required approach directions."""
    if len(setup_directions) <= 1:
        return 3
    axes_used = {direction[1] for direction in setup_directions}
    if len(axes_used) == 1:
        return 3
    if len(axes_used) == 2:
        return 4
    return 5


def map_features_to_setups(
    setups: list[dict],
    features: list[dict],
    acc_map: np.ndarray,
    surface_mask: np.ndarray,
) -> dict[str, list[str]]:
    """Assign detected feature labels to setup ids with a simple heuristic."""
    del acc_map, surface_mask
    mapping = {str(setup["id"]): [] for setup in setups}
    if not setups:
        return mapping

    primary_id = str(setups[0]["id"])
    side_id = primary_id
    for setup in setups:
        if "Z" not in setup["approach_direction"]:
            side_id = str(setup["id"])
            break

    for feature in features:
        feature_type = feature.get("type")
        if not feature_type:
            continue
        target = side_id if feature_type == "rectangular_step" else primary_id
        mapping.setdefault(target, []).append(feature_type)

    return mapping


def map_feature_instances_to_setups(
    setups: list[dict],
    instances: list[dict],
) -> dict[str, list[dict]]:
    """Assign localised feature instances to setup ids by access direction."""
    mapping = {str(setup["id"]): [] for setup in setups}
    if not setups:
        return mapping

    primary_id = str(setups[0]["id"])
    setup_by_direction = {
        setup.get("approach_direction", ""): str(setup["id"]) for setup in setups
    }

    for instance in instances:
        access_directions = instance.get("access_directions", [])
        if not isinstance(access_directions, list):
            access_directions = []
        target_id = None
        for direction in [instance.get("primary_direction")] + access_directions:
            if direction in setup_by_direction:
                target_id = setup_by_direction[direction]
                break
        if target_id is None:
            target_id = primary_id
        mapping.setdefault(target_id, []).append(
            {
                "type": instance.get("type"),
                "instance_id": int(instance.get("instance_id", 0)),
                "confidence": float(instance.get("confidence", 0.0)),
                "primary_direction": instance.get("primary_direction"),
                "volume_voxels": int(instance.get("volume_voxels", 0)),
                "localisation_status": instance.get("localisation_status", "unknown"),
            }
        )

    return mapping


def _single_top_setup(acc_map: np.ndarray, surface_mask: np.ndarray) -> list[dict]:
    """Return the baseline one-sided 2.5D setup."""
    total_surface = int(surface_mask.sum())
    plus_z_idx = DIRECTION_INDEX[DEFAULT_APPROACH_DIRECTION]
    reachable = acc_map[plus_z_idx] & surface_mask
    reachable_count = int(reachable.sum())
    fraction = float(reachable_count / total_surface) if total_surface else 0.0
    return [
        {
            "id": 0,
            "approach_direction": DEFAULT_APPROACH_DIRECTION,
            "rotation_from_previous": "initial",
            "surface_voxel_count": reachable_count,
            "surface_coverage_fraction": fraction,
        }
    ]


def _instance_to_setup_payload(instance: dict, supported: bool, reason: str | None) -> dict:
    payload = {
        "type": instance.get("type"),
        "instance_id": int(instance.get("instance_id", 0)),
        "confidence": float(instance.get("confidence", 0.0)),
        "primary_direction": instance.get("primary_direction"),
        "access_directions": instance.get("access_directions", []),
        "volume_voxels": int(instance.get("volume_voxels", 0)),
        "localisation_status": instance.get("localisation_status", "unknown"),
        "top_accessible": bool(instance.get("top_accessible", DEFAULT_APPROACH_DIRECTION in instance.get("access_directions", []))),
        "bottom_accessible": bool(instance.get("bottom_accessible", False)),
        "side_accessible": bool(instance.get("side_accessible", False)),
        "access_class": instance.get("access_class", "unknown"),
        "opening_span_voxels": int(instance.get("opening_span_voxels", 0)),
        "depth_voxels": int(instance.get("depth_voxels", 0)),
        "aspect_ratio": float(instance.get("aspect_ratio", 0.0)),
        "two_point_five_d_supported": bool(supported),
    }
    if reason:
        payload["unsupported_reason"] = reason
    return payload


def _review_item(code: str, message: str, instance: dict | None = None) -> dict:
    item = {"code": code, "severity": "review", "message": message, "source": "phase3_setup_analysis"}
    if instance is not None:
        item["feature_type"] = instance.get("type")
        item["instance_id"] = int(instance.get("instance_id", 0))
    return item


def _assess_instance_25d(instance: dict) -> tuple[bool, dict | None]:
    feature_type = instance.get("type", "unknown")
    access_directions = instance.get("access_directions", [])
    if not isinstance(access_directions, list):
        access_directions = []
    primary_direction = instance.get("primary_direction")
    directions = [direction for direction in [primary_direction] + access_directions if direction]

    if feature_type in LOCALISATION_REQUIRED_FEATURES and instance.get("localisation_status") != "localised":
        return False, _review_item(
            "UNCERTAIN_LOCALISATION",
            f"{feature_type} instance {int(instance.get('instance_id', 0))} is not localised.",
            instance,
        )

    if DEFAULT_APPROACH_DIRECTION in directions:
        return True, None

    if feature_type == "through_hole" and "-Z" in directions:
        return False, _review_item(
            "BOTTOM_ACCESS_ONLY",
            (
                f"through_hole instance {int(instance.get('instance_id', 0))} appears bottom-only; "
                "manual review required for one-sided machining."
            ),
            instance,
        )

    if any(direction in directions for direction in ("+X", "-X", "+Y", "-Y")):
        return False, _review_item(
            "SIDE_ACCESS_REQUIRED",
            (
                f"{feature_type} instance {int(instance.get('instance_id', 0))} requires side access "
                "outside the +Z 2.5D baseline."
            ),
            instance,
        )

    if "-Z" in directions:
        return False, _review_item(
            "BOTTOM_ACCESS_ONLY",
            (
                f"{feature_type} instance {int(instance.get('instance_id', 0))} requires bottom access "
                "outside the +Z 2.5D baseline."
            ),
            instance,
        )

    return False, _review_item(
        "UNCERTAIN_TOP_ACCESS",
        f"{feature_type} instance {int(instance.get('instance_id', 0))} has uncertain +Z accessibility.",
        instance,
    )


def map_feature_instances_to_single_setup_25d(
    instances: list[dict],
) -> tuple[dict[str, list[dict]], list[dict]]:
    """Assign every instance to setup 0 and tag unsupported 2.5D cases."""
    mapping: dict[str, list[dict]] = {"0": []}
    review_items: list[dict] = []
    for instance in instances:
        supported, item = _assess_instance_25d(instance)
        if item:
            review_items.append(item)
        mapping["0"].append(_instance_to_setup_payload(instance, supported, item["message"] if item else None))
    return mapping, review_items


def _load_metadata(metadata_path: str | None) -> dict:
    if metadata_path is None:
        return {}
    metadata_abs = os.path.abspath(metadata_path)
    if not os.path.exists(metadata_abs):
        raise FileNotFoundError(metadata_path)
    with open(metadata_abs, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _load_pmi_by_instance(pmi_data_path: str | None) -> dict[tuple[str, int], dict]:
    if pmi_data_path is None:
        return {}
    pmi_abs = os.path.abspath(pmi_data_path)
    if not os.path.exists(pmi_abs):
        raise FileNotFoundError(pmi_data_path)
    with open(pmi_abs, encoding="utf-8") as f:
        data = json.load(f)
    result: dict[tuple[str, int], dict] = {}
    for item in data.get("features", []):
        if not isinstance(item, dict) or not item.get("type"):
            continue
        result[(str(item["type"]), int(item.get("instance_id", 0)))] = item
    return result


def _voxel_pitch_mm(grid: np.ndarray, metadata: dict) -> float | None:
    bbox = metadata.get("bounding_box_mm", {})
    dims = [float(bbox.get(axis, 0.0)) for axis in ("x", "y", "z")]
    longest = max(dims) if dims else 0.0
    if longest <= 0 or grid.shape[0] <= 2:
        return None
    return longest / float(grid.shape[0] - 2)


def _tool_reach_feasibility(
    instances: list[dict],
    pitch_mm: float | None,
    pmi_by_instance: dict[tuple[str, int], dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    feature_feasibility: list[dict] = []
    review_items: list[dict] = []
    for instance in instances:
        feature_type = instance.get("type", "unknown")
        instance_id = int(instance.get("instance_id", 0))
        pmi = (pmi_by_instance or {}).get((feature_type, instance_id), {})
        opening_voxels = int(instance.get("opening_span_voxels", 0))
        depth_voxels = int(instance.get("depth_voxels", 0))
        opening_mm = opening_voxels * pitch_mm if pitch_mm else None
        depth_mm = depth_voxels * pitch_mm if pitch_mm else None
        if pmi:
            depth_mm = float(pmi["depth_mm"]) if pmi.get("depth_mm") is not None else depth_mm
            opening_mm = float(
                pmi.get("diameter_mm")
                or pmi.get("width_mm")
                or pmi.get("length_mm")
                or opening_mm
                or 0.0
            )
        ratio = float(depth_voxels / max(1, opening_voxels)) if opening_voxels else math.inf
        if depth_mm is not None and opening_mm and opening_mm > 0:
            ratio = float(depth_mm / opening_mm)
        rule = TOOL_REACH_RULES.get(feature_type, {"max_depth_to_opening_ratio": 6.0, "max_depth_mm": 60.0})
        top_accessible = bool(instance.get("top_accessible", DEFAULT_APPROACH_DIRECTION in instance.get("access_directions", [])))
        tool_reach_ok = top_accessible

        reasons = []
        if not top_accessible:
            tool_reach_ok = False
            reasons.append("Feature is not accessible from +Z.")
        if opening_voxels <= 0 and feature_type in LOCALISATION_REQUIRED_FEATURES:
            tool_reach_ok = False
            reasons.append("Feature opening could not be estimated.")
        if pmi and opening_mm and opening_mm > 0:
            reasons = [reason for reason in reasons if reason != "Feature opening could not be estimated."]
        if (opening_voxels > 0 or (pmi and opening_mm and opening_mm > 0)) and ratio > float(rule["max_depth_to_opening_ratio"]):
            tool_reach_ok = False
            reasons.append(
                f"Depth/opening ratio {ratio:.2f} exceeds {float(rule['max_depth_to_opening_ratio']):.2f}."
            )
        if depth_mm is not None and depth_mm > float(rule["max_depth_mm"]):
            tool_reach_ok = False
            reasons.append(f"Estimated depth {depth_mm:.1f} mm exceeds {float(rule['max_depth_mm']):.1f} mm.")

        feasibility = {
            "instance_id": int(instance.get("instance_id", 0)),
            "type": feature_type,
            "top_accessible": top_accessible,
            "estimated_depth_mm": depth_mm,
            "min_opening_mm": opening_mm,
            "dimension_source": "pmi_brep" if pmi else "voxel",
            "depth_voxels": depth_voxels,
            "opening_span_voxels": opening_voxels,
            "aspect_ratio": ratio if math.isfinite(ratio) else None,
            "max_depth_to_opening_ratio": float(rule["max_depth_to_opening_ratio"]),
            "max_depth_mm": float(rule["max_depth_mm"]),
            "tool_reach_ok": tool_reach_ok,
            "reasons": reasons,
        }
        feature_feasibility.append(feasibility)
        for reason in reasons:
            review_items.append(
                _review_item("TOOL_REACH_LIMIT", f"{feature_type} instance {feasibility['instance_id']}: {reason}", instance)
            )
    return feature_feasibility, review_items


def _write_json_atomic(data: dict, path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=directory, delete=False, suffix=".tmp") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
        tmp = f.name
    os.replace(tmp, path)


def _write_npy_atomic(path: str, array: np.ndarray) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp.npy"
    np.save(tmp, array)
    os.replace(tmp, path)


def _load_features(features_path: str | None) -> list[dict]:
    if features_path is None:
        return []
    features_abs = os.path.abspath(features_path)
    if not os.path.exists(features_abs):
        raise FileNotFoundError(features_path)
    with open(features_abs, encoding="utf-8") as f:
        data = json.load(f)
    features = data.get("features", [])
    return features if isinstance(features, list) else []


def _load_feature_instances(feature_instances_path: str | None) -> list[dict]:
    if feature_instances_path is None:
        return []
    instances_abs = os.path.abspath(feature_instances_path)
    if not os.path.exists(instances_abs):
        raise FileNotFoundError(feature_instances_path)
    with open(instances_abs, encoding="utf-8") as f:
        data = json.load(f)
    instances = data.get("instances", [])
    return instances if isinstance(instances, list) else []


def analyse_setups(
    voxel_path: str,
    output_dir: str,
    features_path: str | None = None,
    feature_instances_path: str | None = None,
    metadata_path: str | None = None,
    pmi_data_path: str | None = None,
    coverage_threshold: float = 0.99,
) -> dict:
    """Run the full setup analysis pipeline."""
    voxel_abs = os.path.abspath(voxel_path)
    if not os.path.exists(voxel_abs):
        raise FileNotFoundError(voxel_path)

    output_abs = os.path.abspath(output_dir)
    os.makedirs(output_abs, exist_ok=True)

    grid = _validate_grid(np.load(voxel_abs))
    metadata = _load_metadata(metadata_path)
    pmi_by_instance = _load_pmi_by_instance(pmi_data_path)
    pitch_mm = _voxel_pitch_mm(grid, metadata)
    surface_mask = compute_surface_mask(grid)
    total_surface = int(surface_mask.sum())
    if total_surface == 0:
        raise RuntimeError("Surface mask is empty - degenerate geometry.")

    acc_map = compute_accessibility_map(grid)
    direction_coverage = {
        label: float(((acc_map[idx] & surface_mask).sum()) / total_surface)
        for idx, label in enumerate(DIRECTION_LABELS)
    }

    legacy_setups = greedy_setup_assignment(acc_map, surface_mask, coverage_threshold)
    setups = _single_top_setup(acc_map, surface_mask)
    selected_indices = [DIRECTION_INDEX[setup["approach_direction"]] for setup in setups]
    if selected_indices:
        selected_union = np.any(acc_map[selected_indices], axis=0) & surface_mask
    else:
        selected_union = np.zeros_like(surface_mask, dtype=bool)

    covered_surface = int(selected_union.sum())
    inaccessible_surface = int((surface_mask & ~selected_union).sum())
    inaccessible_fraction = float(inaccessible_surface / total_surface)
    warnings: list[str] = []

    features = _load_features(features_path)
    feature_instances = _load_feature_instances(feature_instances_path)
    features_per_setup = map_features_to_setups(setups, features, acc_map, surface_mask)
    feature_instances_per_setup, access_review_items = map_feature_instances_to_single_setup_25d(
        feature_instances
    )
    feature_feasibility, reach_review_items = _tool_reach_feasibility(
        feature_instances, pitch_mm, pmi_by_instance
    )
    if feature_instances:
        features_per_setup = {
            setup_id: [instance["type"] for instance in instances if instance.get("type")]
            for setup_id, instances in feature_instances_per_setup.items()
        }
    review_items = access_review_items + reach_review_items
    unsupported_reasons = [item["message"] for item in review_items]
    review_codes = [item["code"] for item in review_items]
    warnings.extend(unsupported_reasons)
    tool_reach_compatible = not reach_review_items
    two_point_five_d_compatible = not review_items

    accessibility_path = os.path.join(output_abs, "accessibility_map.npy")
    surface_path = os.path.join(output_abs, "surface_mask.npy")
    analysis_path = os.path.join(output_abs, "setup_analysis.json")
    _write_npy_atomic(accessibility_path, acc_map)
    _write_npy_atomic(surface_path, surface_mask)

    result = {
        "setup_count": len(setups),
        "axis_requirement": 3,
        "setup_mode": SETUP_MODE,
        "two_point_five_d_compatible": two_point_five_d_compatible,
        "tool_reach_compatible": tool_reach_compatible,
        "feature_feasibility": feature_feasibility,
        "review_items": review_items,
        "review_codes": sorted(set(review_codes)),
        "tool_reach_warnings": [item["message"] for item in reach_review_items],
        "unsupported_reasons": unsupported_reasons,
        "requires_rotation": False,
        "setups": setups,
        "legacy_multi_direction_setups": legacy_setups,
        "legacy_axis_requirement": infer_axis_requirement(
            [setup["approach_direction"] for setup in legacy_setups]
        ),
        "direction_coverage": direction_coverage,
        "total_surface_voxels": total_surface,
        "covered_surface_voxels": covered_surface,
        "inaccessible_surface_voxels": inaccessible_surface,
        "inaccessible_fraction": inaccessible_fraction,
        "features_per_setup": features_per_setup,
        "feature_instances_per_setup": feature_instances_per_setup,
        "voxel_file": voxel_abs,
        "metadata_file": os.path.abspath(metadata_path) if metadata_path else None,
        "pmi_data_file": os.path.abspath(pmi_data_path) if pmi_data_path else None,
        "accessibility_map_file": accessibility_path,
        "surface_mask_file": surface_path,
        "warnings": warnings,
    }
    _write_json_atomic(result, analysis_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse setup and stock rotation requirements from a voxel grid."
    )
    parser.add_argument("voxel_path", help="Path to voxel_{R}.npy")
    parser.add_argument("output_dir", help="Directory for output files")
    parser.add_argument("--features", default=None, help="Path to features.json")
    parser.add_argument("--feature-instances", default=None, help="Path to feature_instances.json")
    parser.add_argument("--metadata", default=None, help="Path to metadata.json from Phase 1")
    parser.add_argument("--pmi-data", default=None, help="Path to pmi_data.json from Phase 2")
    parser.add_argument("--threshold", type=float, default=0.99)
    args = parser.parse_args()

    result = analyse_setups(
        args.voxel_path,
        args.output_dir,
        features_path=args.features,
        feature_instances_path=args.feature_instances,
        metadata_path=args.metadata,
        pmi_data_path=args.pmi_data,
        coverage_threshold=args.threshold,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
