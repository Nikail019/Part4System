"""Voxel feature instance localisation for the RPP baseline pipeline."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import deque

import numpy as np


DIRECTION_LABELS = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
LOCALISABLE_TYPES = {
    "through_hole",
    "blind_hole",
    "rectangular_pocket",
    "circular_pocket",
    "rectangular_slot",
    "circular_slot",
    "rectangular_step",
    "triangular_pocket",
}
FALLBACK_TYPES = {"flat_face", "chamfer", "fillet", "boss"}
SIDE_DIRECTIONS = {"+X", "-X", "+Y", "-Y"}


def _write_json_atomic(data: dict, path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=directory, delete=False, suffix=".tmp") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
        tmp = f.name
    os.replace(tmp, path)


def _validate_grid(grid: np.ndarray) -> np.ndarray:
    if grid.ndim != 3 or len(set(grid.shape)) != 1:
        raise ValueError("voxel array must be 3-D and cubic.")
    return grid.astype(bool)


def _load_features(features_path: str) -> list[dict]:
    if not os.path.exists(features_path):
        raise FileNotFoundError(features_path)
    with open(features_path, encoding="utf-8") as f:
        data = json.load(f)
    features = data.get("features", [])
    if not isinstance(features, list):
        raise ValueError("features JSON must contain a 'features' list.")
    return features


def _load_pmi_features(pmi_data_path: str | None) -> dict[str, list[dict]]:
    if pmi_data_path is None:
        return {}
    pmi_abs = os.path.abspath(pmi_data_path)
    if not os.path.exists(pmi_abs):
        raise FileNotFoundError(pmi_data_path)
    with open(pmi_abs, encoding="utf-8") as f:
        data = json.load(f)
    result: dict[str, list[dict]] = {}
    for item in data.get("features", []):
        if isinstance(item, dict) and item.get("type"):
            result.setdefault(str(item["type"]), []).append(item)
    return result


def _occupied_bbox(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(grid)
    if coords.size == 0:
        raise RuntimeError("Cannot localise features in an empty voxel grid.")
    return coords.min(axis=0), coords.max(axis=0)


def _stock_box_mask(grid: np.ndarray) -> np.ndarray:
    lo, hi = _occupied_bbox(grid)
    mask = np.zeros_like(grid, dtype=bool)
    mask[lo[0] : hi[0] + 1, lo[1] : hi[1] + 1, lo[2] : hi[2] + 1] = True
    return mask


def _connected_components(mask: np.ndarray, min_voxels: int = 4) -> list[dict]:
    visited = np.zeros_like(mask, dtype=bool)
    components: list[dict] = []
    neighbours = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    for seed in np.argwhere(mask):
        seed_tuple = tuple(int(v) for v in seed)
        if visited[seed_tuple]:
            continue

        queue = deque([seed_tuple])
        visited[seed_tuple] = True
        coords = []
        while queue:
            current = queue.popleft()
            coords.append(current)
            for dx, dy, dz in neighbours:
                nxt = (current[0] + dx, current[1] + dy, current[2] + dz)
                if (
                    0 <= nxt[0] < mask.shape[0]
                    and 0 <= nxt[1] < mask.shape[1]
                    and 0 <= nxt[2] < mask.shape[2]
                    and mask[nxt]
                    and not visited[nxt]
                ):
                    visited[nxt] = True
                    queue.append(nxt)

        if len(coords) >= min_voxels:
            arr = np.asarray(coords, dtype=int)
            lo = arr.min(axis=0)
            hi = arr.max(axis=0)
            centroid = arr.mean(axis=0)
            components.append(
                {
                    "coords": arr,
                    "bbox_min": lo,
                    "bbox_max": hi,
                    "centroid": centroid,
                    "volume_voxels": int(len(coords)),
                    "spans": hi - lo + 1,
                }
            )

    return sorted(components, key=lambda item: item["volume_voxels"], reverse=True)


def _component_access_directions(component: dict, stock_lo: np.ndarray, stock_hi: np.ndarray) -> list[str]:
    lo = component["bbox_min"]
    hi = component["bbox_max"]
    directions = []
    if hi[0] >= stock_hi[0]:
        directions.append("+X")
    if lo[0] <= stock_lo[0]:
        directions.append("-X")
    if hi[1] >= stock_hi[1]:
        directions.append("+Y")
    if lo[1] <= stock_lo[1]:
        directions.append("-Y")
    if hi[2] >= stock_hi[2]:
        directions.append("+Z")
    if lo[2] <= stock_lo[2]:
        directions.append("-Z")
    return directions or ["+Z"]


def _primary_direction(access_directions: list[str], feature_type: str) -> str:
    if feature_type in {"through_hole", "blind_hole", "rectangular_pocket", "circular_pocket", "rectangular_slot", "circular_slot", "triangular_pocket"}:
        for preferred in ["+Z", "-Z"]:
            if preferred in access_directions:
                return preferred
    return access_directions[0] if access_directions else "+Z"


def _shape_hint(component: dict) -> str:
    spans = sorted(int(v) for v in component["spans"])
    if spans[2] >= spans[0] * 3 and spans[1] <= spans[0] * 2:
        return "columnar"
    if spans[2] >= spans[1] * 2:
        return "slot_like"
    return "pocket_like"


def _component_metrics(component: dict, access_directions: list[str]) -> dict:
    spans = [int(v) for v in component["spans"]]
    side_open = any(direction in access_directions for direction in SIDE_DIRECTIONS)
    top_open = "+Z" in access_directions
    bottom_open = "-Z" in access_directions
    opening_span = min(spans[0], spans[1])
    depth = spans[2]
    access_class = "top"
    if top_open and bottom_open:
        access_class = "through_z"
    elif side_open and not top_open:
        access_class = "side_only"
    elif bottom_open and not top_open:
        access_class = "bottom_only"
    elif side_open and top_open:
        access_class = "top_and_side"
    elif not top_open:
        access_class = "uncertain"
    return {
        "top_accessible": top_open,
        "bottom_accessible": bottom_open,
        "side_accessible": side_open,
        "access_class": access_class,
        "opening_span_voxels": int(opening_span),
        "depth_voxels": int(depth),
        "aspect_ratio": float(depth / max(1, opening_span)),
    }


def _make_instance(
    feature: dict,
    instance_id: int,
    component: dict | None,
    stock_lo: np.ndarray,
    stock_hi: np.ndarray,
    pmi: dict | None = None,
) -> dict:
    feature_type = feature.get("type", "unknown")
    confidence = float(feature.get("confidence", 0.0))

    if component is None:
        centroid = ((stock_lo + stock_hi) / 2.0).round().astype(int)
        bbox_min = stock_lo
        bbox_max = stock_hi
        access_directions = ["+Z"]
        volume_voxels = 0
        status = "estimated"
        shape_hint = "fallback"
        metrics = {
            "top_accessible": True,
            "bottom_accessible": False,
            "side_accessible": False,
            "access_class": "estimated_top",
            "opening_span_voxels": 0,
            "depth_voxels": 0,
            "aspect_ratio": 0.0,
        }
    else:
        centroid = component["centroid"].round().astype(int)
        bbox_min = component["bbox_min"]
        bbox_max = component["bbox_max"]
        access_directions = _component_access_directions(component, stock_lo, stock_hi)
        volume_voxels = int(component["volume_voxels"])
        status = "localised"
        shape_hint = _shape_hint(component)
        metrics = _component_metrics(component, access_directions)

    instance = {
        "type": feature_type,
        "instance_id": int(instance_id),
        "confidence": confidence,
        "centroid_voxel": [int(v) for v in centroid],
        "bbox_voxel": [
            [int(v) for v in bbox_min],
            [int(v) for v in bbox_max],
        ],
        "volume_voxels": int(volume_voxels),
        "primary_direction": _primary_direction(access_directions, feature_type),
        "access_directions": access_directions,
        "localisation_status": status,
        "shape_hint": shape_hint,
        **metrics,
    }
    if pmi:
        instance["dimension_source"] = pmi.get("dimension_source", "pmi_brep")
        for key in (
            "diameter_mm",
            "depth_mm",
            "depth_ratio",
            "width_mm",
            "length_mm",
            "rough_passes",
            "peck_required",
            "threaded",
            "thread_spec",
        ):
            if key in pmi:
                instance[key] = pmi[key]
    return instance


def localise_feature_instances(
    voxel_path: str,
    features_path: str,
    output_dir: str,
    min_component_voxels: int = 4,
    pmi_data_path: str | None = None,
) -> dict:
    """Create instance-level feature approximations from binary voxels."""
    voxel_abs = os.path.abspath(voxel_path)
    features_abs = os.path.abspath(features_path)
    if not os.path.exists(voxel_abs):
        raise FileNotFoundError(voxel_path)

    grid = _validate_grid(np.load(voxel_abs))
    features = _load_features(features_abs)
    pmi_by_type = _load_pmi_features(pmi_data_path)
    stock_lo, stock_hi = _occupied_bbox(grid)
    stock_mask = _stock_box_mask(grid)
    removal_mask = stock_mask & ~grid
    components = _connected_components(removal_mask, min_component_voxels)

    instances = []
    warnings: list[str] = []
    used_components: set[int] = set()
    counters: dict[str, int] = {}

    for feature in features:
        feature_type = feature.get("type")
        if not feature_type:
            continue

        pmi_entries = pmi_by_type.get(feature_type, [])
        repeat_count = max(1, len(pmi_entries)) if feature_type in LOCALISABLE_TYPES else 1

        for repeat_idx in range(repeat_count):
            instance_id = counters.get(feature_type, 0)
            counters[feature_type] = instance_id + 1
            pmi = pmi_entries[repeat_idx] if repeat_idx < len(pmi_entries) else None

            component = None
            if feature_type in LOCALISABLE_TYPES:
                for idx, candidate in enumerate(components):
                    if idx not in used_components:
                        component = candidate
                        used_components.add(idx)
                        break
            elif feature_type not in FALLBACK_TYPES:
                warnings.append(f"No localisation rule for feature type: {feature_type}")

            instance = _make_instance(feature, instance_id, component, stock_lo, stock_hi, pmi=pmi)
            if instance["localisation_status"] == "estimated":
                warnings.append(
                    f"Estimated fallback localisation for {feature_type} instance {instance_id}."
                )
            instances.append(instance)

    output_abs = os.path.abspath(output_dir)
    os.makedirs(output_abs, exist_ok=True)
    output_path = os.path.join(output_abs, "feature_instances.json")
    result = {
        "instances": instances,
        "instance_count": len(instances),
        "component_count": len(components),
        "source_files": {
            "voxel": voxel_abs,
            "features": features_abs,
            "pmi_data": os.path.abspath(pmi_data_path) if pmi_data_path else None,
        },
        "feature_instances_file": output_path,
        "warnings": warnings,
    }
    _write_json_atomic(result, output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Localise feature instances in a voxel grid.")
    parser.add_argument("voxel_path")
    parser.add_argument("features_path")
    parser.add_argument("output_dir")
    parser.add_argument("--min-component-voxels", type=int, default=4)
    parser.add_argument("--pmi-data", default=None, dest="pmi_data")
    args = parser.parse_args()
    result = localise_feature_instances(
        args.voxel_path,
        args.features_path,
        args.output_dir,
        min_component_voxels=args.min_component_voxels,
        pmi_data_path=args.pmi_data,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
