"""Inference entry point for Phase 2 machining feature recognition."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import deque

import numpy as np
import torch

from models.feature_net import FEATURE_NAMES, load_model

HOLE_TYPES = {"through_hole", "blind_hole"}
RECESS_TYPES = {
    "rectangular_pocket",
    "circular_pocket",
    "rectangular_slot",
    "circular_slot",
    "rectangular_step",
    "triangular_pocket",
}
EDGE_DETAIL_TYPES = {"chamfer", "fillet"}
DEFAULT_UNCERTAINTY_MARGIN = 0.05
GEOMETRY_VALIDATED_TYPES = HOLE_TYPES | RECESS_TYPES
SLOT_TYPES = {"rectangular_slot", "circular_slot"}
POCKET_TYPES = {"rectangular_pocket", "circular_pocket", "triangular_pocket"}


def _normalise_threshold(value: float | None) -> float:
    return float(value if value is not None else 0.5)


def _prediction_status(
    feature_type: str,
    confidence: float,
    threshold: float,
    uncertainty_margin: float,
) -> str:
    if confidence >= threshold:
        if feature_type in EDGE_DETAIL_TYPES:
            return "training_excluded"
        return "detected"
    if confidence >= max(0.0, threshold - uncertainty_margin):
        return "uncertain"
    return "rejected"


def _deduplicate_features(features: list[dict]) -> list[dict]:
    """Keep the highest-confidence prediction for each feature class."""
    by_type: dict[str, dict] = {}
    for feature in features:
        feature_type = feature.get("type")
        if not feature_type:
            continue
        current = by_type.get(feature_type)
        confidence = float(feature.get("confidence", 0.0))
        if current is None or confidence > float(current.get("confidence", 0.0)):
            by_type[feature_type] = feature
    return sorted(by_type.values(), key=lambda item: float(item.get("confidence", 0.0)), reverse=True)


def _occupied_bbox(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(grid)
    if coords.size == 0:
        raise RuntimeError("Cannot inspect geometry in an empty voxel grid.")
    return coords.min(axis=0), coords.max(axis=0)


def _connected_components(mask: np.ndarray, min_voxels: int = 4) -> list[dict]:
    visited = np.zeros_like(mask, dtype=bool)
    components: list[dict] = []
    neighbours = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))

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
            spans = hi - lo + 1
            components.append(
                {
                    "bbox_min": lo,
                    "bbox_max": hi,
                    "spans": spans,
                    "volume_voxels": int(len(coords)),
                }
            )

    return sorted(components, key=lambda item: item["volume_voxels"], reverse=True)


def _voxel_geometry_evidence(grid: np.ndarray, min_component_voxels: int = 4) -> dict:
    grid = grid.astype(bool)
    stock_lo, stock_hi = _occupied_bbox(grid)
    stock_mask = np.zeros_like(grid, dtype=bool)
    stock_mask[stock_lo[0] : stock_hi[0] + 1, stock_lo[1] : stock_hi[1] + 1, stock_lo[2] : stock_hi[2] + 1] = True
    removal_mask = stock_mask & ~grid
    components = _connected_components(removal_mask, min_component_voxels)

    evidence = {
        "component_count": len(components),
        "through_z_components": 0,
        "top_blind_components": 0,
        "top_recess_components": 0,
        "top_slot_components": 0,
        "side_open_components": 0,
    }
    for component in components:
        lo = component["bbox_min"]
        hi = component["bbox_max"]
        spans = [int(v) for v in component["spans"]]
        xy_short = max(1, min(spans[0], spans[1]))
        xy_long = max(spans[0], spans[1])
        top_open = hi[2] >= stock_hi[2]
        bottom_open = lo[2] <= stock_lo[2]
        side_open = lo[0] <= stock_lo[0] or hi[0] >= stock_hi[0] or lo[1] <= stock_lo[1] or hi[1] >= stock_hi[1]
        through_z = top_open and bottom_open
        top_blind = top_open and not bottom_open
        elongated_xy = xy_long / xy_short >= 2.0
        columnar = spans[2] / xy_short >= 1.5

        if through_z:
            evidence["through_z_components"] += 1
        if top_blind:
            evidence["top_blind_components"] += 1
        if top_blind and not columnar:
            evidence["top_recess_components"] += 1
        if top_blind and elongated_xy:
            evidence["top_slot_components"] += 1
        if side_open:
            evidence["side_open_components"] += 1

    return evidence


def _geometry_supports_feature(feature_type: str, evidence: dict) -> tuple[bool, str]:
    if feature_type == "flat_face":
        return True, "baseline_flat_face"
    if feature_type == "through_hole":
        return evidence["through_z_components"] > 0, "through_z_void"
    if feature_type == "blind_hole":
        return evidence["top_blind_components"] > 0, "top_blind_void"
    if feature_type in SLOT_TYPES:
        return evidence["top_slot_components"] > 0, "top_elongated_void"
    if feature_type in POCKET_TYPES:
        return evidence["top_recess_components"] > 0 or evidence["top_blind_components"] > 0, "top_recess_void"
    if feature_type == "rectangular_step":
        return evidence["side_open_components"] > 0 or evidence["top_recess_components"] > 0, "side_or_top_recess_void"
    return True, "not_geometry_validated"


def clean_feature_predictions(
    all_scores: dict[str, float],
    thresholds: dict[str, float],
    uncertainty_margin: float = DEFAULT_UNCERTAINTY_MARGIN,
) -> dict:
    """Convert raw class probabilities into detected, uncertain, and rejected predictions."""
    candidates: list[dict] = []
    detected: list[dict] = []
    uncertain: list[dict] = []
    warnings: list[str] = []

    for name in FEATURE_NAMES:
        confidence = float(all_scores.get(name, 0.0))
        threshold = float(thresholds.get(name, 0.5))
        margin = confidence - threshold
        status = _prediction_status(name, confidence, threshold, uncertainty_margin)
        candidate = {
            "type": name,
            "confidence": confidence,
            "threshold": threshold,
            "margin": margin,
            "status": status,
            "model_supported": name not in EDGE_DETAIL_TYPES,
        }
        candidates.append(candidate)

        if status in {"detected", "training_excluded"}:
            detected.append(candidate)
            if status == "training_excluded":
                warnings.append(
                    f"{name} scored above threshold but this checkpoint was trained with edge-detail labels excluded."
                )
        elif status == "uncertain":
            uncertain.append(candidate)

    detected = _deduplicate_features(detected)
    uncertain = _deduplicate_features(uncertain)
    candidates.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)

    return {
        "features": detected,
        "feature_count": len(detected),
        "candidate_features": candidates,
        "uncertain_features": uncertain,
        "uncertain_count": len(uncertain),
        "prediction_summary": {
            "detected": sum(1 for item in candidates if item["status"] == "detected"),
            "uncertain": sum(1 for item in candidates if item["status"] == "uncertain"),
            "rejected": sum(1 for item in candidates if item["status"] == "rejected"),
            "training_excluded": sum(1 for item in candidates if item["status"] == "training_excluded"),
            "active_feature_count": sum(
                1 for item in detected if item.get("model_supported", True)
            ),
        },
        "warnings": warnings,
    }


def fuse_features_with_voxel_geometry(
    recognition: dict,
    voxel_path: str,
    unsupported_confidence_limit: float = 0.90,
    min_component_voxels: int = 4,
) -> dict:
    """Use simple voxel evidence to validate or suppress low-confidence ML feature labels."""
    voxel_abs = os.path.abspath(voxel_path)
    if not os.path.exists(voxel_abs):
        raise FileNotFoundError(voxel_path)
    grid = np.load(voxel_abs)
    if grid.ndim != 3 or len(set(grid.shape)) != 1:
        raise ValueError("voxel array must be 3-D and cubic.")

    evidence = _voxel_geometry_evidence(grid, min_component_voxels=min_component_voxels)
    warnings = list(recognition.get("warnings", []))
    kept: list[dict] = []
    suppressed: list[dict] = []

    for feature in recognition.get("features", []):
        item = dict(feature)
        feature_type = str(item.get("type", ""))
        confidence = float(item.get("confidence", 0.0))
        supported, support_reason = _geometry_supports_feature(feature_type, evidence)
        item["geometry_support"] = "supported" if supported else "unsupported"
        item["geometry_support_reason"] = support_reason

        if (
            feature_type in GEOMETRY_VALIDATED_TYPES
            and not supported
            and confidence < unsupported_confidence_limit
        ):
            item["status"] = "geometry_suppressed"
            suppressed.append(item)
            warnings.append(
                f"Suppressed {feature_type} ({confidence:.3f}) because voxel geometry found no {support_reason} evidence."
            )
            continue
        if feature_type in GEOMETRY_VALIDATED_TYPES and not supported:
            item["status"] = "unverified_high_confidence"
            warnings.append(
                f"Kept high-confidence {feature_type} ({confidence:.3f}) for review despite weak voxel geometry evidence."
            )
        kept.append(item)

    kept = _deduplicate_features(kept)
    result = dict(recognition)
    result["features"] = kept
    result["feature_count"] = len(kept)
    result["geometry_fusion"] = {
        "applied": True,
        "voxel_file": voxel_abs,
        "unsupported_confidence_limit": float(unsupported_confidence_limit),
        "evidence": evidence,
        "suppressed_features": suppressed,
        "suppressed_count": len(suppressed),
    }
    result["warnings"] = warnings
    return result


def _write_json_atomic(data: dict, path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=directory, delete=False, suffix=".tmp") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
        tmp = f.name
    os.replace(tmp, path)


def recognise_features(
    voxel_path: str,
    model_path: str,
    threshold: float | None = 0.5,
    device: str = "cpu",
    use_checkpoint_thresholds: bool = True,
    uncertainty_margin: float = DEFAULT_UNCERTAINTY_MARGIN,
) -> dict:
    """Predict machining features present in a voxelised part."""
    voxel_abs = os.path.abspath(voxel_path)
    model_abs = os.path.abspath(model_path)
    if not os.path.exists(voxel_abs):
        raise FileNotFoundError(voxel_path)
    if not os.path.exists(model_abs):
        raise FileNotFoundError(model_path)

    voxel = np.load(voxel_abs)
    if voxel.ndim != 3 or len(set(voxel.shape)) != 1:
        raise ValueError("voxel array must be 3-D and cubic.")

    torch_device = torch.device(device)
    model = load_model(model_abs, device=torch_device)
    x = torch.from_numpy(voxel.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(torch_device)

    with torch.no_grad():
        probs = torch.sigmoid(model(x)).detach().cpu().numpy()[0]

    all_scores = {name: float(probs[idx]) for idx, name in enumerate(FEATURE_NAMES)}
    checkpoint_thresholds = {}
    if use_checkpoint_thresholds:
        try:
            checkpoint = torch.load(model_abs, map_location="cpu")
            checkpoint_thresholds = checkpoint.get("class_thresholds", {})
        except Exception:
            checkpoint_thresholds = {}
    default_threshold = _normalise_threshold(threshold)
    thresholds = {name: float(checkpoint_thresholds.get(name, default_threshold)) for name in FEATURE_NAMES}
    cleaned = clean_feature_predictions(
        all_scores,
        thresholds,
        uncertainty_margin=uncertainty_margin,
    )

    return {
        "features": cleaned["features"],
        "feature_count": cleaned["feature_count"],
        "candidate_features": cleaned["candidate_features"],
        "uncertain_features": cleaned["uncertain_features"],
        "uncertain_count": cleaned["uncertain_count"],
        "prediction_summary": cleaned["prediction_summary"],
        "all_scores": all_scores,
        "threshold": default_threshold,
        "thresholds": thresholds,
        "threshold_source": "checkpoint" if checkpoint_thresholds else "argument",
        "uncertainty_margin": float(uncertainty_margin),
        "active_features": [name for name in FEATURE_NAMES if name not in EDGE_DETAIL_TYPES],
        "excluded_features": sorted(EDGE_DETAIL_TYPES),
        "voxel_file": voxel_abs,
        "model_path": model_abs,
        "warnings": cleaned["warnings"],
    }


def reconcile_features_with_brep(
    recognition: dict,
    brep_data: dict,
    unsupported_confidence_limit: float = 0.90,
    edge_detail_confidence_limit: float = 0.70,
) -> dict:
    """Suppress low-confidence CNN labels that contradict simple BRep evidence."""
    features = [dict(feature) for feature in recognition.get("features", [])]
    all_scores = dict(recognition.get("all_scores", {}))
    warnings = list(recognition.get("warnings", []))
    bbox = brep_data.get("bounding_box_mm", {})
    part_z = float(bbox.get("z", 0.0))
    holes = brep_data.get("holes", [])
    recesses = brep_data.get("planar_recesses", [])
    has_holes = bool(holes)
    has_recesses = bool(recesses)
    through_holes = [
        hole for hole in holes if part_z > 0 and float(hole.get("depth_mm", 0.0)) >= part_z * 0.9
    ]
    blind_holes = [hole for hole in holes if hole not in through_holes]

    kept: list[dict] = []
    for feature in features:
        feature_type = feature.get("type")
        confidence = float(feature.get("confidence", 0.0))
        remove_reason = None
        if feature_type == "through_hole" and not has_holes and confidence < unsupported_confidence_limit:
            remove_reason = "no cylindrical hole geometry measured"
        elif feature_type == "blind_hole" and not blind_holes and confidence < unsupported_confidence_limit:
            remove_reason = "no blind-hole geometry measured"
        elif feature_type in RECESS_TYPES and not has_recesses and confidence < unsupported_confidence_limit:
            remove_reason = "no recess geometry measured"
        elif (
            feature_type in EDGE_DETAIL_TYPES
            and not has_holes
            and not has_recesses
            and confidence < edge_detail_confidence_limit
        ):
            remove_reason = "no supporting edge-detail geometry measured"

        if remove_reason:
            warnings.append(
                f"Suppressed {feature_type} ({confidence:.3f}) because BRep check found {remove_reason}."
            )
            continue
        kept.append(feature)

    present = {feature.get("type") for feature in kept}
    if through_holes and "through_hole" not in present:
        kept.append(
            {
                "type": "through_hole",
                "confidence": max(float(all_scores.get("through_hole", 0.0)), 0.95),
                "source": "brep_reconciled",
            }
        )
        warnings.append("Added through_hole because BRep measurement found through-hole geometry.")
    if blind_holes and "blind_hole" not in present:
        kept.append(
            {
                "type": "blind_hole",
                "confidence": max(float(all_scores.get("blind_hole", 0.0)), 0.95),
                "source": "brep_reconciled",
            }
        )
        warnings.append("Added blind_hole because BRep measurement found blind-hole geometry.")

    if not any(feature.get("type") == "flat_face" for feature in kept):
        kept.insert(0, {"type": "flat_face", "confidence": 1.0, "source": "baseline"})

    kept.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
    result = dict(recognition)
    result["features"] = kept
    result["feature_count"] = len(kept)
    result["reconciliation"] = {
        "applied": True,
        "holes_measured": len(holes),
        "through_holes_measured": len(through_holes),
        "blind_holes_measured": len(blind_holes),
        "recesses_measured": len(recesses),
    }
    result["warnings"] = warnings
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Recognise machining features from a voxel grid.")
    parser.add_argument("voxel_path")
    parser.add_argument("model_path")
    parser.add_argument("output_dir")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    result = recognise_features(
        args.voxel_path,
        args.model_path,
        threshold=args.threshold,
        device=args.device,
    )
    output_path = os.path.join(os.path.abspath(args.output_dir), "features.json")
    _write_json_atomic(result, output_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
