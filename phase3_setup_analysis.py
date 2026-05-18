"""Phase 3 setup and stock rotation analysis from voxel grids."""

from __future__ import annotations

import argparse
import json
import os
import tempfile

import numpy as np


DIRECTION_LABELS = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
DIRECTION_INDEX = {label: idx for idx, label in enumerate(DIRECTION_LABELS)}
NUM_DIRECTIONS = 6

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


def analyse_setups(
    voxel_path: str,
    output_dir: str,
    features_path: str | None = None,
    coverage_threshold: float = 0.99,
) -> dict:
    """Run the full setup analysis pipeline."""
    voxel_abs = os.path.abspath(voxel_path)
    if not os.path.exists(voxel_abs):
        raise FileNotFoundError(voxel_path)

    output_abs = os.path.abspath(output_dir)
    os.makedirs(output_abs, exist_ok=True)

    grid = _validate_grid(np.load(voxel_abs))
    surface_mask = compute_surface_mask(grid)
    total_surface = int(surface_mask.sum())
    if total_surface == 0:
        raise RuntimeError("Surface mask is empty - degenerate geometry.")

    acc_map = compute_accessibility_map(grid)
    direction_coverage = {
        label: float(((acc_map[idx] & surface_mask).sum()) / total_surface)
        for idx, label in enumerate(DIRECTION_LABELS)
    }

    setups = greedy_setup_assignment(acc_map, surface_mask, coverage_threshold)
    selected_indices = [DIRECTION_INDEX[setup["approach_direction"]] for setup in setups]
    if selected_indices:
        selected_union = np.any(acc_map[selected_indices], axis=0) & surface_mask
    else:
        selected_union = np.zeros_like(surface_mask, dtype=bool)

    covered_surface = int(selected_union.sum())
    inaccessible_surface = int((surface_mask & ~selected_union).sum())
    inaccessible_fraction = float(inaccessible_surface / total_surface)
    warnings: list[str] = []
    if inaccessible_surface > 0:
        warnings.append(
            f"{inaccessible_surface} surface voxels are inaccessible from selected setups."
        )

    setup_directions = [setup["approach_direction"] for setup in setups]
    features = _load_features(features_path)
    features_per_setup = map_features_to_setups(setups, features, acc_map, surface_mask)

    accessibility_path = os.path.join(output_abs, "accessibility_map.npy")
    surface_path = os.path.join(output_abs, "surface_mask.npy")
    analysis_path = os.path.join(output_abs, "setup_analysis.json")
    _write_npy_atomic(accessibility_path, acc_map)
    _write_npy_atomic(surface_path, surface_mask)

    result = {
        "setup_count": len(setups),
        "axis_requirement": infer_axis_requirement(setup_directions),
        "setups": setups,
        "direction_coverage": direction_coverage,
        "total_surface_voxels": total_surface,
        "covered_surface_voxels": covered_surface,
        "inaccessible_surface_voxels": inaccessible_surface,
        "inaccessible_fraction": inaccessible_fraction,
        "features_per_setup": features_per_setup,
        "voxel_file": voxel_abs,
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
    parser.add_argument("--threshold", type=float, default=0.99)
    args = parser.parse_args()

    result = analyse_setups(
        args.voxel_path,
        args.output_dir,
        features_path=args.features,
        coverage_threshold=args.threshold,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
