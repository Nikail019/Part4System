# AGENTS.md — Phase 3: Voxel → Setup & Stock Rotation Analysis

## Context

Phases 1 and 2 are complete and passing (44 + 22 = 66 tests).

- `phase1_voxeliser.py`  → `voxel_{R}.npy` + `metadata.json`
- `phase2_feature_recognition.py` → `features.json`

Phase 3 is **purely geometric and analytical — no ML model, no training data**.
It reads the voxel grid, computes which directions a cutting tool can access each
surface voxel from, groups the surface into setups, and infers the minimum axis
count required. The output feeds Phase 4 (process plan generation) so it knows
how many setups to plan and which operations can share a fixture.

---

## What Phase 3 Must Deliver

```
INPUT:  voxel_64.npy          (R, R, R) bool  — from Phase 1
        features.json         (optional)       — from Phase 2

OUTPUT: setup_analysis.json   — structured setup plan
        accessibility_map.npy — (6, R, R, R) bool
        surface_mask.npy      — (R, R, R) bool
```

### `setup_analysis.json` schema

```json
{
  "setup_count": 2,
  "axis_requirement": 3,

  "setups": [
    {
      "id": 0,
      "approach_direction": "+Z",
      "rotation_from_previous": "initial",
      "surface_voxel_count": 1820,
      "surface_coverage_fraction": 0.74
    },
    {
      "id": 1,
      "approach_direction": "-Z",
      "rotation_from_previous": "flip_around_X_180",
      "surface_voxel_count": 640,
      "surface_coverage_fraction": 0.26
    }
  ],

  "direction_coverage": {
    "+X": 0.18, "-X": 0.18,
    "+Y": 0.22, "-Y": 0.22,
    "+Z": 0.74, "-Z": 0.26
  },

  "total_surface_voxels": 2460,
  "covered_surface_voxels": 2460,
  "inaccessible_surface_voxels": 0,
  "inaccessible_fraction": 0.0,

  "features_per_setup": {
    "0": ["flat_face", "rectangular_pocket", "through_hole"],
    "1": ["flat_face"]
  },

  "voxel_file": "/abs/path/to/voxel_64.npy",
  "accessibility_map_file": "/abs/path/to/accessibility_map.npy",
  "surface_mask_file": "/abs/path/to/surface_mask.npy",
  "warnings": []
}
```

`features_per_setup` is populated only when `features.json` is provided.
When not provided, the key is present but maps each setup id to an empty list.

---

## Coordinate Convention

```
Voxel array axes:  grid[x, y, z]
Direction indices: 0=+X  1=-X  2=+Y  3=-Y  4=+Z  5=-Z

Direction labels:
  "+X"  tool approaches from x = +∞  (right side)
  "-X"  tool approaches from x = -∞  (left side)
  "+Y"  tool approaches from y = +∞  (back)
  "-Y"  tool approaches from y = -∞  (front)
  "+Z"  tool approaches from z = +∞  (top — default first setup)
  "-Z"  tool approaches from z = -∞  (bottom)

DIRECTION_LABELS = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
DIRECTION_INDEX  = {label: i for i, label in enumerate(DIRECTION_LABELS)}
```

---

## Repository Additions

```
rpp-mvp/
├── phase3_setup_analysis.py     # IMPLEMENT — main module
└── tests/
    └── test_phase3.py           # IMPLEMENT — unit tests
```

No new subdirectories, no model files, no training data.

---
---

# IMPLEMENTATION — `phase3_setup_analysis.py`

---

## Constants

```python
DIRECTION_LABELS = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
DIRECTION_INDEX  = {label: i for i, label in enumerate(DIRECTION_LABELS)}
NUM_DIRECTIONS   = 6

# Rotation description when transitioning between setups
# Key: (from_direction, to_direction) → human-readable rotation string
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
# For any pair not in the dict, fall back to "refixtured"
```

---

## Core Geometric Functions

### `compute_surface_mask`

```python
def compute_surface_mask(grid: np.ndarray) -> np.ndarray:
    """
    Identify surface voxels: occupied voxels with at least one empty
    face-adjacent (6-connected) neighbour.

    Parameters
    ----------
    grid : (R, R, R) bool ndarray

    Returns
    -------
    surface : (R, R, R) bool ndarray
              True where voxel is occupied AND on the part surface.

    Algorithm
    ---------
    A voxel is interior if all 6 face-neighbours are occupied.
    A voxel is on the surface if it is occupied AND NOT interior.

    Use np.pad + boolean shifts to check all 6 neighbours efficiently.
    Do NOT use scipy.ndimage — keep numpy-only for portability.
    """
```

Implementation hint using boolean shifts:

```python
def compute_surface_mask(grid: np.ndarray) -> np.ndarray:
    padded = np.pad(grid, pad_width=1, mode='constant', constant_values=False)
    # A voxel has an empty face-neighbour if any neighbour is False
    has_empty_neighbour = (
        ~padded[2:, 1:-1, 1:-1] |   # +X neighbour
        ~padded[:-2, 1:-1, 1:-1] |   # -X neighbour
        ~padded[1:-1, 2:, 1:-1] |    # +Y neighbour
        ~padded[1:-1, :-2, 1:-1] |   # -Y neighbour
        ~padded[1:-1, 1:-1, 2:] |    # +Z neighbour
        ~padded[1:-1, 1:-1, :-2]     # -Z neighbour
    )
    return grid & has_empty_neighbour
```

---

### `compute_accessibility_map`

```python
def compute_accessibility_map(grid: np.ndarray) -> np.ndarray:
    """
    For each voxel, determine which of 6 tool-approach directions can
    reach it without passing through another occupied voxel.

    A voxel at position p is accessible from direction d if the ray
    from p in direction d to the grid boundary passes through no other
    occupied voxels.

    Parameters
    ----------
    grid : (R, R, R) bool ndarray

    Returns
    -------
    acc_map : (6, R, R, R) bool ndarray
              acc_map[d, x, y, z] is True if the voxel at (x,y,z) is
              accessible from direction DIRECTION_LABELS[d].
              True for ALL voxels (not just surface) — caller applies
              the surface mask.

    Algorithm
    ---------
    Use suffix/prefix cumulative sums along each axis.
    No loops required — fully vectorised with numpy.

    For +Z (index 4):
      accessible_plus_z[x,y,z] = True
        iff grid[x, y, z+1], grid[x, y, z+2], ..., grid[x, y, R-1]
        are ALL False (no occupied voxels above).
      Compute: above_count[x,y,z] = sum of grid[x,y,z+1:]
             = suffix_sum_z[x,y,z] - grid[x,y,z]
      Accessible iff above_count == 0.

    For -Z (index 5):
      below_count[x,y,z] = sum of grid[x,y,:z]
      Accessible iff below_count == 0.

    Apply the same logic for X and Y axes.
    """
```

Implementation:

```python
def compute_accessibility_map(grid: np.ndarray) -> np.ndarray:
    R = grid.shape[0]
    assert grid.shape == (R, R, R), "Grid must be cubic"
    g = grid.astype(np.int32)
    acc = np.ones((NUM_DIRECTIONS, R, R, R), dtype=bool)

    # +X (index 0): no occupied voxels at x+1, x+2, ..., R-1
    suffix_x = np.zeros_like(g)
    suffix_x[:-1] = np.cumsum(g[::-1], axis=0)[-2::-1]
    acc[0] = (suffix_x == 0)

    # -X (index 1): no occupied voxels at 0, 1, ..., x-1
    prefix_x = np.zeros_like(g)
    prefix_x[1:] = np.cumsum(g, axis=0)[:-1]
    acc[1] = (prefix_x == 0)

    # +Y (index 2)
    suffix_y = np.zeros_like(g)
    suffix_y[:, :-1] = np.cumsum(g[:, ::-1], axis=1)[:, -2::-1]
    acc[2] = (suffix_y == 0)

    # -Y (index 3)
    prefix_y = np.zeros_like(g)
    prefix_y[:, 1:] = np.cumsum(g, axis=1)[:, :-1]
    acc[3] = (prefix_y == 0)

    # +Z (index 4)
    suffix_z = np.zeros_like(g)
    suffix_z[:, :, :-1] = np.cumsum(g[:, :, ::-1], axis=2)[:, :, -2::-1]
    acc[4] = (suffix_z == 0)

    # -Z (index 5)
    prefix_z = np.zeros_like(g)
    prefix_z[:, :, 1:] = np.cumsum(g, axis=2)[:, :, :-1]
    acc[5] = (prefix_z == 0)

    return acc
```

---

### `greedy_setup_assignment`

```python
def greedy_setup_assignment(
    acc_map: np.ndarray,
    surface_mask: np.ndarray,
    coverage_threshold: float = 0.99,
) -> list[dict]:
    """
    Greedily select approach directions to cover the part surface.

    Each selected direction becomes one setup. Directions are chosen
    in order of marginal coverage gain (most uncovered surface first).
    Stops when cumulative coverage >= coverage_threshold or all 6
    directions have been evaluated.

    Parameters
    ----------
    acc_map         : (6, R, R, R) bool from compute_accessibility_map
    surface_mask    : (R, R, R) bool from compute_surface_mask
    coverage_threshold : fraction of surface that must be covered

    Returns
    -------
    List of setup dicts, ordered by selection sequence:
        [
          {
            "id": 0,
            "approach_direction": "+Z",
            "rotation_from_previous": "initial",
            "surface_voxel_count": int,
            "surface_coverage_fraction": float,
          },
          ...
        ]

    Notes
    -----
    - +Z is always tried first (conventional top setup).
    - "rotation_from_previous" for the first setup is always "initial".
    - For subsequent setups, look up ROTATION_DESCRIPTIONS; fall back
      to "refixtured" if the pair is not in the dict.
    - surface_coverage_fraction is the MARGINAL fraction added by this
      setup, not the cumulative total.
    """
```

---

### `infer_axis_requirement`

```python
def infer_axis_requirement(setup_directions: list[str]) -> int:
    """
    Infer the minimum CNC axis count from the required approach directions.

    Rules
    -----
    1 direction                          → 3-axis (single setup)
    2+ directions, all on one axis
      e.g. {+Z, -Z} or {+X, -X}        → 3-axis (flip / index)
    2 directions on two different axes
      e.g. {+Z, +X}                     → 4-axis
    3+ directions on two or more axes   → 5-axis

    Parameters
    ----------
    setup_directions : list of direction label strings e.g. ["+Z", "-Z"]

    Returns
    -------
    int — 3, 4, or 5
    """
    if len(setup_directions) <= 1:
        return 3
    axes_used = {d[1] for d in setup_directions}   # {"Z"} or {"Z","X"} etc.
    if len(axes_used) == 1:
        return 3
    elif len(axes_used) == 2:
        return 4
    else:
        return 5
```

---

### `map_features_to_setups`

```python
def map_features_to_setups(
    setups: list[dict],
    features: list[dict],
    acc_map: np.ndarray,
    surface_mask: np.ndarray,
) -> dict[str, list[str]]:
    """
    Assign each detected feature to the first setup that can access it.

    For the MVP, this is a simple heuristic: a feature is assigned to
    the setup whose approach direction has the highest surface coverage.
    Since we don't have per-feature voxel locations from Phase 2 (only
    class labels), we assign each feature to setup 0 unless the feature
    type is geometrically associated with a non-primary direction:

      flat_face       → always setup 0 (top face)
      through_hole    → setup with +Z or -Z (whichever is primary)
      blind_hole      → setup 0 (primary approach)
      rectangular_*   → setup 0
      circular_*      → setup 0
      rectangular_step→ first side setup (non-Z if available, else 0)
      chamfer         → setup 0
      fillet          → setup 0
      boss            → setup 0
      triangular_*    → setup 0

    Parameters
    ----------
    setups   : list of setup dicts from greedy_setup_assignment
    features : list of feature dicts from Phase 2 features.json
               [{"type": str, "confidence": float}, ...]
    acc_map  : (6, R, R, R) bool
    surface_mask : (R, R, R) bool

    Returns
    -------
    dict mapping setup id string → list of feature type strings
    e.g. {"0": ["flat_face", "through_hole"], "1": ["rectangular_step"]}
    """
```

---

## Public Interface

```python
def analyse_setups(
    voxel_path: str,
    output_dir: str,
    features_path: str | None = None,
    coverage_threshold: float = 0.99,
) -> dict:
    """
    Full setup analysis pipeline.

    Parameters
    ----------
    voxel_path         : path to voxel_{R}.npy from Phase 1
    output_dir         : directory for output files (created if absent)
    features_path      : path to features.json from Phase 2 (optional)
    coverage_threshold : greedy coverage stopping criterion (default 0.99)

    Returns
    -------
    dict matching setup_analysis.json schema above.

    Raises
    ------
    FileNotFoundError  — voxel_path does not exist
    ValueError         — voxel array is not 3D or not cubic
    RuntimeError       — surface mask is empty (degenerate geometry)

    Output files written
    --------------------
    {output_dir}/setup_analysis.json     — full result dict
    {output_dir}/accessibility_map.npy   — (6, R, R, R) bool
    {output_dir}/surface_mask.npy        — (R, R, R) bool
    """
```

### Full pipeline flow

```
analyse_setups(voxel_path, output_dir, features_path, coverage_threshold)
│
├─ 1. Validate voxel_path exists           → FileNotFoundError if not
├─ 2. os.makedirs(output_dir, exist_ok=True)
├─ 3. Load grid = np.load(voxel_path).astype(bool)
│      Validate ndim==3 and cubic          → ValueError if not
├─ 4. compute_surface_mask(grid)           → surface_mask (R,R,R)
│      Validate surface_mask.sum() > 0    → RuntimeError if empty
├─ 5. compute_accessibility_map(grid)     → acc_map (6,R,R,R)
├─ 6. Compute direction_coverage dict
│      For each direction d:
│        coverage[d] = (acc_map[i] & surface_mask).sum() / total_surface
├─ 7. greedy_setup_assignment(acc_map, surface_mask, threshold)
│      → setups list
├─ 8. infer_axis_requirement([s["approach_direction"] for s in setups])
│      → axis_requirement int
├─ 9. Compute inaccessible statistics
│      covered = union of acc_map[i] for selected directions & surface
│      inaccessible = surface_mask & ~covered
│      If inaccessible.sum() > 0 → add warning
├─ 10. Load features.json if features_path given → features list
├─ 11. map_features_to_setups(...)        → features_per_setup dict
├─ 12. Save accessibility_map.npy, surface_mask.npy (atomic np.save)
├─ 13. Assemble result dict
├─ 14. _write_json_atomic(result, setup_analysis.json)
└─ 15. return result
```

---

## CLI Entry Point

```python
if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(
        description="Analyse setup and stock rotation requirements from a voxel grid."
    )
    parser.add_argument("voxel_path",   help="Path to voxel_{R}.npy")
    parser.add_argument("output_dir",   help="Directory for output files")
    parser.add_argument("--features",   default=None,
                        help="Path to features.json from Phase 2 (optional)")
    parser.add_argument("--threshold",  type=float, default=0.99,
                        help="Surface coverage threshold (default 0.99)")
    args = parser.parse_args()

    result = analyse_setups(
        args.voxel_path,
        args.output_dir,
        features_path=args.features,
        coverage_threshold=args.threshold,
    )
    print(json.dumps(result, indent=2))
```

---
---

# TESTS — `tests/test_phase3.py`

```python
# tests/test_phase3.py
#
# Prerequisites: pytest tests/test_phase1.py must pass first.
# The Phase 1 CLI output is used:
#   data/processed/simple_block_cli/voxel_64.npy

import os
import json
import numpy as np
import pytest

from phase3_setup_analysis import (
    compute_surface_mask,
    compute_accessibility_map,
    greedy_setup_assignment,
    infer_axis_requirement,
    map_features_to_setups,
    analyse_setups,
    DIRECTION_LABELS,
    NUM_DIRECTIONS,
)

FIXTURE_VOXEL = "data/processed/simple_block_cli/voxel_64.npy"

# ── Synthetic grid helpers ────────────────────────────────────────────────────

def make_solid_cube(R: int = 16) -> np.ndarray:
    """Fully solid cube."""
    return np.ones((R, R, R), dtype=bool)


def make_hollow_cube(R: int = 16, wall: int = 2) -> np.ndarray:
    """Solid cube with a hollow interior."""
    g = np.ones((R, R, R), dtype=bool)
    g[wall:-wall, wall:-wall, wall:-wall] = False
    return g


def make_block_with_blind_pocket(R: int = 32) -> np.ndarray:
    """
    Solid block with a rectangular pocket cut from the top (+Z face).
    The pocket is only accessible from +Z, not from -Z.
    """
    g = np.ones((R, R, R), dtype=bool)
    pw = R // 4
    depth = R // 3
    cx, cy = R // 2, R // 2
    # Cut pocket from top face downward
    g[cx-pw:cx+pw, cy-pw:cy+pw, R-depth:] = False
    return g


def make_block_with_through_hole(R: int = 32) -> np.ndarray:
    """
    Solid block with a through-hole along Z axis.
    The hole interior is accessible from both +Z and -Z.
    """
    g = np.ones((R, R, R), dtype=bool)
    cx, cy = R // 2, R // 2
    radius = R // 8
    for x in range(R):
        for y in range(R):
            if (x - cx)**2 + (y - cy)**2 <= radius**2:
                g[x, y, :] = False
    return g


# ── Surface mask ─────────────────────────────────────────────────────────────

def test_surface_mask_subset_of_occupied():
    g = make_solid_cube(16)
    s = compute_surface_mask(g)
    assert np.all(s <= g), "Surface voxels must be a subset of occupied voxels"


def test_surface_mask_hollow_interior_not_flagged():
    """Interior voxels of a solid cube must NOT be surface voxels."""
    g = make_solid_cube(16)
    s = compute_surface_mask(g)
    # Centre voxel should not be a surface voxel
    c = 8
    assert not s[c, c, c], "Centre voxel of solid cube should not be surface"


def test_surface_mask_outer_layer_flagged():
    """All outer-face voxels of a solid cube must be surface voxels."""
    R = 16
    g = make_solid_cube(R)
    s = compute_surface_mask(g)
    # All voxels on any face should be surface
    assert s[0, :, :].all()
    assert s[-1, :, :].all()
    assert s[:, 0, :].all()
    assert s[:, -1, :].all()
    assert s[:, :, 0].all()
    assert s[:, :, -1].all()


def test_surface_mask_empty_grid():
    g = np.zeros((16, 16, 16), dtype=bool)
    s = compute_surface_mask(g)
    assert s.sum() == 0


def test_surface_mask_shape_preserved():
    g = make_solid_cube(32)
    s = compute_surface_mask(g)
    assert s.shape == g.shape


# ── Accessibility map ─────────────────────────────────────────────────────────

def test_accessibility_map_shape():
    g = make_solid_cube(16)
    a = compute_accessibility_map(g)
    assert a.shape == (NUM_DIRECTIONS, 16, 16, 16)
    assert a.dtype == bool


def test_solid_cube_top_face_accessible_from_plus_z():
    """Top face (+Z face) of a solid cube must be accessible from +Z."""
    R = 16
    g = make_solid_cube(R)
    s = compute_surface_mask(g)
    a = compute_accessibility_map(g)
    plus_z_idx = DIRECTION_LABELS.index("+Z")
    top_face_surface = s[:, :, R-1]
    assert top_face_surface.all(), "Top face should be entirely surface"
    assert a[plus_z_idx, :, :, R-1].all(), \
        "Top face must be accessible from +Z"


def test_solid_cube_bottom_face_accessible_from_minus_z():
    R = 16
    g = make_solid_cube(R)
    a = compute_accessibility_map(g)
    minus_z_idx = DIRECTION_LABELS.index("-Z")
    assert a[minus_z_idx, :, :, 0].all()


def test_solid_cube_top_face_not_accessible_from_minus_z():
    """The top face should NOT be accessible from -Z (solid block below it)."""
    R = 16
    g = make_solid_cube(R)
    a = compute_accessibility_map(g)
    minus_z_idx = DIRECTION_LABELS.index("-Z")
    # Top face voxels: z = R-1. Below them is the solid block.
    assert not a[minus_z_idx, :, :, R-1].any(), \
        "Top face should not be accessible from -Z"


def test_blind_pocket_only_accessible_from_top():
    """Pocket walls are only accessible from +Z, not -Z."""
    g = make_block_with_blind_pocket(32)
    a = compute_accessibility_map(g)
    s = compute_surface_mask(g)
    plus_z_idx  = DIRECTION_LABELS.index("+Z")
    minus_z_idx = DIRECTION_LABELS.index("-Z")
    # Surface fraction accessible from +Z must be > fraction from -Z
    surf_from_plus_z  = (a[plus_z_idx] & s).sum()
    surf_from_minus_z = (a[minus_z_idx] & s).sum()
    assert surf_from_plus_z > surf_from_minus_z


def test_accessibility_map_no_self_occlusion():
    """
    A voxel should not be blocked by itself.
    Single voxel in empty grid must be accessible from all 6 directions.
    """
    g = np.zeros((8, 8, 8), dtype=bool)
    g[4, 4, 4] = True
    a = compute_accessibility_map(g)
    for d_idx in range(NUM_DIRECTIONS):
        assert a[d_idx, 4, 4, 4], \
            f"Single voxel not accessible from {DIRECTION_LABELS[d_idx]}"


def test_accessibility_map_column_blocked():
    """
    Column of voxels: top voxel accessible from +Z, bottom voxel NOT
    accessible from +Z because the column blocks it.
    """
    g = np.zeros((8, 8, 8), dtype=bool)
    g[4, 4, 2] = True   # bottom of column
    g[4, 4, 5] = True   # top of column
    a = compute_accessibility_map(g)
    plus_z_idx = DIRECTION_LABELS.index("+Z")
    assert a[plus_z_idx, 4, 4, 5],  "Top of column accessible from +Z"
    assert not a[plus_z_idx, 4, 4, 2], "Bottom of column blocked from +Z"


# ── Greedy setup assignment ───────────────────────────────────────────────────

def test_solid_cube_needs_two_setups_top_and_bottom():
    """
    A solid block needs at minimum +Z (top) and -Z (bottom) to cover
    all six faces. greedy should select exactly 2 setups.
    """
    R = 16
    g = make_solid_cube(R)
    s = compute_surface_mask(g)
    a = compute_accessibility_map(g)
    setups = greedy_setup_assignment(a, s, coverage_threshold=0.99)
    directions = [st["approach_direction"] for st in setups]
    assert "+Z" in directions
    assert "-Z" in directions


def test_setup_coverage_fractions_sum_to_one():
    """Marginal coverage fractions should sum to approximately 1."""
    g = make_solid_cube(16)
    s = compute_surface_mask(g)
    a = compute_accessibility_map(g)
    setups = greedy_setup_assignment(a, s, coverage_threshold=0.99)
    total = sum(st["surface_coverage_fraction"] for st in setups)
    assert abs(total - 1.0) < 0.02


def test_setup_ids_sequential():
    g = make_solid_cube(16)
    s = compute_surface_mask(g)
    a = compute_accessibility_map(g)
    setups = greedy_setup_assignment(a, s)
    for i, st in enumerate(setups):
        assert st["id"] == i


def test_first_setup_rotation_is_initial():
    g = make_solid_cube(16)
    s = compute_surface_mask(g)
    a = compute_accessibility_map(g)
    setups = greedy_setup_assignment(a, s)
    assert setups[0]["rotation_from_previous"] == "initial"


def test_setup_has_required_keys():
    g = make_solid_cube(16)
    s = compute_surface_mask(g)
    a = compute_accessibility_map(g)
    setups = greedy_setup_assignment(a, s)
    required = {"id", "approach_direction", "rotation_from_previous",
                "surface_voxel_count", "surface_coverage_fraction"}
    for st in setups:
        assert required.issubset(st.keys())


# ── Axis requirement ──────────────────────────────────────────────────────────

def test_axis_requirement_single_direction():
    assert infer_axis_requirement(["+Z"]) == 3


def test_axis_requirement_flip_same_axis():
    assert infer_axis_requirement(["+Z", "-Z"]) == 3


def test_axis_requirement_two_different_axes():
    assert infer_axis_requirement(["+Z", "+X"]) == 4


def test_axis_requirement_three_axes():
    assert infer_axis_requirement(["+Z", "+X", "+Y"]) == 5


def test_axis_requirement_empty():
    assert infer_axis_requirement([]) == 3


# ── Full pipeline ─────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not os.path.exists(FIXTURE_VOXEL),
    reason="Phase 1 CLI output not available"
)
def test_analyse_setups_output_files(tmp_path):
    result = analyse_setups(FIXTURE_VOXEL, str(tmp_path))
    assert (tmp_path / "setup_analysis.json").exists()
    assert (tmp_path / "accessibility_map.npy").exists()
    assert (tmp_path / "surface_mask.npy").exists()


@pytest.mark.skipif(
    not os.path.exists(FIXTURE_VOXEL),
    reason="Phase 1 CLI output not available"
)
def test_analyse_setups_schema(tmp_path):
    result = analyse_setups(FIXTURE_VOXEL, str(tmp_path))
    required = [
        "setup_count", "axis_requirement", "setups",
        "direction_coverage", "total_surface_voxels",
        "covered_surface_voxels", "inaccessible_surface_voxels",
        "inaccessible_fraction", "features_per_setup",
        "voxel_file", "accessibility_map_file", "surface_mask_file",
        "warnings",
    ]
    for key in required:
        assert key in result, f"Missing key: {key}"


@pytest.mark.skipif(
    not os.path.exists(FIXTURE_VOXEL),
    reason="Phase 1 CLI output not available"
)
def test_analyse_setups_direction_coverage_all_classes(tmp_path):
    result = analyse_setups(FIXTURE_VOXEL, str(tmp_path))
    for d in DIRECTION_LABELS:
        assert d in result["direction_coverage"]
        assert 0.0 <= result["direction_coverage"][d] <= 1.0


@pytest.mark.skipif(
    not os.path.exists(FIXTURE_VOXEL),
    reason="Phase 1 CLI output not available"
)
def test_analyse_setups_simple_block_axis_3(tmp_path):
    """A simple rectangular block should only need 3-axis machining."""
    result = analyse_setups(FIXTURE_VOXEL, str(tmp_path))
    assert result["axis_requirement"] == 3


@pytest.mark.skipif(
    not os.path.exists(FIXTURE_VOXEL),
    reason="Phase 1 CLI output not available"
)
def test_analyse_setups_accessibility_map_shape(tmp_path):
    analyse_setups(FIXTURE_VOXEL, str(tmp_path))
    acc = np.load(tmp_path / "accessibility_map.npy")
    vox = np.load(FIXTURE_VOXEL)
    R = vox.shape[0]
    assert acc.shape == (NUM_DIRECTIONS, R, R, R)
    assert acc.dtype == bool


@pytest.mark.skipif(
    not os.path.exists(FIXTURE_VOXEL),
    reason="Phase 1 CLI output not available"
)
def test_analyse_setups_surface_mask_shape(tmp_path):
    analyse_setups(FIXTURE_VOXEL, str(tmp_path))
    vox  = np.load(FIXTURE_VOXEL)
    surf = np.load(tmp_path / "surface_mask.npy")
    assert surf.shape == vox.shape
    assert surf.dtype == bool


@pytest.mark.skipif(
    not os.path.exists(FIXTURE_VOXEL),
    reason="Phase 1 CLI output not available"
)
def test_analyse_setups_features_per_setup_empty_when_no_features(tmp_path):
    """Without features.json, features_per_setup values should be empty lists."""
    result = analyse_setups(FIXTURE_VOXEL, str(tmp_path))
    for setup_id, feats in result["features_per_setup"].items():
        assert isinstance(feats, list)


@pytest.mark.skipif(
    not os.path.exists(FIXTURE_VOXEL),
    reason="Phase 1 CLI output not available"
)
def test_analyse_setups_output_paths_absolute(tmp_path):
    result = analyse_setups(FIXTURE_VOXEL, str(tmp_path))
    assert os.path.isabs(result["voxel_file"])
    assert os.path.isabs(result["accessibility_map_file"])
    assert os.path.isabs(result["surface_mask_file"])


def test_analyse_setups_file_not_found():
    with pytest.raises(FileNotFoundError):
        analyse_setups("no_such_voxel.npy", "/tmp/out")


def test_analyse_setups_creates_output_dir(tmp_path):
    if not os.path.exists(FIXTURE_VOXEL):
        pytest.skip("Phase 1 CLI output not available")
    new_dir = tmp_path / "nested" / "subdir"
    assert not new_dir.exists()
    analyse_setups(FIXTURE_VOXEL, str(new_dir))
    assert new_dir.exists()
```

---
---

# QUICK-START COMMANDS

```bash
# 1. Run setup analysis on Phase 1 output (no Phase 2 features)
python phase3_setup_analysis.py \
    data/processed/simple_block_cli/voxel_64.npy \
    data/processed/simple_block_cli/

# 2. Run with Phase 2 features (if available)
python phase3_setup_analysis.py \
    data/processed/simple_block_cli/voxel_64.npy \
    data/processed/simple_block_cli/ \
    --features data/processed/simple_block_cli/features.json

# 3. Run all tests
pytest tests/test_phase3.py -v

# 4. Run all phases together
pytest tests/ -v
```

---
---

# ACCEPTANCE CRITERIA

- [ ] `pytest tests/test_phase3.py -v` — all tests pass
- [ ] `pytest tests/test_phase1.py tests/test_phase2.py tests/test_phase3.py -v`
      — all 66 + Phase 3 tests pass
- [ ] CLI produces `setup_analysis.json`, `accessibility_map.npy`,
      `surface_mask.npy` for `simple_block_cli/voxel_64.npy`
- [ ] `simple_block.stp` produces `axis_requirement == 3`
- [ ] `accessibility_map.npy` shape is `(6, 64, 64, 64)` and dtype is `bool`
- [ ] Top face of simple block (`z = R-1`) is 100% accessible from `+Z`
- [ ] `direction_coverage` contains all 6 direction keys with values in [0, 1]
- [ ] `inaccessible_fraction` is 0.0 for a solid rectangular block
      (all surface voxels reachable from some principal direction)
- [ ] `features_per_setup` is populated when `features.json` is passed
- [ ] All output paths in JSON are absolute
- [ ] `warnings` key is always present as a list

---
---

# NOTES FOR CODEX

1. **Numpy-only, no scipy.** Do not import scipy for the geometric
   computations. The suffix/prefix cumsum approach is sufficient and
   keeps the dependency footprint small.

2. **The accessibility map covers ALL voxels, not just surface.**
   Apply the surface_mask when computing coverage fractions.
   Storing the full (6, R, R, R) map allows Phase 4 to query
   accessibility for arbitrary voxel positions.

3. **Greedy should try +Z first.** By convention, the first setup
   is always the top (+Z) approach. Start the greedy selection from
   +Z regardless of which direction has the theoretical best marginal
   coverage.

4. **coverage_threshold=0.99 not 1.0.** Some surface voxels on
   sharp interior edges may not be accessible from any principal
   direction (undercuts). Using 0.99 prevents an infinite loop
   while flagging truly inaccessible geometry in the warnings list.

5. **features_per_setup keys are strings, not ints.** JSON object
   keys are always strings. Use `str(setup_id)` as the dict key.

6. **Atomic writes for all three output files.** Use temp-file +
   os.replace for JSON. For .npy files, write to a temp path then
   rename:
   ```python
   tmp = voxel_path + ".tmp.npy"
   np.save(tmp, array)
   os.replace(tmp, voxel_path)
   ```

7. **inaccessible_surface_voxels counts surface voxels not covered
   by ANY of the selected setups** — not all 6 directions. If the
   greedy algorithm selected only 2 directions and 1% of surface is
   inaccessible from both, that 1% is reported here.

8. **The surface_mask.npy dtype must be bool**, not uint8 or int.
   Downstream phases use boolean indexing directly.
