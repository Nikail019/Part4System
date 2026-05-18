# AGENTS.md — Phase 4: Process Plan Generation

## Context

Phases 1–3 are complete and passing (100 tests).

- Phase 1 → `voxel_{R}.npy` + `metadata.json`
- Phase 2 → `features.json`
- Phase 3 → `setup_analysis.json` + `accessibility_map.npy` + `surface_mask.npy`

Phase 4 reads the outputs of all three upstream phases and produces a sequenced,
setup-aware process plan as a structured JSON document. This phase is
**rule-based** — no ML model, no training. The novelty is the integration of
voxel-derived features with voxel-derived setup analysis into a coherent plan.

---

## What Phase 4 Must Deliver

```
INPUTS:
  metadata.json          from Phase 1   (part dimensions, volume)
  features.json          from Phase 2   (detected feature types + confidence)
  setup_analysis.json    from Phase 3   (setup count, axis requirement,
                                         features_per_setup)

OUTPUT:
  process_plan.json      ordered list of machining operations
```

### `process_plan.json` schema

```json
{
  "operations": [
    {
      "step":               1,
      "setup_id":           0,
      "approach_direction": "+Z",
      "feature_type":       "flat_face",
      "operation_type":     "face_mill_rough",
      "tool_type":          "face_mill",
      "phase":              "roughing",
      "notes":              "Establish datum reference surface"
    },
    {
      "step":               2,
      "setup_id":           0,
      "approach_direction": "+Z",
      "feature_type":       "rectangular_pocket",
      "operation_type":     "endmill_rough",
      "tool_type":          "flat_endmill",
      "phase":              "roughing",
      "notes":              ""
    }
  ],

  "operation_count":  8,
  "setup_count":      2,
  "axis_requirement": 3,
  "tool_list":        ["face_mill", "flat_endmill", "centre_drill", "twist_drill"],

  "source_files": {
    "metadata":       "/abs/path/to/metadata.json",
    "features":       "/abs/path/to/features.json",
    "setup_analysis": "/abs/path/to/setup_analysis.json"
  },

  "process_plan_file": "/abs/path/to/process_plan.json",
  "warnings":          []
}
```

---

## Operation Map

Define this as a module-level constant. Each feature type maps to an ordered
list of operations. Operations are applied in list order within their phase.

```python
# Mapping from feature_type → list of operation dicts
# Each dict: {"type": str, "tool": str, "phase": "roughing"|"finishing"}

OPERATION_MAP = {
    "flat_face": [
        {"type": "face_mill_rough",        "tool": "face_mill",      "phase": "roughing"},
        {"type": "face_mill_finish",       "tool": "face_mill",      "phase": "finishing"},
    ],
    "through_hole": [
        {"type": "centre_drill",           "tool": "centre_drill",   "phase": "roughing"},
        {"type": "drill",                  "tool": "twist_drill",    "phase": "roughing"},
    ],
    "blind_hole": [
        {"type": "centre_drill",           "tool": "centre_drill",   "phase": "roughing"},
        {"type": "drill_peck",             "tool": "twist_drill",    "phase": "roughing"},
    ],
    "rectangular_pocket": [
        {"type": "endmill_rough",          "tool": "flat_endmill",   "phase": "roughing"},
        {"type": "endmill_finish",         "tool": "flat_endmill",   "phase": "finishing"},
    ],
    "circular_pocket": [
        {"type": "endmill_rough",          "tool": "flat_endmill",   "phase": "roughing"},
        {"type": "endmill_finish",         "tool": "flat_endmill",   "phase": "finishing"},
    ],
    "rectangular_slot": [
        {"type": "endmill_rough",          "tool": "flat_endmill",   "phase": "roughing"},
        {"type": "endmill_finish",         "tool": "flat_endmill",   "phase": "finishing"},
    ],
    "circular_slot": [
        {"type": "endmill_rough",          "tool": "flat_endmill",   "phase": "roughing"},
        {"type": "endmill_finish",         "tool": "flat_endmill",   "phase": "finishing"},
    ],
    "rectangular_step": [
        {"type": "shoulder_mill_rough",    "tool": "shoulder_mill",  "phase": "roughing"},
        {"type": "shoulder_mill_finish",   "tool": "shoulder_mill",  "phase": "finishing"},
    ],
    "chamfer": [
        {"type": "chamfer_mill",           "tool": "chamfer_mill",   "phase": "finishing"},
    ],
    "fillet": [
        {"type": "ball_endmill_finish",    "tool": "ball_endmill",   "phase": "finishing"},
    ],
    "boss": [
        {"type": "endmill_rough",          "tool": "flat_endmill",   "phase": "roughing"},
        {"type": "endmill_finish",         "tool": "flat_endmill",   "phase": "finishing"},
    ],
    "triangular_pocket": [
        {"type": "endmill_rough",          "tool": "flat_endmill",   "phase": "roughing"},
        {"type": "endmill_finish",         "tool": "flat_endmill",   "phase": "finishing"},
    ],
}
```

---

## Sequencing Rules

Operations must be ordered following these precedence rules, applied in order:

**Rule 1 — Setup ordering.**
All operations in setup 0 come before setup 1, setup 1 before setup 2, etc.
Setup order is taken directly from `setup_analysis.json`.

**Rule 2 — Phase ordering within each setup.**
Within a setup: all `"roughing"` operations come before all `"finishing"`
operations.

**Rule 3 — Feature priority within roughing phase.**
Within the roughing phase of a setup, sort features by this priority table
(lower number = earlier):

```python
ROUGHING_PRIORITY = {
    "flat_face":          0,   # datum establishment — always first
    "rectangular_step":   1,   # large material removal early
    "boss":               2,
    "rectangular_pocket": 3,
    "circular_pocket":    4,
    "triangular_pocket":  5,
    "rectangular_slot":   6,
    "circular_slot":      7,
    "blind_hole":         8,
    "through_hole":       9,
    "chamfer":           10,   # deferred to finishing (no roughing op)
    "fillet":            11,   # deferred to finishing (no roughing op)
}
# Default priority for unrecognised features: 99
```

**Rule 4 — Feature priority within finishing phase.**
Same priority table applies. `flat_face` finishes first, then pockets/steps,
then holes, then chamfers, then fillets last.

**Rule 5 — Duplicate operations.**
If the same feature type appears more than once in `features.json` (possible
due to multi-label output), generate operations only once per feature type per
setup. Deduplicate by feature_type before expanding to operations.

---

## Operation Notes

Populate the `"notes"` field with a brief, human-readable description:

```python
OPERATION_NOTES = {
    "face_mill_rough":       "Establish datum reference surface",
    "face_mill_finish":      "Achieve final face flatness",
    "centre_drill":          "Spot drill for hole location accuracy",
    "drill":                 "Drill through-hole to nominal diameter",
    "drill_peck":            "Peck drill blind hole to depth",
    "endmill_rough":         "Rough pocket / slot to within 0.5mm of final depth",
    "endmill_finish":        "Finish to final profile",
    "shoulder_mill_rough":   "Rough shoulder step",
    "shoulder_mill_finish":  "Finish step to final dimension",
    "chamfer_mill":          "Apply chamfer to edges",
    "ball_endmill_finish":   "Blend fillet radius",
}
# Default for any type not in this dict: ""
```

---

## Repository Additions

```
rpp-mvp/
├── phase4_process_plan.py      # IMPLEMENT — main module
└── tests/
    └── test_phase4.py          # IMPLEMENT — unit tests
```

---
---

# IMPLEMENTATION — `phase4_process_plan.py`

---

## Helper: Load and Validate Inputs

```python
def _load_json(path: str, label: str) -> dict:
    """Load JSON file. Raises FileNotFoundError or ValueError on failure."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")
    with open(path) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {label}: {e}")
```

## Helper: Expand Feature to Operations

```python
def _expand_feature(
    feature_type: str,
    setup_id: int,
    approach_direction: str,
) -> list[dict]:
    """
    Look up feature_type in OPERATION_MAP.
    Return list of operation dicts with setup_id and direction filled in.
    Returns empty list with a warning if feature_type is not in OPERATION_MAP.
    """
```

## Helper: Build Operation List

```python
def _build_operations(
    features_per_setup: dict[str, list[str]],
    setup_list: list[dict],
) -> tuple[list[dict], list[str]]:
    """
    Apply sequencing rules to produce the ordered operation list.

    Parameters
    ----------
    features_per_setup : from setup_analysis.json
                         {"0": ["flat_face", "through_hole"], "1": [...]}
    setup_list         : list of setup dicts from setup_analysis.json
                         [{"id":0, "approach_direction":"+Z", ...}, ...]

    Returns
    -------
    (operations, warnings)
    operations : list of operation dicts (without "step" field yet)
    warnings   : list of warning strings

    Algorithm
    ---------
    For each setup (in order):
      Deduplicate feature list for this setup.
      Separate into roughing features and finishing features using
      ROUGHING_PRIORITY — all features have roughing ops EXCEPT
      chamfer and fillet which are finishing-only.
      Sort roughing features by ROUGHING_PRIORITY.
      Sort finishing features by ROUGHING_PRIORITY.
      Expand each feature to its operations, keeping phase order:
        all roughing ops first, then all finishing ops.
    Concatenate across setups.
    """
```

## Public Interface

```python
def generate_process_plan(
    metadata_path: str,
    features_path: str,
    setup_analysis_path: str,
    output_dir: str,
    confidence_threshold: float = 0.5,
) -> dict:
    """
    Generate a sequenced process plan from Phase 1–3 outputs.

    Parameters
    ----------
    metadata_path       : path to metadata.json from Phase 1
    features_path       : path to features.json from Phase 2
    setup_analysis_path : path to setup_analysis.json from Phase 3
    output_dir          : directory for process_plan.json
    confidence_threshold: features below this confidence are excluded
                          and noted in warnings (default 0.5)

    Returns
    -------
    dict matching process_plan.json schema.

    Raises
    ------
    FileNotFoundError  — any input file does not exist
    ValueError         — any input file is invalid JSON or missing keys

    Output files
    ------------
    {output_dir}/process_plan.json   written atomically
    """
```

### Full pipeline flow

```
generate_process_plan(metadata_path, features_path, setup_path, output_dir)
│
├─ 1.  Load and validate all three input JSON files
│       → FileNotFoundError / ValueError if any missing or malformed
├─ 2.  Filter features by confidence_threshold
│       → add warning for each excluded feature
├─ 3.  Add warning if features list is empty after filtering
├─ 4.  Read features_per_setup from setup_analysis.json
│       If features_per_setup values are all empty lists:
│         fall back to assigning ALL detected features to setup 0
│         add a note to warnings explaining this
├─ 5.  _build_operations(features_per_setup, setup_list)
│       → (raw_ops, build_warnings)
├─ 6.  Number operations: assign "step" field 1..N
├─ 7.  Collect unique tool_list from operations
├─ 8.  Assemble result dict
├─ 9.  _write_json_atomic(result, process_plan.json)
└─ 10. return result
```

### Fallback for empty `features_per_setup`

Phase 3 sets `features_per_setup` values to empty lists when no `features.json`
was provided. In this case Phase 4 must still produce a useful plan:

```python
def _resolve_features_per_setup(
    setup_analysis: dict,
    features: list[dict],
) -> dict[str, list[str]]:
    """
    If setup_analysis has empty features_per_setup, distribute
    all detected features to setup 0 (primary setup).
    Otherwise use features_per_setup as-is.
    """
    fps = setup_analysis.get("features_per_setup", {})
    all_empty = all(len(v) == 0 for v in fps.values())

    if all_empty and features:
        # Assign all features to setup 0
        result = {str(s["id"]): [] for s in setup_analysis["setups"]}
        result["0"] = [f["type"] for f in features]
        return result

    return fps
```

## CLI Entry Point

```python
if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(
        description="Generate a sequenced process plan from Phase 1–3 outputs."
    )
    parser.add_argument("metadata_path",
                        help="Path to metadata.json from Phase 1")
    parser.add_argument("features_path",
                        help="Path to features.json from Phase 2")
    parser.add_argument("setup_analysis_path",
                        help="Path to setup_analysis.json from Phase 3")
    parser.add_argument("output_dir",
                        help="Directory to write process_plan.json")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Feature confidence threshold (default 0.5)")
    args = parser.parse_args()

    result = generate_process_plan(
        args.metadata_path,
        args.features_path,
        args.setup_analysis_path,
        args.output_dir,
        confidence_threshold=args.confidence,
    )
    print(json.dumps(result, indent=2))
```

---
---

# TESTS — `tests/test_phase4.py`

```python
# tests/test_phase4.py
#
# Prerequisites:
#   pytest tests/test_phase1.py tests/test_phase2.py tests/test_phase3.py
#   must all pass first.
#
# Phase 4 tests use both:
#   (a) synthetic in-memory JSON fixtures (no disk I/O needed for most tests)
#   (b) real pipeline output from simple_block_cli/ where available

import os, json, copy
import pytest

from phase4_process_plan import (
    generate_process_plan,
    OPERATION_MAP,
    ROUGHING_PRIORITY,
    OPERATION_NOTES,
    _build_operations,
    _resolve_features_per_setup,
)

# ── Shared fixtures ───────────────────────────────────────────────────────────

SIMPLE_METADATA = {
    "bounding_box_mm": {"x": 100.0, "y": 60.0, "z": 40.0},
    "volume_mm3": 240000.0,
    "surface_area_mm2": 28800.0,
}

SIMPLE_FEATURES = {
    "features": [
        {"type": "flat_face",          "confidence": 0.99},
        {"type": "rectangular_pocket", "confidence": 0.87},
        {"type": "through_hole",       "confidence": 0.81},
    ],
    "feature_count": 3,
    "threshold": 0.5,
}

SIMPLE_SETUP = {
    "setup_count": 2,
    "axis_requirement": 3,
    "setups": [
        {"id": 0, "approach_direction": "+Z",
         "rotation_from_previous": "initial",
         "surface_voxel_count": 1820,
         "surface_coverage_fraction": 0.74},
        {"id": 1, "approach_direction": "-Z",
         "rotation_from_previous": "flip_around_X_180",
         "surface_voxel_count": 640,
         "surface_coverage_fraction": 0.26},
    ],
    "features_per_setup": {
        "0": ["flat_face", "rectangular_pocket", "through_hole"],
        "1": ["flat_face"],
    },
    "warnings": [],
}

CLI_DIR = "data/processed/simple_block_cli"


@pytest.fixture
def input_files(tmp_path):
    """Write synthetic JSON inputs to tmp_path. Return path dict."""
    meta_path   = tmp_path / "metadata.json"
    feat_path   = tmp_path / "features.json"
    setup_path  = tmp_path / "setup_analysis.json"
    meta_path.write_text(json.dumps(SIMPLE_METADATA))
    feat_path.write_text(json.dumps(SIMPLE_FEATURES))
    setup_path.write_text(json.dumps(SIMPLE_SETUP))
    return {
        "metadata": str(meta_path),
        "features": str(feat_path),
        "setup":    str(setup_path),
        "out":      str(tmp_path),
    }


# ── OPERATION_MAP correctness ─────────────────────────────────────────────────

def test_operation_map_all_feature_classes_present():
    """Every Phase 2 feature class must have an entry in OPERATION_MAP."""
    from models.feature_net import FEATURE_NAMES
    for name in FEATURE_NAMES:
        assert name in OPERATION_MAP, f"Missing from OPERATION_MAP: {name}"


def test_operation_map_phases_valid():
    """Every operation phase must be 'roughing' or 'finishing'."""
    for feature, ops in OPERATION_MAP.items():
        for op in ops:
            assert op["phase"] in ("roughing", "finishing"), \
                f"{feature}: invalid phase '{op['phase']}'"


def test_operation_map_required_keys():
    for feature, ops in OPERATION_MAP.items():
        for op in ops:
            assert "type"  in op, f"{feature} op missing 'type'"
            assert "tool"  in op, f"{feature} op missing 'tool'"
            assert "phase" in op, f"{feature} op missing 'phase'"


def test_roughing_priority_all_features_present():
    from models.feature_net import FEATURE_NAMES
    for name in FEATURE_NAMES:
        assert name in ROUGHING_PRIORITY, f"Missing from ROUGHING_PRIORITY: {name}"


def test_flat_face_priority_is_zero():
    assert ROUGHING_PRIORITY["flat_face"] == 0


# ── _build_operations sequencing ─────────────────────────────────────────────

def test_flat_face_is_first_operation():
    features_per_setup = {"0": ["through_hole", "flat_face", "rectangular_pocket"]}
    setup_list = [{"id": 0, "approach_direction": "+Z"}]
    ops, _ = _build_operations(features_per_setup, setup_list)
    assert ops[0]["feature_type"] == "flat_face"


def test_roughing_before_finishing():
    features_per_setup = {"0": ["rectangular_pocket"]}
    setup_list = [{"id": 0, "approach_direction": "+Z"}]
    ops, _ = _build_operations(features_per_setup, setup_list)
    phases = [op["phase"] for op in ops]
    # All roughing ops must come before any finishing op
    seen_finishing = False
    for phase in phases:
        if phase == "finishing":
            seen_finishing = True
        if seen_finishing:
            assert phase == "finishing", "Roughing op found after finishing op"


def test_setup_0_before_setup_1():
    features_per_setup = {"0": ["flat_face"], "1": ["flat_face"]}
    setup_list = [
        {"id": 0, "approach_direction": "+Z"},
        {"id": 1, "approach_direction": "-Z"},
    ]
    ops, _ = _build_operations(features_per_setup, setup_list)
    setup_ids = [op["setup_id"] for op in ops]
    last_0 = max(i for i, s in enumerate(setup_ids) if s == 0)
    first_1 = min(i for i, s in enumerate(setup_ids) if s == 1)
    assert last_0 < first_1, "Setup 0 ops must precede setup 1 ops"


def test_duplicate_features_deduplicated():
    """Same feature type appearing twice should produce operations only once."""
    features_per_setup = {
        "0": ["flat_face", "flat_face", "rectangular_pocket"]
    }
    setup_list = [{"id": 0, "approach_direction": "+Z"}]
    ops, _ = _build_operations(features_per_setup, setup_list)
    flat_ops = [o for o in ops if o["feature_type"] == "flat_face"]
    expected = len(OPERATION_MAP["flat_face"])
    assert len(flat_ops) == expected, \
        f"Expected {expected} flat_face ops, got {len(flat_ops)}"


def test_unknown_feature_generates_warning():
    features_per_setup = {"0": ["unknown_mystery_feature"]}
    setup_list = [{"id": 0, "approach_direction": "+Z"}]
    ops, warnings = _build_operations(features_per_setup, setup_list)
    assert any("unknown_mystery_feature" in w for w in warnings)


def test_chamfer_is_finishing_only():
    """chamfer has no roughing operations — must appear only in finishing."""
    features_per_setup = {"0": ["chamfer"]}
    setup_list = [{"id": 0, "approach_direction": "+Z"}]
    ops, _ = _build_operations(features_per_setup, setup_list)
    for op in ops:
        if op["feature_type"] == "chamfer":
            assert op["phase"] == "finishing"


# ── _resolve_features_per_setup ──────────────────────────────────────────────

def test_resolve_uses_phase3_when_populated():
    setup = copy.deepcopy(SIMPLE_SETUP)
    features = SIMPLE_FEATURES["features"]
    result = _resolve_features_per_setup(setup, features)
    assert result == setup["features_per_setup"]


def test_resolve_falls_back_to_setup0_when_empty():
    setup = copy.deepcopy(SIMPLE_SETUP)
    setup["features_per_setup"] = {"0": [], "1": []}
    features = [{"type": "flat_face", "confidence": 0.99}]
    result = _resolve_features_per_setup(setup, features)
    assert "flat_face" in result["0"]


# ── generate_process_plan ────────────────────────────────────────────────────

def test_output_file_created(input_files):
    generate_process_plan(
        input_files["metadata"],
        input_files["features"],
        input_files["setup"],
        input_files["out"],
    )
    assert os.path.exists(os.path.join(input_files["out"], "process_plan.json"))


def test_schema_keys_present(input_files):
    result = generate_process_plan(
        input_files["metadata"],
        input_files["features"],
        input_files["setup"],
        input_files["out"],
    )
    required = [
        "operations", "operation_count", "setup_count",
        "axis_requirement", "tool_list", "source_files",
        "process_plan_file", "warnings",
    ]
    for key in required:
        assert key in result, f"Missing key: {key}"


def test_operation_steps_sequential(input_files):
    result = generate_process_plan(
        input_files["metadata"],
        input_files["features"],
        input_files["setup"],
        input_files["out"],
    )
    steps = [op["step"] for op in result["operations"]]
    assert steps == list(range(1, len(steps) + 1))


def test_operation_count_matches_list(input_files):
    result = generate_process_plan(
        input_files["metadata"],
        input_files["features"],
        input_files["setup"],
        input_files["out"],
    )
    assert result["operation_count"] == len(result["operations"])


def test_tool_list_is_deduplicated(input_files):
    result = generate_process_plan(
        input_files["metadata"],
        input_files["features"],
        input_files["setup"],
        input_files["out"],
    )
    assert len(result["tool_list"]) == len(set(result["tool_list"]))


def test_operation_required_keys(input_files):
    result = generate_process_plan(
        input_files["metadata"],
        input_files["features"],
        input_files["setup"],
        input_files["out"],
    )
    required = {
        "step", "setup_id", "approach_direction",
        "feature_type", "operation_type", "tool_type",
        "phase", "notes",
    }
    for op in result["operations"]:
        assert required.issubset(op.keys()), f"Operation missing keys: {op}"


def test_axis_requirement_passed_through(input_files):
    result = generate_process_plan(
        input_files["metadata"],
        input_files["features"],
        input_files["setup"],
        input_files["out"],
    )
    assert result["axis_requirement"] == SIMPLE_SETUP["axis_requirement"]


def test_setup_count_passed_through(input_files):
    result = generate_process_plan(
        input_files["metadata"],
        input_files["features"],
        input_files["setup"],
        input_files["out"],
    )
    assert result["setup_count"] == SIMPLE_SETUP["setup_count"]


def test_confidence_threshold_filters_features(input_files, tmp_path):
    """Features below threshold must be excluded from the plan."""
    features_high_threshold = {
        "features": [
            {"type": "flat_face",          "confidence": 0.99},
            {"type": "rectangular_pocket", "confidence": 0.30},  # below 0.5
        ],
        "feature_count": 2,
        "threshold": 0.5,
    }
    feat_path = tmp_path / "features_low.json"
    feat_path.write_text(json.dumps(features_high_threshold))

    result = generate_process_plan(
        input_files["metadata"],
        str(feat_path),
        input_files["setup"],
        input_files["out"],
        confidence_threshold=0.5,
    )
    feature_types = {op["feature_type"] for op in result["operations"]}
    assert "rectangular_pocket" not in feature_types
    assert any("rectangular_pocket" in w for w in result["warnings"])


def test_missing_metadata_raises(input_files):
    with pytest.raises(FileNotFoundError):
        generate_process_plan(
            "no_metadata.json",
            input_files["features"],
            input_files["setup"],
            input_files["out"],
        )


def test_missing_features_raises(input_files):
    with pytest.raises(FileNotFoundError):
        generate_process_plan(
            input_files["metadata"],
            "no_features.json",
            input_files["setup"],
            input_files["out"],
        )


def test_missing_setup_raises(input_files):
    with pytest.raises(FileNotFoundError):
        generate_process_plan(
            input_files["metadata"],
            input_files["features"],
            "no_setup.json",
            input_files["out"],
        )


def test_output_path_is_absolute(input_files):
    result = generate_process_plan(
        input_files["metadata"],
        input_files["features"],
        input_files["setup"],
        input_files["out"],
    )
    assert os.path.isabs(result["process_plan_file"])


def test_warnings_is_list(input_files):
    result = generate_process_plan(
        input_files["metadata"],
        input_files["features"],
        input_files["setup"],
        input_files["out"],
    )
    assert isinstance(result["warnings"], list)


def test_written_json_is_valid(input_files):
    result = generate_process_plan(
        input_files["metadata"],
        input_files["features"],
        input_files["setup"],
        input_files["out"],
    )
    with open(result["process_plan_file"]) as f:
        on_disk = json.load(f)
    assert on_disk["operation_count"] == result["operation_count"]


# ── End-to-end CLI test (uses real Phase 1–3 output) ─────────────────────────

@pytest.mark.skipif(
    not all(os.path.exists(os.path.join(CLI_DIR, f)) for f in [
        "metadata.json", "features.json", "setup_analysis.json"
    ]),
    reason="Real Phase 1–3 CLI outputs not available"
)
def test_full_pipeline_on_simple_block(tmp_path):
    result = generate_process_plan(
        os.path.join(CLI_DIR, "metadata.json"),
        os.path.join(CLI_DIR, "features.json"),
        os.path.join(CLI_DIR, "setup_analysis.json"),
        str(tmp_path),
    )
    assert result["operation_count"] > 0
    # Simple block: flat_face always detected → at least 2 ops
    assert result["operation_count"] >= 2
    feature_types = {op["feature_type"] for op in result["operations"]}
    assert "flat_face" in feature_types
```

---
---

# QUICK-START COMMANDS

```bash
# 1. Run Phase 4 on simple_block_cli outputs
#    (requires Phases 1–3 CLI outputs to already exist)
python phase4_process_plan.py \
    data/processed/simple_block_cli/metadata.json       \
    data/processed/simple_block_cli/features.json       \
    data/processed/simple_block_cli/setup_analysis.json \
    data/processed/simple_block_cli/

# 2. Run Phase 4 tests
pytest tests/test_phase4.py -v

# 3. Run full suite
pytest tests/ -v
```

---
---

# ACCEPTANCE CRITERIA

- [ ] `pytest tests/test_phase4.py -v` — all tests pass
- [ ] `pytest tests/ -v` — all Phase 1–4 tests pass (≥130 total)
- [ ] CLI produces `process_plan.json` for `simple_block_cli/`
- [ ] `flat_face` is the first feature in the first operation
- [ ] All roughing operations precede finishing operations within each setup
- [ ] `operation_count` equals `len(operations)`
- [ ] `tool_list` contains no duplicates
- [ ] Every operation dict contains all 8 required keys
- [ ] Features below `confidence_threshold` are excluded + warned
- [ ] `process_plan_file` path in JSON is absolute and exists on disk
- [ ] `OPERATION_MAP` has an entry for all 12 Phase 2 feature classes

---
---

# NOTES FOR CODEX

1. **OPERATION_MAP must cover all 12 Phase 2 feature classes.**
   The test `test_operation_map_all_feature_classes_present` imports
   `FEATURE_NAMES` from `models/feature_net.py` and checks each one.
   If any class is missing, that test will fail.

2. **`_build_operations` and `_resolve_features_per_setup` must be
   importable directly** (not just callable through `generate_process_plan`).
   Tests import them explicitly for unit testing.

3. **`step` numbering starts at 1**, not 0. The step field is the
   human-readable sequence number for the process plan.

4. **`setup_id` in each operation is an integer**, not a string,
   even though `features_per_setup` keys are strings in JSON.

5. **Atomic JSON write.** Use the same temp-file + os.replace pattern
   as Phases 1–3.

6. **Do not invent tool dimensions.** The MVP does not have exact
   feature dimensions from Phase 2 (only class labels). `tool_type`
   is a generic label (e.g., `"flat_endmill"`) not a sized specification
   (e.g., `"flat_endmill_12mm"`). Sizing is deferred to the PMI
   integration phase.

7. **Empty features list after filtering** is a valid (if unusual)
   input. Return an empty `operations` list, `operation_count: 0`,
   and add a warning. Do not raise an exception.

8. **`features_per_setup` fallback.** If Phase 3 was run without
   a `features.json` input, its `features_per_setup` will have empty
   lists. `_resolve_features_per_setup` must detect this and assign
   all Phase 2 features to setup 0. The test
   `test_resolve_falls_back_to_setup0_when_empty` verifies this.
