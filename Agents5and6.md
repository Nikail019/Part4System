# AGENTS.md — Phase 5 + Phase 6: Machining Time Estimation & Job Quotation

## Context

Phases 1–4 are complete and passing (129 tests).

- Phase 1 → `voxel_{R}.npy` + `metadata.json`
- Phase 2 → `features.json`
- Phase 3 → `setup_analysis.json`
- Phase 4 → `process_plan.json`

Phase 5 reads `process_plan.json` and `metadata.json` to estimate machining
time per operation and total cycle time.

Phase 6 reads the time estimate, loads a factory capability profile, computes
a cost estimate, runs capability checks, and produces a final `quotation.json`
with an ACCEPT or REJECT recommendation.

Together these phases complete the MVP pipeline.

---

## Full Pipeline After These Phases

```
STEP file
    │
    ├─ Phase 1 ──► voxel_64.npy + metadata.json
    ├─ Phase 2 ──► features.json
    ├─ Phase 3 ──► setup_analysis.json
    ├─ Phase 4 ──► process_plan.json
    ├─ Phase 5 ──► time_estimate.json
    └─ Phase 6 ──► quotation.json          ← business-facing output
                   ACCEPT / REJECT + cost + time + capability checks
```

---

## Repository Additions

```
rpp-mvp/
├── phase5_time_estimate.py       # IMPLEMENT
├── phase6_quotation.py           # IMPLEMENT
├── factory_profiles/
│   ├── nash_nz.json              # IMPLEMENT — example factory profile
│   └── generic_3axis.json        # IMPLEMENT — minimal test profile
└── tests/
    ├── test_phase5.py            # IMPLEMENT
    └── test_phase6.py            # IMPLEMENT
```

---
---

# PHASE 5 — Machining Time Estimation

---

## Output: `time_estimate.json`

```json
{
  "total_time_min":     87.3,
  "machining_time_min": 57.3,
  "setup_time_min":     30.0,

  "breakdown": [
    {
      "step":                        1,
      "operation_type":              "face_mill_rough",
      "feature_type":                "flat_face",
      "phase":                       "roughing",
      "estimated_removal_volume_mm3": 12000.0,
      "mrr_mm3_per_min":             8000.0,
      "machining_time_min":          1.50,
      "tool_change_time_min":        2.00,
      "operation_time_min":          3.50
    }
  ],

  "material":                  "aluminium_6061",
  "setup_count":               2,
  "setup_time_per_setup_min":  15.0,
  "tool_change_count":         4,
  "tool_change_time_min":      2.0,

  "assumptions": [
    "Removal volume distributed by feature type weight heuristic",
    "Cutting parameters for aluminium_6061 from standard tables",
    "15.0 min per setup for workholding and alignment",
    "2.0 min per tool change"
  ],

  "metadata_file":       "/abs/path/to/metadata.json",
  "process_plan_file":   "/abs/path/to/process_plan.json",
  "time_estimate_file":  "/abs/path/to/time_estimate.json",
  "warnings":            []
}
```

---

## Constants

### MRR Table

Material Removal Rate in mm³/min per operation type per material.
These are conservative handbook values for a typical workshop VMC.

```python
MRR_TABLE = {
    "aluminium_6061": {
        "face_mill_rough":       8000,
        "face_mill_finish":      2000,
        "endmill_rough":         5000,
        "endmill_finish":         800,
        "shoulder_mill_rough":   4000,
        "shoulder_mill_finish":   600,
        "centre_drill":           500,
        "drill":                 3000,
        "drill_peck":            2000,
        "chamfer_mill":           300,
        "ball_endmill_finish":    400,
    },
    "mild_steel": {
        "face_mill_rough":       3000,
        "face_mill_finish":       800,
        "endmill_rough":         1500,
        "endmill_finish":         300,
        "shoulder_mill_rough":   1200,
        "shoulder_mill_finish":   200,
        "centre_drill":           200,
        "drill":                 1200,
        "drill_peck":             800,
        "chamfer_mill":           100,
        "ball_endmill_finish":    150,
    },
    "stainless_316": {
        "face_mill_rough":       1500,
        "face_mill_finish":       400,
        "endmill_rough":          800,
        "endmill_finish":         150,
        "shoulder_mill_rough":    600,
        "shoulder_mill_finish":   100,
        "centre_drill":           150,
        "drill":                  600,
        "drill_peck":             400,
        "chamfer_mill":            80,
        "ball_endmill_finish":    100,
    },
    "titanium_grade5": {
        "face_mill_rough":        500,
        "face_mill_finish":       150,
        "endmill_rough":          300,
        "endmill_finish":          60,
        "shoulder_mill_rough":    250,
        "shoulder_mill_finish":    50,
        "centre_drill":            80,
        "drill":                  250,
        "drill_peck":             150,
        "chamfer_mill":            40,
        "ball_endmill_finish":     50,
    },
}

DEFAULT_MATERIAL = "aluminium_6061"
TOOL_CHANGE_TIME_MIN  = 2.0    # minutes per unique tool
SETUP_TIME_MIN        = 15.0   # minutes per setup (fixturing + alignment)
DEFAULT_MRR_FALLBACK  = 1000   # mm³/min for unknown operation types
```

### Feature Volume Weight

Fraction of total stock removal attributed to each roughing feature type.
Weights are normalised across detected roughing features at runtime.

```python
FEATURE_VOLUME_WEIGHT = {
    "flat_face":          0.05,
    "rectangular_step":   0.30,
    "boss":               0.25,
    "rectangular_pocket": 0.20,
    "circular_pocket":    0.12,
    "triangular_pocket":  0.15,
    "rectangular_slot":   0.18,
    "circular_slot":      0.10,
    "blind_hole":         0.03,
    "through_hole":       0.03,
    "chamfer":            0.01,
    "fillet":             0.01,
}
```

### Feature Surface Fraction

Fraction of total part surface area attributed to each finishing operation.
Used to estimate finishing pass volume as surface_area × 0.5mm depth.

```python
FEATURE_SURFACE_FRACTION = {
    "flat_face":          0.30,
    "rectangular_step":   0.20,
    "boss":               0.15,
    "rectangular_pocket": 0.20,
    "circular_pocket":    0.12,
    "triangular_pocket":  0.15,
    "rectangular_slot":   0.15,
    "circular_slot":      0.10,
    "blind_hole":         0.05,
    "through_hole":       0.05,
    "chamfer":            0.05,
    "fillet":             0.05,
}

FINISHING_DOC_MM = 0.5   # assumed finishing depth of cut in mm
```

---

## Core Functions — `phase5_time_estimate.py`

### `estimate_removal_volumes`

```python
def estimate_removal_volumes(
    operations: list[dict],
    metadata: dict,
) -> dict[int, float]:
    """
    Estimate material removal volume (mm³) for each operation step.

    Strategy
    --------
    total_removal = raw_stock_volume - part_volume

    raw_stock_volume = raw_stock_mm.x * raw_stock_mm.y * raw_stock_mm.z
    part_volume      = metadata["volume_mm3"]

    For roughing operations:
        Distribute total_removal proportionally by FEATURE_VOLUME_WEIGHT.
        Normalise weights across only the roughing features present.

    For finishing operations:
        volume = surface_area_mm2 * FEATURE_SURFACE_FRACTION[feature] * FINISHING_DOC_MM
        surface_area taken from metadata["surface_area_mm2"].

    Parameters
    ----------
    operations : list of operation dicts from process_plan.json
    metadata   : dict from metadata.json

    Returns
    -------
    dict mapping step (int) → estimated_removal_volume_mm3 (float)
    Volume is always >= 0. Unknown feature types get a small fallback volume.
    """
```

### `estimate_operation_time`

```python
def estimate_operation_time(
    operation: dict,
    removal_volume_mm3: float,
    material: str,
) -> dict:
    """
    Estimate time for a single operation.

    Parameters
    ----------
    operation          : single operation dict from process_plan.json
    removal_volume_mm3 : estimated removal volume for this step
    material           : material string key for MRR_TABLE lookup

    Returns
    -------
    dict with keys:
        mrr_mm3_per_min        : float
        machining_time_min     : float  (removal_volume / MRR)
        tool_change_time_min   : float  (TOOL_CHANGE_TIME_MIN if tool changes)
        operation_time_min     : float  (machining + tool_change)

    Notes
    -----
    Tool change time is added only when the tool_type differs from the
    previous operation's tool_type. Pass the previous tool_type as context
    — or handle this at the caller level.

    Minimum machining_time_min is 0.1 (ten seconds) even for zero volume.
    """
```

### Public Interface

```python
def estimate_time(
    process_plan_path: str,
    metadata_path: str,
    output_dir: str,
    material: str = DEFAULT_MATERIAL,
    setup_time_min: float = SETUP_TIME_MIN,
    tool_change_time_min: float = TOOL_CHANGE_TIME_MIN,
) -> dict:
    """
    Estimate total machining time from a process plan.

    Parameters
    ----------
    process_plan_path  : path to process_plan.json from Phase 4
    metadata_path      : path to metadata.json from Phase 1
    output_dir         : directory to write time_estimate.json
    material           : material key (default "aluminium_6061")
    setup_time_min     : minutes per setup (default 15.0)
    tool_change_time_min : minutes per tool change (default 2.0)

    Returns
    -------
    dict matching time_estimate.json schema.

    Raises
    ------
    FileNotFoundError : either input path does not exist
    ValueError        : material not in MRR_TABLE

    Output
    ------
    {output_dir}/time_estimate.json  written atomically
    """
```

### Full flow

```
estimate_time(...)
│
├─ 1. Load process_plan.json and metadata.json
├─ 2. Validate material is in MRR_TABLE → ValueError if not
│      Add warning if material not found, fall back to DEFAULT_MATERIAL
├─ 3. Compute raw_stock_volume from metadata raw_stock_mm
├─ 4. Compute total_removal = raw_stock_volume - volume_mm3
│      Clamp to >= 0 (part may be close to stock size)
├─ 5. estimate_removal_volumes(operations, metadata)
│      → volumes dict {step: mm3}
├─ 6. For each operation in order:
│       previous_tool = tool from step N-1 (None for step 1)
│       tool_changed = (op["tool_type"] != previous_tool)
│       time_info = estimate_operation_time(op, volumes[step], material)
│       If not tool_changed: set tool_change_time_min = 0.0
│       Append breakdown entry
├─ 7. machining_time_min = sum of all machining_time_min values
│      tool_change_time_total = sum of tool_change_time_min values
│      setup_time_min = setup_count × setup_time_min
│      total_time_min = machining_time_min + tool_change_time_total + setup_time_min
├─ 8. Count unique tool changes
├─ 9. Assemble result dict
├─ 10. Write time_estimate.json atomically
└─ 11. Return result
```

### CLI

```bash
python phase5_time_estimate.py \
    data/processed/simple_block_cli/process_plan.json  \
    data/processed/simple_block_cli/metadata.json      \
    data/processed/simple_block_cli/                   \
    --material aluminium_6061
```

---
---

# PHASE 6 — Factory Profile, Cost & Job Quotation

---

## Factory Profile JSON Schema

Create two example profiles in `factory_profiles/`.

### `factory_profiles/nash_nz.json`

```json
{
  "factory_name": "NASH NZ",
  "currency": "NZD",

  "machines": [
    {
      "id": "VMC_01",
      "type": "VMC",
      "axes": 3,
      "work_envelope_mm": {"x": 500, "y": 400, "z": 300},
      "max_spindle_rpm": 8000,
      "achievable_Ra_um": 1.6,
      "hourly_rate": 120.0
    },
    {
      "id": "VMC_02",
      "type": "VMC",
      "axes": 4,
      "work_envelope_mm": {"x": 600, "y": 500, "z": 400},
      "max_spindle_rpm": 10000,
      "achievable_Ra_um": 0.8,
      "hourly_rate": 160.0
    },
    {
      "id": "LATHE_01",
      "type": "CNC_lathe",
      "axes": 2,
      "work_envelope_mm": {"x": 300, "y": 300, "z": 600},
      "max_spindle_rpm": 4000,
      "achievable_Ra_um": 0.8,
      "hourly_rate": 95.0
    }
  ],

  "materials_available": [
    "aluminium_6061",
    "mild_steel",
    "stainless_316"
  ],

  "weekly_capacity_hours": 40,
  "overhead_factor": 1.15,
  "notes": "Example NASH NZ factory profile for RPP MVP validation"
}
```

### `factory_profiles/generic_3axis.json`

```json
{
  "factory_name": "Generic 3-Axis Shop",
  "currency": "NZD",

  "machines": [
    {
      "id": "VMC_BASIC",
      "type": "VMC",
      "axes": 3,
      "work_envelope_mm": {"x": 400, "y": 300, "z": 250},
      "max_spindle_rpm": 6000,
      "achievable_Ra_um": 3.2,
      "hourly_rate": 90.0
    }
  ],

  "materials_available": [
    "aluminium_6061",
    "mild_steel"
  ],

  "weekly_capacity_hours": 40,
  "overhead_factor": 1.10,
  "notes": "Minimal 3-axis shop for test coverage of REJECT cases"
}
```

---

## Material Properties Table

```python
MATERIAL_PROPERTIES = {
    "aluminium_6061": {
        "density_g_per_mm3":  2.70e-3,
        "price_per_kg":       4.50,    # NZD/kg — update to local market
    },
    "mild_steel": {
        "density_g_per_mm3":  7.85e-3,
        "price_per_kg":       2.00,
    },
    "stainless_316": {
        "density_g_per_mm3":  8.00e-3,
        "price_per_kg":       8.50,
    },
    "titanium_grade5": {
        "density_g_per_mm3":  4.43e-3,
        "price_per_kg":       35.00,
    },
}
```

---

## Capability Checks

Implement each as a function returning `{"pass": bool, ...extra_fields}`.

### Check 1 — Axis Capability

```python
def check_axis_capability(
    required_axes: int,
    machines: list[dict],
) -> dict:
    """
    Pass if at least one machine has axes >= required_axes.
    Returns the best-fit machine id.

    {
      "pass": bool,
      "required": required_axes,
      "best_machine_id": str | None,
      "best_machine_axes": int | None,
      "reason": str    # human-readable, populated only on FAIL
    }
    """
```

### Check 2 — Work Envelope

```python
def check_work_envelope(
    bounding_box_mm: dict,   # {"x": float, "y": float, "z": float}
    machines: list[dict],
    required_axes: int,
) -> dict:
    """
    Pass if the part bounding box fits inside at least one machine
    that also meets the axis requirement.
    Check all three dimensions: part_dim <= envelope_dim for x, y, z.

    {
      "pass": bool,
      "part_mm": bounding_box_mm,
      "best_machine_id": str | None,
      "reason": str
    }
    """
```

### Check 3 — Material Availability

```python
def check_material_available(
    material: str,
    factory: dict,
) -> dict:
    """
    Pass if material is in factory["materials_available"].

    {
      "pass": bool,
      "material": material,
      "available": list[str],
      "reason": str
    }
    """
```

### Check 4 — Capacity

```python
def check_capacity(
    total_time_min: float,
    factory: dict,
) -> dict:
    """
    Pass if total_time_min <= weekly_capacity_hours * 60.

    {
      "pass": bool,
      "required_min": total_time_min,
      "available_min": float,
      "utilisation_fraction": float,
      "reason": str
    }
    """
```

---

## Cost Calculation

```python
def compute_cost(
    time_estimate: dict,
    metadata: dict,
    factory: dict,
    material: str,
    machine: dict,
) -> dict:
    """
    Compute machining and material cost.

    machining_cost = (total_time_min / 60) * machine["hourly_rate"]
    stock_volume   = raw_stock_mm.x * raw_stock_mm.y * raw_stock_mm.z  (mm³)
    material_mass_kg = stock_volume * density_g_per_mm3 / 1000
    material_cost  = material_mass_kg * price_per_kg
    subtotal       = machining_cost + material_cost
    total          = subtotal * factory["overhead_factor"]

    Returns
    -------
    {
      "machining": float,
      "material":  float,
      "subtotal":  float,
      "overhead_factor": float,
      "total":     float,
      "currency":  str
    }
    """
```

---

## Machine Selection

```python
def select_machine(
    factory: dict,
    required_axes: int,
    bounding_box_mm: dict,
) -> dict | None:
    """
    Select the cheapest machine that:
      (a) has axes >= required_axes
      (b) has work envelope >= bounding_box_mm on all three axes

    Returns the machine dict or None if no suitable machine found.
    """
```

---

## Output: `quotation.json`

```json
{
  "recommendation": "ACCEPT",
  "flags": [],

  "estimated_cost": {
    "machining":       175.40,
    "material":         45.20,
    "subtotal":        220.60,
    "overhead_factor":   1.15,
    "total":           253.69,
    "currency":        "NZD"
  },

  "time_summary": {
    "total_min":      87.3,
    "machining_min":  57.3,
    "setup_min":      30.0
  },

  "machine_selected": "VMC_01",

  "capability_checks": {
    "axis_capability":    {"pass": true,  "required": 3, "best_machine_id": "VMC_01", "best_machine_axes": 3},
    "work_envelope":      {"pass": true,  "part_mm": {"x": 100.0, "y": 60.0, "z": 40.0}, "best_machine_id": "VMC_01"},
    "material_available": {"pass": true,  "material": "aluminium_6061"},
    "capacity":           {"pass": true,  "required_min": 87.3, "available_min": 2400.0, "utilisation_fraction": 0.036}
  },

  "factory_name":  "NASH NZ",
  "material":      "aluminium_6061",
  "axis_required": 3,

  "source_files": {
    "process_plan":    "/abs/path",
    "time_estimate":   "/abs/path",
    "metadata":        "/abs/path",
    "factory_profile": "/abs/path"
  },

  "quotation_file": "/abs/path/to/quotation.json",
  "warnings":       []
}
```

`recommendation` is `"ACCEPT"` if all four capability checks pass,
`"REJECT"` otherwise.

`flags` contains one human-readable string per failed check, e.g.:
```json
"flags": [
  "Part requires 4-axis machining. No 4-axis machine available in this factory.",
  "Part bounding box (600 x 400 x 300 mm) exceeds all available work envelopes."
]
```

---

## Public Interface — `phase6_quotation.py`

```python
def generate_quotation(
    process_plan_path: str,
    time_estimate_path: str,
    metadata_path: str,
    factory_profile_path: str,
    output_dir: str,
    material: str = "aluminium_6061",
) -> dict:
    """
    Generate a job quotation with cost estimate and accept/reject recommendation.

    Parameters
    ----------
    process_plan_path    : path to process_plan.json from Phase 4
    time_estimate_path   : path to time_estimate.json from Phase 5
    metadata_path        : path to metadata.json from Phase 1
    factory_profile_path : path to factory profile JSON
    output_dir           : directory to write quotation.json
    material             : material key (default "aluminium_6061")

    Returns
    -------
    dict matching quotation.json schema.

    Raises
    ------
    FileNotFoundError : any input path does not exist
    ValueError        : factory profile missing required keys

    Output
    ------
    {output_dir}/quotation.json  written atomically
    """
```

### Full flow

```
generate_quotation(...)
│
├─ 1.  Load all four input JSON files → FileNotFoundError if any missing
├─ 2.  Validate factory profile has required keys:
│       "machines", "materials_available", "weekly_capacity_hours",
│       "overhead_factor", "currency"
├─ 3.  Read required_axes from process_plan["axis_requirement"]
├─ 4.  Read bounding_box_mm from metadata["bounding_box_mm"]
├─ 5.  Run four capability checks
├─ 6.  select_machine(factory, required_axes, bounding_box_mm)
│       If None → machine_selected = null, cost uses hourly_rate = 0,
│                 add warning
├─ 7.  compute_cost(time_estimate, metadata, factory, material, machine)
├─ 8.  recommendation = "ACCEPT" if all checks pass else "REJECT"
├─ 9.  flags = [check["reason"] for check in failed checks]
├─ 10. Assemble quotation dict
├─ 11. Write quotation.json atomically
└─ 12. Return dict
```

### CLI

```bash
python phase6_quotation.py \
    data/processed/simple_block_cli/process_plan.json   \
    data/processed/simple_block_cli/time_estimate.json  \
    data/processed/simple_block_cli/metadata.json       \
    factory_profiles/nash_nz.json                       \
    data/processed/simple_block_cli/                    \
    --material aluminium_6061
```

---
---

# TESTS — `tests/test_phase5.py`

```python
# tests/test_phase5.py

import os, json
import pytest
from phase5_time_estimate import (
    estimate_time,
    estimate_removal_volumes,
    estimate_operation_time,
    MRR_TABLE,
    FEATURE_VOLUME_WEIGHT,
    DEFAULT_MATERIAL,
    TOOL_CHANGE_TIME_MIN,
    SETUP_TIME_MIN,
)

CLI_DIR = "data/processed/simple_block_cli"

MINIMAL_METADATA = {
    "bounding_box_mm":   {"x": 100.0, "y": 60.0, "z": 40.0},
    "volume_mm3":        180000.0,
    "surface_area_mm2":  26800.0,
    "raw_stock_mm":      {"x": 115.0, "y": 70.0, "z": 45.0},
}

MINIMAL_PLAN = {
    "operations": [
        {"step": 1, "setup_id": 0, "approach_direction": "+Z",
         "feature_type": "flat_face", "operation_type": "face_mill_rough",
         "tool_type": "face_mill", "phase": "roughing", "notes": ""},
        {"step": 2, "setup_id": 0, "approach_direction": "+Z",
         "feature_type": "flat_face", "operation_type": "face_mill_finish",
         "tool_type": "face_mill", "phase": "finishing", "notes": ""},
        {"step": 3, "setup_id": 0, "approach_direction": "+Z",
         "feature_type": "rectangular_pocket", "operation_type": "endmill_rough",
         "tool_type": "flat_endmill", "phase": "roughing", "notes": ""},
    ],
    "setup_count": 1,
    "axis_requirement": 3,
}


@pytest.fixture
def plan_files(tmp_path):
    meta = tmp_path / "metadata.json"
    plan = tmp_path / "process_plan.json"
    meta.write_text(json.dumps(MINIMAL_METADATA))
    plan.write_text(json.dumps(MINIMAL_PLAN))
    return {"metadata": str(meta), "plan": str(plan), "out": str(tmp_path)}


# ── MRR_TABLE coverage ────────────────────────────────────────────────────────

def test_mrr_table_has_default_material():
    assert DEFAULT_MATERIAL in MRR_TABLE


def test_mrr_table_all_materials_have_same_keys():
    key_sets = [set(v.keys()) for v in MRR_TABLE.values()]
    assert all(ks == key_sets[0] for ks in key_sets), \
        "All materials must have the same operation type keys"


def test_feature_volume_weight_all_features():
    from models.feature_net import FEATURE_NAMES
    for name in FEATURE_NAMES:
        assert name in FEATURE_VOLUME_WEIGHT, \
            f"FEATURE_VOLUME_WEIGHT missing: {name}"


# ── estimate_removal_volumes ──────────────────────────────────────────────────

def test_removal_volumes_returns_per_step():
    vols = estimate_removal_volumes(MINIMAL_PLAN["operations"], MINIMAL_METADATA)
    for op in MINIMAL_PLAN["operations"]:
        assert op["step"] in vols
        assert vols[op["step"]] >= 0


def test_roughing_volumes_sum_to_total_removal():
    meta = MINIMAL_METADATA
    stock_vol = meta["raw_stock_mm"]["x"] * meta["raw_stock_mm"]["y"] * meta["raw_stock_mm"]["z"]
    total_removal = max(0.0, stock_vol - meta["volume_mm3"])
    vols = estimate_removal_volumes(MINIMAL_PLAN["operations"], meta)
    roughing_steps = [op["step"] for op in MINIMAL_PLAN["operations"]
                      if op["phase"] == "roughing"]
    roughing_total = sum(vols[s] for s in roughing_steps)
    assert abs(roughing_total - total_removal) < 1.0, \
        f"Roughing volumes {roughing_total:.1f} don't sum to total removal {total_removal:.1f}"


def test_finishing_volume_is_smaller_than_roughing():
    vols = estimate_removal_volumes(MINIMAL_PLAN["operations"], MINIMAL_METADATA)
    roughing = [vols[op["step"]] for op in MINIMAL_PLAN["operations"] if op["phase"] == "roughing"]
    finishing = [vols[op["step"]] for op in MINIMAL_PLAN["operations"] if op["phase"] == "finishing"]
    if roughing and finishing:
        assert max(finishing) < max(roughing)


# ── estimate_operation_time ───────────────────────────────────────────────────

def test_operation_time_returns_required_keys():
    op = MINIMAL_PLAN["operations"][0]
    result = estimate_operation_time(op, removal_volume_mm3=5000.0, material="aluminium_6061")
    for key in ["mrr_mm3_per_min", "machining_time_min",
                "tool_change_time_min", "operation_time_min"]:
        assert key in result


def test_operation_time_all_positive():
    op = MINIMAL_PLAN["operations"][0]
    result = estimate_operation_time(op, 5000.0, "aluminium_6061")
    assert result["machining_time_min"] >= 0.1
    assert result["operation_time_min"] >= result["machining_time_min"]


def test_zero_volume_gives_minimum_time():
    op = MINIMAL_PLAN["operations"][0]
    result = estimate_operation_time(op, 0.0, "aluminium_6061")
    assert result["machining_time_min"] >= 0.1


# ── estimate_time (full pipeline) ─────────────────────────────────────────────

def test_output_file_created(plan_files):
    estimate_time(plan_files["plan"], plan_files["metadata"], plan_files["out"])
    assert os.path.exists(os.path.join(plan_files["out"], "time_estimate.json"))


def test_schema_keys_present(plan_files):
    result = estimate_time(
        plan_files["plan"], plan_files["metadata"], plan_files["out"])
    for key in ["total_time_min", "machining_time_min", "setup_time_min",
                "breakdown", "material", "setup_count", "tool_change_count",
                "assumptions", "time_estimate_file", "warnings"]:
        assert key in result


def test_total_time_equals_components(plan_files):
    result = estimate_time(
        plan_files["plan"], plan_files["metadata"], plan_files["out"])
    computed = (result["machining_time_min"]
                + result["setup_time_min"]
                + result["tool_change_count"] * TOOL_CHANGE_TIME_MIN)
    assert abs(computed - result["total_time_min"]) < 0.1


def test_setup_time_equals_count_times_rate(plan_files):
    result = estimate_time(
        plan_files["plan"], plan_files["metadata"], plan_files["out"],
        setup_time_min=15.0)
    expected = MINIMAL_PLAN["setup_count"] * 15.0
    assert abs(result["setup_time_min"] - expected) < 0.01


def test_breakdown_count_matches_operations(plan_files):
    result = estimate_time(
        plan_files["plan"], plan_files["metadata"], plan_files["out"])
    assert len(result["breakdown"]) == len(MINIMAL_PLAN["operations"])


def test_invalid_material_raises(plan_files):
    with pytest.raises(ValueError):
        estimate_time(
            plan_files["plan"], plan_files["metadata"], plan_files["out"],
            material="unobtainium_99")


def test_file_not_found_plan(plan_files):
    with pytest.raises(FileNotFoundError):
        estimate_time("no_plan.json", plan_files["metadata"], plan_files["out"])


def test_file_not_found_metadata(plan_files):
    with pytest.raises(FileNotFoundError):
        estimate_time(plan_files["plan"], "no_meta.json", plan_files["out"])


def test_output_path_absolute(plan_files):
    result = estimate_time(
        plan_files["plan"], plan_files["metadata"], plan_files["out"])
    assert os.path.isabs(result["time_estimate_file"])


@pytest.mark.skipif(
    not all(os.path.exists(os.path.join(CLI_DIR, f))
            for f in ["process_plan.json", "metadata.json"]),
    reason="Phase 4 CLI output not available"
)
def test_real_pipeline_time_positive(tmp_path):
    result = estimate_time(
        os.path.join(CLI_DIR, "process_plan.json"),
        os.path.join(CLI_DIR, "metadata.json"),
        str(tmp_path),
    )
    assert result["total_time_min"] > 0
    assert result["machining_time_min"] > 0
```

---

# TESTS — `tests/test_phase6.py`

```python
# tests/test_phase6.py

import os, json, copy
import pytest
from phase6_quotation import (
    generate_quotation,
    check_axis_capability,
    check_work_envelope,
    check_material_available,
    check_capacity,
    compute_cost,
    select_machine,
    MATERIAL_PROPERTIES,
)

CLI_DIR       = "data/processed/simple_block_cli"
NASH_PROFILE  = "factory_profiles/nash_nz.json"
BASIC_PROFILE = "factory_profiles/generic_3axis.json"

MINIMAL_FACTORY = {
    "factory_name": "Test Shop",
    "currency": "NZD",
    "machines": [
        {"id": "VMC_01", "type": "VMC", "axes": 3,
         "work_envelope_mm": {"x": 500, "y": 400, "z": 300},
         "achievable_Ra_um": 1.6, "hourly_rate": 120.0}
    ],
    "materials_available": ["aluminium_6061", "mild_steel"],
    "weekly_capacity_hours": 40,
    "overhead_factor": 1.10,
}

MINIMAL_TIME = {
    "total_time_min": 45.0,
    "machining_time_min": 30.0,
    "setup_time_min": 15.0,
    "setup_count": 1,
}

MINIMAL_META = {
    "bounding_box_mm": {"x": 100.0, "y": 60.0, "z": 40.0},
    "volume_mm3": 180000.0,
    "raw_stock_mm": {"x": 115.0, "y": 70.0, "z": 45.0},
}

MINIMAL_PLAN = {
    "axis_requirement": 3,
    "setup_count": 1,
    "operation_count": 4,
}


@pytest.fixture
def input_files(tmp_path):
    factory_path = tmp_path / "factory.json"
    time_path    = tmp_path / "time_estimate.json"
    meta_path    = tmp_path / "metadata.json"
    plan_path    = tmp_path / "process_plan.json"
    factory_path.write_text(json.dumps(MINIMAL_FACTORY))
    time_path.write_text(json.dumps(MINIMAL_TIME))
    meta_path.write_text(json.dumps(MINIMAL_META))
    plan_path.write_text(json.dumps(MINIMAL_PLAN))
    return {
        "factory": str(factory_path),
        "time":    str(time_path),
        "meta":    str(meta_path),
        "plan":    str(plan_path),
        "out":     str(tmp_path),
    }


# ── MATERIAL_PROPERTIES ───────────────────────────────────────────────────────

def test_material_properties_all_have_density():
    for mat, props in MATERIAL_PROPERTIES.items():
        assert "density_g_per_mm3" in props, f"{mat} missing density"
        assert props["density_g_per_mm3"] > 0


def test_material_properties_all_have_price():
    for mat, props in MATERIAL_PROPERTIES.items():
        assert "price_per_kg" in props, f"{mat} missing price"
        assert props["price_per_kg"] > 0


# ── Capability checks ─────────────────────────────────────────────────────────

def test_axis_check_pass():
    result = check_axis_capability(3, MINIMAL_FACTORY["machines"])
    assert result["pass"] is True
    assert result["required"] == 3


def test_axis_check_fail():
    result = check_axis_capability(5, MINIMAL_FACTORY["machines"])
    assert result["pass"] is False
    assert "reason" in result


def test_envelope_check_pass():
    bbox = {"x": 100.0, "y": 60.0, "z": 40.0}
    result = check_work_envelope(bbox, MINIMAL_FACTORY["machines"], required_axes=3)
    assert result["pass"] is True


def test_envelope_check_fail_oversized():
    bbox = {"x": 600.0, "y": 500.0, "z": 400.0}   # bigger than VMC_01
    result = check_work_envelope(bbox, MINIMAL_FACTORY["machines"], required_axes=3)
    assert result["pass"] is False


def test_material_check_pass():
    result = check_material_available("aluminium_6061", MINIMAL_FACTORY)
    assert result["pass"] is True


def test_material_check_fail():
    result = check_material_available("titanium_grade5", MINIMAL_FACTORY)
    assert result["pass"] is False
    assert "reason" in result


def test_capacity_check_pass():
    result = check_capacity(45.0, MINIMAL_FACTORY)    # 45 min << 2400 min/week
    assert result["pass"] is True
    assert result["utilisation_fraction"] < 1.0


def test_capacity_check_fail():
    result = check_capacity(3000.0, MINIMAL_FACTORY)  # 50 hrs > 40 hrs capacity
    assert result["pass"] is False


# ── select_machine ────────────────────────────────────────────────────────────

def test_select_machine_returns_cheapest_valid():
    factory = copy.deepcopy(MINIMAL_FACTORY)
    factory["machines"].append({
        "id": "VMC_PREMIUM", "type": "VMC", "axes": 3,
        "work_envelope_mm": {"x": 600, "y": 500, "z": 400},
        "achievable_Ra_um": 0.4, "hourly_rate": 200.0
    })
    machine = select_machine(factory, required_axes=3,
                             bounding_box_mm={"x": 100, "y": 60, "z": 40})
    assert machine is not None
    assert machine["hourly_rate"] == 120.0   # VMC_01 is cheaper


def test_select_machine_returns_none_when_no_fit():
    machine = select_machine(MINIMAL_FACTORY, required_axes=5,
                             bounding_box_mm={"x": 100, "y": 60, "z": 40})
    assert machine is None


# ── compute_cost ──────────────────────────────────────────────────────────────

def test_compute_cost_total_positive():
    machine = MINIMAL_FACTORY["machines"][0]
    result = compute_cost(MINIMAL_TIME, MINIMAL_META,
                          MINIMAL_FACTORY, "aluminium_6061", machine)
    assert result["total"] > 0


def test_compute_cost_total_equals_subtotal_times_overhead():
    machine = MINIMAL_FACTORY["machines"][0]
    result = compute_cost(MINIMAL_TIME, MINIMAL_META,
                          MINIMAL_FACTORY, "aluminium_6061", machine)
    expected = result["subtotal"] * MINIMAL_FACTORY["overhead_factor"]
    assert abs(result["total"] - expected) < 0.01


def test_compute_cost_currency_matches_factory():
    machine = MINIMAL_FACTORY["machines"][0]
    result = compute_cost(MINIMAL_TIME, MINIMAL_META,
                          MINIMAL_FACTORY, "aluminium_6061", machine)
    assert result["currency"] == MINIMAL_FACTORY["currency"]


# ── generate_quotation ────────────────────────────────────────────────────────

def test_quotation_file_created(input_files):
    generate_quotation(
        input_files["plan"], input_files["time"],
        input_files["meta"], input_files["factory"],
        input_files["out"])
    assert os.path.exists(os.path.join(input_files["out"], "quotation.json"))


def test_quotation_schema_keys(input_files):
    result = generate_quotation(
        input_files["plan"], input_files["time"],
        input_files["meta"], input_files["factory"],
        input_files["out"])
    for key in ["recommendation", "flags", "estimated_cost", "time_summary",
                "machine_selected", "capability_checks", "factory_name",
                "material", "axis_required", "source_files",
                "quotation_file", "warnings"]:
        assert key in result, f"Missing key: {key}"


def test_recommendation_accept_on_capable_factory(input_files):
    result = generate_quotation(
        input_files["plan"], input_files["time"],
        input_files["meta"], input_files["factory"],
        input_files["out"])
    assert result["recommendation"] == "ACCEPT"
    assert result["flags"] == []


def test_recommendation_reject_axis_mismatch(input_files, tmp_path):
    plan_5axis = copy.deepcopy(MINIMAL_PLAN)
    plan_5axis["axis_requirement"] = 5
    plan_path = tmp_path / "plan5.json"
    plan_path.write_text(json.dumps(plan_5axis))
    result = generate_quotation(
        str(plan_path), input_files["time"],
        input_files["meta"], input_files["factory"],
        input_files["out"])
    assert result["recommendation"] == "REJECT"
    assert len(result["flags"]) >= 1


def test_recommendation_reject_wrong_material(input_files, tmp_path):
    result = generate_quotation(
        input_files["plan"], input_files["time"],
        input_files["meta"], input_files["factory"],
        input_files["out"],
        material="titanium_grade5")
    assert result["recommendation"] == "REJECT"


def test_all_four_capability_checks_present(input_files):
    result = generate_quotation(
        input_files["plan"], input_files["time"],
        input_files["meta"], input_files["factory"],
        input_files["out"])
    for check in ["axis_capability", "work_envelope",
                  "material_available", "capacity"]:
        assert check in result["capability_checks"]


def test_quotation_path_absolute(input_files):
    result = generate_quotation(
        input_files["plan"], input_files["time"],
        input_files["meta"], input_files["factory"],
        input_files["out"])
    assert os.path.isabs(result["quotation_file"])


def test_cost_total_positive(input_files):
    result = generate_quotation(
        input_files["plan"], input_files["time"],
        input_files["meta"], input_files["factory"],
        input_files["out"])
    assert result["estimated_cost"]["total"] > 0


def test_written_json_matches_returned(input_files):
    result = generate_quotation(
        input_files["plan"], input_files["time"],
        input_files["meta"], input_files["factory"],
        input_files["out"])
    with open(result["quotation_file"]) as f:
        on_disk = json.load(f)
    assert on_disk["recommendation"] == result["recommendation"]


def test_missing_process_plan_raises(input_files):
    with pytest.raises(FileNotFoundError):
        generate_quotation("no_plan.json", input_files["time"],
                           input_files["meta"], input_files["factory"],
                           input_files["out"])


def test_missing_factory_profile_raises(input_files):
    with pytest.raises(FileNotFoundError):
        generate_quotation(input_files["plan"], input_files["time"],
                           input_files["meta"], "no_factory.json",
                           input_files["out"])


# ── Factory profile files exist ───────────────────────────────────────────────

def test_nash_nz_profile_exists():
    assert os.path.exists(NASH_PROFILE), \
        f"factory_profiles/nash_nz.json not found"


def test_generic_3axis_profile_exists():
    assert os.path.exists(BASIC_PROFILE), \
        f"factory_profiles/generic_3axis.json not found"


def test_nash_nz_profile_valid_json():
    with open(NASH_PROFILE) as f:
        data = json.load(f)
    assert "machines" in data
    assert len(data["machines"]) >= 1


# ── End-to-end full pipeline test ─────────────────────────────────────────────

@pytest.mark.skipif(
    not all(os.path.exists(os.path.join(CLI_DIR, f))
            for f in ["process_plan.json", "metadata.json"]) or
    not os.path.exists(NASH_PROFILE),
    reason="Full pipeline CLI outputs or factory profile not available"
)
def test_full_pipeline_simple_block(tmp_path):
    from phase5_time_estimate import estimate_time
    time_result = estimate_time(
        os.path.join(CLI_DIR, "process_plan.json"),
        os.path.join(CLI_DIR, "metadata.json"),
        str(tmp_path),
    )
    time_path = os.path.join(str(tmp_path), "time_estimate.json")

    result = generate_quotation(
        os.path.join(CLI_DIR, "process_plan.json"),
        time_path,
        os.path.join(CLI_DIR, "metadata.json"),
        NASH_PROFILE,
        str(tmp_path),
        material="aluminium_6061",
    )
    assert result["recommendation"] in ("ACCEPT", "REJECT")
    assert result["estimated_cost"]["total"] > 0
    print(f"\nFull pipeline result: {result['recommendation']}")
    print(f"  Total cost: {result['estimated_cost']['currency']} "
          f"{result['estimated_cost']['total']:.2f}")
    print(f"  Total time: {result['time_summary']['total_min']:.1f} min")
```

---
---

# QUICK-START COMMANDS

```bash
# 1. Estimate machining time
python phase5_time_estimate.py \
    data/processed/simple_block_cli/process_plan.json  \
    data/processed/simple_block_cli/metadata.json      \
    data/processed/simple_block_cli/

# 2. Generate quotation
python phase6_quotation.py \
    data/processed/simple_block_cli/process_plan.json   \
    data/processed/simple_block_cli/time_estimate.json  \
    data/processed/simple_block_cli/metadata.json       \
    factory_profiles/nash_nz.json                       \
    data/processed/simple_block_cli/

# 3. Run Phase 5 + 6 tests
pytest tests/test_phase5.py tests/test_phase6.py -v

# 4. Run full suite
pytest tests/ -v
```

---
---

# ACCEPTANCE CRITERIA

## Phase 5

- [ ] `pytest tests/test_phase5.py -v` — all tests pass
- [ ] `total_time_min == machining_time_min + setup_time_min + tool_change total`
- [ ] `len(breakdown) == len(operations)` in process_plan.json
- [ ] `roughing_volumes.sum() == total_removal_volume` (within 1 mm³)
- [ ] CLI produces `time_estimate.json` with `total_time_min > 0`
- [ ] Invalid material raises `ValueError`
- [ ] `FEATURE_VOLUME_WEIGHT` covers all 12 Phase 2 feature classes

## Phase 6

- [ ] `pytest tests/test_phase6.py -v` — all tests pass
- [ ] `factory_profiles/nash_nz.json` and `generic_3axis.json` exist and are valid
- [ ] `recommendation == "ACCEPT"` for simple_block + nash_nz + aluminium
- [ ] `recommendation == "REJECT"` for 5-axis required + 3-axis-only factory
- [ ] `recommendation == "REJECT"` for unavailable material
- [ ] `estimated_cost.total > 0` always
- [ ] `total == subtotal * overhead_factor` within 0.01
- [ ] All four capability_checks keys present in output
- [ ] `pytest tests/ -v` — all phases pass (≥170 total)

---
---

# NOTES FOR CODEX

1. **`estimate_removal_volumes` must normalise weights.** The roughing
   volume weights for detected features must sum to exactly `total_removal`.
   Collect all roughing operations, sum their weights, then scale each
   weight so the weighted volumes sum to `total_removal`.

2. **Tool change is tracked by transition, not by count.** A tool change
   occurs when `op[n]["tool_type"] != op[n-1]["tool_type"]`. First
   operation always has a tool change (from "no tool" to first tool).

3. **`check_axis_capability` and `check_work_envelope` both importable.**
   Tests import them directly. Do not make them inner functions of
   `generate_quotation`.

4. **`select_machine` returns cheapest valid machine** (lowest hourly_rate
   among all machines meeting axis + envelope requirements). If a tie,
   either is acceptable.

5. **`compute_cost` requires a machine dict, not a machine id.** Pass the
   result of `select_machine` directly. If `select_machine` returns None,
   set machining_cost = 0 and add a warning.

6. **Material cost uses raw_stock volume, not part volume.** The customer
   pays for the stock block, not just the finished part. Use
   `raw_stock_mm.x * raw_stock_mm.y * raw_stock_mm.z` for material cost.

7. **All four checks must run regardless of earlier failures.** Do not
   short-circuit after the first failed check. The `flags` list must
   contain all failure reasons, not just the first.

8. **Factory profile JSON must be validated.** Check for required keys
   before proceeding. Raise `ValueError` with a clear message if any
   required key is missing.

9. **Atomic writes for all output files** using temp file + `os.replace`.

10. **`MATERIAL_PROPERTIES` must include all materials in both factory
    profile files.** If a material appears in a factory profile but not
    in `MATERIAL_PROPERTIES`, add a warning and use a fallback density
    (7.85e-3 g/mm³, steel default).
