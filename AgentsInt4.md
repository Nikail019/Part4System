# AGENTS.md — STEP AP242 PMI Extraction + Dimensionally-Aware Phase 4

## Goal

Extract non-tolerancing PMI from STEP AP242 files and use it to make
Phase 4 dimensionally-aware. After this phase:

- Hole diameters and depths are known → correct drill sizes selected
- Surface finish (Ra) is known → finishing passes added where Ra < 1.6
- Thread specs are known → tapping operations added
- Material is known → cutting parameters matched to material
- Pocket/step dimensions known → pass count calculated

Tolerancing (GD&T, IT grades) is explicitly deferred to a later phase.

---

## What Gets Added

```
rpp-mvp/
├── step_pmi_extractor.py          # NEW — extracts PMI from STEP AP242
├── phase4_process_plan.py         # MODIFY — consume pmi_data.json
├── run_pipeline.py                # MODIFY — add PMI extraction step
└── tests/
    ├── test_pmi_extractor.py      # NEW
    └── test_phase4.py             # MODIFY — PMI-aware plan tests
```

---

## Two-Layer PMI Extraction Strategy

STEP AP242 files vary widely in PMI richness. Some have full annotations;
most from early-stage design have none. The extractor uses two layers:

**Layer 1 — STEP entity parsing (explicit PMI)**
Reads PMI annotation entities directly from the STEP file text.
Extracts: surface finish Ra, thread designations, material name.

**Layer 2 — B-rep geometry measurement (implicit dimensions)**
Uses cadquery to measure the actual B-rep faces.
Extracts: hole diameters, hole depths, pocket dimensions, step dimensions.
This works on ANY STEP file regardless of PMI annotation.

Layer 1 is attempted first. Layer 2 fills gaps. Together they produce
a complete `pmi_data.json` for every STEP file.

---

## Output: `pmi_data.json`

```json
{
  "source_file":   "/abs/path/to/part.stp",
  "material":      "aluminium_6061",
  "material_source": "pmi",

  "features": [
    {
      "type":       "through_hole",
      "instance_id": 0,
      "diameter_mm": 10.0,
      "depth_mm":    40.0,
      "depth_ratio": 4.0,
      "Ra_um":       1.6,
      "Ra_source":   "pmi",
      "threaded":    false,
      "thread_spec": null,
      "peck_required": true
    },
    {
      "type":       "rectangular_pocket",
      "instance_id": 0,
      "width_mm":   30.0,
      "length_mm":  50.0,
      "depth_mm":   15.0,
      "Ra_um":      3.2,
      "Ra_source":  "geometry",
      "threaded":   false,
      "thread_spec": null,
      "rough_passes": 3
    },
    {
      "type":       "flat_face",
      "instance_id": 0,
      "Ra_um":      3.2,
      "Ra_source":  "pmi"
    }
  ],

  "pmi_data_file": "/abs/path/to/pmi_data.json",
  "warnings":      []
}
```

### Field definitions

- `material_source`: `"pmi"` if read from STEP entity, `"default"` if fallback
- `Ra_source`: `"pmi"` if from annotation, `"geometry"` if defaulted from feature type
- `depth_ratio`: depth / diameter — used to decide peck drilling (> 3.0 → peck)
- `rough_passes`: estimated from depth / standard axial depth of cut
- `peck_required`: True if depth_ratio > 3.0

---

## Part 1 — STEP Entity Parser

### Constants

```python
# Default Ra values by feature type when no PMI annotation found
DEFAULT_RA = {
    "flat_face":           3.2,
    "through_hole":        1.6,
    "blind_hole":          1.6,
    "rectangular_pocket":  3.2,
    "circular_pocket":     3.2,
    "rectangular_slot":    3.2,
    "circular_slot":       3.2,
    "rectangular_step":    3.2,
    "chamfer":             6.3,
    "fillet":              3.2,
    "boss":                3.2,
    "triangular_pocket":   3.2,
}

# Standard axial depth of cut per material for rough pass counting
ROUGH_DOC_MM = {
    "aluminium_6061":  5.0,
    "mild_steel":      2.0,
    "stainless_316":   1.5,
    "titanium_grade5": 0.8,
}

PECK_DEPTH_RATIO = 3.0   # peck drill if depth/diameter > this

# Material name normalisation — map common STEP material strings
# to our internal keys
MATERIAL_NAME_MAP = {
    "6061":             "aluminium_6061",
    "6061-t6":          "aluminium_6061",
    "aluminium":        "aluminium_6061",
    "aluminum":         "aluminium_6061",
    "al":               "aluminium_6061",
    "mild steel":       "mild_steel",
    "ms":               "mild_steel",
    "1018":             "mild_steel",
    "316":              "stainless_316",
    "316l":             "stainless_316",
    "stainless":        "stainless_316",
    "ss316":            "stainless_316",
    "titanium":         "titanium_grade5",
    "ti-6al-4v":        "titanium_grade5",
    "grade 5":          "titanium_grade5",
}
```

### Layer 1 — STEP Text Parser

```python
def parse_step_pmi(step_path: str) -> dict:
    """
    Parse PMI annotations from STEP AP242 file text.

    Searches for these entity types:
      SURFACE_TEXTURE_PARAMETER  → Ra value
      EXTERNALLY_DEFINED_FEATURE_DEFINITION → thread specs
      MATERIAL_DESIGNATION / MATERIAL  → material name

    Returns dict with keys:
      "material":       str | None
      "surface_finish": list of {"Ra_um": float, "face_ref": str}
      "threads":        list of {"spec": str, "face_ref": str}

    All values are None or empty list if not found.
    This function never raises — returns empty dict on any error.
    """
    result = {
        "material":       None,
        "surface_finish": [],
        "threads":        [],
    }

    try:
        with open(step_path, "r", errors="ignore") as f:
            content = f.read().upper()
    except Exception:
        return result

    # Material detection
    # Matches: MATERIAL_DESIGNATION('6061-T6',...) or MATERIAL('6061',...))
    import re
    mat_patterns = [
        r"MATERIAL_DESIGNATION\s*\(\s*'([^']+)'",
        r"MATERIAL\s*\(\s*'([^']+)'",
        r"PRODUCT_DEFINITION_FORMATION\s*\(\s*'([^']+)'",
    ]
    for pat in mat_patterns:
        m = re.search(pat, content)
        if m:
            raw = m.group(1).strip().lower()
            for key, normalised in MATERIAL_NAME_MAP.items():
                if key in raw:
                    result["material"] = normalised
                    break
        if result["material"]:
            break

    # Surface finish Ra detection
    # Matches: SURFACE_TEXTURE_PARAMETER(...,'RA',...,MEASURE_VALUE(1.6),...)
    # or simpler: SURFACE_TEXTURE_PARAMETER with a numeric value
    ra_pattern = r"SURFACE_TEXTURE_PARAMETER\s*\([^)]*?(\d+\.?\d*)[^)]*\)"
    for m in re.finditer(ra_pattern, content):
        try:
            val = float(m.group(1))
            if 0.05 <= val <= 50.0:   # sanity range for Ra in microns
                result["surface_finish"].append({
                    "Ra_um":    val,
                    "face_ref": m.group(0)[:40],
                })
        except ValueError:
            pass

    # Thread detection
    # Matches: EXTERNALLY_DEFINED_FEATURE_DEFINITION('M10X1.5-6H',...)
    thread_pattern = (
        r"EXTERNALLY_DEFINED_FEATURE_DEFINITION\s*\(\s*'([^']*(?:M\d|UNC|UNF|NPT|G\d)[^']*)'"
    )
    for m in re.finditer(thread_pattern, content):
        result["threads"].append({
            "spec":     m.group(1).strip(),
            "face_ref": m.group(0)[:40],
        })

    return result
```

### Layer 2 — B-rep Geometry Measurement

```python
def measure_brep_features(step_path: str) -> dict:
    """
    Measure B-rep face geometry to extract feature dimensions.

    Uses cadquery to load the STEP file and inspect face types:
      CYLINDER → hole or boss (radius gives diameter)
      PLANE    → flat features (area, normal, bounding box)

    Returns:
    {
      "holes": [{"diameter_mm": float, "depth_mm": float}, ...],
      "planar_recesses": [{"width_mm": float, "length_mm": float,
                           "depth_mm": float}, ...],
      "bounding_box_mm": {"x": float, "y": float, "z": float},
    }
    """
    import cadquery as cq

    result = {
        "holes":            [],
        "planar_recesses":  [],
        "bounding_box_mm":  {"x": 0.0, "y": 0.0, "z": 0.0},
    }

    try:
        shape  = cq.importers.importStep(step_path).val()
        bb     = shape.BoundingBox()
        result["bounding_box_mm"] = {
            "x": round(bb.xmax - bb.xmin, 3),
            "y": round(bb.ymax - bb.ymin, 3),
            "z": round(bb.zmax - bb.zmin, 3),
        }

        for face in shape.Faces():
            geo = face.geomType()

            if geo == "CYLINDER":
                # Cylindrical face → hole or boss
                # Diameter estimated from face bounding box
                fbb     = face.BoundingBox()
                x_span  = fbb.xmax - fbb.xmin
                y_span  = fbb.ymax - fbb.ymin
                diameter = round((x_span + y_span) / 2.0, 2)
                depth    = round(fbb.zmax - fbb.zmin, 2)

                if diameter > 0.5 and depth > 0.5:
                    result["holes"].append({
                        "diameter_mm": diameter,
                        "depth_mm":    depth,
                    })

            elif geo == "PLANE":
                fbb    = face.BoundingBox()
                width  = round(fbb.xmax - fbb.xmin, 2)
                length = round(fbb.ymax - fbb.ymin, 2)
                depth  = round(fbb.zmax - fbb.zmin, 2)

                # Classify as recess if significantly smaller than bounding box
                bb_x = result["bounding_box_mm"]["x"]
                bb_y = result["bounding_box_mm"]["y"]

                if (width < bb_x * 0.9 and length < bb_y * 0.9
                        and width > 3.0 and length > 3.0
                        and depth > 1.0):
                    result["planar_recesses"].append({
                        "width_mm":  width,
                        "length_mm": length,
                        "depth_mm":  depth,
                    })

    except Exception as e:
        result["warnings"] = [f"B-rep measurement error: {e}"]

    return result
```

---

## Part 2 — Feature Attribute Assembly

```python
def assemble_feature_attributes(
    detected_features: list[dict],   # from Phase 2 features.json
    step_pmi: dict,                   # from parse_step_pmi()
    brep_data: dict,                  # from measure_brep_features()
    default_material: str = "aluminium_6061",
) -> dict:
    """
    Match extracted PMI and geometry to detected feature types.
    Produces the pmi_data.json structure.

    Matching strategy:
      - Sort holes by diameter descending
      - Match to through_hole / blind_hole detections in order
      - Sort planar_recesses by area descending
      - Match to pocket / slot / step detections in order
      - Surface finish: use first Ra annotation found (part-level default)
      - Threads: match to hole features if thread specs found
      - Material: from PMI if found, else default_material

    This matching is approximate for multi-feature parts.
    It is accurate for single-instance features.
    """
    material = step_pmi.get("material") or default_material
    mat_source = "pmi" if step_pmi.get("material") else "default"

    # Global Ra from PMI (first annotation found, used as part-level default)
    global_Ra = None
    global_Ra_source = "default"
    if step_pmi.get("surface_finish"):
        vals = [sf["Ra_um"] for sf in step_pmi["surface_finish"]]
        global_Ra = round(sum(vals) / len(vals), 2)
        global_Ra_source = "pmi"

    holes          = sorted(brep_data.get("holes", []),
                            key=lambda h: h["diameter_mm"], reverse=True)
    recesses       = sorted(brep_data.get("planar_recesses", []),
                            key=lambda r: r["width_mm"] * r["length_mm"],
                            reverse=True)
    thread_specs   = [t["spec"] for t in step_pmi.get("threads", [])]
    hole_types     = ["through_hole", "blind_hole"]
    recess_types   = ["rectangular_pocket", "circular_pocket",
                      "rectangular_slot", "circular_slot",
                      "rectangular_step", "triangular_pocket"]

    hole_queue    = list(holes)
    recess_queue  = list(recesses)
    thread_queue  = list(thread_specs)

    features_out = []
    instance_counters = {}

    for feat in detected_features:
        ftype = feat["type"]
        idx   = instance_counters.get(ftype, 0)
        instance_counters[ftype] = idx + 1

        # Determine Ra for this feature
        Ra_um     = global_Ra or DEFAULT_RA.get(ftype, 3.2)
        Ra_source = global_Ra_source if global_Ra else "default"

        entry = {
            "type":        ftype,
            "instance_id": idx,
            "Ra_um":       Ra_um,
            "Ra_source":   Ra_source,
            "threaded":    False,
            "thread_spec": None,
        }

        if ftype in hole_types:
            if hole_queue:
                h = hole_queue.pop(0)
                entry["diameter_mm"] = h["diameter_mm"]
                entry["depth_mm"]    = h["depth_mm"]
                entry["depth_ratio"] = round(
                    h["depth_mm"] / max(h["diameter_mm"], 0.1), 2)
                entry["peck_required"] = entry["depth_ratio"] > PECK_DEPTH_RATIO
                # Assign thread spec if available
                if thread_queue:
                    entry["threaded"]    = True
                    entry["thread_spec"] = thread_queue.pop(0)
            else:
                # No geometric data — use defaults from bounding box
                bb   = brep_data.get("bounding_box_mm", {})
                diam = round(min(bb.get("x", 20), bb.get("y", 20)) * 0.15, 1)
                dep  = round(bb.get("z", 40) * 0.6, 1)
                entry["diameter_mm"]  = diam
                entry["depth_mm"]     = dep
                entry["depth_ratio"]  = round(dep / max(diam, 0.1), 2)
                entry["peck_required"] = entry["depth_ratio"] > PECK_DEPTH_RATIO

        elif ftype in recess_types:
            if recess_queue:
                r = recess_queue.pop(0)
                entry["width_mm"]  = r["width_mm"]
                entry["length_mm"] = r["length_mm"]
                entry["depth_mm"]  = r["depth_mm"]
                doc = ROUGH_DOC_MM.get(material, 2.0)
                entry["rough_passes"] = max(1, int(
                    (r["depth_mm"] / doc) + 0.5))
            else:
                bb = brep_data.get("bounding_box_mm", {})
                entry["width_mm"]    = round(bb.get("x", 50) * 0.4, 1)
                entry["length_mm"]   = round(bb.get("y", 40) * 0.4, 1)
                entry["depth_mm"]    = round(bb.get("z", 30) * 0.3, 1)
                entry["rough_passes"] = 2

        features_out.append(entry)

    return {
        "material":       material,
        "material_source": mat_source,
        "features":       features_out,
    }
```

---

## Part 3 — Public Interface

### `step_pmi_extractor.py`

```python
def extract_pmi(
    step_path: str,
    features_path: str,
    output_dir: str,
    default_material: str = "aluminium_6061",
) -> dict:
    """
    Extract PMI from a STEP file and produce pmi_data.json.

    Parameters
    ----------
    step_path        : path to original .stp / .step file
    features_path    : path to features.json from Phase 2
    output_dir       : directory to write pmi_data.json
    default_material : fallback material if not found in STEP

    Returns
    -------
    dict matching pmi_data.json schema.

    Raises
    ------
    FileNotFoundError : step_path or features_path does not exist

    Output
    ------
    {output_dir}/pmi_data.json  written atomically
    """
```

### Full flow

```
extract_pmi(step_path, features_path, output_dir, default_material)
│
├─ 1. Validate both input paths exist
├─ 2. Load features.json → detected_features list
├─ 3. parse_step_pmi(step_path)      → step_pmi dict
├─ 4. measure_brep_features(step_path) → brep_data dict
├─ 5. assemble_feature_attributes(
│        detected_features, step_pmi, brep_data, default_material)
│      → assembled dict
├─ 6. Build pmi_data result dict
├─ 7. Write pmi_data.json atomically
└─ 8. Return result
```

### CLI

```bash
python step_pmi_extractor.py \
    tests/fixtures/complex_prismatic.stp \
    data/processed/complex_prismatic/features.json \
    data/processed/complex_prismatic/ \
    --material aluminium_6061
```

---

## Part 4 — Update Phase 4 to Consume PMI

### Modify `phase4_process_plan.py`

#### New signature

```python
def generate_process_plan(
    metadata_path: str,
    features_path: str,
    setup_analysis_path: str,
    output_dir: str,
    confidence_threshold: float = 0.5,
    pmi_data_path: str | None = None,   # NEW — optional
) -> dict:
```

If `pmi_data_path` is None or the file doesn't exist, Phase 4 falls back
to the existing rule-based logic unchanged. When PMI is available, it
uses dimensional information to enrich operations.

#### New PMI-aware operation expansion

Replace or augment `_expand_feature` with a PMI-aware version:

```python
def _expand_feature_with_pmi(
    feature_type: str,
    setup_id: int,
    approach_direction: str,
    pmi: dict | None,         # single feature entry from pmi_data.json
    material: str = "aluminium_6061",
) -> list[dict]:
    """
    Expand a feature to operations using PMI data when available.
    Falls back to OPERATION_MAP when PMI is None.

    PMI-aware decisions:

    HOLES:
      All holes: centre_drill → drill
      depth_ratio > 3.0  → use drill_peck instead of drill
      Ra < 1.6           → add boring pass
      threaded = True    → add tap operation

    POCKETS / SLOTS / STEPS:
      rough_passes > 1   → repeat endmill_rough N times in operation list
      Ra < 1.6           → add endmill_finish with light doc

    FACES:
      Ra < 0.8           → add face_mill_finish as second pass
    """
```

#### Tool size selection

Add a tool size to the `tool_type` field when dimensions are available:

```python
def _select_tool_size(feature_type: str, pmi: dict | None) -> str:
    """
    Return a sized tool label e.g. 'twist_drill_10mm'.
    Falls back to generic label when no PMI.
    """
    if pmi is None:
        return OPERATION_MAP_TOOL_DEFAULTS.get(feature_type, "generic_tool")

    if feature_type in ("through_hole", "blind_hole"):
        d = pmi.get("diameter_mm")
        if d:
            d_rounded = _round_to_standard_drill(d)
            return f"twist_drill_{d_rounded}mm"

    if feature_type in ("rectangular_pocket", "circular_pocket",
                        "rectangular_slot", "triangular_pocket"):
        w = pmi.get("width_mm")
        if w:
            em_size = _select_endmill_size(w)
            return f"flat_endmill_{em_size}mm"

    return OPERATION_MAP_TOOL_DEFAULTS.get(feature_type, "generic_tool")


def _round_to_standard_drill(diameter_mm: float) -> float:
    """Round to nearest standard drill size from preferred number series."""
    STANDARD_DRILLS = [
        1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
        5.5, 6.0, 6.5, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 22.0, 25.0,
        28.0, 30.0, 32.0, 35.0, 38.0, 40.0, 42.0, 45.0, 50.0,
    ]
    return min(STANDARD_DRILLS, key=lambda x: abs(x - diameter_mm))


def _select_endmill_size(pocket_width_mm: float) -> float:
    """
    Select endmill diameter as ~40% of pocket width,
    rounded to standard size.
    """
    STANDARD_ENDMILLS = [3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32]
    target = pocket_width_mm * 0.40
    return min(STANDARD_ENDMILLS, key=lambda x: abs(x - target))
```

#### New operation type for tapping

Add to `OPERATION_MAP`:
```python
"tap": {"type": "tap", "tool": "tap", "phase": "roughing"}
```

Add to `OPERATION_NOTES`:
```python
"tap": "Thread hole to specified tap size",
```

---

## Part 5 — Update `run_pipeline.py`

### Add PMI extraction as a new step between Phase 2 and Phase 3

Add to `PHASE_NAMES`:
```python
PHASE_NAMES = {
    1: "STEP → Voxel",
    2: "Voxel → Features",
    "2b": "PMI Extraction",    # NEW
    3: "Setup Analysis",
    4: "Process Plan",
    5: "Time Estimate",
    6: "Quotation",
}
```

Actually, to keep phase numbering clean and avoid breaking existing tests,
implement PMI extraction as part of Phase 2's output step rather than a
new numbered phase. Add it to `run_phase2`:

```python
def run_phase2(args, paths):
    # ... existing feature recognition code ...

    # Run PMI extraction immediately after feature recognition
    from step_pmi_extractor import extract_pmi
    pmi_result = extract_pmi(
        step_path=args.step_file,
        features_path=features_path,
        output_dir=args.output,
        default_material=args.material,
    )

    return {
        "features_file": features_path,
        "pmi_data_file": pmi_result["pmi_data_file"],
        "feature_count": result["feature_count"],
        "model_used":    model_path,
        "duration_sec":  round(time.time() - t0, 2),
    }
```

Update `PHASE_PATH_KEYS`:
```python
PHASE_PATH_KEYS = {
    ...
    2: ["features_file", "pmi_data_file"],   # add pmi_data_file
    ...
}
```

Update `run_phase4` to pass `pmi_data_path`:
```python
def run_phase4(args, paths):
    result = generate_process_plan(
        paths["metadata_file"],
        paths["features_file"],
        paths["setup_analysis_file"],
        args.output,
        confidence_threshold=args.confidence,
        pmi_data_path=paths.get("pmi_data_file"),   # NEW
    )
    ...
```

---

## Part 6 — Tests

### `tests/test_pmi_extractor.py`

```python
import os, json
import pytest
from step_pmi_extractor import (
    extract_pmi,
    parse_step_pmi,
    measure_brep_features,
    assemble_feature_attributes,
    MATERIAL_NAME_MAP,
    DEFAULT_RA,
    PECK_DEPTH_RATIO,
)

SIMPLE_BLOCK   = "tests/fixtures/simple_block.stp"
BLOCK_HOLES    = "tests/fixtures/block_with_holes.stp"
COMPLEX        = "tests/fixtures/complex_prismatic.stp"
CLI_FEATURES   = "data/processed/simple_block_cli/features.json"


# ── parse_step_pmi ────────────────────────────────────────────────────────────

def test_parse_step_pmi_returns_dict():
    result = parse_step_pmi(SIMPLE_BLOCK)
    assert isinstance(result, dict)
    assert "material" in result
    assert "surface_finish" in result
    assert "threads" in result


def test_parse_step_pmi_never_raises():
    """Must not raise on any valid file."""
    for path in [SIMPLE_BLOCK, BLOCK_HOLES, COMPLEX]:
        result = parse_step_pmi(path)
        assert isinstance(result, dict)


def test_parse_step_pmi_nonexistent_file():
    result = parse_step_pmi("nonexistent.stp")
    assert result["material"] is None
    assert result["surface_finish"] == []


# ── measure_brep_features ─────────────────────────────────────────────────────

def test_measure_brep_bounding_box_positive():
    result = measure_brep_features(SIMPLE_BLOCK)
    bb = result["bounding_box_mm"]
    assert bb["x"] > 0
    assert bb["y"] > 0
    assert bb["z"] > 0


def test_measure_brep_simple_block_dimensions():
    """simple_block is 100×60×40 mm — check within 5%."""
    result = measure_brep_features(SIMPLE_BLOCK)
    bb = result["bounding_box_mm"]
    dims = sorted([bb["x"], bb["y"], bb["z"]])
    assert abs(dims[0] - 40) / 40 < 0.05
    assert abs(dims[1] - 60) / 60 < 0.05
    assert abs(dims[2] - 100) / 100 < 0.05


def test_measure_brep_block_with_holes_finds_cylinders():
    """block_with_holes has 3 Ø10mm holes — should find cylindrical faces."""
    result = measure_brep_features(BLOCK_HOLES)
    assert len(result["holes"]) >= 1


def test_measure_brep_hole_diameter_reasonable():
    result = measure_brep_features(BLOCK_HOLES)
    for hole in result["holes"]:
        assert 1.0 <= hole["diameter_mm"] <= 200.0
        assert hole["depth_mm"] > 0


def test_measure_brep_complex_finds_recesses():
    """complex_prismatic has a pocket and step — should find planar recesses."""
    result = measure_brep_features(COMPLEX)
    assert len(result["planar_recesses"]) >= 1


def test_measure_brep_never_raises():
    for path in [SIMPLE_BLOCK, BLOCK_HOLES, COMPLEX]:
        result = measure_brep_features(path)
        assert "bounding_box_mm" in result


# ── assemble_feature_attributes ───────────────────────────────────────────────

def test_assemble_returns_all_features():
    features = [
        {"type": "flat_face",   "confidence": 0.99},
        {"type": "through_hole","confidence": 0.85},
    ]
    step_pmi = {"material": None, "surface_finish": [], "threads": []}
    brep     = {"holes": [{"diameter_mm": 10.0, "depth_mm": 40.0}],
                "planar_recesses": [], "bounding_box_mm": {"x":100,"y":60,"z":40}}
    result = assemble_feature_attributes(features, step_pmi, brep)
    assert len(result["features"]) == 2


def test_assemble_hole_gets_diameter():
    features = [{"type": "through_hole", "confidence": 0.85}]
    step_pmi = {"material": None, "surface_finish": [], "threads": []}
    brep     = {"holes": [{"diameter_mm": 10.0, "depth_mm": 40.0}],
                "planar_recesses": [], "bounding_box_mm": {"x":100,"y":60,"z":40}}
    result = assemble_feature_attributes(features, step_pmi, brep)
    hole_feat = result["features"][0]
    assert hole_feat["diameter_mm"] == 10.0
    assert hole_feat["depth_mm"] == 40.0


def test_assemble_deep_hole_sets_peck():
    features = [{"type": "through_hole", "confidence": 0.85}]
    step_pmi = {"material": None, "surface_finish": [], "threads": []}
    # depth_ratio = 40/10 = 4.0 > PECK_DEPTH_RATIO
    brep = {"holes": [{"diameter_mm": 10.0, "depth_mm": 40.0}],
            "planar_recesses": [], "bounding_box_mm": {"x":100,"y":60,"z":40}}
    result = assemble_feature_attributes(features, step_pmi, brep)
    assert result["features"][0]["peck_required"] is True


def test_assemble_shallow_hole_no_peck():
    features = [{"type": "through_hole", "confidence": 0.85}]
    step_pmi = {"material": None, "surface_finish": [], "threads": []}
    # depth_ratio = 10/10 = 1.0 < PECK_DEPTH_RATIO
    brep = {"holes": [{"diameter_mm": 10.0, "depth_mm": 10.0}],
            "planar_recesses": [], "bounding_box_mm": {"x":100,"y":60,"z":40}}
    result = assemble_feature_attributes(features, step_pmi, brep)
    assert result["features"][0]["peck_required"] is False


def test_assemble_thread_assigned_to_hole():
    features = [{"type": "through_hole", "confidence": 0.85}]
    step_pmi = {"material": None, "surface_finish": [],
                "threads": [{"spec": "M10X1.5", "face_ref": "..."}]}
    brep = {"holes": [{"diameter_mm": 10.0, "depth_mm": 20.0}],
            "planar_recesses": [], "bounding_box_mm": {"x":100,"y":60,"z":40}}
    result = assemble_feature_attributes(features, step_pmi, brep)
    assert result["features"][0]["threaded"] is True
    assert result["features"][0]["thread_spec"] == "M10X1.5"


def test_assemble_pmi_material_used():
    features = [{"type": "flat_face", "confidence": 0.99}]
    step_pmi = {"material": "mild_steel", "surface_finish": [], "threads": []}
    brep = {"holes": [], "planar_recesses": [],
            "bounding_box_mm": {"x":100,"y":60,"z":40}}
    result = assemble_feature_attributes(features, step_pmi, brep)
    assert result["material"] == "mild_steel"
    assert result["material_source"] == "pmi"


def test_assemble_default_material_when_no_pmi():
    features = [{"type": "flat_face", "confidence": 0.99}]
    step_pmi = {"material": None, "surface_finish": [], "threads": []}
    brep = {"holes": [], "planar_recesses": [],
            "bounding_box_mm": {"x":100,"y":60,"z":40}}
    result = assemble_feature_attributes(
        features, step_pmi, brep, default_material="stainless_316")
    assert result["material"] == "stainless_316"
    assert result["material_source"] == "default"


# ── extract_pmi (full pipeline) ───────────────────────────────────────────────

@pytest.mark.skipif(
    not os.path.exists(CLI_FEATURES),
    reason="Phase 2 CLI output not available"
)
def test_extract_pmi_creates_file(tmp_path):
    result = extract_pmi(SIMPLE_BLOCK, CLI_FEATURES, str(tmp_path))
    assert os.path.exists(os.path.join(tmp_path, "pmi_data.json"))


@pytest.mark.skipif(
    not os.path.exists(CLI_FEATURES),
    reason="Phase 2 CLI output not available"
)
def test_extract_pmi_schema(tmp_path):
    result = extract_pmi(SIMPLE_BLOCK, CLI_FEATURES, str(tmp_path))
    for key in ["source_file", "material", "material_source",
                "features", "pmi_data_file", "warnings"]:
        assert key in result


@pytest.mark.skipif(
    not os.path.exists(CLI_FEATURES),
    reason="Phase 2 CLI output not available"
)
def test_extract_pmi_path_absolute(tmp_path):
    result = extract_pmi(SIMPLE_BLOCK, CLI_FEATURES, str(tmp_path))
    assert os.path.isabs(result["pmi_data_file"])


def test_extract_pmi_missing_step_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_pmi("no_such.stp", CLI_FEATURES, str(tmp_path))


def test_extract_pmi_missing_features_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_pmi(SIMPLE_BLOCK, "no_features.json", str(tmp_path))


# ── Phase 4 PMI-aware plan ────────────────────────────────────────────────────

@pytest.mark.skipif(
    not os.path.exists(CLI_FEATURES),
    reason="Phase 2 CLI output not available"
)
def test_phase4_pmi_adds_peck_for_deep_hole(tmp_path):
    """
    Inject a pmi_data.json with a deep hole and verify
    Phase 4 adds drill_peck instead of drill.
    """
    import json
    from phase4_process_plan import generate_process_plan

    pmi = {
        "material": "aluminium_6061", "material_source": "default",
        "features": [
            {"type": "flat_face",    "instance_id": 0,
             "Ra_um": 3.2, "Ra_source": "default",
             "threaded": False, "thread_spec": None},
            {"type": "through_hole", "instance_id": 0,
             "diameter_mm": 10.0, "depth_mm": 40.0, "depth_ratio": 4.0,
             "Ra_um": 1.6, "Ra_source": "default",
             "threaded": False, "thread_spec": None,
             "peck_required": True},
        ],
        "pmi_data_file": str(tmp_path / "pmi_data.json"), "warnings": []
    }
    pmi_path = tmp_path / "pmi_data.json"
    pmi_path.write_text(json.dumps(pmi))

    # Minimal process plan inputs
    meta_path  = "data/processed/simple_block_cli/metadata.json"
    feat_path  = CLI_FEATURES
    setup_path = "data/processed/simple_block_cli/setup_analysis.json"
    if not all(os.path.exists(p) for p in [meta_path, feat_path, setup_path]):
        pytest.skip("CLI outputs not available")

    result = generate_process_plan(
        meta_path, feat_path, setup_path, str(tmp_path),
        pmi_data_path=str(pmi_path),
    )
    op_types = [op["operation_type"] for op in result["operations"]]
    assert "drill_peck" in op_types


@pytest.mark.skipif(
    not os.path.exists(CLI_FEATURES),
    reason="Phase 2 CLI output not available"
)
def test_phase4_pmi_adds_tap_for_threaded_hole(tmp_path):
    """Threaded hole in PMI data → tap operation in plan."""
    import json
    from phase4_process_plan import generate_process_plan

    pmi = {
        "material": "aluminium_6061", "material_source": "default",
        "features": [
            {"type": "flat_face",    "instance_id": 0,
             "Ra_um": 3.2, "Ra_source": "default",
             "threaded": False, "thread_spec": None},
            {"type": "through_hole", "instance_id": 0,
             "diameter_mm": 10.0, "depth_mm": 20.0, "depth_ratio": 2.0,
             "Ra_um": 1.6, "Ra_source": "default",
             "threaded": True, "thread_spec": "M10X1.5",
             "peck_required": False},
        ],
        "pmi_data_file": str(tmp_path / "pmi_data.json"), "warnings": []
    }
    pmi_path = tmp_path / "pmi_data.json"
    pmi_path.write_text(json.dumps(pmi))

    meta_path  = "data/processed/simple_block_cli/metadata.json"
    feat_path  = CLI_FEATURES
    setup_path = "data/processed/simple_block_cli/setup_analysis.json"
    if not all(os.path.exists(p) for p in [meta_path, feat_path, setup_path]):
        pytest.skip("CLI outputs not available")

    result = generate_process_plan(
        meta_path, feat_path, setup_path, str(tmp_path),
        pmi_data_path=str(pmi_path),
    )
    op_types = [op["operation_type"] for op in result["operations"]]
    assert "tap" in op_types


def test_phase4_without_pmi_unchanged(tmp_path):
    """
    Phase 4 with pmi_data_path=None must produce same output
    as before this change (backward compatible).
    """
    import json
    from phase4_process_plan import generate_process_plan

    # Use synthetic inputs matching existing test patterns
    # (existing test_phase4.py fixtures are already valid)
    pass   # covered by existing test_phase4.py tests which pass None
```

---

## Quick-Start Commands

```bash
# Extract PMI from complex_prismatic fixture
python step_pmi_extractor.py \
    tests/fixtures/complex_prismatic.stp \
    data/processed/simple_block_cli/features.json \
    data/processed/simple_block_cli/

# Run full pipeline with PMI (auto-detected)
python run_pipeline.py \
    tests/fixtures/complex_prismatic.stp \
    factory_profiles/nash_nz.json

# Run tests
pytest tests/test_pmi_extractor.py -v
pytest tests/ -v
```

---

## Acceptance Criteria

- [ ] `step_pmi_extractor.py` runs on all 3 fixtures without error
- [ ] `pmi_data.json` created with correct schema for all fixtures
- [ ] `block_with_holes` fixture produces ≥ 1 hole with `diameter_mm` set
- [ ] Deep hole (depth/diameter > 3) sets `peck_required: True`
- [ ] Threaded hole in PMI → `tap` operation appears in Phase 4 output
- [ ] `complex_prismatic` produces `operation_count >= 8` (vs 12 previously,
      now dimensionally-informed)
- [ ] Phase 4 with `pmi_data_path=None` produces identical output to
      current behaviour (backward compatible)
- [ ] `pytest tests/test_pmi_extractor.py -v` — all tests pass
- [ ] `pytest tests/ -v` — all existing 216 tests still pass

---

## Notes for Codex

1. **`parse_step_pmi` must never raise.** It reads untrusted file content.
   Wrap everything in try/except and return partial results.

2. **`measure_brep_features` must never raise.** cadquery can fail on
   complex geometry. Return empty lists and log a warning.

3. **Backward compatibility is mandatory.** `generate_process_plan` with
   `pmi_data_path=None` must produce identical output to the current
   implementation. Add PMI logic in a branch, do not restructure the
   existing code path.

4. **B-rep cylinder detection is approximate.** cadquery's `face.geomType()`
   returns `"CYLINDER"` for cylindrical faces. The diameter estimate from
   the bounding box is approximate — error ≤ 15% for standard holes.
   This is acceptable for tool class selection.

5. **`_round_to_standard_drill` and `_select_endmill_size` must be
   importable** from `phase4_process_plan` for testing.

6. **The `tap` operation must be added to both `OPERATION_MAP` and
   `OPERATION_NOTES`** in `phase4_process_plan.py`. It is a roughing
   operation (comes after drilling, before finishing).

7. **`pmi_data_file` path in output must be absolute.**
   Use `os.path.abspath()` when assembling the result dict.

8. **Material from PMI overrides `args.material` in the pipeline.**
   If the STEP file says "316 stainless" and the user passed
   `--material aluminium_6061`, PMI wins. Log a warning noting the
   override so the user is aware.
