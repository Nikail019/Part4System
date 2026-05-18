# AGENTS.md — Pipeline Runner: End-to-End STEP → Quotation

## Context

All six phases are implemented and passing (178 tests).

This AGENTS.md implements a single pipeline runner that chains all six
phases in sequence, producing a `pipeline_manifest.json` alongside all
intermediate outputs. After this is done, a user can go from a raw STEP
file to a job quotation in one command.

---

## Usage

```bash
# Full pipeline
python run_pipeline.py part.stp factory_profiles/nash_nz.json \
    --material    aluminium_6061          \
    --output      data/processed/my_part/ \
    --model       checkpoints/best.pt     \
    --resolution  64                      \
    --confidence  0.5

# Resume from a specific phase (skips completed phases)
python run_pipeline.py part.stp factory_profiles/nash_nz.json \
    --output data/processed/my_part/ \
    --resume-from 4

# Dry run (validate inputs, print plan, do not execute)
python run_pipeline.py part.stp factory_profiles/nash_nz.json \
    --output data/processed/my_part/ \
    --dry-run
```

---

## Repository Additions

```
rpp-mvp/
├── run_pipeline.py          # IMPLEMENT — main pipeline runner
└── tests/
    └── test_pipeline.py     # IMPLEMENT — integration tests
```

---

## Output Directory Layout

After a complete run, the output directory contains:

```
{output_dir}/
├── pipeline_manifest.json   # master index of all outputs + timing
├── voxel_64.npy             # Phase 1
├── mesh.stl                 # Phase 1
├── metadata.json            # Phase 1
├── features.json            # Phase 2
├── setup_analysis.json      # Phase 3
├── accessibility_map.npy    # Phase 3
├── surface_mask.npy         # Phase 3
├── process_plan.json        # Phase 4
├── time_estimate.json       # Phase 5
└── quotation.json           # Phase 6
```

---

## `pipeline_manifest.json` Schema

```json
{
  "step_file":        "/abs/path/to/part.stp",
  "factory_profile":  "/abs/path/to/nash_nz.json",
  "material":         "aluminium_6061",
  "model_path":       "/abs/path/to/best.pt",
  "output_dir":       "/abs/path/to/output/",
  "resolution":       64,
  "confidence":       0.5,
  "timestamp":        "2025-01-01T12:00:00",

  "phases_completed": [1, 2, 3, 4, 5, 6],
  "phases_skipped":   [],

  "phase_outputs": {
    "1": {
      "status":          "completed",
      "duration_sec":    2.31,
      "voxel_file":      "/abs/path/voxel_64.npy",
      "metadata_file":   "/abs/path/metadata.json",
      "mesh_file":       "/abs/path/mesh.stl"
    },
    "2": {
      "status":          "completed",
      "duration_sec":    0.84,
      "features_file":   "/abs/path/features.json",
      "feature_count":   3,
      "model_used":      "/abs/path/best.pt"
    },
    "3": {
      "status":          "completed",
      "duration_sec":    1.12,
      "setup_analysis_file":   "/abs/path/setup_analysis.json",
      "setup_count":           2,
      "axis_requirement":      3
    },
    "4": {
      "status":          "completed",
      "duration_sec":    0.18,
      "process_plan_file":  "/abs/path/process_plan.json",
      "operation_count":    13
    },
    "5": {
      "status":          "completed",
      "duration_sec":    0.09,
      "time_estimate_file": "/abs/path/time_estimate.json",
      "total_time_min":     97.7
    },
    "6": {
      "status":          "completed",
      "duration_sec":    0.07,
      "quotation_file":  "/abs/path/quotation.json",
      "recommendation":  "ACCEPT",
      "total_cost":      229.73,
      "currency":        "NZD"
    }
  },

  "summary": {
    "recommendation":  "ACCEPT",
    "total_cost":      229.73,
    "currency":        "NZD",
    "total_time_min":  97.7,
    "operation_count": 13,
    "setup_count":     2,
    "axis_requirement": 3,
    "flags":           []
  },

  "total_duration_sec": 4.61,
  "warnings":           []
}
```

---

## Constants and Configuration

```python
DEFAULT_RESOLUTION  = 64
DEFAULT_MATERIAL    = "aluminium_6061"
DEFAULT_CONFIDENCE  = 0.5
DEFAULT_SETUP_TIME  = 15.0
DEFAULT_TOOL_CHANGE = 2.0

PHASE_NAMES = {
    1: "STEP → Voxel",
    2: "Voxel → Features",
    3: "Setup Analysis",
    4: "Process Plan",
    5: "Time Estimate",
    6: "Quotation",
}

PHASE_OUTPUT_FILES = {
    1: ["voxel_64.npy", "metadata.json", "mesh.stl"],
    2: ["features.json"],
    3: ["setup_analysis.json", "accessibility_map.npy", "surface_mask.npy"],
    4: ["process_plan.json"],
    5: ["time_estimate.json"],
    6: ["quotation.json"],
}
```

---

## Implementation — `run_pipeline.py`

### Argument Parsing

```python
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RPP Pipeline: STEP file → job quotation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py part.stp factory_profiles/nash_nz.json
  python run_pipeline.py part.stp factory_profiles/nash_nz.json \\
      --material mild_steel --output data/processed/my_part/
  python run_pipeline.py part.stp factory_profiles/nash_nz.json \\
      --resume-from 4 --output data/processed/my_part/
        """,
    )
    parser.add_argument("step_file",
        help="Path to input STEP (.stp / .step) file")
    parser.add_argument("factory_profile",
        help="Path to factory profile JSON")
    parser.add_argument("--material",      default=DEFAULT_MATERIAL,
        help=f"Material key (default: {DEFAULT_MATERIAL})")
    parser.add_argument("--output",        default=None,
        help="Output directory (default: data/processed/<stem>/)")
    parser.add_argument("--model",         default=None,
        help="Path to trained Phase 2 model checkpoint (.pt)")
    parser.add_argument("--resolution",    type=int, default=DEFAULT_RESOLUTION,
        help=f"Voxel resolution (default: {DEFAULT_RESOLUTION})")
    parser.add_argument("--confidence",    type=float, default=DEFAULT_CONFIDENCE,
        help=f"Feature confidence threshold (default: {DEFAULT_CONFIDENCE})")
    parser.add_argument("--resume-from",   type=int, default=1,
        dest="resume_from", choices=range(1, 7),
        help="Resume pipeline from this phase (skips earlier phases)")
    parser.add_argument("--dry-run",       action="store_true",
        help="Validate inputs and print execution plan without running")
    parser.add_argument("--quiet",         action="store_true",
        help="Suppress per-phase progress output")
    return parser.parse_args()
```

### Phase Completion Detection

```python
def phase_is_complete(phase: int, output_dir: str, resolution: int = 64) -> bool:
    """
    Return True if all expected output files for this phase exist.
    Uses PHASE_OUTPUT_FILES, substituting actual resolution in voxel filename.
    """
    files = PHASE_OUTPUT_FILES[phase]
    for f in files:
        actual = f.replace("voxel_64.npy", f"voxel_{resolution}.npy")
        if not os.path.exists(os.path.join(output_dir, actual)):
            return False
    return True
```

### Phase Runners

Implement one function per phase. Each must:
- Accept an `args` namespace and a `paths` dict of previously computed file paths
- Return a dict of newly created file paths + timing info
- Print a one-line status update unless `args.quiet`
- Let exceptions propagate — the main runner handles them

```python
def run_phase1(args: argparse.Namespace, paths: dict) -> dict:
    """
    Run Phase 1: STEP → Voxel.
    Returns {"voxel_file": str, "metadata_file": str, "mesh_file": str,
             "duration_sec": float}
    """
    from phase1_voxeliser import process_step_file
    t0 = time.time()
    result = process_step_file(
        args.step_file,
        args.output,
        resolution=args.resolution,
    )
    return {
        "voxel_file":    result["voxel_file"],
        "metadata_file": os.path.join(args.output, "metadata.json"),
        "mesh_file":     result["mesh_file"],
        "duration_sec":  round(time.time() - t0, 2),
    }


def run_phase2(args: argparse.Namespace, paths: dict) -> dict:
    """
    Run Phase 2: Voxel → Features.
    Falls back to a default feature set if no model checkpoint is available.
    Returns {"features_file": str, "feature_count": int,
             "model_used": str | None, "duration_sec": float}
    """
    from phase2_feature_recognition import recognise_features
    import json

    t0 = time.time()
    voxel_file = paths["voxel_file"]
    features_path = os.path.join(args.output, "features.json")

    if args.model and os.path.exists(args.model):
        result = recognise_features(
            voxel_file,
            args.model,
            threshold=args.confidence,
        )
        model_used = args.model
    else:
        # No trained model available — use fallback
        result = _default_features(args.confidence)
        model_used = None
        # warn is collected in calling code

    _write_json_atomic(result, features_path)
    return {
        "features_file": features_path,
        "feature_count": result["feature_count"],
        "model_used":    model_used,
        "duration_sec":  round(time.time() - t0, 2),
    }


def run_phase3(args: argparse.Namespace, paths: dict) -> dict:
    """Run Phase 3: Setup Analysis."""
    from phase3_setup_analysis import analyse_setups
    t0 = time.time()
    result = analyse_setups(
        paths["voxel_file"],
        args.output,
        features_path=paths.get("features_file"),
    )
    return {
        "setup_analysis_file": os.path.join(args.output, "setup_analysis.json"),
        "setup_count":         result["setup_count"],
        "axis_requirement":    result["axis_requirement"],
        "duration_sec":        round(time.time() - t0, 2),
    }


def run_phase4(args: argparse.Namespace, paths: dict) -> dict:
    """Run Phase 4: Process Plan."""
    from phase4_process_plan import generate_process_plan
    t0 = time.time()
    result = generate_process_plan(
        paths["metadata_file"],
        paths["features_file"],
        paths["setup_analysis_file"],
        args.output,
        confidence_threshold=args.confidence,
    )
    return {
        "process_plan_file": result["process_plan_file"],
        "operation_count":   result["operation_count"],
        "duration_sec":      round(time.time() - t0, 2),
    }


def run_phase5(args: argparse.Namespace, paths: dict) -> dict:
    """Run Phase 5: Time Estimate."""
    from phase5_time_estimate import estimate_time
    t0 = time.time()
    result = estimate_time(
        paths["process_plan_file"],
        paths["metadata_file"],
        args.output,
        material=args.material,
    )
    return {
        "time_estimate_file": result["time_estimate_file"],
        "total_time_min":     result["total_time_min"],
        "duration_sec":       round(time.time() - t0, 2),
    }


def run_phase6(args: argparse.Namespace, paths: dict) -> dict:
    """Run Phase 6: Quotation."""
    from phase6_quotation import generate_quotation
    t0 = time.time()
    result = generate_quotation(
        paths["process_plan_file"],
        paths["time_estimate_file"],
        paths["metadata_file"],
        args.factory_profile,
        args.output,
        material=args.material,
    )
    return {
        "quotation_file":   result["quotation_file"],
        "recommendation":   result["recommendation"],
        "total_cost":       result["estimated_cost"]["total"],
        "currency":         result["estimated_cost"]["currency"],
        "duration_sec":     round(time.time() - t0, 2),
    }
```

### Default Feature Fallback

```python
def _default_features(threshold: float) -> dict:
    """
    Return a minimal feature set when no model checkpoint is available.
    Always includes flat_face. Used as a fallback so the pipeline
    can complete without a trained Phase 2 model.
    """
    from models.feature_net import FEATURE_NAMES
    flat_face_idx = FEATURE_NAMES.index("flat_face")
    all_scores = {name: 0.1 for name in FEATURE_NAMES}
    all_scores["flat_face"] = 0.99
    features = [{"type": "flat_face", "confidence": 0.99}]
    return {
        "features":      features,
        "feature_count": 1,
        "all_scores":    all_scores,
        "threshold":     threshold,
        "voxel_file":    "",
        "model_path":    "fallback_no_model",
    }
```

### Main Runner

```python
PHASE_RUNNERS = {
    1: run_phase1,
    2: run_phase2,
    3: run_phase3,
    4: run_phase4,
    5: run_phase5,
    6: run_phase6,
}

# Keys produced by each phase, consumed by later phases
PHASE_PATH_KEYS = {
    1: ["voxel_file", "metadata_file", "mesh_file"],
    2: ["features_file"],
    3: ["setup_analysis_file"],
    4: ["process_plan_file"],
    5: ["time_estimate_file"],
    6: ["quotation_file"],
}


def run_pipeline(args: argparse.Namespace) -> dict:
    """
    Execute all phases in order, respecting --resume-from and phase caching.

    Returns the completed pipeline_manifest dict.
    """
    import time as time_module
    pipeline_start = time_module.time()

    os.makedirs(args.output, exist_ok=True)
    manifest_path = os.path.join(args.output, "pipeline_manifest.json")

    # Load existing manifest if resuming
    manifest = _load_or_create_manifest(args, manifest_path)

    paths   = _collect_existing_paths(args)
    overall_warnings = list(manifest.get("warnings", []))

    for phase_num in range(1, 7):
        phase_name = PHASE_NAMES[phase_num]

        # Skip if before resume point
        if phase_num < args.resume_from:
            if not args.quiet:
                print(f"  Phase {phase_num} [{phase_name}] — skipped (before resume point)")
            manifest["phases_skipped"].append(phase_num)
            continue

        # Skip if already complete (cached outputs exist)
        if phase_num >= args.resume_from and phase_is_complete(
                phase_num, args.output, args.resolution):
            if not args.quiet:
                print(f"  Phase {phase_num} [{phase_name}] — cached ✓")
            _update_paths_from_cache(phase_num, args, paths)
            manifest["phases_completed"].append(phase_num)
            continue

        # Run the phase
        if not args.quiet:
            print(f"  Phase {phase_num} [{phase_name}] ...", end=" ", flush=True)

        phase_result = PHASE_RUNNERS[phase_num](args, paths)

        # Check for Phase 2 fallback warning
        if phase_num == 2 and phase_result.get("model_used") is None:
            w = ("No trained model found. Phase 2 used fallback feature set "
                 "(flat_face only). Train a model with: "
                 "python training/train_feature_net.py")
            overall_warnings.append(w)
            if not args.quiet:
                print(f"\n    ⚠  {w}")

        # Update paths for downstream phases
        for key in PHASE_PATH_KEYS[phase_num]:
            if key in phase_result:
                paths[key] = phase_result[key]

        duration = phase_result.get("duration_sec", 0.0)
        if not args.quiet:
            print(f"done ({duration:.2f}s)")

        manifest["phase_outputs"][str(phase_num)] = {
            "status": "completed",
            **phase_result,
        }
        manifest["phases_completed"].append(phase_num)
        _write_json_atomic(manifest, manifest_path)

    # Final summary
    total_duration = round(time_module.time() - pipeline_start, 2)
    manifest["total_duration_sec"] = total_duration
    manifest["warnings"] = overall_warnings
    manifest["summary"] = _build_summary(manifest)
    _write_json_atomic(manifest, manifest_path)

    return manifest
```

### Summary Builder

```python
def _build_summary(manifest: dict) -> dict:
    """Extract key results from phase outputs for the top-level summary."""
    outputs = manifest.get("phase_outputs", {})
    p6 = outputs.get("6", {})
    p5 = outputs.get("5", {})
    p4 = outputs.get("4", {})
    p3 = outputs.get("3", {})
    return {
        "recommendation":  p6.get("recommendation"),
        "total_cost":      p6.get("total_cost"),
        "currency":        p6.get("currency"),
        "total_time_min":  p5.get("total_time_min"),
        "operation_count": p4.get("operation_count"),
        "setup_count":     p3.get("setup_count"),
        "axis_requirement":p3.get("axis_requirement"),
        "flags":           [],   # populated from quotation.json on final read
    }
```

### Terminal Summary Printer

```python
def print_summary(manifest: dict) -> None:
    """Print a formatted summary of the pipeline result."""
    s = manifest.get("summary", {})
    rec = s.get("recommendation", "UNKNOWN")
    rec_symbol = "✓" if rec == "ACCEPT" else "✗"
    currency   = s.get("currency", "")
    cost       = s.get("total_cost")
    time_min   = s.get("total_time_min")
    ops        = s.get("operation_count")
    setups     = s.get("setup_count")
    axes       = s.get("axis_requirement")
    total_sec  = manifest.get("total_duration_sec", 0.0)

    print()
    print("=" * 52)
    print(f"  RPP PIPELINE RESULT")
    print("=" * 52)
    print(f"  Recommendation : {rec_symbol}  {rec}")
    if cost is not None:
        print(f"  Estimated cost : {currency} {cost:.2f}")
    if time_min is not None:
        print(f"  Estimated time : {time_min:.1f} min")
    if ops is not None:
        print(f"  Operations     : {ops}")
    if setups is not None:
        print(f"  Setups         : {setups}  ({axes}-axis)")
    print(f"  Pipeline time  : {total_sec:.1f}s")
    print("=" * 52)

    flags = manifest.get("summary", {}).get("flags", []) or \
            manifest.get("warnings", [])
    if flags:
        print()
        for flag in flags:
            print(f"  ⚠  {flag}")
    print()
```

### Helpers

```python
def _load_or_create_manifest(args: argparse.Namespace, path: str) -> dict:
    """Load existing manifest for resume, or create a fresh one."""
    if args.resume_from > 1 and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {
        "step_file":        os.path.abspath(args.step_file),
        "factory_profile":  os.path.abspath(args.factory_profile),
        "material":         args.material,
        "model_path":       os.path.abspath(args.model) if args.model else None,
        "output_dir":       os.path.abspath(args.output),
        "resolution":       args.resolution,
        "confidence":       args.confidence,
        "timestamp":        datetime.datetime.now().isoformat(timespec="seconds"),
        "phases_completed": [],
        "phases_skipped":   [],
        "phase_outputs":    {},
        "total_duration_sec": 0.0,
        "warnings":         [],
        "summary":          {},
    }


def _collect_existing_paths(args: argparse.Namespace) -> dict:
    """
    Build a paths dict from files that already exist in output_dir.
    Used when resuming so later phases can find earlier outputs.
    """
    R = args.resolution
    candidates = {
        "voxel_file":          os.path.join(args.output, f"voxel_{R}.npy"),
        "metadata_file":       os.path.join(args.output, "metadata.json"),
        "mesh_file":           os.path.join(args.output, "mesh.stl"),
        "features_file":       os.path.join(args.output, "features.json"),
        "setup_analysis_file": os.path.join(args.output, "setup_analysis.json"),
        "process_plan_file":   os.path.join(args.output, "process_plan.json"),
        "time_estimate_file":  os.path.join(args.output, "time_estimate.json"),
        "quotation_file":      os.path.join(args.output, "quotation.json"),
    }
    return {k: v for k, v in candidates.items() if os.path.exists(v)}


def _update_paths_from_cache(phase: int, args: argparse.Namespace, paths: dict) -> None:
    """
    When a phase is skipped (cached), populate paths from the known
    output filenames so downstream phases can find the files.
    """
    R = args.resolution
    additions = {
        1: {
            "voxel_file":    os.path.join(args.output, f"voxel_{R}.npy"),
            "metadata_file": os.path.join(args.output, "metadata.json"),
            "mesh_file":     os.path.join(args.output, "mesh.stl"),
        },
        2: {"features_file":       os.path.join(args.output, "features.json")},
        3: {"setup_analysis_file": os.path.join(args.output, "setup_analysis.json")},
        4: {"process_plan_file":   os.path.join(args.output, "process_plan.json")},
        5: {"time_estimate_file":  os.path.join(args.output, "time_estimate.json")},
        6: {"quotation_file":      os.path.join(args.output, "quotation.json")},
    }
    paths.update(additions.get(phase, {}))


def _write_json_atomic(data: dict, path: str) -> None:
    """Write JSON atomically via temp file + os.replace."""
    dir_ = os.path.dirname(os.path.abspath(path))
    import tempfile
    with tempfile.NamedTemporaryFile("w", dir=dir_, delete=False, suffix=".tmp") as f:
        json.dump(data, f, indent=2)
        tmp = f.name
    os.replace(tmp, path)
```

### Dry Run

```python
def dry_run(args: argparse.Namespace) -> None:
    """Print what the pipeline would do without executing any phase."""
    print()
    print("DRY RUN — no files will be written")
    print()
    print(f"  Input STEP       : {args.step_file}")
    print(f"  Factory profile  : {args.factory_profile}")
    print(f"  Material         : {args.material}")
    print(f"  Output dir       : {args.output}")
    print(f"  Model checkpoint : {args.model or '(none — fallback will be used)'}")
    print(f"  Voxel resolution : {args.resolution}³")
    print(f"  Confidence       : {args.confidence}")
    print(f"  Resume from      : Phase {args.resume_from}")
    print()
    print("  Phases:")
    for phase_num in range(1, 7):
        name = PHASE_NAMES[phase_num]
        status = "SKIP (before resume)" if phase_num < args.resume_from else "RUN"
        cached = phase_is_complete(phase_num, args.output or ".", args.resolution)
        if cached and phase_num >= args.resume_from:
            status = "SKIP (cached outputs exist)"
        print(f"    Phase {phase_num} [{name}] → {status}")
    print()

    # Validate input files exist
    errors = []
    if not os.path.exists(args.step_file):
        errors.append(f"STEP file not found: {args.step_file}")
    if not os.path.exists(args.factory_profile):
        errors.append(f"Factory profile not found: {args.factory_profile}")
    if args.model and not os.path.exists(args.model):
        errors.append(f"Model checkpoint not found: {args.model}")
    if errors:
        print("  ERRORS:")
        for e in errors:
            print(f"    ✗  {e}")
    else:
        print("  Input validation: ✓  all inputs found")
    print()
```

### `__main__` Entry Point

```python
if __name__ == "__main__":
    import datetime

    args = get_args()

    # Default output directory
    if args.output is None:
        stem = os.path.splitext(os.path.basename(args.step_file))[0]
        args.output = os.path.join("data", "processed", stem)

    # Dry run
    if args.dry_run:
        dry_run(args)
        sys.exit(0)

    # Validate required inputs before starting
    errors = []
    if not os.path.exists(args.step_file):
        errors.append(f"STEP file not found: {args.step_file}")
    if not os.path.exists(args.factory_profile):
        errors.append(f"Factory profile not found: {args.factory_profile}")
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print()
        print(f"RPP Pipeline")
        print(f"  Input  : {args.step_file}")
        print(f"  Output : {args.output}")
        print()

    try:
        manifest = run_pipeline(args)
        print_summary(manifest)
    except Exception as e:
        print(f"\nPipeline failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
```

---
---

# TESTS — `tests/test_pipeline.py`

```python
# tests/test_pipeline.py
#
# Integration tests for run_pipeline.py.
# Most tests use synthetic in-memory argument namespaces rather than
# running the full pipeline (which requires STEP files + trained model).
# The end-to-end test is marked skipif and uses the existing CLI outputs.

import os, json, argparse, copy
import pytest

# Import internal helpers directly for unit testing
from run_pipeline import (
    phase_is_complete,
    _default_features,
    _build_summary,
    _collect_existing_paths,
    _update_paths_from_cache,
    _load_or_create_manifest,
    print_summary,
    dry_run,
    PHASE_NAMES,
    PHASE_OUTPUT_FILES,
    DEFAULT_MATERIAL,
    DEFAULT_RESOLUTION,
    DEFAULT_CONFIDENCE,
)

CLI_DIR  = "data/processed/simple_block_cli"
STP_FILE = "tests/fixtures/simple_block.stp"
FACTORY  = "factory_profiles/nash_nz.json"


def make_args(**kwargs) -> argparse.Namespace:
    """Create a minimal args namespace for testing."""
    defaults = {
        "step_file":       STP_FILE,
        "factory_profile": FACTORY,
        "material":        DEFAULT_MATERIAL,
        "output":          "/tmp/rpp_test_output",
        "model":           None,
        "resolution":      DEFAULT_RESOLUTION,
        "confidence":      DEFAULT_CONFIDENCE,
        "resume_from":     1,
        "dry_run":         False,
        "quiet":           True,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ── Constants ─────────────────────────────────────────────────────────────────

def test_phase_names_has_all_six():
    assert set(PHASE_NAMES.keys()) == {1, 2, 3, 4, 5, 6}


def test_phase_output_files_has_all_six():
    assert set(PHASE_OUTPUT_FILES.keys()) == {1, 2, 3, 4, 5, 6}


def test_phase_output_files_all_non_empty():
    for phase, files in PHASE_OUTPUT_FILES.items():
        assert len(files) >= 1, f"Phase {phase} has no output files"


# ── phase_is_complete ─────────────────────────────────────────────────────────

def test_phase_is_complete_false_empty_dir(tmp_path):
    assert not phase_is_complete(1, str(tmp_path))


def test_phase_is_complete_true_when_all_exist(tmp_path):
    for f in PHASE_OUTPUT_FILES[1]:
        actual = f.replace("voxel_64.npy", f"voxel_{DEFAULT_RESOLUTION}.npy")
        (tmp_path / actual).write_text("dummy")
    assert phase_is_complete(1, str(tmp_path))


def test_phase_is_complete_false_partial(tmp_path):
    (tmp_path / "voxel_64.npy").write_text("dummy")
    # metadata.json missing
    assert not phase_is_complete(1, str(tmp_path))


# ── _default_features fallback ────────────────────────────────────────────────

def test_default_features_schema():
    result = _default_features(0.5)
    assert "features" in result
    assert "feature_count" in result
    assert "all_scores" in result


def test_default_features_includes_flat_face():
    result = _default_features(0.5)
    types = [f["type"] for f in result["features"]]
    assert "flat_face" in types


def test_default_features_all_scores_has_all_classes():
    from models.feature_net import FEATURE_NAMES
    result = _default_features(0.5)
    for name in FEATURE_NAMES:
        assert name in result["all_scores"]


def test_default_features_model_path_is_fallback():
    result = _default_features(0.5)
    assert result["model_path"] == "fallback_no_model"


# ── _build_summary ────────────────────────────────────────────────────────────

def test_build_summary_extracts_recommendation():
    manifest = {
        "phase_outputs": {
            "6": {"recommendation": "ACCEPT", "total_cost": 229.73, "currency": "NZD"},
            "5": {"total_time_min": 97.7},
            "4": {"operation_count": 13},
            "3": {"setup_count": 2, "axis_requirement": 3},
        },
        "warnings": [],
    }
    summary = _build_summary(manifest)
    assert summary["recommendation"] == "ACCEPT"
    assert summary["total_cost"] == 229.73
    assert summary["total_time_min"] == 97.7
    assert summary["operation_count"] == 13
    assert summary["setup_count"] == 2


def test_build_summary_handles_missing_phases():
    summary = _build_summary({"phase_outputs": {}, "warnings": []})
    assert summary["recommendation"] is None
    assert summary["total_cost"] is None


# ── _load_or_create_manifest ─────────────────────────────────────────────────

def test_create_manifest_has_required_keys(tmp_path):
    args = make_args(output=str(tmp_path))
    path = str(tmp_path / "pipeline_manifest.json")
    manifest = _load_or_create_manifest(args, path)
    for key in ["step_file", "factory_profile", "material", "output_dir",
                "resolution", "confidence", "timestamp",
                "phases_completed", "phases_skipped", "phase_outputs",
                "total_duration_sec", "warnings", "summary"]:
        assert key in manifest, f"Missing manifest key: {key}"


def test_create_manifest_paths_are_absolute(tmp_path):
    args = make_args(output=str(tmp_path))
    path = str(tmp_path / "pipeline_manifest.json")
    manifest = _load_or_create_manifest(args, path)
    assert os.path.isabs(manifest["step_file"])
    assert os.path.isabs(manifest["output_dir"])


def test_load_existing_manifest(tmp_path):
    args = make_args(output=str(tmp_path), resume_from=3)
    path = str(tmp_path / "pipeline_manifest.json")
    existing = {"step_file": "/some/part.stp", "sentinel": True}
    with open(path, "w") as f:
        json.dump(existing, f)
    loaded = _load_or_create_manifest(args, path)
    assert loaded["sentinel"] is True


# ── _collect_existing_paths ───────────────────────────────────────────────────

def test_collect_paths_empty_dir(tmp_path):
    args = make_args(output=str(tmp_path))
    paths = _collect_existing_paths(args)
    assert paths == {}


def test_collect_paths_finds_existing_files(tmp_path):
    (tmp_path / "metadata.json").write_text("{}")
    (tmp_path / f"voxel_{DEFAULT_RESOLUTION}.npy").write_text("")
    args = make_args(output=str(tmp_path))
    paths = _collect_existing_paths(args)
    assert "metadata_file" in paths
    assert "voxel_file" in paths


# ── _update_paths_from_cache ──────────────────────────────────────────────────

def test_update_paths_phase1(tmp_path):
    args = make_args(output=str(tmp_path))
    paths = {}
    _update_paths_from_cache(1, args, paths)
    assert "voxel_file" in paths
    assert "metadata_file" in paths
    assert "mesh_file" in paths


def test_update_paths_phase2(tmp_path):
    args = make_args(output=str(tmp_path))
    paths = {}
    _update_paths_from_cache(2, args, paths)
    assert "features_file" in paths


# ── dry_run ───────────────────────────────────────────────────────────────────

def test_dry_run_prints_without_creating_files(tmp_path, capsys):
    args = make_args(
        output=str(tmp_path),
        step_file="tests/fixtures/simple_block.stp",
        factory_profile="factory_profiles/nash_nz.json",
    )
    dry_run(args)
    captured = capsys.readouterr()
    assert "DRY RUN" in captured.out
    # No files should be written
    assert list(tmp_path.iterdir()) == []


def test_dry_run_reports_missing_step_file(tmp_path, capsys):
    args = make_args(
        output=str(tmp_path),
        step_file="nonexistent_part.stp",
        factory_profile="factory_profiles/nash_nz.json",
    )
    dry_run(args)
    captured = capsys.readouterr()
    assert "not found" in captured.out.lower() or "error" in captured.out.lower()


# ── print_summary ─────────────────────────────────────────────────────────────

def test_print_summary_accept(capsys):
    manifest = {
        "summary": {
            "recommendation": "ACCEPT",
            "total_cost": 229.73,
            "currency": "NZD",
            "total_time_min": 97.7,
            "operation_count": 13,
            "setup_count": 2,
            "axis_requirement": 3,
            "flags": [],
        },
        "total_duration_sec": 4.6,
        "warnings": [],
    }
    print_summary(manifest)
    captured = capsys.readouterr()
    assert "ACCEPT" in captured.out
    assert "229.73" in captured.out
    assert "97.7" in captured.out


def test_print_summary_reject_shows_flags(capsys):
    manifest = {
        "summary": {
            "recommendation": "REJECT",
            "total_cost": 0.0,
            "currency": "NZD",
            "total_time_min": 45.0,
            "operation_count": 8,
            "setup_count": 1,
            "axis_requirement": 5,
            "flags": ["Part requires 5-axis. No 5-axis machine available."],
        },
        "total_duration_sec": 3.1,
        "warnings": [],
    }
    print_summary(manifest)
    captured = capsys.readouterr()
    assert "REJECT" in captured.out
    assert "5-axis" in captured.out


# ── End-to-end pipeline run ───────────────────────────────────────────────────

@pytest.mark.skipif(
    not os.path.exists(STP_FILE) or not os.path.exists(FACTORY),
    reason="Test fixtures or factory profile not available"
)
def test_full_pipeline_simple_block(tmp_path):
    """
    Run the complete 6-phase pipeline on simple_block.stp.
    Uses fallback features (no model checkpoint).
    """
    from run_pipeline import run_pipeline

    args = make_args(
        step_file=STP_FILE,
        factory_profile=FACTORY,
        material="aluminium_6061",
        output=str(tmp_path),
        model=None,
        quiet=True,
    )
    manifest = run_pipeline(args)

    # All 6 phases should be completed
    assert set(manifest["phases_completed"]) == {1, 2, 3, 4, 5, 6}

    # Output files should exist
    assert os.path.exists(tmp_path / "pipeline_manifest.json")
    assert os.path.exists(tmp_path / "quotation.json")
    assert os.path.exists(tmp_path / f"voxel_{DEFAULT_RESOLUTION}.npy")

    # Summary should be populated
    s = manifest["summary"]
    assert s["recommendation"] in ("ACCEPT", "REJECT")
    assert s["total_time_min"] is not None and s["total_time_min"] > 0
    assert s["operation_count"] is not None and s["operation_count"] > 0


@pytest.mark.skipif(
    not os.path.exists(STP_FILE) or not os.path.exists(FACTORY),
    reason="Test fixtures or factory profile not available"
)
def test_resume_skips_completed_phases(tmp_path):
    """
    Running the pipeline twice should skip all phases on the second run
    because all output files already exist.
    """
    from run_pipeline import run_pipeline

    args = make_args(
        step_file=STP_FILE,
        factory_profile=FACTORY,
        output=str(tmp_path),
        quiet=True,
    )
    run_pipeline(args)  # first run

    manifest2 = run_pipeline(args)  # second run
    # Second run: all phases either completed or cached
    assert len(manifest2["phases_completed"]) == 6
```

---
---

# QUICK-START COMMANDS

```bash
# Full pipeline on simple_block fixture
python run_pipeline.py \
    tests/fixtures/simple_block.stp \
    factory_profiles/nash_nz.json

# Full pipeline with trained model
python run_pipeline.py \
    tests/fixtures/complex_prismatic.stp \
    factory_profiles/nash_nz.json \
    --model checkpoints/best.pt \
    --material mild_steel \
    --output data/processed/complex_prismatic/

# Dry run to validate inputs
python run_pipeline.py \
    tests/fixtures/simple_block.stp \
    factory_profiles/nash_nz.json \
    --dry-run

# Resume from Phase 4 (Phases 1–3 already done)
python run_pipeline.py \
    tests/fixtures/simple_block.stp \
    factory_profiles/nash_nz.json \
    --output data/processed/simple_block/ \
    --resume-from 4

# Run pipeline tests
pytest tests/test_pipeline.py -v

# Run full suite
pytest tests/ -v
```

---
---

# ACCEPTANCE CRITERIA

- [ ] `pytest tests/test_pipeline.py -v` — all tests pass
- [ ] `pytest tests/ -v` — all phases pass (≥200 total)
- [ ] `python run_pipeline.py tests/fixtures/simple_block.stp factory_profiles/nash_nz.json`
      completes without error and prints an ACCEPT/REJECT summary
- [ ] `pipeline_manifest.json` is created with all 6 `phases_completed`
- [ ] All 10 output files exist in the output directory after a full run
- [ ] Running the same command a second time detects cached outputs
      and skips all phases without recomputing
- [ ] `--dry-run` prints the execution plan and exits without writing any files
- [ ] `--resume-from 4` skips Phases 1–3 and runs Phases 4–6 only
- [ ] No trained model → fallback warning printed, pipeline still completes
- [ ] Missing STEP file → clear error message and exit code 1

---
---

# NOTES FOR CODEX

1. **All helpers must be importable directly.** Tests import
   `phase_is_complete`, `_default_features`, `_build_summary`, etc.
   as module-level functions. Do not make them methods of a class or
   nest them inside `run_pipeline()`.

2. **The `--resume-from` logic is phase number, not index.**
   `--resume-from 1` means run all phases. `--resume-from 4` skips
   Phases 1, 2, 3 entirely and runs from Phase 4 onwards. Phases before
   `resume_from` go into `phases_skipped`, not `phases_completed`.

3. **Phase caching is separate from resume-from.** A phase is skipped
   as cached when `phase_is_complete()` returns True AND the phase is
   not before the resume point. A resumed phase that already has output
   files still skips (it's already done from a previous partial run).

4. **The manifest is written after every phase**, not just at the end.
   This lets the user inspect progress and resume cleanly if the process
   is interrupted mid-pipeline.

5. **Phase 2 fallback must not raise.** If `args.model` is None or the
   file doesn't exist, `run_phase2` must still return a valid features
   dict and record `"model_used": null` in the manifest. The fallback
   warning must appear in `manifest["warnings"]` and in terminal output.

6. **Default output directory derivation.** If `--output` is not given,
   derive from the STEP filename stem:
   `data/processed/{stem}/` where `stem = Path(step_file).stem`.

7. **Exit codes.** Input validation errors → `sys.exit(1)`.
   Successful completion → implicit `sys.exit(0)`.
   Pipeline phase failure → print traceback + `sys.exit(1)`.

8. **The `--quiet` flag** suppresses per-phase progress lines but not
   the final summary table. `print_summary` always runs regardless of
   `args.quiet`.
