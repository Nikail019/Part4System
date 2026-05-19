"""End-to-end RPP pipeline runner: STEP file to quotation."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
import tempfile


DEFAULT_RESOLUTION = 32
DEFAULT_MATERIAL = "aluminium_6061"
DEFAULT_CONFIDENCE = 0.5
DEFAULT_SETUP_TIME = 15.0
DEFAULT_TOOL_CHANGE = 2.0
DEFAULT_CHECKPOINT = "checkpoints/best.pt"

PHASE_NAMES = {
    1: "STEP -> Voxel",
    2: "Voxel -> Features",
    3: "Setup Analysis",
    4: "Process Plan",
    5: "Time Estimate",
    6: "Quotation",
}

PHASE_OUTPUT_FILES = {
    1: ["voxel_64.npy", "metadata.json", "mesh.stl"],
    2: ["features.json", "pmi_data.json"],
    3: ["setup_analysis.json", "accessibility_map.npy", "surface_mask.npy"],
    4: ["process_plan.json"],
    5: ["time_estimate.json"],
    6: ["quotation.json"],
}

PHASE_PATH_KEYS = {
    1: ["voxel_file", "metadata_file", "mesh_file"],
    2: ["features_file", "pmi_data_file"],
    3: ["setup_analysis_file"],
    4: ["process_plan_file"],
    5: ["time_estimate_file"],
    6: ["quotation_file"],
}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RPP Pipeline: STEP file -> job quotation",
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
    parser.add_argument("step_file", help="Path to input STEP (.stp / .step) file")
    parser.add_argument("factory_profile", help="Path to factory profile JSON")
    parser.add_argument("--material", default=DEFAULT_MATERIAL)
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--device",
        default=None,
        help="Training device override: mps / cuda / cpu (default: auto)",
    )
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--resume-from", type=int, default=1, dest="resume_from", choices=range(1, 7))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _resolve_output(args: argparse.Namespace) -> None:
    if args.output is None:
        stem = os.path.splitext(os.path.basename(args.step_file))[0]
        args.output = os.path.join("data", "processed", stem)


def phase_is_complete(phase: int, output_dir: str, resolution: int = DEFAULT_RESOLUTION) -> bool:
    for filename in PHASE_OUTPUT_FILES[phase]:
        actual = filename.replace("voxel_64.npy", f"voxel_{resolution}.npy")
        if not os.path.exists(os.path.join(output_dir, actual)):
            return False
    return True


def _write_json_atomic(data: dict, path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=directory, delete=False, suffix=".tmp") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
        tmp = f.name
    os.replace(tmp, path)


def _resolve_model_path(args: argparse.Namespace) -> str | None:
    if args.model and os.path.exists(args.model):
        return os.path.abspath(args.model)
    if os.path.exists(DEFAULT_CHECKPOINT):
        return os.path.abspath(DEFAULT_CHECKPOINT)
    return None


def run_phase1(args: argparse.Namespace, paths: dict) -> dict:
    del paths
    from phase1_voxeliser import process_step_file

    t0 = time.time()
    result = process_step_file(args.step_file, args.output, resolution=args.resolution)
    return {
        "voxel_file": result["voxel_file"],
        "metadata_file": os.path.join(os.path.abspath(args.output), "metadata.json"),
        "mesh_file": result["mesh_file"],
        "duration_sec": round(time.time() - t0, 2),
    }


def run_phase2(args: argparse.Namespace, paths: dict) -> dict:
    from phase2_feature_recognition import recognise_features
    from step_pmi_extractor import extract_pmi

    t0 = time.time()
    voxel_file = paths["voxel_file"]
    features_path = os.path.join(os.path.abspath(args.output), "features.json")
    model_used = _resolve_model_path(args)
    if model_used is None:
        raise FileNotFoundError(
            "No trained model checkpoint found.\n"
            "Train the model first:\n"
            "  python training/synthetic_data_gen.py --count 2000\n"
            "  python training/train_feature_net.py --data data/raw/synthetic "
            "--out checkpoints --epochs 30\n"
            "Or specify a checkpoint with --model path/to/model.pt"
        )
    result = recognise_features(voxel_file, model_used, threshold=args.confidence)
    _write_json_atomic(result, features_path)
    pmi_result = extract_pmi(
        args.step_file,
        features_path,
        args.output,
        default_material=args.material,
    )
    warnings = list(pmi_result.get("warnings", []))
    return {
        "features_file": features_path,
        "pmi_data_file": pmi_result["pmi_data_file"],
        "feature_count": result["feature_count"],
        "model_used": model_used,
        "warnings": warnings,
        "duration_sec": round(time.time() - t0, 2),
    }


def run_phase3(args: argparse.Namespace, paths: dict) -> dict:
    from phase3_setup_analysis import analyse_setups

    t0 = time.time()
    result = analyse_setups(paths["voxel_file"], args.output, features_path=paths.get("features_file"))
    return {
        "setup_analysis_file": os.path.join(os.path.abspath(args.output), "setup_analysis.json"),
        "setup_count": result["setup_count"],
        "axis_requirement": result["axis_requirement"],
        "duration_sec": round(time.time() - t0, 2),
    }


def run_phase4(args: argparse.Namespace, paths: dict) -> dict:
    from phase4_process_plan import generate_process_plan

    t0 = time.time()
    result = generate_process_plan(
        paths["metadata_file"],
        paths["features_file"],
        paths["setup_analysis_file"],
        args.output,
        confidence_threshold=args.confidence,
        pmi_data_path=paths.get("pmi_data_file"),
    )
    return {
        "process_plan_file": result["process_plan_file"],
        "operation_count": result["operation_count"],
        "duration_sec": round(time.time() - t0, 2),
    }


def run_phase5(args: argparse.Namespace, paths: dict) -> dict:
    from phase5_time_estimate import estimate_time

    t0 = time.time()
    result = estimate_time(
        paths["process_plan_file"],
        paths["metadata_file"],
        args.output,
        material=args.material,
        setup_time_min=DEFAULT_SETUP_TIME,
        tool_change_time_min=DEFAULT_TOOL_CHANGE,
    )
    return {
        "time_estimate_file": result["time_estimate_file"],
        "total_time_min": result["total_time_min"],
        "duration_sec": round(time.time() - t0, 2),
    }


def run_phase6(args: argparse.Namespace, paths: dict) -> dict:
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
        "quotation_file": result["quotation_file"],
        "recommendation": result["recommendation"],
        "total_cost": result["estimated_cost"]["total"],
        "currency": result["estimated_cost"]["currency"],
        "duration_sec": round(time.time() - t0, 2),
    }


PHASE_RUNNERS = {
    1: run_phase1,
    2: run_phase2,
    3: run_phase3,
    4: run_phase4,
    5: run_phase5,
    6: run_phase6,
}


def _load_or_create_manifest(args: argparse.Namespace, path: str) -> dict:
    if args.resume_from > 1 and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {
        "step_file": os.path.abspath(args.step_file),
        "factory_profile": os.path.abspath(args.factory_profile),
        "material": args.material,
        "model_path": os.path.abspath(args.model) if args.model else None,
        "output_dir": os.path.abspath(args.output),
        "resolution": args.resolution,
        "confidence": args.confidence,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "phases_completed": [],
        "phases_skipped": [],
        "phase_outputs": {},
        "total_duration_sec": 0.0,
        "warnings": [],
        "summary": {},
    }


def _collect_existing_paths(args: argparse.Namespace) -> dict:
    output = os.path.abspath(args.output)
    candidates = {
        "voxel_file": os.path.join(output, f"voxel_{args.resolution}.npy"),
        "metadata_file": os.path.join(output, "metadata.json"),
        "mesh_file": os.path.join(output, "mesh.stl"),
        "features_file": os.path.join(output, "features.json"),
        "pmi_data_file": os.path.join(output, "pmi_data.json"),
        "setup_analysis_file": os.path.join(output, "setup_analysis.json"),
        "process_plan_file": os.path.join(output, "process_plan.json"),
        "time_estimate_file": os.path.join(output, "time_estimate.json"),
        "quotation_file": os.path.join(output, "quotation.json"),
    }
    return {key: path for key, path in candidates.items() if os.path.exists(path)}


def _update_paths_from_cache(phase: int, args: argparse.Namespace, paths: dict) -> None:
    output = os.path.abspath(args.output)
    additions = {
        1: {
            "voxel_file": os.path.join(output, f"voxel_{args.resolution}.npy"),
            "metadata_file": os.path.join(output, "metadata.json"),
            "mesh_file": os.path.join(output, "mesh.stl"),
        },
        2: {
            "features_file": os.path.join(output, "features.json"),
            "pmi_data_file": os.path.join(output, "pmi_data.json"),
        },
        3: {"setup_analysis_file": os.path.join(output, "setup_analysis.json")},
        4: {"process_plan_file": os.path.join(output, "process_plan.json")},
        5: {"time_estimate_file": os.path.join(output, "time_estimate.json")},
        6: {"quotation_file": os.path.join(output, "quotation.json")},
    }
    paths.update(additions.get(phase, {}))


def _read_json_if_exists(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _apply_pmi_material_override(args: argparse.Namespace, pmi_data_file: str | None) -> list[str]:
    pmi_data = _read_json_if_exists(pmi_data_file)
    if pmi_data.get("material_source") != "pmi":
        return []
    pmi_material = pmi_data.get("material")
    if not pmi_material or pmi_material == args.material:
        return []
    warning = f"PMI material '{pmi_material}' overrides requested material '{args.material}'."
    args.material = pmi_material
    return [warning]


def _build_summary(manifest: dict) -> dict:
    outputs = manifest.get("phase_outputs", {})
    p6 = outputs.get("6", {})
    p5 = outputs.get("5", {})
    p4 = outputs.get("4", {})
    p3 = outputs.get("3", {})
    flags = []
    quotation_file = p6.get("quotation_file")
    quotation = _read_json_if_exists(quotation_file)
    if quotation:
        flags = quotation.get("flags", [])
    return {
        "recommendation": p6.get("recommendation"),
        "total_cost": p6.get("total_cost"),
        "currency": p6.get("currency"),
        "total_time_min": p5.get("total_time_min"),
        "operation_count": p4.get("operation_count"),
        "setup_count": p3.get("setup_count"),
        "axis_requirement": p3.get("axis_requirement"),
        "flags": flags,
    }


def _phase_cached_result(phase: int, args: argparse.Namespace, paths: dict) -> dict:
    _update_paths_from_cache(phase, args, paths)
    output = {}
    if phase == 1:
        output = {
            "voxel_file": paths["voxel_file"],
            "metadata_file": paths["metadata_file"],
            "mesh_file": paths["mesh_file"],
        }
    elif phase == 2:
        features = _read_json_if_exists(paths["features_file"])
        output = {
            "features_file": paths["features_file"],
            "pmi_data_file": paths.get("pmi_data_file"),
            "feature_count": features.get("feature_count", 0),
            "model_used": features.get("model_path"),
        }
    elif phase == 3:
        setup = _read_json_if_exists(paths["setup_analysis_file"])
        output = {
            "setup_analysis_file": paths["setup_analysis_file"],
            "setup_count": setup.get("setup_count"),
            "axis_requirement": setup.get("axis_requirement"),
        }
    elif phase == 4:
        plan = _read_json_if_exists(paths["process_plan_file"])
        output = {
            "process_plan_file": paths["process_plan_file"],
            "operation_count": plan.get("operation_count"),
        }
    elif phase == 5:
        estimate = _read_json_if_exists(paths["time_estimate_file"])
        output = {
            "time_estimate_file": paths["time_estimate_file"],
            "total_time_min": estimate.get("total_time_min"),
        }
    elif phase == 6:
        quote = _read_json_if_exists(paths["quotation_file"])
        output = {
            "quotation_file": paths["quotation_file"],
            "recommendation": quote.get("recommendation"),
            "total_cost": quote.get("estimated_cost", {}).get("total"),
            "currency": quote.get("estimated_cost", {}).get("currency"),
        }
    return {"status": "completed", "duration_sec": 0.0, **output}


def run_pipeline(args: argparse.Namespace) -> dict:
    _resolve_output(args)
    pipeline_start = time.time()
    os.makedirs(args.output, exist_ok=True)
    manifest_path = os.path.join(args.output, "pipeline_manifest.json")
    manifest = _load_or_create_manifest(args, manifest_path)
    manifest["phases_completed"] = []
    manifest["phases_skipped"] = []
    paths = _collect_existing_paths(args)
    warnings = list(dict.fromkeys(manifest.get("warnings", [])))

    for phase_num in range(1, 7):
        phase_name = PHASE_NAMES[phase_num]

        if phase_num < args.resume_from:
            if not args.quiet:
                print(f"  Phase {phase_num} [{phase_name}] - skipped (before resume point)")
            manifest["phases_skipped"].append(phase_num)
            _update_paths_from_cache(phase_num, args, paths)
            continue

        if phase_is_complete(phase_num, args.output, args.resolution):
            if not args.quiet:
                print(f"  Phase {phase_num} [{phase_name}] - cached")
            phase_output = _phase_cached_result(phase_num, args, paths)
            if phase_num == 2:
                for warning in _apply_pmi_material_override(args, paths.get("pmi_data_file")):
                    if warning not in warnings:
                        warnings.append(warning)
                manifest["material"] = args.material
            manifest["phase_outputs"][str(phase_num)] = phase_output
            manifest["phases_completed"].append(phase_num)
            continue

        if not args.quiet:
            print(f"  Phase {phase_num} [{phase_name}] ...", end=" ", flush=True)
        phase_result = PHASE_RUNNERS[phase_num](args, paths)
        for warning in phase_result.get("warnings", []):
            if warning not in warnings:
                warnings.append(warning)
        if phase_num == 2:
            for warning in _apply_pmi_material_override(args, phase_result.get("pmi_data_file")):
                if warning not in warnings:
                    warnings.append(warning)
            manifest["material"] = args.material
        for key in PHASE_PATH_KEYS[phase_num]:
            if key in phase_result:
                paths[key] = phase_result[key]

        duration = phase_result.get("duration_sec", 0.0)
        if not args.quiet:
            print(f"done ({duration:.2f}s)")

        manifest["phase_outputs"][str(phase_num)] = {"status": "completed", **phase_result}
        manifest["phases_completed"].append(phase_num)
        manifest["warnings"] = list(dict.fromkeys(warnings))
        _write_json_atomic(manifest, manifest_path)

    manifest["total_duration_sec"] = round(time.time() - pipeline_start, 2)
    manifest["warnings"] = list(dict.fromkeys(warnings))
    manifest["summary"] = _build_summary(manifest)
    _write_json_atomic(manifest, manifest_path)
    return manifest


def print_summary(manifest: dict) -> None:
    summary = manifest.get("summary", {})
    recommendation = summary.get("recommendation", "UNKNOWN")
    symbol = "OK" if recommendation == "ACCEPT" else "NO"
    currency = summary.get("currency", "")
    cost = summary.get("total_cost")
    time_min = summary.get("total_time_min")
    operations = summary.get("operation_count")
    setups = summary.get("setup_count")
    axes = summary.get("axis_requirement")
    total_sec = manifest.get("total_duration_sec", 0.0)

    print()
    print("=" * 52)
    print("  RPP PIPELINE RESULT")
    print("=" * 52)
    print(f"  Recommendation : {symbol}  {recommendation}")
    if cost is not None:
        print(f"  Estimated cost : {currency} {cost:.2f}")
    if time_min is not None:
        print(f"  Estimated time : {time_min:.1f} min")
    if operations is not None:
        print(f"  Operations     : {operations}")
    if setups is not None:
        print(f"  Setups         : {setups}  ({axes}-axis)")
    print(f"  Pipeline time  : {total_sec:.1f}s")
    print("=" * 52)

    flags = summary.get("flags", []) or manifest.get("warnings", [])
    if flags:
        print()
        for flag in flags:
            print(f"  WARNING: {flag}")
    print()


def dry_run(args: argparse.Namespace) -> None:
    _resolve_output(args)
    print()
    print("DRY RUN - no files will be written")
    print()
    print(f"  Input STEP       : {args.step_file}")
    print(f"  Factory profile  : {args.factory_profile}")
    print(f"  Material         : {args.material}")
    print(f"  Output dir       : {args.output}")
    model_path = _resolve_model_path(args)
    print(f"  Model checkpoint : {model_path or '(none - run training first)'}")
    print(f"  Voxel resolution : {args.resolution}^3")
    print(f"  Confidence       : {args.confidence}")
    print(f"  Resume from      : Phase {args.resume_from}")
    print()
    print("  Phases:")
    for phase_num in range(1, 7):
        status = "SKIP (before resume)" if phase_num < args.resume_from else "RUN"
        if phase_num >= args.resume_from and phase_is_complete(phase_num, args.output, args.resolution):
            status = "SKIP (cached outputs exist)"
        print(f"    Phase {phase_num} [{PHASE_NAMES[phase_num]}] -> {status}")
    print()

    errors = []
    if not os.path.exists(args.step_file):
        errors.append(f"STEP file not found: {args.step_file}")
    if not os.path.exists(args.factory_profile):
        errors.append(f"Factory profile not found: {args.factory_profile}")
    if args.model and not os.path.exists(args.model):
        errors.append(f"Model checkpoint not found: {args.model}")
    if errors:
        print("  ERRORS:")
        for error in errors:
            print(f"    {error}")
    else:
        print("  Input validation: all inputs found")
    print()


def _validate_required_inputs(args: argparse.Namespace) -> list[str]:
    errors = []
    if not os.path.exists(args.step_file):
        errors.append(f"STEP file not found: {args.step_file}")
    if not os.path.exists(args.factory_profile):
        errors.append(f"Factory profile not found: {args.factory_profile}")
    return errors


def main() -> None:
    args = get_args()
    _resolve_output(args)

    if args.dry_run:
        dry_run(args)
        sys.exit(0)

    errors = _validate_required_inputs(args)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print()
        print("RPP Pipeline")
        print(f"  Input  : {args.step_file}")
        print(f"  Output : {args.output}")
        print()

    try:
        manifest = run_pipeline(args)
        print_summary(manifest)
    except Exception as exc:
        print(f"\nPipeline failed: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
