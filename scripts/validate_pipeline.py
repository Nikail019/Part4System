"""Validate the full RPP pipeline across fixture STEP files."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys


FIXTURES = {
    "simple_block": "tests/fixtures/simple_block.stp",
    "block_with_holes": "tests/fixtures/block_with_holes.stp",
    "complex_prismatic": "tests/fixtures/complex_prismatic.stp",
}


def _load_json(output_dir: str, filename: str) -> dict:
    path = os.path.join(output_dir, filename)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_feature_types(output_dir: str) -> list[str]:
    data = _load_json(output_dir, "features.json")
    return [feature["type"] for feature in data.get("features", [])]


CHECKS = [
    ("phase1", "voxel_32.npy exists", lambda d: os.path.exists(os.path.join(d, "voxel_32.npy"))),
    ("phase2", "flat_face always detected", lambda d: "flat_face" in _load_feature_types(d)),
    ("phase2", "at least 1 feature detected", lambda d: _load_json(d, "features.json").get("feature_count", 0) >= 1),
    ("phase3", "single 2.5D setup", lambda d: _load_json(d, "setup_analysis.json").get("setup_count") == 1),
    ("phase3", "3-axis baseline", lambda d: _load_json(d, "setup_analysis.json").get("axis_requirement") == 3),
    ("phase4", "at least 2 operations", lambda d: _load_json(d, "process_plan.json").get("operation_count", 0) >= 2),
    ("phase5", "positive machining time", lambda d: _load_json(d, "time_estimate.json").get("total_time_min", 0) > 0),
    ("phase6", "valid recommendation", lambda d: _load_json(d, "quotation.json").get("recommendation") in ("ACCEPT", "REVIEW", "REJECT")),
    ("phase6", "positive cost", lambda d: _load_json(d, "quotation.json").get("estimated_cost", {}).get("total", 0) > 0),
]


def run_fixture(
    name: str,
    stp_path: str,
    output_dir: str,
    model_path: str,
    factory_path: str,
    material: str,
) -> dict:
    del name
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        sys.executable,
        "run_pipeline.py",
        stp_path,
        factory_path,
        "--model",
        model_path,
        "--material",
        material,
        "--output",
        output_dir,
        "--quiet",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return {
            "status": "FAIL",
            "error": proc.stderr.strip(),
            "checks_passed": 0,
            "checks_total": len(CHECKS),
            "failed_checks": ["Pipeline returned non-zero exit code"],
        }

    passed = []
    failed = []
    for phase, description, check_fn in CHECKS:
        try:
            ok = check_fn(output_dir)
        except Exception:
            ok = False
        (passed if ok else failed).append(f"[{phase}] {description}")

    quotation = _load_json(output_dir, "quotation.json")
    plan = _load_json(output_dir, "process_plan.json")
    setup = _load_json(output_dir, "setup_analysis.json")
    time_estimate = _load_json(output_dir, "time_estimate.json")
    return {
        "status": "PASS" if not failed else "FAIL",
        "checks_passed": len(passed),
        "checks_total": len(CHECKS),
        "features": _load_feature_types(output_dir),
        "operation_count": plan.get("operation_count"),
        "setup_count": setup.get("setup_count"),
        "axis_requirement": setup.get("axis_requirement"),
        "total_time_min": time_estimate.get("total_time_min"),
        "total_cost": quotation.get("estimated_cost", {}).get("total"),
        "currency": quotation.get("estimated_cost", {}).get("currency"),
        "recommendation": quotation.get("recommendation"),
        "failed_checks": failed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--factory", default="factory_profiles/nash_nz.json")
    parser.add_argument("--material", default="aluminium_6061")
    parser.add_argument("--out", default=os.path.join("data", "validation"))
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    results = {}
    overall_pass = True
    print(f"\nValidating pipeline with model: {args.model}")
    print(f"Factory: {args.factory}  Material: {args.material}\n")
    for name, stp_path in FIXTURES.items():
        if not os.path.exists(stp_path):
            print(f"  {name:<20} SKIP (fixture not found)")
            continue
        fixture_out = os.path.join(args.out, name)
        print(f"  {name:<20} ...", end=" ", flush=True)
        result = run_fixture(name, stp_path, fixture_out, args.model, args.factory, args.material)
        results[name] = result
        if result["status"] == "PASS":
            print(
                f"PASS  {result.get('operation_count')} ops  "
                f"{result.get('total_time_min')} min  {result.get('currency')} "
                f"{result.get('total_cost')}  [{result.get('recommendation')}]"
            )
        else:
            print("FAIL")
            overall_pass = False
            for failed in result.get("failed_checks", []):
                print(f"         {failed}")

    report = {
        "model_path": os.path.abspath(args.model),
        "factory": args.factory,
        "material": args.material,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "fixtures": results,
        "overall_status": "PASS" if overall_pass else "FAIL",
        "fixtures_passed": sum(1 for result in results.values() if result["status"] == "PASS"),
        "fixtures_total": len(results),
    }
    report_path = os.path.join(args.out, "validation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"\nOverall: {report['overall_status']} ({report['fixtures_passed']}/{report['fixtures_total']} fixtures)")
    print(f"Report:  {report_path}\n")
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
