import argparse
import json
import os

import pytest

from run_pipeline import (
    DEFAULT_CONFIDENCE,
    DEFAULT_MATERIAL,
    DEFAULT_RESOLUTION,
    PHASE_NAMES,
    PHASE_OUTPUT_FILES,
    _build_summary,
    _collect_existing_paths,
    _load_or_create_manifest,
    _resolve_model_path,
    _update_paths_from_cache,
    dry_run,
    phase_is_complete,
    print_summary,
)


CLI_DIR = "data/processed/simple_block_cli"
STP_FILE = "tests/fixtures/simple_block.stp"
FACTORY = "factory_profiles/nash_nz.json"


def make_args(**kwargs) -> argparse.Namespace:
    defaults = {
        "step_file": STP_FILE,
        "factory_profile": FACTORY,
        "material": DEFAULT_MATERIAL,
        "output": "/tmp/rpp_test_output",
        "model": None,
        "resolution": DEFAULT_RESOLUTION,
        "confidence": DEFAULT_CONFIDENCE,
        "resume_from": 1,
        "dry_run": False,
        "quiet": True,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_phase_names_has_all_six():
    assert set(PHASE_NAMES.keys()) == {1, 2, 3, 4, 5, 6}


def test_phase_output_files_has_all_six():
    assert set(PHASE_OUTPUT_FILES.keys()) == {1, 2, 3, 4, 5, 6}


def test_phase_output_files_all_non_empty():
    for phase, files in PHASE_OUTPUT_FILES.items():
        assert len(files) >= 1, f"Phase {phase} has no output files"


def test_phase_is_complete_false_empty_dir(tmp_path):
    assert not phase_is_complete(1, str(tmp_path))


def test_phase_is_complete_true_when_all_exist(tmp_path):
    for filename in PHASE_OUTPUT_FILES[1]:
        actual = filename.replace("voxel_64.npy", f"voxel_{DEFAULT_RESOLUTION}.npy")
        (tmp_path / actual).write_text("dummy")
    assert phase_is_complete(1, str(tmp_path))


def test_phase_is_complete_false_partial(tmp_path):
    (tmp_path / "voxel_64.npy").write_text("dummy")
    assert not phase_is_complete(1, str(tmp_path))


def test_resolve_model_explicit(tmp_path):
    ckpt = tmp_path / "my_model.pt"
    ckpt.write_text("dummy")
    args = make_args(model=str(ckpt))
    assert _resolve_model_path(args) == str(ckpt.resolve())


def test_resolve_model_auto_detects_best_pt(tmp_path, monkeypatch):
    import run_pipeline

    ckpt = tmp_path / "best.pt"
    ckpt.write_text("dummy")
    monkeypatch.setattr(run_pipeline, "DEFAULT_CHECKPOINT", str(ckpt))
    args = make_args(model=None)
    resolved = _resolve_model_path(args)
    assert resolved is not None
    assert "best.pt" in resolved


def test_resolve_model_returns_none_when_missing(monkeypatch):
    import run_pipeline

    monkeypatch.setattr(run_pipeline, "DEFAULT_CHECKPOINT", "/no/such/path.pt")
    args = make_args(model=None)
    assert _resolve_model_path(args) is None


def test_default_features_removed():
    import run_pipeline

    assert not hasattr(run_pipeline, "_default_features")


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


def test_create_manifest_has_required_keys(tmp_path):
    args = make_args(output=str(tmp_path))
    path = str(tmp_path / "pipeline_manifest.json")
    manifest = _load_or_create_manifest(args, path)
    for key in [
        "step_file",
        "factory_profile",
        "material",
        "output_dir",
        "resolution",
        "confidence",
        "timestamp",
        "phases_completed",
        "phases_skipped",
        "phase_outputs",
        "total_duration_sec",
        "warnings",
        "summary",
    ]:
        assert key in manifest


def test_create_manifest_paths_are_absolute(tmp_path):
    args = make_args(output=str(tmp_path))
    path = str(tmp_path / "pipeline_manifest.json")
    manifest = _load_or_create_manifest(args, path)
    assert os.path.isabs(manifest["step_file"])
    assert os.path.isabs(manifest["output_dir"])


def test_load_existing_manifest(tmp_path):
    args = make_args(output=str(tmp_path), resume_from=3)
    path = str(tmp_path / "pipeline_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"step_file": "/some/part.stp", "sentinel": True}, f)
    loaded = _load_or_create_manifest(args, path)
    assert loaded["sentinel"] is True


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


def test_update_paths_phase4_includes_simulation_input(tmp_path):
    args = make_args(output=str(tmp_path))
    paths = {}
    _update_paths_from_cache(4, args, paths)
    assert "process_plan_file" in paths
    assert "simulation_input_file" in paths


def test_dry_run_prints_without_creating_files(tmp_path, capsys):
    args = make_args(
        output=str(tmp_path),
        step_file="tests/fixtures/simple_block.stp",
        factory_profile="factory_profiles/nash_nz.json",
    )
    dry_run(args)
    captured = capsys.readouterr()
    assert "DRY RUN" in captured.out
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


def test_print_summary_review(capsys):
    manifest = {
        "summary": {
            "recommendation": "REVIEW",
            "total_cost": 120.0,
            "currency": "NZD",
            "total_time_min": 45.0,
            "operation_count": 8,
            "setup_count": 1,
            "axis_requirement": 3,
            "flags": ["Part violates the single-setup +Z 2.5D machining baseline."],
        },
        "total_duration_sec": 3.1,
        "warnings": [],
    }
    print_summary(manifest)
    captured = capsys.readouterr()
    assert "REVIEW" in captured.out
    assert "2.5D" in captured.out


@pytest.mark.skipif(
    not os.path.exists(STP_FILE) or not os.path.exists(FACTORY),
    reason="Test fixtures or factory profile not available",
)
def test_full_pipeline_simple_block(tmp_path):
    from run_pipeline import run_pipeline
    import torch
    from models.feature_net import FeatureNet3D, NUM_CLASSES

    ckpt = tmp_path / "model.pt"
    model = FeatureNet3D(num_classes=NUM_CLASSES)
    model.eval()
    torch.save({"model_state_dict": model.state_dict()}, ckpt)
    args = make_args(
        step_file=STP_FILE,
        factory_profile=FACTORY,
        material="aluminium_6061",
        output=str(tmp_path),
        model=str(ckpt),
        confidence=0.0,
        quiet=True,
    )
    manifest = run_pipeline(args)
    assert set(manifest["phases_completed"]) == {1, 2, 3, 4, 5, 6}
    assert os.path.exists(tmp_path / "pipeline_manifest.json")
    assert os.path.exists(tmp_path / "quotation.json")
    assert os.path.exists(tmp_path / f"voxel_{DEFAULT_RESOLUTION}.npy")
    summary = manifest["summary"]
    assert summary["recommendation"] in ("ACCEPT", "REVIEW", "REJECT")
    assert summary["total_time_min"] is not None and summary["total_time_min"] > 0
    assert summary["operation_count"] is not None and summary["operation_count"] > 0


@pytest.mark.skipif(
    not os.path.exists(STP_FILE) or not os.path.exists(FACTORY),
    reason="Test fixtures or factory profile not available",
)
def test_resume_skips_completed_phases(tmp_path):
    from run_pipeline import run_pipeline
    import torch
    from models.feature_net import FeatureNet3D, NUM_CLASSES

    ckpt = tmp_path / "model.pt"
    model = FeatureNet3D(num_classes=NUM_CLASSES)
    model.eval()
    torch.save({"model_state_dict": model.state_dict()}, ckpt)
    args = make_args(
        step_file=STP_FILE,
        factory_profile=FACTORY,
        output=str(tmp_path),
        model=str(ckpt),
        confidence=0.0,
        quiet=True,
    )
    run_pipeline(args)
    manifest2 = run_pipeline(args)
    assert len(manifest2["phases_completed"]) == 6
