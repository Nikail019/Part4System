"""Web API and static workbench for RPP pipeline visualisation."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from run_pipeline import DEFAULT_CONFIDENCE, DEFAULT_MATERIAL, DEFAULT_RESOLUTION, run_pipeline


ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "data" / "processed"
VALIDATION_DIR = ROOT / "data" / "validation"
STATIC_DIR = ROOT / "web" / "static"
JOB_STATUS_FILE = "job_status.json"
SAFE_JOB_RE = re.compile(r"[^A-Za-z0-9_.-]+")
ARTIFACT_EXTENSIONS = {".json", ".npy", ".stl", ".txt"}
MAX_VIEWER_POINTS = 2500
VIEWER_SAMPLE_SEED = 42
FEATURE_COLORS = {
    "through_hole": "#5aa9ff",
    "blind_hole": "#7cc7ff",
    "rectangular_pocket": "#54b7a7",
    "circular_pocket": "#64d2a8",
    "rectangular_slot": "#f2b84b",
    "circular_slot": "#f0d264",
    "rectangular_step": "#c792ea",
    "boss": "#ee8f5f",
    "flat_face": "#9fb5c4",
    "triangular_pocket": "#6ed0e0",
    "chamfer": "#9ca8b3",
    "fillet": "#9ca8b3",
}


app = FastAPI(title="RPP MVP Workbench", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, suffix=".tmp") as f:
        json.dump(_jsonable(data), f, indent=2)
        f.write("\n")
        tmp = Path(f.name)
    os.replace(tmp, path)


def _safe_job_id(name: str) -> str:
    stem = Path(name).stem if name else "job"
    cleaned = SAFE_JOB_RE.sub("_", stem).strip("._-")
    return cleaned or "job"


def _unique_job_dir(base_id: str) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    candidate = PROCESSED_DIR / base_id
    if not candidate.exists():
        return candidate
    suffix = time.strftime("%Y%m%d_%H%M%S")
    return PROCESSED_DIR / f"{base_id}_{suffix}"


def _candidate_job_dirs() -> list[Path]:
    roots = [PROCESSED_DIR, VALIDATION_DIR]
    dirs: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        dirs.extend(path for path in root.iterdir() if path.is_dir())
    return sorted(dirs, key=lambda path: path.name)


def _job_dir(job_id: str) -> Path:
    safe = _safe_job_id(job_id)
    for root in (PROCESSED_DIR, VALIDATION_DIR):
        candidate = (root / safe).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            continue
        if candidate.exists() and candidate.is_dir():
            return candidate
    raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")


def _artifact_path(job_dir: Path, artifact_name: str) -> Path:
    path = (job_dir / artifact_name).resolve()
    try:
        path.relative_to(job_dir.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid artifact path.") from exc
    if path.suffix not in ARTIFACT_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported artifact type.")
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"Artifact not found: {artifact_name}")
    return path


def _status_for_job(job_dir: Path) -> dict:
    status = _read_json(job_dir / JOB_STATUS_FILE)
    manifest = _read_json(job_dir / "pipeline_manifest.json")
    quotation = _read_json(job_dir / "quotation.json")
    setup = _read_json(job_dir / "setup_analysis.json")
    plan = _read_json(job_dir / "process_plan.json")
    if not status:
        state = "complete" if manifest or quotation else "artifact-only"
        status = {"state": state, "job_id": job_dir.name}
    return {
        **status,
        "job_id": job_dir.name,
        "source": "validation" if VALIDATION_DIR in job_dir.parents else "processed",
        "recommendation": quotation.get("recommendation"),
        "review_codes": quotation.get("review_codes", plan.get("review_codes", setup.get("review_codes", []))),
        "setup_count": setup.get("setup_count", plan.get("setup_count")),
        "axis_requirement": setup.get("axis_requirement", plan.get("axis_requirement")),
        "operation_count": plan.get("operation_count"),
        "has_mesh": (job_dir / "mesh.stl").exists(),
        "has_voxel": any(job_dir.glob("voxel_*.npy")),
    }


def _list_artifacts(job_dir: Path) -> list[dict]:
    artifacts = []
    for path in sorted(job_dir.iterdir(), key=lambda p: p.name):
        if path.is_file() and path.suffix in ARTIFACT_EXTENSIONS:
            artifacts.append(
                {
                    "name": path.name,
                    "size_bytes": path.stat().st_size,
                    "kind": path.suffix.lstrip("."),
                    "url": f"/api/jobs/{job_dir.name}/artifacts/{path.name}",
                }
            )
    return artifacts


def _viewer_transform(metadata: dict, shape: tuple[int, ...]) -> dict:
    bbox = metadata.get("bounding_box_mm", {}) if isinstance(metadata, dict) else {}
    dims = [float(bbox.get(axis, 0.0)) for axis in ("x", "y", "z")]
    if len(shape) < 3 or max(dims, default=0.0) <= 0:
        scale = 1.0
        dims = [float(shape[0]), float(shape[1]), float(shape[2])] if len(shape) >= 3 else [1.0, 1.0, 1.0]
    else:
        scale = max(dims) / max(1.0, float(max(shape) - 2))
    return {
        "scale_mm_per_voxel": scale,
        "bbox_mm": {"x": dims[0], "y": dims[1], "z": dims[2]},
        "center_voxel": [(float(size) - 1.0) / 2.0 for size in shape[:3]],
    }


def _coords_to_viewer_points(coords: np.ndarray, transform: dict) -> np.ndarray:
    if coords.size == 0:
        return coords.astype(float)
    center = np.asarray(transform["center_voxel"], dtype=float)
    scale = float(transform["scale_mm_per_voxel"])
    return (coords.astype(float) - center) * scale


def _voxel_point_to_viewer(point: list | tuple | np.ndarray, transform: dict) -> list[float]:
    coords = np.asarray(point, dtype=float)
    center = np.asarray(transform["center_voxel"], dtype=float)
    scale = float(transform["scale_mm_per_voxel"])
    return ((coords - center) * scale).tolist()


def _feature_overlays(feature_instances: dict, metadata: dict, voxel_shape: tuple[int, ...] | None) -> list[dict]:
    if not voxel_shape:
        voxel_shape = (64, 64, 64)
    transform = _viewer_transform(metadata, voxel_shape)
    overlays = []
    for instance in feature_instances.get("instances", []):
        if instance.get("localisation_status") != "localised":
            continue
        if int(instance.get("volume_voxels", 0)) <= 0:
            continue
        bbox = instance.get("bbox_voxel")
        if not isinstance(bbox, list) or len(bbox) != 2:
            continue
        try:
            v0 = _voxel_point_to_viewer(bbox[0], transform)
            v1 = _voxel_point_to_viewer(bbox[1], transform)
        except Exception:
            continue
        mins = [min(v0[i], v1[i]) for i in range(3)]
        maxs = [max(v0[i], v1[i]) for i in range(3)]
        size = [max(0.5, maxs[i] - mins[i]) for i in range(3)]
        center = [(mins[i] + maxs[i]) / 2.0 for i in range(3)]
        centroid = instance.get("centroid_voxel")
        overlay = {
            "type": instance.get("type", "unknown"),
            "instance_id": instance.get("instance_id"),
            "confidence": instance.get("confidence"),
            "status": instance.get("localisation_status", instance.get("status")),
            "primary_direction": instance.get("primary_direction"),
            "access_class": instance.get("access_class"),
            "bbox_center": center,
            "bbox_size": size,
            "centroid": _voxel_point_to_viewer(centroid, transform) if centroid else center,
            "color": FEATURE_COLORS.get(str(instance.get("type")), "#f2b84b"),
        }
        overlays.append(overlay)
    return overlays


def _setup_overlay(setup: dict, metadata: dict, voxel_shape: tuple[int, ...] | None) -> dict:
    if not voxel_shape:
        voxel_shape = (64, 64, 64)
    transform = _viewer_transform(metadata, voxel_shape)
    bbox = transform.get("bbox_mm", {})
    dims = [float(bbox.get(axis, 0.0)) for axis in ("x", "y", "z")]
    span = max(dims + [float(max(voxel_shape))])
    z_top = dims[2] / 2.0 if dims[2] > 0 else span / 2.0
    setup_list = setup.get("setups", [])
    approach = setup_list[0].get("approach_direction", "+Z") if setup_list else "+Z"
    start = [0.0, 0.0, z_top + span * 0.35]
    end = [0.0, 0.0, z_top + span * 0.05]
    if approach == "-Z":
        start, end = [0.0, 0.0, -z_top - span * 0.35], [0.0, 0.0, -z_top - span * 0.05]
    return {
        "approach_direction": approach,
        "setup_count": setup.get("setup_count"),
        "axis_requirement": setup.get("axis_requirement"),
        "requires_rotation": setup.get("requires_rotation"),
        "two_point_five_d_compatible": setup.get("two_point_five_d_compatible"),
        "arrow_start": start,
        "arrow_end": end,
    }


def _surface_mask_from_solid(arr: np.ndarray) -> np.ndarray:
    solid = arr.astype(bool)
    if solid.ndim != 3:
        return solid
    padded = np.pad(solid, 1, mode="constant", constant_values=False)
    interior = solid.copy()
    for axis in range(3):
        lower = [slice(1, -1), slice(1, -1), slice(1, -1)]
        upper = [slice(1, -1), slice(1, -1), slice(1, -1)]
        lower[axis] = slice(0, -2)
        upper[axis] = slice(2, None)
        interior &= padded[tuple(lower)] & padded[tuple(upper)]
    return solid & ~interior


def _sample_coords(coords: np.ndarray, max_points: int) -> np.ndarray:
    total = int(coords.shape[0])
    if total <= max_points:
        return coords
    rng = np.random.default_rng(VIEWER_SAMPLE_SEED)
    indices = rng.choice(total, size=max_points, replace=False)
    indices.sort()
    return coords[indices]


def _sparse_points(
    array_path: Path,
    metadata: dict,
    max_points: int = MAX_VIEWER_POINTS,
    *,
    surface_only: bool = False,
) -> dict:
    if not array_path.exists():
        return {"shape": None, "points": []}
    arr = np.load(array_path)
    if arr.ndim == 4:
        arr = np.any(arr, axis=0)
    if surface_only:
        arr = _surface_mask_from_solid(arr)
    transform = _viewer_transform(metadata, arr.shape)
    coords = np.argwhere(arr.astype(bool))
    total = int(coords.shape[0])
    coords = _sample_coords(coords, max_points)
    viewer_points = _coords_to_viewer_points(coords, transform)
    return {
        "shape": [int(v) for v in arr.shape],
        "total_points": total,
        "sampled_points": int(coords.shape[0]),
        "points": coords.astype(int).tolist(),
        "viewer_points": viewer_points.tolist(),
        "transform": transform,
    }


def _voxel_file(job_dir: Path) -> Path | None:
    files = sorted(job_dir.glob("voxel_*.npy"))
    return files[0] if files else None


def _run_job(job_dir: Path, step_path: Path, factory_profile: str, material: str, resolution: int, confidence: float, model: str | None) -> None:
    status_path = job_dir / JOB_STATUS_FILE
    _write_json_atomic(status_path, {"state": "running", "job_id": job_dir.name, "started_at": time.time()})
    args = argparse.Namespace(
        step_file=str(step_path),
        factory_profile=factory_profile,
        material=material,
        output=str(job_dir),
        model=model,
        device=None,
        resolution=resolution,
        confidence=confidence,
        resume_from=1,
        dry_run=False,
        quiet=True,
    )
    try:
        manifest = run_pipeline(args)
    except Exception as exc:  # Surface pipeline errors through the API.
        _write_json_atomic(
            status_path,
            {
                "state": "failed",
                "job_id": job_dir.name,
                "error": str(exc),
                "finished_at": time.time(),
            },
        )
        return
    _write_json_atomic(
        status_path,
        {
            "state": "complete",
            "job_id": job_dir.name,
            "finished_at": time.time(),
            "summary": manifest.get("summary", {}),
        },
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "processed_dir": str(PROCESSED_DIR), "validation_dir": str(VALIDATION_DIR)}


@app.get("/api/jobs")
def list_jobs() -> dict:
    return {"jobs": [_status_for_job(path) for path in _candidate_job_dirs()]}


@app.post("/api/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
    step_file: UploadFile = File(...),
    factory_profile: str = Form("factory_profiles/nash_nz.json"),
    material: str = Form(DEFAULT_MATERIAL),
    resolution: int = Form(DEFAULT_RESOLUTION),
    confidence: float = Form(DEFAULT_CONFIDENCE),
    model: str | None = Form(None),
) -> dict:
    if not step_file.filename or Path(step_file.filename).suffix.lower() not in {".stp", ".step"}:
        raise HTTPException(status_code=400, detail="Upload a .stp or .step file.")
    job_dir = _unique_job_dir(_safe_job_id(step_file.filename))
    job_dir.mkdir(parents=True, exist_ok=True)
    upload_path = job_dir / step_file.filename
    with upload_path.open("wb") as f:
        shutil.copyfileobj(step_file.file, f)
    _write_json_atomic(
        job_dir / JOB_STATUS_FILE,
        {"state": "queued", "job_id": job_dir.name, "created_at": time.time()},
    )
    background_tasks.add_task(
        _run_job,
        job_dir,
        upload_path,
        factory_profile,
        material,
        int(resolution),
        float(confidence),
        model,
    )
    return {"job_id": job_dir.name, "state": "queued"}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job_dir = _job_dir(job_id)
    return {"status": _status_for_job(job_dir), "artifacts": _list_artifacts(job_dir)}


@app.get("/api/jobs/{job_id}/artifacts")
def list_job_artifacts(job_id: str) -> dict:
    job_dir = _job_dir(job_id)
    return {"job_id": job_dir.name, "artifacts": _list_artifacts(job_dir)}


@app.get("/api/jobs/{job_id}/artifacts/{artifact_name}", response_model=None)
def get_artifact(job_id: str, artifact_name: str):
    job_dir = _job_dir(job_id)
    path = _artifact_path(job_dir, artifact_name)
    if path.suffix == ".json":
        return _read_json(path)
    return FileResponse(path)


@app.get("/api/jobs/{job_id}/viewer-data")
def get_viewer_data(job_id: str, max_points: int = MAX_VIEWER_POINTS) -> dict:
    job_dir = _job_dir(job_id)
    voxel = _voxel_file(job_dir)
    surface = job_dir / "surface_mask.npy"
    metadata = _read_json(job_dir / "metadata.json")
    setup = _read_json(job_dir / "setup_analysis.json")
    voxel_points = _sparse_points(voxel, metadata, max_points) if voxel else {"shape": None, "points": []}
    feature_instances = _read_json(job_dir / "feature_instances.json")
    voxel_shape = tuple(voxel_points.get("shape") or ()) or None
    return {
        "job_id": job_dir.name,
        "status": _status_for_job(job_dir),
        "metadata": metadata,
        "features": _read_json(job_dir / "features.json"),
        "feature_instances": feature_instances,
        "feature_overlays": _feature_overlays(feature_instances, metadata, voxel_shape),
        "setup_analysis": setup,
        "setup_overlay": _setup_overlay(setup, metadata, voxel_shape),
        "process_plan": _read_json(job_dir / "process_plan.json"),
        "simulation_input": _read_json(job_dir / "simulation_input.json"),
        "time_estimate": _read_json(job_dir / "time_estimate.json"),
        "quotation": _read_json(job_dir / "quotation.json"),
        "mesh_url": f"/api/jobs/{job_dir.name}/artifacts/mesh.stl" if (job_dir / "mesh.stl").exists() else None,
        "voxel_points": voxel_points,
        "surface_points": (
            _sparse_points(surface, metadata, max_points)
            if surface.exists()
            else (_sparse_points(voxel, metadata, max_points, surface_only=True) if voxel else {"shape": None, "points": []})
        ),
        "approach_direction": "+Z",
    }


def main() -> None:
    import uvicorn

    uvicorn.run("web_app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
