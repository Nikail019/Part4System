"""Phase 1 STEP-to-voxel ingestion pipeline for the RPP MVP."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile

import numpy as np
import trimesh


_STEP_BACKEND: str | None = None

try:
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.StlAPI import StlAPI_Writer

    _STEP_BACKEND = "occ"
except ImportError:
    try:
        import cadquery as cq

        _STEP_BACKEND = "cadquery"
    except ImportError:
        pass

if _STEP_BACKEND is None:
    raise ImportError(
        "No STEP backend found.\n"
        "Install pythonOCC:  conda install -c conda-forge pythonocc-core\n"
        "Install cadquery:   pip install cadquery"
    )


UNIT_MULTIPLIERS = {
    "MILLIMETRE": 1.0,
    "MILLIMETER": 1.0,
    "MM": 1.0,
    "METRE": 1000.0,
    "METER": 1000.0,
    "INCH": 25.4,
    "FOOT": 304.8,
    "FT": 304.8,
}


def _detect_unit_multiplier(step_path: str) -> tuple[float, str, bool]:
    """Return (linear mm multiplier, unit name, detected flag)."""
    lines: list[str] = []
    with open(step_path, "r", encoding="utf-8", errors="ignore") as f:
        for index, line in enumerate(f):
            if index > 1000:
                break
            lines.append(line.upper())

    text = "\n".join(lines)
    if re.search(r"SI_UNIT\s*\(\s*\.MILLI\.\s*,\s*\.METRE\.", text):
        return 1.0, "MM", True
    if re.search(r"SI_UNIT\s*\(\s*\$\s*,\s*\.METRE\.", text):
        return 1000.0, "M", True

    for unit_key, multiplier in UNIT_MULTIPLIERS.items():
        if re.search(rf"(?<![A-Z]){re.escape(unit_key)}(?![A-Z])", text):
            return multiplier, unit_key, True
    return 1.0, "MM", False


def _is_null_shape(shape) -> bool:
    is_null = getattr(shape, "IsNull", None)
    return bool(is_null()) if callable(is_null) else shape is None


def _parse_step_occ(step_path: str, stl_path: str) -> dict:
    """Parse STEP with pythonOCC and write an STL mesh."""
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_path)
    if status != IFSelect_RetDone:
        raise ValueError(f"Could not read STEP file: {step_path}")

    transferred = reader.TransferRoots()
    if transferred <= 0:
        raise ValueError("STEP file contains no transferable solid bodies.")

    shape = reader.OneShape()
    if _is_null_shape(shape):
        raise ValueError("STEP file contains no valid solid bodies.")

    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

    volume_props = GProp_GProps()
    brepgprop.VolumeProperties(shape, volume_props)
    volume = volume_props.Mass()
    if volume <= 0:
        raise ValueError("STEP file contains no valid solid bodies.")

    surface_props = GProp_GProps()
    brepgprop.SurfaceProperties(shape, surface_props)
    surface_area = surface_props.Mass()

    mesher = BRepMesh_IncrementalMesh(shape, 0.1, False, 0.5, True)
    mesher.Perform()
    writer = StlAPI_Writer()
    writer.SetASCIIMode(False)
    if not writer.Write(shape, stl_path):
        raise RuntimeError(f"Failed to write STL mesh: {stl_path}")

    mesh = trimesh.load(stl_path, force="mesh")
    return {
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "zmin": zmin,
        "zmax": zmax,
        "volume": volume,
        "surface_area": surface_area,
        "mesh_face_count": int(len(mesh.faces)),
    }


def _parse_step_cadquery(step_path: str, stl_path: str) -> dict:
    """Parse STEP with cadquery fallback and write an STL mesh."""
    try:
        result = cq.importers.importStep(step_path)
        shape = result.val()
    except Exception as exc:
        raise ValueError(f"Could not read STEP file: {step_path}") from exc

    if shape is None:
        raise ValueError("STEP file contains no valid solid bodies.")

    bb = shape.BoundingBox()
    cq.exporters.export(result, stl_path)

    mesh = trimesh.load(stl_path, force="mesh")
    if len(mesh.faces) == 0 or mesh.volume <= 0:
        raise ValueError("STEP file contains no valid solid bodies.")

    return {
        "xmin": bb.xmin,
        "xmax": bb.xmax,
        "ymin": bb.ymin,
        "ymax": bb.ymax,
        "zmin": bb.zmin,
        "zmax": bb.zmax,
        "volume": float(abs(mesh.volume)),
        "surface_area": float(mesh.area),
        "mesh_face_count": int(len(mesh.faces)),
    }


def _pad_or_crop(grid: np.ndarray, target: int) -> np.ndarray:
    """Centre-pad or centre-crop a 3D bool array to (target, target, target)."""
    for axis in range(3):
        if grid.shape[axis] > target:
            start = (grid.shape[axis] - target) // 2
            stop = start + target
            grid = np.take(grid, range(start, stop), axis=axis)

    out = np.zeros((target, target, target), dtype=bool)
    offsets = [(target - grid.shape[i]) // 2 for i in range(3)]
    slices = tuple(slice(offsets[i], offsets[i] + grid.shape[i]) for i in range(3))
    out[slices] = grid
    return out


def _voxelise_mesh(stl_path: str, resolution: int) -> tuple[np.ndarray, bool]:
    """Convert an STL mesh to a solid-filled boolean voxel grid."""
    mesh = trimesh.load(stl_path, force="mesh")
    if len(mesh.faces) == 0:
        raise RuntimeError("Mesh contains no faces.")

    is_watertight = bool(mesh.is_watertight)
    if not is_watertight:
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_winding(mesh)
        trimesh.repair.fix_normals(mesh)
        is_watertight = bool(mesh.is_watertight)

    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    longest = float(mesh.extents.max())
    if longest == 0:
        raise RuntimeError("Mesh has zero extent - degenerate geometry.")

    mesh.apply_scale((resolution - 2) / longest)
    voxel_obj = mesh.voxelized(pitch=1.0).fill()
    grid = _pad_or_crop(voxel_obj.matrix.astype(bool), resolution)

    if grid.sum() == 0:
        raise RuntimeError("Voxelisation produced an empty grid.")

    return grid, is_watertight


def _compute_raw_stock(bbox_mm: dict, allowance: float) -> dict:
    """Apply stock allowance and round up to the nearest 5 mm."""
    def up5(value: float) -> float:
        return float(math.ceil(value * (1.0 + allowance) / 5.0) * 5.0)

    return {axis: up5(dim) for axis, dim in bbox_mm.items()}


def _write_json_atomic(data: dict, path: str) -> None:
    """Write JSON via a same-directory temporary file and atomic rename."""
    directory = os.path.dirname(os.path.abspath(path))
    with tempfile.NamedTemporaryFile("w", dir=directory, delete=False, suffix=".tmp") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
        tmp = f.name
    os.replace(tmp, path)


def _convert_geometry(raw: dict, multiplier: float) -> tuple[dict, float, float]:
    bbox_mm = {
        "x": float((raw["xmax"] - raw["xmin"]) * multiplier),
        "y": float((raw["ymax"] - raw["ymin"]) * multiplier),
        "z": float((raw["zmax"] - raw["zmin"]) * multiplier),
    }
    volume_mm3 = float(raw["volume"] * multiplier**3)
    surface_area_mm2 = float(raw["surface_area"] * multiplier**2)
    return bbox_mm, volume_mm3, surface_area_mm2


def _backend_geometry_multiplier(declared_multiplier: float) -> float:
    """Return the extra multiplier needed after STEP import.

    CadQuery/pythonOCC import through OCCT, which normalises STEP length units
    into millimetres while building the shape. The declared unit is still
    useful metadata, but applying it again makes metre-authored AP242 files
    1000x too large.
    """
    return 1.0


def process_step_file(
    step_path: str,
    output_dir: str,
    resolution: int = 64,
    stock_allowance: float = 0.10,
) -> dict:
    """Convert a STEP file into a voxel grid and metadata object."""
    if resolution < 4:
        raise ValueError("resolution must be at least 4.")

    step_abs = os.path.abspath(step_path)
    if not os.path.exists(step_abs):
        raise FileNotFoundError(step_path)

    output_abs = os.path.abspath(output_dir)
    os.makedirs(output_abs, exist_ok=True)

    voxel_path = os.path.join(output_abs, f"voxel_{resolution}.npy")
    stl_path = os.path.join(output_abs, "mesh.stl")
    metadata_path = os.path.join(output_abs, "metadata.json")

    multiplier, source_unit, detected_unit = _detect_unit_multiplier(step_abs)
    geometry_multiplier = _backend_geometry_multiplier(multiplier)
    parser = _parse_step_occ if _STEP_BACKEND == "occ" else _parse_step_cadquery
    raw = parser(step_abs, stl_path)
    bbox_mm, volume_mm3, surface_area_mm2 = _convert_geometry(raw, geometry_multiplier)

    grid, mesh_is_watertight = _voxelise_mesh(stl_path, resolution)
    np.save(voxel_path, grid)

    occupancy_ratio = float(grid.sum() / grid.size)
    warnings: list[str] = []
    if not detected_unit:
        warnings.append("STEP unit not detected; assuming millimetres.")
    elif multiplier != geometry_multiplier:
        warnings.append(
            f"STEP declares {source_unit}; STEP backend normalised geometry to millimetres."
        )
    if not mesh_is_watertight:
        warnings.append("Mesh is not watertight after repair attempt.")

    metadata = {
        "source_file": step_abs,
        "source_unit": source_unit,
        "resolution": int(resolution),
        "bounding_box_mm": bbox_mm,
        "volume_mm3": volume_mm3,
        "surface_area_mm2": surface_area_mm2,
        "occupancy_ratio": occupancy_ratio,
        "raw_stock_mm": _compute_raw_stock(bbox_mm, stock_allowance),
        "mesh_face_count": int(raw["mesh_face_count"]),
        "mesh_is_watertight": bool(mesh_is_watertight),
        "step_backend": _STEP_BACKEND,
        "voxel_file": voxel_path,
        "mesh_file": stl_path,
        "warnings": warnings,
    }

    _write_json_atomic(metadata, metadata_path)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a STEP file to a voxel grid and metadata."
    )
    parser.add_argument("step_path", help="Path to .stp / .step file")
    parser.add_argument("output_dir", help="Directory for output files")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--stock-allowance", type=float, default=0.10)
    args = parser.parse_args()

    result = process_step_file(
        args.step_path,
        args.output_dir,
        resolution=args.resolution,
        stock_allowance=args.stock_allowance,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
