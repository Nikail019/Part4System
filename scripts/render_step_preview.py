"""Render a quick PNG preview for a STEP/STL file.

Usage:
  python scripts/render_step_preview.py tests/fixtures/simple_block.stp preview.png
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path


def _step_to_stl(step_path: Path, stl_path: Path) -> None:
    import cadquery as cq

    shape = cq.importers.importStep(str(step_path))
    cq.exporters.export(shape, str(stl_path))


def render_mesh_preview(
    mesh_path: str,
    output_path: str,
    width: int = 1400,
    height: int = 1000,
) -> str:
    """Render an STL mesh to PNG with a simple software orthographic renderer."""
    import numpy as np
    import trimesh
    from PIL import Image, ImageDraw

    mesh_abs = os.path.abspath(mesh_path)
    output_abs = os.path.abspath(output_path)

    mesh = trimesh.load(mesh_abs, force="mesh")
    if mesh.is_empty or len(mesh.faces) == 0:
        raise ValueError(f"Mesh has no renderable faces: {mesh_path}")

    vertices = mesh.vertices.astype(float)
    vertices -= vertices.mean(axis=0)

    azimuth = np.deg2rad(35.0)
    elevation = np.deg2rad(28.0)
    rz = np.array(
        [
            [np.cos(azimuth), -np.sin(azimuth), 0.0],
            [np.sin(azimuth), np.cos(azimuth), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(elevation), -np.sin(elevation)],
            [0.0, np.sin(elevation), np.cos(elevation)],
        ]
    )
    projected = vertices @ rz.T @ rx.T
    span = np.ptp(projected[:, :2], axis=0)
    scale = 0.78 * min(width / max(span[0], 1e-6), height / max(span[1], 1e-6))
    xy = projected[:, :2] * scale
    xy[:, 0] += width / 2.0
    xy[:, 1] = height / 2.0 - xy[:, 1]

    image = Image.new("RGB", (int(width), int(height)), (17, 19, 22))
    draw = ImageDraw.Draw(image)
    light_dir = np.array([0.25, -0.35, 0.90])
    light_dir /= np.linalg.norm(light_dir)
    normals = mesh.face_normals @ rz.T @ rx.T
    face_depth = projected[mesh.faces][:, :, 2].mean(axis=1)
    face_order = np.argsort(face_depth)

    base = np.array([145, 167, 179], dtype=float)
    edge = (37, 43, 49)
    for face_idx in face_order:
        face = mesh.faces[face_idx]
        points = [(float(x), float(y)) for x, y in xy[face]]
        intensity = 0.45 + 0.55 * max(0.0, float(np.dot(normals[face_idx], light_dir)))
        color = tuple(int(max(0, min(255, value))) for value in base * intensity)
        draw.polygon(points, fill=color, outline=edge)

    image.save(output_abs)
    return output_abs


def render_step_preview(
    input_path: str,
    output_path: str,
    width: int = 1400,
    height: int = 1000,
) -> str:
    """Render a STEP or STL file to a PNG preview."""
    source = Path(input_path).resolve()
    if not source.exists():
        raise FileNotFoundError(input_path)

    suffix = source.suffix.lower()
    if suffix == ".stl":
        return render_mesh_preview(str(source), output_path, width=width, height=height)
    if suffix not in {".stp", ".step"}:
        raise ValueError("Input must be .stp, .step, or .stl")

    with tempfile.TemporaryDirectory() as tmp:
        stl_path = Path(tmp) / "preview_mesh.stl"
        _step_to_stl(source, stl_path)
        return render_mesh_preview(str(stl_path), output_path, width=width, height=height)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a STEP/STL preview PNG.")
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=1000)
    args = parser.parse_args()

    output = render_step_preview(
        args.input_path,
        args.output_path,
        width=args.width,
        height=args.height,
    )
    print(output)


if __name__ == "__main__":
    main()
