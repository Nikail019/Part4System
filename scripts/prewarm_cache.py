#!/usr/bin/env python3
"""Pre-compute voxel cache for a synthetic dataset directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase1_voxeliser import process_step_file


def _valid_part_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        part_dir
        for part_dir in root.iterdir()
        if part_dir.is_dir()
        and (part_dir / "part.stp").exists()
        and (part_dir / "labels.json").exists()
    )


def prewarm(data_dir: str, resolution: int, max_samples: int | None) -> None:
    root = Path(data_dir)
    parts = _valid_part_dirs(root)
    selected = parts[:max_samples] if max_samples is not None else parts

    voxel_filename = f"voxel_{resolution}.npy"
    to_process = [part_dir for part_dir in selected if not (part_dir / voxel_filename).exists()]

    print(f"Parts total   : {len(parts)}")
    print(f"Selected      : {len(selected)}")
    print(f"Already cached: {len(selected) - len(to_process)}")
    print(f"To process    : {len(to_process)}")

    if not to_process:
        print("Cache already complete. Nothing to do.")
        return

    errors = 0
    for part_dir in tqdm(to_process, desc=f"Voxelising at {resolution}^3"):
        try:
            process_step_file(
                str(part_dir / "part.stp"),
                str(part_dir),
                resolution=resolution,
            )
        except Exception as exc:
            errors += 1
            tqdm.write(f"  SKIP {part_dir.name}: {exc}")

    print(f"\nDone. Cached: {len(to_process) - errors}  Errors: {errors}")
    if errors > 0:
        print(
            f"Note: {errors} parts failed voxelisation and may be skipped "
            "if cache-only training is enabled."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute voxel cache for training data.")
    parser.add_argument("--data", default="data/raw/synthetic")
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None, dest="max_samples")
    args = parser.parse_args()
    prewarm(args.data, args.resolution, args.max_samples)


if __name__ == "__main__":
    main()
