"""Synthetic labelled CAD data generation for Phase 2."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from datetime import date
from pathlib import Path

import cadquery as cq
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.feature_net import FEATURE_NAMES

EXCLUDED_TRAINING_FEATURES = {"chamfer", "fillet"}
ACTIVE_SYNTHETIC_FEATURES = [
    name for name in FEATURE_NAMES if name not in EXCLUDED_TRAINING_FEATURES
]


def make_base_block() -> tuple:
    """Random block 60-150 x 60-120 x 30-80 mm."""
    x = random.uniform(60, 150)
    y = random.uniform(60, 120)
    z = random.uniform(30, 80)
    return cq.Workplane("XY").box(x, y, z), {"x": x, "y": y, "z": z}


def _safe_xy(block: dict, margin: float = 18.0) -> tuple[float, float]:
    max_x = max(1.0, block["x"] / 2 - margin)
    max_y = max(1.0, block["y"] / 2 - margin)
    return random.uniform(-max_x, max_x), random.uniform(-max_y, max_y)


def add_through_hole(wp, block):
    diameter = random.uniform(6, min(20, block["x"] * 0.2, block["y"] * 0.2))
    x, y = _safe_xy(block, diameter)
    return wp.faces(">Z").workplane().center(x, y).hole(diameter)


def add_blind_hole(wp, block):
    diameter = random.uniform(6, min(20, block["x"] * 0.2, block["y"] * 0.2))
    depth = random.uniform(block["z"] * 0.2, block["z"] * 0.6)
    x, y = _safe_xy(block, diameter)
    return wp.faces(">Z").workplane().center(x, y).hole(diameter, depth)


def add_rectangular_pocket(wp, block):
    width = random.uniform(block["x"] * 0.2, block["x"] * 0.55)
    height = random.uniform(block["y"] * 0.2, block["y"] * 0.55)
    depth = random.uniform(block["z"] * 0.2, block["z"] * 0.4)
    return wp.faces(">Z").workplane().rect(width, height).cutBlind(-depth)


def add_circular_pocket(wp, block):
    radius = random.uniform(min(block["x"], block["y"]) * 0.15, min(block["x"], block["y"]) * 0.3)
    depth = random.uniform(block["z"] * 0.15, block["z"] * 0.35)
    x, y = _safe_xy(block, radius * 1.2)
    return wp.faces(">Z").workplane().center(x, y).circle(radius).cutBlind(-depth)


def add_rectangular_slot(wp, block):
    width = random.uniform(8, min(24, block["x"] * 0.35))
    depth = random.uniform(block["z"] * 0.15, block["z"] * 0.35)
    x = random.uniform(-block["x"] * 0.2, block["x"] * 0.2)
    return wp.faces(">Z").workplane().center(x, 0).rect(width, block["y"] * 1.2).cutBlind(-depth)


def add_circular_slot(wp, block):
    outer = random.uniform(min(block["x"], block["y"]) * 0.18, min(block["x"], block["y"]) * 0.32)
    inner = outer * random.uniform(0.55, 0.75)
    depth = random.uniform(block["z"] * 0.1, block["z"] * 0.25)
    return wp.faces(">Z").workplane().circle(outer).circle(inner).cutBlind(-depth)


def add_rectangular_step(wp, block):
    width = random.uniform(block["x"] * 0.2, block["x"] * 0.45)
    depth = random.uniform(block["z"] * 0.15, block["z"] * 0.4)
    x = block["x"] / 2 - width / 2
    return wp.faces(">Z").workplane().center(x, 0).rect(width, block["y"] * 1.1).cutBlind(-depth)


def add_chamfer(wp, block):
    size = random.uniform(2, min(8, block["z"] * 0.12))
    return wp.edges("|Z").chamfer(size)


def add_fillet(wp, block):
    radius = random.uniform(2, min(10, block["x"] * 0.05, block["y"] * 0.05))
    return wp.edges("|Z").fillet(radius)


def add_boss(wp, block):
    radius = random.uniform(6, min(block["x"], block["y"]) * 0.18)
    height = random.uniform(5, block["z"] * 0.3)
    x, y = _safe_xy(block, radius * 1.5)
    return wp.faces(">Z").workplane().center(x, y).circle(radius).extrude(height)


def add_flat_face(wp, block):
    return wp


def add_triangular_pocket(wp, block):
    side = random.uniform(min(block["x"], block["y"]) * 0.2, min(block["x"], block["y"]) * 0.45)
    depth = random.uniform(block["z"] * 0.15, block["z"] * 0.35)
    height = math.sqrt(3) * side / 2
    points = [(-side / 2, -height / 3), (side / 2, -height / 3), (0, 2 * height / 3)]
    return wp.faces(">Z").workplane().polyline(points).close().cutBlind(-depth)


FEATURE_ADDERS = {
    "through_hole": add_through_hole,
    "blind_hole": add_blind_hole,
    "rectangular_pocket": add_rectangular_pocket,
    "circular_pocket": add_circular_pocket,
    "rectangular_slot": add_rectangular_slot,
    "circular_slot": add_circular_slot,
    "rectangular_step": add_rectangular_step,
    "chamfer": add_chamfer,
    "fillet": add_fillet,
    "boss": add_boss,
    "flat_face": add_flat_face,
    "triangular_pocket": add_triangular_pocket,
}


def generate_part(
    min_features: int = 0,
    max_features: int = 4,
    target_feature: str | None = None,
    extra_feature_chance: float = 0.25,
) -> tuple:
    """Build a random labelled part with optional class targeting."""
    wp, block = make_base_block()
    labels = ["flat_face"]
    candidates = [name for name in ACTIVE_SYNTHETIC_FEATURES if name != "flat_face"]
    if target_feature in EXCLUDED_TRAINING_FEATURES:
        target_feature = "flat_face"

    chosen: list[str] = []
    if target_feature and target_feature != "flat_face":
        chosen.append(target_feature)

    if target_feature is None:
        chosen.extend(random.sample(candidates, k=random.randint(min_features, max_features)))
    elif target_feature != "flat_face" and random.random() < extra_feature_chance:
        extras = [name for name in candidates if name != target_feature]
        extra_count = random.randint(0, max(0, min(max_features - 1, 2)))
        chosen.extend(random.sample(extras, k=extra_count))
    elif target_feature == "flat_face" and min_features > 0:
        chosen.extend(random.sample(candidates, k=random.randint(min_features, max_features)))

    chosen = list(dict.fromkeys(chosen))
    for feature in chosen:
        try:
            wp = FEATURE_ADDERS[feature](wp, block)
            labels.append(feature)
        except Exception:
            pass
    return wp, labels


def _write_labels(path: Path, labels: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump({"labels": labels}, f, indent=2)
        f.write("\n")


def generate_dataset(output_dir: str, count: int = 2000, balanced: bool = True) -> None:
    """Generate labelled STEP parts into output_dir."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    existing_manifest = {}
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as f:
            existing_manifest = json.load(f)

    distribution = {name: 0 for name in FEATURE_NAMES}
    distribution.update(existing_manifest.get("classes", {}))
    existing_dirs = [
        int(path.name)
        for path in root.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]
    start_idx = max(existing_dirs, default=-1) + 1
    existing_total = int(existing_manifest.get("total_files", existing_manifest.get("total", start_idx)))
    previous_attempts = int(existing_manifest.get("attempts", 0))
    generated = 0
    attempts = 0

    with tqdm(total=count, desc="Generating synthetic parts") as progress:
        while generated < count and attempts < count * 5:
            attempts += 1
            try:
                target = (
                    ACTIVE_SYNTHETIC_FEATURES[
                        (start_idx + generated) % len(ACTIVE_SYNTHETIC_FEATURES)
                    ]
                    if balanced
                    else None
                )
                part, labels = generate_part(target_feature=target)
                part_dir = root / f"{start_idx + generated:05d}"
                part_dir.mkdir(parents=True, exist_ok=True)
                cq.exporters.export(part, str(part_dir / "part.stp"), exportType="STEP")
                _write_labels(part_dir / "labels.json", labels)
            except Exception:
                continue

            for label in labels:
                distribution[label] += 1
            generated += 1
            progress.update(1)

    total = existing_total + generated
    manifest = {
        "total": total,
        "total_files": total,
        "generated_this_run": generated,
        "attempts": previous_attempts + attempts,
        "date": date.today().isoformat(),
        "balanced": balanced,
        "active_features": ACTIVE_SYNTHETIC_FEATURES,
        "excluded_features": sorted(EXCLUDED_TRAINING_FEATURES),
        "classes": distribution,
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic STEP training data.")
    parser.add_argument("--output", default=os.path.join("data", "raw", "synthetic"))
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--balanced", action="store_true", default=True)
    parser.add_argument("--random", action="store_false", dest="balanced")
    args = parser.parse_args()
    generate_dataset(args.output, args.count, balanced=args.balanced)


if __name__ == "__main__":
    main()
