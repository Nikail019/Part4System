"""Generate minimal STEP fixtures for Phase 1 tests."""

from __future__ import annotations

import os

import cadquery as cq


FIXTURE_DIR = os.path.join("tests", "fixtures")


def make_simple_block():
    """100 x 60 x 40 mm rectangular block. No features."""
    return cq.Workplane("XY").box(100, 60, 40)


def make_block_with_holes():
    """100 x 60 x 40 mm block with 3 x diameter 10 mm through-holes."""
    return (
        cq.Workplane("XY")
        .box(100, 60, 40)
        .faces(">Z")
        .workplane()
        .pushPoints([(-30, 0), (0, 0), (30, 0)])
        .hole(10)
    )


def make_complex_prismatic():
    """Block with a pocket, a step, and a through-hole."""
    return (
        cq.Workplane("XY")
        .box(120, 80, 50)
        .faces(">Z")
        .workplane()
        .rect(60, 40)
        .cutBlind(-15)
        .faces(">X")
        .workplane()
        .rect(80, 20)
        .cutBlind(-20)
        .faces(">Z")
        .workplane()
        .center(40, 0)
        .hole(12)
    )


def main() -> None:
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    fixtures = {
        "simple_block.stp": make_simple_block(),
        "block_with_holes.stp": make_block_with_holes(),
        "complex_prismatic.stp": make_complex_prismatic(),
    }
    for filename, shape in fixtures.items():
        path = os.path.join(FIXTURE_DIR, filename)
        cq.exporters.export(shape, path, exportType="STEP")
        print(f"  Written: {path}")
    print("Fixtures generated.")


if __name__ == "__main__":
    main()
