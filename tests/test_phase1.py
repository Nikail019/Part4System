import json
import os
from pathlib import Path

import numpy as np
import pytest

from phase1_voxeliser import process_step_file


FIXTURES = {
    "simple": "tests/fixtures/simple_block.stp",
    "holes": "tests/fixtures/block_with_holes.stp",
    "complex": "tests/fixtures/complex_prismatic.stp",
}

REQUIRED_METADATA_KEYS = [
    "source_file",
    "source_unit",
    "resolution",
    "bounding_box_mm",
    "volume_mm3",
    "surface_area_mm2",
    "occupancy_ratio",
    "raw_stock_mm",
    "mesh_face_count",
    "mesh_is_watertight",
    "step_backend",
    "voxel_file",
    "mesh_file",
    "warnings",
]


def _require_fixtures():
    missing = [path for path in FIXTURES.values() if not Path(path).exists()]
    if missing:
        pytest.skip("STEP fixtures missing; run python scripts/generate_fixtures.py")


@pytest.mark.parametrize("key", FIXTURES)
def test_output_files_created(key, tmp_path):
    _require_fixtures()
    process_step_file(FIXTURES[key], str(tmp_path))
    assert (tmp_path / "voxel_64.npy").exists()
    assert (tmp_path / "mesh.stl").exists()
    assert (tmp_path / "metadata.json").exists()


@pytest.mark.parametrize("key", FIXTURES)
def test_voxel_shape_default(key, tmp_path):
    _require_fixtures()
    process_step_file(FIXTURES[key], str(tmp_path))
    grid = np.load(tmp_path / "voxel_64.npy")
    assert grid.shape == (64, 64, 64)
    assert grid.dtype == bool


def test_voxel_custom_resolution(tmp_path):
    _require_fixtures()
    process_step_file(FIXTURES["simple"], str(tmp_path), resolution=32)
    grid = np.load(tmp_path / "voxel_32.npy")
    assert grid.shape == (32, 32, 32)


@pytest.mark.parametrize("key", FIXTURES)
def test_occupancy_in_range(key, tmp_path):
    _require_fixtures()
    result = process_step_file(FIXTURES[key], str(tmp_path))
    assert 0.05 <= result["occupancy_ratio"] <= 0.80


def test_voxel_is_solid_not_shell(tmp_path):
    _require_fixtures()
    process_step_file(FIXTURES["simple"], str(tmp_path))
    grid = np.load(tmp_path / "voxel_64.npy")
    cx, cy, cz = [s // 2 for s in grid.shape]
    assert grid[cx, cy, cz], "Centre voxel empty - interior not filled"


@pytest.mark.parametrize("key", FIXTURES)
def test_metadata_keys_present(key, tmp_path):
    _require_fixtures()
    result = process_step_file(FIXTURES[key], str(tmp_path))
    for key_name in REQUIRED_METADATA_KEYS:
        assert key_name in result, f"Missing metadata key: {key_name}"


@pytest.mark.parametrize("key", FIXTURES)
def test_metadata_valid_json_on_disk(key, tmp_path):
    _require_fixtures()
    process_step_file(FIXTURES[key], str(tmp_path))
    with open(tmp_path / "metadata.json", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict)


@pytest.mark.parametrize("key", FIXTURES)
def test_bounding_box_all_positive(key, tmp_path):
    _require_fixtures()
    result = process_step_file(FIXTURES[key], str(tmp_path))
    for axis, val in result["bounding_box_mm"].items():
        assert val > 0, f"bbox {axis} = {val}"


@pytest.mark.parametrize("key", FIXTURES)
def test_volume_positive(key, tmp_path):
    _require_fixtures()
    result = process_step_file(FIXTURES[key], str(tmp_path))
    assert result["volume_mm3"] > 0


@pytest.mark.parametrize("key", FIXTURES)
def test_surface_area_positive(key, tmp_path):
    _require_fixtures()
    result = process_step_file(FIXTURES[key], str(tmp_path))
    assert result["surface_area_mm2"] > 0


def test_simple_block_dimensions(tmp_path):
    _require_fixtures()
    result = process_step_file(FIXTURES["simple"], str(tmp_path))
    bb = result["bounding_box_mm"]
    dims = sorted([bb["x"], bb["y"], bb["z"]])
    assert abs(dims[0] - 40.0) / 40.0 < 0.01
    assert abs(dims[1] - 60.0) / 60.0 < 0.01
    assert abs(dims[2] - 100.0) / 100.0 < 0.01


@pytest.mark.parametrize("key", FIXTURES)
def test_raw_stock_gte_bbox(key, tmp_path):
    _require_fixtures()
    result = process_step_file(FIXTURES[key], str(tmp_path))
    for axis in ("x", "y", "z"):
        assert result["raw_stock_mm"][axis] >= result["bounding_box_mm"][axis]


@pytest.mark.parametrize("key", FIXTURES)
def test_raw_stock_divisible_by_5(key, tmp_path):
    _require_fixtures()
    result = process_step_file(FIXTURES[key], str(tmp_path))
    for axis, val in result["raw_stock_mm"].items():
        assert val % 5 == 0, f"raw_stock {axis} = {val} not divisible by 5"


def test_file_not_found_raises():
    with pytest.raises(FileNotFoundError):
        process_step_file("does_not_exist.stp", "/tmp/out")


def test_output_dir_created_if_absent(tmp_path):
    _require_fixtures()
    new_dir = tmp_path / "subdir" / "nested"
    assert not new_dir.exists()
    process_step_file(FIXTURES["simple"], str(new_dir))
    assert new_dir.exists()


@pytest.mark.parametrize("key", FIXTURES)
def test_output_paths_are_absolute(key, tmp_path):
    _require_fixtures()
    result = process_step_file(FIXTURES[key], str(tmp_path))
    assert os.path.isabs(result["voxel_file"])
    assert os.path.isabs(result["mesh_file"])


@pytest.mark.parametrize("key", FIXTURES)
def test_output_paths_exist_on_disk(key, tmp_path):
    _require_fixtures()
    result = process_step_file(FIXTURES[key], str(tmp_path))
    assert os.path.exists(result["voxel_file"])
    assert os.path.exists(result["mesh_file"])


@pytest.mark.parametrize("key", FIXTURES)
def test_warnings_key_is_list(key, tmp_path):
    _require_fixtures()
    result = process_step_file(FIXTURES[key], str(tmp_path))
    assert isinstance(result["warnings"], list)
