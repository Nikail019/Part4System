import importlib.util

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("fastapi") is None,
    reason="FastAPI is not installed.",
)


def test_health_endpoint():
    from fastapi.testclient import TestClient
    from web_app import app

    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_jobs_endpoint_lists_validation_outputs():
    from fastapi.testclient import TestClient
    from web_app import app

    client = TestClient(app)
    response = client.get("/api/jobs")
    assert response.status_code == 200
    data = response.json()
    assert "jobs" in data
    assert any(job["job_id"] == "simple_block" for job in data["jobs"])


def test_viewer_data_has_sparse_points():
    from fastapi.testclient import TestClient
    from web_app import app

    client = TestClient(app)
    response = client.get("/api/jobs/simple_block/viewer-data?max_points=100")
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == "simple_block"
    assert "voxel_points" in data
    assert data["voxel_points"]["sampled_points"] <= 100
    assert data["surface_points"]["sampled_points"] <= 100
    assert data["mesh_url"].endswith("mesh.stl")
    assert "feature_overlays" in data
    assert "setup_overlay" in data
    assert "simulation_input" in data
    assert data["setup_overlay"]["approach_direction"] in {"+Z", "-Z", "+X", "-X", "+Y", "-Y"}


def test_sparse_points_sampling_is_not_stride_pattern(tmp_path):
    import numpy as np
    from web_app import _sparse_points

    arr = np.ones((12, 12, 12), dtype=bool)
    voxel = tmp_path / "voxel.npy"
    np.save(voxel, arr)

    data = _sparse_points(voxel, {"bounding_box_mm": {"x": 12, "y": 12, "z": 12}}, max_points=30)
    flattened = [point[0] * 144 + point[1] * 12 + point[2] for point in data["points"]]
    deltas = [b - a for a, b in zip(flattened, flattened[1:])]

    assert data["sampled_points"] == 30
    assert len(set(deltas)) > 1


def test_surface_only_sparse_points_extracts_shell(tmp_path):
    import numpy as np
    from web_app import _sparse_points

    arr = np.ones((8, 8, 8), dtype=bool)
    voxel = tmp_path / "voxel.npy"
    np.save(voxel, arr)

    solid = _sparse_points(voxel, {}, max_points=1000)
    surface = _sparse_points(voxel, {}, max_points=1000, surface_only=True)

    assert solid["total_points"] == 512
    assert surface["total_points"] == 296
    assert surface["total_points"] < solid["total_points"]


def test_json_artifact_endpoint():
    from fastapi.testclient import TestClient
    from web_app import app

    client = TestClient(app)
    response = client.get("/api/jobs/simple_block/artifacts/quotation.json")
    assert response.status_code == 200
    assert "recommendation" in response.json()


def test_feature_overlay_conversion_has_viewer_bbox():
    from web_app import _feature_overlays

    metadata = {"bounding_box_mm": {"x": 100.0, "y": 60.0, "z": 40.0}}
    feature_instances = {
        "instances": [
            {
                "type": "rectangular_pocket",
                "instance_id": 2,
                "confidence": 0.91,
                "localisation_status": "localised",
                "volume_voxels": 120,
                "bbox_voxel": [[24, 22, 44], [40, 36, 60]],
                "centroid_voxel": [32, 29, 52],
                "primary_direction": "+Z",
                "access_class": "top",
            }
        ]
    }

    overlays = _feature_overlays(feature_instances, metadata, (64, 64, 64))

    assert len(overlays) == 1
    overlay = overlays[0]
    assert overlay["type"] == "rectangular_pocket"
    assert overlay["instance_id"] == 2
    assert len(overlay["bbox_center"]) == 3
    assert all(value > 0 for value in overlay["bbox_size"])


def test_feature_overlay_skips_estimated_fallback_instances():
    from web_app import _feature_overlays

    metadata = {"bounding_box_mm": {"x": 50.0, "y": 50.0, "z": 50.0}}
    feature_instances = {
        "instances": [
            {
                "type": "rectangular_pocket",
                "instance_id": 0,
                "confidence": 0.99,
                "localisation_status": "estimated",
                "volume_voxels": 0,
                "bbox_voxel": [[0, 0, 0], [31, 31, 31]],
            }
        ]
    }

    assert _feature_overlays(feature_instances, metadata, (32, 32, 32)) == []
