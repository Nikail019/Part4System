"""Inference entry point for Phase 2 machining feature recognition."""

from __future__ import annotations

import argparse
import json
import os
import tempfile

import numpy as np
import torch

from models.feature_net import FEATURE_NAMES, load_model


def _write_json_atomic(data: dict, path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=directory, delete=False, suffix=".tmp") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
        tmp = f.name
    os.replace(tmp, path)


def recognise_features(
    voxel_path: str,
    model_path: str,
    threshold: float = 0.5,
    device: str = "cpu",
) -> dict:
    """Predict machining features present in a voxelised part."""
    voxel_abs = os.path.abspath(voxel_path)
    model_abs = os.path.abspath(model_path)
    if not os.path.exists(voxel_abs):
        raise FileNotFoundError(voxel_path)
    if not os.path.exists(model_abs):
        raise FileNotFoundError(model_path)

    voxel = np.load(voxel_abs)
    if voxel.ndim != 3 or len(set(voxel.shape)) != 1:
        raise ValueError("voxel array must be 3-D and cubic.")

    torch_device = torch.device(device)
    model = load_model(model_abs, device=torch_device)
    x = torch.from_numpy(voxel.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(torch_device)

    with torch.no_grad():
        probs = torch.sigmoid(model(x)).detach().cpu().numpy()[0]

    all_scores = {name: float(probs[idx]) for idx, name in enumerate(FEATURE_NAMES)}
    features = [
        {"type": name, "confidence": score}
        for name, score in all_scores.items()
        if score >= threshold
    ]
    features.sort(key=lambda item: item["confidence"], reverse=True)

    return {
        "features": features,
        "feature_count": len(features),
        "all_scores": all_scores,
        "threshold": float(threshold),
        "voxel_file": voxel_abs,
        "model_path": model_abs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recognise machining features from a voxel grid.")
    parser.add_argument("voxel_path")
    parser.add_argument("model_path")
    parser.add_argument("output_dir")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    result = recognise_features(
        args.voxel_path,
        args.model_path,
        threshold=args.threshold,
        device=args.device,
    )
    output_path = os.path.join(os.path.abspath(args.output_dir), "features.json")
    _write_json_atomic(result, output_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
