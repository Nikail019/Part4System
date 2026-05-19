"""Evaluate a trained FeatureNet3D checkpoint on a dataset split."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.feature_net import FEATURE_NAMES, NUM_CLASSES, load_model
from training.dataset import MachiningFeatureDataset, random_split_dataset


def _get_device() -> str:
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except AttributeError:
        pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def compute_multilabel_confusion(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> np.ndarray:
    """Return per-class binary confusion matrices shaped (C, 2, 2)."""
    cm = np.zeros((num_classes, 2, 2), dtype=np.int64)
    for class_idx in range(num_classes):
        p = preds[:, class_idx].numpy().astype(int)
        t = targets[:, class_idx].numpy().astype(int)
        for pred_val, true_val in zip(p, t):
            cm[class_idx, true_val, pred_val] += 1
    return cm


def _metrics_from_predictions(preds: torch.Tensor, targets: torch.Tensor) -> tuple[dict, float, float, float]:
    tp = (preds * targets).sum(dim=0)
    fp = (preds * (1 - targets)).sum(dim=0)
    fn = ((1 - preds) * targets).sum(dim=0)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    per_class = {}
    for idx, name in enumerate(FEATURE_NAMES):
        per_class[name] = {
            "precision": round(float(precision[idx]), 4),
            "recall": round(float(recall[idx]), 4),
            "f1": round(float(f1[idx]), 4),
        }
    return per_class, float(precision.mean()), float(recall.mean()), float(f1.mean())


def evaluate(
    model_path: str,
    data_path: str,
    out_dir: str,
    threshold: float = 0.5,
    resolution: int = 32,
    device: str | None = None,
    max_samples: int | None = 1000,
) -> dict:
    dataset = MachiningFeatureDataset(data_path, resolution=resolution)
    if max_samples is not None:
        dataset.samples = dataset.samples[:max_samples]
    _, _, test_ds = random_split_dataset(dataset)
    loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
    device_name = device if device else _get_device()
    torch_device = torch.device(device_name)
    model = load_model(model_path, device=torch_device)

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in loader:
            probs = torch.sigmoid(model(x.to(torch_device))).cpu()
            all_preds.append((probs >= threshold).float())
            all_targets.append(y.float())

    if all_preds:
        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
    else:
        preds = torch.zeros(0, NUM_CLASSES)
        targets = torch.zeros(0, NUM_CLASSES)

    per_class, macro_precision, macro_recall, macro_f1 = _metrics_from_predictions(preds, targets)
    cm = compute_multilabel_confusion(preds, targets, NUM_CLASSES)

    out_abs = os.path.abspath(out_dir)
    os.makedirs(out_abs, exist_ok=True)
    cm_path = os.path.join(out_abs, "confusion_matrix.npy")
    np.save(cm_path, cm)
    report = {
        "model_path": os.path.abspath(model_path),
        "data_path": os.path.abspath(data_path),
        "threshold": threshold,
        "resolution": resolution,
        "device": device_name,
        "max_samples": max_samples,
        "test_samples": len(test_ds),
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "per_class": per_class,
        "confusion_matrix_file": cm_path,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(out_abs, "eval_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained feature model.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default=os.path.join("checkpoints", "eval"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument(
        "--device",
        default=None,
        help="Device override: mps / cuda / cpu (default: auto-detect)",
    )
    args = parser.parse_args()
    device = args.device if args.device else _get_device()
    print(f"Device: {device}")
    report = evaluate(
        args.model,
        args.data,
        args.out,
        args.threshold,
        args.resolution,
        device,
        args.max_samples,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
