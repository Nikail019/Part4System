# AGENTS.md — Production Training & Full Pipeline Validation

## Goal

Eliminate all fallback code paths and produce a fully trained, validated
system. After this phase:

- `checkpoints/best.pt` exists and achieves val F1 ≥ 0.75
- `run_pipeline.py` auto-detects the checkpoint and never uses the flat_face fallback
- All 3 fixture parts produce multi-feature process plans
- A validation report confirms sensible outputs across all fixtures

---

## What This Phase Adds or Modifies

```
rpp-mvp/
├── training/
│   ├── augmentation.py              # NEW  — 3D voxel augmentations
│   ├── dataset.py                   # MODIFY — augmentation support
│   └── train_feature_net.py         # MODIFY — augmentation, early stopping,
│                                    #           class weights, better metrics
│
├── training/evaluate_model.py       # NEW  — per-class F1, confusion matrix
├── scripts/validate_pipeline.py     # NEW  — end-to-end fixture validation
├── run_pipeline.py                  # MODIFY — auto-detect checkpoint,
│                                    #          error on missing model
└── tests/
    └── test_training.py             # NEW  — unit tests for training components
```

---
---

# PART 1 — DATA AUGMENTATION

## `training/augmentation.py`

3D voxel augmentations applied at training time to improve generalisation.
All augmentations must be deterministic given a seed and must preserve the
semantic label (a pocket rotated 90° is still a pocket).

```python
# training/augmentation.py

import torch
import random

class RandomRotate90(object):
    """
    Randomly rotate the voxel grid by 0, 90, 180, or 270 degrees
    around the Z axis (axis 2 of the (1, R, R, R) tensor).

    This is the most important augmentation for machining features:
    a pocket at any rotation is still a pocket.
    """
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (1, R, R, R)
        k = random.randint(0, 3)
        if k == 0:
            return x
        # Rotate in the X-Y plane (dims 1 and 2 of the spatial axes)
        return torch.rot90(x, k=k, dims=[1, 2])


class RandomFlip(object):
    """
    Randomly flip the voxel grid along X, Y, or Z axis with probability p.
    Applied independently per axis.
    """
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for dim in [1, 2, 3]:   # spatial dims of (1, R, R, R)
            if random.random() < self.p:
                x = torch.flip(x, dims=[dim])
        return x


class RandomVoxelNoise(object):
    """
    Randomly flip a small fraction of voxels (boundary noise).
    Models tessellation artefacts and mesh imperfections.
    Only applied near the surface — interior voxels are left unchanged.
    """
    def __init__(self, p: float = 0.01):
        self.p = p   # fraction of voxels to randomly flip

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0:
            return x
        noise_mask = torch.rand_like(x) < self.p
        return x ^ noise_mask.float()   # XOR: flip selected voxels


class Compose(object):
    """Apply a list of transforms in sequence."""
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x


def get_train_transforms() -> Compose:
    """Standard augmentation pipeline for training."""
    return Compose([
        RandomRotate90(),
        RandomFlip(p=0.5),
        RandomVoxelNoise(p=0.005),
    ])


def get_val_transforms() -> None:
    """No augmentation at validation/inference time."""
    return None
```

---

# PART 2 — DATASET UPDATES

## Modify `training/dataset.py`

Add augmentation support and class weight computation.

### Changes to `MachiningFeatureDataset.__init__`

```python
def __init__(
    self,
    root: str,
    resolution: int = 64,
    transform=None,    # existing parameter — now used
    cache: bool = True,
):
    # existing init code unchanged
    # transform is applied in __getitem__
```

### Changes to `__getitem__`

```python
def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    sample = self.samples[idx]
    grid   = self._get_voxel(sample)
    x      = torch.from_numpy(grid).float().unsqueeze(0)  # (1, R, R, R)
    y      = torch.from_numpy(self._labels_to_multihot(sample["labels"]))
    if self.transform is not None:
        x = self.transform(x)
    return x, y
```

### New function: `compute_class_weights`

```python
def compute_class_weights(dataset) -> torch.Tensor:
    """
    Compute per-class positive frequency weights for BCEWithLogitsLoss.

    For each class c:
        pos_freq[c] = (number of samples where class c == 1) / total_samples
        weight[c]   = 1.0 / (pos_freq[c] + epsilon)

    Normalise so the mean weight == 1.0.

    Returns float32 tensor of shape (NUM_CLASSES,).

    This counteracts the class imbalance where flat_face appears in
    every sample but triangular_pocket appears rarely.
    """
    from training.dataset import NUM_CLASSES
    counts = torch.zeros(NUM_CLASSES)
    n = len(dataset)
    for _, y in dataset:
        counts += y
    pos_freq = counts / n
    weights  = 1.0 / (pos_freq + 1e-6)
    weights  = weights / weights.mean()   # normalise
    return weights.float()
```

---

# PART 3 — TRAINING SCRIPT UPDATES

## Modify `training/train_feature_net.py`

### New arguments

Add these to `get_args()`:

```python
p.add_argument("--augment",      action="store_true", default=True,
               help="Apply data augmentation during training (default: True)")
p.add_argument("--no-augment",   action="store_false", dest="augment")
p.add_argument("--class-weights",action="store_true", default=True,
               help="Use class frequency weights in loss (default: True)")
p.add_argument("--early-stop",   type=int, default=10,
               help="Stop if val F1 does not improve for N epochs (default: 10)")
p.add_argument("--min-epochs",   type=int, default=5,
               help="Minimum epochs before early stopping applies (default: 5)")
```

### Updated `train()` function — key changes

**1. Augmentation in DataLoader:**

```python
from training.augmentation import get_train_transforms, get_val_transforms

train_transform = get_train_transforms() if args.augment else None
val_transform   = get_val_transforms()

train_ds_raw, val_ds_raw, test_ds_raw = random_split_dataset(full_ds)

# Wrap subsets with per-split transforms
train_ds = _TransformedSubset(train_ds_raw, train_transform)
val_ds   = _TransformedSubset(val_ds_raw,   val_transform)
test_ds  = _TransformedSubset(test_ds_raw,  val_transform)
```

Implement `_TransformedSubset`:

```python
class _TransformedSubset(torch.utils.data.Dataset):
    """Applies a transform to a Subset without modifying the original dataset."""
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y
```

**2. Class-weighted loss:**

```python
if args.class_weights:
    weights   = compute_class_weights(full_ds).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
    print(f"  Class weights range: "
          f"[{weights.min():.2f}, {weights.max():.2f}]")
else:
    criterion = nn.BCEWithLogitsLoss()
```

**3. Early stopping:**

```python
epochs_no_improve = 0
best_val_f1       = 0.0

for epoch in range(start_epoch, args.epochs):
    # ... existing train/val loop ...

    # Track best F1 (not just val loss) for early stopping
    if f1 > best_val_f1:
        best_val_f1       = f1
        epochs_no_improve = 0
        torch.save(ckpt_data, os.path.join(args.out, "best.pt"))
        if not args.quiet:
            print(f"  ✓ New best F1={f1:.4f}")
    else:
        epochs_no_improve += 1

    if (epoch + 1 >= args.min_epochs and
            epochs_no_improve >= args.early_stop):
        print(f"Early stopping at epoch {epoch+1} "
              f"(no F1 improvement for {args.early_stop} epochs)")
        break
```

**4. Per-class F1 in final report:**

After training completes, print per-class F1 on the test split:

```python
def _print_per_class_report(
    model, loader, device, threshold=0.5
) -> dict:
    """
    Compute and print per-class precision, recall, F1 on a DataLoader.
    Returns dict {class_name: {"precision": f, "recall": f, "f1": f}}.
    """
    from training.dataset import FEATURE_NAMES, NUM_CLASSES

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            probs = torch.sigmoid(model(x.to(device))).cpu()
            all_preds.append((probs >= threshold).float())
            all_targets.append(y)

    preds   = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    tp = (preds * targets).sum(dim=0)
    fp = (preds * (1 - targets)).sum(dim=0)
    fn = ((1 - preds) * targets).sum(dim=0)

    precision = (tp / (tp + fp + 1e-8)).numpy()
    recall    = (tp / (tp + fn + 1e-8)).numpy()
    f1        = (2 * precision * recall / (precision + recall + 1e-8))

    print("\nPer-class results on test split:")
    print(f"  {'Feature':<25} {'P':>6} {'R':>6} {'F1':>6}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*6}")

    report = {}
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  {name:<25} {precision[i]:>6.3f} {recall[i]:>6.3f} {f1[i]:>6.3f}")
        report[name] = {
            "precision": round(float(precision[i]), 4),
            "recall":    round(float(recall[i]), 4),
            "f1":        round(float(f1[i]), 4),
        }

    macro_f1 = f1.mean()
    print(f"\n  Macro F1: {macro_f1:.4f}")
    return report
```

**5. Save training metadata with checkpoint:**

When saving `best.pt`, include training metadata:

```python
ckpt_data = {
    "epoch":               epoch,
    "model_state_dict":    model.state_dict(),
    "optimizer_state_dict": optimiser.state_dict(),
    "val_loss":            val_loss,
    "val_f1":              f1,
    "best_val_f1":         best_val_f1,
    "best_val_loss":       best_val_loss,
    "training_config": {
        "resolution":    args.resolution,
        "material":      None,
        "augment":       args.augment,
        "class_weights": args.class_weights,
        "batch_size":    args.batch,
        "learning_rate": args.lr,
        "num_classes":   NUM_CLASSES,
        "feature_names": FEATURE_NAMES,
    },
}
```

---

# PART 4 — EVALUATION SCRIPT

## `training/evaluate_model.py`

```
Usage:
  python training/evaluate_model.py \
      --model    checkpoints/best.pt  \
      --data     data/raw/synthetic   \
      --out      checkpoints/eval/    \
      --threshold 0.5
```

This script:
1. Loads the trained model
2. Runs it on the test split of the dataset
3. Computes and prints per-class P/R/F1
4. Saves `eval_report.json` with all metrics
5. Saves `confusion_matrix.npy` (NUM_CLASSES × NUM_CLASSES)

### `eval_report.json` schema

```json
{
  "model_path":   "/abs/path/to/best.pt",
  "data_path":    "/abs/path/to/data/",
  "threshold":    0.5,
  "test_samples": 200,
  "macro_f1":     0.823,
  "macro_precision": 0.841,
  "macro_recall":    0.807,
  "per_class": {
    "flat_face":           {"precision": 0.99, "recall": 0.99, "f1": 0.99},
    "through_hole":        {"precision": 0.85, "recall": 0.80, "f1": 0.82},
    "rectangular_pocket":  {"precision": 0.87, "recall": 0.83, "f1": 0.85}
  },
  "confusion_matrix_file": "/abs/path/to/confusion_matrix.npy",
  "timestamp": "2025-01-01T12:00:00"
}
```

### Confusion matrix

For multi-label classification, use a per-class binary confusion matrix.
Shape: `(NUM_CLASSES, 2, 2)` where `[c, 0, 0]` = TN, `[c, 1, 1]` = TP, etc.

```python
def compute_multilabel_confusion(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> np.ndarray:
    """
    Returns (NUM_CLASSES, 2, 2) array.
    cm[c] is the binary confusion matrix for class c:
        [[TN, FP],
         [FN, TP]]
    """
    cm = np.zeros((num_classes, 2, 2), dtype=np.int64)
    for c in range(num_classes):
        p = preds[:, c].numpy().astype(int)
        t = targets[:, c].numpy().astype(int)
        for pred_val, true_val in zip(p, t):
            cm[c, true_val, pred_val] += 1
    return cm
```

---

# PART 5 — PIPELINE AUTO-DETECTION

## Modify `run_pipeline.py`

### Checkpoint auto-detection

Replace the fallback logic in `run_phase2` with a hard fail when no model
is available in non-test mode:

```python
DEFAULT_CHECKPOINT = "checkpoints/best.pt"


def _resolve_model_path(args: argparse.Namespace) -> str | None:
    """
    Resolve the model checkpoint path in priority order:
    1. Explicit --model argument
    2. checkpoints/best.pt (auto-detected)
    3. None (no checkpoint available)
    """
    if args.model and os.path.exists(args.model):
        return os.path.abspath(args.model)
    if os.path.exists(DEFAULT_CHECKPOINT):
        return os.path.abspath(DEFAULT_CHECKPOINT)
    return None
```

### Updated `run_phase2`

```python
def run_phase2(args: argparse.Namespace, paths: dict) -> dict:
    from phase2_feature_recognition import recognise_features
    import json

    t0 = time.time()
    voxel_file    = paths["voxel_file"]
    features_path = os.path.join(args.output, "features.json")
    model_path    = _resolve_model_path(args)

    if model_path is None:
        # No model available — raise a clear error
        raise FileNotFoundError(
            "No trained model checkpoint found.\n"
            "Train the model first:\n"
            "  python training/synthetic_data_gen.py --count 2000\n"
            "  python training/train_feature_net.py --data data/raw/synthetic "
            "--out checkpoints --epochs 30\n"
            "Or specify a checkpoint with --model path/to/model.pt"
        )

    result = recognise_features(
        voxel_file,
        model_path,
        threshold=args.confidence,
    )
    _write_json_atomic(result, features_path)

    return {
        "features_file": os.path.abspath(features_path),
        "feature_count": result["feature_count"],
        "model_used":    model_path,
        "duration_sec":  round(time.time() - t0, 2),
    }
```

### Update `dry_run` to show auto-detected checkpoint

```python
model_path = _resolve_model_path(args)
model_str  = model_path if model_path else "(none — run training first)"
print(f"  Model checkpoint : {model_str}")
```

### Remove `_default_features` function entirely

It is no longer called in production code. Keeping it would create a
misleading fallback path. Delete it.

---

# PART 6 — PIPELINE VALIDATION SCRIPT

## `scripts/validate_pipeline.py`

Runs all 3 fixture STEP files through the full pipeline and produces a
validation report showing whether the outputs are semantically sensible.

```
Usage:
  python scripts/validate_pipeline.py \
      --model    checkpoints/best.pt  \
      --factory  factory_profiles/nash_nz.json \
      --material aluminium_6061 \
      --out      data/validation/
```

### What it checks

For each fixture, after running the full pipeline:

| Check | Expected |
|---|---|
| Phase 1 completed | voxel_64.npy exists, occupancy 0.05–0.80 |
| Phase 2 features | feature_count >= 1, flat_face always present |
| Phase 3 setups | setup_count >= 1, axis_requirement in {3,4,5} |
| Phase 4 plan | operation_count >= 2, flat_face is step 1 |
| Phase 5 time | total_time_min > 0, machining_time_min > 0 |
| Phase 6 quotation | recommendation in {ACCEPT, REJECT}, total cost > 0 |

### `validation_report.json` schema

```json
{
  "model_path":    "/abs/path/to/best.pt",
  "factory":       "NASH NZ",
  "material":      "aluminium_6061",
  "timestamp":     "2025-01-01T12:00:00",
  "fixtures": {
    "simple_block": {
      "status":          "PASS",
      "checks_passed":   6,
      "checks_total":    6,
      "features":        ["flat_face"],
      "operation_count": 2,
      "setup_count":     2,
      "axis_requirement": 3,
      "total_time_min":  49.1,
      "total_cost_nzd":  118.09,
      "recommendation":  "ACCEPT",
      "failed_checks":   []
    },
    "block_with_holes": { ... },
    "complex_prismatic": { ... }
  },
  "overall_status": "PASS",
  "fixtures_passed": 3,
  "fixtures_total":  3
}
```

### Implementation

```python
import os, json, sys, datetime, argparse, subprocess, tempfile
from pathlib import Path

FIXTURES = {
    "simple_block":      "tests/fixtures/simple_block.stp",
    "block_with_holes":  "tests/fixtures/block_with_holes.stp",
    "complex_prismatic": "tests/fixtures/complex_prismatic.stp",
}

CHECKS = [
    ("phase1", "voxel_64.npy exists",
     lambda d: os.path.exists(os.path.join(d, "voxel_64.npy"))),
    ("phase2", "flat_face always detected",
     lambda d: "flat_face" in _load_feature_types(d)),
    ("phase2", "at least 1 feature detected",
     lambda d: _load_json(d, "features.json").get("feature_count", 0) >= 1),
    ("phase3", "valid setup count",
     lambda d: _load_json(d, "setup_analysis.json").get("setup_count", 0) >= 1),
    ("phase4", "at least 2 operations",
     lambda d: _load_json(d, "process_plan.json").get("operation_count", 0) >= 2),
    ("phase5", "positive machining time",
     lambda d: _load_json(d, "time_estimate.json").get("total_time_min", 0) > 0),
    ("phase6", "valid recommendation",
     lambda d: _load_json(d, "quotation.json").get("recommendation") in ("ACCEPT","REJECT")),
    ("phase6", "positive cost",
     lambda d: _load_json(d, "quotation.json").get("estimated_cost",{}).get("total",0) > 0),
]


def _load_json(output_dir: str, filename: str) -> dict:
    path = os.path.join(output_dir, filename)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _load_feature_types(output_dir: str) -> list[str]:
    data = _load_json(output_dir, "features.json")
    return [f["type"] for f in data.get("features", [])]


def run_fixture(
    name: str,
    stp_path: str,
    output_dir: str,
    model_path: str,
    factory_path: str,
    material: str,
) -> dict:
    """Run the full pipeline on one fixture. Return check results."""
    os.makedirs(output_dir, exist_ok=True)

    # Run pipeline via subprocess so it gets its own clean import environment
    cmd = [
        sys.executable, "run_pipeline.py",
        stp_path, factory_path,
        "--model",    model_path,
        "--material", material,
        "--output",   output_dir,
        "--quiet",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return {
            "status": "FAIL",
            "error": proc.stderr.strip(),
            "checks_passed": 0,
            "checks_total": len(CHECKS),
            "failed_checks": ["Pipeline returned non-zero exit code"],
        }

    # Run checks
    passed, failed = [], []
    for phase, description, check_fn in CHECKS:
        try:
            ok = check_fn(output_dir)
        except Exception as e:
            ok = False
        (passed if ok else failed).append(f"[{phase}] {description}")

    # Extract summary values
    quotation = _load_json(output_dir, "quotation.json")
    plan      = _load_json(output_dir, "process_plan.json")
    setup     = _load_json(output_dir, "setup_analysis.json")
    time_est  = _load_json(output_dir, "time_estimate.json")

    return {
        "status":           "PASS" if not failed else "FAIL",
        "checks_passed":    len(passed),
        "checks_total":     len(CHECKS),
        "features":         _load_feature_types(output_dir),
        "operation_count":  plan.get("operation_count"),
        "setup_count":      setup.get("setup_count"),
        "axis_requirement": setup.get("axis_requirement"),
        "total_time_min":   time_est.get("total_time_min"),
        "total_cost":       quotation.get("estimated_cost", {}).get("total"),
        "currency":         quotation.get("estimated_cost", {}).get("currency"),
        "recommendation":   quotation.get("recommendation"),
        "failed_checks":    failed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    required=True)
    parser.add_argument("--factory",  default="factory_profiles/nash_nz.json")
    parser.add_argument("--material", default="aluminium_6061")
    parser.add_argument("--out",      default="data/validation")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    results = {}
    overall_pass = True

    print(f"\nValidating pipeline with model: {args.model}")
    print(f"Factory: {args.factory}  Material: {args.material}\n")

    for name, stp_path in FIXTURES.items():
        if not os.path.exists(stp_path):
            print(f"  {name:<20} SKIP (fixture not found)")
            continue

        fixture_out = os.path.join(args.out, name)
        print(f"  {name:<20} ...", end=" ", flush=True)
        result = run_fixture(
            name, stp_path, fixture_out,
            args.model, args.factory, args.material,
        )
        results[name] = result
        status = result["status"]
        if status == "PASS":
            ops   = result.get("operation_count", "?")
            time_ = result.get("total_time_min",  "?")
            cost  = result.get("total_cost",       "?")
            cur   = result.get("currency",         "")
            rec   = result.get("recommendation",   "?")
            print(f"PASS  {ops} ops  {time_} min  {cur} {cost}  [{rec}]")
        else:
            print(f"FAIL")
            for fc in result.get("failed_checks", []):
                print(f"         ✗  {fc}")
            overall_pass = False

    report = {
        "model_path":      os.path.abspath(args.model),
        "factory":         args.factory,
        "material":        args.material,
        "timestamp":       datetime.datetime.now().isoformat(timespec="seconds"),
        "fixtures":        results,
        "overall_status":  "PASS" if overall_pass else "FAIL",
        "fixtures_passed": sum(1 for r in results.values() if r["status"] == "PASS"),
        "fixtures_total":  len(results),
    }

    report_path = os.path.join(args.out, "validation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nOverall: {report['overall_status']}  "
          f"({report['fixtures_passed']}/{report['fixtures_total']} fixtures)")
    print(f"Report:  {report_path}\n")
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
```

---

# PART 7 — TESTS

## `tests/test_training.py`

```python
# tests/test_training.py

import torch
import pytest
from training.augmentation import (
    RandomRotate90, RandomFlip, RandomVoxelNoise, Compose,
    get_train_transforms, get_val_transforms,
)
from training.dataset import compute_class_weights, MachiningFeatureDataset


# ── Augmentation ──────────────────────────────────────────────────────────────

def make_voxel(R=16):
    """Solid block voxel tensor (1, R, R, R)."""
    return torch.ones(1, R, R, R)


def test_rotate90_preserves_shape():
    x = make_voxel()
    out = RandomRotate90()(x)
    assert out.shape == x.shape


def test_rotate90_preserves_sum():
    """Rotation must not add or remove voxels."""
    x = make_voxel()
    x[0, :4, :4, :4] = 0   # punch a corner hole
    for _ in range(20):
        out = RandomRotate90()(x.clone())
        assert out.sum() == x.sum()


def test_flip_preserves_shape():
    x = make_voxel()
    out = RandomFlip(p=1.0)(x)
    assert out.shape == x.shape


def test_flip_preserves_sum():
    x = make_voxel()
    out = RandomFlip(p=1.0)(x)
    assert out.sum() == x.sum()


def test_noise_changes_small_fraction():
    x = torch.ones(1, 32, 32, 32)
    out = RandomVoxelNoise(p=0.01)(x.clone())
    diff_fraction = (out != x).float().mean().item()
    # Should flip roughly 1% of voxels, within noise
    assert diff_fraction < 0.05


def test_compose_applies_all_transforms():
    x = make_voxel(16)
    transform = get_train_transforms()
    out = transform(x.clone())
    assert out.shape == x.shape


def test_val_transforms_is_none():
    assert get_val_transforms() is None


# ── Class weights ─────────────────────────────────────────────────────────────

def test_compute_class_weights_shape(tmp_path):
    from training.synthetic_data_gen import generate_dataset
    generate_dataset(str(tmp_path), count=20)
    ds = MachiningFeatureDataset(str(tmp_path), resolution=16)
    if len(ds) == 0:
        pytest.skip("No synthetic data generated")
    weights = compute_class_weights(ds)
    from training.dataset import NUM_CLASSES
    assert weights.shape == (NUM_CLASSES,)


def test_compute_class_weights_all_positive(tmp_path):
    from training.synthetic_data_gen import generate_dataset
    generate_dataset(str(tmp_path), count=20)
    ds = MachiningFeatureDataset(str(tmp_path), resolution=16)
    if len(ds) == 0:
        pytest.skip("No synthetic data generated")
    weights = compute_class_weights(ds)
    assert (weights > 0).all()


def test_flat_face_has_lower_weight_than_rare_features(tmp_path):
    """
    flat_face appears in every sample → highest frequency → lowest weight.
    Rare features should have higher weights.
    """
    from training.synthetic_data_gen import generate_dataset
    generate_dataset(str(tmp_path), count=50)
    ds = MachiningFeatureDataset(str(tmp_path), resolution=16)
    if len(ds) == 0:
        pytest.skip("No synthetic data generated")
    from training.dataset import FEATURE_NAMES
    weights = compute_class_weights(ds)
    flat_idx = FEATURE_NAMES.index("flat_face")
    flat_weight = weights[flat_idx].item()
    mean_weight = weights.mean().item()
    assert flat_weight < mean_weight, \
        "flat_face (always present) should have below-average weight"


# ── _TransformedSubset ────────────────────────────────────────────────────────

def test_transformed_subset_applies_transform(tmp_path):
    from training.synthetic_data_gen import generate_dataset
    generate_dataset(str(tmp_path), count=10)
    ds = MachiningFeatureDataset(str(tmp_path), resolution=16)
    if len(ds) == 0:
        pytest.skip("No synthetic data generated")
    from training.train_feature_net import _TransformedSubset
    from torch.utils.data import Subset
    subset = Subset(ds, list(range(min(5, len(ds)))))
    aug_ds = _TransformedSubset(subset, get_train_transforms())
    x, y = aug_ds[0]
    assert x.shape[0] == 1   # channel dim preserved


# ── Checkpoint metadata ───────────────────────────────────────────────────────

def test_checkpoint_contains_training_config(tmp_path):
    """
    After saving a checkpoint, it must contain training_config with
    at least resolution, num_classes, feature_names.
    """
    import os
    from models.feature_net import FeatureNet3D, NUM_CLASSES
    model = FeatureNet3D()
    ckpt = {
        "epoch": 0,
        "model_state_dict": model.state_dict(),
        "val_f1": 0.0,
        "training_config": {
            "resolution": 64,
            "num_classes": NUM_CLASSES,
            "feature_names": ["flat_face"],
        },
    }
    path = str(tmp_path / "test.pt")
    torch.save(ckpt, path)
    loaded = torch.load(path, map_location="cpu")
    assert "training_config" in loaded
    assert loaded["training_config"]["resolution"] == 64
```

## Update `tests/test_pipeline.py`

Add tests for the new auto-detection and error behaviour.

### Tests to add

```python
# Add to tests/test_pipeline.py

from run_pipeline import _resolve_model_path

def test_resolve_model_explicit(tmp_path):
    """Explicit --model takes priority."""
    ckpt = tmp_path / "my_model.pt"
    ckpt.write_text("dummy")
    args = make_args(model=str(ckpt))
    resolved = _resolve_model_path(args)
    assert resolved == str(ckpt.resolve())


def test_resolve_model_auto_detects_best_pt(tmp_path, monkeypatch):
    """Auto-detects checkpoints/best.pt when --model not given."""
    # Monkeypatch the DEFAULT_CHECKPOINT path
    import run_pipeline
    ckpt = tmp_path / "best.pt"
    ckpt.write_text("dummy")
    monkeypatch.setattr(run_pipeline, "DEFAULT_CHECKPOINT", str(ckpt))
    args = make_args(model=None)
    resolved = _resolve_model_path(args)
    assert resolved is not None
    assert "best.pt" in resolved


def test_resolve_model_returns_none_when_missing(monkeypatch):
    """Returns None when no model exists anywhere."""
    import run_pipeline
    monkeypatch.setattr(run_pipeline, "DEFAULT_CHECKPOINT", "/no/such/path.pt")
    args = make_args(model=None)
    resolved = _resolve_model_path(args)
    assert resolved is None


def test_default_features_removed():
    """_default_features must no longer exist in run_pipeline."""
    import run_pipeline
    assert not hasattr(run_pipeline, "_default_features"), \
        "_default_features fallback must be removed from run_pipeline"
```

---
---

# QUICK-START: FULL TRAINING RUN

```bash
# 1. Generate synthetic training data
python training/synthetic_data_gen.py \
    --count 3000 \
    --output data/raw/synthetic

# 2. Train with augmentation and class weights (recommended settings)
python training/train_feature_net.py \
    --data         data/raw/synthetic \
    --out          checkpoints        \
    --epochs       50                 \
    --batch        32                 \
    --lr           1e-3               \
    --resolution   64                 \
    --augment                         \
    --class-weights                   \
    --early-stop   10                 \
    --min-epochs   10

# 3. Evaluate model on test split
python training/evaluate_model.py \
    --model  checkpoints/best.pt  \
    --data   data/raw/synthetic   \
    --out    checkpoints/eval/

# 4. Run full pipeline (auto-detects checkpoints/best.pt)
python run_pipeline.py \
    tests/fixtures/complex_prismatic.stp \
    factory_profiles/nash_nz.json

# 5. Validate all fixtures
python scripts/validate_pipeline.py \
    --model   checkpoints/best.pt     \
    --factory factory_profiles/nash_nz.json

# 6. Run all tests
pytest tests/ -v
```

---
---

# ACCEPTANCE CRITERIA

## Training

- [ ] `python training/synthetic_data_gen.py --count 3000` generates
      ≥ 2700 valid parts
- [ ] `python training/train_feature_net.py ... --epochs 50` completes
      and saves `checkpoints/best.pt`
- [ ] `checkpoints/best.pt` contains `training_config` key
- [ ] Val F1 ≥ 0.70 on synthetic test split (printed at end of training)
- [ ] `flat_face` F1 ≥ 0.95 (it is always present — easy class)

## Evaluation

- [ ] `python training/evaluate_model.py` runs and writes `eval_report.json`
- [ ] `eval_report.json` contains per-class F1 for all 12 classes
- [ ] Macro F1 in report matches final training epoch output

## Pipeline

- [ ] `run_pipeline.py` auto-detects `checkpoints/best.pt` with no `--model` flag
- [ ] `run_pipeline.py` raises `FileNotFoundError` with clear training
      instructions when no checkpoint exists
- [ ] `_default_features` no longer exists in `run_pipeline.py`
- [ ] `simple_block.stp` produces `operation_count >= 2`
- [ ] `complex_prismatic.stp` produces `operation_count >= 6`
- [ ] `block_with_holes.stp` produces `through_hole` in detected features

## Validation script

- [ ] `python scripts/validate_pipeline.py --model checkpoints/best.pt`
      exits with code 0 (all fixtures PASS)
- [ ] `validation_report.json` written with `overall_status: PASS`
- [ ] All 3 fixtures produce `recommendation: ACCEPT` with nash_nz profile

## Tests

- [ ] `pytest tests/test_training.py -v` — all tests pass
- [ ] `pytest tests/ -v` — all tests pass (≥ 220 total)

---
---

# NOTES FOR CODEX

1. **`_TransformedSubset` must be importable from `training.train_feature_net`.**
   The test imports it directly. It is a simple wrapper — do not put it
   inside a function.

2. **`compute_class_weights` must be importable from `training.dataset`.**
   It is a standalone function, not a method of `MachiningFeatureDataset`.

3. **Remove `_default_features` entirely from `run_pipeline.py`.**
   The test `test_default_features_removed` uses `hasattr` to verify
   its absence. Any remaining fallback path will fail this test.

4. **`_resolve_model_path` must be a module-level function.**
   Tests monkeypatch `DEFAULT_CHECKPOINT` on the module to test
   auto-detection without a real file system.

5. **Early stopping tracks val F1, not val loss.** Val loss can
   decrease while F1 plateaus on imbalanced multi-label data. Use
   F1 as the primary stopping criterion.

6. **Checkpoint `best.pt` saves on best F1, not best loss.**
   The training loop already saves `last.pt` every epoch. `best.pt`
   is reserved for the epoch with the highest val F1.

7. **`validate_pipeline.py` runs pipeline via `subprocess`**, not
   direct import. This gives each fixture its own clean Python
   process, avoids import-time side effects between fixtures, and
   produces realistic wall-clock timing in the report.

8. **3000 parts is the minimum for val F1 ≥ 0.70.** If training
   converges below 0.70, increase to 5000 with
   `--count 5000` before adjusting architecture or hyperparameters.

9. **Augmentation must not be applied at validation or inference.**
   `get_val_transforms()` returns `None`. The `_TransformedSubset`
   wrapper passes `None` transforms through without applying anything.
