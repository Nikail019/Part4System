# AGENTS.md — Optimised From-Scratch Training

## Goal

Train FeatureNet3D from scratch with all optimisations applied.
Target: production checkpoint in under 15 minutes on CPU.

## What Changes From Previous Attempts

| Problem | Fix |
|---|---|
| Epoch 1 slow (STEP→voxel on-the-fly) | Pre-warm cache before training starts |
| Resolution 64³ wastes disk + time | Resolution 32³ throughout |
| 3000 parts too large for disk | 200 parts (sufficient with augmentation) |
| Model FC layer oversized for 32³ | Reduce FC 512→256 for 32³ input |
| No progress visibility | tqdm per-batch progress bar |

---

## Files to Modify or Create

```
rpp-mvp/
├── scripts/
│   └── prewarm_cache.py          # NEW — pre-compute all voxels before training
├── models/
│   └── feature_net.py            # MODIFY — smaller FC layer for 32³ input
├── training/
│   └── train_feature_net.py      # MODIFY — add --pre-warm flag, tqdm batch bar
└── tests/
    └── test_training.py          # MODIFY — update resolution references
```

---

## Part 1 — Pre-warm Cache Script

### `scripts/prewarm_cache.py`

Runs Phase 1 on every STEP file in the dataset and caches the result
**before** training starts. Training epochs then read pre-computed `.npy`
files instead of running cadquery, making every epoch equally fast.

```python
#!/usr/bin/env python3
"""
Pre-compute voxel cache for a synthetic dataset directory.

Usage:
    python scripts/prewarm_cache.py \
        --data       data/raw/synthetic \
        --resolution 32                 \
        --max-samples 200

This script is idempotent: parts that already have a cached
voxel_{R}.npy are skipped automatically.
"""

import argparse
import os
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from phase1_voxeliser import process_step_file


def prewarm(data_dir: str, resolution: int, max_samples: int | None) -> None:
    root = Path(data_dir)

    # Collect valid part directories
    parts = sorted([
        d for d in root.iterdir()
        if d.is_dir()
        and (d / "part.stp").exists()
        and (d / "labels.json").exists()
    ])

    if max_samples is not None:
        parts = parts[:max_samples]

    voxel_filename = f"voxel_{resolution}.npy"
    to_process = [p for p in parts if not (p / voxel_filename).exists()]

    print(f"Parts total   : {len(parts)}")
    print(f"Already cached: {len(parts) - len(to_process)}")
    print(f"To process    : {len(to_process)}")

    if not to_process:
        print("Cache already complete. Nothing to do.")
        return

    errors = 0
    for part_dir in tqdm(to_process, desc=f"Voxelising at {resolution}³"):
        try:
            process_step_file(
                str(part_dir / "part.stp"),
                str(part_dir),
                resolution=resolution,
            )
        except Exception as e:
            errors += 1
            tqdm.write(f"  SKIP {part_dir.name}: {e}")

    print(f"\nDone. Cached: {len(to_process) - errors}  Errors: {errors}")
    if errors > 0:
        print(f"Note: {errors} parts failed voxelisation and will be "
              f"skipped by the dataset loader automatically.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        default="data/raw/synthetic")
    parser.add_argument("--resolution",  type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None,
                        dest="max_samples")
    args = parser.parse_args()
    prewarm(args.data, args.resolution, args.max_samples)
```

---

## Part 2 — Smaller Model for 32³

### Modify `models/feature_net.py`

The existing `FeatureNet3D` architecture uses a 512-unit FC layer sized
for 64³ input. At 32³, `AdaptiveAvgPool3d(4)` still outputs `128×4×4×4 = 8192`
values — the same as 64³. Reduce FC to 256 units to match the smaller
input scale and train faster.

Find the classifier definition:

```python
self.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(128 * 4 * 4 * 4, 512),   # ← change 512 to 256
    nn.BatchNorm1d(512),                 # ← change 512 to 256
    nn.ReLU(inplace=True),
    nn.Dropout(p=dropout),
    nn.Linear(512, num_classes),         # ← change 512 to 256
)
```

Replace with:

```python
self.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(128 * 4 * 4 * 4, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(inplace=True),
    nn.Dropout(p=dropout),
    nn.Linear(256, num_classes),
)
```

Add a `hidden_dim` parameter so the architecture is configurable:

```python
def __init__(
    self,
    num_classes: int = NUM_CLASSES,
    dropout: float = 0.5,
    hidden_dim: int = 256,          # 256 for 32³, 512 for 64³
):
    super().__init__()
    self.num_classes = num_classes
    self.encoder = nn.Sequential(...)   # unchanged
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(128 * 4 * 4 * 4, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(hidden_dim, num_classes),
    )
```

---

## Part 3 — Training Script: Pre-warm Flag + Progress Bar

### Modify `training/train_feature_net.py`

#### Add `--pre-warm` argument

```python
p.add_argument("--pre-warm", action="store_true", default=False,
    help="Pre-compute voxel cache before training starts")
```

#### Add pre-warm call at the start of `train()`

Insert after dataset is created, before DataLoader construction:

```python
if args.pre_warm:
    from scripts.prewarm_cache import prewarm
    print("Pre-warming voxel cache...")
    prewarm(args.data, args.resolution, args.max_samples)
    print("Cache ready. Starting training.\n")
```

#### Add per-batch tqdm progress bar

Replace the plain `for x, y in train_loader:` loop with:

```python
from tqdm import tqdm

batch_bar = tqdm(
    train_loader,
    desc=f"Epoch {epoch+1:3d}/{args.epochs} train",
    leave=False,
    unit="batch",
)
for x, y in batch_bar:
    x, y = x.to(device), y.to(device)
    optimiser.zero_grad()
    logits = model(x)
    loss   = criterion(logits, y)
    loss.backward()
    optimiser.step()
    train_loss += loss.item() * x.size(0)
    batch_bar.set_postfix(loss=f"{loss.item():.4f}")
```

#### Print epoch summary on one line

```python
print(
    f"Epoch {epoch+1:3d}/{args.epochs} | "
    f"train={train_loss:.4f} | "
    f"val={val_loss:.4f} | "
    f"F1={f1:.4f}"
    + (" ✓ best" if f1 > best_val_f1 else "")
)
```

---

## Part 4 — Training Commands

Run these in order.

### Step 1: Clear stale cache (if not already done)

```bash
find data/raw/synthetic -name "voxel_*.npy" -delete
find data/raw/synthetic -name "mesh.stl" -delete
find data/raw/synthetic -name "metadata.json" -delete
df -h .
```

### Step 2: Pre-warm cache at resolution 32 for 200 parts

This runs once, writes 200 × ~32KB = ~6.4MB of voxel files.
Takes roughly 5–8 minutes (cadquery voxelisation per part).

```bash
python scripts/prewarm_cache.py \
    --data        data/raw/synthetic \
    --resolution  32                 \
    --max-samples 200
```

Expected output:
```
Parts total   : 3000
Already cached: 0
To process    : 200
Voxelising at 32³: 100%|████████████| 200/200 [05:30<00:00,  1.65s/it]
Done. Cached: 200  Errors: 0
```

Check disk impact:
```bash
du -sh data/raw/synthetic
# Expected: ~415MB (original STEPs + 6MB new voxels — negligible increase)
```

### Step 3: Train

All 200 voxels are pre-cached. Every epoch reads `.npy` files directly.

```bash
python training/train_feature_net.py \
    --data          data/raw/synthetic \
    --out           checkpoints        \
    --epochs        40                 \
    --batch         32                 \
    --lr            1e-3               \
    --resolution    32                 \
    --max-samples   200                \
    --augment                          \
    --class-weights                    \
    --early-stop    8                  \
    --min-epochs    8                  \
    --workers       0                  \
    2>&1 | tee checkpoints/train_log.txt
```

Expected timing per epoch after cache warmup: **20–40 seconds on CPU**.
Expected total training time: **8–15 minutes**.

Expected output:
```
Device: cpu
Dataset: 200 parts
  Train: 160  Val: 20  Test: 20
Class weights range: [0.18, 5.20]

Epoch   1/ 40 | train=0.5821 | val=0.5104 | F1=0.2831
Epoch   2/ 40 | train=0.4203 | val=0.3892 | F1=0.4512 ✓ best
...
Epoch  16/ 40 | train=0.1823 | val=0.2101 | F1=0.7124 ✓ best
Epoch  17/ 40 | train=0.1791 | val=0.2134 | F1=0.7089
...
Early stopping at epoch 24
```

### Step 4: Evaluate

```bash
python training/evaluate_model.py \
    --model checkpoints/best.pt \
    --data  data/raw/synthetic  \
    --out   checkpoints/eval/
```

### Step 5: Validate full pipeline

```bash
python scripts/validate_pipeline.py \
    --model   checkpoints/best.pt \
    --factory factory_profiles/nash_nz.json \
    --out     data/validation/
```

### Step 6: Run full test suite

```bash
pytest tests/ -v
```

---

## Part 5 — Update `DEFAULT_RESOLUTION`

### `run_pipeline.py`

```python
DEFAULT_RESOLUTION = 32   # updated from 64
```

---

## Acceptance Criteria

- [ ] `scripts/prewarm_cache.py` runs and caches 200 voxels at 32³
      in `data/raw/synthetic/` with no disk errors
- [ ] `df -h .` shows ≥ 3GB free after pre-warm completes
- [ ] Training starts immediately after pre-warm (no cadquery calls
      during epoch 1)
- [ ] Per-epoch time ≤ 60 seconds on CPU after epoch 1
- [ ] `checkpoints/best.pt` created and contains `training_config`
      with `resolution: 32` and `hidden_dim: 256`
- [ ] `checkpoints/eval/eval_report.json` shows macro F1 ≥ 0.60
- [ ] `flat_face` F1 ≥ 0.90
- [ ] `data/validation/validation_report.json` shows `overall_status: PASS`
- [ ] `run_pipeline.py` uses `DEFAULT_RESOLUTION = 32`
- [ ] `pytest tests/ -v` — all tests pass

---

## Notes for Codex

1. **`prewarm_cache.py` is idempotent.** Parts that already have
   `voxel_32.npy` are skipped. Safe to re-run if interrupted.

2. **Cache warmup errors are non-fatal.** If a STEP file fails
   voxelisation (degenerate geometry, cadquery error), log it and
   continue. The dataset loader already skips parts without cache files.

3. **`hidden_dim=256` must be saved in `training_config`** inside the
   checkpoint so `load_model()` can reconstruct the exact architecture.
   Update `load_model()` to read `hidden_dim` from the checkpoint:

   ```python
   def load_model(checkpoint_path, num_classes=NUM_CLASSES, device="cpu"):
       state = torch.load(checkpoint_path, map_location=device)
       config = state.get("training_config", {})
       hidden_dim = config.get("hidden_dim", 256)
       model = FeatureNet3D(num_classes=num_classes, hidden_dim=hidden_dim)
       weights = state.get("model_state_dict", state)
       model.load_state_dict(weights)
       model.to(device).eval()
       return model
   ```

4. **Tests that hardcode `voxel_64.npy` or `resolution=64` must be
   updated** to use `DEFAULT_RESOLUTION = 32`. Search:
   ```bash
   grep -rn "voxel_64\|resolution.*64\b" tests/
   ```

5. **The prewarm script must be importable as a module** (the training
   script calls `from scripts.prewarm_cache import prewarm`). Ensure
   `scripts/__init__.py` exists.

6. **200 parts × augmentation = ~1600 effective training samples.**
   With `RandomRotate90` (4 rotations) and `RandomFlip` (2 flips per
   axis), the model sees meaningfully different views each epoch.
   This is sufficient for the 12-class problem.
