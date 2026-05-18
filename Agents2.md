# AGENTS.md — Phase 2: Voxel → Feature Recognition

## Context

Phase 1 is complete. `phase1_voxeliser.py` reliably converts STEP files into
`voxel_{R}.npy` grids and `metadata.json`. The cadquery backend is active;
pythonocc-core is not installed.

Phase 2 implements a 3D CNN that reads a voxel grid and outputs a list of
machining feature types present in the part. This feeds directly into Phase 4
(process plan generation), where each detected feature is mapped to a sequence
of machining operations.

---

## What Phase 2 Must Deliver

```
INPUT:  voxel_64.npy   (64, 64, 64) bool ndarray from Phase 1

OUTPUT: features.json
        {
          "features": [
            {"type": "through_hole",        "confidence": 0.94},
            {"type": "rectangular_pocket",  "confidence": 0.87}
          ],
          "feature_count": 2,
          "all_scores": { ... all 12 classes ... },
          "threshold": 0.5,
          "voxel_file": "/abs/path/to/voxel_64.npy",
          "model_path": "/abs/path/to/best.pt"
        }
```

This is **multi-label classification** — a single part can contain multiple
feature types simultaneously. The model outputs a probability per class;
features above a confidence threshold are reported.

---

## Feature Classes

Use these 12 classes for the MVP.

| ID | Class name              | Typical operation(s)               |
|----|-------------------------|------------------------------------|
|  0 | `through_hole`          | centre drill → drill               |
|  1 | `blind_hole`            | centre drill → drill → peck        |
|  2 | `rectangular_pocket`    | rough endmill → finish endmill     |
|  3 | `circular_pocket`       | rough endmill → finish endmill     |
|  4 | `rectangular_slot`      | rough endmill → finish endmill     |
|  5 | `circular_slot`         | rough endmill → finish endmill     |
|  6 | `rectangular_step`      | face mill → shoulder mill          |
|  7 | `chamfer`               | chamfer mill                       |
|  8 | `fillet`                | ball endmill                       |
|  9 | `boss`                  | rough endmill → finish endmill     |
| 10 | `flat_face`             | face mill                          |
| 11 | `triangular_pocket`     | rough endmill → finish endmill     |

`NUM_CLASSES = 12`

---

## Repository Additions

```
rpp-mvp/
├── phase2_feature_recognition.py    # IMPLEMENT — inference module
├── models/
│   └── feature_net.py               # IMPLEMENT — 3D CNN architecture
├── training/
│   ├── train_feature_net.py         # IMPLEMENT — training script
│   ├── dataset.py                   # IMPLEMENT — data loading
│   └── synthetic_data_gen.py        # IMPLEMENT — synthetic part generator
├── checkpoints/
│   └── .gitkeep
└── tests/
    └── test_phase2.py               # IMPLEMENT — unit tests
```

---
---

# DATA PIPELINE

---

## Synthetic Data Generator — `training/synthetic_data_gen.py`

Since MFCAD++ requires manual download, the synthetic generator is the
primary training data source for the MVP. It uses cadquery (already
installed) to programmatically create labelled parts.

```
Usage:
  python training/synthetic_data_gen.py --output data/raw/synthetic --count 2000

Generates per part:
  data/raw/synthetic/00000/part.stp
  data/raw/synthetic/00000/labels.json   {"labels": ["through_hole", "flat_face"]}
```

### Feature name list and constants

```python
NUM_CLASSES = 12
FEATURE_NAMES = [
    "through_hole", "blind_hole", "rectangular_pocket", "circular_pocket",
    "rectangular_slot", "circular_slot", "rectangular_step", "chamfer",
    "fillet", "boss", "flat_face", "triangular_pocket",
]
FEATURE_TO_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}
```

### Base block generator

```python
def make_base_block() -> tuple:
    """Random block 60–150 x 60–120 x 30–80 mm. Returns (workplane, params)."""
    import cadquery as cq, random
    x = random.uniform(60, 150)
    y = random.uniform(60, 120)
    z = random.uniform(30,  80)
    return cq.Workplane("XY").box(x, y, z), {"x": x, "y": y, "z": z}
```

### Feature adder functions

Implement one function per feature. Each accepts `(wp, block)` and returns
a modified `cq.Workplane`. Raise `ValueError` or allow cadquery to raise if
the feature does not fit the block geometry — the caller will catch and skip.

```python
def add_through_hole(wp, block):
    """Ø6–20 mm through hole, randomly offset on top face."""

def add_blind_hole(wp, block):
    """Ø6–20 mm blind hole, depth 20–60% of block Z."""

def add_rectangular_pocket(wp, block):
    """Pocket 20–60% of face area, depth 20–40% of Z."""

def add_circular_pocket(wp, block):
    """Circular pocket, radius 15–35% of min(x,y)."""

def add_rectangular_slot(wp, block):
    """Slot running full Y-length of top face."""

def add_circular_slot(wp, block):
    """Thin annular groove on top face."""

def add_rectangular_step(wp, block):
    """Step cut from one side face."""

def add_chamfer(wp, block):
    """Chamfer on top face edges, size 2–8 mm."""

def add_fillet(wp, block):
    """Fillet on vertical edges, radius 2–10 mm."""

def add_boss(wp, block):
    """Cylindrical boss protruding from top face."""

def add_flat_face(wp, block):
    """No geometry change — flat face always present on any block."""
    return wp

def add_triangular_pocket(wp, block):
    """Equilateral triangular pocket on top face."""

FEATURE_ADDERS = {name: globals()[f"add_{name}"] for name in FEATURE_NAMES}
```

### Part generation and dataset loop

```python
def generate_part(min_features=1, max_features=4) -> tuple:
    """
    Build a random part with 1–4 features stacked on a base block.
    flat_face is always included.
    Returns (workplane, list_of_label_strings).
    """
    wp, block = make_base_block()
    labels = ["flat_face"]
    candidates = [f for f in FEATURE_NAMES if f != "flat_face"]
    chosen = random.sample(candidates, k=random.randint(min_features, max_features))
    for feat in chosen:
        try:
            wp = FEATURE_ADDERS[feat](wp, block)
            labels.append(feat)
        except Exception:
            pass   # feature did not fit — skip silently
    return wp, labels


def generate_dataset(output_dir: str, count: int = 2000) -> None:
    """
    Generate `count` labelled parts into output_dir.
    Retries up to count*5 times to reach target count.
    Writes manifest.json on completion.
    """
    # implement with tqdm progress bar
    # each successful part saved as {output_dir}/{i:05d}/part.stp
    #                              {output_dir}/{i:05d}/labels.json
    # manifest.json records total, date, class distribution
```

---

## Dataset Loader — `training/dataset.py`

```python
class MachiningFeatureDataset(Dataset):
    """
    Loads (voxel_tensor, multihot_label) pairs.

    Directory layout:
        root/00000/part.stp
        root/00000/labels.json   {"labels": [...]}
        root/00000/voxel_64.npy  (cached by Phase 1, generated on first access)

    Args:
        root       : path to directory containing numbered part subdirectories
        resolution : voxel resolution passed to phase1_voxeliser (default 64)
        transform  : optional callable applied to the voxel tensor
        cache      : if True, cache voxel .npy alongside part.stp (default True)
    """

    def __init__(self, root, resolution=64, transform=None, cache=True): ...

    def _scan_root(self) -> list[dict]:
        """Return list of valid sample dicts with keys: step_path, labels, part_dir."""

    def _get_voxel(self, sample: dict) -> np.ndarray:
        """Load cached voxel or run Phase 1 and cache result."""

    def _labels_to_multihot(self, labels: list[str]) -> np.ndarray:
        """Convert label list to float32 multi-hot vector of length NUM_CLASSES."""

    def __len__(self): ...
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (x, y) where x.shape=(1,R,R,R), y.shape=(NUM_CLASSES,)."""


def random_split_dataset(dataset, train_frac=0.80, val_frac=0.10, seed=42):
    """Return (train_subset, val_subset, test_subset)."""
```

---
---

# MODEL — `models/feature_net.py`

```python
class FeatureNet3D(nn.Module):
    """
    3D CNN for multi-label machining feature recognition.

    Input  : (B, 1, R, R, R) float32
    Output : (B, NUM_CLASSES) float32  — raw logits; apply sigmoid at inference

    Architecture:
        Conv3d(1→32, k=3, pad=1) → BN3d → ReLU → MaxPool3d(2)   # R → R/2
        Conv3d(32→64, k=3, pad=1) → BN3d → ReLU → MaxPool3d(2)  # R/2 → R/4
        Conv3d(64→128, k=3, pad=1) → BN3d → ReLU → AdaptiveAvgPool3d(4)
        Flatten
        Linear(128*64, 512) → BN1d → ReLU → Dropout(0.5)
        Linear(512, NUM_CLASSES)
    """
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.5): ...
    def forward(self, x): ...


def load_model(checkpoint_path: str, num_classes=NUM_CLASSES, device="cpu") -> FeatureNet3D:
    """
    Load FeatureNet3D from checkpoint.
    Supports both raw state_dict and {"model_state_dict": ...} wrapped format.
    Raises FileNotFoundError if checkpoint_path does not exist.
    Sets model to eval() mode before returning.
    """
```

---
---

# TRAINING SCRIPT — `training/train_feature_net.py`

```
Usage:
  python training/train_feature_net.py \
      --data    data/raw/synthetic \
      --out     checkpoints        \
      --epochs  30                 \
      --batch   32                 \
      --lr      1e-3
```

### Required arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | `data/raw/synthetic` | Root dir of labelled parts |
| `--out` | `checkpoints` | Directory for checkpoint files |
| `--epochs` | 30 | Number of training epochs |
| `--batch` | 32 | Batch size |
| `--lr` | 1e-3 | Initial learning rate |
| `--resolution` | 64 | Voxel resolution |
| `--workers` | 4 | DataLoader worker count |
| `--resume` | None | Path to checkpoint to resume from |

### Training loop requirements

- Loss: `nn.BCEWithLogitsLoss()` — multi-label, logits input
- Optimiser: `torch.optim.Adam`
- LR scheduler: `ReduceLROnPlateau(mode="min", factor=0.5, patience=5)`
- Print per-epoch: train_loss, val_loss, val_F1
- Save `checkpoints/last.pt` every epoch
- Save `checkpoints/best.pt` when val_loss improves
- Save `checkpoints/history.json` on completion

### Checkpoint format

```python
{
    "epoch": int,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimiser.state_dict(),
    "val_loss": float,
    "best_val_loss": float,
}
```

### F1 helper function

```python
def _compute_f1(model, loader, device, threshold=0.5) -> float:
    """
    Macro-averaged F1 over all NUM_CLASSES classes.
    Applies sigmoid to logits then thresholds.
    Uses epsilon=1e-8 in precision/recall denominators.
    """
```

### Graceful startup failure

If `len(dataset) == 0` after loading, print a clear error message and exit:
```
RuntimeError: No valid parts found in data/raw/synthetic.
Run: python training/synthetic_data_gen.py --count 2000
```

---
---

# INFERENCE MODULE — `phase2_feature_recognition.py`

```python
def recognise_features(
    voxel_path: str,
    model_path: str,
    threshold: float = 0.5,
    device: str = "cpu",
) -> dict:
    """
    Predict machining features present in a voxelised part.

    Parameters
    ----------
    voxel_path : path to voxel_{R}.npy produced by phase1_voxeliser
    model_path : path to trained FeatureNet3D checkpoint (.pt)
    threshold  : sigmoid probability cutoff for positive prediction
    device     : torch device string

    Returns
    -------
    {
      "features":      [{"type": str, "confidence": float}, ...],
      "feature_count": int,
      "all_scores":    {class_name: float, ...},   # all 12 classes
      "threshold":     float,
      "voxel_file":    str,   # absolute path
      "model_path":    str,   # absolute path
    }

    Raises
    ------
    FileNotFoundError  : voxel_path or model_path does not exist
    ValueError         : array is not 3-D or not cubic
    """
```

### Implementation requirements

1. Validate both paths exist before any computation
2. Load voxel with `np.load()`, check `ndim == 3` and all dims equal
3. Expand to `(1, 1, R, R, R)` float32 tensor
4. Run `torch.no_grad()` inference, apply `torch.sigmoid()`
5. Build `all_scores` dict for every class regardless of threshold
6. Build `features` list only for classes where prob >= threshold
7. Sort `features` by confidence descending
8. Return absolute paths in `voxel_file` and `model_path`

### CLI

```
python phase2_feature_recognition.py \
    data/processed/simple_block_cli/voxel_64.npy \
    checkpoints/best.pt                           \
    data/processed/simple_block_cli/

Writes: data/processed/simple_block_cli/features.json
```

Write `features.json` atomically (temp file + os.replace).

---
---

# TESTS — `tests/test_phase2.py`

### Fixtures needed

```python
FIXTURE_VOXEL = "data/processed/simple_block_cli/voxel_64.npy"
# Already exists from Phase 1 CLI smoke test.

@pytest.fixture(scope="session")
def random_checkpoint(tmp_path_factory):
    """Randomly-initialised model saved to temp file. No training required."""
    path = str(tmp_path_factory.mktemp("ckpt") / "random.pt")
    model = FeatureNet3D(num_classes=NUM_CLASSES)
    torch.save({"model_state_dict": model.state_dict()}, path)
    return path
```

### Required tests

```python
# Model architecture
def test_model_output_shape_64():           # (2,1,64,64,64) → (2,12)
def test_model_output_shape_32():           # (1,1,32,32,32) → (1,12)
def test_model_sigmoid_in_range():          # sigmoid(logits) in [0,1]
def test_load_model_file_not_found():       # raises FileNotFoundError

# Synthetic data generator
def test_generate_part_returns_labels():    # labels is non-empty list
def test_generate_part_always_has_flat_face():
def test_generate_part_labels_are_valid():  # all labels in FEATURE_NAMES
def test_generate_dataset_creates_files():  # part.stp + labels.json exist

# Dataset loader
def test_dataset_loads():                   # len(ds) > 0
def test_dataset_item_shapes():             # x=(1,32,32,32), y=(12,)
def test_dataset_label_is_multihot():       # y values all 0 or 1
def test_dataset_voxel_cache():             # voxel_32.npy written after first access

# Inference
def test_recognise_features_returns_dict():
def test_recognise_features_schema():       # all required keys present
def test_all_scores_has_all_classes():      # 12 keys, values in [0,1]
def test_feature_count_matches_list():
def test_threshold_zero_returns_all():      # threshold=0.0 → 12 features
def test_threshold_one_returns_none():      # threshold=1.0 → 0 features
def test_features_sorted_by_confidence():  # descending order
def test_voxel_not_found_raises():
def test_model_not_found_raises():
def test_output_paths_are_absolute():
```

Mark inference tests with:
```python
@pytest.mark.skipif(
    not os.path.exists(FIXTURE_VOXEL),
    reason="Phase 1 CLI output not available"
)
```

Run all tests: `pytest tests/test_phase2.py -v`

---
---

# QUICK-START COMMANDS

```bash
# 1. Generate synthetic training data (~10 min, CPU)
python training/synthetic_data_gen.py --count 2000 --output data/raw/synthetic

# 2. Train (CPU ~60 min / 30 epochs / 2000 parts; GPU much faster)
python training/train_feature_net.py \
    --data data/raw/synthetic \
    --out  checkpoints        \
    --epochs 30 --batch 16

# 3. Inference smoke test
python phase2_feature_recognition.py \
    data/processed/simple_block_cli/voxel_64.npy \
    checkpoints/best.pt \
    data/processed/simple_block_cli/

# 4. Run tests
pytest tests/test_phase2.py -v
```

---
---

# ACCEPTANCE CRITERIA

- [ ] `python training/synthetic_data_gen.py --count 200` generates ≥180
      valid parts with `labels.json` in `data/raw/synthetic/`
- [ ] `MachiningFeatureDataset` loads those parts; `len(ds) >= 100`
- [ ] `FeatureNet3D` forward pass on `(1,1,64,64,64)` → output `(1,12)`
      without error
- [ ] `train_feature_net.py` completes 2 epochs without error on 50 parts:
      `python training/train_feature_net.py --epochs 2 --batch 8 --data data/raw/synthetic`
- [ ] `recognise_features()` returns correct schema with random checkpoint
- [ ] `all_scores` contains all 12 class names, values in [0,1]
- [ ] `threshold=0.0` → 12 features; `threshold=1.0` → 0 features
- [ ] All pytest tests pass: `pytest tests/test_phase2.py -v`
- [ ] After full training (30 epochs, 2000 parts): val F1 ≥ 0.70

---
---

# NOTES FOR CODEX

1. **BCEWithLogitsLoss not BCELoss.** Model outputs raw logits.
   Apply `torch.sigmoid()` only at inference time.

2. **Multi-label not multi-class.** Do not use softmax or cross-entropy.
   A part will have multiple feature types simultaneously.

3. **flat_face is always label 1.** Every generated part has a flat face.
   The model should learn this quickly — a useful sanity check.

4. **Voxel cache prevents re-running Phase 1.** Dataset caches
   `voxel_{resolution}.npy` alongside each `part.stp`. First access calls
   `process_step_file()` which must be importable from project root.

5. **Synthetic generator is non-deterministic.** Some cadquery feature
   combinations fail silently. Generator retries up to 5× target count.

6. **Test skips are intentional.** Inference tests that require
   `simple_block_cli/voxel_64.npy` are skipped if absent — not failures.

7. **Checkpoint format.** Always save as `{"model_state_dict": ..., ...}`.
   `load_model()` handles both raw and wrapped formats but wrapped preferred.

8. **FEATURE_ADDERS dict.** Do not use `globals()[f"add_{name}"]` in
   production code — build the dict explicitly to avoid name collisions.
