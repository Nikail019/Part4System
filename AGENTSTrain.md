# AGENTS.md — Apple Silicon MPS Training & Production Checkpoint

## Goal

Patch MPS device support into all relevant files, generate training data,
train the feature recognition model to production quality, evaluate it,
and validate the full pipeline end-to-end. After this is done:

- `checkpoints/best.pt` exists and achieves macro F1 ≥ 0.70
- `scripts/validate_pipeline.py` exits 0 (all 3 fixtures PASS)
- All 215 existing tests still pass

This machine runs Apple Silicon. Use `mps` as the training device.

---

## Part 1 — Patch MPS Device Detection

### 1.1 `training/train_feature_net.py`

Find the line:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

Replace with:
```python
def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

device = _get_device()
```

Also find the DataLoader for training and set `pin_memory=False` when
device is `"mps"` (pin_memory is only useful for CUDA):

```python
pin = (device == "cuda")
train_loader = DataLoader(
    train_ds, batch_size=args.batch, shuffle=True,
    num_workers=args.workers, pin_memory=pin,
)
val_loader = DataLoader(
    val_ds, batch_size=args.batch, shuffle=False,
    num_workers=args.workers, pin_memory=False,
)
```

Also add `--workers` default override: on MPS, `num_workers > 0` can cause
hangs on some PyTorch versions. Add this guard after device detection:

```python
if device == "mps" and args.workers > 0:
    print(f"  Note: reducing workers to 0 for MPS stability")
    args.workers = 0
```

### 1.2 `training/evaluate_model.py`

Find device detection (likely `"cuda" if ... else "cpu"`) and replace
with the same `_get_device()` function pattern as above.

Add `--device` CLI argument with auto-detection as default:

```python
p.add_argument("--device", default=None,
    help="Device override: mps / cuda / cpu (default: auto-detect)")
```

In `main()`, after parsing:
```python
device = args.device if args.device else _get_device()
print(f"Device: {device}")
```

### 1.3 `phase2_feature_recognition.py`

The `recognise_features()` function already accepts a `device` parameter.
Update its default:

```python
def recognise_features(
    voxel_path: str,
    model_path: str,
    threshold: float = 0.5,
    device: str = "cpu",          # keep cpu default for inference — fast enough
) -> dict:
```

Inference on a single voxel is fast on CPU. No change needed for inference.

### 1.4 `run_pipeline.py`

Add `--device` argument to `get_args()`:

```python
parser.add_argument("--device", default=None,
    help="Training device override: mps / cuda / cpu (default: auto)")
```

In `run_phase2`, pass device through if needed (inference stays CPU).
No other changes required — the pipeline does not train at runtime.

---

## Part 2 — Generate Synthetic Training Data

Run this command. It will take approximately 10–15 minutes.

```bash
python training/synthetic_data_gen.py \
    --count 3000 \
    --output data/raw/synthetic
```

Verify success:
```bash
python -c "
import json
m = json.load(open('data/raw/synthetic/manifest.json'))
print(f'Parts generated: {m[\"total_files\"]}')
assert m['total_files'] >= 2700, f'Too few parts: {m[\"total_files\"]}'
print('Data generation OK')
"
```

If total_files < 2700, run again with `--count 500 --output data/raw/synthetic`
to top up (the generator appends, it does not overwrite existing parts).

---

## Part 3 — Train the Model

Run the full training command. Expected wall-clock time on Apple Silicon M-series:
- M1/M2: ~25–40 minutes for 50 epochs on 3000 parts
- M3/M4: ~15–25 minutes

```bash
python training/train_feature_net.py \
    --data          data/raw/synthetic \
    --out           checkpoints        \
    --epochs        50                 \
    --batch         32                 \
    --lr            1e-3               \
    --resolution    64                 \
    --augment                          \
    --class-weights                    \
    --early-stop    10                 \
    --min-epochs    10                 \
    --workers       0
```

`--workers 0` is required for MPS stability (already enforced by the patch
in Part 1, but explicit here for clarity).

### Expected training output

```
Device: mps
Dataset: 3000 parts
  Train: 2400  Val: 300  Test: 300
Class weights range: [0.21, 4.83]

Epoch   1 | train_loss=0.4821 | val_loss=0.3914 | val_f1=0.3102
Epoch   2 | train_loss=0.3642 | val_loss=0.3201 | val_f1=0.4418
  ✓ New best F1=0.4418
...
Epoch  18 | train_loss=0.1823 | val_loss=0.1941 | val_f1=0.7634
  ✓ New best F1=0.7634
Epoch  19 | train_loss=0.1790 | val_loss=0.1960 | val_f1=0.7581
Epoch  20 | train_loss=0.1751 | val_loss=0.1978 | val_f1=0.7612
...
Early stopping at epoch 28 (no F1 improvement for 10 epochs)
```

### If val F1 plateaus below 0.65 after epoch 15

Stop training and resume with a lower learning rate:

```bash
python training/train_feature_net.py \
    --data     data/raw/synthetic \
    --out      checkpoints        \
    --epochs   50 --lr 3e-4       \
    --resume   checkpoints/last.pt \
    --augment --class-weights     \
    --early-stop 10 --min-epochs 5 \
    --workers 0
```

### If training crashes with MPS error

Some PyTorch versions have MPS bugs with specific ops. If you see
`NotImplementedError` or `RuntimeError: MPS` during training, fall back:

```bash
python training/train_feature_net.py \
    --data data/raw/synthetic --out checkpoints \
    --epochs 50 --batch 32 --lr 1e-3 \
    --augment --class-weights \
    --early-stop 10 --min-epochs 10 \
    --workers 0 \
    # Add this temporarily:
    # and patch _get_device() to return "cpu"
```

CPU training on M-series is still faster than most Intel laptops.

---

## Part 4 — Evaluate the Model

```bash
python training/evaluate_model.py \
    --model checkpoints/best.pt \
    --data  data/raw/synthetic  \
    --out   checkpoints/eval/
```

Verify the report:

```bash
python -c "
import json
report = json.load(open('checkpoints/eval/eval_report.json'))
macro_f1 = report['macro_f1']
flat_f1  = report['per_class']['flat_face']['f1']
print(f'Macro F1:      {macro_f1:.4f}  (target >= 0.70)')
print(f'flat_face F1:  {flat_f1:.4f}  (target >= 0.95)')
assert macro_f1 >= 0.70, f'Macro F1 too low: {macro_f1:.4f}'
assert flat_f1  >= 0.90, f'flat_face F1 too low: {flat_f1:.4f}'
print('Evaluation OK')
"
```

---

## Part 5 — Run Full Pipeline Validation

```bash
python scripts/validate_pipeline.py \
    --model   checkpoints/best.pt \
    --factory factory_profiles/nash_nz.json \
    --material aluminium_6061 \
    --out     data/validation/
```

Expected output:
```
Validating pipeline with model: checkpoints/best.pt
Factory: factory_profiles/nash_nz.json  Material: aluminium_6061

  simple_block         PASS  N ops   XX.X min  NZD XXX.XX  [ACCEPT]
  block_with_holes     PASS  N ops   XX.X min  NZD XXX.XX  [ACCEPT]
  complex_prismatic    PASS  N ops   XX.X min  NZD XXX.XX  [ACCEPT]

Overall: PASS  (3/3 fixtures)
Report:  data/validation/validation_report.json
```

Verify:
```bash
python -c "
import json
r = json.load(open('data/validation/validation_report.json'))
print(f'Overall status: {r[\"overall_status\"]}')
for name, result in r['fixtures'].items():
    ops = result.get('operation_count', '?')
    rec = result.get('recommendation', '?')
    print(f'  {name:<22} {result[\"status\"]}  {ops} ops  [{rec}]')
assert r['overall_status'] == 'PASS', 'Validation failed'
print('Pipeline validation OK')
"
```

---

## Part 6 — Full Pipeline Smoke Test

```bash
python run_pipeline.py \
    tests/fixtures/complex_prismatic.stp \
    factory_profiles/nash_nz.json \
    --output data/processed/complex_prismatic/
```

Confirm the summary shows:
- No mention of "fallback"
- `Operations >= 6`
- `Recommendation: ACCEPT`

---

## Part 7 — Run Full Test Suite

```bash
pytest tests/ -v
```

All 215 existing tests must still pass. If MPS patches break any tests,
the fix is always to ensure `_get_device()` falls back cleanly to CPU
in test environments where MPS may not be available:

```python
def _get_device() -> str:
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except AttributeError:
        pass   # older PyTorch without MPS support
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
```

---

## Acceptance Criteria

- [ ] `training/train_feature_net.py` uses `_get_device()` with MPS support
- [ ] `training/evaluate_model.py` uses `_get_device()` with MPS support
- [ ] `pin_memory=False` when device is `mps`
- [ ] `args.workers` set to 0 when device is `mps`
- [ ] `data/raw/synthetic/manifest.json` shows ≥ 2700 parts
- [ ] `checkpoints/best.pt` exists and is loadable
- [ ] `checkpoints/best.pt` contains `training_config` key
- [ ] `checkpoints/eval/eval_report.json` shows macro F1 ≥ 0.70
- [ ] `checkpoints/eval/eval_report.json` shows flat_face F1 ≥ 0.90
- [ ] `data/validation/validation_report.json` shows `overall_status: PASS`
- [ ] All 3 fixtures show `recommendation: ACCEPT`
- [ ] `complex_prismatic` shows `operation_count >= 6`
- [ ] `pytest tests/ -v` — all tests pass (≥ 215)
- [ ] Full pipeline run on `complex_prismatic.stp` completes without error
      and prints a summary with no "fallback" language

---

## Notes for Codex

1. **Do not change the public API of any function.** Only change device
   selection internals. All existing tests must pass unchanged.

2. **`_get_device()` should be defined once per file** that needs it,
   not imported from a shared module. This avoids circular imports and
   keeps each file self-contained.

3. **MPS guard for workers must print a note** so the user knows why
   workers were reduced. Silent behaviour changes are confusing.

4. **Run commands in order.** Data generation must complete before
   training. Training must complete before evaluation. Evaluation
   before validation.

5. **If `manifest.json` already shows ≥ 2700 parts, skip data generation.**
   Check before running to avoid regenerating data unnecessarily.

6. **Save the training log.** Pipe training output to a file alongside
   the checkpoints for reproducibility:
   ```bash
   python training/train_feature_net.py [args] 2>&1 | tee checkpoints/train_log.txt
   ```

7. **The validation script uses subprocess.** It will import fresh
   Python processes for each fixture. Ensure `run_pipeline.py` is
   importable from the repo root before running validation.
