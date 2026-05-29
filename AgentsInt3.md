# AGENTS.md — Scale Training to 1000 Parts

## Diagnosis

Current result: macro F1 = 0.28 on 20 test samples.

Root causes:
1. 200 parts → 160 train → ~13 samples per class → too few for rare features
2. 20 test samples → F1 is high-variance and unreliable
3. Early stopping at min_epochs=8 may cut training before rare classes converge

Fix: scale to 1000 parts. At 32³ resolution this costs ~80MB of disk.

## Disk Budget Check

```
Current free disk  : 4.9 GB
1000 voxel_32.npy  : 1000 × 32KB   = ~32MB
1000 mesh.stl      : 1000 × 50KB   = ~50MB
1000 metadata.json : 1000 × 2KB    = ~2MB
Checkpoints        :                 ~5MB
─────────────────────────────────────────
Total new writes   :                ~89MB
Remaining free     : ~4.8 GB        ✓ safe
```

---

## Step 1 — Generate 800 More Parts

The existing 200 `part.stp` files in `data/raw/synthetic/` are kept.
Generate 800 additional parts into the same directory to reach ~1000 total.

```bash
python training/synthetic_data_gen.py \
    --count  800 \
    --output data/raw/synthetic
```

Verify total:
```bash
python -c "
import os, json
parts = [d for d in os.scandir('data/raw/synthetic')
         if d.is_dir()
         and os.path.exists(os.path.join(d.path, 'part.stp'))
         and os.path.exists(os.path.join(d.path, 'labels.json'))]
print(f'Total valid parts: {len(parts)}')
assert len(parts) >= 900, f'Expected >= 900 parts, got {len(parts)}'
"
```

---

## Step 2 — Pre-warm Cache for All 1000 Parts

The first 200 are already pre-warmed. This call skips them and processes
only the new parts (idempotent).

```bash
python scripts/prewarm_cache.py \
    --data       data/raw/synthetic \
    --resolution 32

# No --max-samples here: warm ALL parts so we have flexibility
```

Check disk:
```bash
df -h .
du -sh data/raw/synthetic
```

---

## Step 3 — Train on 1000 Parts

Updated hyperparameters for a larger dataset:

```bash
python training/train_feature_net.py \
    --data          data/raw/synthetic \
    --out           checkpoints        \
    --epochs        50                 \
    --batch         32                 \
    --lr            5e-4               \
    --resolution    32                 \
    --max-samples   1000               \
    --augment                          \
    --class-weights                    \
    --early-stop    12                 \
    --min-epochs    15                 \
    --workers       0                  \
    2>&1 | tee checkpoints/train_log.txt
```

Changes from previous run:

| Parameter | Before | Now | Reason |
|---|---|---|---|
| `--max-samples` | 200 | 1000 | 5× more training data |
| `--lr` | 1e-3 | 5e-4 | More conservative with larger dataset |
| `--early-stop` | 8 | 12 | Give rare classes more time to converge |
| `--min-epochs` | 8 | 15 | Prevent premature stopping |

Expected split: 800 train / 100 val / 100 test.
Expected per-epoch time: 60–90 seconds on CPU (5× more data than before).
Expected total training time: 20–35 minutes.

Expected progress:
```
Device: cpu
Dataset: 1000 parts
  Train: 800  Val: 100  Test: 100
Class weights range: [0.15, 5.80]

Epoch   1/ 50 | train=0.5210 | val=0.4801 | F1=0.3421
Epoch   5/ 50 | train=0.3401 | val=0.3102 | F1=0.5234 ✓ best
Epoch  10/ 50 | train=0.2201 | val=0.2341 | F1=0.6812 ✓ best
Epoch  15/ 50 | train=0.1802 | val=0.2001 | F1=0.7201 ✓ best
...
Early stopping at epoch ~30
```

---

## Step 4 — Evaluate

```bash
python training/evaluate_model.py \
    --model checkpoints/best.pt \
    --data  data/raw/synthetic  \
    --out   checkpoints/eval/
```

Target: macro F1 ≥ 0.60. Expected breakdown:

| Class | Expected F1 |
|---|---|
| flat_face | ≥ 0.95 |
| rectangular_pocket | ≥ 0.70 |
| through_hole | ≥ 0.65 |
| blind_hole | ≥ 0.65 |
| rectangular_step | ≥ 0.70 |
| boss | ≥ 0.65 |
| rectangular_slot | ≥ 0.60 |
| circular_pocket | ≥ 0.55 |
| chamfer | ≥ 0.50 |
| fillet | ≥ 0.50 |
| circular_slot | ≥ 0.45 |
| triangular_pocket | ≥ 0.50 |

`circular_slot`, `chamfer`, and `fillet` are geometrically subtle at 32³
and will be the weakest. This is expected and honest to note in the
dissertation.

---

## Step 5 — Validate Full Pipeline

```bash
python scripts/validate_pipeline.py \
    --model   checkpoints/best.pt \
    --factory factory_profiles/nash_nz.json \
    --out     data/validation/
```

---

## Step 6 — Full Test Suite

```bash
pytest tests/ -v
```

All 216 existing tests must pass. No new tests required for this change —
the only difference is data quantity and hyperparameters.

---

## Acceptance Criteria

- [ ] `data/raw/synthetic/` contains ≥ 900 valid parts
- [ ] All parts pre-warmed at 32³ with 0 errors
- [ ] `df -h .` shows ≥ 3.5 GB free after pre-warm
- [ ] `checkpoints/best.pt` replaced with new checkpoint
- [ ] `checkpoints/eval/eval_report.json` shows macro F1 ≥ 0.60
- [ ] `flat_face` F1 ≥ 0.90
- [ ] `data/validation/validation_report.json` shows `overall_status: PASS`
- [ ] `pytest tests/ -v` — 216 tests pass

---

## If F1 is Still Below 0.60 After Training

Do not increase dataset size further — disk is the constraint.
Instead try these in order:

**Option A — Lower learning rate, more epochs:**
```bash
python training/train_feature_net.py \
    --data data/raw/synthetic --out checkpoints \
    --resume checkpoints/last.pt \
    --epochs 80 --lr 1e-4 \
    --max-samples 1000 --resolution 32 \
    --augment --class-weights \
    --early-stop 15 --min-epochs 5 \
    --workers 0
```

**Option B — Check class distribution:**
```bash
python -c "
import os, json, collections
root = 'data/raw/synthetic'
counter = collections.Counter()
parts = [d for d in os.scandir(root) if d.is_dir()][:1000]
for p in parts:
    lp = os.path.join(p.path, 'labels.json')
    if os.path.exists(lp):
        for lbl in json.load(open(lp))['labels']:
            counter[lbl] += 1
print('Class counts in first 1000 parts:')
for cls, cnt in sorted(counter.items(), key=lambda x: x[1]):
    print(f'  {cls:<25} {cnt:>4}  ({cnt/10:.1f}%)')
"
```

If any class appears in fewer than 30 parts, generate more parts with
`synthetic_data_gen.py` until that class reaches ~50 parts.

---

## Notes for Codex

1. **Do not delete existing pre-warmed voxels.** The 200 `voxel_32.npy`
   files already in `data/raw/synthetic/` are still valid. `prewarm_cache.py`
   skips them automatically.

2. **Training overwrites `checkpoints/best.pt`.** The previous 0.28 F1
   checkpoint will be replaced. This is intentional.

3. **`--max-samples 1000` requires ≥ 1000 valid cached parts.** Run
   prewarm before training. If prewarm completes with fewer than 1000
   (due to generation errors), set `--max-samples` to the actual count.

4. **The `--resume` flag in Option A loads `last.pt`, not `best.pt`.**
   `last.pt` has the most recent optimiser state; `best.pt` has the best
   weights but may not have matching optimiser momentum. Always resume
   from `last.pt`.
