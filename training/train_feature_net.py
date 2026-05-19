"""Train FeatureNet3D for multi-label feature recognition."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.feature_net import FEATURE_NAMES, NUM_CLASSES, FeatureNet3D
from training.augmentation import get_train_transforms, get_val_transforms
from training.dataset import MachiningFeatureDataset, compute_class_weights, random_split_dataset


class _TransformedSubset(Dataset):
    """Apply a transform to a Subset without modifying the underlying dataset."""

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y


def _compute_f1(model, loader, device, threshold: float = 0.5) -> float:
    eps = 1e-8
    model.eval()
    tp = torch.zeros(NUM_CLASSES, device=device)
    fp = torch.zeros(NUM_CLASSES, device=device)
    fn = torch.zeros(NUM_CLASSES, device=device)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = (torch.sigmoid(model(x)) >= threshold).float()
            tp += (preds * y).sum(dim=0)
            fp += (preds * (1 - y)).sum(dim=0)
            fn += ((1 - preds) * y).sum(dim=0)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return float(f1.mean().item())


def _classification_stats(model, loader, device, threshold: float = 0.5):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in loader:
            probs = torch.sigmoid(model(x.to(device))).cpu()
            all_preds.append((probs >= threshold).float())
            all_targets.append(y.float())
    if not all_preds:
        empty = torch.zeros(0, NUM_CLASSES)
        return empty, empty
    return torch.cat(all_preds), torch.cat(all_targets)


def _print_per_class_report(model, loader, device, threshold: float = 0.5) -> dict:
    preds, targets = _classification_stats(model, loader, device, threshold)
    if preds.numel() == 0:
        return {name: {"precision": 0.0, "recall": 0.0, "f1": 0.0} for name in FEATURE_NAMES}

    tp = (preds * targets).sum(dim=0)
    fp = (preds * (1 - targets)).sum(dim=0)
    fn = ((1 - preds) * targets).sum(dim=0)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print("\nPer-class results on test split:")
    print(f"  {'Feature':<25} {'P':>6} {'R':>6} {'F1':>6}")
    print(f"  {'-' * 25} {'-' * 6} {'-' * 6} {'-' * 6}")
    report = {}
    for idx, name in enumerate(FEATURE_NAMES):
        print(f"  {name:<25} {precision[idx]:>6.3f} {recall[idx]:>6.3f} {f1[idx]:>6.3f}")
        report[name] = {
            "precision": round(float(precision[idx]), 4),
            "recall": round(float(recall[idx]), 4),
            "f1": round(float(f1[idx]), 4),
        }
    print(f"\n  Macro F1: {float(f1.mean()):.4f}")
    return report


def _run_epoch(model, loader, criterion, optimiser, device, train: bool) -> float:
    model.train(train)
    total_loss = 0.0
    total_samples = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        if train:
            optimiser.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        if train:
            loss.backward()
            optimiser.step()
        total_loss += float(loss.item()) * x.shape[0]
        total_samples += x.shape[0]
    return total_loss / max(1, total_samples)


def _checkpoint_data(
    args,
    epoch: int,
    model,
    optimiser,
    val_loss: float,
    val_f1: float,
    best_val_loss: float,
    best_val_f1: float,
) -> dict:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimiser.state_dict(),
        "val_loss": val_loss,
        "val_f1": val_f1,
        "best_val_f1": best_val_f1,
        "best_val_loss": best_val_loss,
        "training_config": {
            "resolution": args.resolution,
            "material": None,
            "augment": args.augment,
            "class_weights": args.class_weights,
            "batch_size": args.batch,
            "learning_rate": args.lr,
            "num_classes": NUM_CLASSES,
            "feature_names": FEATURE_NAMES,
        },
    }


def _save_checkpoint(path: Path, data: dict) -> None:
    torch.save(data, path)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Phase 2 feature recognizer.")
    parser.add_argument("--data", default=os.path.join("data", "raw", "synthetic"))
    parser.add_argument("--out", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume")
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no-augment", action="store_false", dest="augment")
    parser.add_argument("--class-weights", action="store_true", default=True)
    parser.add_argument("--no-class-weights", action="store_false", dest="class_weights")
    parser.add_argument("--early-stop", type=int, default=10)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def train(args: argparse.Namespace) -> dict:
    dataset = MachiningFeatureDataset(args.data, resolution=args.resolution)
    if len(dataset) == 0:
        raise RuntimeError(
            f"No valid parts found in {args.data}.\n"
            "Run: python training/synthetic_data_gen.py --count 2000"
        )

    train_raw, val_raw, test_raw = random_split_dataset(dataset)
    train_ds = _TransformedSubset(train_raw, get_train_transforms() if args.augment else None)
    val_ds = _TransformedSubset(val_raw, get_val_transforms())
    test_ds = _TransformedSubset(test_raw, get_val_transforms())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    model = FeatureNet3D(num_classes=NUM_CLASSES).to(device)
    if args.class_weights:
        weights = compute_class_weights(dataset).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
        if not args.quiet:
            print(f"  Class weights range: [{weights.min():.2f}, {weights.max():.2f}]")
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=5
    )
    start_epoch = 1
    best_val_loss = float("inf")
    best_val_f1 = -1.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimiser.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", checkpoint.get("val_loss", best_val_loss)))
        best_val_f1 = float(checkpoint.get("best_val_f1", checkpoint.get("val_f1", best_val_f1)))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    epochs_no_improve = 0
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = _run_epoch(model, train_loader, criterion, optimiser, device, train=True)
        with torch.no_grad():
            val_loss = _run_epoch(model, val_loader, criterion, optimiser, device, train=False)
        val_f1 = _compute_f1(model, val_loader, device)
        scheduler.step(val_loss)
        best_val_loss = min(best_val_loss, val_loss)

        data = _checkpoint_data(args, epoch, model, optimiser, val_loss, val_f1, best_val_loss, best_val_f1)
        _save_checkpoint(out_dir / "last.pt", data)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            data = _checkpoint_data(args, epoch, model, optimiser, val_loss, val_f1, best_val_loss, best_val_f1)
            _save_checkpoint(out_dir / "best.pt", data)
            if not args.quiet:
                print(f"  New best F1={val_f1:.4f}")
        else:
            epochs_no_improve += 1

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_f1": val_f1}
        history.append(row)
        if not args.quiet:
            print(
                f"epoch={epoch} train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_F1={val_f1:.4f}"
            )

        if epoch >= args.min_epochs and epochs_no_improve >= args.early_stop:
            print(f"Early stopping at epoch {epoch} (no F1 improvement for {args.early_stop} epochs)")
            break

    report = _print_per_class_report(model, test_loader, device)
    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump({"classes": FEATURE_NAMES, "history": history, "test_report": report}, f, indent=2)
        f.write("\n")
    return {"best_val_f1": best_val_f1, "history": history, "test_report": report}


def main() -> None:
    args = get_args()
    train(args)


if __name__ == "__main__":
    main()
