"""Train FeatureNet3D for multi-label feature recognition."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.feature_net import FEATURE_NAMES, NUM_CLASSES, FeatureNet3D
from training.dataset import MachiningFeatureDataset, random_split_dataset


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


def _save_checkpoint(path: Path, epoch: int, model, optimiser, val_loss: float, best_val_loss: float) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimiser.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Phase 2 feature recognizer.")
    parser.add_argument("--data", default=os.path.join("data", "raw", "synthetic"))
    parser.add_argument("--out", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume")
    args = parser.parse_args()

    dataset = MachiningFeatureDataset(args.data, resolution=args.resolution)
    if len(dataset) == 0:
        raise RuntimeError(
            f"No valid parts found in {args.data}.\n"
            "Run: python training/synthetic_data_gen.py --count 2000"
        )

    train_ds, val_ds, _ = random_split_dataset(dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    model = FeatureNet3D(num_classes=NUM_CLASSES).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=5
    )
    start_epoch = 1
    best_val_loss = float("inf")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimiser.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", checkpoint.get("val_loss", best_val_loss)))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = _run_epoch(model, train_loader, criterion, optimiser, device, train=True)
        with torch.no_grad():
            val_loss = _run_epoch(model, val_loader, criterion, optimiser, device, train=False)
        val_f1 = _compute_f1(model, val_loader, device)
        scheduler.step(val_loss)

        best_val_loss = min(best_val_loss, val_loss)
        _save_checkpoint(out_dir / "last.pt", epoch, model, optimiser, val_loss, best_val_loss)
        if val_loss <= best_val_loss:
            _save_checkpoint(out_dir / "best.pt", epoch, model, optimiser, val_loss, best_val_loss)

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_f1": val_f1}
        history.append(row)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_F1={val_f1:.4f}"
        )

    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump({"classes": FEATURE_NAMES, "history": history}, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
