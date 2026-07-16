from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import (
    H5CPMGDataset,
    compute_normalization,
    find_split_files,
    load_normalization,
    save_normalization,
)
from .model import ConditionalCPMGDeJNet
from .sampling import METADATA_KEYS


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dej_loss(pred: torch.Tensor, target: torch.Tensor, slope_weight: float) -> torch.Tensor:
    profile_loss = F.mse_loss(pred, target)
    pred_slope = pred[:, 1:] - pred[:, :-1]
    target_slope = target[:, 1:] - target[:, :-1]
    slope_loss = F.mse_loss(pred_slope, target_slope)
    return profile_loss + slope_weight * slope_loss


def raw_metrics(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    input_norm: torch.Tensor,
    normalization: dict,
) -> dict[str, float]:
    r2_mean = float(normalization["r2_mean"])
    r2_std = float(normalization["r2_std"])
    pred = pred_norm * r2_std + r2_mean
    target = target_norm * r2_std + r2_mean
    identity = input_norm * r2_std + r2_mean

    err = pred - target
    identity_err = identity - target
    mse = torch.mean(err**2).item()
    identity_mse = torch.mean(identity_err**2).item()
    rmse = math.sqrt(mse)
    identity_rmse = math.sqrt(identity_mse)
    mae = torch.mean(torch.abs(err)).item()
    max_abs = torch.max(torch.abs(err)).item()
    improvement = 0.0
    if identity_rmse > 0:
        improvement = 100.0 * (identity_rmse - rmse) / identity_rmse
    return {
        "rmse": rmse,
        "mae": mae,
        "max_abs": max_abs,
        "identity_rmse": identity_rmse,
        "identity_improvement_pct": improvement,
    }


def evaluate_epoch(
    model: ConditionalCPMGDeJNet,
    loader: DataLoader,
    device: torch.device,
    normalization: dict,
    slope_weight: float,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    count = 0
    preds = []
    targets = []
    inputs = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            metadata = batch["metadata"].to(device)
            target = batch["target"].to(device)
            pred = model(x, metadata)
            loss = dej_loss(pred, target, slope_weight)
            batch_size = x.shape[0]
            loss_sum += loss.item() * batch_size
            count += batch_size
            preds.append(pred.cpu())
            targets.append(target.cpu())
            inputs.append(x[:, 0, :].cpu())

    metrics = raw_metrics(
        torch.cat(preds, dim=0),
        torch.cat(targets, dim=0),
        torch.cat(inputs, dim=0),
        normalization,
    )
    metrics["loss"] = loss_sum / max(count, 1)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CPMG de-J-coupling DNN.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/cpmg_dej"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/cpmg_dej"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--slope-weight", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--metadata-hidden", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=6)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--normalization", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_files = find_split_files(args.data_dir, "train")
    val_files = find_split_files(args.data_dir, "val")

    norm_path = args.normalization or (args.out_dir / "normalization.json")
    if norm_path.exists():
        normalization = load_normalization(norm_path)
    else:
        normalization = compute_normalization(train_files)
        save_normalization(norm_path, normalization)

    train_dataset = H5CPMGDataset(train_files, normalization)
    val_dataset = H5CPMGDataset(val_files, normalization)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    device = select_device(args.device)
    model_kwargs = {
        "input_channels": 2,
        "metadata_dim": len(METADATA_KEYS),
        "hidden_channels": args.hidden_channels,
        "metadata_hidden": args.metadata_hidden,
        "num_blocks": args.num_blocks,
        "kernel_size": args.kernel_size,
    }
    model = ConditionalCPMGDeJNet(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    bad_epochs = 0
    history = []
    best_path = args.out_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            metadata = batch["metadata"].to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x, metadata)
            loss = dej_loss(pred, target, args.slope_weight)
            loss.backward()
            optimizer.step()

            batch_size = x.shape[0]
            train_loss_sum += loss.item() * batch_size
            train_count += batch_size

        train_loss = train_loss_sum / max(train_count, 1)
        val_metrics = evaluate_epoch(model, val_loader, device, normalization, args.slope_weight)
        record = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(record)
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} "
            f"val_loss={val_metrics['loss']:.6f} rmse={val_metrics['rmse']:.6f} "
            f"identity_rmse={val_metrics['identity_rmse']:.6f} "
            f"improvement={val_metrics['identity_improvement_pct']:.2f}%"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_kwargs": model_kwargs,
                    "normalization": normalization,
                    "metadata_keys": list(METADATA_KEYS),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "args": vars(args),
                },
                best_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs")
                break

    with (args.out_dir / "history.json").open("w") as handle:
        json.dump(history, handle, indent=2)
        handle.write("\n")
    print(f"Saved best checkpoint to {best_path}")


if __name__ == "__main__":
    main()
