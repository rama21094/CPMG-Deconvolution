from __future__ import annotations

import json
import math
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from . import train_dynamic_config as cfg
from .checkpoint import load_trusted_checkpoint
from .dataset import save_normalization
from .generate_dataset import build_ncyc_grid
from .in_memory import (
    InMemoryCPMGDataset,
    compute_normalization_from_arrays,
    generate_in_memory_arrays,
)
from .model import ConditionalCPMGDeJNet
from .sampling import METADATA_KEYS, load_range_specs, specs_to_json
from .train import dej_loss, evaluate_epoch, select_device


def config_dict() -> dict:
    values = {}
    for key in dir(cfg):
        if key.isupper():
            value = getattr(cfg, key)
            values[key] = str(value) if isinstance(value, Path) else value
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic in-memory CPMG de-J trainer.")
    parser.add_argument("--device", default=None, help="Override config DEVICE, e.g. cuda:1")
    parser.add_argument("--generation-workers", type=int, default=None)
    parser.add_argument("--generation-chunk-size", type=int, default=None)
    parser.add_argument("--train-profiles-per-block", type=int, default=None)
    parser.add_argument("--normalization-profiles", type=int, default=None)
    parser.add_argument("--val-profiles", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def apply_cli_overrides(args: argparse.Namespace) -> None:
    overrides = {
        "DEVICE": args.device,
        "GENERATION_WORKERS": args.generation_workers,
        "GENERATION_CHUNK_SIZE": args.generation_chunk_size,
        "TRAIN_PROFILES_PER_BLOCK": args.train_profiles_per_block,
        "NORMALIZATION_PROFILES": args.normalization_profiles,
        "VAL_PROFILES": args.val_profiles,
        "MAX_EPOCHS": args.max_epochs,
        "OUT_DIR": args.out_dir,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)
    if args.no_resume:
        cfg.RESUME = False


def generate_arrays(label: str, count: int, seed: int, specs_json: str, ncyc_grid) -> dict:
    print(
        f"Generating {label}: count={count} seed={seed} workers={cfg.GENERATION_WORKERS} "
        f"chunk_size={cfg.GENERATION_CHUNK_SIZE}",
        flush=True,
    )
    return generate_in_memory_arrays(
        num_profiles=count,
        seed=seed,
        specs_json=specs_json,
        ncyc_grid=ncyc_grid,
        t_relax=cfg.T_RELAX,
        max_retries=cfg.MAX_RETRIES,
        workers=cfg.GENERATION_WORKERS,
        chunk_size=cfg.GENERATION_CHUNK_SIZE,
        progress_label=label,
        progress_seconds=cfg.GENERATION_PROGRESS_SECONDS,
    )


def make_loader(dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=False,
    )


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def cosine_lr(epoch_in_block: int, epochs_per_block: int, base_lr: float, min_lr: float) -> float:
    if epochs_per_block <= 1:
        return base_lr
    phase = epoch_in_block / (epochs_per_block - 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * phase))


def make_model(device: torch.device) -> tuple[ConditionalCPMGDeJNet, dict]:
    model_kwargs = {
        "input_channels": 2,
        "metadata_dim": len(METADATA_KEYS),
        "hidden_channels": cfg.HIDDEN_CHANNELS,
        "metadata_hidden": cfg.METADATA_HIDDEN,
        "num_blocks": cfg.NUM_BLOCKS,
        "kernel_size": cfg.KERNEL_SIZE,
    }
    model = ConditionalCPMGDeJNet(**model_kwargs).to(device)
    if cfg.USE_TORCH_COMPILE and hasattr(torch, "compile"):
        model = torch.compile(model)
    return model, model_kwargs


def save_checkpoint(
    path: Path,
    model,
    optimizer: torch.optim.Optimizer,
    model_kwargs: dict,
    normalization: dict,
    epoch: int,
    block_index: int,
    best_val: float,
    bad_epochs: int,
    lr_bad_epochs: int,
    base_lr: float,
    history: list[dict],
    val_metrics: dict,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_kwargs": model_kwargs,
            "normalization": normalization,
            "metadata_keys": list(METADATA_KEYS),
            "epoch": epoch,
            "block_index": block_index,
            "best_val": best_val,
            "bad_epochs": bad_epochs,
            "lr_bad_epochs": lr_bad_epochs,
            "base_lr": base_lr,
            "history": history,
            "val_metrics": val_metrics,
            "config": config_dict(),
        },
        path,
    )


def maybe_resume(model, optimizer, path: Path, device: torch.device):
    if not cfg.RESUME or not path.exists():
        return 0, 0, float("inf"), 0, 0, cfg.LR, []

    checkpoint = load_trusted_checkpoint(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)

    print(f"Resumed from {path} at epoch {checkpoint.get('epoch', 0)}")
    return (
        int(checkpoint.get("epoch", 0)),
        int(checkpoint.get("block_index", 0)),
        float(checkpoint.get("best_val", float("inf"))),
        int(checkpoint.get("bad_epochs", 0)),
        int(checkpoint.get("lr_bad_epochs", 0)),
        float(checkpoint.get("base_lr", cfg.LR)),
        list(checkpoint.get("history", [])),
    )


def main() -> None:
    apply_cli_overrides(parse_args())
    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (cfg.OUT_DIR / "config.json").open("w") as handle:
        json.dump(config_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")

    device = select_device(cfg.DEVICE)
    specs = load_range_specs(cfg.RANGES_CSV)
    specs_json = specs_to_json(specs)
    ncyc_grid = build_ncyc_grid(cfg.NCYC_START, cfg.NCYC_STOP, cfg.NCYC_STEP)

    normalization_arrays = generate_arrays(
        "normalization", cfg.NORMALIZATION_PROFILES, cfg.NORMALIZATION_SEED, specs_json, ncyc_grid
    )
    normalization = compute_normalization_from_arrays(normalization_arrays)
    save_normalization(cfg.OUT_DIR / "normalization.json", normalization)

    val_arrays = generate_arrays("validation", cfg.VAL_PROFILES, cfg.VALIDATION_SEED, specs_json, ncyc_grid)
    val_dataset = InMemoryCPMGDataset(val_arrays, normalization)
    val_loader = make_loader(val_dataset, cfg.BATCH_SIZE, shuffle=False)

    model, model_kwargs = make_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    start_epoch, start_block, best_val, bad_epochs, lr_bad_epochs, base_lr, history = maybe_resume(
        model, optimizer, cfg.OUT_DIR / "last_model.pt", device
    )

    scaler_enabled = bool(cfg.USE_AMP and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    epoch = start_epoch
    block_index = epoch // cfg.EPOCHS_PER_BLOCK
    while epoch < cfg.MAX_EPOCHS:
        train_seed = cfg.TRAIN_SEED + block_index
        train_arrays = generate_arrays(
            f"train block {block_index}",
            cfg.TRAIN_PROFILES_PER_BLOCK,
            train_seed,
            specs_json,
            ncyc_grid,
        )
        train_dataset = InMemoryCPMGDataset(train_arrays, normalization)
        train_loader = make_loader(train_dataset, cfg.BATCH_SIZE, shuffle=True)

        first_epoch_in_block = epoch % cfg.EPOCHS_PER_BLOCK
        for epoch_in_block in range(first_epoch_in_block, cfg.EPOCHS_PER_BLOCK):
            if epoch >= cfg.MAX_EPOCHS:
                break

            lr = cosine_lr(epoch_in_block, cfg.EPOCHS_PER_BLOCK, base_lr, cfg.MIN_LR)
            set_lr(optimizer, lr)
            epoch += 1

            model.train()
            train_loss_sum = 0.0
            train_count = 0
            for batch in train_loader:
                x = batch["x"].to(device)
                metadata = batch["metadata"].to(device)
                target = batch["target"].to(device)

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=scaler_enabled):
                    pred = model(x, metadata)
                    loss = dej_loss(pred, target, cfg.SLOPE_WEIGHT)

                scaler.scale(loss).backward()
                if cfg.GRAD_CLIP_NORM and cfg.GRAD_CLIP_NORM > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()

                batch_size = x.shape[0]
                train_loss_sum += loss.item() * batch_size
                train_count += batch_size

            train_loss = train_loss_sum / max(train_count, 1)
            val_metrics = evaluate_epoch(model, val_loader, device, normalization, cfg.SLOPE_WEIGHT)
            improved = val_metrics["loss"] < best_val
            if improved:
                best_val = val_metrics["loss"]
                bad_epochs = 0
                lr_bad_epochs = 0
            else:
                bad_epochs += 1
                lr_bad_epochs += 1

            if lr_bad_epochs >= cfg.LR_PLATEAU_PATIENCE:
                new_base_lr = max(base_lr * cfg.LR_PLATEAU_FACTOR, cfg.MIN_LR)
                if new_base_lr < base_lr:
                    base_lr = new_base_lr
                    print(f"Reduced base LR to {base_lr:.6g} after validation plateau")
                lr_bad_epochs = 0

            record = {
                "epoch": epoch,
                "block_index": block_index,
                "epoch_in_block": epoch_in_block,
                "train_seed": train_seed,
                "train_loss": train_loss,
                "lr": current_lr(optimizer),
                **val_metrics,
            }
            history.append(record)
            print(
                f"epoch={epoch} block={block_index} train_loss={train_loss:.6f} "
                f"val_loss={val_metrics['loss']:.6f} rmse={val_metrics['rmse']:.6f} "
                f"identity_rmse={val_metrics['identity_rmse']:.6f} "
                f"improvement={val_metrics['identity_improvement_pct']:.2f}% "
                f"lr={current_lr(optimizer):.6g}"
            )

            if improved:
                save_checkpoint(
                    cfg.OUT_DIR / "best_model.pt",
                    model,
                    optimizer,
                    model_kwargs,
                    normalization,
                    epoch,
                    block_index,
                    best_val,
                    bad_epochs,
                    lr_bad_epochs,
                    base_lr,
                    history,
                    val_metrics,
                )

            if cfg.SAVE_EVERY_EPOCH:
                save_checkpoint(
                    cfg.OUT_DIR / "last_model.pt",
                    model,
                    optimizer,
                    model_kwargs,
                    normalization,
                    epoch,
                    block_index,
                    best_val,
                    bad_epochs,
                    lr_bad_epochs,
                    base_lr,
                    history,
                    val_metrics,
                )

            with (cfg.OUT_DIR / "history.json").open("w") as handle:
                json.dump(history, handle, indent=2)
                handle.write("\n")

            if bad_epochs >= cfg.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping after {epoch} epochs")
                print(f"Best checkpoint: {cfg.OUT_DIR / 'best_model.pt'}")
                return

        block_index += 1

    print(f"Training complete. Best checkpoint: {cfg.OUT_DIR / 'best_model.pt'}")


if __name__ == "__main__":
    main()
