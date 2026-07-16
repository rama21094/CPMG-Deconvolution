from __future__ import annotations

import argparse
import json
import math
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from . import experimental_config as cfg
from .checkpoint import load_trusted_checkpoint
from .dataset import save_normalization
from .experimental_data import (
    ExperimentalProfileDataset,
    collate_experimental,
    compute_experimental_normalization,
    generate_experimental_profiles,
)
from .experimental_model import (
    ExperimentalCPMGTransformer,
    masked_experimental_loss,
    masked_experimental_metrics,
)
from .sampling import load_range_specs, specs_to_json
from .train import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train experimental-ready CPMG de-J Transformer.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--generation-workers", type=int, default=None)
    parser.add_argument("--train-profiles-per-block", type=int, default=None)
    parser.add_argument("--normalization-profiles", type=int, default=None)
    parser.add_argument("--val-profiles", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def apply_overrides(args: argparse.Namespace) -> None:
    for key, value in {
        "DEVICE": args.device,
        "GENERATION_WORKERS": args.generation_workers,
        "TRAIN_PROFILES_PER_BLOCK": args.train_profiles_per_block,
        "NORMALIZATION_PROFILES": args.normalization_profiles,
        "VAL_PROFILES": args.val_profiles,
        "MAX_EPOCHS": args.max_epochs,
        "OUT_DIR": args.out_dir,
    }.items():
        if value is not None:
            setattr(cfg, key, value)
    if args.no_resume:
        cfg.RESUME = False


def config_dict() -> dict:
    return {
        key: str(getattr(cfg, key)) if isinstance(getattr(cfg, key), Path) else getattr(cfg, key)
        for key in dir(cfg)
        if key.isupper()
    }


def generation_options() -> dict:
    return {
        "min_length": cfg.MIN_PROFILE_LENGTH,
        "max_length": cfg.MAX_PROFILE_LENGTH,
        "min_nu_hz": cfg.MIN_NU_CPMG_HZ,
        "max_nu_hz": cfg.MAX_NU_CPMG_HZ,
        "min_t_relax": cfg.MIN_T_RELAX,
        "max_t_relax": cfg.MAX_T_RELAX,
        "min_esd": cfg.MIN_ESD,
        "max_esd": cfg.MAX_ESD,
        "max_duplicate_fraction": cfg.MAX_DUPLICATE_FRACTION,
        "max_ncyc_candidate": cfg.MAX_NCYC_CANDIDATE,
        "max_retries": cfg.MAX_RETRIES,
    }


def generate(label: str, count: int, seed: int, specs_json: str) -> list[dict]:
    print(
        f"Generating {label}: count={count} seed={seed} workers={cfg.GENERATION_WORKERS}",
        flush=True,
    )
    return generate_experimental_profiles(
        num_profiles=count,
        seed=seed,
        specs_json=specs_json,
        options=generation_options(),
        workers=cfg.GENERATION_WORKERS,
        chunk_size=cfg.GENERATION_CHUNK_SIZE,
        progress_label=label,
        progress_seconds=cfg.GENERATION_PROGRESS_SECONDS,
    )


def make_loader(profiles: list[dict], normalization: dict, shuffle: bool) -> DataLoader:
    dataset = ExperimentalProfileDataset(profiles, normalization)
    return DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=False,
        collate_fn=partial(collate_experimental, max_length=cfg.MAX_PROFILE_LENGTH),
    )


def make_model(device: torch.device) -> tuple[ExperimentalCPMGTransformer, dict]:
    kwargs = {
        "point_dim": 3,
        "global_dim": 1,
        "model_dim": cfg.MODEL_DIM,
        "num_heads": cfg.NUM_HEADS,
        "num_layers": cfg.NUM_LAYERS,
        "ff_dim": cfg.FF_DIM,
        "dropout": cfg.DROPOUT,
    }
    model = ExperimentalCPMGTransformer(**kwargs).to(device)
    if cfg.USE_TORCH_COMPILE and hasattr(torch, "compile"):
        model = torch.compile(model)
    return model, kwargs


def cosine_lr(epoch_in_block: int, base_lr: float) -> float:
    if cfg.EPOCHS_PER_BLOCK <= 1:
        return base_lr
    phase = epoch_in_block / (cfg.EPOCHS_PER_BLOCK - 1)
    return cfg.MIN_LR + 0.5 * (base_lr - cfg.MIN_LR) * (1.0 + math.cos(math.pi * phase))


def set_lr(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def evaluate(model, loader, device, normalization) -> dict[str, float]:
    model.eval()
    losses = []
    means = []
    log_sigmas = []
    targets = []
    measured = []
    target_raw = []
    masks = []
    with torch.no_grad():
        for batch in loader:
            points = batch["points"].to(device)
            b0 = batch["b0"].to(device)
            target = batch["target"].to(device)
            mask = batch["mask"].to(device)
            mean, log_sigma = model(points, b0, mask)
            loss, _ = masked_experimental_loss(mean, log_sigma, target, mask, cfg.SLOPE_WEIGHT)
            losses.append(loss.item() * points.shape[0])
            means.append(mean.cpu())
            log_sigmas.append(log_sigma.cpu())
            targets.append(target.cpu())
            measured.append(batch["r2_measured"])
            target_raw.append(batch["r2_target_raw"])
            masks.append(batch["mask"])
    metrics = masked_experimental_metrics(
        torch.cat(means),
        torch.cat(log_sigmas),
        torch.cat(targets),
        torch.cat(measured),
        torch.cat(target_raw),
        torch.cat(masks),
        normalization,
    )
    metrics["loss"] = sum(losses) / len(loader.dataset)
    return metrics


def save_checkpoint(
    path: Path,
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
) -> None:
    torch.save(
        {
            "pipeline": "experimental_transformer_v1",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_kwargs": model_kwargs,
            "normalization": normalization,
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


def resume(model, optimizer, path: Path, device: torch.device):
    if not cfg.RESUME or not path.exists():
        return 0, float("inf"), 0, 0, cfg.LR, []
    checkpoint = load_trusted_checkpoint(path)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
    print(f"Resumed {path} at epoch {checkpoint.get('epoch', 0)}")
    return (
        int(checkpoint.get("epoch", 0)),
        float(checkpoint.get("best_val", float("inf"))),
        int(checkpoint.get("bad_epochs", 0)),
        int(checkpoint.get("lr_bad_epochs", 0)),
        float(checkpoint.get("base_lr", cfg.LR)),
        list(checkpoint.get("history", [])),
    )


def main() -> None:
    apply_overrides(parse_args())
    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (cfg.OUT_DIR / "config.json").open("w") as handle:
        json.dump(config_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")

    specs_json = specs_to_json(load_range_specs(cfg.RANGES_CSV))
    norm_profiles = generate("normalization", cfg.NORMALIZATION_PROFILES, cfg.NORMALIZATION_SEED, specs_json)
    normalization = compute_experimental_normalization(norm_profiles)
    save_normalization(cfg.OUT_DIR / "normalization.json", normalization)
    val_profiles = generate("validation", cfg.VAL_PROFILES, cfg.VALIDATION_SEED, specs_json)
    val_loader = make_loader(val_profiles, normalization, shuffle=False)

    device = select_device(cfg.DEVICE)
    model, model_kwargs = make_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    epoch, best_val, bad_epochs, lr_bad_epochs, base_lr, history = resume(
        model, optimizer, cfg.OUT_DIR / "last_model.pt", device
    )
    amp_enabled = bool(cfg.USE_AMP and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    while epoch < cfg.MAX_EPOCHS:
        block_index = epoch // cfg.EPOCHS_PER_BLOCK
        train_seed = cfg.TRAIN_SEED + block_index
        train_profiles = generate(
            f"train block {block_index}", cfg.TRAIN_PROFILES_PER_BLOCK, train_seed, specs_json
        )
        train_loader = make_loader(train_profiles, normalization, shuffle=True)

        for epoch_in_block in range(epoch % cfg.EPOCHS_PER_BLOCK, cfg.EPOCHS_PER_BLOCK):
            if epoch >= cfg.MAX_EPOCHS:
                break
            set_lr(optimizer, cosine_lr(epoch_in_block, base_lr))
            epoch += 1
            model.train()
            loss_sum = 0.0
            count = 0

            for batch in train_loader:
                points = batch["points"].to(device)
                b0 = batch["b0"].to(device)
                target = batch["target"].to(device)
                mask = batch["mask"].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    mean, log_sigma = model(points, b0, mask)
                    loss, _ = masked_experimental_loss(
                        mean, log_sigma, target, mask, cfg.SLOPE_WEIGHT
                    )
                scaler.scale(loss).backward()
                if cfg.GRAD_CLIP_NORM > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                loss_sum += loss.item() * points.shape[0]
                count += points.shape[0]

            train_loss = loss_sum / count
            val_metrics = evaluate(model, val_loader, device, normalization)
            improved = val_metrics["loss"] < best_val
            if improved:
                best_val = val_metrics["loss"]
                bad_epochs = 0
                lr_bad_epochs = 0
            else:
                bad_epochs += 1
                lr_bad_epochs += 1
            if lr_bad_epochs >= cfg.LR_PLATEAU_PATIENCE:
                base_lr = max(base_lr * cfg.LR_PLATEAU_FACTOR, cfg.MIN_LR)
                lr_bad_epochs = 0
                print(f"Reduced base LR to {base_lr:.6g}")

            record = {
                "epoch": epoch,
                "block_index": block_index,
                "train_seed": train_seed,
                "train_loss": train_loss,
                "lr": optimizer.param_groups[0]["lr"],
                **val_metrics,
            }
            history.append(record)
            print(
                f"epoch={epoch} block={block_index} train_loss={train_loss:.6f} "
                f"val_loss={val_metrics['loss']:.6f} rmse={val_metrics['rmse']:.4f} "
                f"improvement={val_metrics['identity_improvement_pct']:.2f}% "
                f"coverage95={val_metrics['coverage_95']:.3f} "
                f"lr={optimizer.param_groups[0]['lr']:.6g}",
                flush=True,
            )

            if improved:
                save_checkpoint(
                    cfg.OUT_DIR / "best_model.pt", model, optimizer, model_kwargs, normalization,
                    epoch, block_index, best_val, bad_epochs, lr_bad_epochs, base_lr, history,
                    val_metrics,
                )
            if cfg.SAVE_EVERY_EPOCH:
                save_checkpoint(
                    cfg.OUT_DIR / "last_model.pt", model, optimizer, model_kwargs, normalization,
                    epoch, block_index, best_val, bad_epochs, lr_bad_epochs, base_lr, history,
                    val_metrics,
                )
            with (cfg.OUT_DIR / "history.json").open("w") as handle:
                json.dump(history, handle, indent=2)
                handle.write("\n")
            if bad_epochs >= cfg.EARLY_STOPPING_PATIENCE:
                print("Early stopping")
                return


if __name__ == "__main__":
    main()
