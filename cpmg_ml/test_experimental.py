from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import experimental_config as cfg
from .checkpoint import load_trusted_checkpoint
from .experimental_data import (
    ExperimentalProfileDataset,
    collate_experimental,
    generate_experimental_profiles,
)
from .experimental_model import ExperimentalCPMGTransformer, masked_experimental_metrics
from .sampling import load_range_specs, specs_to_json
from .train import select_device
from .train_experimental import generation_options


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test experimental-ready CPMG de-J model.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/cpmg_experimental/test_report"))
    parser.add_argument("--num-profiles", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=9901)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--generation-workers", type=int, default=None)
    parser.add_argument("--num-example-plots", type=int, default=6)
    return parser.parse_args()


def save_profile_group(path, profiles, pred, uncertainty, indices, title) -> None:
    import matplotlib.pyplot as plt

    if len(indices) == 0:
        return
    cols = 2
    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), squeeze=False)
    for ax, index in zip(axes.ravel(), indices):
        item = profiles[index]
        length = len(item["nu_cpmg"])
        nu = item["nu_cpmg"]
        prediction = pred[index, :length].numpy()
        sigma = uncertainty[index, :length].numpy()
        ax.errorbar(nu, item["r2_measured"], yerr=item["esd"], fmt="o", ms=3, label="measured")
        ax.plot(nu, item["r2_target"], "-o", ms=3, label="target J=0")
        ax.plot(nu, prediction, "-o", ms=3, label="predicted J=0")
        ax.fill_between(nu, prediction - 1.96 * sigma, prediction + 1.96 * sigma, alpha=0.2)
        rmse = float(np.sqrt(np.mean((prediction - item["r2_target"]) ** 2)))
        ax.set_title(f"index={index}, RMSE={rmse:.3g}")
        ax.set_xlabel("nu_cpmg (Hz)")
        ax.set_ylabel("R2 (s^-1)")
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=8)
    for ax in axes.ravel()[len(indices) :]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = load_trusted_checkpoint(args.checkpoint)
    normalization = checkpoint["normalization"]
    workers = args.generation_workers or cfg.GENERATION_WORKERS
    profiles = generate_experimental_profiles(
        args.num_profiles,
        args.seed,
        specs_to_json(load_range_specs(cfg.RANGES_CSV)),
        generation_options(),
        workers,
        cfg.GENERATION_CHUNK_SIZE,
        "experimental test",
        cfg.GENERATION_PROGRESS_SECONDS,
    )
    loader = DataLoader(
        ExperimentalProfileDataset(profiles, normalization),
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(collate_experimental, max_length=cfg.MAX_PROFILE_LENGTH),
    )
    device = select_device(args.device)
    model = ExperimentalCPMGTransformer(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    means, sigmas, targets, measured, target_raw, masks, esds, nus = [], [], [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            mean, log_sigma = model(
                batch["points"].to(device), batch["b0"].to(device), batch["mask"].to(device)
            )
            means.append(mean.cpu())
            sigmas.append(log_sigma.cpu())
            targets.append(batch["target"])
            measured.append(batch["r2_measured"])
            target_raw.append(batch["r2_target_raw"])
            masks.append(batch["mask"])
            esds.append(batch["esd"])
            nus.append(batch["nu_cpmg"])
    mean = torch.cat(means)
    log_sigma = torch.cat(sigmas)
    mask = torch.cat(masks)
    target_raw_t = torch.cat(target_raw)
    measured_t = torch.cat(measured)
    metrics = masked_experimental_metrics(
        mean, log_sigma, torch.cat(targets), measured_t, target_raw_t, mask, normalization
    )
    with (args.out_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred = mean * normalization["r2_std"] + normalization["r2_mean"]
    uncertainty = torch.exp(log_sigma) * normalization["r2_std"]
    error = (pred - target_raw_t)[mask].numpy()
    uncertainty_valid = uncertainty[mask].numpy()
    esd_valid = torch.cat(esds)[mask].numpy()

    nu_valid = torch.cat(nus)[mask].numpy()
    b0_per_point = np.concatenate(
        [np.full(len(item["nu_cpmg"]), float(item["b0_mhz"])) for item in profiles]
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    axes[0].hist(error, bins=100)
    axes[0].set_title("Prediction residual")
    axes[0].set_xlabel("R2 error (s^-1)")
    axes[1].scatter(uncertainty_valid, np.abs(error), s=5, alpha=0.25)
    axes[1].plot([0, max(uncertainty_valid)], [0, max(uncertainty_valid)], "k--")
    axes[1].set_title("Uncertainty calibration")
    axes[1].set_xlabel("Predicted uncertainty")
    axes[1].set_ylabel("Absolute error")
    axes[2].scatter(esd_valid, np.abs(error), s=5, alpha=0.25)
    axes[2].set_title("Error vs experimental Esd")
    axes[2].set_xlabel("Esd(R2)")
    axes[2].set_ylabel("Absolute error")
    axes[3].scatter(nu_valid, np.abs(error), s=5, alpha=0.25)
    axes[3].set_title("Error vs nu_cpmg")
    axes[3].set_xlabel("nu_cpmg (Hz)")
    axes[3].set_ylabel("Absolute error")
    axes[4].scatter(b0_per_point, np.abs(error), s=5, alpha=0.25)
    axes[4].set_title("Error vs B0")
    axes[4].set_xlabel("B0 (MHz)")
    axes[4].set_ylabel("Absolute error")
    lengths = np.asarray([len(item["nu_cpmg"]) for item in profiles])
    profile_rmse = np.asarray(
        [
            np.sqrt(np.mean((pred[i, :length].numpy() - profiles[i]["r2_target"]) ** 2))
            for i, length in enumerate(lengths)
        ]
    )
    axes[5].scatter(lengths, profile_rmse, s=8, alpha=0.35)
    axes[5].set_title("Profile RMSE vs length")
    axes[5].set_xlabel("Profile length")
    axes[5].set_ylabel("RMSE (s^-1)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out_dir / "experimental_diagnostics.png", dpi=160)
    plt.close(fig)

    count = min(args.num_example_plots, len(profiles))
    rng = np.random.default_rng(args.seed)
    random_indices = rng.choice(len(profiles), size=count, replace=False)
    sorted_error = np.argsort(profile_rmse)
    high_noise = np.argsort([np.mean(item["esd"]) for item in profiles])[-count:][::-1]
    duplicate_counts = [
        len(item["nu_cpmg"]) - len(np.unique(item["nu_cpmg"])) for item in profiles
    ]
    duplicate_heavy = np.argsort(duplicate_counts)[-count:][::-1]
    save_profile_group(
        args.out_dir / "random_examples.png", profiles, pred, uncertainty, random_indices, "Random Profiles"
    )
    save_profile_group(
        args.out_dir / "best_examples.png", profiles, pred, uncertainty, sorted_error[:count], "Best Profiles"
    )
    save_profile_group(
        args.out_dir / "worst_examples.png", profiles, pred, uncertainty, sorted_error[-count:][::-1], "Worst Profiles"
    )
    save_profile_group(
        args.out_dir / "high_noise_examples.png", profiles, pred, uncertainty, high_noise, "High-noise Profiles"
    )
    save_profile_group(
        args.out_dir / "duplicate_heavy_examples.png",
        profiles,
        pred,
        uncertainty,
        duplicate_heavy,
        "Duplicate-heavy Profiles",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Wrote report to {args.out_dir}")


if __name__ == "__main__":
    main()
