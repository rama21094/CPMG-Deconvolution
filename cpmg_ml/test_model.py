from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoint import load_trusted_checkpoint
from .dataset import H5CPMGDataset, find_split_files
from .model import ConditionalCPMGDeJNet
from .sampling import METADATA_KEYS
from .train import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a comprehensive CPMG de-J model test report.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/cpmg_dej"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/cpmg_dej/test_report"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-profiles", type=int, default=None)
    parser.add_argument("--num-random-plots", type=int, default=8)
    parser.add_argument("--num-best-worst", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[ConditionalCPMGDeJNet, dict]:
    checkpoint = load_trusted_checkpoint(checkpoint_path, map_location="cpu")
    model = ConditionalCPMGDeJNet(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint["normalization"]


def denormalize_r2(values: torch.Tensor, normalization: dict) -> torch.Tensor:
    return values * float(normalization["r2_std"]) + float(normalization["r2_mean"])


def collect_predictions(
    model: ConditionalCPMGDeJNet,
    loader: DataLoader,
    device: torch.device,
    normalization: dict,
    max_profiles: int | None,
) -> dict[str, np.ndarray]:
    chunks: dict[str, list[np.ndarray]] = {
        "nu_cp": [],
        "r2_with_j": [],
        "r2_no_j": [],
        "r2_pred": [],
        "metadata": [],
    }
    collected = 0

    with torch.no_grad():
        for batch in loader:
            remaining = None if max_profiles is None else max_profiles - collected
            if remaining is not None and remaining <= 0:
                break

            x = batch["x"].to(device)
            metadata_norm = batch["metadata"].to(device)
            pred_norm = model(x, metadata_norm).cpu()
            pred = denormalize_r2(pred_norm, normalization).numpy()

            batch_size = pred.shape[0]
            if remaining is not None:
                batch_size = min(batch_size, remaining)

            chunks["nu_cp"].append(batch["nu_cp"][:batch_size].numpy())
            chunks["r2_with_j"].append(batch["r2_with_j"][:batch_size].numpy())
            chunks["r2_no_j"].append(batch["r2_no_j"][:batch_size].numpy())
            chunks["r2_pred"].append(pred[:batch_size])
            chunks["metadata"].append(batch["metadata_raw"][:batch_size].numpy())

            collected += batch_size

    if collected == 0:
        raise ValueError("No profiles were collected for testing")

    return {key: np.concatenate(value, axis=0) for key, value in chunks.items()}


def compute_metrics(arrays: dict[str, np.ndarray]) -> tuple[dict[str, float], list[dict[str, float]]]:
    pred = arrays["r2_pred"]
    target = arrays["r2_no_j"]
    identity = arrays["r2_with_j"]
    metadata = arrays["metadata"]

    err = pred - target
    identity_err = identity - target
    profile_mse = np.mean(err**2, axis=1)
    profile_identity_mse = np.mean(identity_err**2, axis=1)
    profile_rmse = np.sqrt(profile_mse)
    profile_identity_rmse = np.sqrt(profile_identity_mse)
    profile_mae = np.mean(np.abs(err), axis=1)
    profile_identity_mae = np.mean(np.abs(identity_err), axis=1)
    profile_max_abs = np.max(np.abs(err), axis=1)
    profile_identity_max_abs = np.max(np.abs(identity_err), axis=1)

    global_rmse = float(np.sqrt(np.mean(err**2)))
    identity_rmse = float(np.sqrt(np.mean(identity_err**2)))
    improvement = 0.0
    if identity_rmse > 0:
        improvement = 100.0 * (identity_rmse - global_rmse) / identity_rmse

    metrics = {
        "profile_count": int(pred.shape[0]),
        "profile_length": int(pred.shape[1]),
        "rmse": global_rmse,
        "mae": float(np.mean(np.abs(err))),
        "max_abs": float(np.max(np.abs(err))),
        "identity_rmse": identity_rmse,
        "identity_mae": float(np.mean(np.abs(identity_err))),
        "identity_max_abs": float(np.max(np.abs(identity_err))),
        "identity_improvement_pct": float(improvement),
        "median_profile_rmse": float(np.median(profile_rmse)),
        "p95_profile_rmse": float(np.percentile(profile_rmse, 95)),
    }

    rows = []
    for idx in range(pred.shape[0]):
        row = {
            "index": int(idx),
            "rmse": float(profile_rmse[idx]),
            "mae": float(profile_mae[idx]),
            "max_abs": float(profile_max_abs[idx]),
            "identity_rmse": float(profile_identity_rmse[idx]),
            "identity_mae": float(profile_identity_mae[idx]),
            "identity_max_abs": float(profile_identity_max_abs[idx]),
        }
        for meta_idx, key in enumerate(METADATA_KEYS):
            row[key] = float(metadata[idx, meta_idx])
        rows.append(row)

    return metrics, rows


def write_json(path: Path, value: dict) -> None:
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def profile_grid_plot(
    path: Path,
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    title: str,
) -> None:
    plt = setup_matplotlib()
    if len(indices) == 0:
        return
    cols = 2
    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), squeeze=False)

    for ax, idx in zip(axes.ravel(), indices):
        nu_cp = arrays["nu_cp"][idx]
        ax.plot(nu_cp, arrays["r2_with_j"][idx], "-o", ms=3, label="with J")
        ax.plot(nu_cp, arrays["r2_no_j"][idx], "-o", ms=3, label="target J=0")
        ax.plot(nu_cp, arrays["r2_pred"][idx], "-o", ms=3, label="predicted J=0")
        rmse = float(np.sqrt(np.mean((arrays["r2_pred"][idx] - arrays["r2_no_j"][idx]) ** 2)))
        ax.set_title(f"index={idx}, RMSE={rmse:.4g}")
        ax.set_xlabel("nu_cp (Hz)")
        ax.set_ylabel("R2_eff (s^-1)")
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=8)

    for ax in axes.ravel()[len(indices) :]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parity_plot(path: Path, arrays: dict[str, np.ndarray]) -> None:
    plt = setup_matplotlib()
    target = arrays["r2_no_j"].reshape(-1)
    pred = arrays["r2_pred"].reshape(-1)
    max_points = 200_000
    if len(target) > max_points:
        rng = np.random.default_rng(0)
        keep = rng.choice(len(target), size=max_points, replace=False)
        target = target[keep]
        pred = pred[keep]

    low = float(min(np.min(target), np.min(pred)))
    high = float(max(np.max(target), np.max(pred)))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(target, pred, s=4, alpha=0.25)
    ax.plot([low, high], [low, high], "k--", lw=1)
    ax.set_xlabel("Target R2_eff J=0 (s^-1)")
    ax.set_ylabel("Predicted R2_eff J=0 (s^-1)")
    ax.set_title("Prediction Parity")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def residual_histogram(path: Path, arrays: dict[str, np.ndarray]) -> None:
    plt = setup_matplotlib()
    residuals = (arrays["r2_pred"] - arrays["r2_no_j"]).reshape(-1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuals, bins=100, alpha=0.85)
    ax.axvline(0.0, color="k", lw=1)
    ax.set_xlabel("Prediction residual (s^-1)")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def rmse_histogram(path: Path, per_profile_rows: list[dict[str, float]]) -> None:
    plt = setup_matplotlib()
    rmse = np.asarray([row["rmse"] for row in per_profile_rows])
    identity_rmse = np.asarray([row["identity_rmse"] for row in per_profile_rows])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(identity_rmse, bins=80, alpha=0.55, label="identity baseline")
    ax.hist(rmse, bins=80, alpha=0.75, label="model")
    ax.set_xlabel("Per-profile RMSE (s^-1)")
    ax.set_ylabel("Count")
    ax.set_title("Per-profile Error Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def mean_error_vs_nu(path: Path, arrays: dict[str, np.ndarray]) -> None:
    plt = setup_matplotlib()
    abs_err = np.abs(arrays["r2_pred"] - arrays["r2_no_j"])
    mean_abs_err = np.mean(abs_err, axis=0)
    p95_abs_err = np.percentile(abs_err, 95, axis=0)
    mean_nu = np.mean(arrays["nu_cp"], axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(mean_nu, mean_abs_err, "-o", label="mean absolute error")
    ax.plot(mean_nu, p95_abs_err, "-o", label="95th percentile absolute error")
    ax.set_xlabel("Mean nu_cp at profile index (Hz)")
    ax.set_ylabel("Absolute error (s^-1)")
    ax.set_title("Error Across CPMG Profile")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def identity_vs_model_plot(path: Path, per_profile_rows: list[dict[str, float]]) -> None:
    plt = setup_matplotlib()
    rmse = np.asarray([row["rmse"] for row in per_profile_rows])
    identity_rmse = np.asarray([row["identity_rmse"] for row in per_profile_rows])
    high = float(max(np.max(rmse), np.max(identity_rmse)))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(identity_rmse, rmse, s=8, alpha=0.45)
    ax.plot([0, high], [0, high], "k--", lw=1)
    ax.set_xlabel("Identity baseline RMSE (s^-1)")
    ax.set_ylabel("Model RMSE (s^-1)")
    ax.set_title("Model Error vs Do-Nothing Baseline")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def error_vs_metadata_plot(path: Path, per_profile_rows: list[dict[str, float]]) -> None:
    plt = setup_matplotlib()
    keys = ["J_IS", "B0_MHz", "B1_N", "k_ex", "p_B", "dw_N", "tau_m", "S2"]
    rmse = np.asarray([row["rmse"] for row in per_profile_rows])
    fig, axes = plt.subplots(4, 2, figsize=(12, 14), squeeze=False)
    for ax, key in zip(axes.ravel(), keys):
        values = np.asarray([row[key] for row in per_profile_rows])
        ax.scatter(values, rmse, s=8, alpha=0.45)
        ax.set_xlabel(key)
        ax.set_ylabel("RMSE (s^-1)")
        ax.grid(True, alpha=0.35)
        if key in {"tau_m", "k_ex"}:
            ax.set_xscale("log")
    fig.suptitle("Error vs Metadata")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_plots(
    out_dir: Path,
    arrays: dict[str, np.ndarray],
    per_profile_rows: list[dict[str, float]],
    num_random: int,
    num_best_worst: int,
    seed: int,
) -> None:
    n_profiles = arrays["r2_pred"].shape[0]
    rmse = np.asarray([row["rmse"] for row in per_profile_rows])
    rng = np.random.default_rng(seed)

    random_count = min(num_random, n_profiles)
    random_indices = rng.choice(n_profiles, size=random_count, replace=False)
    sorted_indices = np.argsort(rmse)
    best_indices = sorted_indices[: min(num_best_worst, n_profiles)]
    worst_indices = sorted_indices[-min(num_best_worst, n_profiles) :][::-1]

    profile_grid_plot(out_dir / "overlay_examples.png", arrays, random_indices, "Random Example Profiles")
    profile_grid_plot(out_dir / "best_examples.png", arrays, best_indices, "Best Example Profiles")
    profile_grid_plot(out_dir / "worst_examples.png", arrays, worst_indices, "Worst Example Profiles")
    parity_plot(out_dir / "parity_pred_vs_target.png", arrays)
    residual_histogram(out_dir / "residual_histogram.png", arrays)
    rmse_histogram(out_dir / "rmse_histogram.png", per_profile_rows)
    mean_error_vs_nu(out_dir / "mean_profile_error_vs_nu_cp.png", arrays)
    identity_vs_model_plot(out_dir / "identity_vs_model_rmse.png", per_profile_rows)
    error_vs_metadata_plot(out_dir / "error_vs_metadata.png", per_profile_rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    model, normalization = load_model(args.checkpoint, device)

    files = find_split_files(args.data_dir, args.split)
    dataset = H5CPMGDataset(files, normalization, include_raw=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    arrays = collect_predictions(model, loader, device, normalization, args.max_profiles)
    metrics, per_profile_rows = compute_metrics(arrays)
    metrics.update(
        {
            "checkpoint": str(args.checkpoint),
            "data_dir": str(args.data_dir),
            "split": args.split,
            "device": str(device),
        }
    )

    write_json(args.out_dir / "metrics.json", metrics)
    write_csv(args.out_dir / "per_profile_metrics.csv", per_profile_rows)
    worst_rows = sorted(per_profile_rows, key=lambda row: row["rmse"], reverse=True)[
        : min(100, len(per_profile_rows))
    ]
    write_csv(args.out_dir / "worst_examples.csv", worst_rows)
    make_plots(
        args.out_dir,
        arrays,
        per_profile_rows,
        args.num_random_plots,
        args.num_best_worst,
        args.seed,
    )

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Wrote comprehensive test report to {args.out_dir}")


if __name__ == "__main__":
    main()
