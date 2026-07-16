from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_trusted_checkpoint
from .experimental_data import parse_spectrometer_profile, write_experimental_prediction_csv
from .experimental_model import ExperimentalCPMGTransformer
from .train import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a J=0 curve from an experimental CPMG profile.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--b0-mhz", type=float, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-plot", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def normalized_input(profile: dict, b0_mhz: float, norm: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    points = np.stack(
        [
            (np.log1p(profile["nu_cpmg"]) - norm["log_nu_mean"]) / norm["log_nu_std"],
            (profile["r2_measured"] - norm["r2_mean"]) / norm["r2_std"],
            (np.log(profile["esd"]) - norm["log_esd_mean"]) / norm["log_esd_std"],
        ],
        axis=1,
    ).astype(np.float32)
    return (
        torch.from_numpy(points).unsqueeze(0),
        torch.tensor([[(b0_mhz - norm["b0_mean"]) / norm["b0_std"]]], dtype=torch.float32),
        torch.ones(1, len(points), dtype=torch.bool),
    )


def save_plot(path: Path, rows: dict[str, np.ndarray]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(
        rows["nu_cpmg"],
        rows["R2_measured"],
        yerr=rows["Esd_R2"],
        fmt="o",
        label="measured with J",
        alpha=0.8,
    )
    ax.plot(rows["nu_cpmg"], rows["R2_J0_pred"], "-o", label="predicted J=0")
    ax.fill_between(
        rows["nu_cpmg"],
        rows["R2_J0_lower95"],
        rows["R2_J0_upper95"],
        alpha=0.25,
        label="95% prediction interval",
    )
    ax.set_xlabel("nu_cpmg (Hz)")
    ax.set_ylabel("R2 (s^-1)")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    profile = parse_spectrometer_profile(args.input)
    checkpoint = load_trusted_checkpoint(args.checkpoint)
    if checkpoint.get("pipeline") != "experimental_transformer_v1":
        raise ValueError("Checkpoint is not an experimental Transformer checkpoint")
    max_length = int(checkpoint.get("config", {}).get("MAX_PROFILE_LENGTH", 64))
    if len(profile["nu_cpmg"]) > max_length:
        raise ValueError(
            f"Profile has {len(profile['nu_cpmg'])} points, but this checkpoint was trained "
            f"for at most {max_length} points."
        )
    model = ExperimentalCPMGTransformer(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])
    device = select_device(args.device)
    model.to(device).eval()

    points, b0, mask = normalized_input(profile, args.b0_mhz, checkpoint["normalization"])
    with torch.no_grad():
        mean_norm, log_sigma_norm = model(points.to(device), b0.to(device), mask.to(device))
    norm = checkpoint["normalization"]
    prediction = mean_norm.cpu().squeeze(0).numpy() * norm["r2_std"] + norm["r2_mean"]
    uncertainty = np.exp(log_sigma_norm.cpu().squeeze(0).numpy()) * norm["r2_std"]
    rows = {
        "nu_cpmg": profile["nu_cpmg"],
        "R2_measured": profile["r2_measured"],
        "Esd_R2": profile["esd"],
        "R2_J0_pred": prediction,
        "R2_J0_uncertainty": uncertainty,
        "R2_J0_lower95": prediction - 1.96 * uncertainty,
        "R2_J0_upper95": prediction + 1.96 * uncertainty,
    }
    write_experimental_prediction_csv(args.out_csv, rows)
    if args.out_plot:
        save_plot(args.out_plot, rows)
    print(f"Wrote {args.out_csv}")
    if args.out_plot:
        print(f"Wrote {args.out_plot}")


if __name__ == "__main__":
    main()
