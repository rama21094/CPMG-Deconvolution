from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .checkpoint import load_trusted_checkpoint
from .dataset import H5CPMGDataset, find_split_files
from .model import ConditionalCPMGDeJNet
from .train import evaluate_epoch, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained CPMG de-J-coupling model.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/cpmg_dej"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--slope-weight", type=float, default=0.1)
    parser.add_argument("--plot-dir", type=Path, default=None)
    parser.add_argument("--num-plots", type=int, default=8)
    return parser.parse_args()


def save_example_plots(model, loader, device, normalization, plot_dir: Path, num_plots: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    r2_mean = float(normalization["r2_mean"])
    r2_std = float(normalization["r2_std"])

    saved = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            metadata = batch["metadata"].to(device)
            pred_norm = model(x, metadata).cpu()
            pred = pred_norm * r2_std + r2_mean
            for i in range(pred.shape[0]):
                fig, ax = plt.subplots(figsize=(6, 4))
                nu_cp = batch["nu_cp"][i].numpy()
                ax.plot(nu_cp, batch["r2_with_j"][i].numpy(), "-o", label="with J")
                ax.plot(nu_cp, batch["r2_no_j"][i].numpy(), "-o", label="target J=0")
                ax.plot(nu_cp, pred[i].numpy(), "-o", label="predicted J=0")
                ax.set_xlabel("nu_cp (Hz)")
                ax.set_ylabel("R2_eff (s^-1)")
                ax.grid(True)
                ax.legend()
                fig.tight_layout()
                fig.savefig(plot_dir / f"example_{saved:03d}.png", dpi=150)
                plt.close(fig)
                saved += 1
                if saved >= num_plots:
                    return


def main() -> None:
    args = parse_args()
    checkpoint = load_trusted_checkpoint(args.checkpoint, map_location="cpu")
    normalization = checkpoint["normalization"]
    model = ConditionalCPMGDeJNet(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])

    device = select_device(args.device)
    model.to(device)

    files = find_split_files(args.data_dir, args.split)
    include_raw = args.plot_dir is not None and args.num_plots > 0
    dataset = H5CPMGDataset(files, normalization, include_raw=include_raw)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    metrics = evaluate_epoch(model, loader, device, normalization, args.slope_weight)
    print(json.dumps(metrics, indent=2, sort_keys=True))

    if include_raw:
        save_example_plots(model, loader, device, normalization, args.plot_dir, args.num_plots)
        print(f"Saved plots to {args.plot_dir}")


if __name__ == "__main__":
    main()
