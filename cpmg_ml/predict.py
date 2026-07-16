from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np
import torch

from .checkpoint import load_trusted_checkpoint
from .model import ConditionalCPMGDeJNet
from .sampling import METADATA_KEYS
from .train import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a J=0 CPMG profile from a trained model.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-h5", type=Path)
    source.add_argument("--input-npz", type=Path)
    parser.add_argument("--index", type=int, default=0, help="Example index for --input-h5")
    return parser.parse_args()


def load_from_h5(path: Path, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    with h5py.File(path, "r") as handle:
        r2_with = handle["r2_with_j"][index].astype(np.float32)
        nu_cp = handle["nu_cp"][index].astype(np.float32)
        metadata = handle["metadata"][index].astype(np.float32)
        target = handle["r2_no_j"][index].astype(np.float32) if "r2_no_j" in handle else None
    return r2_with, nu_cp, metadata, target


def load_from_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    data = np.load(path)
    required = {"r2_with_j", "nu_cp", "metadata"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"NPZ is missing required arrays: {sorted(missing)}")
    target = data["r2_no_j"].astype(np.float32) if "r2_no_j" in data.files else None
    return (
        data["r2_with_j"].astype(np.float32),
        data["nu_cp"].astype(np.float32),
        data["metadata"].astype(np.float32),
        target,
    )


def normalize_inputs(
    r2_with: np.ndarray,
    nu_cp: np.ndarray,
    metadata: np.ndarray,
    normalization: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    if metadata.shape[-1] != len(METADATA_KEYS):
        raise ValueError(f"metadata must have {len(METADATA_KEYS)} values in this order: {METADATA_KEYS}")

    r2_mean = float(normalization["r2_mean"])
    r2_std = float(normalization["r2_std"])
    log_nu_mean = float(normalization["log_nu_mean"])
    log_nu_std = float(normalization["log_nu_std"])
    meta_mean = np.asarray(normalization["metadata_mean"], dtype=np.float32)
    meta_std = np.asarray(normalization["metadata_std"], dtype=np.float32)

    r2_norm = (r2_with - r2_mean) / r2_std
    log_nu_norm = (np.log1p(nu_cp) - log_nu_mean) / log_nu_std
    metadata_norm = (metadata - meta_mean) / meta_std

    x = torch.from_numpy(np.stack([r2_norm, log_nu_norm]).astype(np.float32)).unsqueeze(0)
    meta = torch.from_numpy(metadata_norm.astype(np.float32)).unsqueeze(0)
    return x, meta


def write_prediction(
    path: Path,
    nu_cp: np.ndarray,
    r2_with: np.ndarray,
    pred: np.ndarray,
    target: np.ndarray | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        fieldnames = ["nu_cp", "r2_with_j", "r2_no_j_pred"]
        if target is not None:
            fieldnames.append("r2_no_j_target")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(nu_cp)):
            row = {
                "nu_cp": float(nu_cp[i]),
                "r2_with_j": float(r2_with[i]),
                "r2_no_j_pred": float(pred[i]),
            }
            if target is not None:
                row["r2_no_j_target"] = float(target[i])
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    checkpoint = load_trusted_checkpoint(args.checkpoint, map_location="cpu")
    normalization = checkpoint["normalization"]
    model = ConditionalCPMGDeJNet(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])
    device = select_device(args.device)
    model.to(device)
    model.eval()

    if args.input_h5:
        r2_with, nu_cp, metadata, target = load_from_h5(args.input_h5, args.index)
    else:
        r2_with, nu_cp, metadata, target = load_from_npz(args.input_npz)

    x, meta = normalize_inputs(r2_with, nu_cp, metadata, normalization)
    with torch.no_grad():
        pred_norm = model(x.to(device), meta.to(device)).cpu().squeeze(0).numpy()
    pred = pred_norm * float(normalization["r2_std"]) + float(normalization["r2_mean"])
    write_prediction(args.out_csv, nu_cp, r2_with, pred, target)
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
