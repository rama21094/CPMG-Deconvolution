from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from .dataset import find_split_files
from .sampling import METADATA_KEYS, load_range_specs
from .simulator import b0_mhz_to_tesla


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated CPMG de-J-coupling HDF5 shards.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/cpmg_dej"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--ranges-csv", type=Path, default=None)
    parser.add_argument("--expected-length", type=int, default=40)
    return parser.parse_args()


def check_range(name: str, values: np.ndarray, low: float, high: float) -> None:
    eps = max(abs(low), abs(high), 1.0) * 1.0e-5
    observed_low = float(np.min(values))
    observed_high = float(np.max(values))
    if observed_low < low - eps or observed_high > high + eps:
        raise ValueError(
            f"{name} out of range: observed [{observed_low}, {observed_high}], "
            f"expected [{low}, {high}]"
        )


def validate_file(path: Path, specs: dict, expected_length: int) -> tuple[int, float]:
    with h5py.File(path, "r") as handle:
        r2_with = handle["r2_with_j"][:]
        r2_no = handle["r2_no_j"][:]
        nu_cp = handle["nu_cp"][:]
        metadata = handle["metadata"][:]

    if r2_with.shape != r2_no.shape or r2_with.shape != nu_cp.shape:
        raise ValueError(f"Profile arrays have mismatched shapes in {path}")
    if r2_with.shape[1] != expected_length:
        raise ValueError(f"Expected profile length {expected_length}, got {r2_with.shape[1]} in {path}")
    for name, array in {
        "r2_with_j": r2_with,
        "r2_no_j": r2_no,
        "nu_cp": nu_cp,
        "metadata": metadata,
    }.items():
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite values in {path}")

    if metadata.shape[1] != len(METADATA_KEYS):
        raise ValueError(f"Expected {len(METADATA_KEYS)} metadata values, got {metadata.shape[1]}")

    for idx, key in enumerate(METADATA_KEYS):
        spec = specs[key]
        check_range(key, metadata[:, idx], spec.low, spec.high)

    mean_abs_delta = float(np.mean(np.abs(r2_with - r2_no)))
    return int(r2_with.shape[0]), mean_abs_delta


def main() -> None:
    args = parse_args()
    specs = load_range_specs(args.ranges_csv)
    files = find_split_files(args.data_dir, args.split)

    total = 0
    deltas = []
    for path in files:
        count, mean_abs_delta = validate_file(path, specs, args.expected_length)
        total += count
        deltas.append(mean_abs_delta)
        print(f"{path}: ok count={count} mean_abs_delta={mean_abs_delta:.6g}")

    b0_600 = b0_mhz_to_tesla(600.0)
    if not np.isclose(b0_600, 14.091967, rtol=1.0e-5):
        raise ValueError(f"600 MHz conversion check failed: got {b0_600}")
    print(f"Validated {total} {args.split} examples. 600 MHz = {b0_600:.6f} T.")
    print(f"Mean |with_J - no_J| across shards: {float(np.mean(deltas)):.6g}")


if __name__ == "__main__":
    main()
