from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .sampling import (
    METADATA_KEYS,
    load_range_specs,
    sample_simulation_case,
    specs_to_json,
)
from .simulator import CPMGSimulator, DEFAULT_NCYC_GRID, DEFAULT_T_RELAX


def build_ncyc_grid(start: int, stop: int, step: int) -> np.ndarray:
    if start < 1:
        raise ValueError("ncyc start must be at least 1")
    if stop <= start:
        raise ValueError("ncyc stop must be greater than start")
    if step < 1:
        raise ValueError("ncyc step must be at least 1")
    grid = np.arange(start, stop, step, dtype=np.int32)
    if len(grid) == 0:
        raise ValueError("ncyc grid is empty")
    return grid


def _valid_profile(values: np.ndarray, expected_len: int) -> bool:
    return len(values) == expected_len and bool(np.all(np.isfinite(values)))


def generate_one_example(
    seed: int,
    specs_json: str,
    ncyc_grid: np.ndarray,
    t_relax: float,
    max_retries: int,
    sequence: str = "home",
) -> dict[str, Any]:
    specs = load_specs_from_json(specs_json)
    rng = np.random.default_rng(seed)
    sim = CPMGSimulator()
    expected_len = len(ncyc_grid)

    for _ in range(max_retries):
        params, metadata = sample_simulation_case(rng, specs)
        with_j, no_j = sim.simulate_dej_pair(
            params, ncyc_range=ncyc_grid, t_relax=t_relax, sequence=sequence
        )
        same_axis = len(with_j.nu_cp) == len(no_j.nu_cp) and np.allclose(with_j.nu_cp, no_j.nu_cp)
        no_skips = not with_j.skipped_ncyc and not no_j.skipped_ncyc
        valid = (
            no_skips
            and same_axis
            and _valid_profile(with_j.r2_eff, expected_len)
            and _valid_profile(no_j.r2_eff, expected_len)
            and _valid_profile(with_j.nu_cp, expected_len)
        )
        if valid:
            return {
                "r2_with_j": with_j.r2_eff.astype(np.float32),
                "r2_no_j": no_j.r2_eff.astype(np.float32),
                "nu_cp": with_j.nu_cp.astype(np.float32),
                "metadata": metadata.astype(np.float32),
            }

    raise RuntimeError(f"Could not generate a valid profile after {max_retries} attempts")


def load_specs_from_json(specs_json: str):
    from .sampling import RangeSpec

    raw = json.loads(specs_json)
    return {
        key: RangeSpec(
            low=float(value["low"]),
            high=float(value["high"]),
            distribution=str(value["distribution"]),
        )
        for key, value in raw.items()
    }


def write_shard(
    path: Path,
    examples: list[dict[str, Any]],
    ncyc_grid: np.ndarray,
    t_relax: float,
    specs_json: str,
    compression: str | None,
    sequence: str = "home",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = len(examples)
    profile_len = len(ncyc_grid)
    metadata_dim = len(METADATA_KEYS)

    kwargs = {}
    if compression and compression != "none":
        kwargs = {"compression": compression, "compression_opts": 4 if compression == "gzip" else None}

    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "r2_with_j",
            data=np.stack([item["r2_with_j"] for item in examples]).astype(np.float32),
            shape=(count, profile_len),
            chunks=(min(count, 1024), profile_len),
            **{key: value for key, value in kwargs.items() if value is not None},
        )
        handle.create_dataset(
            "r2_no_j",
            data=np.stack([item["r2_no_j"] for item in examples]).astype(np.float32),
            shape=(count, profile_len),
            chunks=(min(count, 1024), profile_len),
            **{key: value for key, value in kwargs.items() if value is not None},
        )
        handle.create_dataset(
            "nu_cp",
            data=np.stack([item["nu_cp"] for item in examples]).astype(np.float32),
            shape=(count, profile_len),
            chunks=(min(count, 1024), profile_len),
            **{key: value for key, value in kwargs.items() if value is not None},
        )
        handle.create_dataset(
            "metadata",
            data=np.stack([item["metadata"] for item in examples]).astype(np.float32),
            shape=(count, metadata_dim),
            chunks=(min(count, 1024), metadata_dim),
            **{key: value for key, value in kwargs.items() if value is not None},
        )
        handle.attrs["metadata_keys"] = json.dumps(METADATA_KEYS)
        handle.attrs["ncyc_grid"] = json.dumps([int(value) for value in ncyc_grid])
        handle.attrs["t_relax"] = float(t_relax)
        handle.attrs["sequence"] = str(sequence)
        handle.attrs["range_specs"] = specs_json


def generate_examples(
    seeds: list[int],
    specs_json: str,
    ncyc_grid: np.ndarray,
    t_relax: float,
    max_retries: int,
    workers: int,
    sequence: str = "home",
) -> list[dict[str, Any]]:
    if workers <= 1:
        return [
            generate_one_example(seed, specs_json, ncyc_grid, t_relax, max_retries, sequence)
            for seed in seeds
        ]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(
            executor.map(
                generate_one_example,
                seeds,
                [specs_json] * len(seeds),
                [ncyc_grid] * len(seeds),
                [t_relax] * len(seeds),
                [max_retries] * len(seeds),
                [sequence] * len(seeds),
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paired CPMG with-J/no-J HDF5 shards.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/cpmg_dej"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--num-profiles", type=int, default=10_000)
    parser.add_argument("--shard-size", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--ranges-csv", type=Path, default=None)
    parser.add_argument(
        "--sequence",
        choices=["home", "chemex"],
        default="home",
        help="Pulse sequence used for BOTH profiles of each pair. "
             "'chemex' reproduces ChemEx cpmg_15n_ip (what a real spectrometer runs); "
             "note its ncyc means pulses per half-train, so pick the grid accordingly.",
    )
    parser.add_argument("--ncyc-start", type=int, default=int(DEFAULT_NCYC_GRID[0]))
    parser.add_argument("--ncyc-stop", type=int, default=80)
    parser.add_argument("--ncyc-step", type=int, default=2)
    parser.add_argument("--t-relax", type=float, default=DEFAULT_T_RELAX)
    parser.add_argument("--max-retries", type=int, default=100)
    parser.add_argument("--compression", choices=["gzip", "lzf", "none"], default="gzip")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_profiles < 1:
        raise ValueError("--num-profiles must be at least 1")
    if args.shard_size < 1:
        raise ValueError("--shard-size must be at least 1")

    specs = load_range_specs(args.ranges_csv)
    specs_json = specs_to_json(specs)
    ncyc_grid = build_ncyc_grid(args.ncyc_start, args.ncyc_stop, args.ncyc_step)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "range_specs.json").open("w") as handle:
        handle.write(specs_json)
        handle.write("\n")

    rng = np.random.default_rng(args.seed)
    seeds = rng.integers(0, np.iinfo(np.int32).max, size=args.num_profiles, dtype=np.int64).tolist()

    written = 0
    shard_index = 0
    while written < args.num_profiles:
        count = min(args.shard_size, args.num_profiles - written)
        shard_seeds = seeds[written : written + count]
        examples = generate_examples(
            shard_seeds,
            specs_json,
            ncyc_grid,
            args.t_relax,
            args.max_retries,
            args.workers,
            args.sequence,
        )
        shard_path = args.out_dir / f"{args.split}_{shard_index:04d}.h5"
        write_shard(
            shard_path,
            examples,
            ncyc_grid,
            args.t_relax,
            specs_json,
            None if args.compression == "none" else args.compression,
            args.sequence,
        )
        written += count
        shard_index += 1
        print(f"Wrote {shard_path} ({written}/{args.num_profiles})")


if __name__ == "__main__":
    main()
