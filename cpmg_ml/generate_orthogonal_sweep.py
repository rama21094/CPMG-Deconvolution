from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .simulator import CPMGSimulator, b0_mhz_to_tesla


DEFAULT_OUT_DIR = Path("data/orthogonal_sweep_kex_pb_dw")
DEFAULT_T_RELAX = 0.04
DEFAULT_NCYC_GRID = np.arange(1, 80, 2, dtype=np.int32)

DEFAULT_B0_MHZ = [600.0, 800.0]
DEFAULT_J_IS = 92.0
DEFAULT_KEX_RANGE = (300.0, 2000.0, 10)
DEFAULT_PB_RANGE = (0.05, 0.07, 10)
DEFAULT_DW_RANGE = (1.0, 6.0, 15)

FIXED_GUI_DEFAULTS = {
    "B1_N": 5555.0,
    "tau_m": 5.0e-9,
    "tau_e": 50.0e-12,
    "S2": 0.85,
    "r_IS": 1.02e-10,
    "r_eff": 1.86e-10,
    "csa_N": -160.0e-6,
    "theta_N": math.radians(22.0),
}

METADATA_KEYS = [
    "sim_id",
    "B0_MHz",
    "B0_T",
    "J_IS",
    "k_ex",
    "p_B",
    "dw_N",
    "B1_N",
    "tau_m",
    "tau_e",
    "S2",
    "r_IS",
    "r_eff",
    "csa_N",
    "theta_N_rad",
    "theta_N_deg",
    "valid_points",
    "nonfinite_points",
    "skipped_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an orthogonal CPMG simulator sweep over k_ex, p_B, dw_N, and B0."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--compression", choices=["gzip", "lzf", "none"], default="gzip")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--kex-count", type=int, default=DEFAULT_KEX_RANGE[2])
    parser.add_argument("--pb-count", type=int, default=DEFAULT_PB_RANGE[2])
    parser.add_argument("--dw-count", type=int, default=DEFAULT_DW_RANGE[2])
    parser.add_argument("--b0-mhz", type=float, nargs="+", default=DEFAULT_B0_MHZ)
    return parser.parse_args()


def linspace_from_default(default_range: tuple[float, float, int], count: int) -> np.ndarray:
    low, high, _ = default_range
    if count < 1:
        raise ValueError("Grid counts must be at least 1")
    return np.linspace(low, high, count, dtype=np.float64)


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    kex_values = linspace_from_default(DEFAULT_KEX_RANGE, args.kex_count)
    pb_values = linspace_from_default(DEFAULT_PB_RANGE, args.pb_count)
    dw_values = linspace_from_default(DEFAULT_DW_RANGE, args.dw_count)
    b0_values = np.asarray(args.b0_mhz, dtype=np.float64)
    if np.any(b0_values <= 0.0):
        raise ValueError("All B0 MHz values must be positive")

    total_count = int(len(b0_values) * len(kex_values) * len(pb_values) * len(dw_values))
    default_count = (
        len(DEFAULT_B0_MHZ)
        * DEFAULT_KEX_RANGE[2]
        * DEFAULT_PB_RANGE[2]
        * DEFAULT_DW_RANGE[2]
    )

    return {
        "b0_mhz_values": b0_values.tolist(),
        "j_is_hz": DEFAULT_J_IS,
        "k_ex_values": kex_values.tolist(),
        "p_b_values": pb_values.tolist(),
        "dw_n_ppm_values": dw_values.tolist(),
        "fixed_parameters": {
            **FIXED_GUI_DEFAULTS,
            "theta_N_deg": 22.0,
        },
        "ncyc_grid": DEFAULT_NCYC_GRID.astype(int).tolist(),
        "t_relax": DEFAULT_T_RELAX,
        "profile_length": int(len(DEFAULT_NCYC_GRID)),
        "total_simulations": total_count,
        "default_total_simulations": default_count,
        "metadata_keys": METADATA_KEYS,
    }


def make_params(b0_mhz: float, k_ex: float, p_b: float, dw_n: float) -> dict[str, float]:
    return {
        "B0": b0_mhz_to_tesla(b0_mhz),
        "J_IS": DEFAULT_J_IS,
        "k_ex": k_ex,
        "p_B": p_b,
        "dw_N": dw_n,
        **FIXED_GUI_DEFAULTS,
    }


def align_profile(profile_nu: np.ndarray, profile_r2: np.ndarray, skipped_ncyc: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    profile_len = len(DEFAULT_NCYC_GRID)
    aligned_nu = np.full(profile_len, np.nan, dtype=np.float32)
    aligned_r2 = np.full(profile_len, np.nan, dtype=np.float32)
    skipped = set(skipped_ncyc)
    kept_indices = [idx for idx, ncyc in enumerate(DEFAULT_NCYC_GRID) if int(ncyc) not in skipped]

    if len(kept_indices) != len(profile_nu) or len(profile_nu) != len(profile_r2):
        raise ValueError(
            "Could not align simulator output to ncyc grid: "
            f"kept={len(kept_indices)} nu={len(profile_nu)} r2={len(profile_r2)}"
        )

    aligned_nu[kept_indices] = profile_nu.astype(np.float32)
    aligned_r2[kept_indices] = profile_r2.astype(np.float32)
    return aligned_nu, aligned_r2


def metadata_row(
    sim_id: int,
    b0_mhz: float,
    params: dict[str, float],
    valid_points: int,
    nonfinite_points: int,
    skipped_count: int,
) -> dict[str, float]:
    return {
        "sim_id": sim_id,
        "B0_MHz": b0_mhz,
        "B0_T": params["B0"],
        "J_IS": params["J_IS"],
        "k_ex": params["k_ex"],
        "p_B": params["p_B"],
        "dw_N": params["dw_N"],
        "B1_N": params["B1_N"],
        "tau_m": params["tau_m"],
        "tau_e": params["tau_e"],
        "S2": params["S2"],
        "r_IS": params["r_IS"],
        "r_eff": params["r_eff"],
        "csa_N": params["csa_N"],
        "theta_N_rad": params["theta_N"],
        "theta_N_deg": math.degrees(params["theta_N"]),
        "valid_points": valid_points,
        "nonfinite_points": nonfinite_points,
        "skipped_count": skipped_count,
    }


def ensure_output_paths(out_dir: Path, overwrite: bool) -> dict[str, Path]:
    paths = {
        "h5": out_dir / "orthogonal_sweep.h5",
        "metadata_csv": out_dir / "metadata.csv",
        "profiles_csv": out_dir / "profiles_long.csv",
        "config_json": out_dir / "config.json",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output files already exist: {names}. Use --overwrite to replace them.")
    return paths


def write_outputs(
    paths: dict[str, Path],
    config: dict[str, Any],
    r2_eff: np.ndarray,
    nu_cp: np.ndarray,
    metadata: np.ndarray,
    metadata_rows: list[dict[str, Any]],
    skipped_rows: list[str],
    compression: str,
) -> None:
    with paths["config_json"].open("w") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")

    with paths["metadata_csv"].open("w", newline="") as handle:
        fieldnames = METADATA_KEYS + ["skipped_ncyc"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, skipped in zip(metadata_rows, skipped_rows, strict=True):
            writer.writerow({**row, "skipped_ncyc": skipped})

    with paths["profiles_csv"].open("w", newline="") as handle:
        fieldnames = [
            "sim_id",
            "point_index",
            "ncyc",
            "nu_cp",
            "R2_eff",
            "B0_MHz",
            "k_ex",
            "p_B",
            "dw_N",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_idx, row in enumerate(metadata_rows):
            for point_index, ncyc in enumerate(DEFAULT_NCYC_GRID):
                writer.writerow(
                    {
                        "sim_id": int(row["sim_id"]),
                        "point_index": point_index,
                        "ncyc": int(ncyc),
                        "nu_cp": float(nu_cp[row_idx, point_index]),
                        "R2_eff": float(r2_eff[row_idx, point_index]),
                        "B0_MHz": float(row["B0_MHz"]),
                        "k_ex": float(row["k_ex"]),
                        "p_B": float(row["p_B"]),
                        "dw_N": float(row["dw_N"]),
                    }
                )

    h5_kwargs = {}
    if compression != "none":
        h5_kwargs["compression"] = compression
        if compression == "gzip":
            h5_kwargs["compression_opts"] = 4

    with h5py.File(paths["h5"], "w") as handle:
        profile_chunks = (min(len(r2_eff), 256), r2_eff.shape[1])
        handle.create_dataset("r2_eff", data=r2_eff, chunks=profile_chunks, **h5_kwargs)
        handle.create_dataset("nu_cp", data=nu_cp, chunks=profile_chunks, **h5_kwargs)
        handle.create_dataset("ncyc", data=DEFAULT_NCYC_GRID.astype(np.int32))
        handle.create_dataset("metadata", data=metadata.astype(np.float32), chunks=(min(len(metadata), 256), metadata.shape[1]), **h5_kwargs)
        handle.attrs["metadata_keys"] = json.dumps(METADATA_KEYS)
        handle.attrs["config_json"] = json.dumps(config)


def main() -> None:
    args = parse_args()
    if args.progress_every < 1:
        raise ValueError("--progress-every must be at least 1")

    config = build_config(args)
    expected_count = int(config["total_simulations"])
    default_count = int(config["default_total_simulations"])
    if expected_count != default_count:
        print(f"Custom grid requested: generating {expected_count} simulations instead of default {default_count}.")

    paths = ensure_output_paths(args.out_dir, args.overwrite)
    sim = CPMGSimulator()

    r2_eff = np.full((expected_count, len(DEFAULT_NCYC_GRID)), np.nan, dtype=np.float32)
    nu_cp = np.full_like(r2_eff, np.nan)
    metadata = np.full((expected_count, len(METADATA_KEYS)), np.nan, dtype=np.float32)
    metadata_rows: list[dict[str, Any]] = []
    skipped_rows: list[str] = []

    profile_with_skips = 0
    profile_with_nonfinite = 0
    total_nonfinite = 0

    grid = itertools.product(
        config["b0_mhz_values"],
        config["k_ex_values"],
        config["p_b_values"],
        config["dw_n_ppm_values"],
    )

    for sim_id, (b0_mhz, k_ex, p_b, dw_n) in enumerate(grid):
        params = make_params(float(b0_mhz), float(k_ex), float(p_b), float(dw_n))
        profile = sim.simulate_cpmg(params, ncyc_range=DEFAULT_NCYC_GRID, t_relax=DEFAULT_T_RELAX)
        aligned_nu, aligned_r2 = align_profile(profile.nu_cp, profile.r2_eff, profile.skipped_ncyc)

        finite_mask = np.isfinite(aligned_r2)
        skipped_count = len(profile.skipped_ncyc)
        nonfinite_points = int(np.count_nonzero(~finite_mask))
        valid_points = int(np.count_nonzero(finite_mask))

        if skipped_count:
            profile_with_skips += 1
        if nonfinite_points:
            profile_with_nonfinite += 1
            total_nonfinite += nonfinite_points

        row = metadata_row(sim_id, float(b0_mhz), params, valid_points, nonfinite_points, skipped_count)
        r2_eff[sim_id] = aligned_r2
        nu_cp[sim_id] = aligned_nu
        metadata[sim_id] = np.asarray([row[key] for key in METADATA_KEYS], dtype=np.float32)
        metadata_rows.append(row)
        skipped_rows.append(";".join(str(value) for value in profile.skipped_ncyc))

        done = sim_id + 1
        if done % args.progress_every == 0 or done == expected_count:
            print(f"Generated {done}/{expected_count} profiles")

    if len(metadata_rows) != expected_count:
        raise RuntimeError(f"Generated {len(metadata_rows)} simulations, expected {expected_count}")
    if expected_count == default_count and expected_count != 3000:
        raise RuntimeError(f"Default sweep should generate 3000 simulations, got {expected_count}")

    if profile_with_skips:
        print(f"WARNING: {profile_with_skips} profiles skipped one or more ncyc points.")
    if profile_with_nonfinite:
        print(
            "WARNING: "
            f"{profile_with_nonfinite} profiles contain non-finite R2_eff values "
            f"({total_nonfinite} points total)."
        )

    write_outputs(
        paths,
        config,
        r2_eff,
        nu_cp,
        metadata,
        metadata_rows,
        skipped_rows,
        args.compression,
    )

    print("Orthogonal sweep complete.")
    print(f"Total simulations: {expected_count}")
    print(f"Total profile points: {expected_count * len(DEFAULT_NCYC_GRID)}")
    print(f"Profiles with skipped points: {profile_with_skips}")
    print(f"Profiles with non-finite values: {profile_with_nonfinite}")
    print(f"HDF5: {paths['h5']}")
    print(f"Metadata CSV: {paths['metadata_csv']}")
    print(f"Long profile CSV: {paths['profiles_csv']}")
    print(f"Config: {paths['config_json']}")


if __name__ == "__main__":
    main()
