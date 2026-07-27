from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .simulator import CPMGSimulator, b0_mhz_to_tesla


DEFAULT_OUT_DIR = Path("data/chemex_orthogonal_sweep")
DEFAULT_T_RELAX = 0.04

# ChemEx's cpmg_15n_ip sequence applies TWO ncyc-pulse half-trains per labeled
# ncyc (plus one central compensating 180), i.e. 2*ncyc refocusing pulses total,
# whereas the home simulator applies exactly ncyc pulses for a labeled ncyc.
# DEFAULT_NCYC_GRID is therefore the number of pulses the home physics actually
# simulates; it must be even so CHEMEX_NCYC_LABEL (what gets written to the .out
# file's ncyc_cp column, and what ChemEx will read back as its own ncyc) is an
# exact integer matching ChemEx's convention.
DEFAULT_NCYC_GRID = np.arange(2, 82, 2, dtype=np.int32)
CHEMEX_NCYC_LABEL = DEFAULT_NCYC_GRID // 2

DEFAULT_B0_MHZ = [600.0, 800.0]
DEFAULT_J_IS = 92.0
DEFAULT_KEX_RANGE = (300.0, 2000.0, 10)
DEFAULT_PB_RANGE = (0.05, 0.07, 10)
DEFAULT_DW_RANGE = (1.0, 6.0, 15)

DEFAULT_I0 = 1_000_000.0
DEFAULT_RELATIVE_ESD = 0.005
DEFAULT_NUM_REPEATS = 3

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

METADATA_FIELDNAMES = [
    "sim_id",
    "case_id",
    "field_file",
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
    "I0",
    "relative_esd",
    "noise_enabled",
    "seed",
    "repeated_ncyc",
    "valid_points",
    "nonfinite_points",
    "skipped_count",
    "skipped_ncyc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ChemEx-style intensity files from the orthogonal CPMG simulator sweep."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)

    parser.add_argument("--i0", type=float, default=DEFAULT_I0)
    parser.add_argument("--relative-esd", type=float, default=DEFAULT_RELATIVE_ESD)
    parser.add_argument("--num-repeats", type=int, default=DEFAULT_NUM_REPEATS)
    parser.add_argument("--no-noise", action="store_true")
    parser.add_argument("--j-is", type=float, default=DEFAULT_J_IS)

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
    if args.i0 <= 0.0:
        raise ValueError("--i0 must be positive")
    if args.relative_esd < 0.0:
        raise ValueError("--relative-esd must be non-negative")
    if args.num_repeats < 0:
        raise ValueError("--num-repeats must be non-negative")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be at least 1")

    b0_values = np.asarray(args.b0_mhz, dtype=np.float64)
    if np.any(b0_values <= 0.0):
        raise ValueError("All B0 MHz values must be positive")

    kex_values = linspace_from_default(DEFAULT_KEX_RANGE, args.kex_count)
    pb_values = linspace_from_default(DEFAULT_PB_RANGE, args.pb_count)
    dw_values = linspace_from_default(DEFAULT_DW_RANGE, args.dw_count)

    total_count = int(len(b0_values) * len(kex_values) * len(pb_values) * len(dw_values))
    default_count = (
        len(DEFAULT_B0_MHZ)
        * DEFAULT_KEX_RANGE[2]
        * DEFAULT_PB_RANGE[2]
        * DEFAULT_DW_RANGE[2]
    )

    return {
        "b0_mhz_values": b0_values.tolist(),
        "j_is_hz": float(args.j_is),
        "k_ex_values": kex_values.tolist(),
        "p_b_values": pb_values.tolist(),
        "dw_n_ppm_values": dw_values.tolist(),
        "fixed_parameters": {**FIXED_GUI_DEFAULTS, "theta_N_deg": 22.0},
        "ncyc_grid": DEFAULT_NCYC_GRID.astype(int).tolist(),
        "chemex_ncyc_label_grid": CHEMEX_NCYC_LABEL.astype(int).tolist(),
        "base_schedule": [0] + DEFAULT_NCYC_GRID.astype(int).tolist(),
        "t_relax": DEFAULT_T_RELAX,
        "chemex_time_t2_required": DEFAULT_T_RELAX,
        "i0": float(args.i0),
        "relative_esd": float(args.relative_esd),
        "noise_enabled": not args.no_noise,
        "num_repeats": int(args.num_repeats),
        "seed": int(args.seed),
        "total_simulations": total_count,
        "default_total_simulations": default_count,
        "metadata_fieldnames": METADATA_FIELDNAMES,
        "output_layout": "Data/{B0_MHz:.0f}MHz/sim_{sim_id:06d}_1N-HN.out",
    }


def ensure_output_paths(out_dir: Path, overwrite: bool) -> dict[str, Path]:
    paths = {
        "metadata_csv": out_dir / "metadata.csv",
        "config_json": out_dir / "config.json",
        "data_dir": out_dir / "Data",
    }
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{out_dir} already exists and is not empty. Use --overwrite to replace files.")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths["data_dir"].mkdir(parents=True, exist_ok=True)
    return paths


def make_params(b0_mhz: float, k_ex: float, p_b: float, dw_n: float, j_is: float) -> dict[str, float]:
    return {
        "B0": b0_mhz_to_tesla(b0_mhz),
        "J_IS": j_is,
        "k_ex": k_ex,
        "p_B": p_b,
        "dw_N": dw_n,
        **FIXED_GUI_DEFAULTS,
    }


def align_r2(profile_r2: np.ndarray, skipped_ncyc: tuple[int, ...]) -> np.ndarray:
    aligned = np.full(len(DEFAULT_NCYC_GRID), np.nan, dtype=np.float64)
    skipped = set(skipped_ncyc)
    kept_indices = [idx for idx, ncyc in enumerate(DEFAULT_NCYC_GRID) if int(ncyc) not in skipped]
    if len(kept_indices) != len(profile_r2):
        raise ValueError(
            "Could not align simulator output to ncyc grid: "
            f"kept={len(kept_indices)} r2={len(profile_r2)}"
        )
    aligned[kept_indices] = profile_r2.astype(np.float64)
    return aligned


def choose_repeated_ncyc(rng: np.random.Generator, num_repeats: int) -> list[int]:
    if num_repeats == 0:
        return []
    replace = num_repeats > len(DEFAULT_NCYC_GRID)
    repeated = rng.choice(DEFAULT_NCYC_GRID, size=num_repeats, replace=replace)
    rng.shuffle(repeated)
    return [int(value) for value in repeated]


def noisy_intensity(
    clean_intensity: float,
    relative_esd: float,
    rng: np.random.Generator,
    noise_enabled: bool,
) -> tuple[float, float]:
    esd = float(relative_esd * clean_intensity)
    if noise_enabled and esd > 0.0:
        value = float(clean_intensity + rng.normal(0.0, esd))
    else:
        value = float(clean_intensity)
    return max(value, np.finfo(float).tiny), esd


def chemex_rows(
    r2_by_ncyc: dict[int, float],
    repeated_ncyc: list[int],
    i0: float,
    relative_esd: float,
    rng: np.random.Generator,
    noise_enabled: bool,
) -> list[tuple[int, float, float]]:
    rows: list[tuple[int, float, float]] = []
    ref_intensity, ref_esd = noisy_intensity(i0, relative_esd, rng, noise_enabled)
    rows.append((0, ref_intensity, ref_esd))

    for ncyc, label in zip(DEFAULT_NCYC_GRID, CHEMEX_NCYC_LABEL):
        r2_eff = r2_by_ncyc[int(ncyc)]
        clean_intensity = i0 * math.exp(-r2_eff * DEFAULT_T_RELAX)
        intensity, esd = noisy_intensity(clean_intensity, relative_esd, rng, noise_enabled)
        rows.append((int(label), intensity, esd))

    for ncyc in repeated_ncyc:
        label = int(ncyc) // 2
        r2_eff = r2_by_ncyc[int(ncyc)]
        clean_intensity = i0 * math.exp(-r2_eff * DEFAULT_T_RELAX)
        intensity, esd = noisy_intensity(clean_intensity, relative_esd, rng, noise_enabled)
        rows.append((label, intensity, esd))

    return rows


def write_chemex_file(path: Path, rows: list[tuple[int, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("#   ncyc_cp         Intensity      Esd(Int.)\n")
        for ncyc, intensity, esd in rows:
            handle.write(f"{float(ncyc):12.3e} {intensity:16.7e} {esd:14.7e}\n")


def metadata_row(
    sim_id: int,
    case_id: int,
    field_file: Path,
    b0_mhz: float,
    params: dict[str, float],
    args: argparse.Namespace,
    repeated_ncyc: list[int],
    valid_points: int,
    nonfinite_points: int,
    skipped_ncyc: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "sim_id": sim_id,
        "case_id": case_id,
        "field_file": field_file.as_posix(),
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
        "I0": args.i0,
        "relative_esd": args.relative_esd,
        "noise_enabled": not args.no_noise,
        "seed": args.seed,
        "repeated_ncyc": ";".join(str(value) for value in repeated_ncyc),
        "valid_points": valid_points,
        "nonfinite_points": nonfinite_points,
        "skipped_count": len(skipped_ncyc),
        "skipped_ncyc": ";".join(str(value) for value in skipped_ncyc),
    }


def validate_rows(rows: list[tuple[int, float, float]], num_repeats: int) -> None:
    expected_rows = 1 + len(DEFAULT_NCYC_GRID) + num_repeats
    if len(rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} ChemEx rows, got {len(rows)}")
    if rows[0][0] != 0:
        raise RuntimeError("First ChemEx row must be ncyc_cp = 0")
    values = np.asarray([[ncyc, intensity, esd] for ncyc, intensity, esd in rows], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("ChemEx rows contain non-finite values")
    if np.any(values[:, 1] <= 0.0):
        raise RuntimeError("ChemEx intensities must be positive")
    if np.any(values[:, 2] < 0.0):
        raise RuntimeError("ChemEx Esd values must be non-negative")


def main() -> None:
    args = parse_args()
    config = build_config(args)
    expected_count = int(config["total_simulations"])
    default_count = int(config["default_total_simulations"])
    if expected_count != default_count:
        print(f"Custom grid requested: generating {expected_count} simulations instead of default {default_count}.")

    paths = ensure_output_paths(args.out_dir, args.overwrite)
    with paths["config_json"].open("w") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")

    sim = CPMGSimulator()
    rng = np.random.default_rng(args.seed)
    metadata_rows: list[dict[str, Any]] = []
    profile_with_skips = 0
    profile_with_nonfinite = 0
    total_repeated_rows = 0

    b0_values = config["b0_mhz_values"]
    parameter_grid = list(
        itertools.product(
            config["k_ex_values"],
            config["p_b_values"],
            config["dw_n_ppm_values"],
        )
    )

    sim_id = 0
    for field_index, b0_mhz in enumerate(b0_values):
        for case_id, (k_ex, p_b, dw_n) in enumerate(parameter_grid):
            params = make_params(float(b0_mhz), float(k_ex), float(p_b), float(dw_n), float(args.j_is))
            profile = sim.simulate_cpmg(params, ncyc_range=DEFAULT_NCYC_GRID, t_relax=DEFAULT_T_RELAX)
            aligned_r2 = align_r2(profile.r2_eff, profile.skipped_ncyc)

            finite_mask = np.isfinite(aligned_r2)
            valid_points = int(np.count_nonzero(finite_mask))
            nonfinite_points = int(np.count_nonzero(~finite_mask))
            if profile.skipped_ncyc:
                profile_with_skips += 1
            if nonfinite_points:
                profile_with_nonfinite += 1

            if nonfinite_points:
                raise RuntimeError(
                    f"Simulation {sim_id} has {nonfinite_points} non-finite R2_eff values; "
                    "ChemEx intensity export requires finite profiles."
                )

            r2_by_ncyc = {
                int(ncyc): float(r2)
                for ncyc, r2 in zip(DEFAULT_NCYC_GRID, aligned_r2, strict=True)
            }
            repeated_ncyc = choose_repeated_ncyc(rng, args.num_repeats)
            total_repeated_rows += len(repeated_ncyc)
            rows = chemex_rows(
                r2_by_ncyc,
                repeated_ncyc,
                args.i0,
                args.relative_esd,
                rng,
                not args.no_noise,
            )
            validate_rows(rows, args.num_repeats)

            field_dir = paths["data_dir"] / f"{float(b0_mhz):.0f}MHz"
            field_file = field_dir / f"sim_{sim_id:06d}_1N-HN.out"
            write_chemex_file(field_file, rows)

            metadata_rows.append(
                metadata_row(
                    sim_id,
                    case_id + field_index * len(parameter_grid),
                    field_file.relative_to(args.out_dir),
                    float(b0_mhz),
                    params,
                    args,
                    repeated_ncyc,
                    valid_points,
                    nonfinite_points,
                    profile.skipped_ncyc,
                )
            )

            sim_id += 1
            if sim_id % args.progress_every == 0 or sim_id == expected_count:
                print(f"Wrote {sim_id}/{expected_count} ChemEx files")

    if sim_id != expected_count:
        raise RuntimeError(f"Wrote {sim_id} simulations, expected {expected_count}")
    if expected_count == default_count and expected_count != 3000:
        raise RuntimeError(f"Default ChemEx sweep should generate 3000 simulations, got {expected_count}")

    with paths["metadata_csv"].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDNAMES)
        writer.writeheader()
        writer.writerows(metadata_rows)

    if profile_with_skips:
        print(f"WARNING: {profile_with_skips} profiles skipped one or more ncyc points.")
    if profile_with_nonfinite:
        print(f"WARNING: {profile_with_nonfinite} profiles contained non-finite R2_eff values.")

    print("ChemEx sweep complete.")
    print(f"Total .out files written: {sim_id}")
    print(f"Total field simulations: {expected_count}")
    print(f"Total repeated rows: {total_repeated_rows}")
    print(f"Profiles with skipped points: {profile_with_skips}")
    print(f"Profiles with non-finite values: {profile_with_nonfinite}")
    print(f"Metadata CSV: {paths['metadata_csv']}")
    print(f"Config: {paths['config_json']}")
    print(f"Data directory: {paths['data_dir']}")


if __name__ == "__main__":
    main()
