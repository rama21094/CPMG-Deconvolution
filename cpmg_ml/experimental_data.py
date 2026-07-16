from __future__ import annotations

import csv
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass

from .generate_dataset import load_specs_from_json
from .sampling import sample_simulation_case
from .simulator import CPMGSimulator


EXPERIMENTAL_POINT_KEYS = ["nu_cpmg", "r2_measured", "esd", "r2_target"]


def candidate_ncyc(
    b1_hz: float,
    t_relax: float,
    min_nu_hz: float,
    max_nu_hz: float,
    max_candidate: int,
) -> np.ndarray:
    t180 = 1.0 / (2.0 * b1_hz)
    ncyc = np.arange(1, max_candidate + 1, dtype=np.int32)
    valid_time = ncyc * t180 < t_relax
    ncyc = ncyc[valid_time]
    tau_cp = (t_relax - ncyc * t180) / (2.0 * ncyc)
    nu = 1.0 / (4.0 * tau_cp)
    return ncyc[(nu >= min_nu_hz) & (nu <= max_nu_hz)]


def generate_experimental_profile(
    seed: int,
    specs_json: str,
    min_length: int,
    max_length: int,
    min_nu_hz: float,
    max_nu_hz: float,
    min_t_relax: float,
    max_t_relax: float,
    min_esd: float,
    max_esd: float,
    max_duplicate_fraction: float,
    max_ncyc_candidate: int,
    max_retries: int,
) -> dict[str, np.ndarray | float]:
    rng = np.random.default_rng(seed)
    specs = load_specs_from_json(specs_json)
    sim = CPMGSimulator()

    for _ in range(max_retries):
        params, metadata = sample_simulation_case(rng, specs)
        t_relax = float(rng.uniform(min_t_relax, max_t_relax))
        length = int(rng.integers(min_length, max_length + 1))
        max_duplicates = min(length - 1, int(math.floor(length * max_duplicate_fraction)))
        duplicate_count = int(rng.integers(0, max_duplicates + 1)) if max_duplicates > 0 else 0
        unique_count = length - duplicate_count

        candidates = candidate_ncyc(
            params["B1_N"], t_relax, min_nu_hz, max_nu_hz, max_ncyc_candidate
        )
        if len(candidates) < unique_count:
            continue

        selected = np.sort(rng.choice(candidates, size=unique_count, replace=False))
        with_j, no_j = sim.simulate_dej_pair(params, ncyc_range=selected, t_relax=t_relax)
        valid = (
            len(with_j.r2_eff) == unique_count
            and len(no_j.r2_eff) == unique_count
            and not with_j.skipped_ncyc
            and not no_j.skipped_ncyc
            and np.all(np.isfinite(with_j.r2_eff))
            and np.all(np.isfinite(no_j.r2_eff))
            and np.all(np.isfinite(with_j.nu_cp))
        )
        if not valid:
            continue

        source_indices = np.arange(unique_count)
        if duplicate_count:
            duplicate_indices = rng.choice(source_indices, size=duplicate_count, replace=True)
            source_indices = np.concatenate([source_indices, duplicate_indices])

        nu = with_j.nu_cp[source_indices]
        clean_with_j = with_j.r2_eff[source_indices]
        target_no_j = no_j.r2_eff[source_indices]
        esd = np.exp(rng.uniform(np.log(min_esd), np.log(max_esd), size=length))
        measured = clean_with_j + rng.normal(0.0, esd)

        order = np.argsort(nu, kind="stable")
        return {
            "nu_cpmg": nu[order].astype(np.float32),
            "r2_measured": measured[order].astype(np.float32),
            "esd": esd[order].astype(np.float32),
            "r2_target": target_no_j[order].astype(np.float32),
            "r2_clean_with_j": clean_with_j[order].astype(np.float32),
            "b0_mhz": np.float32(metadata[0]),
            "t_relax": np.float32(t_relax),
            "b1_hz": np.float32(params["B1_N"]),
            "j_is": np.float32(params["J_IS"]),
        }

    raise RuntimeError(f"Could not generate valid experimental profile after {max_retries} attempts")


def _generate_chunk(seeds: list[int], specs_json: str, options: dict) -> list[dict]:
    return [generate_experimental_profile(seed, specs_json, **options) for seed in seeds]


def generate_experimental_profiles(
    num_profiles: int,
    seed: int,
    specs_json: str,
    options: dict,
    workers: int,
    chunk_size: int,
    progress_label: str,
    progress_seconds: float,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, np.iinfo(np.int32).max, size=num_profiles, dtype=np.int64)
    chunks = [
        [int(value) for value in seeds[start : start + chunk_size]]
        for start in range(0, num_profiles, chunk_size)
    ]
    started = time.time()
    last_print = started
    profiles: list[dict] = []

    def progress(done_chunks: int, force: bool = False) -> None:
        nonlocal last_print
        now = time.time()
        if not force and now - last_print < progress_seconds:
            return
        rate = len(profiles) / max(now - started, 1.0e-9)
        print(
            f"{progress_label}: {len(profiles)}/{num_profiles} profiles "
            f"({done_chunks}/{len(chunks)} chunks), {rate:.1f} profiles/s, "
            f"elapsed={(now-started)/60.0:.1f} min",
            flush=True,
        )
        last_print = now

    if workers <= 1:
        for index, chunk in enumerate(chunks, start=1):
            profiles.extend(_generate_chunk(chunk, specs_json, options))
            progress(index)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_generate_chunk, chunk, specs_json, options) for chunk in chunks]
            for index, future in enumerate(as_completed(futures), start=1):
                profiles.extend(future.result())
                progress(index)

    progress(len(chunks), force=True)
    return profiles


def compute_experimental_normalization(profiles: list[dict]) -> dict:
    nu = np.concatenate([np.log1p(item["nu_cpmg"]) for item in profiles]).astype(np.float64)
    r2 = np.concatenate(
        [np.asarray(item["r2_measured"]) for item in profiles]
        + [np.asarray(item["r2_target"]) for item in profiles]
    ).astype(np.float64)
    log_esd = np.concatenate([np.log(item["esd"]) for item in profiles]).astype(np.float64)
    b0 = np.asarray([item["b0_mhz"] for item in profiles], dtype=np.float64)

    def stats(values: np.ndarray) -> tuple[float, float]:
        return float(np.mean(values)), float(max(np.std(values), 1.0e-12))

    nu_mean, nu_std = stats(nu)
    r2_mean, r2_std = stats(r2)
    esd_mean, esd_std = stats(log_esd)
    b0_mean, b0_std = stats(b0)
    return {
        "log_nu_mean": nu_mean,
        "log_nu_std": nu_std,
        "r2_mean": r2_mean,
        "r2_std": r2_std,
        "log_esd_mean": esd_mean,
        "log_esd_std": esd_std,
        "b0_mean": b0_mean,
        "b0_std": b0_std,
    }


class ExperimentalProfileDataset(Dataset):
    def __init__(self, profiles: list[dict], normalization: dict):
        if torch is None:
            raise ModuleNotFoundError("PyTorch is required for ExperimentalProfileDataset")
        self.profiles = profiles
        self.norm = normalization

    def __len__(self) -> int:
        return len(self.profiles)

    def __getitem__(self, index: int) -> dict:
        item = self.profiles[index]
        point_features = np.stack(
            [
                (np.log1p(item["nu_cpmg"]) - self.norm["log_nu_mean"]) / self.norm["log_nu_std"],
                (item["r2_measured"] - self.norm["r2_mean"]) / self.norm["r2_std"],
                (np.log(item["esd"]) - self.norm["log_esd_mean"]) / self.norm["log_esd_std"],
            ],
            axis=1,
        ).astype(np.float32)
        return {
            "points": torch.from_numpy(point_features),
            "b0": torch.tensor(
                [(float(item["b0_mhz"]) - self.norm["b0_mean"]) / self.norm["b0_std"]],
                dtype=torch.float32,
            ),
            "target": torch.from_numpy(
                ((item["r2_target"] - self.norm["r2_mean"]) / self.norm["r2_std"]).astype(np.float32)
            ),
            "nu_cpmg": torch.from_numpy(np.asarray(item["nu_cpmg"], dtype=np.float32)),
            "r2_measured": torch.from_numpy(np.asarray(item["r2_measured"], dtype=np.float32)),
            "esd": torch.from_numpy(np.asarray(item["esd"], dtype=np.float32)),
            "r2_target_raw": torch.from_numpy(np.asarray(item["r2_target"], dtype=np.float32)),
        }


def collate_experimental(batch: list[dict], max_length: int = 64) -> dict:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for collate_experimental")
    batch_size = len(batch)
    lengths = [min(len(item["target"]), max_length) for item in batch]
    points = torch.zeros(batch_size, max_length, 3, dtype=torch.float32)
    target = torch.zeros(batch_size, max_length, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    nu = torch.zeros(batch_size, max_length, dtype=torch.float32)
    measured = torch.zeros(batch_size, max_length, dtype=torch.float32)
    esd = torch.zeros(batch_size, max_length, dtype=torch.float32)
    target_raw = torch.zeros(batch_size, max_length, dtype=torch.float32)
    b0 = torch.stack([item["b0"] for item in batch])

    for i, (item, length) in enumerate(zip(batch, lengths)):
        points[i, :length] = item["points"][:length]
        target[i, :length] = item["target"][:length]
        mask[i, :length] = True
        nu[i, :length] = item["nu_cpmg"][:length]
        measured[i, :length] = item["r2_measured"][:length]
        esd[i, :length] = item["esd"][:length]
        target_raw[i, :length] = item["r2_target_raw"][:length]

    return {
        "points": points,
        "b0": b0,
        "target": target,
        "mask": mask,
        "nu_cpmg": nu,
        "r2_measured": measured,
        "esd": esd,
        "r2_target_raw": target_raw,
        "lengths": torch.tensor(lengths, dtype=torch.int64),
    }


def parse_spectrometer_profile(path: str | Path) -> dict[str, np.ndarray]:
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.replace(",", " ").split()
            if len(parts) < 3:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"No numeric nu_cpmg/R2/Esd rows found in {path}")
    values = np.asarray(rows, dtype=np.float32)
    order = np.argsort(values[:, 0], kind="stable")
    values = values[order]
    if np.any(values[:, 0] <= 0) or np.any(values[:, 2] <= 0):
        raise ValueError("nu_cpmg and Esd(R2) must be positive")
    return {
        "nu_cpmg": values[:, 0],
        "r2_measured": values[:, 1],
        "esd": values[:, 2],
    }


def write_experimental_prediction_csv(path: str | Path, rows: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(keys)
        for values in zip(*(rows[key] for key in keys)):
            writer.writerow([float(value) for value in values])
