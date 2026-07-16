from __future__ import annotations

import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from typing import Iterable

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass

from .generate_dataset import build_ncyc_grid, load_specs_from_json
from .sampling import METADATA_KEYS, RangeSpec, sample_simulation_case, specs_to_json
from .simulator import CPMGSimulator


def range_specs_to_json(specs: dict[str, RangeSpec]) -> str:
    return specs_to_json(specs)


def chunk_seeds(seeds: np.ndarray, chunk_size: int) -> list[list[int]]:
    return [
        [int(seed) for seed in seeds[start : start + chunk_size]]
        for start in range(0, len(seeds), chunk_size)
    ]


def _valid_profile(values: np.ndarray, expected_len: int) -> bool:
    return len(values) == expected_len and bool(np.all(np.isfinite(values)))


def _generate_chunk(
    seeds: list[int],
    specs_json: str,
    ncyc_grid: np.ndarray,
    t_relax: float,
    max_retries: int,
) -> dict[str, np.ndarray]:
    specs = load_specs_from_json(specs_json)
    sim = CPMGSimulator()
    expected_len = len(ncyc_grid)
    r2_with_j = []
    r2_no_j = []
    nu_cp = []
    metadata_rows = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        for _ in range(max_retries):
            params, metadata = sample_simulation_case(rng, specs)
            with_j, no_j = sim.simulate_dej_pair(params, ncyc_range=ncyc_grid, t_relax=t_relax)
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
                r2_with_j.append(with_j.r2_eff.astype(np.float32))
                r2_no_j.append(no_j.r2_eff.astype(np.float32))
                nu_cp.append(with_j.nu_cp.astype(np.float32))
                metadata_rows.append(metadata.astype(np.float32))
                break
        else:
            raise RuntimeError(f"Could not generate a valid profile after {max_retries} attempts")

    return {
        "r2_with_j": np.stack(r2_with_j).astype(np.float32),
        "r2_no_j": np.stack(r2_no_j).astype(np.float32),
        "nu_cp": np.stack(nu_cp).astype(np.float32),
        "metadata": np.stack(metadata_rows).astype(np.float32),
    }


def generate_in_memory_arrays(
    num_profiles: int,
    seed: int,
    specs_json: str,
    ncyc_grid: np.ndarray,
    t_relax: float,
    max_retries: int,
    workers: int,
    chunk_size: int,
    progress_label: str = "generation",
    progress_seconds: float = 10.0,
) -> dict[str, np.ndarray]:
    if num_profiles < 1:
        raise ValueError("num_profiles must be at least 1")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")

    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, np.iinfo(np.int32).max, size=num_profiles, dtype=np.int64)
    chunks = chunk_seeds(seeds, chunk_size)
    total_chunks = len(chunks)
    started_at = time.time()
    last_progress_at = started_at

    def print_progress(completed_chunks: int, completed_profiles: int, force: bool = False) -> None:
        nonlocal last_progress_at
        now = time.time()
        if not force and now - last_progress_at < progress_seconds:
            return
        elapsed = max(now - started_at, 1.0e-9)
        rate = completed_profiles / elapsed
        print(
            f"{progress_label}: {completed_profiles}/{num_profiles} profiles "
            f"({completed_chunks}/{total_chunks} chunks), {rate:.1f} profiles/s, "
            f"elapsed={elapsed/60.0:.1f} min",
            flush=True,
        )
        last_progress_at = now

    if workers <= 1:
        results = []
        completed_profiles = 0
        for completed_chunks, chunk in enumerate(chunks, start=1):
            result = _generate_chunk(chunk, specs_json, ncyc_grid, t_relax, max_retries)
            results.append(result)
            completed_profiles += len(chunk)
            print_progress(completed_chunks, completed_profiles)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_generate_chunk, chunk, specs_json, ncyc_grid, t_relax, max_retries)
                for chunk in chunks
            ]
            results = []
            completed_profiles = 0
            for completed_chunks, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                completed_profiles += int(result["r2_with_j"].shape[0])
                print_progress(completed_chunks, completed_profiles)

    print_progress(total_chunks, num_profiles, force=True)

    return {
        "r2_with_j": np.concatenate([item["r2_with_j"] for item in results], axis=0),
        "r2_no_j": np.concatenate([item["r2_no_j"] for item in results], axis=0),
        "nu_cp": np.concatenate([item["nu_cp"] for item in results], axis=0),
        "metadata": np.concatenate([item["metadata"] for item in results], axis=0),
    }


def compute_normalization_from_arrays(arrays: dict[str, np.ndarray]) -> dict:
    r2_with = arrays["r2_with_j"].astype(np.float64)
    r2_no = arrays["r2_no_j"].astype(np.float64)
    nu = np.log1p(arrays["nu_cp"].astype(np.float64))
    metadata = arrays["metadata"].astype(np.float64)

    both_r2 = np.concatenate([r2_with.reshape(-1), r2_no.reshape(-1)])
    r2_mean = float(np.mean(both_r2))
    r2_std = float(max(np.std(both_r2), 1.0e-12))
    log_nu_mean = float(np.mean(nu))
    log_nu_std = float(max(np.std(nu), 1.0e-12))
    meta_mean = np.mean(metadata, axis=0)
    meta_std = np.maximum(np.std(metadata, axis=0), 1.0e-12)

    return {
        "r2_mean": r2_mean,
        "r2_std": r2_std,
        "log_nu_mean": log_nu_mean,
        "log_nu_std": log_nu_std,
        "metadata_mean": meta_mean.astype(float).tolist(),
        "metadata_std": meta_std.astype(float).tolist(),
        "metadata_keys": list(METADATA_KEYS),
    }


class InMemoryCPMGDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], normalization: dict):
        if torch is None:
            raise ModuleNotFoundError("PyTorch is required for InMemoryCPMGDataset. Install requirements-ml.txt.")
        self.arrays = arrays
        self.normalization = normalization
        self.meta_mean = np.asarray(normalization["metadata_mean"], dtype=np.float32)
        self.meta_std = np.asarray(normalization["metadata_std"], dtype=np.float32)
        self.r2_mean = float(normalization["r2_mean"])
        self.r2_std = float(normalization["r2_std"])
        self.log_nu_mean = float(normalization["log_nu_mean"])
        self.log_nu_std = float(normalization["log_nu_std"])

    def __len__(self) -> int:
        return int(self.arrays["r2_with_j"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        r2_with = self.arrays["r2_with_j"][index]
        r2_no = self.arrays["r2_no_j"][index]
        nu_cp = self.arrays["nu_cp"][index]
        metadata = self.arrays["metadata"][index]

        r2_with_norm = (r2_with - self.r2_mean) / self.r2_std
        r2_no_norm = (r2_no - self.r2_mean) / self.r2_std
        log_nu_norm = (np.log1p(nu_cp) - self.log_nu_mean) / self.log_nu_std
        metadata_norm = (metadata - self.meta_mean) / self.meta_std

        return {
            "x": torch.from_numpy(np.stack([r2_with_norm, log_nu_norm]).astype(np.float32)),
            "metadata": torch.from_numpy(metadata_norm.astype(np.float32)),
            "target": torch.from_numpy(r2_no_norm.astype(np.float32)),
        }
