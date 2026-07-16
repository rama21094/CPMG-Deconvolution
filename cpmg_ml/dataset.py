from __future__ import annotations

import bisect
import json
import math
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass

from .sampling import METADATA_KEYS


def find_split_files(data_dir: str | Path, split: str) -> list[Path]:
    files = sorted(Path(data_dir).glob(f"{split}_*.h5"))
    if not files:
        raise FileNotFoundError(f"No HDF5 files found for split {split!r} in {data_dir}")
    return files


def compute_normalization(files: Iterable[str | Path]) -> dict:
    files = [Path(path) for path in files]
    if not files:
        raise ValueError("No files supplied for normalization")

    r2_sum = 0.0
    r2_sq_sum = 0.0
    r2_count = 0
    log_nu_sum = 0.0
    log_nu_sq_sum = 0.0
    log_nu_count = 0
    meta_sum = np.zeros(len(METADATA_KEYS), dtype=np.float64)
    meta_sq_sum = np.zeros(len(METADATA_KEYS), dtype=np.float64)
    meta_count = 0

    for path in files:
        with h5py.File(path, "r") as handle:
            r2_with = handle["r2_with_j"][:].astype(np.float64)
            r2_no = handle["r2_no_j"][:].astype(np.float64)
            nu = np.log1p(handle["nu_cp"][:].astype(np.float64))
            metadata = handle["metadata"][:].astype(np.float64)

        both_r2 = np.concatenate([r2_with.reshape(-1), r2_no.reshape(-1)])
        r2_sum += float(np.sum(both_r2))
        r2_sq_sum += float(np.sum(both_r2**2))
        r2_count += int(both_r2.size)

        log_nu_sum += float(np.sum(nu))
        log_nu_sq_sum += float(np.sum(nu**2))
        log_nu_count += int(nu.size)

        meta_sum += np.sum(metadata, axis=0)
        meta_sq_sum += np.sum(metadata**2, axis=0)
        meta_count += int(metadata.shape[0])

    r2_mean = r2_sum / r2_count
    r2_std = math.sqrt(max(r2_sq_sum / r2_count - r2_mean**2, 1.0e-12))
    log_nu_mean = log_nu_sum / log_nu_count
    log_nu_std = math.sqrt(max(log_nu_sq_sum / log_nu_count - log_nu_mean**2, 1.0e-12))
    meta_mean = meta_sum / meta_count
    meta_std = np.sqrt(np.maximum(meta_sq_sum / meta_count - meta_mean**2, 1.0e-12))

    return {
        "r2_mean": float(r2_mean),
        "r2_std": float(r2_std),
        "log_nu_mean": float(log_nu_mean),
        "log_nu_std": float(log_nu_std),
        "metadata_mean": meta_mean.astype(float).tolist(),
        "metadata_std": meta_std.astype(float).tolist(),
        "metadata_keys": list(METADATA_KEYS),
    }


def save_normalization(path: str | Path, normalization: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(normalization, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_normalization(path: str | Path) -> dict:
    with Path(path).open() as handle:
        return json.load(handle)


class H5CPMGDataset(Dataset):
    def __init__(
        self,
        files: Iterable[str | Path],
        normalization: dict,
        include_raw: bool = False,
    ):
        if torch is None:
            raise ModuleNotFoundError("PyTorch is required for H5CPMGDataset. Install requirements-ml.txt.")
        self.files = [Path(path) for path in files]
        self.normalization = normalization
        self.include_raw = include_raw
        self.lengths = []
        for path in self.files:
            with h5py.File(path, "r") as handle:
                self.lengths.append(int(handle["r2_with_j"].shape[0]))
        self.cumulative = np.cumsum(self.lengths).tolist()
        self._handles: dict[int, h5py.File] = {}

        keys = normalization.get("metadata_keys", METADATA_KEYS)
        if list(keys) != list(METADATA_KEYS):
            raise ValueError("Normalization metadata keys do not match dataset metadata keys")

        self.meta_mean = np.asarray(normalization["metadata_mean"], dtype=np.float32)
        self.meta_std = np.asarray(normalization["metadata_std"], dtype=np.float32)
        self.r2_mean = float(normalization["r2_mean"])
        self.r2_std = float(normalization["r2_std"])
        self.log_nu_mean = float(normalization["log_nu_mean"])
        self.log_nu_std = float(normalization["log_nu_std"])

    def __len__(self) -> int:
        return int(self.cumulative[-1])

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        file_idx = bisect.bisect_right(self.cumulative, index)
        previous = self.cumulative[file_idx - 1] if file_idx > 0 else 0
        return file_idx, index - previous

    def _handle(self, file_idx: int) -> h5py.File:
        if file_idx not in self._handles:
            self._handles[file_idx] = h5py.File(self.files[file_idx], "r")
        return self._handles[file_idx]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        file_idx, local_idx = self._locate(index)
        handle = self._handle(file_idx)

        r2_with = handle["r2_with_j"][local_idx].astype(np.float32)
        r2_no = handle["r2_no_j"][local_idx].astype(np.float32)
        nu_cp = handle["nu_cp"][local_idx].astype(np.float32)
        metadata = handle["metadata"][local_idx].astype(np.float32)

        r2_with_norm = (r2_with - self.r2_mean) / self.r2_std
        r2_no_norm = (r2_no - self.r2_mean) / self.r2_std
        log_nu_norm = (np.log1p(nu_cp) - self.log_nu_mean) / self.log_nu_std
        metadata_norm = (metadata - self.meta_mean) / self.meta_std

        sample = {
            "x": torch.from_numpy(np.stack([r2_with_norm, log_nu_norm]).astype(np.float32)),
            "metadata": torch.from_numpy(metadata_norm.astype(np.float32)),
            "target": torch.from_numpy(r2_no_norm.astype(np.float32)),
        }
        if self.include_raw:
            sample.update(
                {
                    "r2_with_j": torch.from_numpy(r2_with),
                    "r2_no_j": torch.from_numpy(r2_no),
                    "nu_cp": torch.from_numpy(nu_cp),
                    "metadata_raw": torch.from_numpy(metadata),
                }
            )
        return sample

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
