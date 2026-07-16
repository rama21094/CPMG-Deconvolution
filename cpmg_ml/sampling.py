from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .simulator import b0_mhz_to_tesla


METADATA_KEYS = [
    "B0_MHz",
    "B1_N",
    "tau_m",
    "tau_e",
    "S2",
    "r_IS",
    "r_eff",
    "csa_N_ppm",
    "theta_N_deg",
    "J_IS",
    "k_ex",
    "p_B",
    "dw_N",
]


@dataclass(frozen=True)
class RangeSpec:
    low: float
    high: float
    distribution: str = "uniform"

    def sample(self, rng: np.random.Generator) -> float:
        if self.distribution == "log_uniform":
            return float(math.exp(rng.uniform(math.log(self.low), math.log(self.high))))
        if self.distribution == "uniform":
            return float(rng.uniform(self.low, self.high))
        raise ValueError(f"Unknown distribution: {self.distribution}")


DEFAULT_RANGE_SPECS: dict[str, RangeSpec] = {
    "tau_m": RangeSpec(0.5e-9, 50.0e-9, "log_uniform"),
    "tau_e": RangeSpec(1.0e-12, 750.0e-12, "log_uniform"),
    "S2": RangeSpec(0.01, 1.0, "uniform"),
    "csa_N_ppm": RangeSpec(-190.0, -130.0, "uniform"),
    "theta_N_deg": RangeSpec(15.0, 25.0, "uniform"),
    "r_IS": RangeSpec(1.0e-10, 1.1e-10, "uniform"),
    "r_eff": RangeSpec(1.8e-10, 2.5e-10, "uniform"),
    "J_IS": RangeSpec(85.0, 105.0, "uniform"),
    "B0_MHz": RangeSpec(500.0, 1200.0, "uniform"),
    "B1_N": RangeSpec(2000.0, 10000.0, "log_uniform"),
    "k_ex": RangeSpec(50.0, 4000.0, "log_uniform"),
    "p_B": RangeSpec(0.005, 0.15, "uniform"),
    "dw_N": RangeSpec(0.5, 10.0, "uniform"),
}


CSV_VARIABLE_MAP = {
    "Overall Tumbling": ("tau_m", 1.0e-9, "log_uniform"),
    "Internal Motion": ("tau_e", 1.0e-12, "log_uniform"),
    "Order Parameter": ("S2", 1.0, "uniform"),
    "15N CSA": ("csa_N_ppm", 1.0, "uniform"),
    "CSA-DD Angle": ("theta_N_deg", 1.0, "uniform"),
    "N-H Distance": ("r_IS", 1.0e-10, "uniform"),
    "Bath Proton Dist.": ("r_eff", 1.0e-10, "uniform"),
    "Scalar Coupling": ("J_IS", 1.0, "uniform"),
    "RF Field Strength": ("B1_N", 1.0, "log_uniform"),
    "Exchange Rate": ("k_ex", 1.0, "log_uniform"),
    "Minor Population": ("p_B", 0.01, "uniform"),
    "Chem. Shift Diff.": ("dw_N", 1.0, "uniform"),
}


def default_range_csv() -> Path:
    return Path.home() / "Downloads" / "CPMG Simulation - Base Variables - updated.csv"


def _parse_two_numbers(text: str) -> tuple[float, float]:
    text = re.sub(r"(?i)\bs\s*-\s*1\b", " ", text)
    text = re.sub(r"(?i)\bs\s*\^\s*-1\b", " ", text)
    values = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if len(values) < 2:
        raise ValueError(f"Could not parse range from: {text!r}")
    first, second = float(values[0]), float(values[1])
    return min(first, second), max(first, second)


def load_range_specs(csv_path: str | Path | None = None) -> dict[str, RangeSpec]:
    specs = dict(DEFAULT_RANGE_SPECS)
    if csv_path is None:
        csv_path = default_range_csv()
        if not Path(csv_path).exists():
            return specs

    path = Path(csv_path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            variable = (row.get("Variable") or "").strip()
            typical_range = (row.get("Typical Range") or "").strip()
            if not variable or not typical_range:
                continue

            if variable == "Static Field":
                low_t, high_t = _parse_two_numbers(typical_range)
                gamma_h = 267.522e6
                low_mhz = low_t * gamma_h / (2.0 * np.pi) / 1.0e6
                high_mhz = high_t * gamma_h / (2.0 * np.pi) / 1.0e6
                specs["B0_MHz"] = RangeSpec(min(low_mhz, high_mhz), max(low_mhz, high_mhz), "uniform")
                continue

            if variable not in CSV_VARIABLE_MAP:
                continue

            key, scale, distribution = CSV_VARIABLE_MAP[variable]
            low, high = _parse_two_numbers(typical_range)
            specs[key] = RangeSpec(low * scale, high * scale, distribution)

    return specs


def specs_to_json(specs: Mapping[str, RangeSpec]) -> str:
    return json.dumps({key: asdict(value) for key, value in specs.items()}, indent=2, sort_keys=True)


def metadata_vector(metadata: Mapping[str, float]) -> np.ndarray:
    return np.asarray([metadata[key] for key in METADATA_KEYS], dtype=np.float32)


def sample_metadata(rng: np.random.Generator, specs: Mapping[str, RangeSpec]) -> dict[str, float]:
    return {key: specs[key].sample(rng) for key in METADATA_KEYS}


def metadata_to_sim_params(metadata: Mapping[str, float]) -> dict[str, float]:
    return {
        "B0": b0_mhz_to_tesla(metadata["B0_MHz"]),
        "B1_N": metadata["B1_N"],
        "tau_m": metadata["tau_m"],
        "tau_e": metadata["tau_e"],
        "S2": metadata["S2"],
        "r_IS": metadata["r_IS"],
        "r_eff": metadata["r_eff"],
        "csa_N": metadata["csa_N_ppm"] * 1.0e-6,
        "theta_N": math.radians(metadata["theta_N_deg"]),
        "J_IS": metadata["J_IS"],
        "k_ex": metadata["k_ex"],
        "p_B": metadata["p_B"],
        "dw_N": metadata["dw_N"],
    }


def sample_simulation_case(
    rng: np.random.Generator,
    specs: Mapping[str, RangeSpec],
) -> tuple[dict[str, float], np.ndarray]:
    metadata = sample_metadata(rng, specs)
    return metadata_to_sim_params(metadata), metadata_vector(metadata)
