from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from . import experimental_config as cfg
from .experimental_data import (
    ExperimentalProfileDataset,
    collate_experimental,
    compute_experimental_normalization,
    generate_experimental_profiles,
    parse_spectrometer_profile,
)
from .sampling import load_range_specs, specs_to_json


def generation_options() -> dict:
    return {
        "min_length": cfg.MIN_PROFILE_LENGTH,
        "max_length": cfg.MAX_PROFILE_LENGTH,
        "min_nu_hz": cfg.MIN_NU_CPMG_HZ,
        "max_nu_hz": cfg.MAX_NU_CPMG_HZ,
        "min_t_relax": cfg.MIN_T_RELAX,
        "max_t_relax": cfg.MAX_T_RELAX,
        "min_esd": cfg.MIN_ESD,
        "max_esd": cfg.MAX_ESD,
        "max_duplicate_fraction": cfg.MAX_DUPLICATE_FRACTION,
        "max_ncyc_candidate": cfg.MAX_NCYC_CANDIDATE,
        "max_retries": cfg.MAX_RETRIES,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the experimental-ready pipeline.")
    parser.add_argument("--num-profiles", type=int, default=64)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = generate_experimental_profiles(
        args.num_profiles,
        seed=777,
        specs_json=specs_to_json(load_range_specs()),
        options=generation_options(),
        workers=args.workers,
        chunk_size=8,
        progress_label="validation",
        progress_seconds=2.0,
    )
    lengths = np.asarray([len(item["nu_cpmg"]) for item in profiles])
    if np.min(lengths) < 8 or np.max(lengths) > 64:
        raise ValueError(f"Unexpected lengths: {np.min(lengths)}-{np.max(lengths)}")
    if not all(np.all(np.diff(item["nu_cpmg"]) >= 0) for item in profiles):
        raise ValueError("Generated frequencies are not sorted")
    if not all(np.all(item["esd"] > 0) for item in profiles):
        raise ValueError("Generated Esd values must be positive")

    normalization = compute_experimental_normalization(profiles)
    if not all(np.isfinite(value) for value in normalization.values()):
        raise ValueError("Normalization contains non-finite values")

    sample = """#nu_cpmg(Hz) R2(1/s) Esd(R2)
1000 14.7 0.2
66.667 21.0 0.3
66.667 21.2 0.3
"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "profile.txt"
        path.write_text(sample)
        parsed = parse_spectrometer_profile(path)
    if not np.allclose(parsed["nu_cpmg"], [66.667, 66.667, 1000.0]):
        raise ValueError("Parser did not sort and preserve duplicate frequencies")

    duplicate_profiles = sum(len(np.unique(item["nu_cpmg"])) < len(item["nu_cpmg"]) for item in profiles)

    try:
        import torch

        from .experimental_model import ExperimentalCPMGTransformer

        dataset = ExperimentalProfileDataset(profiles, normalization)
        batch = collate_experimental([dataset[i] for i in range(len(dataset))], max_length=64)
        model = ExperimentalCPMGTransformer(
            model_dim=32, num_heads=4, num_layers=2, ff_dim=64, dropout=0.0
        ).eval()
        with torch.no_grad():
            mean_a, _ = model(batch["points"], batch["b0"], batch["mask"])
            changed = batch["points"].clone()
            changed[~batch["mask"]] = 1000.0
            mean_b, _ = model(changed, batch["b0"], batch["mask"])
        if not torch.allclose(mean_a[batch["mask"]], mean_b[batch["mask"]], atol=1.0e-5):
            raise ValueError("Padding mask invariance check failed")
        print("PyTorch padding-mask invariance check passed")
    except ModuleNotFoundError:
        print("PyTorch unavailable; skipped padding-mask invariance check")

    print(
        f"Experimental pipeline validation passed: profiles={len(profiles)}, "
        f"length_range={lengths.min()}-{lengths.max()}, duplicate_profiles={duplicate_profiles}"
    )


if __name__ == "__main__":
    main()
