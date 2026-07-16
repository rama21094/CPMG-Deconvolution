from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_trusted_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    """Load checkpoints written by this project under PyTorch 2.6+.

    PyTorch 2.6 changed torch.load's default to weights_only=True. Our checkpoints
    contain trusted local metadata dictionaries in addition to tensor weights, so
    evaluation/prediction must request the legacy full-checkpoint loader.
    """
    return torch.load(path, map_location=map_location, weights_only=False)
