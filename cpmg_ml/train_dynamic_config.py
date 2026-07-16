"""Editable config for the dynamic in-memory CPMG trainer.

Run from an IDE by opening cpmg_ml/train_dynamic.py and pressing Run, or from a
terminal with:

    python -m cpmg_ml.train_dynamic
"""

from pathlib import Path


# Data generation
TRAIN_PROFILES_PER_BLOCK = 10_000
EPOCHS_PER_BLOCK = 20
MAX_EPOCHS = 2000
VAL_PROFILES = 2_000
NORMALIZATION_PROFILES = 10_000
GENERATION_WORKERS = 16
GENERATION_CHUNK_SIZE = 250
GENERATION_PROGRESS_SECONDS = 10.0
MAX_RETRIES = 100
RANGES_CSV = Path.home() / "Downloads" / "CPMG Simulation - Base Variables - updated.csv"

# CPMG grid
NCYC_START = 1
NCYC_STOP = 80
NCYC_STEP = 2
T_RELAX = 0.04

# Training
OUT_DIR = Path("runs/cpmg_dej_dynamic")
DEVICE = "cuda:1"
BATCH_SIZE = 512
NUM_WORKERS = 0
LR = 1.0e-3
MIN_LR = 1.0e-6
WEIGHT_DECAY = 1.0e-4
SLOPE_WEIGHT = 0.1
GRAD_CLIP_NORM = 1.0
USE_AMP = True
USE_TORCH_COMPILE = False

# Scheduler / stopping
LR_PLATEAU_PATIENCE = 40
LR_PLATEAU_FACTOR = 0.5
EARLY_STOPPING_PATIENCE = 60

# Model
HIDDEN_CHANNELS = 64
METADATA_HIDDEN = 128
NUM_BLOCKS = 6
KERNEL_SIZE = 3

# Seeds
BASE_SEED = 12345
NORMALIZATION_SEED = 1001
VALIDATION_SEED = 2001
TRAIN_SEED = 3001

# Checkpointing
RESUME = True
SAVE_EVERY_EPOCH = True
