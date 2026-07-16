# CPMG De-J-Coupling ML Pipeline

This package trains a supervised model that maps a simulated CPMG profile with normal
amide scalar coupling to the matched profile with `J_IS = 0`.

## Install

```bash
python -m pip install -r requirements-ml.txt
```

PyTorch is not installed in the current local Python environment, so training and
inference require the install step. Dataset generation and validation only require
NumPy, SciPy, and HDF5.

## Generate Smoke Data

```bash
python -m cpmg_ml.generate_dataset --split train --num-profiles 10000 --shard-size 1000 --seed 1
python -m cpmg_ml.generate_dataset --split val --num-profiles 2000 --shard-size 1000 --seed 2
python -m cpmg_ml.generate_dataset --split test --num-profiles 2000 --shard-size 1000 --seed 3
python -m cpmg_ml.validate_dataset --split train
```

By default, the generator uses the updated range CSV from:

```text
~/Downloads/CPMG Simulation - Base Variables - updated.csv
```

If that file moves, pass `--ranges-csv /path/to/file.csv`.

## Train

```bash
python -m cpmg_ml.train --data-dir data/cpmg_dej --out-dir runs/cpmg_dej
```

The best checkpoint is saved to:

```text
runs/cpmg_dej/best_model.pt
```

## Dynamic In-Memory Train

Edit `cpmg_ml/train_dynamic_config.py`, then run:

```bash
python -m cpmg_ml.train_dynamic
```

This trainer repeatedly generates fresh in-memory training data, keeps one fixed
validation set, uses parallel CPU simulation, applies a cosine-with-plateau LR
policy, and saves `best_model.pt` plus `last_model.pt` under the configured
`OUT_DIR`.

## Experimental-Ready Variable-Length Train

This separate pipeline trains on noisy, variable-length, spectrometer-style
profiles and requires only `nu_cpmg`, measured `R2`, `Esd(R2)`, and `B0` at
inference.

Edit `cpmg_ml/experimental_config.py`, then run:

```bash
python -m cpmg_ml.validate_experimental_pipeline --num-profiles 64 --workers 4
python -m cpmg_ml.train_experimental --device cuda:1 --generation-workers 32
```

Test the trained experimental model:

```bash
python -m cpmg_ml.test_experimental \
  --checkpoint runs/cpmg_experimental/best_model.pt \
  --out-dir runs/cpmg_experimental/test_report \
  --device cuda:1
```

Predict from a spectrometer text file:

```bash
python -m cpmg_ml.predict_experimental \
  --checkpoint runs/cpmg_experimental/best_model.pt \
  --input profile.txt \
  --b0-mhz 600 \
  --out-csv predicted_j0.csv \
  --out-plot predicted_j0.png \
  --device cuda:1
```

## Evaluate

```bash
python -m cpmg_ml.evaluate \
  --checkpoint runs/cpmg_dej/best_model.pt \
  --split test \
  --plot-dir runs/cpmg_dej/test_plots
```

## Comprehensive Test Report

```bash
python -m cpmg_ml.test_model \
  --checkpoint runs/cpmg_dej/best_model.pt \
  --data-dir data/cpmg_dej \
  --split test \
  --out-dir runs/cpmg_dej/test_report \
  --device cuda:1
```

This writes metrics, per-profile CSV files, and plots for overlays, residuals,
parity, profile-index error, identity-baseline comparison, and error versus
metadata.

## Predict One Generated Example

```bash
python -m cpmg_ml.predict \
  --checkpoint runs/cpmg_dej/best_model.pt \
  --input-h5 data/cpmg_dej/test_0000.h5 \
  --index 0 \
  --out-csv runs/cpmg_dej/example_prediction.csv
```
