# DNN CPMG J-Coupling Deconvolution Pipeline

## Summary
- Build a supervised profile-to-profile DNN that maps a simulated CPMG profile with normal `J_IS` coupling to the matching profile with `J_IS = 0`.
- Use paired simulations: same sampled physical parameters, same `ncyc` grid, same `nu_cp` axis, only `J_IS` changes between input and target.
- Use a custom PyTorch conditional 1D residual ConvNet instead of a pretrained Hugging Face model, because this is a low-dimensional physics-regression problem rather than text/image/time-series forecasting.
- Create a staged pipeline: smoke dataset first, then 100k-1M+ profiles.

## Key Implementation
- Extract the physics simulator from `32x32_Sim_5thJune.py` into a non-GUI module so dataset generation does not import Tkinter/Matplotlib.
- Add scripts:
  - `cpmg_ml/generate_dataset.py`: sample variables, simulate paired profiles, write HDF5 shards.
  - `cpmg_ml/model.py`: define the conditional de-J-coupling network.
  - `cpmg_ml/train.py`: train/validate the DNN.
  - `cpmg_ml/evaluate.py`: report metrics and plot predicted vs target profiles.
  - `cpmg_ml/predict.py`: load a trained model and deconvolve a new profile.
- Add `requirements-ml.txt`; PyTorch must be installed because local Python currently has NumPy/SciPy/Pandas/HDF5 but not `torch`.

## Dataset Specification
- One training example is one full CPMG profile, not one point.
- Fixed CPMG grid:
  - `T_relax = 0.04 s`
  - `ncyc_grid = np.arange(1, 80, 2)` giving 40 profile points.
  - This avoids finite-pulse skipped points even at minimum `B1 = 2000 Hz`.
- For each sampled parameter set:
  - Input profile: simulate with sampled `J_IS` from 85-105 Hz.
  - Target profile: simulate again with `J_IS = 0 Hz`.
  - Store `R2_eff_with_J`, `R2_eff_no_J`, `nu_cp`, and metadata.
- Store data as HDF5 shards, for example `data/cpmg_dej/train_0000.h5`, with float32 arrays.

## Sampling Ranges
Use the updated CSV as the source of truth:
- `tau_m`: log-uniform, `0.5 ns` to `50 ns`
- `tau_e`: log-uniform, `1 ps` to `750 ps`
- `S2`: uniform, `0.01` to `1.0`
- `csa_N`: uniform, `-190 ppm` to `-130 ppm`
- `theta_N`: uniform, `15 deg` to `25 deg`
- `r_IS`: uniform, `1.0 A` to `1.1 A`
- `r_eff`: uniform, `1.8 A` to `2.5 A`
- `J_IS`: uniform, `85 Hz` to `105 Hz`
- `B0`: uniform as proton frequency, `500 MHz` to `1200 MHz`, converted internally to Tesla
- `B1`: log-uniform or uniform, default log-uniform, `2000 Hz` to `10000 Hz`
- `k_ex`: log-uniform, `50 s^-1` to `4000 s^-1`
- `p_B`: uniform, `0.005` to `0.15`
- `dw_N`: uniform, `0.5 ppm` to `10 ppm`

## Model Design
- Use a conditional residual 1D ConvNet: `ConditionalCPMGDeJNet`.
- Inputs:
  - Channel 1: normalized `R2_eff_with_J`
  - Channel 2: normalized/log-scaled `nu_cp`
  - Metadata vector: normalized physical variables including `B0`, `B1`, `tau_m`, `tau_e`, `S2`, `r_IS`, `r_eff`, `csa_N`, `theta_N`, `J_IS`, `k_ex`, `p_B`, `dw_N`
- Output:
  - predicted `R2_eff_no_J` on the same `nu_cp` grid.
- Architecture:
  - metadata MLP encoder
  - 1D convolution stem
  - 6 residual Conv1D blocks with FiLM conditioning from metadata
  - 1D convolution output head
  - residual prediction: learn correction from `with_J` profile to `no_J` profile
- Loss:
  - profile MSE
  - plus small slope/shape loss using first differences along `nu_cp`
- Metrics:
  - RMSE in `s^-1`
  - MAE in `s^-1`
  - max absolute error
  - improvement over identity baseline, where identity means predicting `with_J` as `no_J`.

## Training Plan
- Stage 1 smoke run:
  - `10k` train profiles
  - `2k` validation profiles
  - `2k` test profiles
- Stage 2:
  - `100k` train profiles
  - `20k` validation profiles
- Stage 3:
  - `1M+` train profiles using HDF5 sharding and multiprocessing generation.
- Training defaults:
  - AdamW optimizer
  - learning rate `1e-3`
  - batch size `512`
  - max `200` epochs
  - early stopping after `20` validation epochs without improvement
  - automatic device selection: CUDA, Apple MPS, then CPU

## Test Plan
- Verify sampled values always fall inside the updated CSV ranges.
- Verify `600 MHz` converts to about `14.09 T`.
- Verify paired examples differ only by `J_IS`.
- Verify generated profiles have fixed length 40 and contain finite values.
- Train on a tiny dataset and confirm the model can overfit 512 examples.
- Validate that the trained model beats the identity baseline on held-out data.
- Save example plots comparing:
  - input with-J profile
  - true no-J target
  - predicted no-J profile

## Assumptions
- The first DNN will use profile + `nu_cp` axis + metadata, as previously chosen.
- The decoupled target is defined mathematically as the same simulator with `J_IS = 0`.
- `dw_N` is sampled as a positive magnitude from `0.5 ppm` to `10 ppm`.
- `T_relax` remains fixed at `40 ms` for v1.
- The GUI remains separate from the ML pipeline for now; integration can come after the model is trained and validated.
