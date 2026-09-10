# Validation progress and next experiments — 9 September 2026

## Outcome

The clean CNN result survives removal of exact training/test duplicates. A first downstream ChemEx pilot shows improved exchange-parameter estimates after correction, but still appreciable residual bias. This is preliminary evidence from one physical exchange group, not a population-level validation.

Existing datasets, checkpoints, and historical result files were preserved. No retraining was performed. New results are retrospective: the old validation set influenced checkpoint selection, and the old test set has already been examined during model development.

## Dataset integrity

`benchmark_audit_500k.json` records SHA-256 hashes for each original shard and exact duplicate exclusions. Equality requires identical float32 metadata, frequency axes, coupled curves, and J=0 targets.

- Training: 500,000 rows, including 59 duplicate occurrences.
- Validation: 20,000 rows, including 4 exact matches to training.
- Test: 20,000 rows, including 7 exact matches to training.
- No nonfinite rows were found.
- Revised evaluation excludes duplicate validation/test rows, leaving 19,993 test profiles. Existing checkpoints retain their original training and selection history.

The generator now supports reserved consecutive seed ranges (`--seed-start`), explicit JSON ranges (`--ranges-json`), and stores per-example simulation seeds. It refuses to overwrite an existing split. The historical random-seed mode remains available for compatibility. Future splits must use disjoint reserved ranges and undergo a cross-split audit.

## Curve evaluation

Clean full-test CNN RMSE after exclusions: **0.053236 s^-1**. Unchanged-input RMSE: **0.819076 s^-1**.

Noisy evaluation used the existing SNR-100 CNN and three independent new noise draws. The table reports mean RMSE and sample standard deviation across these three draws, NOT uncertainty across training seeds or a confidence interval.

| Evaluation | Unchanged input | CNN | Savitzky–Golay control |
|---|---:|---:|---:|
| All points | 25.139 ± 0.089 | 3.369 ± 0.034 | 12.696 ± 0.111 |
| Expected intensity >=0.05 I0 | 1.304 ± 0.002 | 0.459 ± 0.002 | 2.409 ± 0.035 |

All values are in s^-1. The second row retains 724,761 points across 19,663 profiles. At reference SNR 100, its threshold corresponds to expected point-intensity SNR >=5. This is a retrospective truth-based mask, not an implementable experimental quality rule or a universal measurability threshold.

Approximately 1.53–1.55% of noisy intensities were nonpositive before the historical clipping step. Therefore the all-point RMSE is strongly influenced by censoring and cannot be interpreted solely as J correction performance. The clipping policy was retained in this comparison to preserve compatibility with existing checkpoints.

The smoothing window (39 points, polynomial order 2) was selected on the nonduplicate noisy validation set using full-point MSE against the clean J=0 target. This simple control oversmooths; it is not a comprehensive comparison against tuned, uncertainty-aware denoisers. Its clean-data result is not a separately tuned clean benchmark. A dedicated coupled-target denoiser remains to be trained.

## Downstream fitting pilot

One group shared k_ex=600 s^-1 and p_B=0.05, with three residue shifts (1,3,5 ppm), each simulated at 600 and 800 MHz. T=40 ms; 40 frequencies; ChemEx-matched sequence; J=92 Hz in coupled curves; B1=5555 Hz. Remaining simulation settings are explicit in `fit_validation_pilot.py`.

ChemEx fitted PB, KEX_AB, residue shifts and field/residue-specific R2. R1 was fixed to the simulator value; A/B relaxation rates were constrained equal. Thus this is an optimistic controlled test. Two initialisations (PB=.02/KEX=300 and PB=.10/KEX=1200) agreed to reported precision. Corrected curves were converted to pseudo-intensities with fixed fitting weights; these are not calibrated DNN uncertainties. The noisy J=0 and coupled arms use common random numbers for a paired comparison.

| Arm | Fitted k_ex (s^-1) | Fitted p_B | Fitted shifts (ppm) |
|---|---:|---:|---|
| Truth | 600 | 0.050000 | 1, 3, 5 |
| Clean J=0 | 599.830 | 0.050114 | 0.999689, 2.999620, 5.000210 |
| Noisy J=0 | 602.830 | 0.050165 | 0.989019, 3.017450, 5.005900 |
| Noisy coupled | 502.585 | 0.060617 | 1.029110, 2.933070, 4.898030 |
| CNN corrected | 563.597 | 0.054125 | 0.995144, 2.971540, 4.940240 |

Correction reduces signed k_ex bias from -16.2% to -6.1%, and p_B bias from +21.2% to +8.3%. The clean reference closely recovers truth, supporting this fitting setup for this particular system. Broader groups, repeated noise, realistic nuisance-parameter uncertainty, and a denoising control are required before a general claim.

Results: `ml_results/downstream_fit_pilot.json`. Figure: `group_meeting_figs/downstream_fit_pilot_20260909.png`. Input configurations and logs are in the work directory recorded in the JSON; the checked-in script can regenerate them in a new directory.

## Corrections to interpretation

The old 26.3% visibility statistic is a single-noise-draw heuristic, not a recovery ceiling. `recoverability.py` now says so. Historical slides and Word reports have not yet been rewritten; their stronger ceiling/acquisition claims should not be reused as established conclusions.

For a shared reference, first-order R2 covariance is

Cov(R_i,R_j) = (sigma/T)^2 [delta_ij exp(2 R_i T) + 1].

The off-diagonal term comes from the common reference. A new module implements this covariance and an oracle separation statistic sqrt(d^T Cov^-1 d). Monte Carlo tests at high intensity SNR validate the covariance; an analytical repeated-point test checks the common-reference contribution.

A revised 100-profile, 15-acquisition pilot retained 80 profiles valid across all configurations. It excludes points with expected intensity SNR <5 and records all per-profile statistics and failures. At T=40 ms/40 points, the median oracle statistic is 1.495 and 27.5% exceed 3. These are descriptive pilot values, not calibrated detection probabilities or inverse-problem recovery rates. Different seeds, covariance treatment and masks mean these cannot isolate the effect of the covariance correction relative to the historical sweep. No equal-time recommendation is made.

The existing ChemEx matrix comparison now has an executable assertion and portable project path. It still gives maximum element difference zero, **with shared supplied rates and identity/source components excluded**. This verifies matrix assembly, not independent relaxation-rate formulas.

## Next computational stages

1. Specify a fresh confirmatory dataset and disjoint seed ranges; retain broad parameter ranges pending PI discussion. Freeze the evaluation protocol before examining outcomes. Keep OOD/acquisition tests separate from ordinary random splits.
2. Extend the four-arm downstream pilot over exchange regimes and repeated noise draws. Include clean-reference failures, optimiser failures, signed/absolute parameter errors and paired improvements. Do not treat multiple starting points as independent samples.
3. Implement and validate an intensity/mask-aware noise pipeline, then train a coupled-target denoiser and revised correction model. Existing checkpoints do not automatically remain valid after changing preprocessing.
4. Repeat finalist training across at least three seeds with documented compute budgets; include J=0/no-exchange controls and uncertainty in supplied B1/J.
5. Independently verify relaxation-rate formulas and equilibrium/source behaviour against the source paper and limiting cases.
6. Consolidate inference around the tested checkpoint format and document supported sequence/grid/settings.

## Inputs needed for experimental validation

- Exact experimental pulse programme and acquisition settings (including frequencies, T, field strengths, pulse widths and reference acquisition).
- Available repeated coupled measurements and any independently acquired artefact-suppressed/reference measurements.
- PI-agreed target experimental parameter range, when ready; no narrowing has been applied here.

## Reproduction

Run from the project root with the vyoma Python environment, except the ChemEx structural comparison, which uses NMR:

```sh
python -m unittest discover -s tests -p 'test_*.py'
python -m cpmg_ml.benchmark_audit --data-dir data/cpmg_dej_chemex_500k --out docs/ml_results/benchmark_audit_500k.json
python -m cpmg_ml.evaluate_audited
python -m cpmg_ml.acquisition_covariance --n-profiles 100 --workers 4
python -m cpmg_ml.fit_validation_pilot --work-dir /private/tmp/cpmg-new-fit-pilot
python -m cpmg_ml.plot_validation_pilot
```

The fitting work directory must not already exist. The default ChemEx executable is the local NMR environment and can be overridden. Evaluation requires the original local shards and checkpoints. New evaluations write separate JSON files; historical results are retained.
