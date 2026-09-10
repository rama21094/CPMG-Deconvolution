# PI meeting — 11 September 2026

The six-slide update incorporates the unpresented 3 September material and the new validation. The August decks provide historical context; their stronger claims of complete physics validation, a hard recoverability ceiling and no benefit from acquisition redesign should not be reused.

## Main finding to present

The CNN learns the synthetic correction accurately. The downstream pilot now spans kex=300,600,1800 s^-1 with two SNR-100 noise draws each, fixed pB=.05, three shift magnitudes and two fields. All 48 ChemEx runs returned fitted results. The two initialisations agree closely on global parameters.

| Mean absolute relative error | Noisy coupled | CNN corrected | Noisy J=0 reference |
|---|---:|---:|---:|
| kex | 18.47% | 11.04% | 1.66% |
| pB | 21.96% | 11.44% | 1.49% |
| Shift magnitude | 4.09% | 1.58% | 1.09% |

kex and shift-magnitude errors improve in 6/6 cases; pB improves in 5/6. These six cases represent three physical exchange rates with two noise draws, not six independent systems. Known R1 is fixed, geometry is fixed, and fitted R2 is allowed to vary. Shift errors use magnitudes and retain original signed fitted results. Source: `ml_results/fit_expansion_20260910_summary.json`.

The main unresolved observation is kex=300 s^-1: corrected kex remains about 20–22% low and pB about 24–27% high. More correction is needed before claiming accurate exchange-parameter recovery in this regime. The clean J=0 controls recover the parameters closely, so this particular failure is not explained by an inability to fit the clean reference. Broader fitting/noise tests are still needed to distinguish model bias, denoising, and identifiability effects.

## Five decisions to ask for

1. **Target acquisition:** Which exact pulse programme should version 1 support? Obtain the pulse-program file, field strengths, T, pulse widths, phase cycle, frequency grid and reference acquisition.
2. **Experimental comparison:** Which sample has suitable exchange and available repeat measurements? Can we obtain an artefact-suppressed or independently characterised reference? Physical decoupling can change more than J, so a reference is not automatically the synthetic J=0 target.
3. **Operating range:** Which kex, pB, shift, relaxation and SNR ranges matter for the intended application? Keep the broad training distribution until this discussion; report a separate application-specific benchmark if agreed.
4. **Success criterion:** What bias in kex/pB/shift is acceptable, and what fraction of failures is tolerable? When should the tool reject a curve rather than correct it? Agree criteria before viewing the final test results.
5. **Publication scope:** Should the first paper be a bounded computational methods study or require experimental validation? Identify the sample/data provider and acquisition plan accordingly.

## Computational work before claiming readiness

- Investigate the residual bias using repeated noise and additional exchange groups. Include a coupled-target denoiser so that improvement over raw data is not mistaken for J-specific correction.
- Replace the historical weak-intensity clipping assumptions with an explicit intensity/mask-aware protocol, then retrain and calibrate uncertainties.
- Freeze a new benchmark with disjoint reserved seeds, verify overlap, repeat finalist training across at least three seeds, and test J=0/no-exchange controls and held-out regimes.
- Test B1/J uncertainty and missing/irregular frequencies for the chosen experimental acquisition.
- Complete independent relaxation-rate/source-term checks and consolidate prediction around the actual tested CNN checkpoint format.

These tasks can proceed alongside planning measurements. Another major dataset expansion or remote GPU is not the immediate bottleneck for the current small model and pilot experiments.

## Suggested presentation pacing

1. Goal and example curve: 1 minute.
2. Simulator scope and model benchmark: 2 minutes.
3. Noise and evaluation corrections: 2 minutes.
4. Expanded fitting result and residual failure: 3 minutes.
5. Remaining computational work: 1 minute.
6. PI decisions: allow 4–6 minutes of discussion.

The slides include more detailed speaker notes and source filenames. Historical decks and datasets were preserved.
