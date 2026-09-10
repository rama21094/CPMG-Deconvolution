"""Paired acquisition pilot with shared-reference covariance and explicit masks."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from .acquisition_sweep import configs
from .generate_dataset import load_specs_from_json
from .sampling import sample_simulation_case
from .simulator import CPMGSimulator
from .noise_statistics import shared_reference_covariance, oracle_separation


def one(payload):
    seed, specs_json, cfg, sigma = payload
    params, _ = sample_simulation_case(np.random.default_rng(seed), load_specs_from_json(specs_json))
    values, points, failures = [], [], []
    for c in cfg:
        try:
            a, b = CPMGSimulator().simulate_dej_pair(params, ncyc_range=c['grid'], t_relax=c['T'], sequence='chemex')
            if a.skipped_ncyc or b.skipped_ncyc or not (np.isfinite(a.r2_eff).all() and np.isfinite(b.r2_eff).all()):
                raise ValueError('Invalid or incomplete profile')
            keep = np.exp(-a.r2_eff*c['T']) >= 5*sigma
            points.append(int(keep.sum()))
            if keep.any():
                cov = shared_reference_covariance(a.r2_eff[keep], c['T'], sigma)
                values.append(oracle_separation((a.r2_eff-b.r2_eff)[keep], cov))
            else:
                values.append(None)
            failures.append(None)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            if len(points) == len(values):
                points.append(0)
            values.append(None)
            failures.append(str(exc))
    return {'seed': seed, 'D': values, 'retained_points': points, 'failures': failures}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ranges', type=Path, default=Path('data/cpmg_dej_chemex_500k/range_specs.json'))
    p.add_argument('--n-profiles', type=int, default=100)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--out', type=Path, default=Path('docs/ml_results/acquisition_covariance_pilot.json'))
    a = p.parse_args()
    cfg = configs([.02,.04,.06,.08,.10], [1,2,4])
    specs = a.ranges.read_text()
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, [(91000000+i, specs, cfg, .01) for i in range(a.n_profiles)]))
    d = np.array([[np.nan if x is None else x for x in r['D']] for r in rows])
    common = np.isfinite(d).all(1)
    summary = []
    for j,c in enumerate(cfg):
        v = d[common,j]
        summary.append({'T':c['T'], 'step':c['step'], 'n':c['n'],
                        'median_D':float(np.median(v)) if len(v) else None,
                        'fraction_D_above_3':float((v>3).mean()) if len(v) else None})
    result = {'description':'Oracle known-curve separation, first-order shared-reference covariance; expected point SNR>=5. Descriptive D>3, not calibrated recovery. Pilot only; common complete cases may select a biased subset.',
              'sigma_rel':.01, 'ranges':json.loads(specs), 'common_profiles':int(common.sum()),
              'results':summary, 'raw':rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({'common_profiles':result['common_profiles'], 'results':summary}, indent=2))


if __name__ == '__main__':
    main()
