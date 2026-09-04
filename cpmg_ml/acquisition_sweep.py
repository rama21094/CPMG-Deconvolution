"""How much of the J artefact is detectable, as a function of how the experiment
is acquired.

The 26.3% "visible above noise" ceiling reported for the current acquisition
(40 nu points, T_relax = 40 ms) is not a property of the physics -- it is a
property of that sampling choice. This sweeps the acquisition and recomputes
the ceiling, with no model training involved, because detectability is a
property of the data alone.

Detection statistic
-------------------
The artefact d_i = R2_withJ(i) - R2_noJ(i) is a deterministic, smooth vector;
the noise on each point is independent with standard deviation

    sigma_i = (sigma_rel / T) * sqrt(1 + exp(2 * R2_i * T))

(from R2 = -ln(I/I0)/T with intensity noise sigma_rel on both I and I0). The
optimal statistic for deciding whether d is present is the matched-filter SNR

    D = sqrt( sum_i (d_i / sigma_i)^2 ),

which is the right way to compare acquisitions with different numbers of
points: adding points helps only in proportion to the artefact they actually
carry, and unmeasurable points (R2*T large) contribute ~0 automatically
because sigma_i explodes there. D > 3 is a conventional detection threshold.

All configurations are evaluated on the SAME sampled parameter sets, so the
comparison is paired.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .sampling import load_range_specs, sample_simulation_case
from .simulator import CPMGSimulator

NU_MAX = 1000.0            # keep the top of the nu range fixed across configs


def configs(t_list, step_list):
    out = []
    for T in t_list:
        ncyc_max = int(round(NU_MAX * T))
        for st in step_list:
            grid = np.arange(st, ncyc_max + 1, st, dtype=np.int32)
            if len(grid) >= 4:
                out.append({"T": T, "step": st, "n": len(grid), "grid": grid})
    return out


def detectability(with_j, no_j, r2, T, sigma_rel):
    d = with_j - no_j
    sigma = (sigma_rel / T) * np.sqrt(1.0 + np.exp(np.clip(2.0 * r2 * T, None, 700.0)))
    return float(np.sqrt(np.sum((d / sigma) ** 2)))


def one_case(args):
    seed, specs_json, cfg_meta, sigma_rel = args
    from .generate_dataset import load_specs_from_json
    specs = load_specs_from_json(specs_json)
    rng = np.random.default_rng(seed)
    params, _ = sample_simulation_case(rng, specs)
    sim = CPMGSimulator()
    out = []
    for T, grid in cfg_meta:
        try:
            wj, nj = sim.simulate_dej_pair(params, ncyc_range=np.asarray(grid),
                                           t_relax=T, sequence="chemex")
        except Exception:
            out.append(np.nan); continue
        if wj.skipped_ncyc or nj.skipped_ncyc or not np.all(np.isfinite(wj.r2_eff)):
            out.append(np.nan); continue
        out.append(detectability(wj.r2_eff, nj.r2_eff, wj.r2_eff, T, sigma_rel))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-profiles", type=int, default=3000)
    ap.add_argument("--sigma", type=float, default=0.01, help="1/SNR per point")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=90_000_000)
    ap.add_argument("--out", type=Path, default=Path("docs/ml_results/acquisition_sweep.json"))
    ap.add_argument("--save-raw", type=Path, default=Path("docs/ml_results/acquisition_D.npz"),
                    help="raw per-profile detectability matrix; D scales exactly as 1/sigma, "
                         "so this allows any averaging level to be evaluated without re-simulating")
    a = ap.parse_args()

    specs = load_range_specs(None)
    from .sampling import specs_to_json
    specs_json = specs_to_json(specs)

    cfgs = configs([0.02, 0.04, 0.06, 0.08, 0.10], [1, 2, 4])
    cfg_meta = [(c["T"], c["grid"].tolist()) for c in cfgs]
    print(f"{len(cfgs)} acquisitions x {a.n_profiles} profiles, SNR {1/a.sigma:.0f}\n")

    seeds = [a.seed + i for i in range(a.n_profiles)]
    payload = [(s, specs_json, cfg_meta, a.sigma) for s in seeds]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(one_case, payload, chunksize=8))
    D = np.array(rows, dtype=float)              # (n_profiles, n_configs)

    print(f"{'T (ms)':>7}{'step':>6}{'points':>8}{'median D':>11}{'D>3':>8}{'D>5':>8}")
    res = []
    base = None
    for j, c in enumerate(cfgs):
        col = D[:, j]; ok = np.isfinite(col)
        med = float(np.median(col[ok])); f3 = float(100*(col[ok] > 3).mean())
        f5 = float(100*(col[ok] > 5).mean())
        res.append({"T": c["T"], "step": c["step"], "n": c["n"],
                    "median_D": med, "pct_D_gt3": f3, "pct_D_gt5": f5})
        if c["T"] == 0.04 and c["step"] == 1:
            base = (med, f3)
        print(f"{1000*c['T']:>7.0f}{c['step']:>6}{c['n']:>8}{med:>11.2f}{f3:>7.1f}%{f5:>7.1f}%")
    if base:
        print(f"\ncurrent acquisition (T=40 ms, 40 points): median D {base[0]:.2f}, "
              f"{base[1]:.1f}% detectable")
        best = max(res, key=lambda r: r["pct_D_gt3"])
        print(f"best swept acquisition: T={1000*best['T']:.0f} ms, {best['n']} points -> "
              f"{best['pct_D_gt3']:.1f}% detectable "
              f"({best['pct_D_gt3']/base[1]:.2f}x the current acquisition)")
    if a.save_raw:
        a.save_raw.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.save_raw, D=D,
                            T=np.array([c["T"] for c in cfgs]),
                            step=np.array([c["step"] for c in cfgs]),
                            n=np.array([c["n"] for c in cfgs]), sigma_rel=a.sigma)
        print(f"wrote {a.save_raw}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"sigma_rel": a.sigma, "n_profiles": a.n_profiles,
                                 "results": res}, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
