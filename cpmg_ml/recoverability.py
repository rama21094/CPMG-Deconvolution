"""Per-profile recoverability map for CPMG de-J-coupling.

Break-even SNR quoted against the *median* J artefact hides the thing an
experimentalist actually needs to know: for WHICH exchange regimes is the
artefact recoverable at a given SNR? This script reports descriptive criteria:

  1. its maximum artefact exceeds the maximum of one noise realisation, and
  2. the model's residual error is small compared with the artefact.

The first is a heuristic, NOT a necessary condition or a recovery ceiling.
Both are per-profile properties driven by (k_ex, p_B, dw_N), none of which the
model is given.  This bins the held-out test set over those parameters and
reports the fraction of profiles meeting a criterion, so the output is a map of
where the method works rather than a single global number.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .baselines import MODELS, add_intensity_noise, featurise, load_split
from .sampling import METADATA_KEYS

plt.rcParams.update({
    "font.family": "Arial", "font.size": 16, "axes.titlesize": 17,
    "axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.fontsize": 14, "axes.linewidth": 1.2, "lines.linewidth": 2.2,
})


def predict(ckpt: Path, raw: dict, sigma: float, seed: int = 3000):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    d = dict(raw)
    if sigma > 0:
        d["with_j"] = add_intensity_noise(raw["with_j"], raw["t_relax"], sigma,
                                          np.random.default_rng(seed))
    pts, glb, _, scale, _ = featurise(d, ck["stats"])
    m = MODELS[ck["arch"]](pts.shape[-1], glb.shape[-1])
    m.load_state_dict(ck["state_dict"]); m.eval()
    with torch.no_grad():
        pr = torch.cat([m(torch.from_numpy(pts[i:i+4096]), torch.from_numpy(glb[i:i+4096]))
                        for i in range(0, len(pts), 4096)]).numpy()
    return d["with_j"] - scale * pr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/cpmg_dej_chemex_500k"))
    ap.add_argument("--clean-ckpt", type=Path, default=Path("runs/b500k_clean/cnn.pt"))
    ap.add_argument("--noisy-ckpt", type=Path, default=Path("runs/b500k_noise01/cnn.pt"))
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--out", type=Path,
                    default=Path("docs/group_meeting_figs/gm_recoverability.png"))
    a = ap.parse_args()

    raw = load_split(a.data_dir, "test")
    T = raw["t_relax"]
    wj, nj = raw["with_j"], raw["no_j"]
    meta = np.concatenate([h5py.File(f, "r")["metadata"][:]
                           for f in sorted(a.data_dir.glob("test_*.h5"))]).astype(np.float64)

    # Operational retrospective mask; not a universal measurability threshold.
    mask = np.exp(-wj * T) > 0.01

    def pmax(x):                      # per-profile max over measurable points
        e = np.abs(x).copy(); e[~mask] = np.nan
        with np.errstate(all="ignore"):
            return np.nanmax(e, axis=1)

    artefact = pmax(wj - nj)                                   # what must be removed
    err_clean = pmax(predict(a.clean_ckpt, raw, 0.0) - nj)
    err_noisy = pmax(predict(a.noisy_ckpt, raw, a.sigma) - nj)
    noise_amp = pmax(add_intensity_noise(wj, T, a.sigma, np.random.default_rng(7)) - wj)

    good = np.isfinite(artefact) & np.isfinite(err_noisy) & np.isfinite(err_clean)
    # criterion: maximum prediction error <25% of maximum clean J difference
    rec_clean = good & (err_clean < 0.25 * artefact)
    rec_noisy = good & (err_noisy < 0.25 * artefact)
    visible = good & (artefact > noise_amp)                    # single-draw heuristic only

    kex = meta[:, METADATA_KEYS.index("k_ex")]
    pb = meta[:, METADATA_KEYS.index("p_B")]
    dw = meta[:, METADATA_KEYS.index("dw_N")]

    print(f"profiles {good.sum()}   measurable points {100*mask.mean():.1f}%")
    print(f"median artefact {np.nanmedian(artefact[good]):.3f} s-1   "
          f"median noise {np.nanmedian(noise_amp[good]):.3f} s-1")
    for lbl, m_ in [("recoverable, clean", rec_clean),
                    ("artefact visible above noise (SNR %.0f)" % (1/a.sigma), visible),
                    ("recoverable, SNR %.0f" % (1/a.sigma), rec_noisy)]:
        print(f"{lbl:<45}{100*m_.sum()/good.sum():>6.1f}%")

    def grid(xv, yv, sel, xlog, nb=14):
        xe = (np.logspace(np.log10(xv.min()), np.log10(xv.max()), nb+1) if xlog
              else np.linspace(xv.min(), xv.max(), nb+1))
        ye = np.linspace(yv.min(), yv.max(), nb+1)
        out = np.full((nb, nb), np.nan)
        for i in range(nb):
            for j in range(nb):
                c = (xv >= xe[i]) & (xv < xe[i+1]) & (yv >= ye[j]) & (yv < ye[j+1]) & good
                if c.sum() >= 12:
                    out[j, i] = 100.0 * (sel & c).sum() / c.sum()
        return out, xe, ye

    panels = [
        ("$k_{ex}$ (s$^{-1}$)", kex, True,  r"$\Delta\omega$ (ppm)", dw),
        ("$k_{ex}$ (s$^{-1}$)", kex, True,  "$p_B$", pb),
        ("$p_B$", pb, False, r"$\Delta\omega$ (ppm)", dw),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 10.2))
    for col, (xl, xv, xlog, yl, yv) in enumerate(panels):
        for row, (sel, tag) in enumerate([(rec_clean, "no noise"),
                                          (rec_noisy, f"SNR {1/a.sigma:.0f}")]):
            g, xe, ye = grid(xv, yv, sel, xlog)
            ax = axes[row, col]
            im = ax.pcolormesh(xe, ye, g, cmap="viridis", vmin=0, vmax=100,
                               shading="auto")
            if xlog:
                ax.set_xscale("log")
            ax.set_xlabel(xl); ax.set_ylabel(yl)
            ax.set_title(f"{tag}", fontsize=17)
            if col == 2:
                cb = fig.colorbar(im, ax=ax)
                cb.set_label("% recoverable", fontsize=15)
                cb.ax.tick_params(labelsize=13)
    fig.suptitle("Where de-J-coupling works: % of profiles whose J artefact is removed to <25% "
                 "of its size\n(CNN trained on 500k, held-out test set, ChemEx sequence)",
                 fontsize=18, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=160, facecolor="white", bbox_inches="tight")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
