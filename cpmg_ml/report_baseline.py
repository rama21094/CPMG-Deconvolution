"""Diagnostics for a trained de-J-coupling baseline: example overlays, error
distribution, and error broken down by the physics parameters that were NOT
given to the model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .baselines import GLOBAL_IDX, MODELS, featurise, load_split
from .sampling import METADATA_KEYS

plt.rcParams.update({
    "font.family": "Arial", "font.size": 15, "axes.titlesize": 17,
    "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "legend.fontsize": 13, "axes.linewidth": 1.2, "lines.linewidth": 2.0,
})
RUST, TEAL, GREEN, NAVY = "#C1440E", "#1C7293", "#0F7B6C", "#21295C"


def full_metadata(data_dir: Path, split: str) -> np.ndarray:
    files = sorted(Path(data_dir).glob(f"{split}_*.h5"))
    return np.concatenate([h5py.File(f, "r")["metadata"][:] for f in files]).astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("runs/baselines/cnn.pt"))
    ap.add_argument("--data-dir", type=Path, default=Path("data/cpmg_dej_chemex"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, default=Path("docs/group_meeting_figs/gm_dej_baseline.png"))
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    raw = load_split(a.data_dir, a.split)
    pts, glb, resid, scale, _ = featurise(raw, ck["stats"])
    L, gdim = pts.shape[-1], glb.shape[-1]
    model = MODELS[ck["arch"]](L, gdim).to(a.device)
    model.load_state_dict(ck["state_dict"]); model.eval()

    preds = []
    with torch.no_grad():
        for i in range(0, len(pts), 2048):
            preds.append(model(torch.from_numpy(pts[i:i+2048]).to(a.device),
                               torch.from_numpy(glb[i:i+2048]).to(a.device)).cpu().numpy())
    pred_resid = np.concatenate(preds)

    wj, nj, nu = raw["with_j"], raw["no_j"], raw["nu"]
    pred = wj - scale * pred_resid
    err = pred - nj
    ident_err = wj - nj
    rmse = float(np.sqrt((err ** 2).mean()))
    ident = float(np.sqrt((ident_err ** 2).mean()))
    meta = full_metadata(a.data_dir, a.split)

    fig = plt.figure(figsize=(15, 9.2))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.30)

    # --- three example profiles, chosen across artefact size ---
    amp = np.abs(ident_err).max(1)
    picks = [int(np.argsort(amp)[int(q * (len(amp) - 1))]) for q in (0.50, 0.90, 0.995)]
    for j, idx in enumerate(picks):
        ax = fig.add_subplot(gs[0, j])
        ax.plot(nu[idx], wj[idx], "o-", color=RUST, ms=4, label="input (with J)")
        ax.plot(nu[idx], nj[idx], "s-", color=TEAL, ms=4, label="target (J = 0)")
        ax.plot(nu[idx], pred[idx], "--", color=GREEN, lw=2.4, label="model")
        ax.set_xscale("log")
        ax.set_xlabel(r"$\nu_{CPMG}$ (Hz)")
        if j == 0:
            ax.set_ylabel(r"$R_{2,eff}$ (s$^{-1}$)")
        pct = {0: "median", 1: "90th pct", 2: "99.5th pct"}[j]
        ax.set_title(f"{pct} artefact\nmax|Δ| = {amp[idx]:.2f} s$^{{-1}}$", fontsize=15)
        ax.grid(alpha=0.25, which="both")
        if j == 0:
            ax.legend(frameon=False, fontsize=12)

    # --- error distribution vs identity ---
    ax = fig.add_subplot(gs[1, 0])
    bins = np.logspace(-4, 1, 60)
    ax.hist(np.abs(ident_err).ravel(), bins=bins, color=RUST, alpha=0.55, label="identity")
    ax.hist(np.abs(err).ravel(), bins=bins, color=GREEN, alpha=0.75, label="model")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"|error| (s$^{-1}$)"); ax.set_ylabel("count")
    ax.set_title(f"RMSE {ident:.3f} → {rmse:.3f} s$^{{-1}}$\n({100*(1-rmse/ident):.1f}% removed)")
    ax.legend(frameon=False)

    # --- error vs two hidden parameters ---
    for j, key in enumerate(("k_ex", "dw_N")):
        ax = fig.add_subplot(gs[1, 1 + j])
        v = meta[:, METADATA_KEYS.index(key)]
        pm = np.abs(err).max(1)
        nb = 18
        edges = (np.logspace(np.log10(v.min()), np.log10(v.max()), nb + 1) if key == "k_ex"
                 else np.linspace(v.min(), v.max(), nb + 1))
        cen = 0.5 * (edges[1:] + edges[:-1])
        med = [np.median(pm[(v >= edges[i]) & (v < edges[i + 1])]) if
               ((v >= edges[i]) & (v < edges[i + 1])).sum() else np.nan for i in range(nb)]
        base = [np.median(amp[(v >= edges[i]) & (v < edges[i + 1])]) if
                ((v >= edges[i]) & (v < edges[i + 1])).sum() else np.nan for i in range(nb)]
        ax.plot(cen, base, "o-", color=RUST, ms=4, label="identity")
        ax.plot(cen, med, "s-", color=GREEN, ms=4, label="model")
        if key == "k_ex":
            ax.set_xscale("log"); ax.set_xlabel(r"$k_{ex}$ (s$^{-1}$)")
        else:
            ax.set_xlabel(r"$\Delta\omega$ (ppm)")
        ax.set_yscale("log")
        ax.set_ylabel(r"median max|error| (s$^{-1}$)")
        ax.set_title(f"error vs {key}\n(not given to the model)", fontsize=15)
        ax.grid(alpha=0.25, which="both"); ax.legend(frameon=False, fontsize=12)

    fig.suptitle(f"De-J-coupling, {ck['arch']} on held-out test set "
                 f"({len(wj)} profiles, ChemEx sequence)", fontsize=18, y=0.975)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=160, facecolor="white", bbox_inches="tight")
    print(f"saved {a.out}")
    print(f"identity RMSE {ident:.4f} -> model {rmse:.4f} s-1  ({100*(1-rmse/ident):.1f}% removed)")
    print(f"median max|err| {np.median(np.abs(err).max(1)):.4f}  "
          f"p95 {np.percentile(np.abs(err).max(1), 95):.4f} s-1")


if __name__ == "__main__":
    main()
