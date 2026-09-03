"""Baseline ladder for CPMG de-J-coupling: identity -> linear -> MLP -> CNN -> Transformer.

Any claim that an architecture is "the right one" only means something against a
ladder, so this trains all of them under identical data, preprocessing, budget and
metrics, and reports each against the identity baseline (predict the input unchanged).

HONEST INPUTS ONLY
------------------
The pre-existing ConditionalCPMGDeJNet conditions on 13 metadata values including
k_ex, p_B and dw_N. Those are exactly the quantities a real experiment is trying to
measure, so a model that needs them as input cannot be used on real data. Everything
here is restricted to what is actually known at the spectrometer:

    per point : nu_cpmg, R2_eff(with J)
    global    : B0 (MHz), B1_N (Hz), J_IS (Hz)

SCALE HANDLING
--------------
R2_eff spans ~1-150 s-1 across the sampled parameter ranges, while the J artefact is
only a few percent of it. Both are handled by per-profile scaling: each profile is
divided by its own peak R2, the network predicts the *residual* (with_J - no_J) in
those units, and the prediction is mapped back with

    R2_no_J_pred = R2_with_J - scale * residual_pred

so the network spends its capacity on the artefact rather than on copying the input.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

from .sampling import METADATA_KEYS

GLOBAL_KEYS = ["B0_MHz", "B1_N", "J_IS"]          # knowable at the spectrometer
GLOBAL_IDX = [METADATA_KEYS.index(k) for k in GLOBAL_KEYS]


# ------------------------------------------------------------------ data
def load_split(data_dir: Path, split: str) -> dict[str, np.ndarray]:
    files = sorted(Path(data_dir).glob(f"{split}_*.h5"))
    if not files:
        raise FileNotFoundError(f"no {split} shards in {data_dir}")
    wj, nj, nu, meta = [], [], [], []
    for f in files:
        with h5py.File(f, "r") as h:
            wj.append(h["r2_with_j"][:]); nj.append(h["r2_no_j"][:])
            nu.append(h["nu_cp"][:]);     meta.append(h["metadata"][:])
    with h5py.File(files[0], "r") as h:
        t_relax = float(h.attrs["t_relax"])
    return {
        "with_j": np.concatenate(wj).astype(np.float64),
        "no_j":   np.concatenate(nj).astype(np.float64),
        "nu":     np.concatenate(nu).astype(np.float64),
        "glob":   np.concatenate(meta).astype(np.float64)[:, GLOBAL_IDX],
        "t_relax": t_relax,
    }


def add_intensity_noise(r2: np.ndarray, t_relax: float, sigma_rel: float,
                        rng: np.random.Generator) -> np.ndarray:
    """Perturb a R2_eff profile the way a real experiment is perturbed.

    Noise in a CPMG experiment is Gaussian in the *intensity* domain, not in
    R2_eff.  Each point comes from a ratio of two measured peak heights,

        R2_eff = -ln(I / I0) / T,

    so we go back to intensities (I0 = 1 by construction), add independent
    Gaussian noise of standard deviation `sigma_rel` (quoted relative to the
    reference intensity, i.e. 1/SNR) to both the reference and the relaxed
    point, and take the log again.  This makes the induced R2 noise
    *heteroscedastic*: sigma_R2 ~ (sigma_rel/T) * sqrt(1 + exp(2 R2 T)), so
    fast-relaxing points -- exactly the interesting ones -- are far noisier
    than slow ones.  Adding homoscedastic noise directly to R2_eff would make
    the task look easier than it is.
    """
    if sigma_rel <= 0:
        return r2
    inten = np.exp(-r2 * t_relax)
    i_ref = 1.0 + sigma_rel * rng.standard_normal(size=(r2.shape[0], 1))
    i_obs = inten + sigma_rel * rng.standard_normal(size=r2.shape)
    floor = 1e-6
    return -np.log(np.clip(i_obs, floor, None) / np.clip(i_ref, floor, None)) / t_relax


def featurise(d: dict[str, np.ndarray], stats: dict | None = None):
    """Per-profile scaled features. Returns (x_points, x_global, y_residual, scale)."""
    wj, nj, nu, g = d["with_j"], d["no_j"], d["nu"], d["glob"]
    scale = wj.max(axis=1, keepdims=True)                    # (N,1) per-profile peak R2
    r2n = wj / scale                                         # in [~0,1]
    logn = np.log10(nu)
    resid = (wj - nj) / scale                                # target

    if stats is None:
        stats = {
            "logn_mu": float(logn.mean()), "logn_sd": float(logn.std()),
            "g_mu": g.mean(0).tolist(), "g_sd": (g.std(0) + 1e-12).tolist(),
            "logs_mu": float(np.log10(scale).mean()), "logs_sd": float(np.log10(scale).std()),
        }
    logn_z = (logn - stats["logn_mu"]) / stats["logn_sd"]
    g_z = (g - np.array(stats["g_mu"])) / np.array(stats["g_sd"])
    logs_z = (np.log10(scale) - stats["logs_mu"]) / stats["logs_sd"]

    x_pts = np.stack([r2n, logn_z], axis=1)                  # (N, 2, L)
    x_glb = np.concatenate([g_z, logs_z], axis=1)            # (N, 4)
    return (x_pts.astype(np.float32), x_glb.astype(np.float32),
            resid.astype(np.float32), scale.astype(np.float32), stats)


# ---------------------------------------------------------------- models
class LinearBase(nn.Module):
    def __init__(self, L, gdim):
        super().__init__()
        self.f = nn.Linear(2 * L + gdim, L)

    def forward(self, p, g):
        return self.f(torch.cat([p.flatten(1), g], 1))


class MLP(nn.Module):
    def __init__(self, L, gdim, hidden=512, depth=4):
        super().__init__()
        layers, d = [], 2 * L + gdim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        layers += [nn.Linear(d, L)]
        self.f = nn.Sequential(*layers)

    def forward(self, p, g):
        return self.f(torch.cat([p.flatten(1), g], 1))


class FiLMBlock(nn.Module):
    def __init__(self, c, cond, k=5):
        super().__init__()
        self.c1 = nn.Conv1d(c, c, k, padding=k // 2)
        self.c2 = nn.Conv1d(c, c, k, padding=k // 2)
        self.n1 = nn.GroupNorm(8, c); self.n2 = nn.GroupNorm(8, c)
        self.film = nn.Linear(cond, 2 * c)
        self.act = nn.GELU()

    def forward(self, x, cond):
        gamma, beta = self.film(cond).chunk(2, 1)
        h = self.act(self.n1(self.c1(x)))
        h = self.n2(self.c2(h)) * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
        return self.act(x + h)


class CNN(nn.Module):
    """FiLM-conditioned 1D residual ConvNet (the existing repo architecture,
    restricted to honest global inputs)."""

    def __init__(self, L, gdim, ch=96, blocks=6):
        super().__init__()
        self.cond = nn.Sequential(nn.Linear(gdim, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU())
        self.stem = nn.Conv1d(2, ch, 5, padding=2)
        self.blocks = nn.ModuleList([FiLMBlock(ch, 128) for _ in range(blocks)])
        self.head = nn.Sequential(nn.Conv1d(ch, ch, 5, padding=2), nn.GELU(), nn.Conv1d(ch, 1, 1))

    def forward(self, p, g):
        c = self.cond(g)
        h = self.stem(p)
        for b in self.blocks:
            h = b(h, c)
        return self.head(h).squeeze(1)


class Transformer(nn.Module):
    """Encoder over points; same family as ExperimentalCPMGTransformer.

    A learned positional embedding is included: without it the model is
    permutation-invariant apart from the log-nu feature, which handicaps it
    badly on a fixed regular grid relative to a ConvNet. Keeping it optional
    (`use_pos`) lets the fixed-grid and irregular-grid cases be compared fairly.
    """

    def __init__(self, L, gdim, dim=128, heads=8, layers=4, ff=256, use_pos=True):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, L, dim)) if use_pos else None
        if self.pos is not None:
            nn.init.trunc_normal_(self.pos, std=0.02)
        self.pt = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.gl = nn.Sequential(nn.Linear(gdim, dim), nn.GELU(), nn.Linear(dim, dim))
        enc = nn.TransformerEncoderLayer(dim, heads, ff, dropout=0.0,
                                         activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, p, g):
        h = self.pt(p.transpose(1, 2)) + self.gl(g).unsqueeze(1)
        if self.pos is not None:
            h = h + self.pos
        return self.head(self.enc(h)).squeeze(-1)


def _transformer_nopos(L, gdim):
    return Transformer(L, gdim, use_pos=False)


MODELS = {"linear": LinearBase, "mlp": MLP, "cnn": CNN, "transformer": Transformer,
          "transformer_nopos": _transformer_nopos}


# -------------------------------------------------------------- training
def evaluate(model, pts, glb, resid, scale, wj, nj, dev, bs=2048):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(pts), bs):
            preds.append(model(pts[i:i+bs].to(dev), glb[i:i+bs].to(dev)).cpu())
    pred_resid = torch.cat(preds).numpy()
    pred_no_j = wj - scale * pred_resid                      # back to s-1
    err = pred_no_j - nj
    rmse = float(np.sqrt((err ** 2).mean()))
    ident = float(np.sqrt(((wj - nj) ** 2).mean()))
    per_max = np.abs(err).max(1)
    return {
        "rmse": rmse,
        "identity_rmse": ident,
        "artefact_removed_pct": 100.0 * (1 - rmse / ident),
        "median_max_abs_err": float(np.median(per_max)),
        "p95_max_abs_err": float(np.percentile(per_max, 95)),
    }


def train_one(name, tr, va, te, dev, epochs, bs, lr, seed=0, resample=None):
    torch.manual_seed(seed)
    L = tr["pts"].shape[-1]; gdim = tr["glb"].shape[-1]
    model = MODELS[name](L, gdim).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(tr["pts"])
    steps = max(1, n // bs) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    lossf = nn.SmoothL1Loss(beta=0.01)

    best, best_state, t0 = math.inf, None, time.time()
    step = 0
    for ep in range(epochs):
        if resample is not None:
            tr = resample(ep)          # fresh noise draw each epoch (augmentation)
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n - bs + 1, bs):
            idx = perm[i:i+bs]
            p, g, y = tr["pts"][idx].to(dev), tr["glb"][idx].to(dev), tr["resid"][idx].to(dev)
            loss = lossf(model(p, g), y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step < steps - 1:
                sched.step()
            step += 1
        m = evaluate(model, va["pts"], va["glb"], va["resid"], va["scale_np"],
                     va["wj"], va["nj"], dev)
        if m["rmse"] < best:
            best = m["rmse"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    res = evaluate(model, te["pts"], te["glb"], te["resid"], te["scale_np"],
                   te["wj"], te["nj"], dev)
    res.update(name=name, params=n_par, train_s=round(time.time() - t0, 1), val_rmse=best)
    return model, res


def pack(d, stats=None, dev="cpu", sigma_rel=0.0, seed=0):
    """Featurise a split.  Noise is added to the INPUT only -- the target stays
    the clean no-J curve, which is the physical answer we actually want."""
    if sigma_rel > 0:
        d = dict(d)
        d["with_j"] = add_intensity_noise(d["with_j"], d["t_relax"], sigma_rel,
                                          np.random.default_rng(seed))
    pts, glb, resid, scale, stats = featurise(d, stats)
    return {
        "pts": torch.from_numpy(pts), "glb": torch.from_numpy(glb),
        "resid": torch.from_numpy(resid), "scale_np": scale,
        "wj": d["with_j"], "nj": d["no_j"],
    }, stats


def main():
    ap = argparse.ArgumentParser(description="Baseline ladder for CPMG de-J-coupling.")
    ap.add_argument("--data-dir", type=Path, default=Path("data/cpmg_dej_chemex"))
    ap.add_argument("--out-dir", type=Path, default=Path("runs/baselines"))
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="relative intensity noise (1/SNR) added to the INPUT curve; "
                         "0.01 = 1%% of the reference peak height. Target stays clean.")
    a = ap.parse_args()

    a.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={a.device}  data={a.data_dir}")
    raw = {s: load_split(a.data_dir, s) for s in ("train", "val", "test")}
    # Stats are fitted on a noisy draw so that normalisation matches what the
    # model will actually see; val/test get one fixed draw for reproducibility.
    tr, stats = pack(raw["train"], sigma_rel=a.noise, seed=1000)
    va, _ = pack(raw["val"], stats, sigma_rel=a.noise, seed=2000)
    te, _ = pack(raw["test"], stats, sigma_rel=a.noise, seed=3000)
    resample = None
    if a.noise > 0:
        print(f"input noise: sigma_rel={a.noise:.4g} (intensity domain), target clean")
        resample = lambda ep: pack(raw["train"], stats, sigma_rel=a.noise, seed=10_000 + ep)[0]
    print(f"train={len(tr['pts'])}  val={len(va['pts'])}  test={len(te['pts'])}  "
          f"points={tr['pts'].shape[-1]}")

    ident = float(np.sqrt(((te["wj"] - te["nj"]) ** 2).mean()))
    rows = [{"name": "identity", "params": 0, "rmse": ident, "identity_rmse": ident,
             "artefact_removed_pct": 0.0,
             "median_max_abs_err": float(np.median(np.abs(te["wj"] - te["nj"]).max(1))),
             "p95_max_abs_err": float(np.percentile(np.abs(te["wj"] - te["nj"]).max(1), 95)),
             "train_s": 0.0}]

    for name in a.models:
        print(f"\n--- {name} ---", flush=True)
        model, res = train_one(name, tr, va, te, a.device, a.epochs, a.batch_size, a.lr,
                               resample=resample)
        rows.append(res)
        torch.save({"state_dict": model.state_dict(), "stats": stats, "arch": name,
                    "noise": a.noise},
                   a.out_dir / f"{name}.pt")
        print(f"  params={res['params']:,}  test RMSE={res['rmse']:.4f} s-1  "
              f"artefact removed={res['artefact_removed_pct']:.1f}%  ({res['train_s']}s)")

    (a.out_dir / "results.json").write_text(json.dumps(rows, indent=2))
    print(f"\n{'model':<13}{'params':>10}{'test RMSE':>12}{'artefact rm':>13}{'med max err':>13}")
    for r in rows:
        print(f"{r['name']:<13}{r['params']:>10,}{r['rmse']:>12.4f}"
              f"{r['artefact_removed_pct']:>12.1f}%{r['median_max_abs_err']:>13.4f}")
    print(f"\nwrote {a.out_dir/'results.json'}")


if __name__ == "__main__":
    main()
