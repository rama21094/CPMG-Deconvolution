"""Figure: 30x30 matrix comparison against ChemEx's cpmg_hn_dq_zq basis,
before and after fixing the two multiple-quantum-block defects."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, "/Users/shankararamasharma/Desktop/IISc/CPMG Project")
OUT = "/Users/shankararamasharma/Desktop/IISc/CPMG Project/docs/group_meeting_figs"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "Arial", "font.size": 17, "axes.titlesize": 19,
    "axes.labelsize": 17, "xtick.labelsize": 15, "ytick.labelsize": 15,
    "axes.linewidth": 1.4,
})
RUST, GREEN = "#C1440E", "#0F7B6C"

from cpmg_ml.simulator import CPMGSimulator, b0_mhz_to_tesla
from chemex.configuration.conditions import Conditions
from chemex.models.loader import register_kinetic_settings
from chemex.models.model import ModelSpec
from chemex.nmr.basis import Basis
from chemex.nmr.spectrometer import Spectrometer
from chemex.parameters.spin_system import SpinSystem
register_kinetic_settings()

H_MHZ = 600.0
B0 = b0_mhz_to_tesla(H_MHZ)
TAU_M, TAU_E, S2 = 5e-9, 50e-12, 0.85
R_IS, R_EFF, CSA_N, THETA_N = 1.02e-10, 1.86e-10, -160e-6, np.radians(22.0)
J_IS, K_EX, P_B, DW_PPM = 92.0, 300.0, 0.05, 2.0
kab, kba = K_EX*P_B, K_EX*(1-P_B)

sim = CPMGSimulator()
dw_n = DW_PPM * B0 * (sim.gamma_N/1e6)

def build_ours(buggy):
    """buggy=True re-introduces the two defects that were just fixed."""
    def blk(dw):
        L = sim.build_16x16(B0, TAU_M, TAU_E, S2, R_IS, R_EFF, CSA_N, THETA_N, J_IS, dw)
        if buggy:
            mu = L[11,14].real                    # current (correct) value = +mu
            piJ = np.pi*J_IS
            L[11,14] = L[14,11] = -mu             # old inverted sign
            L[12,13] = L[13,12] = mu
            L[11,14] -= piJ; L[14,11] += piJ      # old spurious J terms
            L[12,13] += piJ; L[13,12] -= piJ
        return L
    M = np.zeros((32,32), complex)
    M[:16,:16] = blk(0.0); M[16:,16:] = blk(dw_n)
    for i in range(1,16):
        M[i,i]-=kab; M[i,i+16]+=kba; M[i+16,i]+=kab; M[i+16,i+16]-=kba
    return np.real(M)

GOOD = build_ours(False)
LA = sim.build_16x16(B0, TAU_M, TAU_E, S2, R_IS, R_EFF, CSA_N, THETA_N, J_IS, 0.0)

basis = Basis(type="ixyzsxyz", spin_system="nh", model=ModelSpec.from_name("2st"))
spec = Spectrometer.from_spin_system(SpinSystem.from_name("1N-HN"), basis,
                                     Conditions(h_larmor_frq=H_MHZ))
spec.carrier_i = spec.carrier_s = 0.0
g = lambda a,b: -LA[a,b].real
spec.update({
    "r2_i_a": g(4,4), "r2_i_b": g(4,4), "r1_i_a": g(6,6), "r1_i_b": g(6,6),
    "r2_s_a": g(1,1), "r2_s_b": g(1,1), "r1_s_a": g(3,3), "r1_s_b": g(3,3),
    "r2a_i_a": g(9,9), "r2a_i_b": g(9,9), "r2a_s_a": g(7,7), "r2a_s_b": g(7,7),
    "r2mq_is_a": g(11,11), "r2mq_is_b": g(11,11),
    "r1a_is_a": g(15,15), "r1a_is_b": g(15,15),
    "etaxy_i_a": g(4,9), "etaxy_i_b": g(4,9),
    "etaz_i_a": g(6,15), "etaz_i_b": g(6,15),
    "etaxy_s_a": 0.0, "etaxy_s_b": 0.0, "etaz_s_a": 0.0, "etaz_s_b": 0.0,
    "sigma_is_a": g(6,3), "sigma_is_b": g(6,3),
    "mu_is_a": LA[11,14].real, "mu_is_b": LA[11,14].real,
    "j_is_a": J_IS, "j_is_b": J_IS,
    "cs_i_a": 0.0, "cs_i_b": dw_n/spec.ppm_i, "cs_s_a": 0.0, "cs_s_b": 0.0,
    "kab": kab, "kba": kba, "pa": 1-P_B, "pb": P_B,
})
CX = np.real(np.squeeze(np.asarray(spec._engine.l_free)))

CX2OURS = [4,5,6,1,2,3,9,10,7,8,11,13,12,14,15]
idx = CX2OURS + [i+16 for i in CX2OURS]
D_bad = np.abs(build_ours(True)[np.ix_(idx,idx)] - CX)
D_good = np.abs(GOOD[np.ix_(idx,idx)] - CX)

fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.0))
for ax, D, title, col in [
        (axes[0], D_bad,  "Before: 8 elements differ",  RUST),
        (axes[1], D_good, "After the two fixes",        GREEN)]:
    shown = np.where(D > 0, D, np.nan)
    im = ax.imshow(shown, cmap="autumn_r", norm=LogNorm(vmin=1e-3, vmax=1e3),
                   interpolation="nearest")
    ax.imshow(np.zeros_like(D), cmap="Greys", vmin=0, vmax=1, alpha=0.06)
    n = int((D > 1e-9).sum())
    ax.set_title(f"{title}\nmax |diff| = {D.max():.4g} s$^{{-1}}$  "
                 f"({n} of 900 elements)",
                 fontsize=18, color=col, pad=12, fontweight="bold")
    ax.set_xlabel("operator index"); ax.set_ylabel("operator index")
    ax.set_xticks([0, 15, 29]); ax.set_yticks([0, 15, 29])
    ax.axhline(14.5, color="#888", lw=1.2); ax.axvline(14.5, color="#888", lw=1.2)
    ax.text(7.2, 31.6, "state A", ha="center", fontsize=14, color="#666")
    ax.text(22.2, 31.6, "state B", ha="center", fontsize=14, color="#666")

cb = fig.colorbar(im, ax=axes, fraction=0.032, pad=0.02)
cb.set_label("|ours - ChemEx|  (s$^{-1}$)", fontsize=15)
cb.ax.tick_params(labelsize=13)
fig.suptitle("Our 16-operator Liouvillian vs ChemEx cpmg_hn_dq_zq  (30 x 30, "
             "identical inputs, no pulse sequence)", fontsize=17.5, y=0.99)
fig.savefig(f"{OUT}/gm_dqzq_matrix_match.png", dpi=170, facecolor="white",
            bbox_inches="tight")
print("saved. before max =", D_bad.max(), " after max =", D_good.max())
