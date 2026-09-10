"""Compare our 16-operator Liouvillian against ChemEx's cpmg_hn_dq_zq basis
('ixyzsxyz' = 15 operators per state). No pulse sequence involved -- this
compares the relaxation/exchange matrix equations only.

Our basis (index: label), I = 1H, S = 15N:
   0 E, 1 Hx, 2 Hy, 3 Hz, 4 Nx, 5 Ny, 6 Nz,
   7 2HxNz, 8 2HyNz, 9 2HzNx, 10 2HzNy,
   11 2HxNx, 12 2HxNy, 13 2HyNx, 14 2HyNy, 15 2HzNz

ChemEx 'ixyzsxyz' with spin_system='nh'  ->  i = 15N, s = 1H:
   ix iy iz sx sy sz 2ixsz 2iysz 2izsx 2izsy 2ixsx 2ixsy 2iysx 2iysy 2izsz

ChemEx has no identity operator in this basis, so our row/col 0 (E) -- which
only carries the R1 equilibrium-recovery source terms -- is dropped, leaving
15 per state = 30 x 30 on both sides.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
np.set_printoptions(linewidth=250, precision=5, suppress=True)

from cpmg_ml.simulator import CPMGSimulator, b0_mhz_to_tesla

from chemex.configuration.conditions import Conditions
from chemex.models.loader import register_kinetic_settings
from chemex.models.model import ModelSpec
from chemex.nmr.basis import Basis
from chemex.nmr.spectrometer import Spectrometer
from chemex.parameters.spin_system import SpinSystem

register_kinetic_settings()

# ---------------------------------------------------------------- parameters
H_MHZ = 600.0
B0 = b0_mhz_to_tesla(H_MHZ)
TAU_M, TAU_E, S2 = 5.0e-9, 50.0e-12, 0.85
R_IS, R_EFF = 1.02e-10, 1.86e-10
CSA_N, THETA_N = -160.0e-6, np.radians(22.0)
J_IS = 92.0
K_EX, P_B, DW_PPM = 300.0, 0.05, 2.0

kab, kba = K_EX * P_B, K_EX * (1.0 - P_B)

sim = CPMGSimulator()
dw_n = DW_PPM * B0 * (sim.gamma_N / 1.0e6)          # rad/s, our convention

# ---------------------------------------------------------------- our matrix
LA = sim.build_16x16(B0, TAU_M, TAU_E, S2, R_IS, R_EFF, CSA_N, THETA_N, J_IS, 0.0)
LB = sim.build_16x16(B0, TAU_M, TAU_E, S2, R_IS, R_EFF, CSA_N, THETA_N, J_IS, dw_n)
OURS = np.zeros((32, 32), dtype=complex)
OURS[0:16, 0:16] = LA
OURS[16:32, 16:32] = LB
for i in range(1, 16):
    OURS[i, i] -= kab
    OURS[i, i + 16] += kba
    OURS[i + 16, i] += kab
    OURS[i + 16, i + 16] -= kba
OURS = np.real(OURS)

# --------------------------------------------- extract rates from our matrix
# read each rate off the canonical position, so nothing is re-derived by hand
r2_i   = -LA[4, 4].real                      # Nx / Ny
r1_i   = -LA[6, 6].real                      # Nz
r2_s   = -LA[1, 1].real                      # Hx / Hy
r1_s   = -LA[3, 3].real                      # Hz
r2a_i  = -LA[9, 9].real                      # 2HzNx / 2HzNy  (antiphase 15N)
r2a_s  = -LA[7, 7].real                      # 2HxNz / 2HyNz  (antiphase 1H)
r2mq   = -LA[11, 11].real                    # multiple quantum
r1a_is = -LA[15, 15].real                    # 2HzNz
etaxy_i = -LA[4, 9].real                     # Nx <-> 2HzNx
etaz_i  = -LA[6, 15].real                    # Nz <-> 2HzNz
sigma   = -LA[6, 3].real                     # Nz <-> Hz  (NOE)
# our MQ block carries mu AND a pi*J term; separate them
mu_ours  =  0.5 * (LA[11, 14].real + LA[14, 11].real)
piJ_mq   =  0.5 * (LA[14, 11].real - LA[11, 14].real)

print("=" * 78)
print("RATES READ OFF OUR MATRIX")
print("=" * 78)
for n, v in [("r2_i (Nx)", r2_i), ("r1_i (Nz)", r1_i), ("r2_s (Hx)", r2_s),
             ("r1_s (Hz)", r1_s), ("r2a_i (2HzNx)", r2a_i), ("r2a_s (2HxNz)", r2a_s),
             ("r2mq", r2mq), ("r1a_is (2HzNz)", r1a_is), ("etaxy_i", etaxy_i),
             ("etaz_i", etaz_i), ("sigma_is", sigma), ("mu (MQ)", mu_ours)]:
    print(f"  {n:16s} = {v:12.6f} s-1")
print(f"  {'pi*J in MQ block':16s} = {piJ_mq:12.6f}   (pi*J would be {np.pi*J_IS:.6f})")
print(f"  {'etaxy_s / etaz_s':16s} = not present in our implementation")

# ------------------------------------------------------------ chemex matrix
basis = Basis(type="ixyzsxyz", spin_system="nh", model=ModelSpec.from_name("2st"))
spec = Spectrometer.from_spin_system(SpinSystem.from_name("1N-HN"), basis,
                                     Conditions(h_larmor_frq=H_MHZ))
spec.carrier_i = 0.0
spec.carrier_s = 0.0

# cs_i is supplied in ppm and scaled internally by ppm_i; choose the ppm value
# that reproduces our dw exactly, so the 0.03% gamma-vs-Xi convention
# difference does not contaminate the structural comparison.
cs_i_b_ppm = dw_n / spec.ppm_i

pars = {
    "r2_i_a": r2_i, "r2_i_b": r2_i,
    "r1_i_a": r1_i, "r1_i_b": r1_i,
    "r2_s_a": r2_s, "r2_s_b": r2_s,
    "r1_s_a": r1_s, "r1_s_b": r1_s,
    "r2a_i_a": r2a_i, "r2a_i_b": r2a_i,
    "r2a_s_a": r2a_s, "r2a_s_b": r2a_s,
    "r2mq_is_a": r2mq, "r2mq_is_b": r2mq,
    "r1a_is_a": r1a_is, "r1a_is_b": r1a_is,
    "etaxy_i_a": etaxy_i, "etaxy_i_b": etaxy_i,
    "etaz_i_a": etaz_i,  "etaz_i_b": etaz_i,
    "etaxy_s_a": 0.0, "etaxy_s_b": 0.0,   # absent from our implementation
    "etaz_s_a": 0.0,  "etaz_s_b": 0.0,
    "sigma_is_a": sigma, "sigma_is_b": sigma,
    "mu_is_a": mu_ours, "mu_is_b": mu_ours,
    "j_is_a": J_IS, "j_is_b": J_IS,
    "cs_i_a": 0.0, "cs_i_b": cs_i_b_ppm,
    "cs_s_a": 0.0, "cs_s_b": 0.0,
    "kab": kab, "kba": kba,
    "pa": 1.0 - P_B, "pb": P_B,
}
spec.update(pars)
CX = np.real(np.squeeze(np.asarray(spec._engine.l_free)))

# ------------------------------------------------------- align the two bases
# ChemEx operator order -> our index (within one state block)
CX2OURS = [4, 5, 6, 1, 2, 3, 9, 10, 7, 8, 11, 13, 12, 14, 15]
LABELS = ["Nx", "Ny", "Nz", "Hx", "Hy", "Hz", "2HzNx", "2HzNy", "2HxNz", "2HyNz",
          "2HxNx", "2HyNx", "2HxNy", "2HyNy", "2HzNz"]
idx = [i for i in CX2OURS] + [i + 16 for i in CX2OURS]
NAMES = [f"{l}_A" for l in LABELS] + [f"{l}_B" for l in LABELS]

OURS30 = OURS[np.ix_(idx, idx)]

print()
print("=" * 78)
print("ELEMENT-BY-ELEMENT COMPARISON  (30 x 30)")
print("=" * 78)
D = OURS30 - CX
print(f"max |ours - chemex| = {np.abs(D).max():.6e}")

tol = 1e-9
bad = np.argwhere(np.abs(D) > tol)
if len(bad) == 0:
    print("\n*** THE TWO MATRICES ARE IDENTICAL ***")
else:
    print(f"\n{len(bad)} differing elements:\n")
    print(f"{'row':>10}{'col':>10}{'ours':>14}{'chemex':>14}{'diff':>14}")
    seen = set()
    for r, c in bad:
        key = (NAMES[r], NAMES[c])
        if key in seen:
            continue
        seen.add(key)
        print(f"{NAMES[r]:>10}{NAMES[c]:>10}{OURS30[r,c]:14.5f}{CX[r,c]:14.5f}{D[r,c]:14.5f}")

# Structural equality with shared rates, not independent validation of rates.
np.testing.assert_allclose(OURS30, CX, rtol=0, atol=tol)
