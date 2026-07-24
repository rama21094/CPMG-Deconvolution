"""CW-decoupled 15N CPMG relaxation-dispersion simulator.

Implements the pulse scheme of Hansen, Vallurupalli & Kay,
J. Phys. Chem. B 2008, 112, 5898-5904 ("CW CPMG"): a continuous 1H
decoupling field is held on throughout T_relax so 15N magnetization
stays in-phase the whole time (instead of exchanging with anti-phase
coherence via the 1H-15N scalar coupling, as in a plain CPMG train),
and one extra 15N 180 degree pulse (phase x) sits at the exact center
of the train. That extra pulse makes the whole element self-compensating
for 15N pulse-length errors regardless of whether the echo count on
each side of center is odd or even.

This module does not modify cpmg_ml/simulator.py. It reuses
CPMGSimulator.build_16x16 (the free-precession + relaxation + two-site
exchange physics, unchanged) and only adds new RF-pulse logic and a
new pulse-train timing function for this specific experiment.

Basis and RF-Liouvillian derivation
------------------------------------
CPMGSimulator.build_16x16 implements the 16-operator Cartesian
product-operator basis and relaxation superoperator of:

    Allard, Helgstrand & Hard, "The Complete Homogeneous Master
    Equation for a Heteronuclear Two-Spin System in the Basis of
    Cartesian Product Operators", J. Magn. Reson. 134, 7-16 (1998).

In that basis (I = 1H, S = 15N):

    0=E, 1=Hx, 2=Hy, 3=Hz, 4=Nx, 5=Ny, 6=Nz,
    7=2HxNz, 8=2HyNz, 9=2HzNx, 10=2HzNy,
    11=2HxNx, 12=2HxNy, 13=2HyNx, 14=2HyNy, 15=2HzNz

CPMGSimulator.build_rf_liouvillian only implements an x-phase pulse on
the S (15N) spin. The CW-CPMG scheme needs pulses (and a continuous
field) on both spins at both x and y phase, so build_rf_liouvillian_spin
below generalizes it. The sign convention (which index of each
z-coupled pair gets +field vs -field) was fixed by numerically testing
CPMGSimulator.build_rf_liouvillian's existing N-x-phase behavior
(confirmed: a 90 degree N-x pulse rotates Nz -> +Ny) and then deriving
the other three cases (N-y, H-x, H-y) from Allard et al.'s own
Liouvillian matrix (their Eq. 19) using the same handedness, so all
four cases are mutually consistent with the one convention this
codebase already uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import scipy.linalg as la

from .simulator import (
    DEFAULT_NCYC_GRID,
    DEFAULT_T_RELAX,
    CPMGProfile,
    CPMGSimulator,
)

# (index_a, index_b) pairs that rotate into each other under a hard pulse,
# plus the sign of the +field entry: L[a, b] = sign * w1, L[b, a] = -sign * w1.
_RF_PAIRS: dict[tuple[str, str], tuple[list[tuple[int, int]], int]] = {
    ("N", "x"): ([(5, 6), (12, 7), (14, 8), (10, 15)], +1),
    ("N", "y"): ([(4, 6), (11, 7), (13, 8), (9, 15)], -1),
    ("H", "x"): ([(2, 3), (8, 15), (13, 9), (14, 10)], +1),
    ("H", "y"): ([(1, 3), (7, 15), (11, 9), (12, 10)], -1),
}


class CWCPMGSimulator(CPMGSimulator):
    """CPMGSimulator plus the CW-CPMG (1H-decoupled) pulse-train logic."""

    def build_rf_liouvillian_spin(
        self, spin: str, phase: str, field_hz: float
    ) -> np.ndarray:
        """32x32 (two-site-exchange-doubled) RF Liouvillian for one hard
        pulse or continuous field on one spin, at one phase.

        spin: "H" (proton, I) or "N" (nitrogen, S)
        phase: "x" or "y"
        """
        key = (spin, phase)
        if key not in _RF_PAIRS:
            raise ValueError(f"unsupported spin/phase combination: {spin!r}/{phase!r}")
        pairs, sign = _RF_PAIRS[key]

        l_rf = np.zeros((32, 32), dtype=complex)
        w1 = 2.0 * np.pi * field_hz
        for offset in (0, 16):
            for a, b in pairs:
                l_rf[offset + a, offset + b] = sign * w1
                l_rf[offset + b, offset + a] = -sign * w1
        return l_rf

    def simulate_cw_cpmg(
        self,
        params: Mapping[str, float],
        ncyc_range: np.ndarray = DEFAULT_NCYC_GRID,
        t_relax: float = DEFAULT_T_RELAX,
        b1_h_cw_hz: float = 13000.0,
    ) -> CPMGProfile:
        """Simulate the CW-CPMG relaxation-dispersion profile.

        params: same physical-parameter mapping as CPMGSimulator.simulate_cpmg
            (B0, tau_m, tau_e, S2, r_IS, r_eff, csa_N, theta_N, J_IS, k_ex,
            p_B, dw_N, B1_N).
        b1_h_cw_hz: strength of the continuous 1H decoupling field (Hz),
            held on for the entire T_relax period. The Hansen/Kay paper
            quantizes this to an integer multiple of 2*nu_CPMG for real
            hardware reasons (so the decoupling pulses themselves cannot
            transfer magnetization at any CPMG rate); this exact-numerical
            simulator instead applies a true continuous field, which the
            quantization was only ever an approximation to.
        """
        b0 = params["B0"]
        tau_m = params["tau_m"]
        tau_e = params["tau_e"]
        s2 = params["S2"]
        r_is = params["r_IS"]
        r_eff = params["r_eff"]
        csa_n = params["csa_N"]
        theta_n = params["theta_N"]
        j_is = params["J_IS"]
        k_ex = params["k_ex"]
        p_b = params["p_B"]
        dw_n_ppm = params["dw_N"]
        b1_n_hz = params["B1_N"]

        dw_n = dw_n_ppm * b0 * (self.gamma_N / 1.0e6)
        p_a = 1.0 - p_b
        k_ab = k_ex * p_b
        k_ba = k_ex * p_a

        l_free = np.zeros((32, 32), dtype=complex)
        l_a = self.build_16x16(b0, tau_m, tau_e, s2, r_is, r_eff, csa_n, theta_n, j_is, 0.0)
        l_b = self.build_16x16(b0, tau_m, tau_e, s2, r_is, r_eff, csa_n, theta_n, j_is, dw_n)
        l_free[0:16, 0:16] = l_a
        l_free[16:32, 16:32] = l_b
        for i in range(1, 16):
            l_free[i, i] -= k_ab
            l_free[i, i + 16] += k_ba
            l_free[i + 16, i] += k_ab
            l_free[i + 16, i + 16] -= k_ba

        # Continuous 1H CW decoupling field, held on for the entire T_relax
        # period (including through the 15N pulses).
        l_h_cw = self.build_rf_liouvillian_spin("H", "x", b1_h_cw_hz)
        l_free_cw = l_free + l_h_cw

        t_180 = 1.0 / (2.0 * b1_n_hz)
        l_n_y = self.build_rf_liouvillian_spin("N", "y", b1_n_hz)
        l_n_x = self.build_rf_liouvillian_spin("N", "x", b1_n_hz)
        u_pulse_y = la.expm((l_free_cw + l_n_y) * t_180)
        u_pulse_x_center = la.expm((l_free_cw + l_n_x) * t_180)

        rho_0 = np.zeros(32)
        rho_0[0] = 1.0
        rho_0[16] = 1.0
        rho_0[4] = p_a
        rho_0[20] = p_b

        nu_cp_list = []
        r2_eff_list = []
        skipped_ncyc = []

        for ncyc_value in ncyc_range:
            ncyc = int(ncyc_value)
            # ncyc train (y-phase) pulses + 1 extra central (x-phase) pulse.
            total_pulses = ncyc + 1
            total_pulse_time = total_pulses * t_180
            if total_pulse_time >= t_relax:
                skipped_ncyc.append(ncyc)
                continue

            tau_cp = (t_relax - total_pulse_time) / (2.0 * total_pulses)
            delta = 2.0 * tau_cp + 2.0 * t_180
            nu_cp = 1.0 / (2.0 * delta)

            u_tau = la.expm(l_free_cw * tau_cp)
            u_train_block = u_tau @ u_pulse_y @ u_tau
            u_center_block = u_tau @ u_pulse_x_center @ u_tau

            n1 = (ncyc + 1) // 2  # train pulses before center (ncyc split as evenly as possible)
            n2 = ncyc - n1        # train pulses after center

            u_total = (
                np.linalg.matrix_power(u_train_block, n2)
                @ u_center_block
                @ np.linalg.matrix_power(u_train_block, n1)
            )

            rho_t = u_total @ rho_0
            # A y-phase 180 pulse inverts Nx (rotation about y flips the
            # perpendicular x/z components), so an odd total pulse count
            # flips the sign of the recovered in-phase magnetization. Real
            # spectrometers report a magnitude/phase-sensitive intensity
            # that is always positive regardless of this sign, so take the
            # absolute value here rather than treating it as a decay.
            mag_x = np.abs(np.real(rho_t[4] + rho_t[20]))
            normalized_mag = mag_x / (p_a + p_b)
            if normalized_mag <= 0.0 or not np.isfinite(normalized_mag):
                r2_eff = np.nan
            else:
                r2_eff = -np.log(normalized_mag) / t_relax

            nu_cp_list.append(nu_cp)
            r2_eff_list.append(r2_eff)

        return CPMGProfile(
            nu_cp=np.asarray(nu_cp_list, dtype=np.float64),
            r2_eff=np.asarray(r2_eff_list, dtype=np.float64),
            skipped_ncyc=tuple(skipped_ncyc),
        )
