from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import scipy.constants as const
import scipy.linalg as la


DEFAULT_T_RELAX = 0.04
DEFAULT_NCYC_GRID = np.arange(1, 80, 2, dtype=np.int32)

# (index_a, index_b) pairs that rotate into each other under a hard pulse on
# one spin, plus the sign of the +field entry: L[a,b] = sign*w1, L[b,a] = -sign*w1.
# Basis order: 0=E,1=Hx,2=Hy,3=Hz,4=Nx,5=Ny,6=Nz,7=2HxNz,8=2HyNz,9=2HzNx,
# 10=2HzNy,11=2HxNx,12=2HxNy,13=2HyNx,14=2HyNy,15=2HzNz. Sign convention fixed
# by testing the existing N-x-phase behavior below (a 90 N-x pulse rotates
# Nz -> +Ny) and deriving the other three cases with the same handedness.
_RF_PAIRS: dict[tuple[str, str], tuple[list[tuple[int, int]], int]] = {
    ("N", "x"): ([(5, 6), (12, 7), (14, 8), (10, 15)], +1),
    ("N", "y"): ([(4, 6), (11, 7), (13, 8), (9, 15)], -1),
    ("H", "x"): ([(2, 3), (8, 15), (13, 9), (14, 10)], +1),
    ("H", "y"): ([(1, 3), (7, 15), (11, 9), (12, 10)], -1),
}


def b0_mhz_to_tesla(b0_mhz: float, gamma_h: float = 267.522e6) -> float:
    """Convert proton spectrometer frequency in MHz to magnetic field in Tesla."""
    return float(2.0 * np.pi * b0_mhz * 1.0e6 / gamma_h)


@dataclass(frozen=True)
class CPMGProfile:
    nu_cp: np.ndarray
    r2_eff: np.ndarray
    skipped_ncyc: tuple[int, ...]


class CPMGSimulator:
    """Headless CPMG simulator extracted from the Tk GUI script."""

    def __init__(self):
        self.mu_0 = const.mu_0
        self.hbar = const.hbar
        self.gamma_H = 267.522e6
        self.gamma_N = -27.116e6

    def j_comp(self, omega: float, tau_m: float, tau_e: float, s2: float) -> float:
        tau_inv = (1.0 / tau_m) + (1.0 / tau_e)
        tau = 1.0 / tau_inv
        term1 = (s2 * tau_m) / (1.0 + (omega * tau_m) ** 2)
        term2 = ((1.0 - s2) * tau) / (1.0 + (omega * tau) ** 2)
        return (2.0 / 5.0) * (term1 + term2)

    def build_16x16(
        self,
        b0: float,
        tau_m: float,
        tau_e: float,
        s2: float,
        r_is: float,
        r_eff: float,
        csa_n: float,
        theta_n: float,
        j_is: float,
        dw_offset: float,
    ) -> np.ndarray:
        l_mat = np.zeros((16, 16), dtype=complex)

        omega_h = -self.gamma_H * b0
        omega_n = -self.gamma_N * b0

        j0 = self.j_comp(0.0, tau_m, tau_e, s2)
        jh = self.j_comp(omega_h, tau_m, tau_e, s2)
        jn = self.j_comp(omega_n, tau_m, tau_e, s2)
        jh_minus_n = self.j_comp(omega_h - omega_n, tau_m, tau_e, s2)
        jh_plus_n = self.j_comp(omega_h + omega_n, tau_m, tau_e, s2)
        j2h = self.j_comp(2.0 * omega_h, tau_m, tau_e, s2)

        d2 = (self.mu_0 / (4.0 * np.pi)) ** 2 * (
            self.hbar * self.gamma_H * self.gamma_N / r_is**3
        ) ** 2
        d2_h = (self.mu_0 / (4.0 * np.pi)) ** 2 * (
            self.hbar * self.gamma_H**2 / r_eff**3
        ) ** 2
        c2_n = (1.0 / 3.0) * (omega_n * csa_n) ** 2

        p2_cos = 0.5 * (3.0 * np.cos(theta_n) ** 2 - 1.0)
        # sqrt(3) corrects a missing factor versus Allard, Helgstrand & Hard
        # (1998) Eq. 33-34: Ad*Ac = sqrt(9*d2 * 3*c2_n) = sqrt(3)*sqrt(d2*c2_n)*3,
        # so dc must include sqrt(3) for eta_z/eta_xy to match delta_S/eta_S.
        dc = np.sqrt(3.0) * np.sqrt(d2) * np.sqrt(c2_n) * p2_cos

        rho_h = (d2_h / 4.0) * (j0 + 3.0 * jh + 6.0 * j2h)
        lambda_h = (d2_h / 8.0) * (5.0 * j0 + 9.0 * jh + 6.0 * j2h)

        r1_n = (d2 / 4.0) * (jh_minus_n + 3.0 * jn + 6.0 * jh_plus_n) + c2_n * jn
        r2_n = (d2 / 8.0) * (
            4.0 * j0 + jh_minus_n + 3.0 * jn + 6.0 * jh + 6.0 * jh_plus_n
        ) + (c2_n / 6.0) * (4.0 * j0 + 3.0 * jn)

        r1_h = (d2 / 4.0) * (jh_minus_n + 3.0 * jh + 6.0 * jh_plus_n) + rho_h
        r2_h = (d2 / 8.0) * (
            4.0 * j0 + jh_minus_n + 3.0 * jh + 6.0 * jn + 6.0 * jh_plus_n
        ) + lambda_h

        r2_sz_nx = (
            (d2 / 8.0)
            * (4.0 * j0 + jh_minus_n + 3.0 * jn + 6.0 * jh + 6.0 * jh_plus_n)
            + (d2 / 4.0) * (jh_minus_n + 3.0 * jh + 6.0 * jh_plus_n)
            + (c2_n / 6.0) * (4.0 * j0 + 3.0 * jn)
            + rho_h
        )
        r2_hz_nx = (
            (d2 / 8.0)
            * (4.0 * j0 + jh_minus_n + 3.0 * jh + 6.0 * jn + 6.0 * jh_plus_n)
            + (d2 / 4.0) * (jh_minus_n + 3.0 * jn + 6.0 * jh_plus_n)
            + (c2_n / 3.0) * jn
            + lambda_h
        )

        r_mq = (d2 / 8.0) * (
            jh_minus_n + 3.0 * jh + 3.0 * jn + 6.0 * jh_plus_n
        ) + (c2_n / 6.0) * (4.0 * j0 + 3.0 * jn) + lambda_h
        r1_is2sp = (d2 / 4.0) * (3.0 * jn + 3.0 * jh) + (c2_n / 3.0) * jn + rho_h

        eta_z = dc * jn
        eta_xy = (dc / 6.0) * (4.0 * j0 + 3.0 * jn)

        sigma = (d2 / 4.0) * (-jh_minus_n + 6.0 * jh_plus_n)
        mu_mq = (d2 / 4.0) * (-0.5 * jh_minus_n + 3.0 * jh_plus_n)

        m_h0 = 1.0
        m_n0 = self.gamma_N / self.gamma_H
        theta_i = r1_h * m_h0 + sigma * m_n0
        theta_s = sigma * m_h0 + r1_n * m_n0
        theta_is = eta_z * m_n0

        l_mat[3, 0] = 2.0 * theta_i
        l_mat[6, 0] = 2.0 * theta_s
        l_mat[15, 0] = 2.0 * theta_is

        l_mat[1, 1] = l_mat[2, 2] = -r2_h
        l_mat[3, 3] = -r1_h
        l_mat[4, 4] = l_mat[5, 5] = -r2_n
        l_mat[6, 6] = -r1_n
        l_mat[7, 7] = l_mat[8, 8] = -r2_hz_nx
        l_mat[9, 9] = l_mat[10, 10] = -r2_sz_nx
        l_mat[11, 11] = l_mat[12, 12] = l_mat[13, 13] = l_mat[14, 14] = -r_mq
        l_mat[15, 15] = -r1_is2sp

        l_mat[3, 6] = l_mat[6, 3] = -sigma
        l_mat[4, 9] = l_mat[9, 4] = -eta_xy
        l_mat[5, 10] = l_mat[10, 5] = -eta_xy
        l_mat[6, 15] = l_mat[15, 6] = -eta_z
        # Allard Eq. [19] is written as  d(sigma)/dt = -P * sigma,  so every
        # element of the printed matrix must be negated on the way into l_mat.
        # The printed matrix has P[2IxSx,2IySy] = -mu_mq and P[2IxSy,2IySx] =
        # +mu_mq, hence l_mat gets +mu_mq and -mu_mq respectively. These two
        # lines previously carried the printed signs un-negated, which swapped
        # the DQ and ZQ relaxation rates (R_DQ must exceed R_ZQ, since
        # mu_mq = (Ad^2/36)[-J(wI-wS)/2 + 3J(wI+wS)] > 0).
        l_mat[11, 14] = l_mat[14, 11] = mu_mq
        l_mat[12, 13] = l_mat[13, 12] = -mu_mq

        l_mat[4, 5] = -dw_offset
        l_mat[5, 4] = dw_offset
        l_mat[9, 10] = -dw_offset
        l_mat[10, 9] = dw_offset
        l_mat[11, 12] = -dw_offset
        l_mat[12, 11] = dw_offset
        l_mat[13, 14] = -dw_offset
        l_mat[14, 13] = dw_offset

        pi_j = np.pi * j_is
        l_mat[1, 8] = -pi_j
        l_mat[8, 1] = pi_j
        l_mat[2, 7] = pi_j
        l_mat[7, 2] = -pi_j
        l_mat[4, 10] = -pi_j
        l_mat[10, 4] = pi_j
        l_mat[5, 9] = pi_j
        l_mat[9, 5] = -pi_j
        # No J-coupling terms in the multiple-quantum block. The scalar-coupling
        # Hamiltonian 2*pi*J*IzSz commutes exactly with all four MQ operators
        # (2IxSx, 2IxSy, 2IySx, 2IySy), so J cannot connect them -- DQ and ZQ
        # coherences are not split by J in a two-spin system. Allard Eq. [19]
        # correspondingly shows no pi*J entries in those four rows. Four lines
        # adding +/-pi_j to [11,14], [14,11], [12,13], [13,12] were removed here.

        return l_mat

    def build_rf_liouvillian(self, b1_n_hz: float) -> np.ndarray:
        l_rf = np.zeros((32, 32), dtype=complex)
        w_sx = 2.0 * np.pi * b1_n_hz

        for offset in (0, 16):
            l_rf[offset + 5, offset + 6] = w_sx
            l_rf[offset + 6, offset + 5] = -w_sx
            l_rf[offset + 12, offset + 7] = w_sx
            l_rf[offset + 7, offset + 12] = -w_sx
            l_rf[offset + 14, offset + 8] = w_sx
            l_rf[offset + 8, offset + 14] = -w_sx
            l_rf[offset + 10, offset + 15] = w_sx
            l_rf[offset + 15, offset + 10] = -w_sx

        return l_rf

    def build_rf_liouvillian_spin(
        self, spin: str, phase: str, field_hz: float
    ) -> np.ndarray:
        """32x32 RF Liouvillian for one hard pulse or continuous field on one
        spin, at one phase. spin: "H" (1H) or "N" (15N). phase: "x" or "y".

        build_rf_liouvillian(b1_n_hz) above is the special case
        build_rf_liouvillian_spin("N", "x", b1_n_hz); a -x phase pulse is
        simply the negative of the x-phase Liouvillian (H_rf flips sign).
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

    def simulate_cpmg(
        self,
        params: Mapping[str, float],
        ncyc_range: np.ndarray = DEFAULT_NCYC_GRID,
        t_relax: float = DEFAULT_T_RELAX,
        sequence: str = "home",
    ) -> CPMGProfile:
        """Simulate a CPMG relaxation-dispersion profile.

        sequence: "home" (default) is this simulator's own idealized train --
            magnetization starts in-phase on Nx, evolves through ncyc plain
            (tau-180x-tau) blocks over t_relax, and Nx is read directly at the
            end. "chemex" instead runs ChemEx's cpmg_15n_ip pulse sequence
            (see simulate_cpmg_chemex_sequence) through this simulator's full
            32x32 physics, for a same-sequence, physics-only comparison against
            ChemEx. NOTE: ncyc has a DIFFERENT meaning for each sequence --
            for "home" it is the total refocusing-pulse count; for "chemex" it
            is ChemEx's own convention (pulses per half-train, so the real
            total is 2*ncyc+1) and t_relax is passed through as ChemEx's
            time_t2. Do not convert between them implicitly; pick the ncyc
            grid to match whichever sequence you select.
        """
        if sequence == "chemex":
            return self.simulate_cpmg_chemex_sequence(
                params, ncyc_range=ncyc_range, time_t2=t_relax
            )
        if sequence != "home":
            raise ValueError(f"unknown sequence {sequence!r}; expected 'home' or 'chemex'")
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

        l_rf = self.build_rf_liouvillian(b1_n_hz)
        l_pulse = l_free + l_rf

        t_180 = 1.0 / (2.0 * b1_n_hz)
        u_pulse = la.expm(l_pulse * t_180)

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
            total_pulse_time = ncyc * t_180
            if total_pulse_time >= t_relax:
                skipped_ncyc.append(ncyc)
                continue

            tau_cp = (t_relax - total_pulse_time) / (2.0 * ncyc)
            nu_cp = 1.0 / (4.0 * tau_cp)
            u_tau = la.expm(l_free * tau_cp)
            u_block = u_tau @ u_pulse @ u_tau
            u_total = np.linalg.matrix_power(u_block, ncyc)

            rho_t = u_total @ rho_0
            mag_x = np.real(rho_t[4] + rho_t[20])
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

    def simulate_cpmg_chemex_sequence(
        self,
        params: Mapping[str, float],
        ncyc_range: np.ndarray = DEFAULT_NCYC_GRID,
        time_t2: float = DEFAULT_T_RELAX,
        time_equil: float = 2.0e-3,
    ) -> CPMGProfile:
        """Simulate CPMG relaxation dispersion using ChemEx's own cpmg_15n_ip
        pulse sequence (chemex/experiments/catalog/cpmg_15n_ip.py,
        Cpmg15NIpSequence.calculate), propagated through this simulator's full
        32x32 Allard-1998 physics instead of ChemEx's reduced 6x6 {Nx,Ny,Nz} x
        {A,B} basis. Comparing this against a real ChemEx run isolates genuine
        physics differences (proton bath, CSA-DD cross-correlation, J-coupling)
        from pulse-sequence/timing-convention differences, since both sides
        then use the identical sequence definition.

        Sequence (thermal equilibrium starts on Nz_A + Nz_B, matching
        ChemEx's start term "iz"; detection reads Nz_A only, matching
        ChemEx's detection term "[iz_a]"):

            Nz --delay(t_neg)--> p90x -->
                [ delay(tau_cp) -> p180y -> delay(tau_cp) ]^ncyc -->
                p180pmx (average of +x/-x 180, ChemEx's phase-cycled
                pulse-imperfection self-compensation) -->
                [ delay(tau_cp) -> p180y -> delay(tau_cp) ]^ncyc -->
                p90x --> delay(t_neg) --> delay(time_equil) --> detect Nz_A

        ncyc here is ChemEx's own convention: pulses PER HALF-TRAIN (real
        total refocusing-pulse count is 2*ncyc + 1). tau_cp = time_t2/(4*ncyc)
        - pw90, and the reported nu_cpmg = ncyc/time_t2, exactly as in ChemEx.
        t_neg = -2*pw90/pi is ChemEx's finite-pulse-width correction; it is a
        NEGATIVE-duration delay, which is simply the inverse propagator
        (expm of a negative time), not a special case numerically.
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

        pw90 = 1.0 / (4.0 * b1_n_hz)
        t_180 = 1.0 / (2.0 * b1_n_hz)
        t_neg = -2.0 * pw90 / np.pi

        l_rf_x = self.build_rf_liouvillian_spin("N", "x", b1_n_hz)
        l_rf_y = self.build_rf_liouvillian_spin("N", "y", b1_n_hz)

        u_90x = la.expm((l_free + l_rf_x) * pw90)
        u_180y = la.expm((l_free + l_rf_y) * t_180)
        u_180x = la.expm((l_free + l_rf_x) * t_180)
        u_180negx = la.expm((l_free - l_rf_x) * t_180)
        u_180pmx = 0.5 * (u_180x + u_180negx)

        u_neg = la.expm(l_free * t_neg)
        u_equil = la.expm(l_free * time_equil)

        rho_0 = np.zeros(32)
        rho_0[0] = 1.0
        rho_0[16] = 1.0
        rho_0[6] = p_a
        rho_0[22] = p_b

        def detect(rho: np.ndarray) -> float:
            # ChemEx's detection operator is "[iz_a]" only (suffix_detect="_a"
            # in Cpmg15NIpSettings) -- it reads the ground-state (A) Nz
            # component alone, NOT the population-weighted sum over both
            # exchanging states. Summing both (as the "home" sequence's
            # from-Nx observable does) silently breaks the detected signal's
            # sensitivity to several pulse/offset sign conventions and was
            # verified numerically to cause a ~5% R2,eff error at low ncyc
            # against a real ChemEx run; reading index 6 alone reproduces
            # ChemEx to ~0.1% (residual is the ~0.03% gammaN/gammaH vs IUPAC
            # Xi-ratio difference in dw, not a sequence bug).
            return float(np.real(rho[6]))

        i0 = detect(u_equil @ u_90x @ u_180pmx @ (u_90x @ rho_0))

        part1 = u_neg @ (u_90x @ rho_0)
        u_part2 = u_equil @ u_90x @ u_neg

        nu_cp_list = []
        r2_eff_list = []
        skipped_ncyc = []

        for ncyc_value in ncyc_range:
            ncyc = int(ncyc_value)
            tau_cp = time_t2 / (4.0 * ncyc) - pw90
            if tau_cp <= 0.0:
                skipped_ncyc.append(ncyc)
                continue

            u_tau = la.expm(l_free * tau_cp)
            echo = u_tau @ u_180y @ u_tau
            cpmg = np.linalg.matrix_power(echo, ncyc)

            rho_t = u_part2 @ (cpmg @ (u_180pmx @ (cpmg @ part1)))
            intensity = detect(rho_t)
            normalized = intensity / i0
            if normalized <= 0.0 or not np.isfinite(normalized):
                r2_eff = np.nan
            else:
                r2_eff = -np.log(normalized) / time_t2

            nu_cp_list.append(ncyc / time_t2)
            r2_eff_list.append(r2_eff)

        return CPMGProfile(
            nu_cp=np.asarray(nu_cp_list, dtype=np.float64),
            r2_eff=np.asarray(r2_eff_list, dtype=np.float64),
            skipped_ncyc=tuple(skipped_ncyc),
        )

    def simulate_dej_pair(
        self,
        params: Mapping[str, float],
        ncyc_range: np.ndarray = DEFAULT_NCYC_GRID,
        t_relax: float = DEFAULT_T_RELAX,
        sequence: str = "home",
    ) -> tuple[CPMGProfile, CPMGProfile]:
        """Matched (J-coupled, J-free) profile pair for de-J-coupling training.

        Both profiles use the same pulse sequence and the same ncyc grid, so the
        only difference between them is J_IS. sequence="chemex" runs ChemEx's
        cpmg_15n_ip sequence, which is what a real spectrometer executes;
        remember that its ncyc convention differs from "home" (pulses per
        half-train rather than total pulse count), so the ncyc grid should be
        chosen for whichever sequence is selected.
        """
        with_j = self.simulate_cpmg(
            params, ncyc_range=ncyc_range, t_relax=t_relax, sequence=sequence
        )
        no_j_params = dict(params)
        no_j_params["J_IS"] = 0.0
        no_j = self.simulate_cpmg(
            no_j_params, ncyc_range=ncyc_range, t_relax=t_relax, sequence=sequence
        )
        return with_j, no_j
