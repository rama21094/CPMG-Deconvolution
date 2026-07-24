from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import scipy.constants as const
import scipy.linalg as la


DEFAULT_T_RELAX = 0.04
DEFAULT_NCYC_GRID = np.arange(1, 80, 2, dtype=np.int32)


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
        l_mat[11, 14] = l_mat[14, 11] = -mu_mq
        l_mat[12, 13] = l_mat[13, 12] = mu_mq

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
        l_mat[11, 14] -= pi_j
        l_mat[14, 11] += pi_j
        l_mat[12, 13] += pi_j
        l_mat[13, 12] -= pi_j

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

    def simulate_cpmg(
        self,
        params: Mapping[str, float],
        ncyc_range: np.ndarray = DEFAULT_NCYC_GRID,
        t_relax: float = DEFAULT_T_RELAX,
    ) -> CPMGProfile:
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

    def simulate_dej_pair(
        self,
        params: Mapping[str, float],
        ncyc_range: np.ndarray = DEFAULT_NCYC_GRID,
        t_relax: float = DEFAULT_T_RELAX,
    ) -> tuple[CPMGProfile, CPMGProfile]:
        with_j = self.simulate_cpmg(params, ncyc_range=ncyc_range, t_relax=t_relax)
        no_j_params = dict(params)
        no_j_params["J_IS"] = 0.0
        no_j = self.simulate_cpmg(no_j_params, ncyc_range=ncyc_range, t_relax=t_relax)
        return with_j, no_j
