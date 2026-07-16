# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 23:58:31 2025

@author: shank
"""

#!/usr/bin/env python3
# cpmg_sim.py
# Works in Spyder IDE or from the CLI. No ipywidgets required.

import argparse
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

# -----------------------------
# Bloch–McConnell core
# -----------------------------

def create_evolution_matrix(R2A, R2B, R1A, R1B, wA, wB, kAB, kBA):
    """
    Creates the 6x6 Bloch–McConnell evolution matrix (L) for free precession.
    Order of magnetization vector: [IxA, IyA, IzA, IxB, IyB, IzB]
    """
    # Relaxation
    R = np.diag([-R2A, -R2A, -R1A, -R2B, -R2B, -R1B])

    # Precession (chemical shifts, rad/s)
    Omega = np.zeros((6, 6))
    # Site A
    Omega[0, 1] = -wA
    Omega[1, 0] =  wA
    # Site B
    Omega[3, 4] = -wB
    Omega[4, 3] =  wB

    # Chemical exchange
    K = np.zeros((6, 6))
    # A -> B
    K[0:3, 0:3] -= kAB * np.eye(3)
    K[3:6, 0:3] += kAB * np.eye(3)
    # B -> A
    K[3:6, 3:6] -= kBA * np.eye(3)
    K[0:3, 3:6] += kBA * np.eye(3)

    return R + Omega + K


def simulate_cpmg_profile(kex=500.0, pB=0.02, delta_nu=150.0, R2_base=10.0, R1=2.0,
                          Tex_ms=40.0, fmin=50.0, fmax=2000.0, npoints=40):
    """
    Numerically simulates a CPMG dispersion using Bloch–McConnell formalism.

    Parameters
    ----------
    kex : float
        Exchange rate (s^-1), kex = kAB + kBA
    pB : float
        Minor-state population (0 < pB < 1), pA = 1 - pB
    delta_nu : float
        Chemical shift difference between sites (Hz)
    R2_base : float
        Intrinsic transverse relaxation rate R2 for both sites (s^-1)
    R1 : float
        Longitudinal relaxation rate R1 for both sites (s^-1)
    Tex_ms : float
        Total CPMG time in milliseconds (converted to seconds internally)
    fmin, fmax : float
        Min/Max CPMG frequencies in Hz
    npoints : int
        Number of CPMG frequency points (log-spaced)

    Returns
    -------
    nu_cpmg : ndarray, shape (npoints,)
    R2eff : ndarray, shape (npoints,)
    """
    # Exchange partitioning
    pA = 1.0 - pB
    kAB = kex * pB
    kBA = kex * pA

    # Chemical shifts in rad/s
    delta_w = delta_nu * 2.0 * np.pi
    wA = 0.0
    wB = delta_w

    # Relaxation rates
    R2A = R2_base
    R2B = R2_base
    R1A = R1
    R1B = R1

    # Frequencies and delays
    nu_cpmg = np.logspace(np.log10(fmin), np.log10(fmax), int(npoints))
    tau = 1.0 / (4.0 * nu_cpmg)  # seconds
    Tex = Tex_ms / 1000.0        # convert ms -> s

    # Initial magnetization (equilibrium along +z, then 90x to -y)
    M0 = np.array([0.0, 0.0, pA, 0.0, 0.0, pB], dtype=float)
    M_initial = np.array([0.0, -pA, 0.0, 0.0, -pB, 0.0], dtype=float)

    # 180°_y pulse block (applies to each site independently)
    P180 = np.array([[0, 1, 0],
                     [1, 0, 0],
                     [0, 0, -1]], dtype=float)
    P180_6 = np.zeros((6, 6), dtype=float)
    P180_6[0:3, 0:3] = P180
    P180_6[3:6, 3:6] = P180

    # Precompute L (independent of tau)
    L = create_evolution_matrix(R2A, R2B, R1A, R1B, wA, wB, kAB, kBA)

    R2eff = np.zeros_like(nu_cpmg, dtype=float)

    # Numerical stability helpers
    eps = 1e-15
    I_initial = M_initial[1] + M_initial[4]
    if np.isclose(I_initial, 0.0):
        # Shouldn't happen with our choice, but guard anyway
        I_initial = np.sign(I_initial) * max(abs(I_initial), eps) if I_initial != 0 else eps

    for i, t in enumerate(tau):
        # Number of echoes N ≈ Tex / (2*tau); each CPMG block has two echoes
        N_echos = int(np.floor(Tex / (2.0 * t)))
        N_blocks = max(1, N_echos // 2)  # ensure at least one block for numerical meaning

        # Free precession over tau
        E = expm(L * t)

        # One CPMG block: E – 180y – E
        G = E @ P180_6 @ E

        # Total propagator
        Gtot = np.linalg.matrix_power(G, int(N_blocks))

        M_final = Gtot @ M_initial
        I_final = M_final[1] + M_final[4]

        # Use magnitude to avoid negative ratios; clip to eps to avoid log(0)
        ratio = np.abs(I_final) / max(np.abs(I_initial), eps)
        ratio = max(ratio, eps)

        # R2_eff from ln(I_final / I_initial) over *total* evolution time
        total_time = 2.0 * t * N_echos if N_echos > 0 else 2.0 * t * N_blocks
        total_time = max(total_time, eps)
        R2eff[i] = -(1.0 / total_time) * np.log(ratio)

    return nu_cpmg, R2eff


def plot_profile(nu_cpmg, R2eff, title=None, show=True, save_png=None):
    plt.figure(figsize=(8, 5))
    plt.plot(nu_cpmg, R2eff, 'o-', label='Simulated CPMG Profile')
    plt.xlabel(r'CPMG Frequency $\nu_{\mathrm{CPMG}}$ (Hz)')
    plt.ylabel(r'Effective Relaxation Rate $R_{2,\mathrm{eff}}$ (s$^{-1}$)')
    if title:
        plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    if save_png:
        plt.savefig(save_png, dpi=200, bbox_inches='tight')
    if show:
        plt.show()


# -----------------------------
# Optional matplotlib sliders
# -----------------------------

def interactive_ui(defaults):
    from matplotlib.widgets import Slider
    # Initial compute
    nu, r2e = simulate_cpmg_profile(**defaults)
    fig, ax = plt.subplots(figsize=(8, 5))
    line, = ax.plot(nu, r2e, 'o-')
    ax.set_xlabel(r'CPMG Frequency $\nu_{\mathrm{CPMG}}$ (Hz)')
    ax.set_ylabel(r'$R_{2,\mathrm{eff}}$ (s$^{-1}$)')
    ax.set_title(f"CPMG Simulation (kex={defaults['kex']:.1f} s$^{{-1}}$)")
    ax.grid(True, linestyle='--', alpha=0.7)

    # Sliders layout
    plt.subplots_adjust(left=0.12, bottom=0.40)

    ax_kex = plt.axes([0.12, 0.33, 0.78, 0.03])
    ax_pB  = plt.axes([0.12, 0.29, 0.78, 0.03])
    ax_dn  = plt.axes([0.12, 0.25, 0.78, 0.03])
    ax_R2  = plt.axes([0.12, 0.21, 0.78, 0.03])
    ax_R1  = plt.axes([0.12, 0.17, 0.78, 0.03])
    ax_Tex = plt.axes([0.12, 0.13, 0.78, 0.03])

    s_kex = Slider(ax_kex, 'kex (s⁻¹)', 50, 5000, valinit=defaults['kex'], valstep=10)
    s_pB  = Slider(ax_pB,  'pB',        0.001, 0.1, valinit=defaults['pB'], valstep=0.001)
    s_dn  = Slider(ax_dn,  'Δν (Hz)',   10, 1000, valinit=defaults['delta_nu'], valstep=5)
    s_R2  = Slider(ax_R2,  'R2 (s⁻¹)',  5, 50, valinit=defaults['R2_base'], valstep=1)
    s_R1  = Slider(ax_R1,  'R1 (s⁻¹)',  1, 5, valinit=defaults['R1'], valstep=0.1)
    s_Tex = Slider(ax_Tex, 'Tex (ms)',  10, 100, valinit=defaults['Tex_ms'], valstep=5)

    def update(_):
        params = dict(
            kex=s_kex.val,
            pB=s_pB.val,
            delta_nu=s_dn.val,
            R2_base=s_R2.val,
            R1=s_R1.val,
            Tex_ms=s_Tex.val,
            fmin=defaults['fmin'],
            fmax=defaults['fmax'],
            npoints=defaults['npoints'],
        )
        nu, r2e = simulate_cpmg_profile(**params)
        line.set_xdata(nu)
        line.set_ydata(r2e)
        ax.set_xlim(nu.min(), nu.max())
        ax.set_ylim(0, max(1e-6, r2e.max()*1.1))
        ax.set_title(f"CPMG Simulation (kex={params['kex']:.1f} s$^{{-1}}$)")
        fig.canvas.draw_idle()

    for s in (s_kex, s_pB, s_dn, s_R2, s_R1, s_Tex):
        s.on_changed(update)

    plt.show()


# -----------------------------
# CLI
# -----------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="CPMG relaxation dispersion simulator (Bloch–McConnell, 2-site)."
    )
    p.add_argument('--kex', type=float, default=500.0, help='Exchange rate kex (s^-1)')
    p.add_argument('--pB', type=float, default=0.02, help='Minor-state population pB (0-1)')
    p.add_argument('--delta-nu', type=float, default=150.0, help='Chemical shift difference Δν (Hz)')
    p.add_argument('--R2', dest='R2_base', type=float, default=10.0, help='Intrinsic R2 (s^-1)')
    p.add_argument('--R1', type=float, default=2.0, help='Intrinsic R1 (s^-1)')
    p.add_argument('--Tex-ms', type=float, default=40.0, help='Total CPMG time Tex (milliseconds)')
    p.add_argument('--fmin', type=float, default=50.0, help='Min CPMG frequency (Hz)')
    p.add_argument('--fmax', type=float, default=2000.0, help='Max CPMG frequency (Hz)')
    p.add_argument('--npoints', type=int, default=40, help='Number of frequency points (log-spaced)')
    p.add_argument('--interactive', action='store_true', help='Use matplotlib sliders for parameters')
    p.add_argument('--no-show', action='store_true', help='Do not display the plot window')
    p.add_argument('--save-png', type=str, default=None, help='Path to save PNG of the plot')
    p.add_argument('--save-csv', type=str, default=None, help='Path to save CSV (nu,R2eff)')
    return p


def main():
    args = build_parser().parse_args()
    defaults = dict(kex=args.kex, pB=args.pB, delta_nu=args.delta_nu, R2_base=args.R2_base,
                    R1=args.R1, Tex_ms=args.Tex_ms, fmin=args.fmin, fmax=args.fmax,
                    npoints=args.npoints)

    if args.interactive:
        interactive_ui(defaults)
        return

    nu, r2e = simulate_cpmg_profile(**defaults)

    if args.save_csv:
        import csv
        with open(args.save_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['nu_cpmg_Hz', 'R2eff_s^-1'])
            writer.writerows(zip(nu, r2e))

    title = f"CPMG Simulation (kex={args.kex:.1f} s$^{{-1}}$)"
    plot_profile(nu, r2e, title=title, show=not args.no_show, save_png=args.save_png)


if __name__ == '__main__':
    main()
