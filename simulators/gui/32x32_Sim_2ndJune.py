import tkinter as tk
from tkinter import ttk
import numpy as np
import scipy.linalg as la
import scipy.constants as const
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Physics Engine ---

class CPMG_Simulator:
    def __init__(self):
        # Physical Constants
        self.mu_0 = const.mu_0
        self.hbar = const.hbar
        self.gamma_H = 267.522e6   # 1H gyromagnetic ratio (rad s^-1 T^-1)
        self.gamma_N = -27.116e6   # 15N gyromagnetic ratio (rad s^-1 T^-1)

    def J_comp(self, omega, tau_m, tau_e, S2):
        """Lipari-Szabo Spectral Density Function"""
        tau_inv = (1.0 / tau_m) + (1.0 / tau_e)
        tau = 1.0 / tau_inv
        term1 = (S2 * tau_m) / (1.0 + (omega * tau_m)**2)
        term2 = ((1.0 - S2) * tau) / (1.0 + (omega * tau)**2)
        return (2.0 / 5.0) * (term1 + term2)

    def build_16x16(self, B0, tau_m, tau_e, S2, r_IS, r_eff, csa_N, theta_N, J_IS, dw_offset):
        """Builds the exact 16x16 Homogeneous Master Equation Matrix"""
        L = np.zeros((16, 16), dtype=complex)
        
        omega_H = -self.gamma_H * B0
        omega_N = -self.gamma_N * B0

        # Spectral densities
        j0 = self.J_comp(0, tau_m, tau_e, S2)
        jH = self.J_comp(omega_H, tau_m, tau_e, S2)
        jN = self.J_comp(omega_N, tau_m, tau_e, S2)
        jH_minus_N = self.J_comp(omega_H - omega_N, tau_m, tau_e, S2)
        jH_plus_N = self.J_comp(omega_H + omega_N, tau_m, tau_e, S2)
        j2H = self.J_comp(2 * omega_H, tau_m, tau_e, S2) # For homonuclear leak
        
        # Interaction Constants
        d2 = (self.mu_0 / (4.0 * np.pi))**2 * (self.hbar * self.gamma_H * self.gamma_N / r_IS**3)**2
        d2_H = (self.mu_0 / (4.0 * np.pi))**2 * (self.hbar * self.gamma_H**2 / r_eff**3)**2
        c2_N = (1.0 / 3.0) * (omega_N * csa_N)**2
        
        P2_cos = 0.5 * (3.0 * np.cos(theta_N)**2 - 1.0)
        dc = np.sqrt(d2) * np.sqrt(c2_N) * P2_cos

        # Eq 39 & 40: Homonuclear Leakage
        rho_H = (d2_H / 4.0) * (j0 + 3*jH + 6*j2H)
        lambda_H = (d2_H / 8.0) * (5*j0 + 9*jH + 6*j2H)

        # Eq 23-30: Auto-relaxation rates
        R1_N = (d2 / 4.0) * (jH_minus_N + 3*jN + 6*jH_plus_N) + c2_N * jN
        R2_N = (d2 / 8.0) * (4*j0 + jH_minus_N + 3*jN + 6*jH + 6*jH_plus_N) + (c2_N / 6.0) * (4*j0 + 3*jN)
        
        R1_H = (d2 / 4.0) * (jH_minus_N + 3*jH + 6*jH_plus_N) + rho_H
        R2_H = (d2 / 8.0) * (4*j0 + jH_minus_N + 3*jH + 6*jN + 6*jH_plus_N) + lambda_H
        
        # Anti-phase rates
        R2_SzNx = (d2 / 8.0) * (4*j0 + jH_minus_N + 3*jN + 6*jH + 6*jH_plus_N) + \
                  (d2 / 4.0) * (jH_minus_N + 3*jH + 6*jH_plus_N) + (c2_N / 6.0) * (4*j0 + 3*jN) + rho_H
        R2_HzNx = (d2 / 8.0) * (4*j0 + jH_minus_N + 3*jH + 6*jN + 6*jH_plus_N) + \
                  (d2 / 4.0) * (jH_minus_N + 3*jN + 6*jH_plus_N) + (c2_N / 3.0) * jN + lambda_H
                  
        R_MQ = (d2 / 8.0) * (jH_minus_N + 3*jH + 3*jN + 6*jH_plus_N) + (c2_N / 6.0) * (4*j0 + 3*jN) + lambda_H
        R1_IS2sp = (d2 / 4.0) * (3*jN + 3*jH) + (c2_N / 3.0) * jN + rho_H

        # Eq 33-34: DD-CSA Cross-correlated relaxation
        eta_z = dc * jN
        eta_xy = (dc / 6.0) * (4*j0 + 3*jN)

        # Eq 31-32: Cross-Relaxation (NOE and MQ)
        sigma = (d2 / 4.0) * (-jH_minus_N + 6*jH_plus_N)
        mu_mq = (d2 / 4.0) * (-0.5*jH_minus_N + 3*jH_plus_N)

        # Eq 20-22: Thermal Equilibrium Returns
        # Normalized equilibrium magnetizations (M_H0 = 1, M_N0 = gamma_N / gamma_H)
        M_H0 = 1.0
        M_N0 = self.gamma_N / self.gamma_H
        Theta_I = R1_H * M_H0 + sigma * M_N0
        Theta_S = sigma * M_H0 + R1_N * M_N0
        Theta_IS = eta_z * M_N0

        # --- Populate Matrix ---
        
        # 1. Thermal Return (Column 0 = E/2)
        L[3, 0] = 2.0 * Theta_I   # Iz
        L[6, 0] = 2.0 * Theta_S   # Sz
        L[15, 0] = 2.0 * Theta_IS # 2IzSz

        # 2. Diagonal Relaxation
        L[1,1] = L[2,2] = -R2_H
        L[3,3] = -R1_H
        L[4,4] = L[5,5] = -R2_N
        L[6,6] = -R1_N
        L[7,7] = L[8,8] = -R2_HzNx
        L[9,9] = L[10,10] = -R2_SzNx
        L[11,11] = L[12,12] = L[13,13] = L[14,14] = -R_MQ
        L[15,15] = -R1_IS2sp

        # 3. Cross-Relaxation
        L[3,6] = L[6,3] = -sigma                    # Iz <-> Sz
        L[4,9] = L[9,4] = -eta_xy                   # Sx <-> 2IzSx
        L[5,10] = L[10,5] = -eta_xy                 # Sy <-> 2IzSy
        L[6,15] = L[15,6] = -eta_z                  # Sz <-> 2IzSz
        L[11,14] = L[14,11] = -mu_mq                # 2IxSx <-> 2IySy
        L[12,13] = L[13,12] = mu_mq                 # 2IxSy <-> 2IySx

        # 4. Coherent Evolution (Chemical Shift & Scalar Coupling)
        L[4, 5] = -dw_offset; L[5, 4] = dw_offset
        L[9, 10] = -dw_offset; L[10, 9] = dw_offset
        L[11, 12] = -dw_offset; L[12, 11] = dw_offset
        L[13, 14] = -dw_offset; L[14, 13] = dw_offset

        pi_J = np.pi * J_IS
        L[1, 8] = -pi_J; L[8, 1] = pi_J
        L[2, 7] = pi_J; L[7, 2] = -pi_J
        L[4, 10] = -pi_J; L[10, 4] = pi_J
        L[5, 9] = pi_J; L[9, 5] = -pi_J
        L[11, 14] -= pi_J; L[14, 11] += pi_J 
        L[12, 13] += pi_J; L[13, 12] -= pi_J 

        return L

    def build_RF_Liouvillian(self, B1_N_Hz):
        """Builds the 32x32 RF matrix for a finite X-axis pulse on the S spin"""
        L_RF = np.zeros((32, 32), dtype=complex)
        w_Sx = 2.0 * np.pi * B1_N_Hz
        
        # Apply to both State A (0-15) and State B (16-31)
        for offset in [0, 16]:
            L_RF[offset+5, offset+6] = w_Sx;  L_RF[offset+6, offset+5] = -w_Sx   # Sy <-> Sz
            L_RF[offset+12, offset+7] = w_Sx; L_RF[offset+7, offset+12] = -w_Sx  # 2IxSy <-> 2IxSz
            L_RF[offset+14, offset+8] = w_Sx; L_RF[offset+8, offset+14] = -w_Sx  # 2IySy <-> 2IySz
            L_RF[offset+10, offset+15] = w_Sx; L_RF[offset+15, offset+10] = -w_Sx # 2IzSy <-> 2IzSz
            
        return L_RF

    def simulate_cpmg(self, params, ncyc_range, T_relax=0.04):
        # Unpack parameters
        B0 = params['B0']; tau_m = params['tau_m']; tau_e = params['tau_e']
        S2 = params['S2']; r_IS = params['r_IS']; r_eff = params['r_eff']
        csa_N = params['csa_N']; theta_N = params['theta_N']; J_IS = params['J_IS']
        k_ex = params['k_ex']; p_B = params['p_B']; dw_N_ppm = params['dw_N']
        B1_N_Hz = params['B1_N']
        
        dw_N = dw_N_ppm * B0 * (self.gamma_N / 1e6)
        p_A = 1.0 - p_B
        k_AB = k_ex * p_B
        k_BA = k_ex * p_A

        # Build Static Matrices
        L_free = np.zeros((32, 32), dtype=complex)
        L_A = self.build_16x16(B0, tau_m, tau_e, S2, r_IS, r_eff, csa_N, theta_N, J_IS, 0.0)
        L_B = self.build_16x16(B0, tau_m, tau_e, S2, r_IS, r_eff, csa_N, theta_N, J_IS, dw_N)
        
        L_free[0:16, 0:16] = L_A
        L_free[16:32, 16:32] = L_B

        # Apply Bloch-McConnell Exchange to Free Matrix
        for i in range(1, 16):
            L_free[i, i] -= k_AB; L_free[i, i+16] += k_BA
            L_free[i+16, i] += k_AB; L_free[i+16, i+16] -= k_BA

        # Build RF Pulse Matrix
        L_RF = self.build_RF_Liouvillian(B1_N_Hz)
        L_pulse = L_free + L_RF
        
        t_180 = 1.0 / (2.0 * B1_N_Hz)
        U_pulse = la.expm(L_pulse * t_180)

        # Initial State (Transverse Sx magnetization)
        rho_0 = np.zeros(32)
        rho_0[0] = 1.0; rho_0[16] = 1.0 # Unity operator E/2
        rho_0[4] = p_A; rho_0[20] = p_B # Sx states
        
        nu_cp_list = []
        R2_eff_list = []
        skipped_ncyc = []

        for ncyc in ncyc_range:
            # Calculate free precession delay accounting for finite pulse width
            total_pulse_time = ncyc * t_180
            if total_pulse_time >= T_relax:
                skipped_ncyc.append(int(ncyc))
                continue # Pulses take up the entire window, unphysical for CPMG
                
            tau_cp = (T_relax - total_pulse_time) / (2.0 * ncyc)
            nu_cp = 1.0 / (4.0 * tau_cp) # Standard mapping
            
            U_tau = la.expm(L_free * tau_cp)
            
            # CPMG Block: tau_cp - Finite Pulse - tau_cp
            U_block = U_tau @ U_pulse @ U_tau
            U_total = np.linalg.matrix_power(U_block, int(ncyc))
            
            rho_t = U_total @ rho_0
            mag_x = np.real(rho_t[4] + rho_t[20])
            
            R2_eff = -np.log(mag_x / (p_A + p_B)) / T_relax
            nu_cp_list.append(nu_cp)
            R2_eff_list.append(R2_eff)
            
        return nu_cp_list, R2_eff_list, skipped_ncyc

# --- GUI Application ---

class CPMG_App:
    def __init__(self, root):
        self.root = root
        self.root.title("Full Rigorous CPMG Simulator (Finite Pulses)")
        self.sim = CPMG_Simulator()
        self.lines = []
        
        # Extended Parameters including Leakage and B1 Field
        self.vars = {
            'B0_MHz': tk.DoubleVar(value=600.0),
            'B1_N': tk.DoubleVar(value=5555.0),          # RF Field Strength Hz
            'ncyc_start': tk.IntVar(value=1),
            'ncyc_stop': tk.IntVar(value=40),
            'ncyc_step': tk.IntVar(value=2),
            'tau_m': tk.DoubleVar(value=5.0e-9),        
            'tau_e': tk.DoubleVar(value=50e-12),        
            'S2': tk.DoubleVar(value=0.85),
            'r_IS': tk.DoubleVar(value=1.02e-10),       
            'r_eff': tk.DoubleVar(value=1.86e-10),      # Effective bath H distance
            'csa_N': tk.DoubleVar(value=-160e-6),       
            'theta_N': tk.DoubleVar(value=22.0*(np.pi/180)), 
            'J_IS': tk.DoubleVar(value=92.0),           
            'k_ex': tk.DoubleVar(value=500.0),          
            'p_B': tk.DoubleVar(value=0.05),
            'dw_N': tk.DoubleVar(value=1.5)
        }

        self.setup_ui()

    def setup_ui(self):
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        ttk.Label(control_frame, text="CPMG Controls", font=('Helvetica', 11, 'bold')).grid(row=0, column=0, columnspan=2, pady=10)

        row = 1
        row = self.add_section(control_frame, "Field / Sequence", [
            ('B0_MHz', 'B0 1H Frequency (MHz)'),
            ('B1_N', '15N B1 Field (Hz)'),
            ('ncyc_start', 'ncyc Start'),
            ('ncyc_stop', 'ncyc Stop'),
            ('ncyc_step', 'ncyc Step')
        ], row)

        row = self.add_section(control_frame, "Dynamics / Exchange", [
            ('k_ex', 'k_ex (s^-1)'),
            ('p_B', 'Population p_B'),
            ('dw_N', 'Delta Omega (ppm)')
        ], row)

        row = self.add_section(control_frame, "Relaxation Model", [
            ('tau_m', 'tau_m (s)'),
            ('tau_e', 'tau_e (s)'),
            ('S2', 'Order Param S2'),
            ('r_IS', 'N-H Distance (m)'),
            ('r_eff', 'Bath H Dist (m)'),
            ('csa_N', '15N CSA (ppm)'),
            ('theta_N', 'Theta CSA/DD (rad)'),
            ('J_IS', 'Scalar Coup J (Hz)')
        ], row)

        ttk.Button(control_frame, text="Plot / Overlay Profile", command=self.run_simulation).grid(row=row, column=0, columnspan=2, pady=15)
        ttk.Button(control_frame, text="Clear Plot", command=self.clear_plot).grid(row=row+1, column=0, columnspan=2)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(control_frame, textvariable=self.status_var, foreground="blue", wraplength=260).grid(row=row+2, column=0, columnspan=2, sticky='w', pady=10)

        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.fig, self.ax = plt.subplots(figsize=(7, 5))
        self.ax.set_xlabel(r'$\nu_{cp}$ (Hz)')
        self.ax.set_ylabel(r'$R_{2,eff}$ ($s^{-1}$)')
        self.ax.set_title('Simulated CPMG Dispersion Profile')
        self.ax.grid(True)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def add_section(self, parent, title, fields, row):
        ttk.Label(parent, text=title, font=('Helvetica', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky='w', pady=(12, 4))
        row += 1
        for key, text in fields:
            ttk.Label(parent, text=text).grid(row=row, column=0, sticky='w')
            ttk.Entry(parent, textvariable=self.vars[key], width=12).grid(row=row, column=1, padx=5, pady=2)
            row += 1
        return row

    def validate_inputs(self, params, ncyc_range):
        checks = [
            (params['B0_MHz'] > 0, "B0 frequency must be greater than 0 MHz."),
            (params['B1_N'] > 0, "B1 field must be greater than 0 Hz."),
            (params['ncyc_start'] >= 1, "ncyc start must be at least 1."),
            (params['ncyc_stop'] > params['ncyc_start'], "ncyc stop must be greater than ncyc start."),
            (params['ncyc_step'] >= 1, "ncyc step must be at least 1."),
            (len(ncyc_range) > 0, "ncyc range is empty."),
            (0.0 <= params['p_B'] <= 1.0, "Population p_B must be between 0 and 1.")
        ]
        for passed, message in checks:
            if not passed:
                raise ValueError(message)

    def run_simulation(self):
        try:
            params = {k: v.get() for k, v in self.vars.items()}
            ncyc_start = int(params['ncyc_start'])
            ncyc_stop = int(params['ncyc_stop'])
            ncyc_step = int(params['ncyc_step'])
            params['ncyc_start'] = ncyc_start
            params['ncyc_stop'] = ncyc_stop
            params['ncyc_step'] = ncyc_step
            if ncyc_step < 1:
                raise ValueError("ncyc step must be at least 1.")
            ncyc_range = np.arange(ncyc_start, ncyc_stop, ncyc_step)
            self.validate_inputs(params, ncyc_range)

            B0_MHz = params['B0_MHz']
            params['B0'] = 2.0 * np.pi * B0_MHz * 1.0e6 / self.sim.gamma_H

            nu_cp, R2_eff, skipped_ncyc = self.sim.simulate_cpmg(params, ncyc_range)
            if len(nu_cp) == 0:
                raise ValueError("No valid CPMG points remain after finite-pulse timing checks.")

            label = f"B0={B0_MHz:g}MHz, B1={params['B1_N']:g}Hz, ncyc={ncyc_start}:{ncyc_stop}:{ncyc_step}"
            line, = self.ax.plot(nu_cp, R2_eff, '-o', label=label)
            self.lines.append(line)
            
            self.ax.legend()
            self.canvas.draw()
            status = f"Plotted {len(nu_cp)} points. Internal B0 = {params['B0']:.3f} T."
            if skipped_ncyc:
                status += f" Skipped {len(skipped_ncyc)} high-ncyc point(s): {skipped_ncyc}."
            self.status_var.set(status)
        except Exception as e:
            self.status_var.set(f"Simulation Error: {e}")

    def clear_plot(self):
        for line in self.lines:
            line.remove()
        self.lines.clear()
        self.ax.legend_ = None
        self.canvas.draw()
        self.status_var.set("Plot cleared.")

if __name__ == "__main__":
    root = tk.Tk()
    app = CPMG_App(root)
    root.mainloop()
