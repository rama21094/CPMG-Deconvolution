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

    def build_16x16(self, B0, tau_m, tau_e, S2, r_IS, csa_N, theta_N, J_IS, dw_offset):
        """Builds the rigorous 16x16 Homogeneous Master Equation Matrix"""
        L = np.zeros((16, 16), dtype=complex)
        
        omega_H = -self.gamma_H * B0
        omega_N = -self.gamma_N * B0

        # Spectral densities
        j0 = self.J_comp(0, tau_m, tau_e, S2)
        jH = self.J_comp(omega_H, tau_m, tau_e, S2)
        jN = self.J_comp(omega_N, tau_m, tau_e, S2)
        jH_minus_N = self.J_comp(omega_H - omega_N, tau_m, tau_e, S2)
        jH_plus_N = self.J_comp(omega_H + omega_N, tau_m, tau_e, S2)
        
        # Interaction Constants
        d2 = (self.mu_0 / (4.0 * np.pi))**2 * (self.hbar * self.gamma_H * self.gamma_N / r_IS**3)**2
        c2_N = (1.0 / 3.0) * (omega_N * csa_N)**2
        
        # Cross-correlation prefactor
        P2_cos = 0.5 * (3.0 * np.cos(theta_N)**2 - 1.0)
        dc = np.sqrt(d2) * np.sqrt(c2_N) * P2_cos

        # Auto-relaxation rates
        R1_N = (d2 / 4.0) * (jH_minus_N + 3*jN + 6*jH_plus_N) + c2_N * jN
        R2_N = (d2 / 8.0) * (4*j0 + jH_minus_N + 3*jN + 6*jH + 6*jH_plus_N) + (c2_N / 6.0) * (4*j0 + 3*jN)
        R1_H = (d2 / 4.0) * (jH_minus_N + 3*jH + 6*jH_plus_N)
        R2_H = (d2 / 8.0) * (4*j0 + jH_minus_N + 3*jH + 6*jN + 6*jH_plus_N)
        R2_HzNx = (d2 / 8.0) * (4*j0 + jH_minus_N + 3*jN + 6*jH + 6*jH_plus_N) + \
                  (d2 / 4.0) * (jH_minus_N + 3*jH + 6*jH_plus_N) + (c2_N / 6.0) * (4*j0 + 3*jN)
        R_MQ = (d2 / 8.0) * (jH_minus_N + 3*jH + 3*jN + 6*jH_plus_N) + (c2_N / 6.0) * (4*j0 + 3*jN)

        # Cross-correlated relaxation rates
        eta_z = dc * jN
        eta_xy = (dc / 6.0) * (4*j0 + 3*jN)

        # Populate Relaxation Diagonal
        L[1,1] = L[2,2] = -R2_H
        L[3,3] = -R1_H
        L[4,4] = L[5,5] = -R2_N
        L[6,6] = -R1_N
        L[7,7] = L[8,8] = L[9,9] = L[10,10] = -R2_HzNx
        L[11,11] = L[12,12] = L[13,13] = L[14,14] = -R_MQ
        L[15,15] = -(R1_H + R1_N)

        # DD-CSA Cross-Correlation
        L[4,9] = L[9,4] = -eta_xy
        L[5,10] = L[10,5] = -eta_xy
        L[6,15] = L[15,6] = -eta_z

        # Coherent Evolution (Chemical Shift & Scalar Coupling)
        L[4, 5] = -dw_offset; L[5, 4] = dw_offset
        L[9, 10] = -dw_offset; L[10, 9] = dw_offset
        L[11, 12] = -dw_offset; L[12, 11] = dw_offset
        L[13, 14] = -dw_offset; L[14, 13] = dw_offset

        pi_J = np.pi * J_IS
        L[1, 8] = -pi_J; L[8, 1] = pi_J
        L[2, 7] = pi_J; L[7, 2] = -pi_J
        L[4, 10] = -pi_J; L[10, 4] = pi_J
        L[5, 9] = pi_J; L[9, 5] = -pi_J
        L[11, 14] = -pi_J; L[14, 11] = pi_J 
        L[12, 13] = pi_J; L[13, 12] = -pi_J 

        return L

    def simulate_cpmg(self, params, ncyc_range, T_relax=0.04):
        """Builds the 32x32 matrix and applies the CPMG propagators"""
        
        # Unpack parameters
        B0 = params['B0']; tau_m = params['tau_m']; tau_e = params['tau_e']
        S2 = params['S2']; r_IS = params['r_IS']; csa_N = params['csa_N']
        theta_N = params['theta_N']; J_IS = params['J_IS']; k_ex = params['k_ex']
        p_B = params['p_B']; dw_N_ppm = params['dw_N']
        
        # Derived Exchange
        dw_N = dw_N_ppm * B0 * (self.gamma_N / 1e6) # rad/s
        p_A = 1.0 - p_B
        k_AB = k_ex * p_B
        k_BA = k_ex * p_A

        L_full = np.zeros((32, 32), dtype=complex)
        L_A = self.build_16x16(B0, tau_m, tau_e, S2, r_IS, csa_N, theta_N, J_IS, 0.0)
        L_B = self.build_16x16(B0, tau_m, tau_e, S2, r_IS, csa_N, theta_N, J_IS, dw_N)
        
        L_full[0:16, 0:16] = L_A
        L_full[16:32, 16:32] = L_B

        # Apply Bloch-McConnell Exchange
        for i in range(1, 16):
            L_full[i, i] -= k_AB; L_full[i, i+16] += k_BA
            L_full[i+16, i] += k_AB; L_full[i+16, i+16] -= k_BA

        # 180 degree pulse on Nitrogen (assumed ideal)
        P_180 = np.eye(32)
        for offset in [0, 16]:
            P_180[offset+5, offset+5] = -1 # Sy -> -Sy
            P_180[offset+6, offset+6] = -1 # Sz -> -Sz
            P_180[offset+7, offset+7] = -1 # 2IxSz -> -2IxSz
            P_180[offset+8, offset+8] = -1 # 2IySz -> -2IySz

        # Initial State (Sx magnetization on A and B)
        rho_0 = np.zeros(32)
        rho_0[4] = p_A; rho_0[20] = p_B
        
        nu_cp_list = []
        R2_eff_list = []

        for ncyc in ncyc_range:
            tau_cp = T_relax / (4.0 * ncyc)
            nu_cp = 1.0 / (4.0 * tau_cp)
            
            U_tau = la.expm(L_full * tau_cp)
            U_block = U_tau @ P_180 @ U_tau @ U_tau @ P_180 @ U_tau
            U_total = np.linalg.matrix_power(U_block, int(ncyc))
            
            rho_t = U_total @ rho_0
            mag_x = np.real(rho_t[4] + rho_t[20])
            
            R2_eff = -np.log(mag_x / (p_A + p_B)) / T_relax
            nu_cp_list.append(nu_cp)
            R2_eff_list.append(R2_eff)
            
        return nu_cp_list, R2_eff_list

# --- GUI Application ---

class CPMG_App:
    def __init__(self, root):
        self.root = root
        self.root.title("Rigorous CPMG Dispersion Simulator")
        self.sim = CPMG_Simulator()
        self.lines = [] # Keep track of plotted lines for overlay
        
        # Default Literature Values
        self.vars = {
            'B0': tk.DoubleVar(value=14.1),             # Tesla
            'tau_m': tk.DoubleVar(value=5.0e-9),        # s
            'tau_e': tk.DoubleVar(value=50e-12),        # s
            'S2': tk.DoubleVar(value=0.85),
            'r_IS': tk.DoubleVar(value=1.02e-10),       # m
            'csa_N': tk.DoubleVar(value=-160e-6),       # ppm
            'theta_N': tk.DoubleVar(value=22.0*(np.pi/180)), # rad
            'J_IS': tk.DoubleVar(value=92.0),           # Hz
            'k_ex': tk.DoubleVar(value=500.0),          # s^-1
            'p_B': tk.DoubleVar(value=0.05),
            'dw_N': tk.DoubleVar(value=1.5)             # ppm
        }

        self.setup_ui()

    def setup_ui(self):
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        ttk.Label(control_frame, text="Independent Variables", font=('Helvetica', 12, 'bold')).grid(row=0, column=0, columnspan=2, pady=10)

        # Generate Input Fields dynamically
        labels = {
            'B0': 'B0 Field (T)', 'tau_m': 'tau_m (s)', 'tau_e': 'tau_e (s)', 
            'S2': 'Order Param S2', 'r_IS': 'N-H Distance (m)', 'csa_N': '15N CSA (ppm)',
            'theta_N': 'Theta CSA/DD (rad)', 'J_IS': 'Scalar Coup J (Hz)', 
            'k_ex': 'Exchange Rate k_ex (s^-1)', 'p_B': 'Population p_B', 'dw_N': 'Delta Omega 15N (ppm)'
        }
        
        row = 1
        for key, text in labels.items():
            ttk.Label(control_frame, text=text).grid(row=row, column=0, sticky='w')
            ttk.Entry(control_frame, textvariable=self.vars[key], width=12).grid(row=row, column=1, padx=5, pady=2)
            row += 1

        ttk.Button(control_frame, text="Plot / Overlay Profile", command=self.run_simulation).grid(row=row, column=0, columnspan=2, pady=15)
        ttk.Button(control_frame, text="Clear Plot", command=self.clear_plot).grid(row=row+1, column=0, columnspan=2)

        # Plotting Area
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.fig, self.ax = plt.subplots(figsize=(7, 5))
        self.ax.set_xlabel(r'$\nu_{cp}$ (Hz)')
        self.ax.set_ylabel(r'$R_{2,eff}$ ($s^{-1}$)')
        self.ax.set_title('Simulated CPMG Dispersion Profile')
        self.ax.grid(True)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def run_simulation(self):
        # Extract current parameters from GUI
        params = {k: v.get() for k, v in self.vars.items()}
        
        ncyc_range = np.arange(1, 40, 2)
        
        try:
            nu_cp, R2_eff = self.sim.simulate_cpmg(params, ncyc_range)
            
            # Label based on J_IS for easy comparison
            label = f"J={params['J_IS']} Hz, k_ex={params['k_ex']} s-1"
            line, = self.ax.plot(nu_cp, R2_eff, '-o', label=label)
            self.lines.append(line)
            
            self.ax.legend()
            self.canvas.draw()
            
        except Exception as e:
            print(f"Simulation Error: {e}")

    def clear_plot(self):
        for line in self.lines:
            line.remove()
        self.lines.clear()
        self.ax.legend_ = None
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = CPMG_App(root)
    root.mainloop()