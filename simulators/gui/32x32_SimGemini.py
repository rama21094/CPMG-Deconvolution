import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk
from scipy.linalg import expm

class TwoSpinExchangeCPMGSimulator:
    """Exact 32x32 CPMG simulation for a Two-Spin System with Two-Site Chemical Exchange"""
    
    def __init__(self, R2_I=12.0, R2_S=15.0, J=90.0, 
                 kex=1000.0, pB=0.05, dw_I_ppm=2.0, dw_S_ppm=0.0, 
                 B0=600.0, timeT2=40.0, ncycMax=30):
        
        # Relaxation rates (Applied equally to State A and B for simplicity)
        self.l_I = R2_I    
        self.l_S = R2_S    
        self.r_I, self.r_S = 2.0, 2.0
        self.r_Ia, self.r_Sa = 2.0, 2.0
        self.l_mq = R2_I + R2_S 
        self.r_IS2sp = self.r_I + self.r_S 
        
        # Cross-relaxation (set to 0 for standard baseline)
        self.s, self.d_S, self.h_S, self.m_mq = 0.0, 0.0, 0.0, 0.0
        self.U_I, self.U_S, self.U_IS = -1.0, -1.0, 0.0
        
        # Exchange kinetics
        self.pB = pB
        self.pA = 1.0 - pB
        self.kAB = kex * self.pB
        self.kBA = kex * self.pA
        
        # Frequencies (rad/s)
        # State A is assumed on-resonance (0.0), State B is offset by dw
        self.V_I_A = 0.0
        self.V_S_A = 0.0
        self.V_I_B = dw_I_ppm * B0 * 2.0 * np.pi
        self.V_S_B = dw_S_ppm * B0 * 2.0 * np.pi
        
        self.piJ = np.pi * J        
        self.timeT2 = timeT2 / 1000.0  
        self.ncycMax = ncycMax

    def build_16x16_state_matrix(self, V_I, V_S):
        """Builds the 16x16 Gamma matrix for a specific chemical state."""
        G = np.zeros((16, 16), dtype=complex)
        
        G[1, 1] = self.l_I; G[1, 2] = V_I; G[1, 8] = self.piJ
        G[2, 1] = -V_I; G[2, 2] = self.l_I; G[2, 7] = -self.piJ
        G[3, 0] = -2*self.U_I; G[3, 3] = self.r_I; G[3, 6] = self.s
        G[4, 4] = self.l_S; G[4, 5] = V_S; G[4, 9] = self.h_S; G[4, 10] = self.piJ
        G[5, 4] = -V_S; G[5, 5] = self.l_S; G[5, 9] = -self.piJ; G[5, 10] = self.h_S
        G[6, 0] = -2*self.U_S; G[6, 3] = self.s; G[6, 6] = self.r_S; G[6, 15] = self.d_S
        G[7, 2] = self.piJ; G[7, 7] = self.r_Ia; G[7, 8] = V_I
        G[8, 1] = -self.piJ; G[8, 7] = -V_I; G[8, 8] = self.r_Ia
        G[9, 4] = self.h_S; G[9, 5] = self.piJ; G[9, 9] = self.r_Sa; G[9, 10] = V_S
        G[10, 4] = -self.piJ; G[10, 5] = self.h_S; G[10, 9] = -V_S; G[10, 10] = self.r_Sa
        G[11, 11] = self.l_mq; G[11, 12] = V_S; G[11, 13] = V_I; G[11, 14] = -self.m_mq
        G[12, 11] = -V_S; G[12, 12] = self.l_mq; G[12, 13] = self.m_mq; G[12, 14] = V_I
        G[13, 11] = -V_I; G[13, 12] = self.m_mq; G[13, 13] = self.l_mq; G[13, 14] = V_S
        G[14, 11] = -self.m_mq; G[14, 12] = -V_I; G[14, 13] = -V_S; G[14, 14] = self.l_mq
        G[15, 0] = -2*self.U_IS; G[15, 6] = self.d_S; G[15, 15] = self.r_IS2sp
        
        return G

    def build_32x32_exchange_matrix(self):
        """Snaps the A and B states together with exchange rate cross-terms."""
        G_A = self.build_16x16_state_matrix(self.V_I_A, self.V_S_A)
        G_B = self.build_16x16_state_matrix(self.V_I_B, self.V_S_B)
        
        I_16 = np.eye(16, dtype=complex)
        
        # Add departure rates to the diagonals
        G_A_total = G_A + (self.kAB * I_16)
        G_B_total = G_B + (self.kBA * I_16)
        
        # Create arrival cross-blocks
        K_AB_block = -self.kBA * I_16
        K_BA_block = -self.kAB * I_16
        
        # Assemble the 32x32 block matrix
        top_half = np.hstack((G_A_total, K_AB_block))
        bottom_half = np.hstack((K_BA_block, G_B_total))
        G_32 = np.vstack((top_half, bottom_half))
        
        return G_32

    # def get_ideal_pi_pulse_32(self):
    #     """180-degree pulse applied simultaneously to State A and State B."""
    #     P_16 = np.eye(16, dtype=complex)
    #     flip_indices = [1, 3, 7, 9, 10, 11, 12, 15]
    #     for idx in flip_indices:
    #         P_16[idx, idx] = -1.0
            
    #     # Duplicate the pulse matrix for both states
    #     P_32 = np.block([
    #         [P_16, np.zeros((16, 16), dtype=complex)],
    #         [np.zeros((16, 16), dtype=complex), P_16]
    #     ])
    #     return P_32
    
    def get_ideal_pi_pulse_32(self):
        """180-degree pulse applied along the X-axis to satisfy the Meiboom-Gill condition."""
        P_16 = np.eye(16, dtype=complex)
        
        # FIX: These are the exact Cartesian indices for a 180-pulse on the X-axis
        # It flips Iy(2), Iz(3), 2IySz(8), 2IzSx(9), 2IzSy(10), 2IySx(13), 2IySy(14), 2IzSz(15)
        flip_indices = [2, 3, 8, 9, 10, 13, 14, 15]
        
        for idx in flip_indices:
            P_16[idx, idx] = -1.0
            
        # Duplicate the pulse matrix for both State A and State B
        P_32 = np.block([
            [P_16, np.zeros((16, 16), dtype=complex)],
            [np.zeros((16, 16), dtype=complex), P_16]
        ])
        return P_32

    def calculate_R2eff(self, ncyc):
        if ncyc == 0: return self.l_I, 0.0

        tcpmg = self.timeT2 / (4.0 * ncyc)
        vcpmg = 1.0 / (4.0 * tcpmg)
        
        Gamma_32 = self.build_32x32_exchange_matrix()
        
        E_tau = expm(-Gamma_32 * tcpmg)
        E_2tau = expm(-Gamma_32 * 2.0 * tcpmg)
        P_y = self.get_ideal_pi_pulse_32()
        
        cycle_prop = E_tau @ P_y @ E_2tau @ P_y @ E_tau
        total_prop = np.linalg.matrix_power(cycle_prop, ncyc)
        
        # Initialize 32-element vector based on thermal equilibrium populations
        v_0 = np.zeros(32, dtype=complex)
        # State A initial magnetization
        v_0[0] = self.pA    # Identity operator A
        v_0[1] = self.pA    # Ix magnetization A
        # State B initial magnetization
        v_0[16] = self.pB   # Identity operator B
        v_0[17] = self.pB   # Ix magnetization B
        
        v_final = total_prop @ v_0
        
        # Sum the Ix magnetization from both State A (index 1) and State B (index 17)
        Ix_total = np.real(v_final[1] + v_final[17])
        
        if Ix_total <= 0: return self.l_I, vcpmg
            
        # Total initial Ix magnetization is exactly 1.0 (pA + pB)
        R2eff = -(1.0 / self.timeT2) * np.log(Ix_total / 1.0)
        return R2eff, vcpmg

    def generate_profile(self):
        ncyc_vals = np.arange(1, self.ncycMax + 1)
        vcpmg_vals = []
        R2eff_vals = []
        
        for ncyc in ncyc_vals:
            r2, v = self.calculate_R2eff(ncyc)
            vcpmg_vals.append(v)
            R2eff_vals.append(r2)
            
        return np.array(vcpmg_vals), np.array(R2eff_vals)

class Exchange32GUI:
    def __init__(self, master):
        self.master = master
        master.title("32x32 Two-Spin Chemical Exchange CPMG Simulator")
        master.geometry("1400x800")
        
        self.params = {
            'kex': tk.DoubleVar(value=1000.0),
            'pB': tk.DoubleVar(value=0.05),
            'dw_I': tk.DoubleVar(value=2.0),
            'dw_S': tk.DoubleVar(value=0.0),
            'R2_I': tk.DoubleVar(value=12.0),
            'R2_S': tk.DoubleVar(value=15.0),
            'J': tk.DoubleVar(value=90.0),
            'B0': tk.DoubleVar(value=600.0),
            'timeT2': tk.DoubleVar(value=40.0),
            'ncycMax': tk.IntVar(value=30)
        }
        
        self.setup_gui()
        self.update_plot()
        
    def setup_gui(self):
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        control_frame = ttk.LabelFrame(main_frame, text="Matrix Parameters", padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        row = 0
        ttk.Label(control_frame, text="Exchange Dynamics", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=3, pady=5, sticky=tk.W); row += 1
        self.create_slider(control_frame, row, "kex (s⁻¹)", 'kex', 50, 5000, 50); row += 1
        self.create_slider(control_frame, row, "pB", 'pB', 0.01, 0.20, 0.01); row += 1
        self.create_slider(control_frame, row, "Δω Spin I (ppm)", 'dw_I', -5.0, 5.0, 0.1); row += 1
        self.create_slider(control_frame, row, "Δω Spin S (ppm)", 'dw_S', -5.0, 5.0, 0.1); row += 1
        
        ttk.Label(control_frame, text="Spin Properties", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=3, pady=5, sticky=tk.W); row += 1
        self.create_slider(control_frame, row, "R2 Spin I (s⁻¹)", 'R2_I', 1, 50, 0.5); row += 1
        self.create_slider(control_frame, row, "R2 Spin S (s⁻¹)", 'R2_S', 1, 50, 0.5); row += 1
        self.create_slider(control_frame, row, "J-Coupling (Hz)", 'J', 0, 150, 1); row += 1
        
        ttk.Label(control_frame, text="Experimental Setup", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=3, pady=5, sticky=tk.W); row += 1
        self.create_slider(control_frame, row, "B0 (MHz)", 'B0', 400, 950, 50); row += 1
        self.create_slider(control_frame, row, "T2 time (ms)", 'timeT2', 10, 100, 5); row += 1
        self.create_slider(control_frame, row, "Max cycles", 'ncycMax', 10, 50, 1); row += 1
        
        ttk.Button(control_frame, text="Update Plot", command=self.update_plot).grid(row=row, column=0, columnspan=3, pady=20)
        
        self.fig = Figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_slider(self, parent, row, label, var_name, min_val, max_val, step):
        var = self.params[var_name]
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=2)
        val_label = ttk.Label(parent, text=f"{var.get():.2f}")
        val_label.grid(row=row, column=2, sticky=tk.E, pady=2)
        
        def update_lbl(v):
            val_label.config(text=f"{float(v):.2f}")
            self.update_plot()
            
        slider = ttk.Scale(parent, from_=min_val, to=max_val, variable=var, command=update_lbl, orient=tk.HORIZONTAL)
        slider.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=10)

    def update_plot(self, *args):
        self.ax.clear()
        p = {k: v.get() for k, v in self.params.items()}
        
        sim = TwoSpinExchangeCPMGSimulator(
            kex=p['kex'], pB=p['pB'], dw_I_ppm=p['dw_I'], dw_S_ppm=p['dw_S'],
            R2_I=p['R2_I'], R2_S=p['R2_S'], J=p['J'], 
            B0=p['B0'], timeT2=p['timeT2'], ncycMax=int(p['ncycMax'])
        )
        
        vcpmg, R2eff = sim.generate_profile()
        
        self.ax.plot(vcpmg, R2eff, 'o-', color='darkblue', linewidth=2, markersize=6)
        self.ax.set_xlabel('νCPMG (Hz)', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('R2eff (s⁻¹)', fontsize=12, fontweight='bold')
        self.ax.set_title('32x32 Exact Exchange Dispersion Profile', fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = Exchange32GUI(root)
    root.mainloop()