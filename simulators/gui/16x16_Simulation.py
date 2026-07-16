import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk
from scipy.linalg import expm

class TwoSpinCPMGSimulator:
    """Exact CPMG simulation using the 16x16 Homogeneous Master Equation Matrix"""
    
    def __init__(self, l_I=10.0, l_S=10.0, r_I=2.0, r_S=2.0, 
                 offset_I_ppm=0.0, offset_S_ppm=0.0, B0=600.0, 
                 J=15.0, timeT2=30.0, ncycMax=30):
        
        # Auto-relaxation rates (transverse and longitudinal)
        self.l_I = l_I    # Ix, Iy relaxation (equivalent to R2_I)
        self.l_S = l_S    # Sx, Sy relaxation (equivalent to R2_S)
        self.r_I = r_I    # Iz relaxation
        self.r_S = r_S    # Sz relaxation
        self.r_Ia = r_I   
        self.r_Sa = r_S
        self.l_mq = l_I + l_S 
        self.r_IS2sp = r_I + r_S 
        
        # Cross-relaxation terms (simplified to 0 for a baseline profile)
        self.s, self.d_S, self.h_S, self.m_mq = 0.0, 0.0, 0.0, 0.0
        
        # Thermal equilibrium components
        self.U_I, self.U_S, self.U_IS = -1.0, -1.0, 0.0
        
        # Frequencies and coupling
        # Convert ppm to rad/s using the B0 field
        self.V_I = offset_I_ppm * B0 * 2 * np.pi  
        self.V_S = offset_S_ppm * B0 * 2 * np.pi  
        self.piJ = np.pi * J        
        
        # CPMG setup
        self.timeT2 = timeT2 / 1000.0  # Convert ms to seconds
        self.ncycMax = ncycMax

    def build_gamma_free(self):
        """Builds the 16x16 Gamma matrix for FREE EVOLUTION."""
        G = np.zeros((16, 16), dtype=float)
        
        # Row 2 (Ix) and Row 3 (Iy)
        G[1, 1] = self.l_I; G[1, 2] = self.V_I; G[1, 8] = self.piJ
        G[2, 1] = -self.V_I; G[2, 2] = self.l_I; G[2, 7] = -self.piJ
        
        # Row 4 (Iz)
        G[3, 0] = -2*self.U_I; G[3, 3] = self.r_I; G[3, 6] = self.s
        
        # Row 5 (Sx) and Row 6 (Sy)
        G[4, 4] = self.l_S; G[4, 5] = self.V_S; G[4, 9] = self.h_S; G[4, 10] = self.piJ
        G[5, 4] = -self.V_S; G[5, 5] = self.l_S; G[5, 9] = -self.piJ; G[5, 10] = self.h_S
        
        # Row 7 (Sz)
        G[6, 0] = -2*self.U_S; G[6, 3] = self.s; G[6, 6] = self.r_S; G[6, 15] = self.d_S
        
        # Row 8 (2IxSz) and Row 9 (2IySz)
        G[7, 2] = self.piJ; G[7, 7] = self.r_Ia; G[7, 8] = self.V_I
        G[8, 1] = -self.piJ; G[8, 7] = -self.V_I; G[8, 8] = self.r_Ia
        
        # Row 10 (2IzSx) and Row 11 (2IzSy)
        G[9, 4] = self.h_S; G[9, 5] = self.piJ; G[9, 9] = self.r_Sa; G[9, 10] = self.V_S
        G[10, 4] = -self.piJ; G[10, 5] = self.h_S; G[10, 9] = -self.V_S; G[10, 10] = self.r_Sa
        
        # Row 12 (2IxSx) to Row 15 (2IySy) - Multiple Quantum
        G[11, 11] = self.l_mq; G[11, 12] = self.V_S; G[11, 13] = self.V_I; G[11, 14] = -self.m_mq
        G[12, 11] = -self.V_S; G[12, 12] = self.l_mq; G[12, 13] = self.m_mq; G[12, 14] = self.V_I
        G[13, 11] = -self.V_I; G[13, 12] = self.m_mq; G[13, 13] = self.l_mq; G[13, 14] = self.V_S
        G[14, 11] = -self.m_mq; G[14, 13] = -self.V_S; G[14, 14] = self.l_mq
        
        # Row 16 (2IzSz)
        G[15, 0] = -2*self.U_IS; G[15, 6] = self.d_S; G[15, 15] = self.r_IS2sp
        
        return G

    def get_ideal_pi_pulse_I(self):
        """Transformation matrix for an ideal 180-degree pulse on I spin along Y-axis."""
        P = np.eye(16)
        flip_indices = [1, 3, 7, 9, 10, 11, 12, 15]
        for idx in flip_indices:
            P[idx, idx] = -1.0
        return P

    def calculate_R2eff(self, ncyc):
        """Calculates R2,eff from the Ix magnetization decay for a given ncyc."""
        if ncyc == 0: return self.l_I, 0.0

        tcpmg = self.timeT2 / (4.0 * ncyc)
        vcpmg = 1.0 / (4.0 * tcpmg)
        
        Gamma = self.build_gamma_free()
        
        # Suppress numpy warnings during matrix exponential calculations
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            # Propagators - compute once
            E_tau = expm(-Gamma * tcpmg)
            E_2tau = expm(-Gamma * 2.0 * tcpmg)
            P_y = self.get_ideal_pi_pulse_I()
            
            # One cycle: tau -> Pi -> 2tau -> Pi -> tau
            cycle_prop = E_tau @ P_y @ E_2tau @ P_y @ E_tau
            
            # Replace matrix power with iterative multiplication for stability
            total_prop = np.eye(16)
            for _ in range(ncyc):
                total_prop = total_prop @ cycle_prop
                # Check for numerical overflow/underflow
                if np.any(np.isnan(total_prop)) or np.any(np.isinf(total_prop)):
                    # Return safe default values if matrix becomes invalid
                    return self.l_I, vcpmg
            
            # Start pure Ix
            v_0 = np.zeros(16); v_0[0] = 1.0; v_0[1] = 1.0
            
            # Clamp total_prop to avoid NaN/inf propagation
            total_prop = np.nan_to_num(total_prop, nan=0.0, posinf=1e308, neginf=-1e308)
            v_final = total_prop @ v_0
            
            Ix_final = v_final[1]
            
            # Handle edge cases: very small or invalid values
            if Ix_final <= 0 or np.isnan(Ix_final) or np.isinf(Ix_final):
                return self.l_I, vcpmg
            
            # Clamp the ratio to prevent log of invalid values
            ratio = np.clip(Ix_final / v_0[1], 1e-308, 1e308)
            
            try:
                R2eff = -(1.0 / self.timeT2) * np.log(ratio)
                # Clamp R2eff to reasonable physical range
                R2eff = np.clip(R2eff, 0, 1000)
            except (ValueError, RuntimeWarning):
                return self.l_I, vcpmg
        
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

class TwoSpinGUI:
    def __init__(self, master):
        self.master = master
        master.title("16x16 Homogeneous Two-Spin CPMG Simulator")
        master.geometry("1400x800")
        
        # Parameters representing the 16x16 matrix physics
        self.params = {
            'l_I': tk.DoubleVar(value=12.0),
            'l_S': tk.DoubleVar(value=15.0),
            'offset_I': tk.DoubleVar(value=0.1),
            'offset_S': tk.DoubleVar(value=0.0),
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
        ttk.Label(control_frame, text="Relaxation Rates (s⁻¹)", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=3, pady=5, sticky=tk.W); row += 1
        self.create_slider(control_frame, row, "R2 Spin I (l_I)", 'l_I', 1, 50, 0.5); row += 1
        self.create_slider(control_frame, row, "R2 Spin S (l_S)", 'l_S', 1, 50, 0.5); row += 1
        
        ttk.Label(control_frame, text="Spin Properties", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=3, pady=5, sticky=tk.W); row += 1
        self.create_slider(control_frame, row, "Offset I (ppm)", 'offset_I', -5.0, 5.0, 0.1); row += 1
        self.create_slider(control_frame, row, "Offset S (ppm)", 'offset_S', -5.0, 5.0, 0.1); row += 1
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
        val_label = ttk.Label(parent, text=f"{var.get():.1f}")
        val_label.grid(row=row, column=2, sticky=tk.E, pady=2)
        
        def update_lbl(v):
            val_label.config(text=f"{float(v):.1f}")
            self.update_plot()
            
        slider = ttk.Scale(parent, from_=min_val, to=max_val, variable=var, command=update_lbl, orient=tk.HORIZONTAL)
        slider.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=10)

    def update_plot(self, *args):
        self.ax.clear()
        p = {k: v.get() for k, v in self.params.items()}
        
        sim = TwoSpinCPMGSimulator(
            l_I=p['l_I'], l_S=p['l_S'], 
            offset_I_ppm=p['offset_I'], offset_S_ppm=p['offset_S'], 
            B0=p['B0'], J=p['J'], timeT2=p['timeT2'], ncycMax=int(p['ncycMax'])
        )
        
        vcpmg, R2eff = sim.generate_profile()
        
        self.ax.plot(vcpmg, R2eff, 'o-', color='darkred', linewidth=2, markersize=6)
        self.ax.set_xlabel('νCPMG (Hz)', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('R2eff (s⁻¹)', fontsize=12, fontweight='bold')
        self.ax.set_title('16x16 Homogeneous Matrix Dispersion Profile', fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = TwoSpinGUI(root)
    root.mainloop()