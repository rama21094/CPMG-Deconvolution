# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 11:16:52 2025

@author: shank
"""

"""
Exact CPMG Relaxation Dispersion Simulator
==========================================
Uses matrix exponential approach to solve Bloch-McConnell equations
Based on rigorous numerical methods for publication-quality simulations

Requirements:
    pip install numpy matplotlib scipy

Usage:
    python cpmg_simulator.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from scipy.linalg import expm
import csv


class CPMGSimulator:
    """Exact CPMG simulation using matrix exponentials"""
    
    def __init__(self, R2A=10.0, R2B=10.0, kex=750.0, pB=0.03, 
                 dwH=1.0, B0=600.0, timeT2=30.0, ncycMax=30, 
                 J=0.0, includeJCoupling=False):
        """
        Initialize CPMG simulator parameters
        
        Parameters:
        -----------
        R2A : float
            Intrinsic R2 at site A (s^-1)
        R2B : float
            Intrinsic R2 at site B (s^-1)
        kex : float
            Exchange rate kex = kAB + kBA (s^-1)
        pB : float
            Population of minor state B (0-1)
        dwH : float
            Chemical shift difference (ppm)
        B0 : float
            Magnetic field strength (MHz)
        timeT2 : float
            Total CPMG relaxation time (ms)
        ncycMax : int
            Maximum number of CPMG cycles
        J : float
            J-coupling constant (Hz)
        includeJCoupling : bool
            Include J-coupling in simulation
        """
        self.R2A = R2A
        self.R2B = R2B
        self.kex = kex
        self.pB = pB
        self.dwH = dwH
        self.B0 = B0
        self.timeT2 = timeT2
        self.ncycMax = ncycMax
        self.J = J
        self.includeJCoupling = includeJCoupling
        
    def calculate_R2eff(self, ncyc, add_noise=False, noise_level=0.02):
        """
        Calculate R2eff for a given number of CPMG cycles using exact matrix exponentials
        
        This implements the exact solution to the Bloch-McConnell equations:
        I(t) = [EMP * EMN * EMN * EMP]^ncyc * I0
        
        Parameters:
        -----------
        ncyc : int
            Number of CPMG cycles
        add_noise : bool
            Add Gaussian noise to simulate experimental error
        noise_level : float
            Fractional noise level (e.g., 0.02 = 2%)
            
        Returns:
        --------
        R2eff : float
            Effective transverse relaxation rate (s^-1)
        vcpmg : float
            CPMG frequency (Hz)
        """
        pA = 1.0 - self.pB
        
        # Exchange rate constants
        kA = self.kex * self.pB  # kAB
        kB = self.kex * pA       # kBA
        
        # Chemical shift in rad/s
        dw = self.dwH * self.B0 * 2.0 * np.pi
        
        # J-coupling in rad/s
        dwJ = self.J * 2.0 * np.pi if self.includeJCoupling else 0.0
        
        # Time for one tau period (in seconds)
        tcpmg = (self.timeT2 / 1000.0) / (4.0 * ncyc)
        
        # CPMG frequency
        vcpmg = 1.0 / (4.0 * tcpmg)
        
        # Build Bloch-McConnell evolution matrices
        # BMP: evolution with +i*omega (positive frequency)
        BMP = np.array([
            [-self.R2A - kA,          kB],
            [kA,                      -self.R2B - kB + 1j*(dw + dwJ)]
        ], dtype=complex)
        
        # BMN: evolution with -i*omega (negative frequency, after 180° pulse)
        BMN = np.array([
            [-self.R2A - kA,          kB],
            [kA,                      -self.R2B - kB - 1j*(dw + dwJ)]
        ], dtype=complex)
        
        # Calculate matrix exponentials using scipy
        EMP = expm(BMP * tcpmg)
        EMN = expm(BMN * tcpmg)
        
        # Initial magnetization (equilibrium populations)
        I0 = np.array([pA, self.pB], dtype=complex)
        I = I0.copy()
        
        # Apply CPMG sequence: [EMP * EMN * EMN * EMP]^ncyc
        # One cycle = tau(+w) - 180 - 2tau(-w) - 180 - tau(+w)
        for k in range(ncyc):
            I = EMP @ I          # tau with +frequency
            I = EMN @ EMN @ I    # 2*tau with -frequency (after 180°)
            I = EMP @ I          # tau with +frequency (after second 180°)
        
        # Calculate R2eff from decay of site A magnetization
        I_final = np.real(I[0])
        I0_val = np.real(I0[0])
        
        if I_final <= 0 or I0_val <= 0:
            return self.R2A, vcpmg
        
        R2eff = (1000.0 / self.timeT2) * np.log(I0_val / I_final)
        
        # Add noise if requested
        if add_noise:
            noise = noise_level * R2eff * np.random.randn()
            R2eff += noise
        
        return R2eff, vcpmg
    
    def generate_profile(self, add_noise=False, noise_level=0.02):
        """
        Generate full CPMG relaxation dispersion profile
        
        Returns:
        --------
        data : dict
            Dictionary containing 'ncyc', 'vcpmg', 'R2eff', 'error'
        """
        ncyc_values = np.arange(1, self.ncycMax + 1)
        vcpmg_values = []
        R2eff_values = []
        error_values = []
        
        for ncyc in ncyc_values:
            R2eff, vcpmg = self.calculate_R2eff(ncyc, add_noise, noise_level)
            vcpmg_values.append(vcpmg)
            R2eff_values.append(R2eff)
            error_values.append(noise_level * R2eff if add_noise else 0.0)
        
        return {
            'ncyc': ncyc_values,
            'vcpmg': np.array(vcpmg_values),
            'R2eff': np.array(R2eff_values),
            'error': np.array(error_values)
        }


class CPMGSimulatorGUI:
    """GUI for interactive CPMG simulation"""
    
    def __init__(self, master):
        self.master = master
        master.title("Exact CPMG Relaxation Dispersion Simulator")
        master.geometry("1400x900")
        
        # Initialize parameters
        self.params = {
            'R2A': tk.DoubleVar(value=10.0),
            'R2B': tk.DoubleVar(value=10.0),
            'kex': tk.DoubleVar(value=750.0),
            'pB': tk.DoubleVar(value=0.03),
            'dwH': tk.DoubleVar(value=1.0),
            'B0': tk.DoubleVar(value=600.0),
            'timeT2': tk.DoubleVar(value=30.0),
            'ncycMax': tk.IntVar(value=30),
            'J': tk.DoubleVar(value=0.0),
            'noise_level': tk.DoubleVar(value=0.02),
        }
        
        self.include_noise = tk.BooleanVar(value=False)
        self.include_J = tk.BooleanVar(value=False)
        self.show_comparison = tk.BooleanVar(value=False)
        
        self.setup_gui()
        self.update_plot()
        
    def setup_gui(self):
        """Setup the GUI layout"""
        # Main container
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Left panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Parameters", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Right panel - Plot
        plot_frame = ttk.Frame(main_frame)
        plot_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Configure grid weights
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=3)
        main_frame.rowconfigure(0, weight=1)
        
        # Create parameter controls
        row = 0
        
        # Exchange parameters section
        ttk.Label(control_frame, text="Exchange Dynamics", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=3, pady=(0, 5), sticky=tk.W)
        row += 1
        
        self.create_slider(control_frame, row, "kex (s⁻¹)", 'kex', 50, 5000, 50)
        row += 1
        self.create_slider(control_frame, row, "pB (%)", 'pB', 0.005, 0.30, 0.005, scale=100)
        row += 1
        self.create_slider(control_frame, row, "Δω (ppm)", 'dwH', 0.1, 15.0, 0.1)
        row += 1
        
        ttk.Separator(control_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        # Relaxation parameters section
        ttk.Label(control_frame, text="Relaxation Rates", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=3, pady=(0, 5), sticky=tk.W)
        row += 1
        
        self.create_slider(control_frame, row, "R2A (s⁻¹)", 'R2A', 1, 50, 0.5)
        row += 1
        self.create_slider(control_frame, row, "R2B (s⁻¹)", 'R2B', 1, 50, 0.5)
        row += 1
        
        ttk.Separator(control_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        # Experimental setup section
        ttk.Label(control_frame, text="Experimental Setup", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=3, pady=(0, 5), sticky=tk.W)
        row += 1
        
        self.create_slider(control_frame, row, "B0 (MHz)", 'B0', 400, 950, 50)
        row += 1
        self.create_slider(control_frame, row, "T2 time (ms)", 'timeT2', 10, 100, 5)
        row += 1
        self.create_slider(control_frame, row, "Max cycles", 'ncycMax', 10, 50, 1)
        row += 1
        
        ttk.Separator(control_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        # J-coupling section
        ttk.Checkbutton(control_frame, text="Include J-Coupling", 
                       variable=self.include_J, command=self.update_plot).grid(
            row=row, column=0, columnspan=3, sticky=tk.W)
        row += 1
        
        self.create_slider(control_frame, row, "J (Hz)", 'J', 0, 200, 5)
        row += 1
        
        ttk.Separator(control_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        # Noise section
        ttk.Checkbutton(control_frame, text="Add Experimental Noise", 
                       variable=self.include_noise, command=self.update_plot).grid(
            row=row, column=0, columnspan=3, sticky=tk.W)
        row += 1
        
        self.create_slider(control_frame, row, "Noise (%)", 'noise_level', 0.001, 0.10, 0.005, scale=100)
        row += 1
        
        ttk.Separator(control_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        # Comparison option
        ttk.Checkbutton(control_frame, text="Show comparison (with/without J)", 
                       variable=self.show_comparison, command=self.update_plot).grid(
            row=row, column=0, columnspan=3, sticky=tk.W)
        row += 1
        
        ttk.Separator(control_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=row, column=0, columnspan=3, pady=10)
        
        ttk.Button(button_frame, text="Update Plot", command=self.update_plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Export Data", command=self.export_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Preset: Fast Exchange", 
                  command=self.preset_fast).pack(side=tk.LEFT, padx=5)
        row += 1
        
        ttk.Button(button_frame, text="Preset: Intermediate", 
                  command=self.preset_intermediate).pack(side=tk.LEFT, padx=5)
        
        # Info section
        row += 1
        info_frame = ttk.LabelFrame(control_frame, text="Interpretation", padding="5")
        info_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        self.info_text = tk.Text(info_frame, height=8, width=40, wrap=tk.WORD)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        # Setup matplotlib figure
        self.fig = Figure(figsize=(10, 7))
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_slider(self, parent, row, label, var_name, min_val, max_val, step, scale=1):
        """Create a labeled slider"""
        var = self.params[var_name]
        
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=2)
        
        value_label = ttk.Label(parent, text=f"{var.get()*scale:.2f}")
        value_label.grid(row=row, column=2, sticky=tk.E, pady=2)
        
        def update_label(val):
            value_label.config(text=f"{float(val)*scale:.2f}")
            self.update_plot()
        
        slider = ttk.Scale(parent, from_=min_val, to=max_val, 
                          variable=var, command=update_label, orient=tk.HORIZONTAL)
        slider.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=5)
        
        parent.columnconfigure(1, weight=1)
        
    def update_plot(self, *args):
        """Update the plot with current parameters"""
        self.ax.clear()
        
        # Get parameters
        params = {key: var.get() for key, var in self.params.items()}
        
        # Create simulator
        sim = CPMGSimulator(
            R2A=params['R2A'],
            R2B=params['R2B'],
            kex=params['kex'],
            pB=params['pB'],
            dwH=params['dwH'],
            B0=params['B0'],
            timeT2=params['timeT2'],
            ncycMax=int(params['ncycMax']),
            J=params['J'],
            includeJCoupling=self.include_J.get()
        )
        
        # Generate profile
        data = sim.generate_profile(
            add_noise=self.include_noise.get(),
            noise_level=params['noise_level']
        )
        
        # Plot main data
        if self.include_noise.get():
            self.ax.errorbar(data['vcpmg'], data['R2eff'], yerr=data['error'],
                           fmt='o', color='blue', label='With noise', 
                           capsize=3, markersize=6)
        else:
            self.ax.plot(data['vcpmg'], data['R2eff'], 'o-', 
                        color='blue', linewidth=2, markersize=6, 
                        label='R2eff')
        
        # Show comparison if requested
        if self.show_comparison.get():
            sim_noJ = CPMGSimulator(
                R2A=params['R2A'],
                R2B=params['R2B'],
                kex=params['kex'],
                pB=params['pB'],
                dwH=params['dwH'],
                B0=params['B0'],
                timeT2=params['timeT2'],
                ncycMax=int(params['ncycMax']),
                J=0.0,
                includeJCoupling=False
            )
            data_noJ = sim_noJ.generate_profile(add_noise=False)
            self.ax.plot(data_noJ['vcpmg'], data_noJ['R2eff'], 's--', 
                        color='orange', linewidth=2, markersize=5,
                        label='Without J-coupling')
        
        # Formatting
        self.ax.set_xlabel('νCPMG (Hz)', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('R2eff (s⁻¹)', fontsize=12, fontweight='bold')
        self.ax.set_title('CPMG Relaxation Dispersion Profile', fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        self.canvas.draw()
        
        # Update info text
        self.update_info(sim, data)
        
    def update_info(self, sim, data):
        """Update interpretation info"""
        self.info_text.delete(1.0, tk.END)
        
        dw_hz = sim.dwH * sim.B0
        ratio = sim.kex / (dw_hz * 2 * np.pi)
        
        if ratio > 2:
            regime = "Fast exchange (kex >> Δω)"
        elif ratio > 0.5:
            regime = "Intermediate exchange (kex ≈ Δω)"
        else:
            regime = "Slow exchange (kex << Δω)"
        
        R2ex_max = np.max(data['R2eff']) - sim.R2A
        pA = 1 - sim.pB
        
        info = f"""Exchange Regime: {regime}

Δω = {dw_hz:.1f} Hz at {sim.B0:.0f} MHz
kex = {sim.kex:.0f} s⁻¹
pA = {pA*100:.1f}%, pB = {sim.pB*100:.1f}%

R2,ex (max) = {R2ex_max:.2f} s⁻¹
ΔR2 = {sim.R2B - sim.R2A:.1f} s⁻¹

vcpmg range: {np.min(data['vcpmg']):.1f} - {np.max(data['vcpmg']):.1f} Hz
"""
        
        if sim.includeJCoupling and sim.J > 0:
            info += f"\nJ-coupling: {sim.J:.1f} Hz (included)"
        
        self.info_text.insert(1.0, info)
        
    def export_data(self):
        """Export data to CSV file"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        # Get current data
        params = {key: var.get() for key, var in self.params.items()}
        sim = CPMGSimulator(
            R2A=params['R2A'],
            R2B=params['R2B'],
            kex=params['kex'],
            pB=params['pB'],
            dwH=params['dwH'],
            B0=params['B0'],
            timeT2=params['timeT2'],
            ncycMax=int(params['ncycMax']),
            J=params['J'],
            includeJCoupling=self.include_J.get()
        )
        
        data = sim.generate_profile(
            add_noise=self.include_noise.get(),
            noise_level=params['noise_level']
        )
        
        # Write to CSV
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['# CPMG Relaxation Dispersion Data'])
            writer.writerow([f'# kex={sim.kex}, pB={sim.pB}, dw={sim.dwH} ppm'])
            writer.writerow(['ncyc', 'vcpmg(Hz)', 'R2eff(s-1)', 'error(s-1)'])
            
            for i in range(len(data['ncyc'])):
                writer.writerow([
                    data['ncyc'][i],
                    f"{data['vcpmg'][i]:.4f}",
                    f"{data['R2eff'][i]:.6f}",
                    f"{data['error'][i]:.6f}"
                ])
        
        messagebox.showinfo("Export Successful", f"Data exported to:\n{filename}")
        
    def preset_fast(self):
        """Load fast exchange preset"""
        self.params['kex'].set(2000)
        self.params['pB'].set(0.05)
        self.params['dwH'].set(2.0)
        self.params['R2A'].set(15)
        self.params['R2B'].set(15)
        self.update_plot()
        
    def preset_intermediate(self):
        """Load intermediate exchange preset"""
        self.params['kex'].set(750)
        self.params['pB'].set(0.03)
        self.params['dwH'].set(1.0)
        self.params['R2A'].set(10)
        self.params['R2B'].set(10)
        self.update_plot()


def main():
    """Main entry point"""
    root = tk.Tk()
    app = CPMGSimulatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()