# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 08:23:20 2025

@author: shank
"""

"""
Exact CPMG Relaxation Dispersion Simulator with 7×7 Matrix Formulation
======================================================================
Models two-state exchange (G ⇌ E) with full relaxation dynamics including
longitudinal (R1) and transverse (R2) relaxation for both states.

Based on the full Liouville space representation:
State vector = [E/2, Ix^G, Iy^G, Iz^G, Ix^E, Iy^E, Iz^E]

Requirements:
    pip install numpy matplotlib scipy

Usage:
    python cpmg_simulator_7x7.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from scipy.linalg import expm
import csv


class CPMGSimulator7x7:
    """Exact CPMG simulation using 7×7 matrix for two-state exchange"""
    
    def __init__(self, R2G=10.0, R2E=10.0, R1G=1.5, R1E=1.5, 
                 kGE=500.0, kEG=50.0, dwG=0.0, dwE=1.0, 
                 B0=600.0, timeT2=30.0, ncycMax=30):
        """
        Initialize CPMG simulator with 7×7 matrix formulation
        
        Parameters:
        -----------
        R2G, R2E : float
            Transverse relaxation rates for ground (G) and excited (E) states (s^-1)
        R1G, R1E : float
            Longitudinal relaxation rates for ground and excited states (s^-1)
        kGE : float
            Exchange rate from G → E (s^-1)
        kEG : float
            Exchange rate from E → G (s^-1)
        dwG : float
            Offset of ground state from carrier (ppm)
        dwE : float
            Offset of excited state from carrier (ppm)
        B0 : float
            Magnetic field strength (MHz)
        timeT2 : float
            Total CPMG relaxation time (ms)
        ncycMax : int
            Maximum number of CPMG cycles
        """
        self.R2G = R2G
        self.R2E = R2E
        self.R1G = R1G
        self.R1E = R1E
        self.kGE = kGE
        self.kEG = kEG
        self.dwG = dwG
        self.dwE = dwE
        self.B0 = B0
        self.timeT2 = timeT2
        self.ncycMax = ncycMax
        
        # Calculate equilibrium populations
        kex = kGE + kEG
        self.pG = kEG / kex if kex > 0 else 0.95
        self.pE = kGE / kex if kex > 0 else 0.05
        
    def build_evolution_matrix(self, omega_sign=1):
        """
        Build the 7×7 evolution matrix for the two-state system
        
        The state vector is: [E/2, Ix^G, Iy^G, Iz^G, Ix^E, Iy^E, Iz^E]
        
        Parameters:
        -----------
        omega_sign : int
            +1 for positive frequency evolution, -1 for negative (after 180° pulse)
            
        Returns:
        --------
        M : 7×7 numpy array
            Evolution matrix
        """
        # Convert offsets from ppm to rad/s
        omega_G = self.dwG * self.B0 * 2.0 * np.pi * omega_sign
        omega_E = self.dwE * self.B0 * 2.0 * np.pi * omega_sign
        
        # For 15N at typical fields, omega_1 (RF field) is typically small
        # For off-resonance effects during pulses, but ~0 during free evolution
        omega_1 = 0.0  # Can be modified for on-resonance RF effects
        
        # Equilibrium magnetizations (Boltzmann populations)
        # For simplicity, assume thermal equilibrium proportional to populations
        Ieq_G = self.pG
        Ieq_E = self.pE
        
        # Build the 7×7 matrix following the equation structure
        M = np.zeros((7, 7), dtype=complex)
        
        # Row 0: E/2 (identity) - doesn't evolve
        M[0, 0] = 0
        
        # Row 1: Ix^G
        M[1, 1] = -self.R2G - self.kGE
        M[1, 2] = -omega_G
        M[1, 3] = omega_1
        M[1, 4] = self.kEG
        
        # Row 2: Iy^G
        M[2, 1] = omega_G
        M[2, 2] = -self.R2G - self.kGE
        M[2, 5] = self.kEG
        
        # Row 3: Iz^G
        M[3, 0] = 2 * self.R1G * Ieq_G
        M[3, 1] = -omega_1
        M[3, 3] = -self.R1G - self.kGE
        M[3, 6] = self.kEG
        
        # Row 4: Ix^E
        M[4, 1] = self.kGE
        M[4, 4] = -self.R2E - self.kEG
        M[4, 5] = -omega_E
        M[4, 6] = omega_1
        
        # Row 5: Iy^E
        M[5, 2] = self.kGE
        M[5, 4] = omega_E
        M[5, 5] = -self.R2E - self.kEG
        
        # Row 6: Iz^E
        M[6, 0] = 2 * self.R1E * Ieq_E
        M[6, 3] = self.kGE
        M[6, 4] = -omega_1
        M[6, 6] = -self.R1E - self.kEG
        
        return M
    
    def apply_180_pulse(self, state):
        """
        Apply ideal 180° pulse along x-axis
        Effect: Iy → -Iy, Iz → -Iz, Ix → Ix
        
        Parameters:
        -----------
        state : numpy array (7,)
            Current state vector
            
        Returns:
        --------
        new_state : numpy array (7,)
            State after 180° pulse
        """
        new_state = state.copy()
        # E/2 unchanged (index 0)
        # Ix^G unchanged (index 1)
        new_state[2] = -state[2]  # Iy^G → -Iy^G
        new_state[3] = -state[3]  # Iz^G → -Iz^G
        # Ix^E unchanged (index 4)
        new_state[5] = -state[5]  # Iy^E → -Iy^E
        new_state[6] = -state[6]  # Iz^E → -Iz^E
        
        return new_state
    
    def calculate_R2eff(self, ncyc, add_noise=False, noise_level=0.02):
        """
        Calculate R2eff using exact 7×7 matrix evolution
        
        CPMG sequence: 90°x - [τ - 180°x - τ]n
        We track the evolution through: τ(+ω) - 180° - τ(-ω) - 180° - ...
        
        Parameters:
        -----------
        ncyc : int
            Number of CPMG cycles (number of 180° pulses)
        add_noise : bool
            Add Gaussian noise to simulate experimental error
        noise_level : float
            Fractional noise level
            
        Returns:
        --------
        R2eff : float
            Effective transverse relaxation rate (s^-1)
        vcpmg : float
            CPMG frequency (Hz)
        """
        # Time for one τ period (in seconds)
        tcpmg = (self.timeT2 / 1000.0) / (4.0 * ncyc)
        
        # CPMG frequency
        vcpmg = 1.0 / (4.0 * tcpmg)
        
        # Build evolution matrices
        # M_plus: evolution with positive frequency
        # M_minus: evolution with negative frequency (equivalent to 180° refocusing)
        M_plus = self.build_evolution_matrix(omega_sign=+1)
        M_minus = self.build_evolution_matrix(omega_sign=-1)
        
        # Calculate matrix exponentials
        E_plus = expm(M_plus * tcpmg)
        E_minus = expm(M_minus * tcpmg)
        
        # Initial state after 90°x pulse
        # 90°x converts Iz to -Iy
        # State vector: [E/2, Ix^G, Iy^G, Iz^G, Ix^E, Iy^E, Iz^E]
        initial_state = np.array([
            0.5,      # E/2
            0.0,      # Ix^G = 0
            -self.pG, # Iy^G = -pG (from 90°x on Iz^G)
            0.0,      # Iz^G = 0 (converted to Iy)
            0.0,      # Ix^E = 0
            -self.pE, # Iy^E = -pE (from 90°x on Iz^E)
            0.0       # Iz^E = 0
        ], dtype=complex)
        
        state = initial_state.copy()
        
        # CPMG sequence: τ(+) - 180° - τ(-) - 180° - τ(+) - 180° - τ(-) - 180° ...
        # Each full cycle is: τ(+) - 180° - 2τ(-) - 180° - τ(+)
        # But we implement it as alternating τ periods with 180° pulses
        
        for k in range(ncyc):
            # First τ period (positive frequency)
            state = E_plus @ state
            
            # 180° pulse
            state = self.apply_180_pulse(state)
            
            # Second τ period (effectively negative frequency due to refocusing)
            state = E_minus @ state
            
            # 180° pulse
            state = self.apply_180_pulse(state)
            
            # Third τ period (positive frequency)
            state = E_plus @ state
            
            # 180° pulse
            state = self.apply_180_pulse(state)
            
            # Fourth τ period (negative frequency)
            state = E_minus @ state
            
            # 180° pulse (if not last cycle)
            if k < ncyc - 1:
                state = self.apply_180_pulse(state)
        
        # Extract transverse magnetization
        # We observe Iy (or total transverse magnetization)
        Iy_G_final = np.real(state[2])
        Iy_E_final = np.real(state[5])
        
        # Total observable transverse magnetization
        Iy_total_initial = -self.pG - self.pE  # Initial Iy after 90°
        Iy_total_final = Iy_G_final + Iy_E_final
        
        # Calculate R2eff from decay
        if np.abs(Iy_total_final) < 1e-10 or np.abs(Iy_total_initial) < 1e-10:
            return self.R2G, vcpmg
        
        # Handle sign issues (magnetization can become negative)
        ratio = np.abs(Iy_total_final / Iy_total_initial)
        
        if ratio <= 0 or ratio > 1:
            return self.R2G, vcpmg
        
        R2eff = -(1000.0 / self.timeT2) * np.log(ratio)
        
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
    """GUI for interactive CPMG simulation with 7×7 matrix"""
    
    def __init__(self, master):
        self.master = master
        master.title("CPMG Simulator - 7×7 Matrix Formulation (Two-State Exchange)")
        master.geometry("1500x900")
        
        # Initialize parameters
        self.params = {
            'R2G': tk.DoubleVar(value=10.0),
            'R2E': tk.DoubleVar(value=10.0),
            'R1G': tk.DoubleVar(value=1.5),
            'R1E': tk.DoubleVar(value=1.5),
            'kGE': tk.DoubleVar(value=500.0),
            'kEG': tk.DoubleVar(value=50.0),
            'dwG': tk.DoubleVar(value=0.0),
            'dwE': tk.DoubleVar(value=1.0),
            'B0': tk.DoubleVar(value=600.0),
            'timeT2': tk.DoubleVar(value=30.0),
            'ncycMax': tk.IntVar(value=30),
            'noise_level': tk.DoubleVar(value=0.02),
        }
        
        self.include_noise = tk.BooleanVar(value=False)
        
        self.setup_gui()
        self.update_plot()
        
    def setup_gui(self):
        """Setup the GUI layout"""
        # Main container
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Left panel - Controls (with scrollbar)
        control_canvas = tk.Canvas(main_frame, width=400)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=control_canvas.yview)
        control_frame = ttk.Frame(control_canvas)
        
        control_frame.bind(
            "<Configure>",
            lambda e: control_canvas.configure(scrollregion=control_canvas.bbox("all"))
        )
        
        control_canvas.create_window((0, 0), window=control_frame, anchor="nw")
        control_canvas.configure(yscrollcommand=scrollbar.set)
        
        control_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Right panel - Plot
        plot_frame = ttk.Frame(main_frame)
        plot_frame.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Configure grid weights
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)
        main_frame.columnconfigure(2, weight=3)
        main_frame.rowconfigure(0, weight=1)
        
        # Create parameter controls
        row = 0
        
        # Title
        ttk.Label(control_frame, text="7×7 Matrix Parameters", 
                 font=('Arial', 12, 'bold')).grid(
            row=row, column=0, columnspan=3, pady=(0, 10), sticky=tk.W)
        row += 1
        
        # Relaxation rates - Ground State
        ttk.Label(control_frame, text="Ground State (G) Relaxation", 
                 font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=3, pady=(5, 5), sticky=tk.W)
        row += 1
        
        self.create_slider(control_frame, row, "R2G (s⁻¹)", 'R2G', 1, 50, 0.5)
        row += 1
        self.create_slider(control_frame, row, "R1G (s⁻¹)", 'R1G', 0.1, 10, 0.1)
        row += 1
        
        ttk.Separator(control_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        # Relaxation rates - Excited State
        ttk.Label(control_frame, text="Excited State (E) Relaxation", 
                 font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=3, pady=(0, 5), sticky=tk.W)
        row += 1
        
        self.create_slider(control_frame, row, "R2E (s⁻¹)", 'R2E', 1, 50, 0.5)
        row += 1
        self.create_slider(control_frame, row, "R1E (s⁻¹)", 'R1E', 0.1, 10, 0.1)
        row += 1
        
        ttk.Separator(control_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        # Exchange rates
        ttk.Label(control_frame, text="Exchange Dynamics (G ⇌ E)", 
                 font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=3, pady=(0, 5), sticky=tk.W)
        row += 1
        
        self.create_slider(control_frame, row, "kGE (G→E, s⁻¹)", 'kGE', 10, 2000, 10)
        row += 1
        self.create_slider(control_frame, row, "kEG (E→G, s⁻¹)", 'kEG', 10, 2000, 10)
        row += 1
        
        # Display kex and populations
        info_frame = ttk.Frame(control_frame)
        info_frame.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=5)
        self.kex_label = ttk.Label(info_frame, text="", foreground='blue')
        self.kex_label.pack()
        row += 1
        
        ttk.Separator(control_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        # Chemical shifts
        ttk.Label(control_frame, text="Chemical Shifts (Offsets)", 
                 font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=3, pady=(0, 5), sticky=tk.W)
        row += 1
        
        self.create_slider(control_frame, row, "ωG (ppm)", 'dwG', -5.0, 5.0, 0.1)
        row += 1
        self.create_slider(control_frame, row, "ωE (ppm)", 'dwE', -5.0, 5.0, 0.1)
        row += 1
        
        # Display Δω
        self.dw_label = ttk.Label(control_frame, text="", foreground='blue')
        self.dw_label.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=5)
        row += 1
        
        ttk.Separator(control_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        # Experimental setup
        ttk.Label(control_frame, text="Experimental Setup", 
                 font=('Arial', 10, 'bold')).grid(
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
        
        # Noise
        ttk.Checkbutton(control_frame, text="Add Experimental Noise", 
                       variable=self.include_noise, command=self.update_plot).grid(
            row=row, column=0, columnspan=3, sticky=tk.W)
        row += 1
        
        self.create_slider(control_frame, row, "Noise (%)", 'noise_level', 0.001, 0.10, 0.005, scale=100)
        row += 1
        
        ttk.Separator(control_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=row, column=0, columnspan=3, pady=10)
        
        ttk.Button(button_frame, text="Update Plot", command=self.update_plot).pack(side=tk.TOP, pady=2)
        ttk.Button(button_frame, text="Export Data", command=self.export_data).pack(side=tk.TOP, pady=2)
        ttk.Button(button_frame, text="Preset: Slow Exchange", 
                  command=self.preset_slow).pack(side=tk.TOP, pady=2)
        ttk.Button(button_frame, text="Preset: Intermediate", 
                  command=self.preset_intermediate).pack(side=tk.TOP, pady=2)
        ttk.Button(button_frame, text="Preset: Fast Exchange", 
                  command=self.preset_fast).pack(side=tk.TOP, pady=2)
        row += 1
        
        # Info section
        info_text_frame = ttk.LabelFrame(control_frame, text="Interpretation", padding="5")
        info_text_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        self.info_text = tk.Text(info_text_frame, height=10, width=45, wrap=tk.WORD)
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
        
        # Update kex and population display
        kex = params['kGE'] + params['kEG']
        pG = params['kEG'] / kex if kex > 0 else 0.95
        pE = params['kGE'] / kex if kex > 0 else 0.05
        
        self.kex_label.config(text=f"kex = {kex:.1f} s⁻¹ | pG = {pG*100:.1f}% | pE = {pE*100:.1f}%")
        
        # Update Δω display
        dw = params['dwE'] - params['dwG']
        dw_hz = dw * params['B0']
        self.dw_label.config(text=f"Δω = {dw:.2f} ppm = {dw_hz:.1f} Hz")
        
        # Create simulator
        sim = CPMGSimulator7x7(
            R2G=params['R2G'],
            R2E=params['R2E'],
            R1G=params['R1G'],
            R1E=params['R1E'],
            kGE=params['kGE'],
            kEG=params['kEG'],
            dwG=params['dwG'],
            dwE=params['dwE'],
            B0=params['B0'],
            timeT2=params['timeT2'],
            ncycMax=int(params['ncycMax'])
        )
        
        # Generate profile
        data = sim.generate_profile(
            add_noise=self.include_noise.get(),
            noise_level=params['noise_level']
        )
        
        # Plot
        if self.include_noise.get():
            self.ax.errorbar(data['vcpmg'], data['R2eff'], yerr=data['error'],
                           fmt='o', color='darkblue', label='R2eff (with noise)', 
                           capsize=4, markersize=7, linewidth=2)
        else:
            self.ax.plot(data['vcpmg'], data['R2eff'], 'o-', 
                        color='darkblue', linewidth=2.5, markersize=7, 
                        label='R2eff', markerfacecolor='lightblue', markeredgewidth=2)
        
        # Formatting
        self.ax.set_xlabel('νCPMG (Hz)', fontsize=13, fontweight='bold')
        self.ax.set_ylabel('R2eff (s⁻¹)', fontsize=13, fontweight='bold')
        self.ax.set_title('CPMG Relaxation Dispersion Profile (7×7 Matrix Model)', 
                         fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.ax.legend(fontsize=11)
        
        # Add horizontal line at R2G for reference
        self.ax.axhline(y=params['R2G'], color='red', linestyle='--', 
                       alpha=0.5, label=f'R2G = {params["R2G"]:.1f}')
        self.ax.legend(fontsize=10)
        
        self.canvas.draw()
        
        # Update info text
        self.update_info(sim, data, params)
        
    def update_info(self, sim, data, params):
        """Update interpretation info"""
        self.info_text.delete(1.0, tk.END)
        
        kex = params['kGE'] + params['kEG']
        dw = (params['dwE'] - params['dwG']) * params['B0']
        
        ratio = kex / (np.abs(dw) * 2 * np.pi) if dw != 0 else 1000
        
        if ratio > 2:
            regime = "Fast exchange (kex >> Δω)"
        elif ratio > 0.5:
            regime = "Intermediate exchange (kex ≈ Δω)"
        else:
            regime = "Slow exchange (kex << Δω)"
        
        R2ex_max = np.max(data['R2eff']) - params['R2G']
        
        info = f"""Exchange Regime: {regime}

Two-State Model (G ⇌ E):
  kGE = {params['kGE']:.1f} s⁻¹
  kEG = {params['kEG']:.1f} s⁻¹
  kex = {kex:.1f} s⁻¹
  
Populations:
  pG = {sim.pG*100:.1f}%
  pE = {sim.pE*100:.1f}%

Chemical Shifts:
  ωG = {params['dwG']:.2f} ppm
  ωE = {params['dwE']:.2f} ppm
  Δω = {dw:.1f} Hz at {params['B0']:.0f} MHz

Relaxation Rates:
  R2G = {params['R2G']:.1f} s⁻¹
  R2E = {params['R2E']:.1f} s⁻¹
  R1G = {params['R1G']:.1f} s⁻¹
  R1E = {params['R1E']:.1f} s⁻¹

Exchange Contribution:
  R2,ex (max) = {R2ex_max:.2f} s⁻¹

CPMG Parameters:
  vcpmg range: {np.min(data['vcpmg']):.1f} - {np.max(data['vcpmg']):.1f} Hz
  T2 relaxation time: {params['timeT2']:.0f} ms
"""
        
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
        sim = CPMGSimulator7x7(
            R2G=params['R2G'],
            R2E=params['R2E'],
            R1G=params['R1G'],
            R1E=params['R1E'],
            kGE=params['kGE'],
            kEG=params['kEG'],
            dwG=params['dwG'],
            dwE=params['dwE'],
            B0=params['B0'],
            timeT2=params['timeT2'],
            ncycMax=int(params['ncycMax'])
        )
        
        data = sim.generate_profile(
            add_noise=self.include_noise.get(),
            noise_level=params['noise_level']
        )
        
        kex = params['kGE'] + params['kEG']
        
        # Write to CSV
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['# CPMG Relaxation Dispersion Data - 7x7 Matrix Model'])
            writer.writerow([f'# Two-State Exchange: G ⇌ E'])
            writer.writerow([f'# kGE={params["kGE"]:.1f}, kEG={params["kEG"]:.1f}, kex={kex:.1f} s-1'])
            writer.writerow([f'# pG={sim.pG:.3f}, pE={sim.pE:.3f}'])
            writer.writerow([f'# R2G={params["R2G"]:.1f}, R2E={params["R2E"]:.1f} s-1'])
            writer.writerow([f'# R1G={params["R1G"]:.1f}, R1E={params["R1E"]:.1f} s-1'])
            writer.writerow([f'# dwG={params["dwG"]:.2f}, dwE={params["dwE"]:.2f} ppm'])
            writer.writerow([f'# B0={params["B0"]:.0f} MHz'])
            writer.writerow([''])
            writer.writerow(['ncyc', 'vcpmg(Hz)', 'R2eff(s-1)', 'error(s-1)'])
            
            for i in range(len(data['ncyc'])):
                writer.writerow([
                    data['ncyc'][i],
                    f"{data['vcpmg'][i]:.4f}",
                    f"{data['R2eff'][i]:.6f}",
                    f"{data['error'][i]:.6f}"
                ])
        
        messagebox.showinfo("Export Successful", f"Data exported to:\n{filename}")
        
    def preset_slow(self):
        """Load slow exchange preset"""
        self.params['kGE'].set(50)
        self.params['kEG'].set(500)
        self.params['dwG'].set(0.0)
        self.params['dwE'].set(2.0)
        self.params['R2G'].set(10)
        self.params['R2E'].set(10)
        self.params['R1G'].set(1.5)
        self.params['R1E'].set(1.5)
        self.update_plot()
        
    def preset_intermediate(self):
        """Load intermediate exchange preset"""
        self.params['kGE'].set(500)
        self.params['kEG'].set(50)
        self.params['dwG'].set(0.0)
        self.params['dwE'].set(1.0)
        self.params['R2G'].set(10)
        self.params['R2E'].set(12)
        self.params['R1G'].set(1.5)
        self.params['R1E'].set(1.8)
        self.update_plot()
        
    def preset_fast(self):
        """Load fast exchange preset"""
        self.params['kGE'].set(1500)
        self.params['kEG'].set(500)
        self.params['dwG'].set(0.0)
        self.params['dwE'].set(0.5)
        self.params['R2G'].set(15)
        self.params['R2E'].set(15)
        self.params['R1G'].set(1.5)
        self.params['R1E'].set(1.5)
        self.update_plot()


def main():
    """Main entry point"""
    root = tk.Tk()
    app = CPMGSimulatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()