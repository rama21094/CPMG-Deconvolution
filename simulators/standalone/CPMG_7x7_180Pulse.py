# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 09:00:49 2025

@author: shank
"""

"""
CPMG Relaxation Dispersion Simulator with Finite Pulse Widths
==============================================================
7×7 matrix formulation including realistic pulse durations

This version properly accounts for:
- Finite 180° pulse widths
- Evolution during pulses (chemical shift, exchange, relaxation)
- RF field effects during pulses
- Off-resonance effects

Requirements:
    pip install numpy matplotlib scipy

Usage:
    python cpmg_simulator_finite_pulse.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from scipy.linalg import expm
import csv


class CPMGSimulatorFinitePulse:
    """CPMG simulation with finite pulse widths using 7×7 matrix"""
    
    def __init__(self, R2G=10.0, R2E=10.0, R1G=1.5, R1E=1.5, 
                 kGE=500.0, kEG=50.0, dwG=0.0, dwE=1.0, 
                 B0=600.0, timeT2=30.0, ncycMax=30,
                 pulse_width=80.0, omega1=15000.0, use_finite_pulses=True):
        """
        Initialize CPMG simulator with finite pulse widths
        
        Parameters:
        -----------
        R2G, R2E : float
            Transverse relaxation rates (s^-1)
        R1G, R1E : float
            Longitudinal relaxation rates (s^-1)
        kGE, kEG : float
            Exchange rates (s^-1)
        dwG, dwE : float
            Chemical shift offsets (ppm)
        B0 : float
            Magnetic field (MHz)
        timeT2 : float
            Total CPMG time (ms)
        ncycMax : int
            Maximum CPMG cycles
        pulse_width : float
            180° pulse width (μs)
        omega1 : float
            RF field strength (Hz) - typically 10-25 kHz
        use_finite_pulses : bool
            If True, use finite pulses; if False, instantaneous
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
        self.pulse_width = pulse_width  # in μs
        self.omega1 = np.pi/2 * (self.pulse_width / 1e6) # in Hz
        self.use_finite_pulses = use_finite_pulses
        
        # Calculate populations
        kex = kGE + kEG
        self.pG = kEG / kex if kex > 0 else 0.95
        self.pE = kGE / kex if kex > 0 else 0.05
        
    def build_evolution_matrix(self, omega_sign=1, rf_on=False):
        """
        Build 7×7 evolution matrix
        
        Parameters:
        -----------
        omega_sign : int
            +1 or -1 for frequency evolution direction
        rf_on : bool
            True during pulses (ω1 ≠ 0), False during free evolution
            
        Returns:
        --------
        M : 7×7 complex array
        """
        # Chemical shifts in rad/s
        omega_G = self.dwG * self.B0 * 2.0 * np.pi * omega_sign * 1e-6 * 1/9.87
        omega_E = self.dwE * self.B0 * 2.0 * np.pi * omega_sign * 1e-6 * 1/9.87
        
        # RF field in rad/s (only active during pulses)
        omega_1 = self.omega1 * 2.0 * np.pi if rf_on else 0.0
        
        # Equilibrium magnetizations
        Ieq_G = self.pG
        Ieq_E = self.pE
        
        # Build 7×7 matrix
        M = np.zeros((7, 7), dtype=complex)
        
        # Row 0: E/2 (identity)
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
    
    def apply_finite_180_pulse(self, state, tau_pw):
        """
        Apply finite-width 180° pulse by evolving under RF field
        
        During the pulse:
        - RF field ω1 is ON
        - Chemical shift evolution continues
        - Exchange continues
        - Relaxation continues
        
        Parameters:
        -----------
        state : array (7,)
            Current state vector
        tau_pw : float
            Pulse width in seconds
            
        Returns:
        --------
        new_state : array (7,)
            State after pulse
        """
        # Build matrix with RF field ON
        M_pulse = self.build_evolution_matrix(omega_sign=1, rf_on=True)
        # print(M_pulse)
        
        # Evolve for pulse duration
        E_pulse = expm(M_pulse * tau_pw)
        new_state = E_pulse @ state
        
        return new_state
    
    def apply_instantaneous_180_pulse(self, state):
        """
        Apply ideal instantaneous 180°x pulse
        Effect: Iy → -Iy, Iz → -Iz, Ix → Ix
        """
        new_state = state.copy()
        new_state[2] = -state[2]  # Iy^G → -Iy^G
        new_state[3] = -state[3]  # Iz^G → -Iz^G
        new_state[5] = -state[5]  # Iy^E → -Iy^E
        new_state[6] = -state[6]  # Iz^E → -Iz^E
        return new_state
    
    def calculate_R2eff(self, ncyc, add_noise=False, noise_level=0.02):
        """
        Calculate R2eff with finite or instantaneous pulse widths
        
        CPMG sequence timing:
        90°x - τ - [180°x(τpw) - τ]n
        
        For finite pulses: τ_eff = τ - τpw/2
        For instantaneous pulses: τ_eff = τ, τpw = 0
        
        Returns:
        --------
        R2eff : float
        vcpmg : float
        tau_effective : float
        """
        # Pulse width in seconds
        tau_pw = self.pulse_width / 1e6 if self.use_finite_pulses else 0.0
        
        # Total relaxation time
        total_time = self.timeT2 / 1000.0  # convert ms to s
        
        # For CPMG: time = ncyc * [4*tau + 4*tau_pw]
        # We want: ncyc * [4*tau + 4*tau_pw] = total_time
        
        if self.use_finite_pulses:
            # Total time includes pulse widths
            # ncyc * (4*tau + 4*tau_pw) = total_time
            # tau = (total_time/ncyc - 4*tau_pw) / 4
            tau = (total_time / ncyc - 4 * tau_pw) / 4.0
            tau_eff = tau  # Free evolution time
            
            if tau <= 0:
                # Pulse widths exceed total time - not physical
                return self.R2G, 0.0, 0.0
        else:
            # Instantaneous pulses: no time spent in pulses
            # ncyc * 4*tau = total_time
            tau = total_time / (4.0 * ncyc)
            tau_eff = tau
        
        # CPMG frequency: 1/(4*tau) where tau is center-to-center spacing
        if self.use_finite_pulses:
            vcpmg = 1.0 / (4.0 * (tau + tau_pw))
        else:
            vcpmg = 1.0 / (4.0 * tau)
        
        # Build evolution matrices for free evolution
        M_plus = self.build_evolution_matrix(omega_sign=+1, rf_on=False)
        M_minus = self.build_evolution_matrix(omega_sign=-1, rf_on=False)
        
        # Calculate propagators for free evolution
        if self.use_finite_pulses:
            E_plus = expm(M_plus * tau_eff)
            E_minus = expm(M_minus * tau_eff)
        else:
            E_plus = expm(M_plus * tau)
            E_minus = expm(M_minus * tau)
        
        # Initial state after 90°x pulse
        initial_state = np.array([
            0.5,      # E/2
            0.0,      # Ix^G = 0
            -self.pG, # Iy^G = -pG
            0.0,      # Iz^G = 0
            0.0,      # Ix^E = 0
            -self.pE, # Iy^E = -pE
            0.0       # Iz^E = 0
        ], dtype=complex)
        
        state = initial_state.copy()
        
        # CPMG sequence: τ - 180° - τ - 180° - τ - 180° - τ - 180° ...
        # Each cycle: τ(+) - 180° - 2τ(-) - 180° - τ(+)
        # Implemented as: τ(+) - 180° - τ(-) - 180° - τ(+) - 180° - τ(-) - 180°
        
        for k in range(ncyc):
            # First τ (positive frequency)
            state = E_plus @ state
            
            # 180° pulse
            if self.use_finite_pulses:
                state = self.apply_finite_180_pulse(state, 2*tau_pw)
            else:
                state = self.apply_instantaneous_180_pulse(state)
            
            # Second τ (negative frequency)
            state = E_minus @ state
            
            # 180° pulse
            if self.use_finite_pulses:
                state = self.apply_finite_180_pulse(state, 2*tau_pw)
            else:
                state = self.apply_instantaneous_180_pulse(state)
            
            # Third τ (positive frequency)
            state = E_plus @ state
            
            # 180° pulse
            if self.use_finite_pulses:
                state = self.apply_finite_180_pulse(state, 2*tau_pw)
            else:
                state = self.apply_instantaneous_180_pulse(state)
            
            # Fourth τ (negative frequency)
            state = E_minus @ state
            
            # 180° pulse (except last cycle)
            if k < ncyc - 1:
                if self.use_finite_pulses:
                    state = self.apply_finite_180_pulse(state, 2*tau_pw)
                else:
                    state = self.apply_instantaneous_180_pulse(state)
        
        # Extract observable magnetization
        Iy_total_initial = -self.pG #- self.pE
        Iy_total_final = np.real(state[2]) #+ np.real(state[5])
        
        ratio = np.abs(Iy_total_final / Iy_total_initial)
        
        if ratio <= 0 or ratio > 1:
            return self.R2G, vcpmg, tau_eff
        
        R2eff = -(1000.0 / self.timeT2) * np.log(ratio)
        
        if add_noise:
            noise = noise_level * R2eff * np.random.randn()
            R2eff += noise
        
        return R2eff, vcpmg, tau_eff
    
    def generate_profile(self, add_noise=False, noise_level=0.02):
        """Generate CPMG profile"""
        ncyc_values = np.arange(1, self.ncycMax + 1)
        vcpmg_values = []
        R2eff_values = []
        error_values = []
        tau_eff_values = []
        
        for ncyc in ncyc_values:
            R2eff, vcpmg, tau_eff = self.calculate_R2eff(ncyc, add_noise, noise_level)
            vcpmg_values.append(vcpmg)
            R2eff_values.append(R2eff)
            error_values.append(noise_level * R2eff if add_noise else 0.0)
            tau_eff_values.append(tau_eff)
        
        return {
            'ncyc': ncyc_values,
            'vcpmg': np.array(vcpmg_values),
            'R2eff': np.array(R2eff_values),
            'error': np.array(error_values),
            'tau_eff': np.array(tau_eff_values)
        }


class CPMGSimulatorGUI:
    """GUI for CPMG simulator with finite pulse options"""
    
    def __init__(self, master):
        self.master = master
        master.title("CPMG Simulator - Finite Pulse Widths (7×7 Matrix)")
        master.geometry("1600x900")
        
        # Parameters
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
            'pulse_width': tk.DoubleVar(value=80.0),
            'omega1': tk.DoubleVar(value=15000.0),
            'noise_level': tk.DoubleVar(value=0.02),
        }
        
        self.include_noise = tk.BooleanVar(value=False)
        self.use_finite_pulses = tk.BooleanVar(value=True)
        self.show_comparison = tk.BooleanVar(value=False)
        
        self.setup_gui()
        self.update_plot()
        
    def setup_gui(self):
        """Setup GUI layout"""
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Left panel with scrollbar
        control_canvas = tk.Canvas(main_frame, width=450)
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
        
        # Configure weights
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)
        main_frame.columnconfigure(2, weight=3)
        main_frame.rowconfigure(0, weight=1)
        
        row = 0
        
        # Title
        ttk.Label(control_frame, text="Finite Pulse Width Model", 
                 font=('Arial', 12, 'bold')).grid(
            row=row, column=0, columnspan=3, pady=(0, 10), sticky=tk.W)
        row += 1
        
        # Pulse parameters section (HIGHLIGHTED)
        pulse_frame = ttk.LabelFrame(control_frame, text="⚡ Pulse Parameters (NEW)", 
                                     padding="10")
        pulse_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        pulse_row = 0
        
        ttk.Checkbutton(pulse_frame, text="Use Finite Pulse Widths (vs Instantaneous)", 
                       variable=self.use_finite_pulses, 
                       command=self.update_plot).grid(
            row=pulse_row, column=0, columnspan=3, sticky=tk.W, pady=5)
        pulse_row += 1
        
        self.create_slider_in_frame(pulse_frame, pulse_row, "180° Pulse Width (μs)", 
                                    'pulse_width', 20, 200, 5)
        pulse_row += 1
        
        self.create_slider_in_frame(pulse_frame, pulse_row, "ω1 / 2π (kHz)", 
                                    'omega1', 5000, 30000, 1000, scale=0.001)
        pulse_row += 1
        
        ttk.Label(pulse_frame, text="Typical: τpw = 50-100 μs, ω1 = 10-25 kHz", 
                 font=('Arial', 8, 'italic'), foreground='gray').grid(
            row=pulse_row, column=0, columnspan=3, sticky=tk.W)
        
        row += 1
        
        # Comparison option
        ttk.Checkbutton(control_frame, text="Show Comparison (Finite vs Instantaneous)", 
                       variable=self.show_comparison, command=self.update_plot).grid(
            row=row, column=0, columnspan=3, sticky=tk.W, pady=5)
        row += 1
        
        ttk.Separator(control_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        # Exchange parameters
        ttk.Label(control_frame, text="Exchange Dynamics", 
                 font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=3, pady=(0, 5), sticky=tk.W)
        row += 1
        
        self.create_slider(control_frame, row, "kGE (s⁻¹)", 'kGE', 10, 2000, 10)
        row += 1
        self.create_slider(control_frame, row, "kEG (s⁻¹)", 'kEG', 10, 2000, 10)
        row += 1
        
        self.kex_label = ttk.Label(control_frame, text="", foreground='blue')
        self.kex_label.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=5)
        row += 1
        
        ttk.Separator(control_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        # Relaxation rates
        ttk.Label(control_frame, text="Relaxation Rates", 
                 font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=3, pady=(0, 5), sticky=tk.W)
        row += 1
        
        self.create_slider(control_frame, row, "R2G (s⁻¹)", 'R2G', 1, 50, 0.5)
        row += 1
        self.create_slider(control_frame, row, "R2E (s⁻¹)", 'R2E', 1, 50, 0.5)
        row += 1
        self.create_slider(control_frame, row, "R1G (s⁻¹)", 'R1G', 0.1, 10, 0.1)
        row += 1
        self.create_slider(control_frame, row, "R1E (s⁻¹)", 'R1E', 0.1, 10, 0.1)
        row += 1
        
        ttk.Separator(control_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        # Chemical shifts
        ttk.Label(control_frame, text="Chemical Shifts", 
                 font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=3, pady=(0, 5), sticky=tk.W)
        row += 1
        
        self.create_slider(control_frame, row, "ωG (ppm)", 'dwG', -5.0, 5.0, 0.1)
        row += 1
        self.create_slider(control_frame, row, "ωE (ppm)", 'dwE', -5.0, 5.0, 0.1)
        row += 1
        
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
        ttk.Checkbutton(control_frame, text="Add Noise", 
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
        ttk.Button(button_frame, text="Preset: Standard", 
                  command=self.preset_standard).pack(side=tk.TOP, pady=2)
        row += 1
        
        # Info
        info_frame = ttk.LabelFrame(control_frame, text="Interpretation", padding="5")
        info_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        self.info_text = tk.Text(info_frame, height=12, width=50, wrap=tk.WORD)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        # Plot
        self.fig = Figure(figsize=(10, 7))
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_slider(self, parent, row, label, var_name, min_val, max_val, step, scale=1):
        """Create slider in main control frame"""
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
    
    def create_slider_in_frame(self, parent, row, label, var_name, min_val, max_val, step, scale=1):
        """Create slider in a sub-frame (for nested frames like pulse parameters)"""
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
        """Update plot with current parameters"""
        self.ax.clear()
        
        params = {key: var.get() for key, var in self.params.items()}
        
        # Update labels
        kex = params['kGE'] + params['kEG']
        pG = params['kEG'] / kex if kex > 0 else 0.95
        pE = params['kGE'] / kex if kex > 0 else 0.05
        
        self.kex_label.config(text=f"kex = {kex:.1f} s⁻¹ | pG = {pG*100:.1f}% | pE = {pE*100:.1f}%")
        
        dw = params['dwE'] - params['dwG']
        dw_hz = dw * params['B0']
        self.dw_label.config(text=f"Δω = {dw:.2f} ppm = {dw_hz:.1f} Hz")
        
        # Create simulator with finite pulses
        sim_finite = CPMGSimulatorFinitePulse(
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
            ncycMax=int(params['ncycMax']),
            pulse_width=params['pulse_width'],
            omega1=params['omega1'],
            use_finite_pulses=self.use_finite_pulses.get()
        )
        
        # Generate profile
        data = sim_finite.generate_profile(
            add_noise=self.include_noise.get(),
            noise_level=params['noise_level']
        )
        
        # Plot main data
        if self.include_noise.get():
            self.ax.errorbar(data['vcpmg'], data['R2eff'], yerr=data['error'],
                           fmt='o', color='darkblue', 
                           label='Finite pulses' if self.use_finite_pulses.get() else 'Instantaneous',
                           capsize=4, markersize=7, linewidth=2)
        else:
            label = 'Finite pulses' if self.use_finite_pulses.get() else 'Instantaneous pulses'
            self.ax.plot(data['vcpmg'], data['R2eff'], 'o-', 
                        color='darkblue', linewidth=2.5, markersize=7, 
                        label=label, markerfacecolor='lightblue', markeredgewidth=2)
        
        # Show comparison if requested
        if self.show_comparison.get():
            sim_instant = CPMGSimulatorFinitePulse(
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
                ncycMax=int(params['ncycMax']),
                pulse_width=params['pulse_width'],
                omega1=params['omega1'],
                use_finite_pulses=False  # Force instantaneous
            )
            data_instant = sim_instant.generate_profile(add_noise=False)
            self.ax.plot(data_instant['vcpmg'], data_instant['R2eff'], 's--', 
                        color='red', linewidth=2, markersize=6,
                        label='Instantaneous (comparison)', alpha=0.7)
        
        # Formatting
        self.ax.set_xlabel('νCPMG (Hz)', fontsize=13, fontweight='bold')
        self.ax.set_ylabel('R2eff (s⁻¹)', fontsize=13, fontweight='bold')
        
        title = 'CPMG Dispersion: Finite Pulse Width Model' if self.use_finite_pulses.get() else 'CPMG Dispersion: Instantaneous Pulse Model'
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.ax.legend(fontsize=11)
        
        # Reference line
        self.ax.axhline(y=params['R2G'], color='gray', linestyle=':', 
                       alpha=0.5, linewidth=1.5, label=f'R2G = {params["R2G"]:.1f}')
        self.ax.legend(fontsize=10)
        
        self.canvas.draw()
        
        # Update info
        self.update_info(sim_finite, data, params)
        
    def update_info(self, sim, data, params):
        """Update info text"""
        self.info_text.delete(1.0, tk.END)
        
        kex = params['kGE'] + params['kEG']
        dw = (params['dwE'] - params['dwG']) * params['B0']
        
        ratio = kex / (np.abs(dw) * 2 * np.pi) if dw != 0 else 1000
        
        if ratio > 2:
            regime = "Fast exchange"
        elif ratio > 0.5:
            regime = "Intermediate exchange"
        else:
            regime = "Slow exchange"
        
        R2ex_max = np.max(data['R2eff']) - params['R2G']
        
        # Calculate pulse width impact
        tau_pw = params['pulse_width']  # μs
        tau_typical = (params['timeT2'] / params['ncycMax']) / 4.0  # ms -> convert to μs
        tau_typical_us = tau_typical * 1000
        pulse_fraction = (tau_pw / tau_typical_us) * 100 if tau_typical_us > 0 else 0
        
        info = f"""Exchange Regime: {regime}

Two-State Model (G ⇌ E):
  kGE = {params['kGE']:.1f} s⁻¹
  kEG = {params['kEG']:.1f} s⁻¹
  kex = {kex:.1f} s⁻¹
  pG = {sim.pG*100:.1f}%, pE = {sim.pE*100:.1f}%

Chemical Shifts:
  Δω = {dw:.1f} Hz at {params['B0']:.0f} MHz

Relaxation:
  R2G = {params['R2G']:.1f}, R2E = {params['R2E']:.1f} s⁻¹
  R1G = {params['R1G']:.1f}, R1E = {params['R1E']:.1f} s⁻¹

Pulse Parameters:
  180° pulse width: {tau_pw:.1f} μs
  RF field (ω1/2π): {params['omega1']/1000:.1f} kHz
  τ (typical): {tau_typical_us:.1f} μs
  Pulse occupies: {pulse_fraction:.1f}% of τ
  
  Mode: {'FINITE PULSES' if self.use_finite_pulses.get() else 'INSTANTANEOUS'}

Exchange Contribution:
  R2,ex (max) = {R2ex_max:.2f} s⁻¹

CPMG Range:
  {np.min(data['vcpmg']):.1f} - {np.max(data['vcpmg']):.1f} Hz
"""
        
        if pulse_fraction > 20:
            info += "\n⚠️ Warning: Pulse width > 20% of τ\n   Finite pulse effects significant!"
        
        self.info_text.insert(1.0, info)
        
    def export_data(self):
        """Export data to CSV"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        params = {key: var.get() for key, var in self.params.items()}
        sim = CPMGSimulatorFinitePulse(
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
            ncycMax=int(params['ncycMax']),
            pulse_width=params['pulse_width'],
            omega1=params['omega1'],
            use_finite_pulses=self.use_finite_pulses.get()
        )
        
        data = sim.generate_profile(
            add_noise=self.include_noise.get(),
            noise_level=params['noise_level']
        )
        
        kex = params['kGE'] + params['kEG']
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['# CPMG Data - Finite Pulse Width Model'])
            writer.writerow([f'# Pulse mode: {"Finite" if self.use_finite_pulses.get() else "Instantaneous"}'])
            writer.writerow([f'# Pulse width: {params["pulse_width"]:.1f} us'])
            writer.writerow([f'# RF field (omega1/2pi): {params["omega1"]/1000:.1f} kHz'])
            writer.writerow([f'# kGE={params["kGE"]:.1f}, kEG={params["kEG"]:.1f}, kex={kex:.1f} s-1'])
            writer.writerow([f'# R2G={params["R2G"]:.1f}, R2E={params["R2E"]:.1f} s-1'])
            writer.writerow([''])
            writer.writerow(['ncyc', 'vcpmg(Hz)', 'R2eff(s-1)', 'error(s-1)', 'tau_eff(us)'])
            
            for i in range(len(data['ncyc'])):
                writer.writerow([
                    data['ncyc'][i],
                    f"{data['vcpmg'][i]:.4f}",
                    f"{data['R2eff'][i]:.6f}",
                    f"{data['error'][i]:.6f}",
                    f"{data['tau_eff'][i]*1e6:.2f}"
                ])
        
        messagebox.showinfo("Export Successful", f"Data exported to:\n{filename}")
        
    def preset_standard(self):
        """Load standard preset"""
        self.params['kGE'].set(500)
        self.params['kEG'].set(50)
        self.params['dwG'].set(0.0)
        self.params['dwE'].set(1.0)
        self.params['R2G'].set(10)
        self.params['R2E'].set(12)
        self.params['R1G'].set(1.5)
        self.params['R1E'].set(1.8)
        self.params['pulse_width'].set(80)
        self.params['omega1'].set(15000)
        self.use_finite_pulses.set(True)
        self.update_plot()


def main():
    """Main entry point"""
    root = tk.Tk()
    app = CPMGSimulatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()