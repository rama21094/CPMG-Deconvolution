"""GUI for the CW-decoupled 15N CPMG simulator (cpmg_ml.simulator_cw_cpmg).

Mirrors the layout/conventions of simulators/gui/32x32_Sim_5thJune.py, but
plots profiles from CWCPMGSimulator.simulate_cw_cpmg instead of duplicating
any physics here. Does not modify cpmg_ml/simulator.py or
cpmg_ml/simulator_cw_cpmg.py.
"""

import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cpmg_ml.simulator import CPMGSimulator  # noqa: E402
from cpmg_ml.simulator_cw_cpmg import CWCPMGSimulator  # noqa: E402


class CWCPMGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CW-Decoupled 15N CPMG Simulator (Hansen/Vallurupalli/Kay, JPCB 2008)")
        self.sim = CWCPMGSimulator()
        self.plain_sim = CPMGSimulator()
        self.lines = []

        self.vars = {
            "B0_MHz": tk.DoubleVar(value=600.0),
            "B1_N": tk.DoubleVar(value=5555.0),
            "B1_H_cw": tk.DoubleVar(value=13000.0),
            "ncyc_start": tk.IntVar(value=1),
            "ncyc_stop": tk.IntVar(value=80),
            "ncyc_step": tk.IntVar(value=2),
            "k_ex": tk.DoubleVar(value=500.0),
            "p_B": tk.DoubleVar(value=0.05),
            "dw_N": tk.DoubleVar(value=1.5),
            "tau_m": tk.DoubleVar(value=5.0e-9),
            "tau_e": tk.DoubleVar(value=50e-12),
            "S2": tk.DoubleVar(value=0.85),
            "r_IS": tk.DoubleVar(value=1.02e-10),
            "r_eff": tk.DoubleVar(value=1.86e-10),
            "csa_N": tk.DoubleVar(value=-160e-6),
            "theta_N": tk.DoubleVar(value=22.0 * (np.pi / 180)),
            "J_IS": tk.DoubleVar(value=92.0),
        }
        self.overlay_plain = tk.BooleanVar(value=True)

        self.setup_ui()

    def setup_ui(self):
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(
            control_frame, text="CW-CPMG Controls", font=("Helvetica", 11, "bold")
        ).grid(row=0, column=0, columnspan=2, pady=10)

        row = 1
        row = self.add_section(
            control_frame,
            "Field / Sequence",
            [
                ("B0_MHz", "B0 1H Frequency (MHz)"),
                ("B1_N", "15N B1 Field (Hz)"),
                ("B1_H_cw", "1H CW Decoupling Field (Hz)"),
                ("ncyc_start", "ncyc Start"),
                ("ncyc_stop", "ncyc Stop"),
                ("ncyc_step", "ncyc Step"),
            ],
            row,
        )

        row = self.add_section(
            control_frame,
            "Dynamics / Exchange",
            [
                ("k_ex", "k_ex (s^-1)"),
                ("p_B", "Population p_B"),
                ("dw_N", "Delta Omega (ppm)"),
            ],
            row,
        )

        row = self.add_section(
            control_frame,
            "Relaxation Model",
            [
                ("tau_m", "tau_m (s)"),
                ("tau_e", "tau_e (s)"),
                ("S2", "Order Param S2"),
                ("r_IS", "N-H Distance (m)"),
                ("r_eff", "Bath H Dist (m)"),
                ("csa_N", "15N CSA (ppm)"),
                ("theta_N", "Theta CSA/DD (rad)"),
                ("J_IS", "Scalar Coup J (Hz)"),
            ],
            row,
        )

        ttk.Checkbutton(
            control_frame,
            text="Also overlay plain CPMG (no decoupling)",
            variable=self.overlay_plain,
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(12, 4))
        row += 1

        ttk.Button(
            control_frame, text="Plot / Overlay Profile", command=self.run_simulation
        ).grid(row=row, column=0, columnspan=2, pady=15)
        row += 1
        ttk.Button(control_frame, text="Clear Plot", command=self.clear_plot).grid(
            row=row, column=0, columnspan=2
        )
        row += 1
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(
            control_frame, textvariable=self.status_var, foreground="blue", wraplength=280
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=10)

        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots(figsize=(7, 5))
        self.ax.set_xlabel(r"$\nu_{cp}$ (Hz)")
        self.ax.set_ylabel(r"$R_{2,eff}$ ($s^{-1}$)")
        self.ax.set_title("CW-CPMG Dispersion Profile")
        self.ax.grid(True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def add_section(self, parent, title, fields, row):
        ttk.Label(parent, text=title, font=("Helvetica", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(12, 4)
        )
        row += 1
        for key, text in fields:
            ttk.Label(parent, text=text).grid(row=row, column=0, sticky="w")
            ttk.Entry(parent, textvariable=self.vars[key], width=12).grid(
                row=row, column=1, padx=5, pady=2
            )
            row += 1
        return row

    def autoscale_plot(self):
        self.ax.relim()
        self.ax.autoscale_view()

    def validate_inputs(self, params, ncyc_range):
        checks = [
            (params["B0_MHz"] > 0, "B0 frequency must be greater than 0 MHz."),
            (params["B1_N"] > 0, "15N B1 field must be greater than 0 Hz."),
            (params["B1_H_cw"] > 0, "1H CW field must be greater than 0 Hz."),
            (params["ncyc_start"] >= 1, "ncyc start must be at least 1."),
            (params["ncyc_stop"] > params["ncyc_start"], "ncyc stop must be greater than ncyc start."),
            (params["ncyc_step"] >= 1, "ncyc step must be at least 1."),
            (len(ncyc_range) > 0, "ncyc range is empty."),
            (0.0 <= params["p_B"] <= 1.0, "Population p_B must be between 0 and 1."),
        ]
        for passed, message in checks:
            if not passed:
                raise ValueError(message)

    def run_simulation(self):
        try:
            params = {k: v.get() for k, v in self.vars.items()}
            ncyc_start = int(params["ncyc_start"])
            ncyc_stop = int(params["ncyc_stop"])
            ncyc_step = int(params["ncyc_step"])
            params["ncyc_start"] = ncyc_start
            params["ncyc_stop"] = ncyc_stop
            params["ncyc_step"] = ncyc_step
            ncyc_range = np.arange(ncyc_start, ncyc_stop, ncyc_step)
            self.validate_inputs(params, ncyc_range)

            b0_mhz = params["B0_MHz"]
            sim_params = dict(params)
            sim_params["B0"] = 2.0 * np.pi * b0_mhz * 1.0e6 / self.sim.gamma_H
            sim_params["B1_N"] = params["B1_N"]

            profile = self.sim.simulate_cw_cpmg(
                sim_params, ncyc_range=ncyc_range, b1_h_cw_hz=params["B1_H_cw"]
            )
            if len(profile.nu_cp) == 0:
                raise ValueError("No valid CPMG points remain after finite-pulse timing checks.")

            label = (
                f"CW: B0={b0_mhz:g}MHz, B1(15N)={params['B1_N']:g}Hz, "
                f"B1(1H-CW)={params['B1_H_cw']:g}Hz, ncyc={ncyc_start}:{ncyc_stop}:{ncyc_step}"
            )
            line, = self.ax.plot(profile.nu_cp, profile.r2_eff, "-o", label=label)
            self.lines.append(line)

            status = f"Plotted {len(profile.nu_cp)} CW-CPMG points. Internal B0 = {sim_params['B0']:.3f} T."
            if profile.skipped_ncyc:
                status += f" Skipped {len(profile.skipped_ncyc)} high-ncyc point(s)."

            if self.overlay_plain.get():
                plain_profile = self.plain_sim.simulate_cpmg(sim_params, ncyc_range=ncyc_range)
                plain_label = (
                    f"Plain (no decoupling): B0={b0_mhz:g}MHz, B1={params['B1_N']:g}Hz, "
                    f"ncyc={ncyc_start}:{ncyc_stop}:{ncyc_step}"
                )
                plain_line, = self.ax.plot(
                    plain_profile.nu_cp, plain_profile.r2_eff, "--s", label=plain_label
                )
                self.lines.append(plain_line)
                status += f" Overlaid {len(plain_profile.nu_cp)} plain-CPMG points."

            self.autoscale_plot()
            self.ax.legend(fontsize=8)
            self.canvas.draw()
            self.status_var.set(status)
        except Exception as e:
            self.status_var.set(f"Simulation Error: {e}")

    def clear_plot(self):
        for line in self.lines:
            line.remove()
        self.lines.clear()
        self.ax.legend_ = None
        self.autoscale_plot()
        self.canvas.draw()
        self.status_var.set("Plot cleared.")


if __name__ == "__main__":
    root = tk.Tk()
    app = CWCPMGApp(root)
    root.mainloop()
