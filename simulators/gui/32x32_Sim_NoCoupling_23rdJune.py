import importlib.util
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


BASE_SCRIPT = Path(__file__).with_name("32x32_Sim_5thJune.py")
MAGIC_ANGLE_DEG = 54.735610317245346
T_RELAX = 0.04


def load_base_simulator():
    spec = importlib.util.spec_from_file_location("cpmg_gui_base_5thjune", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Could not load simulator from {BASE_SCRIPT}")
    spec.loader.exec_module(module)
    return module.CPMG_Simulator


CPMG_Simulator = load_base_simulator()


class CPMG_NoCoupling_App:
    def __init__(self, root):
        self.root = root
        self.root.title("CPMG Simulator - Normal vs No Coupling / No DD / Magic Angle")
        self.sim = CPMG_Simulator()
        self.lines = []

        self.vars = {
            "B0_MHz": tk.DoubleVar(value=600.0),
            "B1_N": tk.DoubleVar(value=5555.0),
            "ncyc_start": tk.IntVar(value=1),
            "ncyc_stop": tk.IntVar(value=80),
            "ncyc_step": tk.IntVar(value=2),
            "tau_m": tk.DoubleVar(value=5.0e-9),
            "tau_e": tk.DoubleVar(value=50.0e-12),
            "S2": tk.DoubleVar(value=0.85),
            "r_IS": tk.DoubleVar(value=1.02e-10),
            "r_eff": tk.DoubleVar(value=1.86e-10),
            "csa_N": tk.DoubleVar(value=-160e-6),
            "theta_N": tk.DoubleVar(value=22.0 * (np.pi / 180.0)),
            "J_IS": tk.DoubleVar(value=92.0),
            "k_ex": tk.DoubleVar(value=500.0),
            "p_B": tk.DoubleVar(value=0.05),
            "dw_N": tk.DoubleVar(value=1.5),
            "no_coupling_r_IS": tk.DoubleVar(value=1.0e-6),
            "magic_angle_deg": tk.DoubleVar(value=MAGIC_ANGLE_DEG),
            "nu_cp_mode": tk.StringVar(value="constant_time"),
        }

        self.setup_ui()

    def setup_ui(self):
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(control_frame, text="CPMG Controls", font=("Helvetica", 11, "bold")).grid(
            row=0, column=0, columnspan=2, pady=10
        )

        row = 1
        row = self.add_section(
            control_frame,
            "Field / Sequence",
            [
                ("B0_MHz", "B0 1H Frequency (MHz)"),
                ("B1_N", "15N B1 Field (Hz)"),
                ("ncyc_start", "ncyc Start"),
                ("ncyc_stop", "ncyc Stop"),
                ("ncyc_step", "ncyc Step"),
            ],
            row,
        )
        row = self.add_nu_cp_mode(control_frame, row)

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
                ("csa_N", "15N CSA"),
                ("theta_N", "Theta CSA/DD (rad)"),
                ("J_IS", "Scalar Coup J (Hz)"),
            ],
            row,
        )

        row = self.add_section(
            control_frame,
            "No-Coupling Target Overrides",
            [
                ("no_coupling_r_IS", "Large r_IS (m)"),
                ("magic_angle_deg", "Magic Angle (deg)"),
            ],
            row,
        )

        ttk.Button(control_frame, text="Plot Current Profile", command=self.plot_current).grid(
            row=row, column=0, columnspan=2, pady=(15, 4), sticky="ew"
        )
        ttk.Button(
            control_frame,
            text="Plot No-Coupling Target",
            command=self.plot_no_coupling_target,
        ).grid(row=row + 1, column=0, columnspan=2, pady=4, sticky="ew")
        ttk.Button(control_frame, text="Plot Both", command=self.plot_both).grid(
            row=row + 2, column=0, columnspan=2, pady=4, sticky="ew"
        )
        ttk.Button(control_frame, text="Clear Plot", command=self.clear_plot).grid(
            row=row + 3, column=0, columnspan=2, pady=(4, 0), sticky="ew"
        )

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(control_frame, textvariable=self.status_var, foreground="blue", wraplength=280).grid(
            row=row + 4, column=0, columnspan=2, sticky="w", pady=10
        )

        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots(figsize=(7, 5))
        self.ax.set_xlabel(r"$\nu_{cp}$ (Hz)")
        self.ax.set_ylabel(r"$R_{2,eff}$ ($s^{-1}$)")
        self.ax.set_title("CPMG Profile: Current vs No-Coupling Target")
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
            ttk.Entry(parent, textvariable=self.vars[key], width=13).grid(row=row, column=1, padx=5, pady=2)
            row += 1
        return row

    def add_nu_cp_mode(self, parent, row):
        ttk.Label(parent, text="nu_cp Mapping", font=("Helvetica", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(12, 4)
        )
        row += 1
        ttk.Radiobutton(
            parent,
            text="Constant time: ncyc / T",
            variable=self.vars["nu_cp_mode"],
            value="constant_time",
        ).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        ttk.Radiobutton(
            parent,
            text="Standard: 1 / (4 tau_cp)",
            variable=self.vars["nu_cp_mode"],
            value="standard",
        ).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        return row

    def collect_params(self):
        params = {key: value.get() for key, value in self.vars.items()}
        params["ncyc_start"] = int(params["ncyc_start"])
        params["ncyc_stop"] = int(params["ncyc_stop"])
        params["ncyc_step"] = int(params["ncyc_step"])
        ncyc_range = np.arange(params["ncyc_start"], params["ncyc_stop"], params["ncyc_step"])
        self.validate_inputs(params, ncyc_range)
        params["B0"] = 2.0 * np.pi * params["B0_MHz"] * 1.0e6 / self.sim.gamma_H
        return params, ncyc_range

    def validate_inputs(self, params, ncyc_range):
        checks = [
            (params["B0_MHz"] > 0.0, "B0 frequency must be greater than 0 MHz."),
            (params["B1_N"] > 0.0, "B1 field must be greater than 0 Hz."),
            (params["ncyc_start"] >= 1, "ncyc start must be at least 1."),
            (params["ncyc_stop"] > params["ncyc_start"], "ncyc stop must be greater than ncyc start."),
            (params["ncyc_step"] >= 1, "ncyc step must be at least 1."),
            (len(ncyc_range) > 0, "ncyc range is empty."),
            (0.0 <= params["p_B"] <= 1.0, "Population p_B must be between 0 and 1."),
            (params["r_IS"] > 0.0, "r_IS must be greater than 0."),
            (params["r_eff"] > 0.0, "r_eff must be greater than 0."),
            (params["no_coupling_r_IS"] > 0.0, "Large r_IS override must be greater than 0."),
            (
                params["nu_cp_mode"] in {"constant_time", "standard"},
                "nu_cp mapping must be constant_time or standard.",
            ),
        ]
        for passed, message in checks:
            if not passed:
                raise ValueError(message)

    def make_no_coupling_params(self, params):
        target = dict(params)
        target["J_IS"] = 0.0
        target["r_IS"] = params["no_coupling_r_IS"]
        target["theta_N"] = np.deg2rad(params["magic_angle_deg"])
        return target

    def run_profile(self, params, ncyc_range):
        sim_params = dict(params)
        sim_params.pop("B0_MHz", None)
        sim_params.pop("ncyc_start", None)
        sim_params.pop("ncyc_stop", None)
        sim_params.pop("ncyc_step", None)
        sim_params.pop("no_coupling_r_IS", None)
        sim_params.pop("magic_angle_deg", None)
        nu_cp, r2_eff, skipped_ncyc = self.sim.simulate_cpmg(sim_params, ncyc_range, T_relax=T_RELAX)
        if len(nu_cp) == 0:
            raise ValueError("No valid CPMG points remain after finite-pulse timing checks.")
        nu_cp = self.map_nu_cp(params, ncyc_range, skipped_ncyc)
        return nu_cp, r2_eff, skipped_ncyc

    def map_nu_cp(self, params, ncyc_range, skipped_ncyc):
        skipped = set(skipped_ncyc)
        kept_ncyc = np.asarray([int(ncyc) for ncyc in ncyc_range if int(ncyc) not in skipped], dtype=float)
        if params["nu_cp_mode"] == "constant_time":
            return kept_ncyc / T_RELAX

        t_180 = 1.0 / (2.0 * params["B1_N"])
        tau_cp = (T_RELAX - kept_ncyc * t_180) / (2.0 * kept_ncyc)
        return 1.0 / (4.0 * tau_cp)

    def nu_cp_mode_label(self, params):
        if params["nu_cp_mode"] == "constant_time":
            return "nu_cp=ncyc/T"
        return "nu_cp=1/(4 tau_cp)"

    def add_line(self, nu_cp, r2_eff, label, style="-o"):
        line, = self.ax.plot(nu_cp, r2_eff, style, label=label)
        self.lines.append(line)

    def plot_current(self):
        try:
            params, ncyc_range = self.collect_params()
            nu_cp, r2_eff, skipped_ncyc = self.run_profile(params, ncyc_range)
            label = (
                f"Current: J={params['J_IS']:g}Hz, rIS={params['r_IS']:.2e}m, "
                f"theta={np.rad2deg(params['theta_N']):.2f}deg, {self.nu_cp_mode_label(params)}"
            )
            self.add_line(nu_cp, r2_eff, label)
            self.finish_plot(params, len(nu_cp), skipped_ncyc, "current profile")
        except Exception as exc:
            self.status_var.set(f"Simulation Error: {exc}")

    def plot_no_coupling_target(self):
        try:
            params, ncyc_range = self.collect_params()
            target = self.make_no_coupling_params(params)
            nu_cp, r2_eff, skipped_ncyc = self.run_profile(target, ncyc_range)
            label = (
                f"Target: J=0, rIS={target['r_IS']:.1e}m, "
                f"theta={np.rad2deg(target['theta_N']):.2f}deg, {self.nu_cp_mode_label(target)}"
            )
            self.add_line(nu_cp, r2_eff, label, "--s")
            self.finish_plot(target, len(nu_cp), skipped_ncyc, "no-coupling target")
        except Exception as exc:
            self.status_var.set(f"Simulation Error: {exc}")

    def plot_both(self):
        try:
            params, ncyc_range = self.collect_params()
            target = self.make_no_coupling_params(params)
            current_nu, current_r2, current_skipped = self.run_profile(params, ncyc_range)
            target_nu, target_r2, target_skipped = self.run_profile(target, ncyc_range)
            self.add_line(
                current_nu,
                current_r2,
                f"Current: J={params['J_IS']:g}Hz, rIS={params['r_IS']:.2e}m, theta={np.rad2deg(params['theta_N']):.2f}deg, {self.nu_cp_mode_label(params)}",
            )
            self.add_line(
                target_nu,
                target_r2,
                f"Target: J=0, rIS={target['r_IS']:.1e}m, theta={np.rad2deg(target['theta_N']):.2f}deg, {self.nu_cp_mode_label(target)}",
                "--s",
            )
            skipped = sorted(set(current_skipped) | set(target_skipped))
            self.finish_plot(params, len(current_nu), skipped, "current + no-coupling target")
        except Exception as exc:
            self.status_var.set(f"Simulation Error: {exc}")

    def finish_plot(self, params, point_count, skipped_ncyc, description):
        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.legend()
        self.canvas.draw()
        status = (
            f"Plotted {description}: {point_count} points. "
            f"B0={params['B0_MHz']:g} MHz ({params['B0']:.3f} T). "
            f"{self.nu_cp_mode_label(params)}."
        )
        if skipped_ncyc:
            status += f" Skipped ncyc: {skipped_ncyc}."
        self.status_var.set(status)

    def clear_plot(self):
        for line in self.lines:
            line.remove()
        self.lines.clear()
        self.ax.legend_ = None
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()
        self.status_var.set("Plot cleared.")


if __name__ == "__main__":
    root = tk.Tk()
    app = CPMG_NoCoupling_App(root)
    root.mainloop()
