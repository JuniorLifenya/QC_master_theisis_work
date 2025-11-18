# nv_gw_super_final.py
"""
NV-center ⇆ gravitational-wave simulator - integrated & robust version.

Key features:
 - Class-based API (SimulatorParams + NVGWSuperSimulator)
 - sesolve (pure state) and mesolve (Lindblad) modes
 - Function-based time-dependent coefficient (no Cython compile)
 - Collapse operators for T1/T2 optional
 - FFT analysis, frequency scan utility, CSV export
 - Demo-mode for visible dynamics (amplifies tiny physical parameters)
 - Robust sanity checks (no IndexError if solver omits stored states)
"""
from __future__ import annotations
import csv
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Sequence

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("NVGWSuperFinal")

# ---------- Parameters dataclass ----------
@dataclass
class SimulatorParams:
    D_hz: float = 2.87e9                  # NV zero-field splitting (Hz)
    gamma_e_hz_per_T: float = 28e9        # gyromagnetic ratio (Hz/T)
    Bz_T: float = 0.0                     # static B field (T)

    h_max: float = 1e-18                  # GW strain amplitude (realistic)
    kappa: float = 1e-3                   # coupling constant (toy default)
    f_gw_hz: float = 100.0                # GW frequency (Hz)

    t_final_s: float = 0.1                # simulation time (s)
    nsteps: int = 2000                    # time steps
    demo_mode: bool = False               # amplify parameters if True

    T1_s: Optional[float] = None          # optional T1
    T2_s: Optional[float] = None          # optional T2

    use_mesolve: bool = False             # use mesolve (open system) when True
    save_csv: bool = False
    csv_filename: str = "nv_gw_results.csv"

    def validate(self):
        if self.nsteps < 2:
            raise ValueError("nsteps must be >= 2")
        if self.t_final_s <= 0:
            raise ValueError("t_final_s must be > 0")


# ---------- Simulator class ----------
class NVGWSuperSimulator:
    def __init__(self, params: SimulatorParams):
        params.validate()
        self.p = params
        # spin operators
        self.Sx = qt.jmat(1, "x")
        self.Sy = qt.jmat(1, "y")
        self.Sz = qt.jmat(1, "z")
        # basis
        self.psi_p1 = qt.basis(3, 0)
        self.psi_0 = qt.basis(3, 1)
        self.psi_m1 = qt.basis(3, 2)
        # quadrupole-like operators
        self.Op_plus = self.Sx**2 - self.Sy**2
        self.Op_cross = self.Sx * self.Sy + self.Sy * self.Sx

        # demo-mode amplification
        if self.p.demo_mode:
            log.info("DEMO MODE ON: amplifying h_max and kappa for visualization")
            self.p.h_max *= 1e6      # amplify strain
            self.p.kappa *= 1e12     # amplify coupling

        # convert to angular frequencies (rad/s) for propagation with t in seconds
        self.D = 2 * np.pi * self.p.D_hz
        self.gamma_e = 2 * np.pi * self.p.gamma_e_hz_per_T
        self.omega_gw = 2 * np.pi * self.p.f_gw_hz

    def gw_coeff(self, t: float, args: Dict) -> float:
        """QuTiP expects signature (t, args) -> float."""
        return args["A"] * np.sin(args["omega"] * t)

    def get_hamiltonian(self):
        """Return QuTiP time-dependent Hamiltonian in list form."""
        H_static = self.D * (self.Sz**2) + self.gamma_e * self.p.Bz_T * self.Sz
        H_int_op = self.p.kappa * self.Op_plus
        H_td = [H_static, [H_int_op, self.gw_coeff]]
        return H_td

    def get_collapse_operators(self):
        c_ops = []
        if self.p.T2_s is not None:
            gamma_T2 = 1.0 / self.p.T2_s
            c_ops.append(np.sqrt(gamma_T2) * self.Sz)
        if self.p.T1_s is not None:
            gamma_T1 = 1.0 / self.p.T1_s
            c_ops.append(np.sqrt(gamma_T1) * (self.psi_0 * self.psi_p1.dag()))
            c_ops.append(np.sqrt(gamma_T1) * (self.psi_0 * self.psi_m1.dag()))
        return c_ops

    def run(self) -> Tuple[qt.Result, np.ndarray]:
        """Run the simulation (sesolve or mesolve depending on params)."""
        tlist = np.linspace(0.0, self.p.t_final_s, self.p.nsteps)
        H_td = self.get_hamiltonian()
        args = {"A": self.p.h_max, "omega": self.omega_gw}

        if not self.p.use_mesolve:
            psi0 = self.psi_0
            e_ops = [self.psi_p1.proj(), self.psi_0.proj(), self.psi_m1.proj(), self.Sz]
            log.info("Running sesolve (pure-state evolution)...")
            result = qt.sesolve(H_td, psi0, tlist, e_ops=e_ops, args=args)
        else:
            rho0 = self.psi_0.proj()
            c_ops = self.get_collapse_operators()
            e_ops = [self.psi_p1.proj(), self.psi_0.proj(), self.psi_m1.proj(), self.Sz]
            log.info(f"Running mesolve (open-system) with {len(c_ops)} collapse operators...")
            result = qt.mesolve(H_td, rho0, tlist, c_ops, e_ops=e_ops, args=args)

        # ---------- Robust sanity-check (no IndexError) ----------
        try:
            if hasattr(result, "expect") and len(result.expect) >= 2:
                # e_ops ordering: [P_plus1, P_0, P_minus1, Sz] -> index 1 is P_0
                initial_P0 = float(np.real(result.expect[1][0]))
                log.info(f"Sanity: initial P(|0>) from expect = {initial_P0:.6f}")
            elif hasattr(result, "states") and result.states:
                first = result.states[0]
                if not self.p.use_mesolve:
                    norm0 = float(first.norm())
                    if abs(norm0 - 1.0) > 1e-8:
                        log.warning(f"Initial state norm != 1 ({norm0})")
                    else:
                        log.info(f"Sanity: initial state norm = {norm0:.6f}")
                else:
                    try:
                        trace0 = float(np.real(first.tr()))
                        if abs(trace0 - 1.0) > 1e-8:
                            log.warning(f"Initial density matrix trace != 1 ({trace0})")
                        else:
                            log.info(f"Sanity: initial density matrix trace = {trace0:.6f}")
                    except Exception:
                        log.debug("Could not compute trace of first element in result.states")
            else:
                log.info("No 'expect' or stored 'states' available for sanity checks (this is OK).")
        except Exception as ex:
            log.warning(f"Sanity check encountered unexpected error: {ex}")

        return result, tlist

    # ---------- Analysis helpers ----------
    def compute_matrix_element(self, bra: qt.Qobj, op: qt.Qobj, ket: qt.Qobj) -> float:
        q = (bra.dag() * op * ket)
        try:
            val = q.full()[0, 0]
        except Exception:
            val = complex(q)
        return float(np.abs(val))

    def estimate_rabi_frequency(self, state_i: qt.Qobj, state_j: qt.Qobj) -> float:
        H_int_op = self.p.kappa * self.Op_plus
        matel = self.compute_matrix_element(state_i, H_int_op, state_j)
        return matel

    def compute_fft(self, signal: np.ndarray, tlist: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dt = tlist[1] - tlist[0]
        yf = rfft(signal - np.mean(signal))
        xf = rfftfreq(len(signal), dt)
        return xf, np.abs(yf)

    def freq_scan(self, f_array_hz: Sequence[float], which_outcome: str = "+1") -> Tuple[np.ndarray, np.ndarray]:
        final_p = np.zeros(len(f_array_hz))
        orig_f = self.p.f_gw_hz
        for i, f in enumerate(f_array_hz):
            self.p.f_gw_hz = f
            self.omega_gw = 2 * np.pi * f
            result, tlist = self.run()
            if hasattr(result, "expect") and len(result.expect) >= 3:
                if which_outcome == "+1":
                    final_p[i] = np.real(result.expect[0][-1])
                elif which_outcome == "0":
                    final_p[i] = np.real(result.expect[1][-1])
                else:
                    final_p[i] = np.real(result.expect[2][-1])
            else:
                final_p[i] = np.nan
            log.debug(f"freq scan {i+1}/{len(f_array_hz)} f={f:.3e}Hz -> {final_p[i]:.3e}")
        self.p.f_gw_hz = orig_f
        self.omega_gw = 2 * np.pi * orig_f
        return np.array(f_array_hz), final_p

    def analyze_and_plot(self, result: qt.Result, tlist: np.ndarray, save_csv: Optional[str] = None):
        p_plus1, p_0, p_minus1, exp_Sz = [np.real(x) for x in result.expect]
        omega_rabi_rad_s = self.estimate_rabi_frequency(self.psi_p1, self.psi_m1)
        omega_rabi_hz = omega_rabi_rad_s / (2 * np.pi)
        log.info(f"Estimated Rabi frequency |+1> <-> |-1> (theoretical) = {omega_rabi_hz:.6e} Hz")

        freqs_hz, fft_amp = self.compute_fft(p_0, tlist)
        t_plot_ms = tlist * 1e3

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
        fig.suptitle(f"NV center + GW (demo_mode={self.p.demo_mode}, mesolve={self.p.use_mesolve})", fontsize=14)

        axes[0, 0].plot(t_plot_ms, p_plus1, label="P(|+1>)", color="tab:red", lw=1.8)
        axes[0, 0].plot(t_plot_ms, p_0, label="P(|0>)", color="tab:blue", lw=1.6)
        axes[0, 0].plot(t_plot_ms, p_minus1, label="P(|-1>)", color="tab:green", lw=1.8)
        axes[0, 0].set_xlabel("Time (ms)"); axes[0, 0].set_ylabel("Population"); axes[0, 0].set_title("Populations"); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.2)

        axes[0, 1].plot(t_plot_ms, self.p.h_max * np.sin(self.omega_gw * tlist), color="purple", lw=1.6)
        axes[0, 1].set_xlabel("Time (ms)"); axes[0, 1].set_ylabel("h_+(t)"); axes[0, 1].set_title("Gravitational wave strain (reference)"); axes[0, 1].grid(alpha=0.2)

        axes[1, 0].plot(t_plot_ms, exp_Sz, color="tab:orange", lw=1.6)
        axes[1, 0].set_xlabel("Time (ms)"); axes[1, 0].set_ylabel("<S_z>"); axes[1, 0].set_title("Spin expectation"); axes[1, 0].grid(alpha=0.2)

        axes[1, 1].plot(freqs_hz/1e3, fft_amp, color="magenta", lw=1.2)
        axes[1, 1].set_xlabel("Frequency (kHz)"); axes[1, 1].set_ylabel("FFT amplitude"); axes[1, 1].set_title("FFT of P(|0>)"); axes[1, 1].grid(alpha=0.2)

        plt.show()

        if save_csv or self.p.save_csv:
            filename = save_csv or self.p.csv_filename
            log.info(f"Saving results to {filename}")
            header = ["time_s", "P_plus1", "P_0", "P_minus1", "exp_Sz"]
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for i, t in enumerate(tlist):
                    writer.writerow([t, p_plus1[i], p_0[i], p_minus1[i], exp_Sz[i]])
            log.info("CSV saved.")

        return {
            "p_plus1": p_plus1,
            "p_0": p_0,
            "p_minus1": p_minus1,
            "exp_Sz": exp_Sz,
            "freqs_hz": freqs_hz,
            "fft_amp": fft_amp,
            "omega_rabi_hz": omega_rabi_hz,
        }


# ---------- main demo ----------
def main_demo():
    params = SimulatorParams(
        D_hz=2.87e9,
        gamma_e_hz_per_T=28e9,
        Bz_T=1e-4,
        h_max=1e-18,
        kappa=1e-3,
        f_gw_hz=2e3,
        t_final_s=1e-3,
        nsteps=4000,
        demo_mode=True,
        use_mesolve=False,
        save_csv=False,
    )

    sim = NVGWSuperSimulator(params)
    result, tlist = sim.run()

    # Safe printing of initial population
    if hasattr(result, "expect") and len(result.expect) >= 2:
        print(f"Initial P(|0>) = {result.expect[1][0]:.6f}")
    else:
        print("Initial population info not available in result.expect. (That's OK.)")

    out = sim.analyze_and_plot(result, tlist)

    # small coarse frequency scan (demo) - remove/skip if too slow
    fscan = np.linspace(0.5 * params.f_gw_hz, 2.0 * params.f_gw_hz, 8)
    print("Running a coarse frequency scan (demo). This runs multiple solves and may take time...")
    freqs, finals = sim.freq_scan(fscan, which_outcome="+1")
    print("Freq scan results (Hz):", freqs)
    print("Final P(+1):", finals)

    # Now run mesolve with decoherence as demonstration (shorter run)
    params.use_mesolve = True
    params.T2_s = 500e-6
    params.t_final_s = 200e-6
    params.nsteps = 2000
    sim2 = NVGWSuperSimulator(params)
    result2, tlist2 = sim2.run()
    sim2.analyze_and_plot(result2, tlist2)

if __name__ == "__main__":
    main_demo()

