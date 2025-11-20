# nv_gw_merged.py
"""
Merged concise NV-center GW simulator.
Features:
 - SimulatorParams dataclass
 - NVGW class: get_hamiltonian, run (sesolve/mesolve), analyze, plot, freq_scan
 - Robust sanity checks, units-consistent (angular freq internally)
 - Demo-mode for visible dynamics
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Dict
import logging

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("NVGW")

@dataclass
class SimulatorParams:
    D_hz: float = 2.87e9             # zero-field splitting (Hz)
    gamma_e_hz_per_T: float = 28e9   # gyromagnetic ratio (Hz/T)
    Bz_T: float = 0.0                # static field (T)
    h_max: float = 1e-18             # GW strain amplitude
    kappa: float = 1e-3              # coupling (rad/s per strain or similar)
    f_gw_hz: float = 100.0           # GW frequency (Hz)
    t_final_s: float = 0.1           # simulation time (s)
    nsteps: int = 2000               # time steps
    demo_mode: bool = False          # amplify h and kappa for visibility
    T1_s: Optional[float] = None     # T1 relaxation (s)
    T2_s: Optional[float] = None     # T2 dephasing (s)
    use_mesolve: bool = False        # run mesolve if True
    def validate(self):
        assert self.nsteps >= 2 and self.t_final_s > 0

class NVGW:
    def __init__(self, params: SimulatorParams):
        params.validate()
        self.p = params
        # spin-1 op
        self.Sx = qt.jmat(1, "x")
        self.Sy = qt.jmat(1, "y")
        self.Sz = qt.jmat(1, "z")
        self.psi_p1 = qt.basis(3, 0)
        self.psi_0  = qt.basis(3, 1)
        self.psi_m1 = qt.basis(3, 2)
        self.Op_plus = self.Sx**2 - self.Sy**2
        # demo amplification for visibility
        if self.p.demo_mode:
            log.info("demo_mode ON: amplifying h_max and kappa for visualization")
            self.p.h_max *= 1e6
            self.p.kappa *= 1e12
        # convert frequencies to angular frequency for use with t in seconds
        self.D = 2 * np.pi * self.p.D_hz
        self.gamma_e = 2 * np.pi * self.p.gamma_e_hz_per_T
        self.omega_gw = 2 * np.pi * self.p.f_gw_hz

    def gw_coeff(self, t: float, args: Dict) -> float:
        return args["A"] * np.sin(args["omega"] * t)

    def get_hamiltonian(self):
        H0 = self.D * (self.Sz**2)
        if self.p.Bz_T != 0.0:
            H0 += self.gamma_e * self.p.Bz_T * self.Sz
        H_int = self.p.kappa * self.Op_plus
        return [H0, [H_int, self.gw_coeff]]

    def get_collapse_ops(self):
        c_ops = []
        if self.p.T2_s:
            gamma_T2 = 1.0 / self.p.T2_s
            c_ops.append(np.sqrt(gamma_T2) * self.Sz)
        if self.p.T1_s:
            gamma_T1 = 1.0 / self.p.T1_s
            c_ops.append(np.sqrt(gamma_T1) * (self.psi_0 * self.psi_p1.dag()))
            c_ops.append(np.sqrt(gamma_T1) * (self.psi_0 * self.psi_m1.dag()))
        return c_ops

    def run(self):
        tlist = np.linspace(0.0, self.p.t_final_s, self.p.nsteps)
        H_td = self.get_hamiltonian()
        args = {"A": self.p.h_max, "omega": self.omega_gw}
        e_ops = [self.psi_p1.proj(), self.psi_0.proj(), self.psi_m1.proj(), self.Sz]
        if self.p.use_mesolve and (self.p.T1_s or self.p.T2_s):
            rho0 = self.psi_0.proj()
            c_ops = self.get_collapse_ops()
            log.info("Running mesolve (open system)...")
            result = qt.mesolve(H_td, rho0, tlist, c_ops, e_ops=e_ops, args=args)
        else:
            psi0 = self.psi_0
            log.info("Running sesolve (pure state)...")
            result = qt.sesolve(H_td, psi0, tlist, e_ops=e_ops, args=args)
        # robust sanity: prefer result.expect if available
        if hasattr(result, "expect") and len(result.expect) >= 2:
            initial_P0 = float(np.real(result.expect[1][0]))
            log.info(f"Initial P(|0>) = {initial_P0:.6f}")
        else:
            log.info("No expect data available for sanity check.")
        self.result = result
        self.tlist = tlist
        return result, tlist

    def compute_matrix_element(self, bra, op, ket):
        q = (bra.dag() * op * ket)
        try:
            val = q.full()[0, 0]
        except Exception:
            val = complex(q)
        return float(np.abs(val))

    def estimate_rabi(self):
        # element of H_int between |+1> and |-1> or between |+1> and |0> depending on interest
        H_int = self.p.kappa * self.Op_plus
        matel = self.compute_matrix_element(self.psi_p1, H_int, self.psi_m1)
        # Rabi amplitude induced by h_max: Omega ≈ |kappa * h_max * <i|Op|j>|
        Omega_rad_s = np.abs(matel) * self.p.h_max
        return Omega_rad_s, Omega_rad_s / (2*np.pi)

    def analyze(self):
        if not hasattr(self, "result"):
            raise RuntimeError("Run simulation first")
        p_plus1, p_0, p_minus1, exp_Sz = [np.real(x) for x in self.result.expect]
        gw_strain = self.p.h_max * np.sin(self.omega_gw * self.tlist)
        omega_rabi_rad, omega_rabi_hz = self.estimate_rabi()
        xf, yf = self.fft(p_0, self.tlist)
        metrics = {
            "p_plus1": p_plus1, "p_0": p_0, "p_minus1": p_minus1,
            "exp_Sz": exp_Sz, "gw_strain": gw_strain,
            "omega_rabi_hz": omega_rabi_hz,
            "fft_freqs": xf, "fft_amp": yf
        }
        return metrics

    def fft(self, signal, tlist):
        dt = tlist[1] - tlist[0]
        yf = rfft(signal - np.mean(signal))
        xf = rfftfreq(len(signal), dt)
        return xf, np.abs(yf)

    def plot(self, metrics):
        t_ms = self.tlist * 1e3
        p_plus1 = metrics["p_plus1"]; p_0 = metrics["p_0"]; p_minus1 = metrics["p_minus1"]
        gw = metrics["gw_strain"]
        fig, axes = plt.subplots(2,2, figsize=(10,8), constrained_layout=True)
        axes[0,0].plot(t_ms, p_0, label="P(|0>)"); axes[0,0].plot(t_ms, p_plus1, label="P(|+1>)"); axes[0,0].plot(t_ms, p_minus1, label="P(|-1>)")
        axes[0,0].set_xlabel("Time (ms)"); axes[0,0].set_ylabel("Population"); axes[0,0].legend(); axes[0,0].grid(True)
        axes[0,1].plot(t_ms, gw); axes[0,1].set_xlabel("Time (ms)"); axes[0,1].set_ylabel("h_+(t)"); axes[0,1].grid(True)
        axes[1,0].plot(t_ms, metrics["exp_Sz"]); axes[1,0].set_xlabel("Time (ms)"); axes[1,0].set_ylabel("<Sz>"); axes[1,0].grid(True)
        freqs = metrics["fft_freqs"]; amp = metrics["fft_amp"]
        axes[1,1].plot(freqs, amp); axes[1,1].set_xlabel("Freq (Hz)"); axes[1,1].set_ylabel("FFT amp"); axes[1,1].grid(True)
        plt.show()

    def freq_scan(self, f_array_hz: Sequence[float], which="+1"):
        finals = []
        orig_f = self.p.f_gw_hz
        orig_omega = self.omega_gw
        for f in f_array_hz:
            self.p.f_gw_hz = f
            self.omega_gw = 2*np.pi*f
            res, t = self.run()
            if hasattr(res, "expect") and len(res.expect) >= 3:
                if which=="+1": finals.append(np.real(res.expect[0][-1]))
                elif which=="0": finals.append(np.real(res.expect[1][-1]))
                else: finals.append(np.real(res.expect[2][-1]))
            else:
                finals.append(np.nan)
        self.p.f_gw_hz = orig_f
        self.omega_gw = orig_omega
        return np.array(f_array_hz), np.array(finals)

# ---------------- simple demo ----------------
if __name__ == "__main__":
    # Demo params
    params = SimulatorParams(demo_mode=True, f_gw_hz=2000.0, t_final_s=1e-3, nsteps=2000)
    sim = NVGW(params)
    res, t = sim.run()
    metrics = sim.analyze()
    print(f"Estimated Rabi frequency (Hz): {metrics['omega_rabi_hz']:.6e}")
    sim.plot(metrics)
