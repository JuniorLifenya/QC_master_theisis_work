"""
NV_GW_Enterprise_Simulation_FAST.py

UPDATES:
1. SPEED: Replaced Python-callback Hamiltonian with Array-Interpolation (C++ speed).
2. UI: Added Progress Bar (tqdm/QuTiP native).
3. STABILITY: Adjusted tolerances for fast prototyping.
"""

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Optional
from scipy.fft import fft, fftfreq

# ==============================================================================
# 1. CONFIGURATION LAYER
# ==============================================================================

@dataclass
class NVConfig:
    """Immutable configuration object."""
    D_splitting: float = 2.87e9       
    gamma_e: float = 28e9             
    Bz: float = 0.02                  
    
    f_gw: float = 560.0               
    h_strain: float = 5e-6            
    kappa: float = 1e10               
    
    t_final: float = 0.1              
    n_steps: int = 10000              # Optimized for speed/resolution balance
    
    T2: Optional[float] = None        

    @property
    def omega_gw(self) -> float:
        return 2 * np.pi * self.f_gw

# ==============================================================================
# 2. PHYSICS ENGINE LAYER
# ==============================================================================

class NVEngine:
    def __init__(self, config: NVConfig):
        self.cfg = config
        self._build_operators()

    def _build_operators(self):
        self.Sx = qt.jmat(1, 'x')
        self.Sy = qt.jmat(1, 'y')
        self.Sz = qt.jmat(1, 'z')
        self.psi_p1 = qt.basis(3, 0)
        self.psi_0  = qt.basis(3, 1)
        self.psi_m1 = qt.basis(3, 2)
        self.Op_interaction = self.Sx**2 - self.Sy**2

    def get_hamiltonian(self, tlist: np.ndarray) -> list:
        """
        Constructs Hamiltonian using Array Interpolation (FAST).
        We pre-calculate the GW signal so the solver doesn't query Python every step.
        """
        # 1. Static Hamiltonian
        H0 = (self.cfg.D_splitting * self.Sz**2) + \
             (self.cfg.gamma_e * self.cfg.Bz * self.Sz)

        # 2. Time-Dependent Hamiltonian (Pre-calculated Array)
        # This creates a Cubic Spline in C++ backend automatically
        gw_signal = self.cfg.h_strain * np.sin(self.cfg.omega_gw * tlist)
        
        # [H_static, [H_operator, numpy_array_of_coefficients]]
        H_interaction = [self.cfg.kappa * self.Op_interaction, gw_signal]
        
        return [H0, H_interaction]

    def get_collapse_ops(self) -> List[qt.Qobj]:
        c_ops = []
        if self.cfg.T2:
            c_ops.append(np.sqrt(1.0 / self.cfg.T2) * self.Sz)
        return c_ops

    def get_observables(self) -> List[qt.Qobj]:
        return [self.psi_p1.proj(), self.psi_0.proj(), self.psi_m1.proj(), self.Sz]

# ==============================================================================
# 3. SOLVER LAYER (Now with Progress Bar)
# ==============================================================================

class QuantumSolver:
    def __init__(self, engine: NVEngine):
        self.engine = engine

    def run(self) -> Tuple[qt.Result, np.ndarray]:
        
        # Define time list first
        tlist = np.linspace(0, self.engine.cfg.t_final, self.engine.cfg.n_steps)
        
        # Pass tlist to engine to generate the pre-calculated array
        H = self.engine.get_hamiltonian(tlist)
        c_ops = self.engine.get_collapse_ops()
        e_ops = self.engine.get_observables()
        
        rho0 = self.engine.psi_p1 
        if c_ops: rho0 = rho0 * rho0.dag()

        # --- STABILITY OPTIONS & PROGRESS BAR ---
        options = {
            'nsteps': 1_000_000,      
            'atol': 1e-10,            # Slightly relaxed for speed (still high precision)
            'rtol': 1e-8,
            'max_step': 1e-5,         
            'progress_bar': 'text'    # <--- THIS SHOWS THE PROGRESS BAR
        }

        print(f"[*] Starting Simulation ({self.engine.cfg.n_steps} steps)... Please wait.")
        print(f"[*] Mode: {'MESOLVE (Open System)' if c_ops else 'SESOLVE (Unitary)'}")
        
        if c_ops:
            result = qt.mesolve(H, rho0, tlist, c_ops=c_ops, e_ops=e_ops, options=options)
        else:
            result = qt.sesolve(H, rho0, tlist, e_ops=e_ops, options=options)
            
        return result, tlist

# ==============================================================================
# 4. VISUALIZATION LAYER
# ==============================================================================

class DataVisualizer:
    @staticmethod
    def analyze_and_plot(result: qt.Result, tlist: np.ndarray, config: NVConfig):
        p_p1, p_0, p_m1, Sz_exp = [np.real(x) for x in result.expect]
        t_ms = tlist * 1000
        
        # FFT Analysis
        signal = p_p1 - p_m1
        N = len(tlist)
        dt = tlist[1] - tlist[0]
        yf = fft(signal - np.mean(signal))
        xf = fftfreq(N, dt)
        mask = xf > 0
        freqs = xf[mask]
        amps = np.abs(yf[mask])
        dominant_freq = freqs[np.argmax(amps)]

        fig = plt.figure(figsize=(14, 9))
        gs = fig.add_gridspec(2, 2)

        # Plot 1: Rabi Flopping
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(t_ms, p_p1, 'r-', lw=2, label=r'$|+1\rangle$')
        ax1.plot(t_ms, p_m1, 'g--', lw=2, label=r'$|-1\rangle$')
        ax1.set_title(f'Rabi Oscillations (Detected f={dominant_freq:.1f} Hz)')
        ax1.set_xlabel('Time (ms)'); ax1.set_ylabel('Population')
        ax1.legend(); ax1.grid(True, alpha=0.3)

        # Plot 2: GW Input
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(t_ms, config.h_strain * np.sin(config.omega_gw * tlist), 'purple', lw=1.5)
        ax2.set_title('Gravitational Wave Strain')
        ax2.set_xlabel('Time (ms)'); ax2.grid(True, alpha=0.3)

        # Plot 3: FFT
        ax3 = fig.add_subplot(gs[1, :])
        ax3.plot(freqs, amps, 'darkcyan', lw=2)
        ax3.set_title('Spectral Signature (FFT)')
        ax3.set_xlabel('Frequency (Hz)'); ax3.set_xlim(0, dominant_freq * 3)
        ax3.fill_between(freqs, amps, alpha=0.1, color='darkcyan')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    config = NVConfig() # Defaults are fine
    engine = NVEngine(config)
    solver = QuantumSolver(engine)

    try:
        result, tlist = solver.run()
        DataVisualizer.analyze_and_plot(result, tlist, config)
        print("\n[Success] Simulation Complete.")
    except KeyboardInterrupt:
        print("\n[!] User Aborted.")
    except Exception as e:
        print(f"\n[!] Error: {e}")