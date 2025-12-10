import qutip as qt
import numpy as np
import logging
from src.nv_quantum_setup import NVCenter

# Configure logger
logger = logging.getLogger(__name__)

class SimulationEngine:
    """Manages the time-evolution of the quantum system."""

    def __init__(self, system: NVCenter):
        self.system = system
        self.cfg = system.cfg

    def _strain_func(self, t, args):
        """Time-dependent coefficient for Hamiltonian: h(t)"""
        return args['h_max'] * np.sin(args['omega_gw'] * t)

    def run(self):
        """Executes the simulation using QuTiP mesolve/sesolve."""
        logger.info(f"Starting Simulation: {self.cfg.n_steps} steps, T_final={self.cfg.t_final}s")
        
        # 1. Construct Hamiltonian: H = H0 + [H_int, h(t)]
        H0 = self.system.get_static_hamiltonian()
        H_int = self.system.get_interaction_operator()
        
        # QuTiP format for time-dependent H
        H = [H0, [H_int, self._strain_func]]
        args = {'h_max': self.cfg.h_max, 'omega_gw': self.cfg.omega_gw}
        
        # --- UPDATE 1: Track Sy as well ---
        e_ops = [
            self.system.psi_0 * self.system.psi_0.dag(),   # 0: Pop |0>
            self.system.psi_p1 * self.system.psi_p1.dag(), # 1: Pop |+1>
            self.system.psi_m1 * self.system.psi_m1.dag(), # 2: Pop |-1>
            self.system.Sz,                                # 3: Sz
            self.system.Sx,                                # 4: Sx
            self.system.Sy                                 # 5: Sy (Added)
        ]
        
        c_ops = self.system.get_collapse_ops() if self.cfg.use_mesolve else []
        options = {'nsteps': 1_000_000, 'progress_bar': False} # Turn off bar for loops
        
        if c_ops:
            rho0 = self.system.psi_initial * self.system.psi_initial.dag()
            result = qt.mesolve(H, rho0, self.cfg.tlist, c_ops, e_ops, args=args, options=options)
        else:
            result = qt.sesolve(H, self.system.psi_initial, self.cfg.tlist, e_ops, args=args, options=options)
            
        return result
# --- UPDATE 2: Frequency Scan Logic ---
    def frequency_scan(self, freqs: np.ndarray):
        """
        Scans over a range of GW frequencies to find resonance.
        Returns: (frequencies, max_population_transfer)
        """
        logger.info(f"🔍 Starting Frequency Scan over {len(freqs)} points...")
        
        original_f = self.cfg.f_gw
        results = []

        for f in freqs:
            # Update Config
            self.cfg.f_gw = f
            self.cfg.omega_gw = 2 * np.pi * f
            
            # Run Quietly
            res = self.run()
            
            # Metric: Max population in |+1> (Resonance indicator)
            # We take the maximum value reached during the simulation
            max_p1 = np.max(res.expect[1]) 
            results.append(max_p1)

        # Restore original state
        self.cfg.f_gw = original_f
        self.cfg.omega_gw = 2 * np.pi * original_f
        
        return freqs, np.array(results)