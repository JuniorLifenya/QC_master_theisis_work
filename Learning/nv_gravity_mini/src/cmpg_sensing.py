import numpy as np
import qutip as qt
import logging
from src.Engin_timeStepper import SimulationEngine

logger = logging.getLogger(__name__)

class SensingEngine(SimulationEngine):
    """
    Subclass specifically for Experimental Sequences (CPMG, DD).
    Inherits hardware/Hamiltonian logic from SimulationEngine.
    """

    def __init__(self, system, n_pulses=8):
        super().__init__(system)
        self.n_pulses = n_pulses
        # Setup specific to qubit sensing subspace (|0> and |-1>)
        self._setup_sensing_params()

    def _setup_sensing_params(self):
        """Calculate timings based on resonance condition: 2*tau = 1/f_gw"""
        self.tau = (1 / self.cfg.f_gw) / 2
        self.total_time = self.n_pulses * self.tau
        
        # Qubit subspace operators (Mapping Spin-1 to effective Spin-1/2)
        # Using the Sz transition between |0> and |-1>
        self.sz_eff = qt.sigmaz()
        self.sx_eff = qt.sigmax()
        self.sy_eff = qt.sigmay()
        
        # Rabi drive frequency for pulses
        self.omega_rabi = 500.0  # MHz (Strong limit)
        self.pulse_width = np.pi / self.omega_rabi

    def _control_func(self, t, args):
        """Pulse sequence function: Returns 1.0 when pulse is ON."""
        for k in range(args['n_pulses']):
            center_time = (k + 0.5) * args['tau']
            if abs(t - center_time) < (args['width'] / 2):
                return 1.0
        return 0.0

    def run_cpmg(self):
        """Executes the CPMG sequence simulation."""
        # 1. Prepare initial state: Superposition (|0> + |-1>)/sqrt(2)
        # Standard in sensing: pi/2 pulse creates coherence
        psi_init = (qt.basis(2, 0) + qt.basis(2, 1)).unit()

        # 2. Hamiltonians
        # H_gw: effective coupling is Sz (detuning) in rotating frame
        H_gw = self.cfg.h_max * self.sz_eff
        # H_control: Rabi drive around Sx axis
        H_control = self.omega_rabi * self.sx_eff

        # 3. Assemble Time-Dependent Hamiltonian
        # Note: We use the system's gw_signal logic
        H = [
            [H_gw, lambda t, args: np.cos(args['omega_gw'] * t)],
            [H_control, self._control_func]
        ]

        args = {
            'omega_gw': self.cfg.omega_gw,
            'tau': self.tau,
            'n_pulses': self.n_pulses,
            'width': self.pulse_width
        }

        # 4. Simulation time list (Higher density for pulses)
        tlist = np.linspace(0, self.total_time, self.cfg.n_steps)
        
        result = qt.mesolve(H, psi_init, tlist, [], 
                           [self.sx_eff, self.sy_eff, self.sz_eff], 
                           args=args)
        return result, tlist