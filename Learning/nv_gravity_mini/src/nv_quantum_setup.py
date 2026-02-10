import qutip as qt
import numpy as np
from src.config import SimulationConfig
cfg = SimulationConfig()


# --------------------------------------------------------------------
# Main simulation class, QuantumSystem
# --------------------------------------------------------------------
class QuantumSystem:
    """Base class for any quantum system simulation"""

    def __init__(self,config: SimulationConfig):
        self.cfg = config
        self.setup_operators() # These are methods defined below
        self.setup_states()

    def setup_operators(self):
        raise NotImplementedError("Subclasses must implement this!") # Here we force subclasses to implement this method
        # Which means that any subclass of QuantumSystem must have this method implemented
    
    def setup_states(self):
        raise NotImplementedError("Subclasses must implement this!")
    
    def get_hamiltonian(self):
        raise NotImplementedError("Subclasses must implement this!")
    
# --------------------------------------------------------------------
# Subclass for NV Center specific implementation< 
# --------------------------------------------------------------------    
class NVCenter(QuantumSystem):

    """NV Center quantum system implementation"""

    def setup_operators(self):
        #Define spin operators for spin-1 system
        self.Sx = qt.jmat(1, 'x')
        self.Sy = qt.jmat(1, 'y')
        self.Sz = qt.jmat(1, 'z')

        #Square of operators
        self.Sx2 = self.Sx * self.Sx # Better then using dag og np (numpy)
        self.Sy2 = self.Sy * self.Sy
        self.Sz2 = self.Sz * self.Sz

        # GW Interaction Operators
        self.Op_plus = self.Sx2 - self.Sy2
        self.Op_cross = self.Sx * self.Sy + self.Sy * self.Sx
    
    """ Define initial states """
    def setup_states(self):
        self.psi_p1 = qt.basis(3, 0)  # |+1>
        self.psi_0 = qt.basis(3, 1)   # |0>
        self.psi_m1 = qt.basis(3, 2)  # |-1>
    # Initial state (usually prepared in |0>)
        self.psi_initial = self.psi_0

    def setup_analysis_colors(self):
        self.colors = {
            'p0': '#1f77b4', 'p1': '#d62728', 'm1': '#2ca02c',
            'gw': "#53257f", 'sz': '#ff7f0e', 'bg': '#f8f9fa','we': "#956416" 
        }
    

    
    
        
    def get_static_hamiltonian(self) -> qt.Qobj:
        """H0 = D * Sz^2 + gamma * B * Sz"""
        H0 = self.cfg.D * self.Sz2
        if self.cfg.Bz != 0:
            H0 += self.cfg.gamma_e * self.cfg.Bz * self.Sz
        return H0

    def get_interaction_operator(self) -> qt.Qobj:
        """Returns the operator part of the GW Hamiltonian."""
        return self.cfg.kappa * self.Op_plus

    def get_collapse_ops(self) -> list:
        """Construct collapse operators for noise simulation."""
        c_ops = []
        
        # T2 Dephasing (z-axis noise)
        if self.cfg.gamma_T2 > 0:
            c_ops.append(np.sqrt(self.cfg.gamma_T2) * self.Sz)
            
        # T1 Relaxation (decay to ground state or thermal mix)
        # Simplified: Decay from +1 to 0 and -1 to 0
        if self.cfg.gamma_T1 > 0:
            c_ops.append(np.sqrt(self.cfg.gamma_T1) * (self.psi_0 * self.psi_p1.dag()))
            c_ops.append(np.sqrt(self.cfg.gamma_T1) * (self.psi_0 * self.psi_m1.dag()))
            
        return c_ops
                