import qutip as qt
import numpy as np
from src.config import SimulateParameters
cfg = SimulateParameters()

# --------------------------------------------------------------------
# Main simulation class, QuantumSystem
# --------------------------------------------------------------------
class QuantumSystem:
    """Base class for any quantum system simulation"""

    def __init__(self,config: SimulateParameters):
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
    

    
    def get_hamiltonian_0(self):
        """Construct the Hamiltonian for the NV center"""
        H0 = self.cfg.D * self.Sz2 
        if self.cfg.Bz != 0.0:

            H0 += self.cfg.gamma_e * self.cfg.Bz * self.Sz
        return H0

    def get_gw_interaction_operator(self):
        """Gravitational wave interaction Hamiltonian"""
        #h_t = self.params.h_max * np.sin(2 * np.pi * self.params.f_gw * t) * np.exp(-t / self.params.kappa)
        H_gw = self.cfg.kappa * (self.Op_plus)
        # This is a postulated form, please adjust based on actual physics
        return H_gw    
