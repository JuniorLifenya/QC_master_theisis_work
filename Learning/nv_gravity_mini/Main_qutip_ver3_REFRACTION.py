# Compared to the previouse version, this version includes refraction at the interfaces of different media.
from dataclasses import dataclass
from typing import Optional
import qutip as qt
import numpy as np 
import matplotlib.pyplot as plt

# --------------------------------------------------------------------
# Small tutorial on OOP in Python
# --------------------------------------------------------------------

class NVcenter_demo:
    def__init__(self, D=2.87e9, Bz=0.0) # type: ignore 
    """" Init runs autumatically when we create an object"""
    """ Self refers to THIS specific NV center instance not all NV centers"""
    self.D = D  # type: ignore # Zero-field splitting in Hz
    self.Bz = Bz  # type: ignore # Magnetic field along the NV axis in Tesla
    self.setup_operators() # type: ignore

    def some_method(self):
        # All methods need "self" to access the object'sdata
        return self.D * self .Sz*Sz  # type: ignore
# usage example
nv1 = NVcenter_demo(D=2.87e9, Bz=0.01)
nv2 = NVcenter_demo(D=2.87e9, Bz=0.02)
H1 = nv1.some_method()
H2 = nv2.some_method()
H_total = H1 + H2

# --------------------------------------------------------------------
# Data classes for simulation parameters, actually used in the simulation
# --------------------------------------------------------------------

@dataclass
class SimulateParameters: 
    """"Here we have all parameters needed for the simulation"""

     # NV Physics
    D: float = 2.87e9
    gamma_e: float = 28e9
    Bz: float = 0.0
    
    # GW Parameters  
    f_gw: float = 1000
    h_max: float = 1e-20
    kappa: float = 1e15
    
    # Simulation
    t_final: float = 0.001
    n_steps: int = 1000

    def __post__init__(self): # type: ignore Here we can compute derived parameters
        self.dt = self.t_final / self.n_steps

# --------------------------------------------------------------------
# Main simulation class, QuantumSystem
# --------------------------------------------------------------------

class QuantumSystem:
    """Base class for any quantum system simulation"""

    def __init__(self,params):
        self.params = params
        self.setup_operators() # These are methods defined below
        self.setup_states()

    def setup_operators(self):
        raise NotImplementedError("Subclasses must implement this!")
    
    def setup_states(self):
        raise NotImplementedError("Subclasses must implement this!")
    
    def get_hamiltonian(self):
        raise NotImplementedError("Subclasses must implement this!")
    
# --------------------------------------------------------------------
# Subclass for NV Center specific implementation
# --------------------------------------------------------------------    
class NVCenter(QuantumSystem):

    """NV Center quantum system implementation"""
    def setup_operators(self):
        """Define spin operators for spin-1 system"""
        self.Sx = np.array([[0, 1/np.sqrt(2), 0],
                            [1/np.sqrt(2), 0, 1/np.sqrt(2)],
                            [0, 1/np.sqrt(2), 0]])
        self.Sy = np.array([[0, -1j/np.sqrt(2), 0],
                            [1j/np.sqrt(2), 0, -1j/np.sqrt(2)],
                            [0, 1j/np.sqrt(2), 0]])
        self.Sz = np.array([[1, 0, 0],
                            [0, 0, 0],
                            [0, 0, -1]])
        #Square of operators
        self.Sx2 = np.dot(self.Sx, self.Sx)
        self.Sy2 = np.dot(self.Sy, self.Sy)
        self.Sz2 = np.dot(self.Sz, self.Sz)
    
    """ Define initial states """
    def setup_states(self):
        self.psi_p1 = qt.basis(3, 0)  # |+1>
        self.psi_0 = qt.basis(3, 1)   # |0>
        self.psi_m1 = qt.basis(3, 2)  # |-1>

    """Construct the Hamiltonian for the NV center"""
    def get_hamiltonian(self):

        Bz = 0.0
        H0 = self.params.D * self.Sz2 
        if self.params.Bz != 0.0:
            H0 += self.params.gamma_e * self.params.Bz * self.Sz
        return H0

    def get_gw_interaction(self, t):
        """Gravitational wave interaction Hamiltonian"""
        h_t = self.params.h_max * np.sin(2 * np.pi * self.params.f_gw * t) * np.exp(-t / self.params.kappa)

        H_gw = h_t * (self.Sx2 - self.Sy2)
        # Alternatively H_gw = self.params.kappa *(self.Sx2 - self.Sy2)
        return H_gw
    

# --------------------------------------------------------------------
# Simulation engine class
# --------------------------------------------------------------------   
