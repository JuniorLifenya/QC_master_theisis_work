from dataclasses import dataclass,field
from typing import Optional
import numpy as np
import warnings
warnings.filterwarnings('ignore')



# --------------------------------------------------------------------
# Data classes for simulation parameters, actually used in the simulation
# --------------------------------------------------------------------
@dataclass
class SimulationConfig: 
    """"Here we have all parameters needed for the simulation"""

     # NV Physics
    D: float = 2.87e9
    gamma_e: float = 28e9
    Bz: float = 0.01

    # GW Parameters  
    f_gw: float = 1e5
    h_max: float = 1e-20
    kappa: float = 1e15
    
    # Simulation
    t_final: float = 0.001
    n_steps: int = 5000
    use_mesolve: bool = False # We initiate the parameters for mesolve
    # Best definition of property


    
    # Options for output
    save_animation: bool = False
    demo_mode: bool = True # We will use this
    Realistic_mode: bool = False # This is what we will have inside INTERN


    # @property Alternative methode
    # def omega_gw(self):
    #     return 2*np.pi * self.f_gw


    # --- Decoherence ---
    T1: float = 0.0         # Longitudinal relaxation (s), 0 = infinite
    T2: float = 100e-6      # Transverse relaxation (s)

    def __post_init__(self):
        """Calculate derived units after initialization."""
        self.omega_gw = 2 * np.pi * self.f_gw
        self.gamma_T1 = 1.0 / self.T1 if self.T1 > 0 else 0.0
        self.gamma_T2 = 1.0 / self.T2 if self.T2 > 0 else 0.0

    @property
    def tlist(self):
        return np.linspace(0, self.t_final, self.n_steps)
    

    

    