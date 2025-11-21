from dataclasses import dataclass
from typing import Optional
import numpy as np

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
    f_gw: float = 2000
    h_max: float = 1e-20
    kappa: float = 1e15
    
    # Simulation
    t_final: float = 0.001
    n_steps: int = 1000

    # Best definition of property
    @property
    def omega_gw(self):
        return 2*np.pi * self.f_gw
    
    #def __post_init__(self): # This runs automatically after init ALTERNATIVE
        #self.omega_gw = 2 * np.pi * self.f_gw
        #self.dt = self.t_final / self.n_steps

    # Decoherence (optional)
    T2: Optional[float] = None   # seconds, None = no dephasing



