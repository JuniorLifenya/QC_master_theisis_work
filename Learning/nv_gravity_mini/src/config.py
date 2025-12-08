from dataclasses import dataclass
from typing import Optional
import numpy as np
import warnings
warnings.filterwarnings('ignore')



# --------------------------------------------------------------------
# Data classes for simulation parameters, actually used in the simulation
# --------------------------------------------------------------------
@dataclass
class SimulateParameters: 
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
    

    

    