from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import warnings
from pydantic import BaseModel, Field, field_validator
warnings.filterwarnings('ignore')

# --------------------------------------------------------------------
# Data classes for simulation parameters, actually used in the simulation
# --------------------------------------------------------------------
class SimulationConfig(BaseModel): 
    
    """"
    Here we have all parameters needed for the simulation
    Here we try to add some Pydantic, 
    which is a widely used Python library for data validation,
    parsing,and serialization using standard Python type hints
    """

    # NV Physics
    D: float = Field(2.87e9, description = "Zfs in GHz")
    gamma_e: float = 28e9
    Bz: float = 0.01

    # GW Parameters  
    f_gw: float = Field(1e5, ge=0, description = "GW freq (Hz)")

    @field_validator('f_gw')
    @classmethod
    def validate_f_gw(cls, v):
        if v <= 0:
            raise ValueError("GW frequency must be positive")
        return v
    
    @property
    def omega_gw(self):
        return 2 * np.pi * self.f_gw
    
    h_max: float = 1e-20
    kappa: float = 1e15
    
    # Simulation
    t_final: float = 0.001
    n_steps: int = 5000
    use_mesolve: bool = False # We initiate the parameters for mesolve

    # Options for output
    save_animation: bool = False
    demo_mode: bool = True # We will use this
    Realistic_mode: bool = False # This is what we will have inside INTERN

    # --- Decoherence ---
    T1: float = 0.0         # Longitudinal relaxation (s), 0 = infinite
    T2: float = 100e-6      # Transverse relaxation (s)

    # In Pydantic, you generally use model_post_init or handle derived 
    # properties differently than standard dataclasses.
    # We will compute derived properties as properties.
    
    @property
    def gamma_T1(self):
         return 1.0 / self.T1 if self.T1 > 0 else 0.0
         
    @property
    def gamma_T2(self):
         return 1.0 / self.T2 if self.T2 > 0 else 0.0

    @property
    def tlist(self):
        return np.linspace(0, self.t_final, self.n_steps)