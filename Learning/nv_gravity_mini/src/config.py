from dataclasses import dataclass
from typing import Optional
import numpy as np
import qutip as qt

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


    def constructing_gamma_matrices():
        """We construct the Dirac Gamma matrices in the standard (Dirac) representation """

        # We know pauli matrices in Qutip
        sigma_x = qt.sigmax() #σ_x 
        sigma_y = qt.sigmay() #σ_y
        sigma_z = qt.sigmaz() #σ_z
        I2 = qt.qeye(2) # The 2x2 identity matrix 


        # Gamma matrices using tensor products
        gamma_0 = qt.tensor(sigma_z, I2)  # γ⁰ = σ_z ⊗ I
        gamma_1 = qt.tensor(qt.sigmay(), sigma_x)  # γ¹ = iσ_y ⊗ σ_x
        gamma_2 = qt.tensor(qt.sigmay(), sigma_y)  # γ² = iσ_y ⊗ σ_y  
        gamma_3 = qt.tensor(qt.sigmay(), sigma_z)  # γ³ = iσ_y ⊗ σ_z
        gamma_5 = qt.tensor(sigma_x, I2)  # γ⁵ = σ_x ⊗ I

        gamma_mu = gamma_0+gamma_1+gamma_2+gamma_3
    
        return gamma_0, gamma_1, gamma_2, gamma_3, gamma_5


# --------------------------------------------------------------------
# Alternative: chiral (Weyl) representation
# --------------------------------------------------------------------

#def construct_chiral_gamma_matrices():
    """Construct gamma matrices in chiral representation"""
    I2 = qt.qeye(2)
    sigma_x = qt.sigmax()
    sigma_y = qt.sigmay()
    sigma_z = qt.sigmaz()
    
    gamma_0 = qt.tensor(sigma_x, I2)  # γ⁰
    gamma_1 = qt.tensor(qt.sigmay(), sigma_x)  # γ¹
    gamma_2 = qt.tensor(qt.sigmay(), sigma_y)  # γ²
    gamma_3 = qt.tensor(qt.sigmay(), sigma_z)  # γ³
    gamma_5 = qt.tensor(sigma_z, I2)  # γ⁵
    
    #return gamma_0, gamma_1, gamma_2, gamma_3, gamma_5
    
# --------------------------------------------------------------------