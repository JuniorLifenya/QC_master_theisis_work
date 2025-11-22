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

    #def get_gw_interaction_operator(self):
        """Gravitational wave interaction Hamiltonian"""
        #h_t = self.params.h_max * np.sin(2 * np.pi * self.params.f_gw * t) * np.exp(-t / self.params.kappa)
        H_gw = self.cfg.kappa * (self.Op_plus)
        # This is a postulated form, please adjust based on actual physics
        return H_gw    
    

    def get_fw_gw_interaction_operator(self):

        """Minimal FW-style interaction for testing"""
        # Use your spin matrices with proper GW coupling structure
        Sx, Sy, Sz = self.Sx, self.Sy, self.Sz
        
        # From PDF we have: H = ψ†[(G_ab/4)(ω_0^{ab} + α·ω_i^{ab}) + βm + α·p]ψ
        # We really have it in the form: H = α[p + (-Gab/4)ωi^{ab}] + βm + (Gab/4)ω0^{ab}

        # With ω_μ^{ab} ≈ k/2 (∂^ah_μ^b - ∂^bh_μ^a) 

        # # Alternative (later) with this : (i/4)ωμ^{ab}(Gab) ≈ (ik/4)(∂^ahμ^bGab)
        # Then we obtain something else: H = α[p + (-kGab/4)∂^ahi^b] + βm + (-kGab/4)∂^ah0^b

        


        #Implementation of the spin-connection coupling 
        w_i = self.calculate_w_i() # ωi^{ab} term
        w_0 = self.calculate_w_0() # ω0^{ab} term
        
        # Spatial spin connection coupling with α
        H_i = (self.cfg.Gab / 4) * (
            Sx * w_i[0] +
            Sy * w_i[1] + 
            Sz * w_i[2] 
        )

        # Temporal spin-connection coupling (ω0^{ab} term)
        H_t = (self.cfg.Gab / 4) * w_0 * qt.qeye(3)

        # Effective momentum coupling from GW strain derivatives 
        H_p = self.build_p_coupling()

        return H_i + H_t + H_p
    
    def calculate_w_i(self):
    # From your PDF: ω_b^a ≈ k/2 (G_ab - ∂_k^a)
    # For simulation, return placeholder operators
        return [qt.sigmax(), qt.sigmay(), qt.sigmaz()]

    def calculate_w_0(self):
        """Calculate temporal spin-connection component ω_0^a"""
        # Placeholder - should come from GW time derivatives
        return qt.sigmax()

    def build_momentum_coupling(self):
        """Build the momentum-spin coupling terms"""
        Sx, Sy, Sz = self.Sx, self.Sy, self.Sz
        
        # From your PDF: terms like α·p where p includes GW effects
        # Use strain derivative operators
        p_x, p_y, p_z = [qt.sigmax(), qt.sigmay(), qt.sigmaz()]
        
        return (Sx * p_x + Sy * p_y + Sz * p_z) * self.cfg.kappa