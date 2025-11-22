import qutip as qt
import numpy as np
from src.config import SimulateParameters
from src.Dirac_config import DiracMatrices
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
    
    # Usage example in your NVCenter class
    def get_Fw_gw_interaction_operator(self):
        """Proper implementation with sigma_ab = (i/2)[γ_a, γ_b]"""
        
        # First construct the gamma matrices
        dirac = DiracMatrices()
        gamma0, gamma1, gamma2, gamma3, gamma5 = dirac.gamma_matrices
        
        # Define sigma_ab = (i/2)[γ_a, γ_b] for the required combinations
        # In your PDF, you need σ_ab where a,b = 0,1,2,3
        
        # Example: σ_01 = (i/2)[γ_0, γ_1]
        sigma_01 = (1j/2) * (gamma0 * gamma1 - gamma1 * gamma0)
        sigma_02 = (1j/2) * (gamma0 * gamma2 - gamma2 * gamma0) 
        sigma_03 = (1j/2) * (gamma0 * gamma3 - gamma3 * gamma0)
        sigma_12 = (1j/2) * (gamma1 * gamma2 - gamma2 * gamma1)
        sigma_13 = (1j/2) * (gamma1 * gamma3 - gamma3 * gamma1)
        sigma_23 = (1j/2) * (gamma2 * gamma3 - gamma3 * gamma2)
        
        # Store in a dictionary for easy access
        sigma_dict = {
            (0,1): sigma_01, (0,2): sigma_02, (0,3): sigma_03,
            (1,2): sigma_12, (1,3): sigma_13, (2,3): sigma_23
        }
        
        # Now calculate the spin connection terms ω_μ^{ab}
        # From your PDF: ω_μ^{ab} ≈ k/2 (∂^a h_μ^b - ∂^b h_μ^a)
        omega_mu = self.calculate_spin_connection_components()
        
        # Build the spin connection operator: Γ_μ = (1/4) ω_μ^{ab} σ_ab
        Gamma_mu = self.build_spin_connection_operator(omega_mu, sigma_dict)
        
        # The full interaction Hamiltonian from your PDF
        H_gw = self.build_full_interaction_hamiltonian(Gamma_mu, dirac)
        
        return H_gw
        
    def calculate_spin_connection_components(self):
        """Calculate ω_μ^{ab} components from GW strain derivatives"""
        # This is simplified - you'll need to implement based on your specific GW model
        # ω_μ^{ab} ≈ k/2 (∂^a h_μ^b - ∂^b h_μ^a)
        
        omega_components = {}
        
        # Placeholder values - replace with actual GW strain derivative calculations
        for mu in range(4):  # μ = 0,1,2,3
            for a in range(4):
                for b in range(4):
                    if a != b:
                        # Simple placeholder - should come from actual h_μν derivatives
                        omega_components[(mu, a, b)] = self.cfg.kappa * 0.1 * (a - b)
        
        return omega_components

    def build_spin_connection_operator(self, omega_mu, sigma_dict):
        """Build Γ_μ = (1/4) ω_μ^{ab} σ_ab"""
        Gamma_operators = []
        
        for mu in range(4):
            Gamma_mu = qt.qzero(4)  # Start with zero operator
            
            for (a, b), sigma_ab in sigma_dict.items():
                if a < b:  # Avoid double counting
                    omega_component = omega_mu.get((mu, a, b), 0)
                    Gamma_mu += (1/4) * omega_component * sigma_ab
            
            Gamma_operators.append(Gamma_mu)
        
        return Gamma_operators

    def build_full_interaction_hamiltonian(self, Gamma_mu, dirac):
        """Build the full interaction Hamiltonian from your PDF"""
        alpha_x, alpha_y, alpha_z = dirac.alpha_matrices
        
        # From your PDF structure:
        # H = α·(p + something) + βm + spin_connection_terms
        
        # Spatial components (Γ_1, Γ_2, Γ_3)
        H_spatial = (
            alpha_x * Gamma_mu[1] +
            alpha_y * Gamma_mu[2] + 
            alpha_z * Gamma_mu[3]
        )
        
        # Temporal component (Γ_0)
        H_temporal = dirac.gamma0 * Gamma_mu[0]
        
        # Mass term
        H_mass = dirac.beta * self.cfg.mass
        
        return H_spatial + H_temporal + H_mass
                