import qutip as qt
import numpy as np 

class DiracMatrices:
        """
        Dirac gamma matrices utility.
        Note: These are 4x4 matrices. 
        """

        def __init__(self):
            self.setup_gamma_matrices()
            self.setup_alpha_beta_matrices()
        
        def setup_gamma_matrices(self):
            """Constructing Dirac Gamma matrices in Dirac representation"""

            # We know Pauli Matrices in Qutip
            sigma_x = qt.sigmax() #σ_x 
            sigma_y = qt.sigmay() #σ_y
            sigma_z = qt.sigmaz() #σ_z
       
            I2 = qt.qeye(2) # The 2x2 identity matrix 
            zero = qt.qzero(2)
       
            # Gamma matrices (4x4)
            self.gamma0 = qt.tensor(sigma_z, I2)  # γ⁰= σ_z ⊗ I
            self.gamma1 = 1j * qt.tensor(sigma_y, sigma_x)  # γ¹ = iσ_y ⊗ σ_x
            self.gamma2 = 1j * qt.tensor(sigma_y, sigma_y)  # γ² = iσ_y ⊗ σ_y 
            self.gamma3 = 1j * qt.tensor(sigma_y, sigma_z)  # γ³ = iσ_y ⊗ σ_z
            self.gamma5 = qt.tensor(sigma_x, I2)  # γ⁵ = σ_x ⊗ I

            # Store in list for easy access
            self.gamma_matrices = [self.gamma0, self.gamma1, self.gamma2, self.gamma3, self.gamma5]
            #self.gamma_mu = self.gamma0 + self.gamma1 + self.gamma2 + self.gamma3
            #self.gamma_nu = self.gamma0 + self.gamma1 + self.gamma2 + self.gamma3

            #self.Gab = (np.sqrt(-1)/2)(self.gamma_mu*self.gamma_nu - self.gamma_nu*self.gamma_mu)


        def setup_alpha_beta_matrices(self):

            """Construct alpha and beta matrices for Dirac Hamiltonian"""
            # Alpha matrices: αᵢ = γ⁰γᵢ
            self.alpha1 = self.gamma0 * self.gamma1
            self.alpha2 = self.gamma0 * self.gamma2
            self.alpha3 = self.gamma0 * self.gamma3
            
            # Beta matrix: β = γ⁰
            self.beta = self.gamma0
            
            self.alpha_matrices = [self.alpha1, self.alpha2, self.alpha3]
        def get_sigma_ab(self, a: int, b: int) -> qt.Qobj:
            """Calculate sigma_ab = (i/2)[gamma_a, gamma_b]"""
            commutation = self.gammas[a] * self.gammas[b] - self.gammas[b] * self.gammas[a]
            return (1j / 2) * commutation

        def get_spin_connection_operator(self, omega_mu, derivative_terms):
            """
            Construct spin connection operator for gravitational coupling
            Based on your PDF: Γ_μ = (i/4) ω_μ^{ab} σ_{ab}. Where σ_{ab} = (i/2)[γ_a, γ_b]
            and ω_μ^{ab} = k(∂^a h_{μ}^b)
            """
            omega_mu = derivative_terms  # Placeholder for actual derivative terms
            spin_connection = 0
            for a in range(4):
                for b in range(4):
                    sigma_ab = self.get_sigma_ab(a, b)
                    spin_connection += (1j / 4) * omega_mu[a][b] * sigma_ab
            return spin_connection
            
# --------------------------------------------------------------------
# Alternative: chiral (Weyl) representation
# --------------------------------------------------------------------

def construct_chiral_gamma_matrices():
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