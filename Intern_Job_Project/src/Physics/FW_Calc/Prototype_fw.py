import sympy as sp
from sympy.physics.matrices import msigma, mgamma

def commutator(A, B):
    return A * B - B * A

def anticommutator(A, B):
    return A * B + B * A

def derive_interaction():
    print("--- 1. Initializing Symbolic Dirac Algebra ---")
    # Define symbols
    m, c, hbar, kappa = sp.symbols('m c hbar kappa', real=True, positive=True)
    px, py, pz = sp.symbols('p_x p_y p_z', commutative=False) # Momentum operators
    
    # 4x4 Dirac Matrices (Standard Representation)
    I4 = sp.eye(4)
    Zero4 = sp.zeros(4)
    # Pauli matrices (2x2)
    sx = msigma(1)
    sy = msigma(2)
    sz = msigma(3)
    
    # Gamma Matrices (4x4)
    # gamma^0
    gamma0 = sp.Matrix(sp.BlockMatrix([[I4[:2,:2], Zero4[:2,:2]], [Zero4[:2,:2], -I4[:2,:2]]]))
    # gamma^i
    gamma1 = sp.Matrix(sp.BlockMatrix([[Zero4[:2,:2], sx], [-sx, Zero4[:2,:2]]]))
    gamma2 = sp.Matrix(sp.BlockMatrix([[Zero4[:2,:2], sy], [-sy, Zero4[:2,:2]]]))
    gamma3 = sp.Matrix(sp.BlockMatrix([[Zero4[:2,:2], sz], [-sz, Zero4[:2,:2]]]))
    
    gammas = [gamma0, gamma1, gamma2, gamma3]
    
    # Alpha and Beta
    beta = gamma0
    alpha1 = gamma0 * gamma1
    alpha2 = gamma0 * gamma2
    alpha3 = gamma0 * gamma3
    alphas = [alpha1, alpha2, alpha3]

    # Sigma_ab = (i/2) [gamma_a, gamma_b]
    # We create a function to get sigma_ab easily
    def get_sigma(a, b):
        return (sp.I / 2) * commutator(gammas[a], gammas[b])

    print("--- 2. Defining the Spin Connection Term ---")
    # Your result: Gamma_mu = (i/4) * kappa * (partial^a h_mu^b) * sigma_ab
    # Let's test a specific GW component to see the interaction.
    # Case: h_plus traveling in z, so h_11 = -h_22 = h(t).
    # We look at the SPATIAL connection Gamma_1 (mu=1).
    # Relevant term: partial^2 h_1^1  (a=2, b=1 => sigma_21)
    
    # Let's symbolically represent the gradient of h as a scalar 'dh'
    dh = sp.symbols('partial_h') 
    
    # Example: A term in Gamma_1 coming from d_y h_xx
    # sigma_21 is proportional to Sigma_z (Even)
    sigma_21 = get_sigma(2, 1) 
    Gamma_1_term = (sp.I / 4) * kappa * dh * sigma_21
    
    # The Hamiltonian Interaction term: H_int = -i * alpha_1 * Gamma_1
    H_int_odd = -sp.I * alpha1 * Gamma_1_term
    
    print("Checking Parity of the H_int term (alpha_1 * sigma_21)...")
    # Check if it commutes with beta
    check_even = commutator(H_int_odd, beta)
    if check_even == sp.zeros(4):
        print("-> It commutes with Beta. It is EVEN.")
    else:
        print("-> It anti-commutes. It is ODD.")
        
    print("\n--- 3. Performing the FW Step: [O, E] ---")
    # Let's calculate the Commutator [H_int_odd, E_kinetic]
    # O = H_int_odd (The interaction term we found above)
    # E = beta * m * c**2 (Mass term)
    
    O_term = H_int_odd
    E_term = beta * m * c**2
    
    # The FW correction is (1/2m) * beta * [O, O] ... wait, usually it's [O, E] in higher orders
    # Let's look at [O, E] specifically as you asked.
    
    comm_OE = commutator(O_term, E_term)
    
    # The result is likely a 4x4 matrix. We want to see the Top-Left 2x2 block (Electron).
    top_left = comm_OE[:2, :2]
    
    print("Result of [O, E] (Top-Left 2x2 Block):")
    # Simplification to make it readable
    sp.pprint(sp.simplify(top_left))
    
    print("\n--- 4. Interpreting the Result ---")
    # If the result is proportional to Identity -> Scalar Potential
    # If proportional to Pauli Matrices -> Spin Interaction
    
    # Let's project onto sigma_z
    coeff_sz = 0.5 * sp.trace(top_left * sz)
    print("Coefficient of Sigma_z:")
    sp.pprint(sp.simplify(coeff_sz))

if __name__ == "__main__":
    derive_interaction()