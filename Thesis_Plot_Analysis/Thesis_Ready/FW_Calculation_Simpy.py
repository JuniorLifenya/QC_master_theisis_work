import sympy as sp
from sympy import symbols, I, LeviCivita, KroneckerDelta, expand, Matrix, diag, zeros, eye, trace, diff, simplify, Rational, Sum, Product, IndexedBase, Idx, Function


# We Define Basic Symbols and Constants
m , kappa = symbols('m kappa', real=True,positive=True)
p = [symbols(f'p_{i}', commutative=False) for i in range(3)]
x = symbols('x0 x1 x2 x3', real=True)  # x0=t, x1,x2,x3=space
alpha = IndexedBase('alpha', commutative=False) #Alpha matrices
h_bar = symbols('h_bar', real=True, positive=True) #Planck's constant


#=====================Now we have to define the Dirac matrices (Alpha and Beta matrices)=======================
# But first we define pauli matrices to make it easier for us
# def alpha(i):
#     Sigma =[ sympy.Matrix([[0, 1], [1, 0]]), 
#              sympy.Matrix([[0, -I], [I, 0]]),
#              sympy.Matrix([[1, 0], [0, -1]])]
#     zeroes = sympy.zeros(2)
#     print (f"for visualization properly {([[zeroes]])} \n  {([Sigma[i]])} \n  {([Sigma[i]])} \n {([[zeroes]])}")
#     return sympy.Matrix([[zeroes, Sigma[i]], [Sigma[i], zeroes]])
# print(alpha(2)) # Example of alpha_2 matrix

beta = sp.diag(1,1,-1,-1) # Beta matrix # Alpha_0
Sigma_n = symbols('Sigma_n', commutative= False)
alpha_i,alpha_j = symbols('alpha_i alpha_j', commutative = False) # Alpha_i matrices (i=1,2,3)

def alpha_alpha_prodcut(expr):
    # This function defines the product of two alpha matrices using the anticommutation relations
    """
    Applies the rule: alpha^k * alpha^l = delta^kl + i * epsilon^kln * Sigma_n
    """
    return expr.replace(alpha_i*alpha_j, KroneckerDelta(i,j) + I * LeviCivita(i,j,n)*Sigma_n)

def apply_operator_product_rule(expr, op_p, field_h):
    """
    Applies: p(h) -> -I*grad(h) + h*p
    """
    # We represent the gradient of h as a new symbol 'dh'
    dh = symbols(f'd{field_h}') 
    
    # Replace the order: op_p * field_h -> field_h * op_p - I * dh
    return expr.replace(op_p * field_h, field_h * op_p - I * dh)

def push_p_right(expr):
    expr = sp.expand(expr)
    for k in range(3):
        p_k = p[k]
        f = sp.Wild('f')
        # Match p_k * any expression
        expr = expr.replace(p_k * f, lambda f: -I * diff(f, x[k+1]) + f * p_k)
    return expr

#============================Now we construct Odd and even operators from the Thesis ==========================
#Odd operators are those that anti-commute with beta, while even operators commute with beta.
# For exampple, the mass term m*beta is an even operator, while the kinetic term alpha_i * p_i is an odd operator.

hij = Matrix(4, 4, lambda i,j: Function(f'h_{{{i}{j}}}')(*x))
hio = Matrix(4, 4, lambda i,j: Function(f'h_{{{i}0}}')(*x))
hoj = Matrix(4, 4, lambda i,j: Function(f'h_{{0{j}}}')(*x))
h = sum(hij[i,i] for i in range(3))   # spatial trace

O = sum(alpha(i) * p[i] for i in range(3)) +  (kappa/2)* sum(hij*alpha(i)*p[j]-h*alpha(k)*p[k] for i,j,k in range (3)) # Odd operator (kinetic term)

# Even operator (epsilon) commutes with beta
# For gravitational interaction, epsilon might include h_00 terms
epsilon = sp.zeros(4) + (kappa/2)*hio*m*beta # Even operator (gravitational potential term)
#==============================================================================================================

#=========================== Now we have defined all the necessary Hamiltonians ===============================
# Now we can define the Dirac Hamiltonian in curved spacetime
H = beta * m*(1 + (kappa/2)* h) + O + epsilon
H_TT = beta * m + (kappa/2)* (hij*alpha(i)*p[j] for i,j in range(3)) + sum(alpha(i) * p[i] for i in range(3))
#=============================================================================================================

#========================== Now the resulting Hamiltonian in terms of commutation and operators ==============
H_eff = beta*(m + O**2/(2*m) - O**4/(8*m*m*m)) + epsilon - commutator(O,commutator(O,epsilon))/(8*m*m) # Effective Hamiltonian after FW transformation
H_eff_TT = beta*(m + O**2/(2*m) ) # In our case we Ignore the epsilon term and higher order terms(relativistic correction) in O for the TT gauge 
#==============================================================================================================

#========================== Now we can simplify the resulting Hamiltonian ==============================
m, kappa = symbols('m kappa', real=True, positive=True)

# Odd operator O = O1 + O2
O1 = sum(alpha(i) * p[i] for i in range(3))
term1 = sum(h[i,j] * alpha(i) * p[j] for i in range(3) for j in range(3))
O2 = (kappa/2) * (term1 - h * sum(alpha(k)*p[k] for k in range(3)))
O = O1 + O2

# Even operator epsilon = epsilon1 + epsilon2
epsilon1 = (kappa/2) * sum(h[i,0] * p[i] for i in range(3))
epsilon2 = (kappa/2) * h * beta * m
epsilon = epsilon1 + epsilon2

# Compute [O, epsilon]
comm_O_eps = ordered_comm(O, epsilon)
sp.pprint(comm_O_eps)
