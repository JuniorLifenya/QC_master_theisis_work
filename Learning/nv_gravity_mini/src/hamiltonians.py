import numpy as np
import qutip as qt
from qutip import Bloch, about, basis, sigmam,sigmax, sigmay, sigmaz, mesolve

#----------- Simple setup --------------------------------------------------#

# We will start with a simple model of an NV-center as a spin-1 system.
# In doing so we define H = [delta/2]sigmax 
# We define an additional collaps operator , describing the dissipation
# Of energy from the qubit to an external environment. C = [sqrt(g)]sigmaz

delta = 2* np.pi 

g = 0.25 

# Hamiltonian 
H = (delta/2) * sigmax()

# Collapse operators
c_ops = [np.sqrt(g) * sigmaz()]

# Initial state
psi0 = basis(2,0)  # Ground state

# Time vector
tlist = np.linspace(0, 5, 100)



#----------- Simple setup --------------------------------------------------#








#----------More advanced definitions---------------------------------------#
# Define NV-center spin-1 operators
Sx = qt.jmat(1, 'x')
Sy = qt.jmat(1, 'y')
Sz = qt.jmat(1, 'z')

Sx2 = Sx * Sx
Sy2 = Sy * Sy
Sz2 = Sz * Sz

#---------Base Hamiltonian ------------------------------------------------#
# Zero-field splitting constant (in Hz)
D = 2.87e9
gamma_e = 28e9  # Hz/T (electron gyromagnetic ratio)

def get_base_hamiltonian(Bz=0.0):
    """
    Returns the static NV Hamiltonian H_0.
    Args:
        Bz (float): Magnetic field in Tesla (for Zeeman splitting)
    """
    # Spin-1 operators (3x3 matrices as Qobj)
    Sz = qt.jmat(1, 'z')
    Sz2 = Sz ** 2

    # NV center constants
    D = 2.87e9  # Hz (zero-field splitting ~ 2.87 GHz)
    gamma_e = 28e9  # Hz/T (electron gyromagnetic ratio)

    # Build Hamiltonian
    H_0 = D * Sz2
    if Bz != 0.0:
        H_0 += gamma_e * Bz * Sz

    return H_0

#----------Define stuff----------------------------------------------------#

def get_interaction_hamiltonian(strain_plus, strain_cross):
    """
    Returns the interaction Hamiltonian H_int from gravitational strain.
    THIS IS THE KEY PHYSICS YOU NEED TO DERIVE.
    A first guess based on your supervisor's quadrupole comment:
    H_int ~ h_plus(t) * (Sx^2 - Sy^2) + h_cross(t) * (Sx*Sy + Sy*Sx)

    Args:
        strain_plus (float): h_plus strain component
        strain_cross (float): h_cross strain component
    """
    # This coupling constant 'kappa' is what you need to calculate/estimate from your theory!
    kappa = 1e10  # Placeholder! This is a crucial parameter from your derivation.

    H_int = kappa * (
        strain_plus * (Sx2 - Sy2) +
        strain_cross * (Sx * Sy + Sy * Sx)
    )
    return H_int

def get_total_hamiltonian(t, args):
    """
    Function of time 't' for QuTiP to use in time-dependent problems.
    'args' is a dictionary expected to contain:
        'Bz': magnetic field
        'h_plus_func': function h_plus(t)
        'h_cross_func': function h_cross(t)
    """
    h_p = args['h_plus_func'](t)
    h_c = args['h_cross_func'](t)
    
    H_0 = get_base_hamiltonian(args['Bz'])
    H_int = get_interaction_hamiltonian(h_p, h_c)
    
    return H_0 + H_int