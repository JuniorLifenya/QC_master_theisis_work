import qutip as qt
import numpy as np 
import matplotlib.pyplot as plt


#========= NV-center under Gravitational Wave Simulation using QuTiP ==========#
print("Setting first up the NV-center as a spin-1 system")

# Spin-1 operators as (3x3 matrices)
Sx = qt.jmat(1, 'x')
Sy = qt.jmat(1, 'y')
Sz = qt.jmat(1, 'z')

# Squared operators
Sx2 = Sx ** 2
Sy2 = Sy ** 2
Sz2 = Sz ** 2

# ================= Basis states and Hamiltonian Definitions ==================#

psi_plus = qt.basis(3, 2)   # |ms = +1> 
psi_0    = qt.basis(3, 1)   # |ms = 0>
psi_minus = qt.basis(3, 0)  # |ms = -1>

# Nv center Hamiltonian (Zero-field splitting + Zeeman term)

D = 2.87e9  # Hz (zero-field splitting ~ 2.87 GHz)
gamma_e = 28e9  # Hz/T (electron gyromagnetic ratio)

H_nv = D * Sz2  # Zero-field splitting term

print(" ✓ Setup done and Hamiltonian without magnetic field:\n", H_nv)

# ================ Gravitational Wave Interaction Hamiltonian =================#
print("Defining the interaction Hamiltonian from gravitational wave strain")

# Simple monochromatic GW waveform function
def h_plus(t, f_gw = 1000, h_max = 1e-20):
    """GW strain: h_plus(t) = h_max * sin(2π f_gw t)"""

    return h_max * np.sin(2 * np.pi * f_gw * t)

def h_cross(t):
    """No cross polarization in this simple example"""
    return 0.0

# Interaction Hamiltonian (MUST BE DERIVED FROM THEORY, MAIN PHYSICS)
# Based on quadrupole coupling: H_int = κ h_plus(t) (S_x² - S_y²)

kappa = 1e15  # Placeholder coupling constant (to be derived from FW transformation)
H_int_operator = kappa * (Sx2 - Sy2)  # Operator part of H_int

def H_int(t, args = None):
    """Time-dependent interaction Hamiltonian from GW strain"""
    h_p = h_plus(t)
    h_c = h_cross(t)
    return h_p * H_int_operator  # Only h_plus contributes in this simple example

print("✓ GW interaction setup complete")

# ==================== Time Evolution Simulation ====================#
print("Setting up time evolution simulation under GW influence")

# Total Hamiltonian 
H = [H_nv, [H_int, lambda t, args: 1.0]]  # QuTiP format for time-dependent Hamiltonian, the lambda  t is needed to match QuTiP's expected format
# Alternative: H = [H_nv, [H_int, 't']] # if H_int is defined as a function of t

# Initial state: ( We start in |0> )  
psi0 = psi_0

# Time vector (math GW timescale)
tlist = np.llinspace(0, 0.01, 1000)  # 10 ms total time, 1000 points

# We run the simulation (NO decoherence for simplicity)
result = qt.sesolve(H, psi0, tlist, [])
print("✓ Time evolution simulation complete")
