# Exercise: Create a "spin gym" with different challenges
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import qutip as qt

def spin_gym():
    challenges = {
        "warmup": "Verify [Sx, Sy] = iħSz for spin-1",
        "rabi_race": "Find fastest Rabi oscillations between |0> and |+1>",
        "decoherence_dodge": "Simulate T2 decoherence and find when coherence drops below 50%",
        "magnetic_maze": "Navigate state from |0> to |-1> using only B-field pulses",
        "gw_detector": "Detect a hidden GW signal in noisy population data"
    }
    return challenges

#================= Spin Gymnastics Challenges =================#
# Warmup Challenge #
Sx = qt.jmat(1, 'x') # We define the spin-1 operators
Sy = qt.jmat(1, 'y')
Sz = qt.jmat(1, 'z')


def spin_commutation(a, b):
    """Compute the commutator [a, b]"""
    return a * b - b * a
print("Verifying spin-1 commutation relation [Sx, Sy] = iħSz:")
print(spin_commutation(Sx, Sy))
print("Expected: i * Sz")
print("\n")

def print_matrix(mat, name):
    """Helper function to print matrices nicely"""
    print(f"{name} =\n{mat.full()}\n")
print_matrix(Sx, "Sx")
print_matrix(Sy, "Sy")
print_matrix(Sz, "Sz")

def verify_commutation():
    """Verify the commutation relations for spin-1 operators"""
    comm_Sx_Sy = spin_commutation(Sx, Sy)
    expected = 1j * Sz
    if np.allclose(comm_Sx_Sy.full(), expected.full()):
        print("✓ Commutation relation [Sx, Sy] = iħSz verified!")
    else:
        print("✗ Commutation relation [Sx, Sy] = iħSz NOT verified!")

verify_commutation()

#================= Level 2 Challenges =================#


