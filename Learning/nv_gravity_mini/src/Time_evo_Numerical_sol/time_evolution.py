# src/evolution.py
import qutip as qt
from ..Hamiltonians.hamiltonians import get_base_hamiltonian

def simulate_nv_evolution(initial_state, times, args):
    Bz = args['Bz']
    h_plus_func = args['h_plus_func']
    h_cross_func = args['h_cross_func']

    # Base Hamiltonian (static Zeeman term, for example)
    H0 = get_base_hamiltonian(Bz)  # must return a Qobj

    # Spin-1 operators
    Sx = qt.jmat(1, 'x')   # 3x3 operator
    Sy = qt.jmat(1, 'y')

    # Time-dependent Hamiltonian
    H_td = [H0, [Sx, h_plus_func], [Sy, h_cross_func]]

    # Solve Schrödinger equation
    result = qt.sesolve(H_td, initial_state, times, args=args)

    print(Sx)
    return result
 