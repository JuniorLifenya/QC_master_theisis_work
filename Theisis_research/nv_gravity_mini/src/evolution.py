import qutip as qt
from .hamiltonians import get_total_hamiltonian

def simulate_nv_evolution(initial_state, times, args):
    """
    Runs the time evolution simulation.
    """
    # Create the time-dependent Hamiltonian list for QuTiP
    # The string 't' tells QuTiP which variable is time
    H_td = [[get_total_hamiltonian, 't']]
    
    # Run the simulation using QuTiP's solver
    result = qt.sesolve(H_td, initial_state, times, args=args)
    return result