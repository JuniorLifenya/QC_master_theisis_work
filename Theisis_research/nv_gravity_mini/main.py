import numpy as np
import matplotlib.pyplot as plt
from src.hamiltonians import get_base_hamiltonian
from src.gravitational_waves import simple_gw_waveform # type: ignore
from src.evolution import simulate_nv_evolution
import qutip as qt

# 1. Simulation Parameters
times = np.linspace(0, 10e-9, 1000)  # 10 ns simulation
Bz = 0.01  # 10 mT magnetic field
f_gw = 1e9  # 1 GHz GW (unrealistic, but good for a clear signal)
h_max = 1e-20 # Strain amplitude

# 2. Define the time-dependent strain functions
def h_plus(t):
    return simple_gw_waveform(t, f_gw, h_max)
def h_cross(t):
    return 0.0  # No cross polarization for this simple example

# Pack arguments for QuTiP
args = {
    'Bz': Bz,
    'h_plus_func': h_plus,
    'h_cross_func': h_cross
}

# 3. Set initial state (e.g., the m_s=0 state)
psi0 = qt.basis(3, 1)  # Represents |0>

# 4. Run the simulation
result = simulate_nv_evolution(psi0, times, args)

# 5. Calculate and plot the population in the |0> state
p0 = np.zeros(len(times))
for i, state in enumerate(result.states):
    # The population is the absolute square of the projection
    p0[i] = np.abs(state[1][0][0])**2  # Gets the amplitude of the m_s=0 component

plt.figure(figsize=(10, 5))
plt.plot(times * 1e9, p0) # Plot time in ns
plt.xlabel('Time (ns)')
plt.ylabel('Population in |0> state')
plt.title('Effect of a Monochromatic Gravitational Wave on an NV-center')
plt.grid(True)
plt.savefig('outputs/figures/simple_simulation.png')
plt.show()