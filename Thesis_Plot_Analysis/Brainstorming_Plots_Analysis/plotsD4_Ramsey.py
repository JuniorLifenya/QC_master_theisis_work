# Ramsey Interferometry Signal
"""
Plot 5: Ramsey fringes with and without a gravitational wave.
The GW induces a phase shift φ = ΔE T / ℏ, where ΔE ∝ h_+ cos(ω T).
For simplicity we plot the population in |1> after the second π/2 pulse.
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = np.linspace(0, 20, 500)   # free evolution time (arb. units)
h_plus = 0.5                 # GW amplitude (arbitrary)
omega = 2 * np.pi / 10        # GW frequency (1/10 of time unit)
delta_E = h_plus * np.cos(omega * T)   # energy shift

phi = delta_E * T             # accumulated phase
# Ramsey probability (standard formula for a π/2 - τ - π/2 sequence)
P = 0.5 * (1 + np.cos(phi))

plt.figure(figsize=(8, 5))
plt.plot(T, P, linewidth=2)
plt.xlabel('Free evolution time $T$')
plt.ylabel('Probability in $|1\\rangle$')
plt.title('Ramsey interferometry with gravitational wave')
plt.grid(alpha=0.3)
plt.savefig('figures/Ramsey_signal.png', dpi=300, bbox_inches='tight')
plt.show()