"""
lense_thirring_qutip_bulletproof.py
Final, stable QuTiP 5.x script for Lense-Thirring precession.
Explicitly maps colors to every data point to bypass the IndexError.
"""

import numpy as np
from qutip import sigmax, sigmay, sigmaz, basis, sesolve, Bloch, expect
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)

# 1. Physics Parameters
omega_L  = 2 * np.pi * 1.0  
omega_LT = 0.25 * omega_L   

H_mag = (omega_L / 2.0) * sigmaz()
H_lt  = (omega_LT / 2.0) * sigmax()
H_total = H_mag + H_lt

# 2. Evolution
psi0 = (basis(2, 0) + basis(2, 1)).unit()
t = np.linspace(0, 10, 400) # Reduced points slightly for stability

res_mag = sesolve(H_mag, psi0, t)
res_total = sesolve(H_total, psi0, t)

def get_coords(result):
    x = expect(sigmax(), result.states)
    y = expect(sigmay(), result.states)
    z = expect(sigmaz(), result.states)
    return [x, y, z]

coords_mag = get_coords(res_mag)
coords_total = get_coords(res_total)

# 3. Build the Bloch Sphere
b = Bloch()

# Colors
c_mag = '#2171B5'   # Blue
c_total = '#E31A1C' # Red

# THE ULTIMATE FIX: 
# Instead of 2 colors, we give it a list for every single point in the arrays.
# This prevents the internal 'indperm' logic from ever going out of range.
b.point_color = [c_mag] * len(t) + [c_total] * len(t)
b.vector_color = [c_mag, c_total]

# Add points as a single massive batch or individual sets
b.add_points(coords_mag, meth='l') 
b.add_points(coords_total, meth='l') 

# Add the final state vectors (arrows)
b.add_states(res_mag.states[-1])
b.add_states(res_total.states[-1])

# Final Styling
b.font_size = 14
b.zlabel = ['$|0\\rangle$', '$|1\\rangle$']
b.title = "Lense-Thirring Precession: Unitary Evolution"

# Render
b.show()
plt.savefig("figures/lense_thirring_qutip_final.pdf", bbox_inches='tight')
plt.show()

print("Saved: figures/lense_thirring_qutip_final.pdf")