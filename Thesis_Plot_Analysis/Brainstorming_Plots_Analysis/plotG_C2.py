"""
lense_thirring_bloch.py

Visualizes Lense-Thirring frame-dragging on an NV-center Bloch sphere.
Compares:
1. Pure Larmor precession (Magnetic field only) - Blue
2. Lense-Thirring 'Wobble' (Magnetic + Gravitomagnetic coupling) - Red
"""

import numpy as np
from qutip import Bloch, basis, sesolve, sigmax, sigmay, sigmaz, expect
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# Ensure output directory exists
os.makedirs("figures", exist_ok=True)

# ----------------------------------------------------------------------
# 1. Setup Hamiltonians
# ----------------------------------------------------------------------
omega_L = 2 * np.pi * 1.0   # Larmor frequency (Z-axis)
omega_LT = 0.2 * omega_L    # Exaggerated Lense-Thirring (X-axis)

H_total = (omega_L / 2.0) * sigmaz() + (omega_LT / 2.0) * sigmax()
H_larmor = (omega_L / 2.0) * sigmaz()

# ----------------------------------------------------------------------
# 2. Initial State and Solver
# ----------------------------------------------------------------------
psi0 = (basis(2, 0) + basis(2, 1)).unit()  # Spin along X-axis
tlist = np.linspace(0, 5, 100) 

result_total = sesolve(H_total, psi0, tlist)
result_larmor = sesolve(H_larmor, psi0, tlist)

# ----------------------------------------------------------------------
# 3. PRE-CALCULATE COORDINATES (The Fix!)
# Extract expectation values to get pure (x,y,z) arrays for the trails
# ----------------------------------------------------------------------
x_L = expect(sigmax(), result_larmor.states)
y_L = expect(sigmay(), result_larmor.states)
z_L = expect(sigmaz(), result_larmor.states)

x_T = expect(sigmax(), result_total.states)
y_T = expect(sigmay(), result_total.states)
z_T = expect(sigmaz(), result_total.states)

# ----------------------------------------------------------------------
# 4. Animation Setup
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(8, 8))
b = Bloch(fig=fig)
b.view = [45, 20]

def update(frame):
    b.clear()
    
    # 1. Add current position as Vectors
    b.add_states(result_larmor.states[frame], kind='vector')
    b.add_states(result_total.states[frame], kind='vector')
    
    # 2. Add historical trails as Raw Points (Grouping them into a single dataset)
    if frame > 0:
        # Grouped arrays for Larmor
        pnts_L = [x_L[:frame], y_L[:frame], z_L[:frame]]
        b.add_points(pnts_L)
        
        # Grouped arrays for Total (LT)
        pnts_T = [x_T[:frame], y_T[:frame], z_T[:frame]]
        b.add_points(pnts_T)
        
        # Because we added exactly 2 datasets, we only need 2 colors!
        b.point_color = ['#2171B5', '#E31A1C']
        b.point_marker = ['o', 'o']
        b.point_size = [8, 8]
    else:
        b.point_color = []

    # 3. Vector styling
    b.vector_color = ['#2171B5', '#E31A1C']
    
    b.render()
    return fig

# ----------------------------------------------------------------------
# 5. Create and Save Animation
# ----------------------------------------------------------------------
print("Generating animation frames...")
ani = FuncAnimation(fig, update, frames=len(tlist), blit=False)

gif_path = 'figures/lense_thirring_animation.gif'
ani.save(gif_path, fps=20)
print(f"Animation successfully saved to {gif_path}")

plt.show()