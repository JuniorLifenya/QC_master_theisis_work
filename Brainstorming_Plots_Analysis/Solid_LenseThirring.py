"""
lense_thirring_bloch_static.py

Visualizes Lense-Thirring frame-dragging on an NV-center Bloch sphere.
Pure Scipy/Matplotlib implementation. Zero QuTiP dependencies.
Generates a static high-quality PDF for the thesis.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os

# Ensure output directory exists
os.makedirs("figures", exist_ok=True)

# ----------------------------------------------------------------------
# 1. Physics Parameters
# ----------------------------------------------------------------------
omega_L = 2 * np.pi * 1.0   # Primary Larmor frequency (Z-axis)
omega_LT = 0.2 * omega_L    # Lense-Thirring frame-dragging (X-axis)

# Initial State: |+> state (pointing along X-axis)
S0 = [1.0, 0.0, 0.0]
t = np.linspace(0, 5, 500)  # Smooth time array

# ----------------------------------------------------------------------
# 2. Bloch Equation Solvers (dS/dt = Omega x S)
# ----------------------------------------------------------------------
def bloch_larmor(S, t):
    """Pure magnetic precession strictly around Z-axis."""
    Omega = np.array([0, 0, omega_L])
    return np.cross(Omega, S)

def bloch_total(S, t):
    """Magnetic (Z) + Gravitomagnetic Lense-Thirring (X) precession."""
    Omega = np.array([omega_LT, 0, omega_L])
    return np.cross(Omega, S)

# Solve the ODEs
sol_larmor = odeint(bloch_larmor, S0, t)
sol_total = odeint(bloch_total, S0, t)

# ----------------------------------------------------------------------
# 3. 3D Plotting Setup
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Draw a subtle wireframe sphere to represent the Bloch Sphere
u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
x_sphere = np.cos(u)*np.sin(v)
y_sphere = np.sin(u)*np.sin(v)
z_sphere = np.cos(v)
ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.1, linewidth=0.5)

# Plot the Precession Trails
ax.plot(sol_larmor[:,0], sol_larmor[:,1], sol_larmor[:,2], 
        color='#2171B5', lw=2.5, label='Pure Larmor Precession (Flat)', zorder=3)

ax.plot(sol_total[:,0], sol_total[:,1], sol_total[:,2], 
        color='#E31A1C', lw=3.0, label='Lense-Thirring Precession (Wobble)', zorder=4)

# Draw Final State Vectors (Arrows)
# Pure Larmor Final
ax.quiver(0, 0, 0, sol_larmor[-1,0], sol_larmor[-1,1], sol_larmor[-1,2], 
          color='#2171B5', arrow_length_ratio=0.15, lw=2)
# Total Final
ax.quiver(0, 0, 0, sol_total[-1,0], sol_total[-1,1], sol_total[-1,2], 
          color='#E31A1C', arrow_length_ratio=0.15, lw=2.5)

# Draw XYZ Axes for reference
ax.quiver(0,0,0, 1.2,0,0, color='black', alpha=0.5, arrow_length_ratio=0.1)
ax.quiver(0,0,0, 0,1.2,0, color='black', alpha=0.5, arrow_length_ratio=0.1)
ax.quiver(0,0,0, 0,0,1.2, color='black', alpha=0.5, arrow_length_ratio=0.1)
ax.text(1.3, 0, 0, 'x', fontsize=12)
ax.text(0, 1.3, 0, 'y', fontsize=12)
ax.text(0, 0, 1.3, 'z ($B_0$)', fontsize=12)

# Formatting
ax.set_title("Lense-Thirring Frame-Dragging on the Bloch Sphere", fontsize=16, pad=20)
ax.view_init(elev=20, azim=45)
ax.set_axis_off()
ax.legend(loc='lower center', fontsize=12, bbox_to_anchor=(0.5, 0.05))

# ----------------------------------------------------------------------
# 4. Save and Show
# ----------------------------------------------------------------------
pdf_path = "figures/lense_thirring_bloch_static.pdf"
plt.tight_layout()
plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
print(f"Saved highly-optimized static plot to: {pdf_path}")

plt.show()
