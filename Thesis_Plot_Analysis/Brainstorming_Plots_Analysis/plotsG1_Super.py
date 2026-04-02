# Visualize the shearing of a global coordinate grid against a rigid local tetrad.
# This script calculates the 3D deformation of a lattice under $h_+$ and $h_\times$ strains
# to conceptually motivate the spin connection $\omega_\mu^{ab}$ in an NV center.
# Place in Ch2 2.2


import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure the output directory exists
os.makedirs("figures", exist_ok=True)

# ─── RIGOROUS GW POLARIZATION SIMULATOR (3D) ──────────────────────
def compute_3d_deformation(X, Y, Z, phase, h_plus=0.0, h_cross=0.0):
    """
    Computes the deformed coordinates of a 3D lattice in the TT gauge.
    A GW propagating along the z-axis stretches/squeezes the x-y plane.
    """
    # Linearized TT Gauge perturbations
    dX = 0.5 * (h_plus * X + h_cross * Y) * np.cos(phase)
    dY = 0.5 * (h_cross * X - h_plus * Y) * np.cos(phase)
    dZ = np.zeros_like(Z)  # Transverse wave, no deformation along z
    
    return X + dX, Y + dY, Z + dZ

def plot_3d_lattice(ax, x, y, z, color, alpha=0.5, lw=1.0, label=None):
    """Helper function to draw a 3D wireframe lattice from 3D meshgrid arrays."""
    # Plot lines along X, Y, and Z directions
    drawn_label = False
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            line_label = label if not drawn_label else None
            ax.plot(x[i, j, :], y[i, j, :], z[i, j, :], color=color, alpha=alpha, lw=lw, label=line_label)
            drawn_label = True
            
    for i in range(x.shape[0]):
        for k in range(x.shape[2]):
            ax.plot(x[i, :, k], y[i, :, k], z[i, :, k], color=color, alpha=alpha, lw=lw)
            
    for j in range(x.shape[1]):
        for k in range(x.shape[2]):
            ax.plot(x[:, j, k], y[:, j, k], z[:, j, k], color=color, alpha=alpha, lw=lw)

def plot_rigid_tetrad(ax):
    """Draws the rigid local inertial frame (the tetrad) at the origin."""
    length = 1.2
    # e_1 (x-axis)
    ax.quiver(0, 0, 0, length, 0, 0, color='k', arrow_length_ratio=0.15, lw=2.5)
    ax.text(length + 0.1, 0, 0, r'$e_{\hat{1}}$', fontsize=12, fontweight='bold')
    
    # e_2 (y-axis)
    ax.quiver(0, 0, 0, 0, length, 0, color='k', arrow_length_ratio=0.15, lw=2.5)
    ax.text(0, length + 0.1, 0, r'$e_{\hat{2}}$', fontsize=12, fontweight='bold')
    
    # e_3 (z-axis)
    ax.quiver(0, 0, 0, 0, 0, length, color='k', arrow_length_ratio=0.15, lw=2.5)
    ax.text(0, 0, length + 0.1, r'$e_{\hat{3}}$', fontsize=12, fontweight='bold')

# ─── SETUP GRID ───────────────────────────────────────────────────
# Create a 3x3x3 grid centered at the origin
grid_pts = np.linspace(-1, 1, 3)
X0, Y0, Z0 = np.meshgrid(grid_pts, grid_pts, grid_pts, indexing='ij')

fig = plt.figure(figsize=(14, 6.5))
fig.suptitle(r'Local Tetrad vs. Global Coordinate Shear ($TT$ Gauge)', fontsize=16)

# Wave parameters
h_amp = 0.5  # Exaggerated amplitude for visual clarity
phase = 0.0  # Maximum amplitude slice

# ─── LEFT: PLUS POLARIZATION ──────────────────────────────────────
ax1 = fig.add_subplot(121, projection='3d')

# Calculate deformed grid
Xp, Yp, Zp = compute_3d_deformation(X0, Y0, Z0, phase, h_plus=h_amp, h_cross=0.0)

# Plotting
plot_3d_lattice(ax1, X0, Y0, Z0, color='gray', alpha=0.2, lw=1.0, label=r'Global. grid $x^\mu$ (Unperturbed)')
plot_3d_lattice(ax1, Xp, Yp, Zp, color='#2171B5', alpha=0.8, lw=1.5, label=r'Sheared grid ($h_+$)')
plot_rigid_tetrad(ax1)

ax1.set_title(r"Plus ($+$) Polarization: $h_{xx} = -h_{yy}$", fontsize=14, pad=10)
ax1.set_xlim(-1.5, 1.5); ax1.set_ylim(-1.5, 1.5); ax1.set_zlim(-1.5, 1.5)
ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('z')
ax1.view_init(elev=20, azim=35)
# Avoid duplicate legend entries from the lattice looping
handles, labels = ax1.get_legend_handles_labels()
ax1.legend([handles[0], handles[-1]], [labels[0], labels[-1]], loc='upper left')

# ─── RIGHT: CROSS POLARIZATION ────────────────────────────────────
ax2 = fig.add_subplot(122, projection='3d')

# Calculate deformed grid
Xc, Yc, Zc = compute_3d_deformation(X0, Y0, Z0, phase, h_plus=0.0, h_cross=h_amp)

# Plotting
plot_3d_lattice(ax2, X0, Y0, Z0, color='gray', alpha=0.2, lw=1.0, label=r'Global. grid $x^\mu$ (Unperturbed)')
plot_3d_lattice(ax2, Xc, Yc, Zc, color='#E31A1C', alpha=0.8, lw=1.5, label=r'Sheared grid ($h_\times$)')
plot_rigid_tetrad(ax2)

ax2.set_title(r"Cross ($\times$) Polarization: $h_{xy} = h_{yx}$", fontsize=14, pad=10)
ax2.set_xlim(-1.5, 1.5); ax2.set_ylim(-1.5, 1.5); ax2.set_zlim(-1.5, 1.5)
ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('z')
ax2.view_init(elev=20, azim=35)
handles, labels = ax2.get_legend_handles_labels()
ax2.legend([handles[0], handles[-1]], [labels[0], labels[-1]], loc='upper left')

plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig("figures/GW_Tetrad_Deformation_Thesis.png", dpi=300, bbox_inches='tight')
plt.show()
print("Saved: figures/GW_Tetrad_Deformation_Thesis.png")


