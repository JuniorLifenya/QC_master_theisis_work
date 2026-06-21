import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Ensure output directory
import os
os.makedirs("Thesis_Ready_Plots", exist_ok=True)

# ----------------------------------------------------------------------
# 1. Define the kinetic‑energy surface in momentum space (isotropic)
# ----------------------------------------------------------------------
def kinetic_energy_iso(px, py, m=1.0):
    """Standard isotropic kinetic energy (scaled by 2m for simplicity)."""
    return (px**2 + py**2) / (2*m)

# ----------------------------------------------------------------------
# 2. Add GW strain term (plus polarisation)
# ----------------------------------------------------------------------
def gw_strain_plus(px, py, h_plus=0.3):
    """Plus‑polarised strain term h_+ (p_x^2 - p_y^2)."""
    return h_plus * (px**2 - py**2)

# ----------------------------------------------------------------------
# 3. Combine to get deformed kinetic energy surface
# ----------------------------------------------------------------------
def total_energy(px, py, h_plus=0.3, m=1.0, kappa=1.0):
    """Kinetic energy + GW coupling."""
    iso = kinetic_energy_iso(px, py, m)
    strain = kappa * gw_strain_plus(px, py, h_plus)
    return iso + strain

# ----------------------------------------------------------------------
# 4. Create a circular momentum orbit (p = p0 (cos θ, sin θ))
# ----------------------------------------------------------------------
theta = np.linspace(0, 2*np.pi, 200)
p0 = 1.5
circle_px = p0 * np.cos(theta)
circle_py = p0 * np.sin(theta)
circle_E = total_energy(circle_px, circle_py, h_plus=0.3)

# ----------------------------------------------------------------------
# 5. Meshgrid for the surface
# ----------------------------------------------------------------------
p_range = np.linspace(-2.5, 2.5, 100)
PX, PY = np.meshgrid(p_range, p_range)
E_surface = total_energy(PX, PY, h_plus=0.3)

# ----------------------------------------------------------------------
# 6. 3D plot
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('#f5f5f5')

# Draw the deformed kinetic‑energy surface
surf = ax.plot_surface(PX, PY, E_surface, cmap=cm.viridis, alpha=0.7,
                       edgecolor='none', linewidth=0, antialiased=True)

# Overlay a faint wireframe to emphasise curvature
ax.plot_wireframe(PX, PY, E_surface, color='black', alpha=0.12,
                  rstride=5, cstride=5)

# Draw the circular momentum orbit on the surface
ax.plot(circle_px, circle_py, circle_E, color='crimson', lw=3,
        label='Momentum orbit (electron in circular motion)')

# Mark some points with arrows to show speed variation
for i in [0, 25, 50, 75, 100, 125, 150, 175]:
    px_i = circle_px[i]
    py_i = circle_py[i]
    E_i = circle_E[i]
    # Tangential velocity direction (in momentum plane)
    vx = -np.sin(theta[i])
    vy = np.cos(theta[i])
    ax.quiver(px_i, py_i, E_i, 0.15*vx, 0.15*vy, 0,
              color='darkred', arrow_length_ratio=0.2, lw=2)

# Labels
ax.set_xlabel(r'$p_x$', fontsize=12)
ax.set_ylabel(r'$p_y$', fontsize=12)
ax.set_zlabel(r'$E_{\rm kin}$', fontsize=12)
ax.set_title(
    'Electron kinetic energy landscape\n'
    r'GW plus polarisation ($h_+ = 0.3$) squashes the paraboloid',
    fontsize=14, fontweight='bold')

# Legend & colour bar
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15,
                    label=r'Kinetic energy $E(p_x,p_y)$')
ax.legend(loc='upper left')

# View angle
ax.view_init(elev=25, azim=-60)

plt.tight_layout()
plt.savefig("Thesis_Ready_Plots/electron_momentum_GW_landscape.png",
            dpi=300, bbox_inches='tight')
plt.show()
print("Saved: Thesis_Ready_Plots/electron_momentum_GW_landscape.png")