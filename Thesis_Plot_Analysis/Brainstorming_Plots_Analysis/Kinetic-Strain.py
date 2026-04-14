import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure the output directory exists
os.makedirs('figures', exist_ok=True)

# ─── FIG 11: KINETIC STRAIN VS ISOTROPIC MOMENTUM SPACE ───────────

px = np.linspace(-3.5, 3.5, 200)
PX, PY = np.meshgrid(px, px)

h = 0.5
phases = [0, np.pi/4, np.pi/2, 3*np.pi/4]

# We create 5 columns: 1 for the reference, 4 for the time phases
fig, axes = plt.subplots(2, 5, figsize=(22, 9))

for row, pol in enumerate(['plus', 'cross']):
    # 1. Plot the Standard Isotropic Reference (Column 0)
    ax_ref = axes[row, 0]
    V_iso = (PX**2 + PY**2)
    ax_ref.contourf(PX, PY, V_iso, levels=20, cmap='viridis')
    ax_ref.set_aspect('equal')
    ax_ref.set_title("Standard Isotropic", fontweight='bold')
    ax_ref.set_ylabel(r'$p_y/p_0$' if row == 0 else r'$p_y/p_0$')
    if row == 1: ax_ref.set_xlabel(r'$p_x/p_0$')

    # 2. Plot the Kinetic Strain Phases (Columns 1 to 4)
    for col, phi in enumerate(phases):
        # We use col + 1 to leave the first column for the reference
        ax = axes[row, col + 1]
        
        # Handle the 1.57 (pi/2) null-point by checking if cos is near zero
        phi_eff = phi + 1e-9 if np.isclose(np.cos(phi), 0) else phi
        
        # Calculate the gravito-magnetic strain potential
        V = h * np.cos(phi_eff) * (PX**2 - PY**2 if pol == 'plus' else PX * PY)
        
        # Symmetric color scale centering on zero
        lim = max(abs(V).max(), 1e-9)
        cf = ax.contourf(PX, PY, V, levels=20, cmap='RdBu_r', vmin=-lim, vmax=lim)
        ax.contour(PX, PY, V, levels=[0], colors='k', linewidths=0.8)
        
        ax.set_aspect('equal')
        ax.set_title(fr'${"h_+" if pol=="plus" else "h_\\times"}$, $\omega t={phi:.2f}$')
        if row == 1: ax.set_xlabel(r'$p_x/p_0$')

fig.suptitle(r'Kinetic strain $h_{ij}\hat{p}^i\hat{p}^j$ in Momentum Space' + \
             '\n' + r'(Comparison: Isotropic Background vs. Gravito-Magnetic Perturbations)', 
             fontsize=16, y=0.97)

plt.tight_layout()
plt.savefig('figures/fig11_kinetic_strain.png', bbox_inches='tight')
plt.show()

print("File saved: figures/fig11_kinetic_strain.png")