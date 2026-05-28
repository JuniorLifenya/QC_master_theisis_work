"""
Plot 2 — Spherical Harmonic Density Comparison
================================================
Side-by-side |Y_1^{+1}|² and |Y_1^{-1}|² then the linear combinations
that diagonalize the secular matrix under GW perturbation (Δm = ±2 mixing).
The GW mixes |2p,m=+1⟩ ↔ |2p,m=-1⟩, forming symmetric/antisymmetric combos.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

os.makedirs("/home/claude/thesis_plots", exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10,
    'figure.dpi': 150,
})

BG   = '#F9F7F3'
CG   = '#CCCCCC'

# ── Spherical harmonic functions ──────────────────────────────────────────
# Y_1^{+1} = -sqrt(3/8π) sin θ e^{iφ}   → |Y_1^{+1}|² = (3/8π) sin²θ
# Y_1^{-1} = +sqrt(3/8π) sin θ e^{-iφ}  → |Y_1^{-1}|² = (3/8π) sin²θ
# Both have the SAME density — the difference is in the phase.
# GW mixes them into:
#   |+⟩ = (|m=+1⟩ + |m=-1⟩)/√2  ← symmetric — elongated along x
#   |-⟩ = (|m=+1⟩ - |m=-1⟩)/√2  ← antisymmetric — elongated along y

def sph_harm_density(l, m, theta, phi):
    """Return real |Y_l^m(θ,φ)|² on a (theta,phi) grid."""
    if l == 1 and abs(m) == 1:
        # |Y_1^{±1}|² = (3/8π) sin²θ  — phase independent
        return (3.0/(8.0*np.pi)) * np.sin(theta)**2
    elif l == 1 and m == 0:
        # |Y_1^0|² = (3/4π) cos²θ
        return (3.0/(4.0*np.pi)) * np.cos(theta)**2
    return np.zeros_like(theta)

def mixed_density(theta, phi, plus=True):
    """
    Density of the GW-eigenstate superposition
    |±⟩ = (|m=+1⟩ ± |m=-1⟩)/√2
    Full complex wave functions:
       Y_1^{+1} = -sqrt(3/8π) sinθ e^{+iφ}
       Y_1^{-1} = +sqrt(3/8π) sinθ e^{-iφ}
    Combination (m=+1) + (m=-1):  ∝ sinθ(e^{iφ} - e^{-iφ}) = 2i sinθ sinφ → ∝ sinθ sinφ
    Combination (m=+1) - (m=-1):  ∝ sinθ(e^{iφ} + e^{-iφ}) = 2 sinθ cosφ  → ∝ sinθ cosφ
    """
    norm = np.sqrt(3.0/(8.0*np.pi))
    if plus:
        # (|+1⟩ + |-1⟩)/√2: real part ~ sinθ sinφ
        psi = norm * np.sin(theta) * (-np.exp(1j*phi) + np.exp(-1j*phi)) / np.sqrt(2)
    else:
        # (|+1⟩ - |-1⟩)/√2: real part ~ sinθ cosφ
        psi = norm * np.sin(theta) * (-np.exp(1j*phi) - np.exp(-1j*phi)) / np.sqrt(2)
    return np.abs(psi)**2

def make_cartesian(r_density, theta, phi):
    """Convert spherical probability density to (x,y,z) colored by density."""
    x = r_density * np.sin(theta) * np.cos(phi)
    y = r_density * np.sin(theta) * np.sin(phi)
    z = r_density * np.cos(theta)
    return x, y, z

# ── Grid ──────────────────────────────────────────────────────────────────
n_theta, n_phi = 120, 240
theta_grid = np.linspace(0.001, np.pi - 0.001, n_theta)
phi_grid   = np.linspace(0, 2*np.pi, n_phi)
THETA, PHI = np.meshgrid(theta_grid, phi_grid, indexing='ij')

# Compute the four densities
d_p1   = sph_harm_density(1,  1, THETA, PHI)  # |Y_1^{+1}|²
d_m1   = sph_harm_density(1, -1, THETA, PHI)  # |Y_1^{-1}|²
d_plus = mixed_density(THETA, PHI, plus=True)   # |+1⟩+|-1⟩
d_mns  = mixed_density(THETA, PHI, plus=False)  # |+1⟩-|-1⟩

# ── Figure ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 7.5), facecolor=BG)
fig.suptitle(
    r"$|2p\rangle$ Subspace: Bare States vs GW-Eigenstate Linear Combinations"
    "\n"
    r"GW quadrupole mixes $|m=+1\rangle \leftrightarrow |m=-1\rangle$ via $\Delta m = \pm 2$",
    fontsize=13, fontweight='bold', y=1.01, color='#1a1a1a')

ELEV, AZIM = 28, 50
CMAP_BARE   = 'plasma'
CMAP_MIXED  = 'viridis'

panels = [
    (d_p1,   r"$|Y_1^{+1}|^2$",
     r"Bare state $|2p,\,m=+1\rangle$""\n"r"Density $\propto \sin^2\theta$ (axisymmetric)", CMAP_BARE,   '#5B1A8A'),
    (d_m1,   r"$|Y_1^{-1}|^2$",
     r"Bare state $|2p,\,m=-1\rangle$""\n"r"Identical density — difference is in phase $e^{\mp i\phi}$", CMAP_BARE, '#5B1A8A'),
    (d_mns,  r"$\propto\sin\theta\cos\phi$",
     r"GW eigenstate $\frac{|{+1}\rangle - |{-1}\rangle}{\sqrt{2}}$"
     "\n"r"Elongated along $x$ — lower energy eigenstate", CMAP_MIXED, '#175A3A'),
    (d_plus, r"$\propto\sin\theta\sin\phi$",
     r"GW eigenstate $\frac{|{+1}\rangle + |{-1}\rangle}{\sqrt{2}}$"
     "\n"r"Elongated along $y$ — upper energy eigenstate", CMAP_MIXED, '#1B3870'),
]

axes = []
for idx, (density, sym_label, title, cmap, tc) in enumerate(panels):
    ax = fig.add_subplot(1, 4, idx+1, projection='3d', facecolor=BG)
    axes.append(ax)

    # Scale the sphere by density
    R = density / density.max()
    x, y, z = make_cartesian(R, THETA, PHI)

    surf = ax.plot_surface(x, y, z, facecolors=plt.colormaps[cmap](density/density.max()),
                            alpha=0.92, linewidth=0, antialiased=True)

    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_zlim(-1.1, 1.1)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_axis_off()
    ax.set_title(title, fontsize=9, pad=8, color=tc, fontweight='bold')

    # Axis labels (small arrows)
    L = 1.3
    for (dx,dy,dz,lbl) in [(L,0,0,'x'),(0,L,0,'y'),(0,0,L,'z')]:
        ax.quiver(0,0,0,dx,dy,dz, color='#555', lw=0.8, arrow_length_ratio=0.15, alpha=0.5)
        ax.text(dx*1.05, dy*1.05, dz*1.05, lbl, fontsize=7, color='#666', ha='center')

    # Symmetry label below
    ax.text2D(0.5, -0.04, sym_label, transform=ax.transAxes,
              ha='center', fontsize=10, color=tc, math_fontfamily='stix')

# ── Separator arrow showing GW-induced mixing ─────────────────────────────
fig.text(0.49, 0.48, r"$H_{\rm GW},\ \Delta m=\pm 2$  $\longrightarrow$",
         ha='center', fontsize=13, color='#C47C0A', fontweight='bold')
fig.text(0.49, 0.42, "diagonalization\nof secular matrix",
         ha='center', fontsize=8.5, color='#C47C0A', style='italic')

# ── Bottom caption ────────────────────────────────────────────────────────
fig.text(0.5, -0.04,
    r"Left two: the bare states $|m=\pm1\rangle$ are degenerate (same density shape, different orbital phase). "
    "GW perturbation lifts this degeneracy by mixing them.\n"
    r"Right two: eigenstates of the secular matrix — $p_x$-like and $p_y$-like orbitals — "
    r"split by $\Delta E = 2|V_{\rm FNC}|$, the GW-induced energy splitting.",
    ha='center', fontsize=9, color='#444', style='italic',
    bbox=dict(boxstyle='round,pad=0.5', fc=BG, ec='#ccc', lw=0.8))

plt.tight_layout(rect=[0, 0.08, 1, 0.98])
plt.savefig("Thesis_Ready_Plots/plot2_spherical_harmonics.png",
            dpi=300, bbox_inches='tight', facecolor=BG)
print("Saved: plot2_spherical_harmonics.png")
plt.close()