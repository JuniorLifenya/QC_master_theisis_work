# Effective Hamiltonian Term Magnitude Comparison

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ─── physical constants (SI) ──────────────────────────────────
hbar  = 1.055e-34    # J·s
c     = 3.0e8        # m/s
G     = 6.674e-11    # m^3 kg^-1 s^-2
m_e   = 9.109e-31    # kg
a_0   = 5.292e-11    # m  (Bohr radius)
alpha = 1/137.0      # fine structure constant

# ─── GW parameters ────────────────────────────────────────────
h_strain = 1e-21     # dimensionless strain (LIGO event)
f_gw     = 100.0     # Hz
omega_gw = 2*np.pi*f_gw
k_gw     = omega_gw / c  # 1/m

# ─── characteristic electron momentum in hydrogen ─────────────
p_char = alpha * m_e * c      # kg m/s
E_kin  = p_char**2 / (2*m_e)  # J

# ─── gravitational coupling ───────────────────────────────────
kappa = np.sqrt(32 * np.pi * G) / c**2  # m/kg  (natural units conversion)

# Convert kappa to dimensionless by noting
# kappa * m_e * c^2 / hbar = coupling rate scale
# We compute all terms in units of E_kin for comparison

def term_magnitude(label, expr_J):
    """Return magnitude in units of electron kinetic energy."""
    return abs(expr_J) / E_kin

h = h_strain   # dimensionless
k = k_gw       # 1/m in SI — but we compare as dimensionless ratios

# ─── compute each term's energy scale (in Joules) ─────────────
# Using: kappa^2 = 32 pi G / c^4 in SI
# E_grav = (kappa / (2m_e c^2 / hbar^2)) * h * p^2 / m_e
# More directly: Delta E = (sqrt(32piG)/c^2) * h * p^2 / (2 m_e)

kappa_SI = np.sqrt(32 * np.pi * G) / c**2  # s/(kg m)

terms = {
    "Rest mass\n$\\beta m(1+\\kappa h/2)$":
        kappa_SI * h * m_e * c**2 / 2,

    "Gravito-el. potential\n$\\frac{\\kappa}{2}h_{i0}\\hat{p}^i$":
        kappa_SI * h * p_char * c / 2,   # rough scale with p_char

    "Free kinetic\n$\\frac{\\hat{p}^2}{2m}$":
        E_kin,

    "Rel. correction\n$-\\frac{\\hat{p}^4}{8m^3}$":
        p_char**4 / (8 * m_e**3 * c**4),

    "Isotropic kinetic\n$-\\frac{3\\kappa h}{4m}\\hat{p}^2$":
        3 * kappa_SI * h * p_char**2 / (4 * m_e),

    "Kinetic strain\n$\\frac{\\kappa}{2m}h_{ij}\\hat{p}^i\\hat{p}^j$":
        kappa_SI * h * p_char**2 / (2 * m_e),

    "Gravito-Zeeman\n$\\frac{\\kappa}{4m}\\vec{\\Sigma}\\cdot\\hat{B}_g$":
        kappa_SI * h * k_gw * p_char / (4 * m_e),  # Bg ~ k*h*p

    "Momentum-grad.\n$\\frac{\\kappa}{2m}i(\\nabla h)\\cdot\\hat{p}$":
        kappa_SI * h * k_gw * p_char / (2 * m_e),

    "Grav. spin-orbit\n$\\frac{3\\kappa}{8m}\\vec{\\Sigma}\\cdot[(\\nabla h)\\times\\hat{p}]$":
        3 * kappa_SI * h * k_gw * p_char / (8 * m_e),

    "Darwin (trace)\n$\\frac{\\kappa}{16m}\\nabla^2 h$":
        kappa_SI * h * k_gw**2 / (16 * m_e),  # Darwin is scalar — scale by Bohr

    "Thomas precession\n$\\frac{\\kappa}{8m^2}\\vec{\\Sigma}\\cdot[\\hat{p}\\times\\hat{E}_g]$":
        kappa_SI * h * k_gw * p_char**2 / (8 * m_e**2 * c),
}

labels = list(terms.keys())
values = np.array([abs(v) for v in terms.values()])

# normalise to free kinetic energy
norm = E_kin
values_norm = values / norm

# ─── plot ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

colors = plt.cm.RdYlGn_r(np.linspace(0.05, 0.95, len(labels)))
bars = ax.barh(range(len(labels)), np.log10(values_norm),
               color=colors, edgecolor='k', linewidth=0.5)

ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=8.5)
ax.set_xlabel(r"$\log_{10}(\Delta E\,/\,E_{\rm kin})$", fontsize=11)
ax.set_title(
    f"Effective Hamiltonian term magnitudes\n"
    f"$h = 10^{{-21}}$, $f_{{\\rm GW}} = 100$ Hz, hydrogen $1s$ state",
    fontsize=11)
ax.axvline(0, color='navy', lw=1.2, ls='--', label='$= E_{\\rm kin}$')
ax.axvline(-30, color='crimson', lw=1.0, ls=':', alpha=0.7,
           label='$10^{-30}\\,E_{\\rm kin}$ (detection threshold)')
ax.legend(fontsize=9, loc='lower right')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3, lw=0.5)

plt.tight_layout()
plt.savefig("figures/hamiltonian_term_magnitudes.png",
            bbox_inches="tight", dpi=300)
plt.show()
print("Saved: hamiltonian_term_magnitudes.pdf")