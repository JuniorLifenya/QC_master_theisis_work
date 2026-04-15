

"""
new_thesis_figures.py
=====================
Daniel Junior Lifenya Fondo — UiB 2026
"Quantum Effects of Gravitational Waves"

Generates 4 new publication-quality figures:
  1. fig_hydrogen_radial_overlap   -> boughn_rothman_radial.png  (Ch. 6 §6.2)
  2. fig_literature_landscape      -> literature_landscape.png   (Ch. 6 §6.4)
  3. fig_decoherence_budget        -> decoherence_budget.png     (Ch. 6 §6.3)
  4. fig_gw_strain_waterfall       -> gw_strain_waterfall.png    (Ch. 5 §5.4)

Run: python new_thesis_figures.py
All figures saved to ./figures/
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import os

os.makedirs('figures', exist_ok=True)

# Physical constants (SI)
hbar  = 1.054571817e-34
m_e   = 9.10938356e-31
c     = 2.99792458e8
G     = 6.67430e-11
eV    = 1.60217663e-19
alpha = 1/137.035999
a_0   = hbar/(m_e*alpha*c)
l_Pl  = np.sqrt(hbar*G/c**3)

# ─── Shared style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':'serif',
    'font.serif':['Palatino','Georgia','Times New Roman','DejaVu Serif'],
    'font.size':11,
    'axes.labelsize':12,
    'axes.titlesize':11,
    'legend.fontsize':9.5,
    'axes.spines.top':False,
    'axes.spines.right':False,
    'axes.grid':True,
    'grid.alpha':0.22,
    'grid.linewidth':0.5,
    'figure.dpi':150,
    'lines.linewidth':2.0,
    'xtick.direction':'in',
    'ytick.direction':'in',
})


def fig_hydrogen_radial_overlap():
    """
    Visualises why the 1s→3d transition dominates graviton absorption
    (Boughn & Rothman 2006). Shows radial probability densities and the
    quadrupole overlap integrand.
    Physics: the quadrupole matrix element is ∝ ∫ R_3d(r) r² R_1s(r) r² dr
    """
    r = np.linspace(0.001, 28, 2000)   # r in units of a_0

    # Exact hydrogenic radial wavefunctions (not normalised for display)
    R_1s = 2 * np.exp(-r)
    R_3d = (2/(81*np.sqrt(30))) * r**2 * np.exp(-r/3)

    # Radial probability densities
    P_1s = R_1s**2 * r**2
    P_3d = R_3d**2 * r**2

    # Quadrupole integrand: ∫ R_3d · r² · R_1s · r² dr  (proportional to matrix element)
    # The r⁴ weighting strongly suppresses short-range and selects the mid-range overlap
    integrand_raw = R_3d * r**2 * R_1s * r**2
    integrand = integrand_raw / np.max(np.abs(integrand_raw))

    # Normalise densities for display
    P_1s /= np.max(P_1s)
    P_3d /= np.max(P_3d)

    # Also compute 2p for comparison (dipole-forbidden by GW)
    R_2p = (1/(2*np.sqrt(6))) * r * np.exp(-r/2)
    P_2p = R_2p**2 * r**2 / np.max(R_2p**2 * r**2)

    fig, (ax_main, ax_int) = plt.subplots(2, 1, figsize=(9, 7.5),
                                           gridspec_kw={'height_ratios':[1.5,1],'hspace':0.35})

    # Upper panel: wavefunctions
    ax = ax_main
    ax.plot(r, P_1s, color='#3a7bd5', lw=2.2, label=r'$|R_{1s}|^2 r^2$ (initial state)')
    ax.plot(r, P_3d, color='#e05a2b', lw=2.2, label=r'$|R_{3d}|^2 r^2$ (final state, GW-allowed)')
    ax.plot(r, P_2p, color='#888898', lw=1.5, ls='--', alpha=0.6, label=r'$|R_{2p}|^2 r^2$ (EM dipole — not driven by GW)')

    ax.fill_between(r, 0, integrand, where=integrand>0.02,
                    color='#e05a2b', alpha=0.12, label='Quadrupole overlap integrand')

    ax.axvline(3**2*1, color='#e05a2b', lw=0.9, ls=':', alpha=0.6)
    ax.text(10.5, 0.78, r'$\langle r\rangle_{3d} \approx 10.5\,a_0$', fontsize=9, color='#e05a2b')
    ax.axvline(1.5, color='#3a7bd5', lw=0.9, ls=':', alpha=0.6)
    ax.text(1.6, 0.88, r'$\langle r\rangle_{1s}=1.5\,a_0$', fontsize=9, color='#3a7bd5')

    ax.set_xlabel(r'Radius $r\;(a_0)$')
    ax.set_ylabel('Normalised radial probability density')
    ax.set_title(r'Boughn–Rothman graviton absorption: $1s \to 3d$ radial overlap' '\n'
                 r'Selection rules: $\Delta l = \pm2$, $\Delta m = \pm2$ (quadrupole)')
    ax.set_xlim(0, 28); ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', framealpha=0.9)

    # Lower panel: integrand and cumulative
    ax = ax_int
    ax.fill_between(r, 0, integrand, where=integrand>0, color='#e05a2b', alpha=0.3)
    ax.plot(r, integrand, color='#e05a2b', lw=2.2, label=r'Overlap integrand $r^4 R_{3d}(r)R_{1s}(r)$')

    # Cumulative integral (shows where the matrix element is built up)
    cumul = np.cumsum(integrand) * (r[1]-r[0])
    cumul /= cumul[-1]
    ax.plot(r, cumul, color='#5a9bd5', lw=1.8, ls='--', label='Cumulative (normalised)')
    ax.axhline(0.5, color='gray', lw=0.8, ls=':', alpha=0.6)
    ax.text(0.3, 0.55, '50% accumulated', fontsize=8.5, color='gray')

    ax.set_xlabel(r'Radius $r\;(a_0)$')
    ax.set_ylabel('Quadrupole overlap')
    ax.set_title(r'Transition matrix element $\propto \int R_{3d}(r)\,r^4\,R_{1s}(r)\,dr$   '
                 r'$\rightarrow$ cross section $\sigma_{\rm abs} = 0.31\,\ell_{\rm Pl}^2$')
    ax.set_xlim(0, 28); ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper right', framealpha=0.9)

    plt.savefig('figures/boughn_rothman_radial.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print('saved: figures/boughn_rothman_radial.png')
if __name__ == '__main__':
    print('Generating new thesis figures...')
    fig_hydrogen_radial_overlap()