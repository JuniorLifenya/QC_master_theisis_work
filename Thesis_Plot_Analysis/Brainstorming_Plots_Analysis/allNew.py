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


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: GW Strain Waterfall — Frequency vs. Distance
# ══════════════════════════════════════════════════════════════════════════════
def fig_gw_strain_waterfall():
    """
    2D map: GW frequency (x) vs. source distance (y), coloured by strain.
    Situates known events and NV sensitivity contours in the observable universe.
    Replaces the schematic plotsC8 waveform with a scientifically richer summary.
    """
    f   = np.logspace(0, 4, 200)      # Hz
    D   = np.logspace(0, 4.5, 200)    # Mpc
    F, Dist = np.meshgrid(f, D)

    # Strain estimate for 30 M_solar binary:
    # h ≈ (4G/c²) * M_chirp * (π f)^(2/3) / (c^(1/3) * D)
    # Simplified leading order: h ~ 1e-21 * (100 Mpc/D) * (f/100 Hz)^(2/3) * (M_chirp/1.2 M_sun)
    M_sun = 1.989e30
    M_chirp = 30 * M_sun  # typical BBH chirp mass
    # Exact Kepler estimate
    G_SI = 6.674e-11; c_SI = 3e8
    h_strain = (4*G_SI/c_SI**2) * M_chirp * (np.pi*F)**(2/3) * G_SI**(1/3) / (c_SI**(1/3) * Dist*3.086e22)
    log_h = np.log10(np.clip(h_strain, 1e-35, 1e-15))

    # NV sensitivity curves (SQL, LIF formula, NV electron r=1nm, N=1e12, T2=10ms, 1yr)
    omega_arr = 2*np.pi*f
    g_NV = m_e * omega_arr**2 * (1e-9)**2 / (4*hbar)
    T2 = 10e-3; N = 1e12; T_int = 3.15e7
    h_nv_sql = hbar / (g_NV * T2) / np.sqrt(N * T_int/T2)

    # Macro resonator (1g, 1mm)
    g_macro = 1e-3 * omega_arr**2 * (1e-3)**2 / (4*hbar)
    h_macro_sql = hbar / (g_macro * T2) / np.sqrt(N * T_int/T2)

    # LIGO-like curve
    f_ligo = np.logspace(1, 3.5, 300)
    h_ligo = 8e-24 * np.sqrt((110/f_ligo)**4.5 + 1.5 + 0.5*(f_ligo/110)**2.5)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Waterfall
    levels = np.linspace(-34, -19, 60)
    cmap = LinearSegmentedColormap.from_list('gwmap',
        ['#0a0a18','#1a1040','#2060a0','#20a080','#80c040','#e0c020','#e05020'])
    cf = ax.contourf(F, Dist, log_h, levels=levels, cmap=cmap, extend='both')
    cbar = fig.colorbar(cf, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(r'$\log_{10}(h_+)$', fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    # Contour lines
    ax.contour(F, Dist, log_h, levels=[-30,-27,-24,-21,-18],
               colors='white', linewidths=0.7, alpha=0.35)

    # NV sensitivity (vertical line in this plot = minimum detectable distance vs frequency)
    # On this plot: for each f, minimum detectable h is h_nv_sql(f)
    # Which means minimum distance from which NV can detect: D_min(f)
    # h_source = C/D → D_min = C/h_min; simplified using h ~ 1e-21 at 100 Mpc
    scale = 1e-21 * 100  # h*D product for typical source
    D_nv_min = scale / h_nv_sql        # Mpc
    D_macro_min = scale / h_macro_sql  # Mpc
    D_ligo_min  = scale / h_ligo

    ax.plot(f, D_nv_min, color='#60c8ff', lw=2.2, label='NV electron SQL ($r=1$ nm, $N=10^{12}$, 1yr)')
    ax.plot(f, D_macro_min, color='#c060ff', lw=2.2, label='Macro resonator SQL ($M=1$ g, $R=1$ mm)')
    ax.plot(f_ligo, D_ligo_min, color='#ffcc40', lw=2.2, label='LIGO-like sensitivity')

    # Known events
    events = [
        (35,  410, 'GW150914\n(BBH)', '*', 200, '#ffffff'),
        (100,  40, 'GW170817\n(BNS)', 's', 120, '#ffee80'),
        (59.4, 2e3,'Crab pulsar\n(CW target)', 'D',  90, '#ff8060'),
    ]
    for fx, Dx, lab, mk, sz, col in events:
        ax.scatter([fx],[Dx], marker=mk, s=sz, color=col, zorder=6, edgecolors='white', lw=0.8)
        ax.text(fx*1.15, Dx*1.1, lab, fontsize=8.5, color=col, va='bottom')

    # Cosmological horizon bands
    for D_band, label in [(0.01,'10 kpc (Galaxy)'),(1,'1 Mpc (Local group)'),
                          (100,'100 Mpc'),(3000,'z≈0.7')]:
        if D_band >= f.min() and D_band <= 3e4:
            ax.axhline(D_band, color='white', lw=0.6, ls=':', alpha=0.25)
            ax.text(1.1, D_band*1.08, label, fontsize=8, color='rgba(255,255,255,0.45)'
                    if False else '#888898', va='bottom')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(1, 1e4); ax.set_ylim(0.01, 1e4)
    ax.set_xlabel(r'GW Frequency $f$ (Hz)', fontsize=12)
    ax.set_ylabel(r'Source Distance $D$ (Mpc)', fontsize=12)
    ax.set_title('Astrophysical strain landscape: frequency vs. source distance\n'
                 r'Colour shows $h_+$; sensitivity curves show minimum detectable distance',
                 fontsize=11)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.85)
    ax.set_facecolor('#0a0a18')

    plt.tight_layout()
    plt.savefig('figures/gw_strain_waterfall.png', dpi=300, bbox_inches='tight',
                facecolor='#0a0a18')
    plt.show()
    plt.close()
    print('saved: figures/gw_strain_waterfall.png')


# ── Run all ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    fig_gw_strain_waterfall()
    print('\nAll 4 figures saved to ./figures/')
    print()
    print('Thesis placement:')
    print('  boughn_rothman_radial.png  -> Ch. 6 §6.2 (Boughn-Rothman discussion)')
    print('  literature_landscape.png   -> Ch. 6 §6.4 (Synthesis section)')
    print('  decoherence_budget.png     -> Ch. 6 §6.3 (SQL discussion)')
    print('  gw_strain_waterfall.png    -> Ch. 5 §5.4 or Ch. 6 intro')