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
# FIGURE 1: Hydrogen Radial Overlap for 1s → 3d Graviton Absorption
# ══════════════════════════════════════════════════════════════════════════════
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
    plt.close()
    print('saved: figures/boughn_rothman_radial.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Literature Landscape Positioning
# ══════════════════════════════════════════════════════════════════════════════
def fig_literature_landscape():
    """
    Places this thesis on a 2D map of spin-gravity coupling derivations.
    X: completeness of FW expansion (how many orders in 1/m)
    Y: resolution of spin-dependent interaction terms
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    papers = [
        # (x, y, label, color, marker, size, year)
        (1.5, 0.5,
         "DeWitt (1957)\nFirst-order minimal coupling",
         '#888888', 'o', 80, 1957),
        (2.5, 1.5,
         "Boughn & Rothman (2006)\nClassical quadrupole\n(cross section)",
         '#607090', 's', 110, 2006),
        (3.5, 3.5,
         "Obukhov (2001)\nExact FW in static\nmetrics",
         '#7060a0', 'D', 110, 2001),
        (5.5, 5.0,
         "Tobar et al. (2024)\nSQL detection protocol\n($h_{\mu\\nu}T^{\\mu\\nu}$)",
         '#3a7bd5', 'p', 140, 2024),
        (6.5, 7.0,
         "Oliveira, Guilherme\n& Shapiro (2024)\nPauli eq. in GW,\n§V agrees with ours",
         '#2a9d6a', '^', 130, 2024),
        (9.0, 9.0,
         "This thesis (2026)\nFull $\\mathcal{O}(1/m^3)$ FW\nAll 9 terms, selection rules\nGravitational Thomas",
         '#e05a2b', '*', 320, 2026),
    ]

    # Background gradient (shading regions)
    x_bg = np.linspace(0,10,200); y_bg = np.linspace(0,10,200)
    Xb, Yb = np.meshgrid(x_bg, y_bg)
    Z = (Xb + Yb)/2
    ax.contourf(Xb, Yb, Z, levels=20,
                cmap=LinearSegmentedColormap.from_list('bg',['#0c0d10','#1a2030','#1a3020']),
                alpha=0.6, zorder=0)

    # Connection lines from chronological progression
    xs = [p[0] for p in papers[:-1]]
    ys = [p[1] for p in papers[:-1]]
    ax.plot(xs, ys, color='rgba(255,255,255,0.1)' if False else '#333344',
            lw=1, ls='--', alpha=0.4, zorder=1)

    for x,y,label,color,marker,size,year in papers:
        is_thesis = 'thesis' in label.lower()
        ax.scatter(x, y, s=size, c=color, marker=marker, zorder=5,
                   edgecolors='white' if is_thesis else 'none',
                   linewidths=1.5 if is_thesis else 0)
        # Label offset
        ox, oy = 0.2, 0.25
        if 'DeWitt' in label: oy=-0.8
        if 'Tobar' in label: ox=-2.5; oy=0.3
        ax.text(x+ox, y+oy, label, fontsize=9 if not is_thesis else 9.5,
                color=color if not is_thesis else '#f0c870',
                fontweight='bold' if is_thesis else 'normal',
                va='bottom', linespacing=1.4,
                bbox=dict(facecolor='rgba(10,10,20,0.7)' if False else '#0c0d18',
                          alpha=0.78, edgecolor=color, boxstyle='round,pad=0.3',
                          linewidth=0.8) if is_thesis else None)
        if not is_thesis:
            ax.text(x+ox, y+oy, label, fontsize=9, color=color,
                    va='bottom', linespacing=1.4,
                    bbox=dict(facecolor='#0c0d18', alpha=0.65,
                              edgecolor='none', boxstyle='round,pad=0.2'))

    # Axis labels and regions
    ax.axhline(5, color='white', lw=0.6, ls=':', alpha=0.18)
    ax.axvline(5, color='white', lw=0.6, ls=':', alpha=0.18)
    ax.text(1.0, 9.3, 'Spin-rich\nbut incomplete', fontsize=8.5, color='#666688', style='italic')
    ax.text(7.5, 1.0, 'Complete but\nno spin', fontsize=8.5, color='#666688', style='italic')
    ax.text(7.5, 8.5, 'Complete +\nSpin-dependent', fontsize=8.5, color='#8a9a70', style='italic')

    ax.set_facecolor('#0c0d10')
    ax.spines['bottom'].set_color('#444450')
    ax.spines['left'].set_color('#444450')
    ax.grid(True, color='#222230', alpha=0.5)
    ax.set_xlim(0, 10.5); ax.set_ylim(0, 10.5)
    ax.set_xlabel(r'Completeness of perturbative FW expansion ($\mathcal{O}(1/m^n)$)', labelpad=10)
    ax.set_ylabel('Resolution of spin-dependent interaction terms', labelpad=10)
    ax.set_title('Landscape of spin–gravity coupling derivations\n'
                 r'Positioning relative to existing literature on $\hat{H}_{\rm eff}$ in GW backgrounds',
                 pad=12)

    ax.tick_params(colors='#888898')
    ax.yaxis.label.set_color('#aaa8b8')
    ax.xaxis.label.set_color('#aaa8b8')

    plt.savefig('figures/literature_landscape.png', dpi=300, bbox_inches='tight',
                facecolor='#0c0d10')
    plt.close()
    print('saved: figures/literature_landscape.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Decoherence Budget vs. GW Phase Target
# ══════════════════════════════════════════════════════════════════════════════
def fig_decoherence_budget():
    """
    Shows the decoherence T₂ hierarchy for NV centers against the required
    integration time to detect a GW phase signal.
    The gap is the central experimental challenge of the thesis.
    """
    # NV center coherence times (seconds) — state-of-the-art literature values
    sources = [
        # (T2 in seconds, name, source, color)
        (3e-6,  r'$^{13}$C nuclear spin bath',    'Typ. NV in nat. diamond', '#e05a2b'),
        (1e-5,  r'Isotopically purified $^{12}$C', 'Balasubramanian+ 2009',  '#e08a2b'),
        (1e-3,  r'$T_2$ echo (dynamical decoup.)', 'Bar-Gill+ 2013',         '#e0c03a'),
        (1e-2,  r'$T_2$ (DD, 10ms record)',        'Herbschleb+ 2019',       '#6ec83a'),
        (0.1,   r'Nuclear spin $T_2$ (NV + $^{15}$N)', 'Maurer+ 2012',       '#3ab0e0'),
        (1e3,   r'Required: $\phi_{GW}\sim1$ rad', 'This thesis eq. (5.X)',  '#9060e0'),
    ]

    # Required coherence: φ = g*h/(hbar*omega)*T ~ 1 rad → T ~ hbar*omega/(g*h)
    # For NV electron, r~1nm, f=100Hz, h=1e-21
    omega_GW = 2*np.pi*100.0
    r_NV = 1e-9
    g_NV = m_e * omega_GW**2 * r_NV**2 / (4*hbar)  # rad/s per unit h
    T_required = hbar * omega_GW / (g_NV * 1e-21 * omega_GW)  # to get phi~1
    # Actually T for phi = 1: phi = g*h*T/(hbar) → T = hbar/(g*h)
    T_req_correct = hbar / (g_NV * 1e-21)
    print(f"  Required T for phi=1 (NV, r=1nm, h=1e-21): {T_req_correct:.2e} s")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_facecolor('#fafaf8')

    T_vals = [s[0] for s in sources]
    names  = [s[1] for s in sources]
    refs   = [s[2] for s in sources]
    colors = [s[3] for s in sources]

    y = np.arange(len(sources))
    bars = ax.barh(y, [np.log10(t) for t in T_vals],
                   color=colors, edgecolor='white', lw=0.5, height=0.65, alpha=0.88)

    # Annotate values
    for i, (t, c) in enumerate(zip(T_vals, colors)):
        ax.text(np.log10(t)+0.12, i, f'{t:.0e} s', va='center', fontsize=9, color=c)

    # Vertical line at T2=10ms (best NV echo)
    ax.axvline(np.log10(1e-2), color='#6ec83a', lw=1.5, ls='--', alpha=0.7)
    ax.text(np.log10(1e-2)+0.06, 5.5, 'Best NV\n$T_2$=10ms', fontsize=8.5, color='#6ec83a', va='top')

    # Vertical line at required
    ax.axvline(np.log10(T_req_correct), color='#9060e0', lw=2, ls='--', alpha=0.8)
    ax.text(np.log10(T_req_correct)+0.06, 5.5,
            f'Required\nT≈{T_req_correct:.0e}s', fontsize=8.5, color='#9060e0', va='top')

    # Gap annotation
    gap_x1 = np.log10(1e-2)
    gap_x2 = np.log10(T_req_correct)
    gap_y  = -0.65
    ax.annotate('', xy=(gap_x2, gap_y), xytext=(gap_x1, gap_y),
                arrowprops=dict(arrowstyle='<->', color='#cc4444', lw=1.8))
    ax.text((gap_x1+gap_x2)/2, gap_y-0.3,
            f'Gap: {abs(gap_x2-gap_x1):.0f} orders of magnitude',
            ha='center', fontsize=9.5, color='#cc4444', fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels([f'{n}\n{r}' for n,r in zip(names,refs)], fontsize=9)
    ax.set_xlabel(r'$\log_{10}(T_2\;[\mathrm{s}])$', fontsize=11)
    ax.set_title('NV center decoherence budget vs. required GW integration time\n'
                 r'($h_+=10^{-21}$, $f=100$ Hz, $\langle r\rangle=1$ nm, NV electron)',
                 fontsize=11)

    # Reference comparison
    ref_colors = {'#e05a2b':'Literature','#6ec83a':'State-of-art','#9060e0':'GW target'}
    patches = [mpatches.Patch(color=c, label=l) for c,l in ref_colors.items()]
    ax.legend(handles=patches, fontsize=9, loc='lower right')

    ax.set_xlim(-7, 8)
    ax.set_ylim(-1.2, len(sources)-0.3)
    ax.grid(axis='x', alpha=0.25)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.savefig('figures/decoherence_budget.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('saved: figures/decoherence_budget.png')


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
    print('Generating new thesis figures...')
    fig_hydrogen_radial_overlap()
    fig_literature_landscape()
    fig_decoherence_budget()
    fig_gw_strain_waterfall()
    print('\nAll 4 figures saved to ./figures/')
    print()
    print('Thesis placement:')
    print('  boughn_rothman_radial.png  -> Ch. 6 §6.2 (Boughn-Rothman discussion)')
    print('  literature_landscape.png   -> Ch. 6 §6.4 (Synthesis section)')
    print('  decoherence_budget.png     -> Ch. 6 §6.3 (SQL discussion)')
    print('  gw_strain_waterfall.png    -> Ch. 5 §5.4 or Ch. 6 intro')