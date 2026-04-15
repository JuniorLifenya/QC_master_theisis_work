"""
fig_gso_thomas_darwin.py
========================
Three publication-quality figures for:
  Fig A — Gravitational Spin-Orbit coupling
  Fig B — Thomas Precession
  Fig C — Gravitomagnetic Darwin term (Zitterbewegung)

Physics:
  T9a (GSO):    H_gso = -(3κβ/8m) Σ·[(∇h)×p̂]
  T9b (Thomas): H_th  =  (κ/8m²) Σ·[p̂×Ê_g]
  T10  (Darwin): H_D   =  (κβ/16m)∇²h
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D
import os

os.makedirs('figures', exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 11,
    'figure.dpi': 150,
    'text.usetex': False,
})

GOLD   = '#e8a020'
BLUE   = '#3a7bd5'
RED    = '#c03030'
GREEN  = '#2a9050'
PURPLE = '#8050c0'
TEAL   = '#1a8080'
GREY   = '#666666'

# ═══════════════════════════════════════════════════════════════════════════════
# ██  FIG A — GRAVITATIONAL SPIN-ORBIT COUPLING
# ═══════════════════════════════════════════════════════════════════════════════
# Layout: 4 panels
#   (0) Orbital geometry + (∇h)×p vector field
#   (1) Spin precession on Bloch sphere over one orbit
#   (2) Effective GSO field magnitude |B_gso| as electron orbits
#   (3) Energy splitting Δm=±1/2 as function of orbital angle

def make_figA():
    fig = plt.figure(figsize=(22, 10))
    gs  = gridspec.GridSpec(2, 4, figure=fig,
                            wspace=0.38, hspace=0.42,
                            left=0.05, right=0.97,
                            top=0.88, bottom=0.10)

    # ── physics parameters ───────────────────────────────────────────────
    h0     = 0.5          # exaggerated
    omega  = 2.0          # GW angular frequency (arb. units)
    a0     = 1.0          # Bohr radius (arb. units)
    k_gw   = omega        # k = ω/c = ω in natural units here
    kappa  = 0.08         # κ coupling (scaled for visibility)
    m      = 1.0          # electron mass

    N_orbit = 360
    phi_orb = np.linspace(0, 2*np.pi, N_orbit)

    # Electron position on circular orbit of radius a0
    x_e = a0 * np.cos(phi_orb)
    y_e = a0 * np.sin(phi_orb)

    # GW phase at electron position (wave propagating in z, evaluated at z=0, t=0)
    # h_+ = h0 cos(kx) → ∂h/∂x = -h0 k sin(kx), ∂h/∂y = 0
    # Gradient ∇h at electron position (x_e, y_e, 0):
    dhdx = -h0 * k_gw * np.sin(k_gw * x_e)
    dhdy = np.zeros_like(phi_orb)

    # Momentum p̂ (tangent to orbit, normalized)
    px = -np.sin(phi_orb)   # tangent direction
    py =  np.cos(phi_orb)
    pz =  np.zeros_like(phi_orb)

    # GSO effective field: (∇h) × p̂ — z-component
    # (∇h × p̂)_z = (∂h/∂x)(p_y) - (∂h/∂y)(p_x)
    B_gso_z = dhdx * py - dhdy * px
    # x and y components
    B_gso_x = dhdy * pz - np.zeros_like(phi_orb) * py   # = 0 here
    B_gso_y = np.zeros_like(phi_orb) * px - dhdx * pz   # = 0 here

    B_gso_mag = np.abs(B_gso_z)
    # GSO energy: ε = -C * Σ_z * B_gso_z, C = 3κβ/8m
    C_gso = 3 * kappa / (8 * m)
    E_up   =  C_gso * B_gso_z   # m_s = +1/2
    E_down = -C_gso * B_gso_z   # m_s = -1/2

    # ── Panel 0: Orbital geometry + GSO field vectors ────────────────────
    ax0 = fig.add_subplot(gs[:, 0])
    theta_grid = np.linspace(0, 2*np.pi, 200)
    ax0.plot(np.cos(theta_grid), np.sin(theta_grid),
             'k--', lw=1.2, alpha=0.4, label='Orbit (radius $a_0$)')
    ax0.plot(0, 0, 'ko', ms=6, zorder=5)
    ax0.text(0.05, 0.05, 'Nucleus', fontsize=8)

    # Arrow subset for clarity
    skip = 18
    for i in range(0, N_orbit, skip):
        xe, ye = x_e[i], y_e[i]
        Bz = B_gso_z[i]
        # Color by sign
        col = RED if Bz > 0 else BLUE
        # GSO field vector (out-of-plane direction drawn as colored circles)
        size = min(abs(Bz) * 60 + 0.01, 120)
        ax0.scatter(xe, ye, s=size, color=col, zorder=4, alpha=0.85)
        # Momentum tangent
        ax0.annotate('', xy=(xe + 0.18*px[i], ye + 0.18*py[i]),
                     xytext=(xe, ye),
                     arrowprops=dict(arrowstyle='->', color=GOLD, lw=1.4))
        # Gradient ∇h
        if abs(dhdx[i]) > 0.01:
            ax0.annotate('', xy=(xe + 0.22*dhdx[i], ye),
                         xytext=(xe, ye),
                         arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.4))

    ax0.set_aspect('equal')
    ax0.set_xlim(-2.0, 2.0); ax0.set_ylim(-2.0, 2.0)
    ax0.set_xlabel(r'$x/a_0$'); ax0.set_ylabel(r'$y/a_0$')
    ax0.set_title('Orbital geometry\n' + r'$\hat{H}_{\rm GSO} = -\frac{3\kappa}{8m}\,\vec{\Sigma}\cdot[(\nabla h)\times\hat{\vec{p}}]$',
                  fontsize=10)

    legend_els = [
        mpatches.Patch(color=GOLD,  label=r'Momentum $\hat{\vec{p}}$ (tangent)'),
        mpatches.Patch(color=GREEN, label=r'GW gradient $\nabla h$'),
        mpatches.Patch(color=RED,   label=r'$(\nabla h\times\hat{\vec{p}})_z > 0$  [out]'),
        mpatches.Patch(color=BLUE,  label=r'$(\nabla h\times\hat{\vec{p}})_z < 0$  [in]'),
    ]
    ax0.legend(handles=legend_els, loc='upper right', fontsize=7.5, framealpha=0.9)

    # ── Panel 1: Bloch sphere spin precession ────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1], projection='3d')
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax1.plot_surface(xs, ys, zs, alpha=0.07, color='lightblue')

    # Spin precession: spin starts at +z, precesses around z-axis
    # (GSO drives Larmor precession with rate Ω_gso ∝ B_gso_z)
    # Phase accumulated as function of orbital angle φ:
    n_trace = 300
    phi_t = np.linspace(0, 2*np.pi, n_trace)
    Omega_gso = C_gso * h0 * k_gw * np.sin(phi_t) * 0.5
    prec_angle = np.cumsum(Omega_gso) * (2*np.pi / n_trace) * 0.35
    theta_spin = np.pi/4  # spin tilted from z-axis

    sx = np.sin(theta_spin) * np.cos(prec_angle)
    sy = np.sin(theta_spin) * np.sin(prec_angle)
    sz = np.cos(theta_spin) * np.ones_like(prec_angle)

    # Color by prec_angle
    from matplotlib.collections import LineCollection
    points = np.array([sx, sy, sz]).T.reshape(-1, 1, 3)
    segs   = np.concatenate([points[:-1], points[1:]], axis=1)
    colors_spin = plt.cm.viridis(np.linspace(0, 1, len(segs)))
    for i in range(0, len(segs), 4):
        ax1.plot([segs[i,0,0], segs[i,1,0]],
                 [segs[i,0,1], segs[i,1,1]],
                 [segs[i,0,2], segs[i,1,2]],
                 color=colors_spin[i], lw=1.6)

    # Axes
    for d, c, l in [([1.3,0,0],RED,'x'), ([0,1.3,0],GREEN,'y'), ([0,0,1.3],BLUE,'z')]:
        ax1.quiver(0,0,0,d[0],d[1],d[2],color=c,lw=1.4,arrow_length_ratio=0.15)
        ax1.text(*[di*1.15 for di in d], f'$\\hat{{{l}}}$', fontsize=9)

    # Start and end arrows
    ax1.quiver(0,0,0,sx[0],sy[0],sz[0],color=GOLD,lw=2.5,arrow_length_ratio=0.2)
    ax1.quiver(0,0,0,sx[-1],sy[-1],sz[-1],color=PURPLE,lw=2.5,arrow_length_ratio=0.2)
    ax1.text(sx[0]*1.18, sy[0]*1.18, sz[0]*1.18, 'Start', fontsize=8, color=GOLD)
    ax1.text(sx[-1]*1.18, sy[-1]*1.18, sz[-1]*1.18, 'End', fontsize=8, color=PURPLE)

    ax1.set_xlim(-1.3,1.3); ax1.set_ylim(-1.3,1.3); ax1.set_zlim(-1.3,1.3)
    ax1.set_title('Spin precession\nover one orbit', fontsize=10)
    ax1._axis3don = False

    # ── Panel 2: GSO field magnitude vs orbital angle ────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.fill_between(np.degrees(phi_orb), 0, B_gso_mag,
                     alpha=0.35, color=TEAL)
    ax2.plot(np.degrees(phi_orb), B_gso_mag, color=TEAL, lw=2.2)
    ax2.set_xlabel('Orbital angle (deg)')
    ax2.set_ylabel(r'$|(\nabla h \times \hat{\vec{p}})_z|$  [arb.]')
    ax2.set_title('GSO coupling strength\nvs orbital position', fontsize=10)
    ax2.axhline(0, color='k', lw=0.8, ls='--')
    ax2.set_xlim(0, 360)
    ax2.set_xticks([0, 90, 180, 270, 360])

    # ── Panel 3: Energy splitting ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.plot(np.degrees(phi_orb), E_up,   color=RED,  lw=2.2, label=r'$m_s=+\frac{1}{2}$')
    ax3.plot(np.degrees(phi_orb), E_down, color=BLUE, lw=2.2, label=r'$m_s=-\frac{1}{2}$')
    ax3.fill_between(np.degrees(phi_orb), E_up, E_down,
                     alpha=0.15, color=PURPLE, label=r'Splitting $\Delta E$')
    ax3.axhline(0, color='k', lw=0.8, ls='--')
    ax3.set_xlabel('Orbital angle (deg)')
    ax3.set_ylabel(r'$\Delta E_{\rm GSO}$  [arb.]')
    ax3.set_title('GSO energy splitting\n' + r'$m_s=\pm\frac{1}{2}$  vs orbital phase',
                  fontsize=10)
    ax3.legend(fontsize=9, loc='upper right')
    ax3.set_xlim(0, 360)
    ax3.set_xticks([0, 90, 180, 270, 360])

    # ── Panel lower row: magnetic moment + orbit diagram ─────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    # Show how magnetic moment (spin) rotates as electron orbits
    # in the presence of GSO coupling
    skip2 = 24
    phi_sub = phi_orb[::skip2]
    xe_s, ye_s = x_e[::skip2], y_e[::skip2]
    E_s = E_up[::skip2]
    B_s = B_gso_z[::skip2]

    # Spin arrow length encodes |splitting|
    max_E = max(abs(E_up)) + 1e-9
    for i, (xe, ye, Ei, Bi) in enumerate(zip(xe_s, ye_s, E_s, B_s)):
        length = 0.35 * abs(Ei) / max_E + 0.1
        col = RED if Bi > 0 else BLUE
        # Spin arrow: up if Ei > 0 (favorable for m_s=+1/2), down if unfavorable
        ax4.annotate('', xy=(xe, ye + length * np.sign(Ei)),
                     xytext=(xe, ye),
                     arrowprops=dict(arrowstyle='->', color=col, lw=2.2))

    ax4.plot(np.cos(theta_grid), np.sin(theta_grid), 'k--', lw=1.2, alpha=0.5)
    ax4.plot(0, 0, 'ko', ms=6)
    ax4.set_aspect('equal')
    ax4.set_xlim(-2, 2); ax4.set_ylim(-2, 2)
    ax4.set_xlabel(r'$x/a_0$')
    ax4.set_ylabel(r'$y/a_0$')
    ax4.set_title('Spin orientation\nalong orbit', fontsize=10)
    legend_sp = [
        mpatches.Patch(color=RED,  label=r'$\uparrow$ preferred ($m_s=+\frac{1}{2}$)'),
        mpatches.Patch(color=BLUE, label=r'$\downarrow$ preferred ($m_s=-\frac{1}{2}$)'),
    ]
    ax4.legend(handles=legend_sp, fontsize=8, loc='upper right')

    # ── Panel lower right: schematic comparison with EM spin-orbit ────────
    ax5 = fig.add_subplot(gs[1, 2:])
    ax5.axis('off')
    comparison = [
        ('', 'EM Spin-Orbit', 'Grav. Spin-Orbit (T9a)'),
        ('Coupling field', r'$\vec{E}$ (Coulomb)',
                           r'$\nabla h$ (GW gradient)'),
        ('Effective B field', r'$\vec{B}_{\rm eff} = \frac{1}{2mc^2}\vec{E}\times\hat{\vec{p}}$',
                              r'$\vec{B}_{g,\rm eff} = (\nabla h)\times\hat{\vec{p}}$'),
        ('Hamiltonian', r'$\frac{e}{2m^2c^2}\vec{S}\cdot(\vec{E}\times\hat{\vec{p}})$',
                        r'$-\frac{3\kappa\beta}{8m}\vec{\Sigma}\cdot[(\nabla h)\times\hat{\vec{p}}]$'),
        ('Coefficient', r'$\frac{e}{2m^2c^2}$ (exact)',
                        r'$-\frac{3\kappa\beta}{8m}$ (from FW)'),
        ('Requires', 'Spatial E-field gradient',
                     r'Spatial GW gradient $\partial_k h_{ij}\neq0$'),
        ('Uniform field?', 'Spin precesses', r'No precession ($\nabla h=0$)'),
    ]
    col_x = [0.01, 0.30, 0.70]
    header_y = 0.92
    row_h = 0.12
    for ci, header in enumerate(['Property', 'EM Spin-Orbit', 'Grav. Spin-Orbit (T9a)']):
        ax5.text(col_x[ci], header_y, header, fontsize=10, fontweight='bold',
                 color=['#333333', '#1a3a6a', RED][ci],
                 transform=ax5.transAxes)
    ax5.plot([0,1],[header_y-0.03, header_y-0.03], 'k-', lw=0.8, transform=ax5.transAxes)
    for ri, row in enumerate(comparison[1:]):
        ypos = header_y - (ri+1)*row_h - 0.02
        for ci, cell in enumerate(row):
            ax5.text(col_x[ci], ypos, cell, fontsize=9,
                     color=['#333333', '#1a3a6a', RED][ci],
                     transform=ax5.transAxes, va='top')

    fig.suptitle(
        r'Fig A — Gravitational Spin-Orbit Coupling:  '
        r'$\hat{H}_{\rm GSO} = -\frac{3\kappa\beta}{8m}\,\vec{\Sigma}\cdot[(\nabla h)\times\hat{\vec{p}}]$'
        '\n'
        r'The GW spatial gradient $\nabla h$ acts as an effective gravitomagnetic field '
        r'in the electron rest frame, precessing the spin as the electron orbits.',
        fontsize=12, y=0.98
    )
    plt.savefig('figures/figA_spin_orbit.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: figures/figA_spin_orbit.png")


# ═══════════════════════════════════════════════════════════════════════════════
# ██  FIG B — THOMAS PRECESSION
# ═══════════════════════════════════════════════════════════════════════════════

def make_figB():
    fig = plt.figure(figsize=(22, 10))
    gs  = gridspec.GridSpec(2, 4, figure=fig,
                            wspace=0.38, hspace=0.42,
                            left=0.05, right=0.97,
                            top=0.88, bottom=0.10)

    # ── parameters ───────────────────────────────────────────────────────
    kappa  = 0.10
    m      = 1.0
    h0     = 0.5
    omega  = 2.0
    a0     = 1.0
    alpha  = 1/137.0   # fine structure (for velocity estimate)
    v_orb  = alpha * 1.0  # |p|/m ~ α for H atom

    N = 360
    phi = np.linspace(0, 2*np.pi, N)

    # Electron orbital position and velocity
    x_e = a0 * np.cos(phi)
    y_e = a0 * np.sin(phi)
    px_e = -np.sin(phi)  # unit tangent
    py_e =  np.cos(phi)

    # Gravitoelectric field Ê_g (from h_{i0} in FNC, or ∂_t h_{ij} p^j component)
    # In TT gauge + FNC: Ê_g ≈ ∂_t h_ij p^j / m
    # For a propagating wave h_+(t) = h0 cos(ω t - k z), ∂_t h = -ω h0 sin(ω t)
    # At the electron position, Ê_g ~ ω h0 sin(ω t) × p̂
    # Thomas term: κ/(8m²) Σ·(p̂ × Ê_g)
    # At the electron position on orbit (t = φ/Ω_orb for circular orbit):
    omega_h = omega   # GW frequency
    omega_orb = 1.0   # orbital frequency
    t_orb = phi / omega_orb   # time along orbit

    # E_g components (from gravitational analogue)
    Eg_x = omega_h * h0 * np.sin(omega_h * t_orb) * px_e
    Eg_y = omega_h * h0 * np.sin(omega_h * t_orb) * py_e

    # Thomas coupling: p × E_g (z-component)
    # (p × E_g)_z = px * Eg_y - py * Eg_x
    pxEg_z = px_e * Eg_y - py_e * Eg_x

    C_thomas = kappa / (8 * m**2)
    E_thomas_up   =  C_thomas * pxEg_z
    E_thomas_down = -C_thomas * pxEg_z

    # Thomas precession rate (proportional to centripetal acceleration × v)
    # Ω_Thomas = -v²/(2c²) Ω_orb = -(α²/2) Ω_orb (gravitational analogue)
    # We visualize the cumulative precession angle
    dOmega_thomas = C_thomas * np.abs(pxEg_z)
    precession_angle = np.cumsum(dOmega_thomas) * (2*np.pi/N) * 15

    # ── Panel 0: 3D schematic — lab vs rest frame ─────────────────────────
    ax0 = fig.add_subplot(gs[:, 0])
    # Draw two coordinate frames: lab (black) and co-moving rest frame (colored)
    # at several orbital positions
    circle_t = np.linspace(0, 2*np.pi, 200)
    ax0.plot(np.cos(circle_t), np.sin(circle_t), 'k--', lw=1.2, alpha=0.35)
    ax0.plot(0, 0, 'ko', ms=7, zorder=5)
    ax0.text(0.05, -0.12, 'Nucleus', fontsize=8.5)

    skip = 30
    for i in range(0, N, skip):
        xe, ye = x_e[i], y_e[i]
        angle_acc = precession_angle[i]

        # Lab frame axes at electron position (static)
        ax0.annotate('', xy=(xe+0.25, ye),
                     xytext=(xe, ye),
                     arrowprops=dict(arrowstyle='->', color='#aaaaaa', lw=1.2))
        ax0.annotate('', xy=(xe, ye+0.25),
                     xytext=(xe, ye),
                     arrowprops=dict(arrowstyle='->', color='#aaaaaa', lw=1.2))

        # Co-moving frame axes (rotated by Thomas angle)
        cos_a, sin_a = np.cos(angle_acc), np.sin(angle_acc)
        ex = np.array([cos_a, -sin_a]) * 0.28
        ey = np.array([sin_a,  cos_a]) * 0.28
        col = plt.cm.plasma(i / N)
        ax0.annotate('', xy=(xe+ex[0], ye+ex[1]),
                     xytext=(xe, ye),
                     arrowprops=dict(arrowstyle='->', color=col, lw=2.0))
        ax0.annotate('', xy=(xe+ey[0], ye+ey[1]),
                     xytext=(xe, ye),
                     arrowprops=dict(arrowstyle='->', color=col, lw=2.0))

    ax0.set_aspect('equal')
    ax0.set_xlim(-1.9, 1.9); ax0.set_ylim(-1.9, 1.9)
    ax0.set_xlabel(r'$x/a_0$'); ax0.set_ylabel(r'$y/a_0$')
    ax0.set_title('Lab frame (grey) vs\nco-moving frame (coloured)\nThomas rotation accumulates',
                  fontsize=10)

    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(cmap='plasma', norm=Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax0, orientation='horizontal',
                        pad=0.12, fraction=0.04)
    cbar.set_label('Orbital phase $0 \\to 2\\pi$', fontsize=8)

    # ── Panel 1: Thomas precession angle accumulation ────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(np.degrees(phi), np.degrees(precession_angle),
             color=PURPLE, lw=2.4)
    ax1.fill_between(np.degrees(phi), 0, np.degrees(precession_angle),
                     alpha=0.2, color=PURPLE)
    ax1.set_xlabel('Orbital angle (deg)')
    ax1.set_ylabel('Thomas angle (deg)')
    ax1.set_title('Cumulative Thomas\nprecession angle', fontsize=10)
    ax1.set_xlim(0, 360)
    ax1.set_xticks([0, 90, 180, 270, 360])
    ax1.axhline(0, color='k', lw=0.8, ls='--')
    ax1.text(0.97, 0.05, r'$\delta\theta_T \propto \frac{v^2}{c^2}\sim\alpha^2$',
             transform=ax1.transAxes, ha='right', fontsize=10, color=PURPLE)

    # ── Panel 2: Bloch sphere — Thomas precession of spin ─────────────────
    ax2 = fig.add_subplot(gs[0, 2], projection='3d')
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    ax2.plot_surface(np.outer(np.cos(u), np.sin(v)),
                     np.outer(np.sin(u), np.sin(v)),
                     np.outer(np.ones_like(u), np.cos(v)),
                     alpha=0.07, color='lightblue')

    # Spin precesses around the Thomas axis (in the orbital plane)
    # Two cases: pure Larmor vs Larmor + Thomas correction
    # Larmor (GSO): precesses around z-axis at ω_L
    # Thomas correction shifts the precession axis
    n_pts = 300
    phi_t = np.linspace(0, 4*np.pi, n_pts)  # two full orbits
    omega_L = 0.3     # Larmor frequency (arb)
    omega_T = 0.05    # Thomas correction

    # Pure Larmor (blue)
    s_L = np.array([np.sin(np.pi/4)*np.cos(omega_L*phi_t),
                    np.sin(np.pi/4)*np.sin(omega_L*phi_t),
                    np.cos(np.pi/4)*np.ones(n_pts)])
    # Larmor + Thomas (red)
    s_T = np.array([np.sin(np.pi/4)*np.cos((omega_L + omega_T)*phi_t),
                    np.sin(np.pi/4)*np.sin((omega_L + omega_T)*phi_t),
                    np.cos(np.pi/4)*np.ones(n_pts)])

    ax2.plot(s_L[0], s_L[1], s_L[2], color=BLUE,   lw=2.0,
             label='Larmor only', alpha=0.7)
    ax2.plot(s_T[0], s_T[1], s_T[2], color=RED,    lw=2.0,
             label='Larmor+Thomas', alpha=0.9)

    for d, c, l in [([1.3,0,0],RED,'x'), ([0,1.3,0],GREEN,'y'), ([0,0,1.3],BLUE,'z')]:
        ax2.quiver(0,0,0,d[0],d[1],d[2],color=c,lw=1.2,arrow_length_ratio=0.15)
        ax2.text(*[di*1.2 for di in d], f'$\\hat{{{l}}}$', fontsize=8)

    ax2.set_xlim(-1.3,1.3); ax2.set_ylim(-1.3,1.3); ax2.set_zlim(-1.3,1.3)
    ax2.set_title('Spin trajectory:\nLarmor vs Larmor+Thomas', fontsize=10)
    ax2.legend(loc='upper left', fontsize=8)
    ax2._axis3don = False

    # ── Panel 3: Energy splitting from Thomas term ────────────────────────
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.plot(np.degrees(phi), E_thomas_up,   color=RED,  lw=2.2,
             label=r'$m_s=+\frac{1}{2}$')
    ax3.plot(np.degrees(phi), E_thomas_down, color=BLUE, lw=2.2,
             label=r'$m_s=-\frac{1}{2}$')
    ax3.fill_between(np.degrees(phi), E_thomas_up, E_thomas_down,
                     alpha=0.15, color=PURPLE)
    ax3.axhline(0, color='k', lw=0.8, ls='--')
    ax3.set_xlabel('Orbital angle (deg)')
    ax3.set_ylabel(r'$\Delta E_{\rm Thomas}$  [arb.]')
    ax3.set_title('Thomas energy splitting\n' + r'$\hat{H}_T = \frac{\kappa}{8m^2}\vec{\Sigma}\cdot[\hat{\vec{p}}\times\hat{\vec{E}}_g]$',
                  fontsize=10)
    ax3.legend(fontsize=9)
    ax3.set_xlim(0, 360)
    ax3.set_xticks([0, 90, 180, 270, 360])

    # ── Lower row: comparison table EM Thomas vs grav Thomas ──────────────
    ax4 = fig.add_subplot(gs[1, 1:])
    ax4.axis('off')
    rows = [
        ('Origin', 'Lorentz boost of E-field into moving frame',
                   r'Boost of grav. E-field $\hat{\vec{E}}_g = \partial_t \hat{\vec{A}}_g$ into electron rest frame'),
        ('Coupling', r'$\frac{e}{4m^2c^2}\vec{S}\cdot(\hat{\vec{p}}\times\vec{E})$',
                     r'$\frac{\kappa}{8m^2}\vec{\Sigma}\cdot(\hat{\vec{p}}\times\hat{\vec{E}}_g)$'),
        ('Coeff ratio', r'EM : $e/4m^2c^2$',
                        r'Grav : $\kappa/8m^2$ — factor of $\frac{1}{2}$ relative to GSO'),
        ('Precession rate', r'$\Omega_T^{\rm EM} = -\frac{v^2}{2c^2}\Omega_{\rm orb}$',
                            r'$\Omega_T^{\rm grav} \propto \frac{\kappa p^2}{8m^2}|\hat{\vec{E}}_g|$'),
        ('Magnitude (H 2p)', r'$\sim\alpha^4 m_e c^2 \sim 10^{-4}$~eV',
                              r'$\sim\kappa h \alpha^2 m_e \sim 10^{-65}$~eV (at $h=10^{-21}$)'),
    ]
    col_x = [0.01, 0.28, 0.65]
    for ci, hdr in enumerate(['Property', 'EM Thomas', 'Gravitational Thomas (T9b)']):
        ax4.text(col_x[ci], 0.94, hdr, fontsize=10, fontweight='bold',
                 color=['#333333','#1a3a6a',RED][ci], transform=ax4.transAxes)
    ax4.plot([0,1],[0.90,0.90], 'k-', lw=0.8, transform=ax4.transAxes)
    row_h = 0.15
    for ri, row in enumerate(rows):
        ypos = 0.88 - ri*row_h
        for ci, cell in enumerate(row):
            ax4.text(col_x[ci], ypos, cell, fontsize=9, va='top',
                     color=['#333333','#1a3a6a',RED][ci],
                     transform=ax4.transAxes)

    fig.suptitle(
        r'Fig B — Thomas Precession:  '
        r'$\hat{H}_T = \frac{\kappa}{8m^2}\,\vec{\Sigma}\cdot[\hat{\vec{p}}\times\hat{\vec{E}}_g]$'
        '\n'
        r'The gravitational analogue of Thomas precession: '
        r'a Lorentz boost of the gravitoelectric field $\hat{\vec{E}}_g$ '
        r'into the electron rest frame produces a spin-dependent energy shift.',
        fontsize=12, y=0.98
    )
    plt.savefig('figures/figB_thomas_precession.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: figures/figB_thomas_precession.png")


# ═══════════════════════════════════════════════════════════════════════════════
# ██  FIG C — GRAVITOMAGNETIC DARWIN TERM (Zitterbewegung)
# ═══════════════════════════════════════════════════════════════════════════════

def make_figC():
    fig = plt.figure(figsize=(22, 10))
    gs  = gridspec.GridSpec(2, 4, figure=fig,
                            wspace=0.38, hspace=0.42,
                            left=0.05, right=0.97,
                            top=0.88, bottom=0.10)

    # ── parameters ───────────────────────────────────────────────────────
    kappa  = 0.10
    m      = 1.0
    lambda_C = 0.02    # Compton wavelength in units of a0 (α ≈ 1/137)
    h0     = 0.5
    k_gw   = 2.0

    # ── Panel 0: Zitterbewegung — electron position distribution ─────────
    ax0 = fig.add_subplot(gs[0, 0])

    # Zitterbewegung: electron trembles in a disk of radius ~λ_C
    # We show the smeared wavepacket compared to a point particle
    N_zbw = 300
    t_zbw = np.linspace(0, 4*np.pi, N_zbw)
    omega_C = 1.0 / lambda_C  # Compton frequency

    # Point electron trajectory (circular orbit + no trembling)
    r_orb = 1.0
    x_point = r_orb * np.cos(t_zbw * 0.05)
    y_point = r_orb * np.sin(t_zbw * 0.05)

    # Zitterbewegung trajectory: adds rapid oscillation at λ_C
    x_zbw = x_point + lambda_C * np.cos(omega_C * t_zbw)
    y_zbw = y_point + lambda_C * np.sin(omega_C * t_zbw)

    ax0.plot(x_point, y_point, color=BLUE, lw=2.2,
             label='Point electron', zorder=3)
    ax0.plot(x_zbw, y_zbw, color=RED, lw=0.8, alpha=0.6,
             label=r'With Zitterbewegung ($\lambda_C$)')

    # Draw the Compton wavelength circle at a point
    theta_c = np.linspace(0, 2*np.pi, 100)
    idx0 = 30  # pick a point on the orbit
    ax0.plot(x_point[idx0] + lambda_C*np.cos(theta_c),
             y_point[idx0] + lambda_C*np.sin(theta_c),
             color=GOLD, lw=2.5, ls='-.',
             label=r'Trembling radius $\sim\lambda_C$')
    ax0.annotate('', xy=(x_point[idx0]+lambda_C*1.0, y_point[idx0]),
                 xytext=(x_point[idx0], y_point[idx0]),
                 arrowprops=dict(arrowstyle='->', color=GOLD, lw=2.0))
    ax0.text(x_point[idx0]+lambda_C*1.1, y_point[idx0]+0.02,
             r'$\lambda_C$', fontsize=10, color=GOLD)

    ax0.set_aspect('equal')
    ax0.set_xlim(-1.4, 1.4); ax0.set_ylim(-1.4, 1.4)
    ax0.set_xlabel(r'$x/a_0$'); ax0.set_ylabel(r'$y/a_0$')
    ax0.set_title('Zitterbewegung:\nelectron smeared over $\\lambda_C$', fontsize=10)
    ax0.legend(fontsize=8, loc='upper right')

    # ── Panel 1: Effective potential — point vs smeared ───────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    x1d = np.linspace(-3, 3, 500)

    # GW h_+(x) = h0 cos(k_gw x)   →   ∇²h = -k² h0 cos(kx)
    h_x   = h0 * np.cos(k_gw * x1d)
    d2h_x = -k_gw**2 * h0 * np.cos(k_gw * x1d)

    # Point particle samples h(x) directly
    # Smeared particle samples <h(x+δ)> ≈ h(x) + ½λ_C² ∇²h
    h_smeared = h_x + 0.5 * lambda_C**2 * d2h_x

    ax1.plot(x1d, h_x,       color=BLUE, lw=2.5, label=r'$h(x)$ (point particle)')
    ax1.plot(x1d, h_smeared, color=RED,  lw=2.5, ls='--',
             label=r'$\langle h\rangle$ (smeared, Darwin)')
    ax1.fill_between(x1d, h_x, h_smeared, alpha=0.2, color=PURPLE,
                     label=r'Darwin correction $\propto\nabla^2 h$')
    ax1.axhline(0, color='k', lw=0.8, ls='--')
    ax1.set_xlabel(r'$x$ [arb.]')
    ax1.set_ylabel(r'Effective GW coupling [arb.]')
    ax1.set_title('Darwin: smeared electron\nsamples spatially averaged field', fontsize=10)
    ax1.legend(fontsize=8.5, loc='upper right')

    # ── Panel 2: ∇²h map in 2D ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    N2 = 200
    x2  = np.linspace(-3, 3, N2)
    X2, Y2 = np.meshgrid(x2, x2)
    h_2d   = h0 * np.cos(k_gw * X2)        # h_+(x,y) — wave in x
    d2h_2d = -k_gw**2 * h0 * np.cos(k_gw * X2)   # ∇²h

    lim2 = abs(d2h_2d).max()
    norm2 = TwoSlopeNorm(vmin=-lim2, vcenter=0, vmax=lim2)
    im = ax2.contourf(X2, Y2, d2h_2d, levels=40, cmap='RdBu_r', norm=norm2)
    ax2.contour(X2, Y2, d2h_2d, levels=[0], colors='k', linewidths=1.0)
    plt.colorbar(im, ax=ax2, label=r'$\nabla^2 h$  [arb.]', fraction=0.04, pad=0.02)

    # Overlay some electron wavepackets (circles of radius λ_C)
    centers = [(0, 0), (np.pi/k_gw, 0), (2*np.pi/k_gw, 0)]
    for (cx, cy) in centers:
        circle = plt.Circle((cx, cy), lambda_C*8, fill=False,
                             color=GOLD, lw=2.2, ls='-.')
        ax2.add_patch(circle)
        ax2.plot(cx, cy, 'o', color=GOLD, ms=5, zorder=5)

    ax2.set_aspect('equal')
    ax2.set_xlim(-3, 3); ax2.set_ylim(-3, 3)
    ax2.set_xlabel(r'$x$  [arb.]'); ax2.set_ylabel(r'$y$  [arb.]')
    ax2.set_title(r'$\nabla^2 h$ field (background)' + '\n'
                  + r'+ electron positions (gold circles $\sim\lambda_C$)',
                  fontsize=10)

    # ── Panel 3: Darwin energy correction vs position ─────────────────────
    ax3 = fig.add_subplot(gs[0, 3])
    C_darwin = kappa / (16 * m)
    E_darwin = C_darwin * d2h_x
    ax3.plot(x1d, E_darwin, color=TEAL, lw=2.5)
    ax3.fill_between(x1d, 0, E_darwin, alpha=0.25, color=TEAL)
    ax3.axhline(0, color='k', lw=0.8, ls='--')
    ax3.set_xlabel(r'$x$  [arb.]')
    ax3.set_ylabel(r'$\Delta E_{\rm Darwin}$  [arb.]')
    ax3.set_title(r'Darwin energy shift $\frac{\kappa\beta}{16m}\nabla^2 h$' + '\n'
                  + 'vs electron position', fontsize=10)
    ax3.text(0.97, 0.95, r'$\Delta E_D \propto k_{\rm GW}^2 h$',
             transform=ax3.transAxes, ha='right', va='top',
             fontsize=10, color=TEAL)

    # ── Lower row: three-way comparison table ─────────────────────────────
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis('off')
    table_data = [
        ('Term', 'Darwin (EM)', 'Darwin (Gravitational, T10)', 'Key distinction'),
        ('Hamiltonian',
         r'$\frac{e^2}{8m^2c^2}\nabla^2 V_{\rm Coulomb}$',
         r'$\frac{\kappa\beta}{16m}\nabla^2 h$',
         r'GW version: $\nabla^2 h$ not $\nabla^2 V$'),
        ('Physical origin',
         r'Electron smeared over $\lambda_C$; samples $\langle V\rangle$ not $V$',
         r'Same: electron trembles, samples $\langle h\rangle$ not $h$',
         r'Same Zitterbewegung origin'),
        ('Non-zero when',
         r'$\nabla^2 V \neq 0$ (s-state, cusp)',
         r'$\nabla^2 h \neq 0$ (non-plane-wave, near-source)',
         r'Vanishes for pure plane GW in vacuum'),
        ('Magnitude (H 1s)',
         r'$\sim\alpha^4 m_e c^2 / 4 \sim 1.8\times10^{-4}$~eV',
         r'$\sim\kappa h k_{\rm GW}^2 / 16m \sim 10^{-79}$~eV',
         r'Smallest GW term by far'),
        ('With momentum term',
         r'Combine with $\nabla\cdot E$ for gauge invariance',
         r'T10 combines $\nabla^2 h$ and $\nabla\cdot\hat{E}_g$ ',
         r'Hermiticity requires pairing'),
    ]
    col_x2 = [0.01, 0.19, 0.48, 0.77]
    hdr_y = 0.95
    ax4.text(col_x2[0], hdr_y, table_data[0][0], fontsize=9.5, fontweight='bold',
             transform=ax4.transAxes)
    ax4.text(col_x2[1], hdr_y, table_data[0][1], fontsize=9.5, fontweight='bold',
             color='#1a3a6a', transform=ax4.transAxes)
    ax4.text(col_x2[2], hdr_y, table_data[0][2], fontsize=9.5, fontweight='bold',
             color=TEAL, transform=ax4.transAxes)
    ax4.text(col_x2[3], hdr_y, table_data[0][3], fontsize=9.5, fontweight='bold',
             color=GREY, transform=ax4.transAxes)
    ax4.plot([0,1],[0.91,0.91], 'k-', lw=0.8, transform=ax4.transAxes)
    for ri, row in enumerate(table_data[1:]):
        ypos = 0.89 - ri * 0.15
        colors_row = ['#333333', '#1a3a6a', TEAL, GREY]
        for ci, cell in enumerate(row):
            ax4.text(col_x2[ci], ypos, cell, fontsize=8.5, va='top',
                     color=colors_row[ci], transform=ax4.transAxes)

    fig.suptitle(
        r'Fig C — Gravitomagnetic Darwin Term:  '
        r'$\hat{H}_D = \frac{\kappa\beta}{16m}\nabla^2 h - \frac{\kappa}{16m^2}\nabla\cdot\hat{\vec{E}}_g$'
        '\n'
        r'Zitterbewegung smears the electron over $\sim\!\lambda_C$; '
        r'it then samples the spatial average $\langle h\rangle\approx h + \frac{\lambda_C^2}{2}\nabla^2 h$ '
        r'instead of the local field value.',
        fontsize=12, y=0.98
    )
    plt.savefig('figures/figC_darwin_term.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: figures/figC_darwin_term.png")


# ─── run all three ───────────────────────────────────────────────────────────
make_figA()
make_figB()
make_figC()
print("\nAll three figures complete.")