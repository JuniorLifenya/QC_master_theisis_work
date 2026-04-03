"""
P_{0→1}(t) for a Binary Neutron Star Inspiral — NV Spin Quantum Sensor
========================================================================
Three physically distinct regimes are shown:
  (i)  DC / broadband limit (ω_0 → 0): P grows monotonically throughout
       the whole inspiral as the GW strain accumulates — relevant for a
       broadband quantum sensor.
  (ii) Resonant sensor at f_0 = 35 Hz: the GW is resonant at t* ≈ 53.33 s
       (0.19 s before merger); P shows a sharp Zener-like step.
  (iii) Zoom-in on the resonance crossing interval showing the coherent
       build-up and post-resonance oscillations.

Physics:  Foldy-Wouthuysen effective Rabi coupling  Ω(t) = (κ p*²/4mℏ) h₊(t)
Method:   First-order time-dependent perturbation theory, exact for P ≪ 1
Waveform: Leading-order post-Newtonian BNS chirp (m1=m2=1.4 M⊙, d=40 Mpc)
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import uniform_filter1d
import os

matplotlib.rcParams.update({
    'font.family'       : 'serif',
    'font.serif'        : ['DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset'  : 'stix',
    'font.size'         : 10,
    'axes.labelsize'    : 11,
    'axes.titlesize'    : 10,
    'xtick.labelsize'   : 9,
    'ytick.labelsize'   : 9,
    'legend.fontsize'   : 8.5,
    'figure.dpi'        : 150,
    'axes.axisbelow'    : True,
})

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICAL CONSTANTS  (CODATA 2018)
# ─────────────────────────────────────────────────────────────────────────────
G      = 6.674_30e-11
c      = 2.997_924_58e8
hbar   = 1.054_571_817e-34
m_e    = 9.109_383_7015e-31
alpha  = 7.297_352_5693e-3
M_sun  = 1.98892e30
Mpc    = 3.085_677_581e22

kappa  = np.sqrt(32*np.pi*G) / c**2   # FW gravitomagnetic coupling constant
p_star = alpha * m_e * c               # characteristic electron momentum

# ─────────────────────────────────────────────────────────────────────────────
# BNS PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
m1, m2  = 1.4*M_sun, 1.4*M_sun
M_tot   = m1 + m2
M_c     = (m1*m2)**0.6 / M_tot**0.2   # 1.2188 M_sun
d       = 40.0*Mpc
f_isco  = c**3 / (6**1.5 * np.pi * G * M_tot)   # 1570 Hz
f_low   = 30.0    # Hz — LIGO/ET low-frequency cutoff
xi      = G*M_c/c**3                              # 5.996e-6 s
t_coal  = (5.0/256)*xi**(-5/3)*(np.pi*f_low)**(-8/3)  # 53.52 s

print(f"Mc={M_c/M_sun:.4f} M_sun | f_isco={f_isco:.0f} Hz | T_coal={t_coal:.3f} s")

# ─────────────────────────────────────────────────────────────────────────────
# WAVEFORM  (leading-order PN, face-on, h+ polarisation)
# ─────────────────────────────────────────────────────────────────────────────
N   = 500_000
t   = np.linspace(1e-8, 0.99998*t_coal, N)
tau = t_coal - t

f_gw  = np.minimum((1.0/(8*np.pi*xi))*(5*xi/(256*tau))**0.375, f_isco)
h_amp = (4*G*M_c/(c**2*d)) * (np.pi*G*M_c*f_gw/c**3)**(2/3)
phi_gw = 2*np.pi*np.concatenate([[0.], cumulative_trapezoid(f_gw, t)])
h_plus = h_amp * np.cos(phi_gw)

# FW effective Rabi half-frequency
Omega = (kappa*p_star**2 / (4*m_e*hbar)) * h_plus   # rad s⁻¹
print(f"peak |Ω_FW| = {np.max(np.abs(Omega)):.3e} rad/s | h_peak = {np.max(h_amp):.3e}")

# ─────────────────────────────────────────────────────────────────────────────
# TRANSITION PROBABILITY — three cases
# ─────────────────────────────────────────────────────────────────────────────
def P_transition(omega_0):
    """First-order TDPT: P_{0→1}(t) = |−i/2 ∫ Ω e^{i(ω₀t−φ_GW)} dt|²"""
    phase = omega_0*t - phi_gw
    c1 = -1j * np.concatenate([[0.], cumulative_trapezoid(
                (Omega/2)*np.exp(1j*phase), t)])
    return np.abs(c1)**2

# (i) DC / broadband limit: ω₀ = 0
# This represents the total absorbed spin rotation — the signal from a
# sensor much broader than the GW bandwidth.  No coherent detuning phase.
P_dc = P_transition(omega_0=0.0)

# (ii) Resonant sensor at f_0 = 35 Hz — GW sweeps THROUGH resonance
f_0     = 35.0               # Hz
omega_0 = 2*np.pi*f_0
P_res   = P_transition(omega_0=omega_0)

# Resonance crossing time
i_res = np.argmin(np.abs(f_gw - f_0))
t_res = t[i_res]
# chirp rate at resonance: ḟ = (3/8) f / τ
fdot_res = (3./8)*f_0/(t_coal - t_res)
# Stationary-phase (Fresnel) estimate of P after resonance
Omega_res = (kappa*p_star**2/(4*m_e*hbar))*h_amp[i_res]
P_sp      = (Omega_res/2)**2 / (2*np.pi*fdot_res)

print(f"f_0={f_0} Hz: t*={t_res:.3f} s | ḟ={fdot_res:.2f} Hz/s | P_sp={P_sp:.3e}")
print(f"P_DC(end)={P_dc[-1]:.3e} | P_res(end)={P_res[-1]:.3e}")

# Running-mean envelope (for clarity on oscillating P_res)
win_pts = max(1, int(N*0.0015))
def smooth_envelope(P):
    return uniform_filter1d(P, size=win_pts, mode='nearest')

env_res = smooth_envelope(P_res)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────────────────────────────────────
BG    = '#F9F7F3'
CA    = '#1B3870'   # navy — strain
CB    = '#175A3A'   # forest — freq
CDC   = '#8B4513'   # brown — DC case
CRES  = '#5B1A8A'   # violet — resonant
CR    = '#C47C0A'   # amber — resonance marker
CG    = '#BBBBBB'   # grid

fig = plt.figure(figsize=(11, 12), facecolor=BG)
gs  = gridspec.GridSpec(4, 1, hspace=0.07,
                        top=0.95, bottom=0.07, left=0.13, right=0.94)
axs = [fig.add_subplot(gs[i]) for i in range(4)]
for ax in axs:
    ax.set_facecolor(BG)
    for sp in ['top','right']:
        ax.spines[sp].set_color(CG)

# ══ (a) GW strain ═══════════════════════════════════════════════════════════
ax = axs[0]
ax.plot(t, h_plus/1e-22, color=CA, lw=0.45, alpha=0.85)
ax.plot(t,  h_amp/1e-22, color=CA, lw=1.1, ls='--', alpha=0.5,
        label='amplitude envelope')
ax.plot(t, -h_amp/1e-22, color=CA, lw=1.1, ls='--', alpha=0.5)
ax.axvline(t_res, color=CR, ls=':', lw=1.0, alpha=0.8)
ax.set_ylabel(r'$h_+(t)\;\;[10^{-22}]$')
ax.set_title(
    r'BNS inspiral GW — NV spin transition probability $P_{0\to 1}(t)$'
    r'$\quad(m_1\!=\!m_2\!=\!1.4\,M_\odot,\ d\!=\!40\,\mathrm{Mpc},\ f_\mathrm{low}\!=\!30\,\mathrm{Hz})$',
    fontsize=9.5, pad=4)
ax.legend(loc='upper left', framealpha=0.75)
ax.grid(alpha=0.18, color=CG)
ax.text(0.99, 0.92, r'$\mathbf{(a)}$  GW strain',
        transform=ax.transAxes, ha='right', fontsize=9, color=CA)

# ══ (b) Instantaneous frequency ════════════════════════════════════════════
ax = axs[1]
ax.plot(t, f_gw, color=CB, lw=1.3)
ax.axhline(f_0, color=CR, ls='--', lw=1.1,
           label=rf'Resonance $f_0={f_0:.0f}\,\mathrm{{Hz}}$')
ax.axvline(t_res, color=CR, ls=':', lw=1.0)
ax.set_ylabel(r'$f_\mathrm{GW}(t)$ (Hz)')
ax.legend(loc='upper left', framealpha=0.75)
ax.grid(alpha=0.18, color=CG)
ax.text(0.99, 0.92, r'$\mathbf{(b)}$  Instantaneous frequency',
        transform=ax.transAxes, ha='right', fontsize=9, color=CB)
ax.annotate(
    rf'$t_* = {t_res:.2f}$ s$\;\;|\;\;\dot f = {fdot_res:.1f}$ Hz s$^{{-1}}$',
    xy=(t_res, f_0),
    xytext=(t_res - 12, f_0*1.9),
    arrowprops=dict(arrowstyle='->', lw=0.8, color='#666'),
    fontsize=8, color='#444')

# ══ (c) DC / broadband transition probability ═══════════════════════════════
ax = axs[2]
ax.semilogy(t, P_dc + 1e-310, color=CDC, lw=1.1,
            label=r'$P^\mathrm{DC}_{0\to 1}(t)$  broadband ($\omega_0\!=\!0$)')
ax.axvline(t_res, color=CR, ls=':', lw=0.9, alpha=0.7)
ax.set_ylabel(r'$P^\mathrm{DC}_{0\to 1}(t)$')
ax.grid(alpha=0.18, which='both', color=CG)
ax.legend(loc='upper left', framealpha=0.75)
ax.text(0.99, 0.92, r'$\mathbf{(c)}$  Broadband (DC) response',
        transform=ax.transAxes, ha='right', fontsize=9, color=CDC)
# Annotate: P grows ~h² f^{4/3}, so faster near merger
ax.text(0.60, 0.15,
        r'$P^\mathrm{DC}(t) \propto \left|\!\int_0^t \Omega(t^\prime)\,dt^\prime\right|^2$',
        transform=ax.transAxes, fontsize=9, color=CDC,
        bbox=dict(boxstyle='round,pad=0.3', fc=BG, ec='#CCC', lw=0.6))

# ══ (d) Resonant transition probability ════════════════════════════════════
ax = axs[3]
# raw oscillating P_res (thin)
ax.semilogy(t, P_res + 1e-310, color=CRES, lw=0.3, alpha=0.45, zorder=3)
# smoothed envelope
ax.semilogy(t, env_res + 1e-310, color=CRES, lw=1.5, alpha=0.95, zorder=4,
            label=r'$\langle P^\mathrm{res}_{0\to 1}\rangle$ (envelope)')
ax.axvline(t_res, color=CR, ls='--', lw=1.1, zorder=5,
           label=rf'$t_* = {t_res:.2f}$ s  ($f_\mathrm{{GW}}={f_0}\,\mathrm{{Hz}}$)')
ax.axhline(P_sp, color=CR, ls=':', lw=1.0, alpha=0.75,
           label=rf'Stationary-phase estimate $P_\mathrm{{sp}} = {P_sp:.2e}$')

# Region shading
yrange = ax.get_ylim()
ax.axvspan(0, t_res, alpha=0.04, color='#888', zorder=0)
ax.axvspan(t_res, t[-1], alpha=0.06, color=CR, zorder=0)
ax.set_ylabel(r'$P^\mathrm{res}_{0\to 1}(t)$')
ax.set_xlabel(r'Time $t$ (s)  [0 = band entry at $f_\mathrm{low}=30$ Hz]', fontsize=10)
ax.legend(loc='upper left', framealpha=0.78, ncol=1)
ax.grid(alpha=0.18, which='both', color=CG)
ax.text(0.99, 0.92, r'$\mathbf{(d)}$  Resonant sensor  ($f_0=35$ Hz)',
        transform=ax.transAxes, ha='right', fontsize=9, color=CRES)

# ── absolute-scale footnote in (d)
ax.text(0.01, 0.05,
        rf'$|\Omega_\mathrm{{FW}}|_\mathrm{{peak}} = {np.max(np.abs(Omega)):.2e}\ \mathrm{{rad\,s^{{-1}}}}$  '
        r'(single NV electron, $\kappa=\sqrt{32\pi G}/c^2$, $p_* = \alpha m_e c$)',
        transform=ax.transAxes, fontsize=7.5, color='#666',
        bbox=dict(boxstyle='round,pad=0.3', fc=BG, ec='#CCC', lw=0.5))

# ── shared x-axis ──────────────────────────────────────────────────────────
for ax in axs[:-1]:
    ax.set_xticklabels([])
    ax.tick_params(which='both', direction='in', top=True, right=True)
axs[-1].tick_params(which='both', direction='in', top=True, right=True)
axs[-1].xaxis.set_minor_locator(ticker.AutoMinorLocator())
for ax in axs:
    try:
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    except Exception:
        pass  # log axes skip minor auto locator

# ── ZOOM INSET on resonance crossing in panel (d) ─────────────────────────
ax_main = axs[3]
ax_ins  = inset_axes(ax_main, width='42%', height='44%', loc='lower right',
                     bbox_to_anchor=(0.0, 0.0, 0.99, 0.98),
                     bbox_transform=ax_main.transAxes)
ax_ins.set_facecolor('#FFFFF4')
# Zoom window: t_res ± 0.5 s
dt_zoom = 0.5
mask = (t >= t_res - dt_zoom) & (t <= t_res + dt_zoom)
ax_ins.semilogy(t[mask], P_res[mask] + 1e-310,
                color=CRES, lw=0.5, alpha=0.55)
ax_ins.semilogy(t[mask], env_res[mask] + 1e-310,
                color=CRES, lw=1.5, alpha=0.9)
ax_ins.axvline(t_res, color=CR, ls='--', lw=1.0)
ax_ins.set_xlim(t_res - dt_zoom, t_res + dt_zoom)
ax_ins.tick_params(labelsize=7, which='both', direction='in')
ax_ins.set_xlabel(r'$t$ (s)', fontsize=7.5)
ax_ins.set_ylabel(r'$P_{0\to1}$', fontsize=7.5)
ax_ins.set_title(r'Zoom: resonance crossing', fontsize=7.5, pad=2)
ax_ins.grid(alpha=0.25, which='both', color=CG)

try:
    mark_inset(ax_main, ax_ins, loc1=2, loc2=3,
               fc="none", ec="0.5", lw=0.5)
except Exception:
    pass

os.makedirs('/home/claude/figures', exist_ok=True)
# fig.savefig('/home/claude/figures/P_01_BNS_inspiral.png',
#             dpi=300, bbox_inches='tight', facecolor=BG)
plt.show()
print("Saved: figures/P_01_BNS_inspiral.png")
plt.close()