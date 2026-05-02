"""
fig_ch5_sensitivity_map.py
Chapter 5 — Quantum sensor sensitivity vs classical GW detectors.
Uses rigorous LIF/FNC coupling formula (verified against B&R).
"""
import numpy as np
import matplotlib.pyplot as plt
import os; os.makedirs("figures", exist_ok=True)

c    = 2.99792458e8
hbar = 1.054571817e-34
G    = 6.67430e-11
m_e  = 9.10938356e-31
alpha = 1/137.035999
a_0  = hbar/(m_e*alpha*c)

f = np.logspace(1, 4, 1000)

# LIGO A+ approximate sensitivity
def ligo(f):
    f0 = 110.; h0 = 8e-24
    n = h0*np.sqrt((f0/f)**4.5 + 1.5 + 0.5*(f/f0)**2.5)
    n[f < 10] = np.nan
    return n

# Einstein Telescope approximate
def et(f):
    f0 = 100.; h0 = 5e-25
    n = h0*np.sqrt((f0/f)**4 + 1 + (f/f0)**2)
    n[f < 2] = np.nan
    return n

# NV ensemble using LIF coupling (thesis derivation)
# g = m_e * omega^2 * <r^2>_{2p} / 4 = m_e*(2pi f)^2 * 5 a_0^2 / 4
# h_min = hbar / (g * sqrt(tau * T2))
def nv_sql(f, T2=1e-2, tau=3.15e7, N_spin=1e12):
    w = 2*np.pi*f
    g = m_e * w**2 * 5 * a_0**2 / 4   # J per unit h (LIF, H-like)
    # For ensemble: multiply g by N_spin coherently
    g_ens = g * N_spin
    return hbar / (g_ens * np.sqrt(tau * T2))

def nv_squeezed(f, xi_db=15, **kw):
    return nv_sql(f, **kw) * 10**(-xi_db/20)

h_l = ligo(f.copy())
h_e = et(f.copy())
h_n = nv_sql(f)
h_s = nv_squeezed(f)

fig, ax = plt.subplots(figsize=(10, 6.5))
ax.loglog(f, h_l, color='#636363', lw=2,   label='LIGO A+ design')
ax.loglog(f, h_e, color='#969696', lw=2, ls='--', label='Einstein Telescope')
ax.loglog(f, h_n, color='steelblue', lw=2.5, label=r'NV ensemble SQL ($N=10^{12}$, $T_2=10$ ms, 1 yr)')
ax.loglog(f, h_s, color='red',       lw=2.5, label=r'NV + 15 dB squeezing')

ax.fill_between(f, h_s, h_n, color='red', alpha=0.10)

# Astrophysical markers
ax.plot(35,   1e-21, 'k*', ms=14, zorder=6, label='GW150914 (BBH, 35 Hz)')
ax.plot(100,  5e-22, 'ks', ms=10, zorder=6, label='GW170817 (BNS, 100 Hz)')

ax.set_xlim(10, 1e4); ax.set_ylim(1e-28, 1e-18)
ax.set_xlabel(r'GW frequency $f$ [Hz]', fontsize=12)
ax.set_ylabel(r'Strain sensitivity $h_{\rm min}$ [1/$\sqrt{\rm Hz}$]', fontsize=12)
ax.set_title('Quantum Sensing vs Classical Detectors\n'
             r'NV sensitivity from: $h_{\rm min}=\hbar/(N g_{\rm LIF}\sqrt{\tau T_2^*})$,'
             r'  $g_{\rm LIF}=m_e\omega^2\langle r^2\rangle/4$', fontsize=11)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, which='both', alpha=0.25)

ax.text(0.02, 0.06,
        r'$N_{\rm spins}=10^{12}$,  $T_2=10$ ms' '\n'
        r'$\tau_{\rm int}=1$ yr,  LIF coupling (Ch.4)',
        transform=ax.transAxes, fontsize=9,
        bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray'))

plt.tight_layout()
plt.show()
plt.savefig('figures/fig_ch5_sensitivity_map.png', dpi=200, bbox_inches='tight')
print("Saved: figures/fig_ch5_sensitivity_map.png")
# Print key numbers for thesis text
for f0, lab in [(100, '100 Hz'), (1000, '1 kHz')]:
    w = 2*np.pi*f0
    g = m_e*w**2*5*a_0**2/4 * 1e12
    hm = hbar/(g*np.sqrt(3.15e7*1e-2))
    print(f"  h_min({lab}, SQL, 1yr) = {hm:.2e}")
