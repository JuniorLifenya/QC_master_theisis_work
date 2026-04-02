# Sensitivity Curve: Minimum Detectable Strain
import numpy as np
import matplotlib.pyplot as plt

# ─── NV center parameters ─────────────────────────────────────
hbar       = 1.055e-34   # J·s
c          = 3.0e8       # m/s
G          = 6.674e-11   # m³/(kg·s²)
m_e        = 9.109e-31   # kg
alpha_fs   = 1/137.0
p_char     = alpha_fs * m_e * c

kappa_SI   = np.sqrt(32 * np.pi * G) / c**2

# Delta E / h_strain (strain sensitivity per unit strain, in J)
dE_dh      = kappa_SI * p_char**2 / (2 * m_e)   # kinetic strain channel

T2_star    = 1e-3        # s  (dephasing time — NV coherence)
T2         = 1e-2        # s  (echo coherence time)
N_spins    = 1e6         # number of NV centers
T_int      = 3.156e7     # s  (1 year integration)

f_arr = np.logspace(0, 5, 500)   # 1 Hz – 100 kHz

def h_min_SQL(f, T2_time, N, T_total):
    """
    Minimum detectable strain at standard quantum limit.
    h_min ~ hbar / (dE_dh * T2 * sqrt(N * T_total / T2))
    """
    n_meas = T_total / T2_time
    # sensitivity per shot ~ hbar / (dE_dh * T2_time)
    h_min_single = hbar / (dE_dh * T2_time)
    
    # Calculate the scalar minimum strain
    strain_scalar = h_min_single / np.sqrt(N * n_meas)
    
    # Multiply by an array of 1s to match the shape of the frequency array
    return strain_scalar * np.ones_like(f)

# ─── compute sensitivity curves ───────────────────────────────
h_NV_Tstar = h_min_SQL(f_arr, T2_star, N_spins, T_int)
h_NV_T2    = h_min_SQL(f_arr, T2,      N_spins, T_int)
h_NV_T2_single = h_min_SQL(f_arr, T2, 1,       T_int)

# ─── LIGO design sensitivity (approximate analytical fit) ──────
def ligo_strain(f):
    """Approximate LIGO O4 design sensitivity."""
    f0 = 100.0
    h0 = 1e-23
    return h0 * np.sqrt((f0/f)**4 + 1 + (f/f0)**2)

h_LIGO = ligo_strain(f_arr)

# Typical GW events
events = {
    'GW150914\n($h \\sim 10^{-21}$)': (35, 1e-21, 'o', '#E31A1C'),
    'GW170817\n($h \\sim 10^{-22}$)': (100, 1e-22, 's', '#FF7F00'),
}

# ─── plot ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6.5))

ax.loglog(f_arr, h_NV_T2,    color='#2171B5', lw=2.0,
          label=r'NV ensemble ($T_2 = 10\,$ms, $N=10^6$, 1 yr)')
ax.loglog(f_arr, h_NV_Tstar, color='#6BAED6', lw=1.5, ls='--',
          label=r'NV ensemble ($T_2^* = 1\,$ms)')
ax.loglog(f_arr, h_NV_T2_single, color='#9ECAE1', lw=1.2, ls=':',
          label=r'Single NV ($T_2 = 10\,$ms, 1 yr)')
ax.loglog(f_arr, h_LIGO,     color='#FC4E2A', lw=2.0,
          label='LIGO O4 design sensitivity')

# Shot-noise floor
ax.axhline(1e-30, color='gray', lw=0.8, ls='--', alpha=0.5)
ax.text(2, 1.4e-30, r'$\Delta E/E \sim 10^{-30}$',
        fontsize=8.5, color='gray')

for label, (f_ev, h_ev, mk, col) in events.items():
    ax.scatter([f_ev], [h_ev], marker=mk, s=80, color=col,
               zorder=6, label=label)

ax.set_xlabel('Frequency (Hz)', fontsize=12)
ax.set_ylabel(r'Strain sensitivity $h_{\rm min}$', fontsize=12)
ax.set_title('NV-center gravitational wave sensitivity\n'
             'vs LIGO O4 design curve\n'
             r'(kinetic strain channel: $\hat{H}_{\rm strain} = '
             r'\frac{\kappa}{2m_e}h_{ij}\hat{p}^i\hat{p}^j$)',
             fontsize=11)
ax.set_xlim(1, 1e5)
ax.set_ylim(1e-34, 1e-18)
ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
ax.grid(which='both', alpha=0.2, lw=0.4)

plt.tight_layout()
plt.savefig("figures/nv_sensitivity_curve.png",
            bbox_inches="tight", dpi=300)
plt.show()
print("Saved: nv_sensitivity_curve.pdf")