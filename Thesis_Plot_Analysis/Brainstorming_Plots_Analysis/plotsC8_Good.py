# GW Waveform and Strain Time Series
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp

# ─── GW chirp signal (inspiral approximation) ─────────────────
t_merge = 1.0      # s (merger time)
t       = np.linspace(0, t_merge, 10000)
f0      = 30.0     # starting frequency (Hz)
f1      = 300.0    # merger frequency (Hz)
h0      = 1e-21    # peak strain

# Chirp frequency evolution: f(t) ~ (t_merge - t)^(-3/8)
tau = t_merge - t
tau = np.maximum(tau, 1e-4)   # avoid division by zero near merger

f_chirp = f0 * (tau / t_merge)**(-3/8)
f_chirp = np.minimum(f_chirp, f1)

# Phase by integration of frequency
dt = t[1] - t[0]
phase = 2 * np.pi * np.cumsum(f_chirp) * dt

# Amplitude envelope: grows near merger
amplitude = h0 * (tau / t_merge)**(-1/4)
amplitude[tau < 0.01] = 0   # ringdown cutoff

h_plus  =  amplitude * np.cos(phase)
h_cross =  amplitude * np.sin(phase)

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

ax = axes[0]
ax.plot(t, h_plus  / 1e-21, color='#2171B5', lw=1.2,
        label='$h_+$ polarization')
ax.plot(t, h_cross / 1e-21, color='#E31A1C', lw=1.2, alpha=0.7,
        label='$h_\\times$ polarization')
ax.set_ylabel('Strain $h\\;[10^{-21}]$', fontsize=10)
ax.set_title('GW chirp signal from binary inspiral\n'
             '(schematic, 30–300 Hz, $M_{\\rm tot}=60\\,M_\\odot$)',
             fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.25)
ax.axvline(t_merge - 0.001, color='gray', lw=0.8, ls='--', alpha=0.6)
ax.text(t_merge - 0.05, 0.7, 'Merger', fontsize=8, color='gray',
        ha='right')

ax = axes[1]
ax.plot(t, f_chirp, color='#35978F', lw=1.5)
ax.set_ylabel('Instantaneous\nfrequency (Hz)', fontsize=10)
ax.grid(alpha=0.25)

ax = axes[2]
# NV energy shift: Delta E ~ kappa * h * <p^2> / (2m)
hbar  = 1.055e-34
c     = 3.0e8
G     = 6.674e-11
m_e   = 9.109e-31
alpha = 1/137.0
p_char = alpha * m_e * c
kappa_SI = np.sqrt(32*np.pi*G) / c**2

dE = kappa_SI * np.abs(h_plus) * p_char**2 / (2 * m_e)  # Joules
dE_Hz = dE / (2*np.pi*hbar)  # in Hz units

ax.semilogy(t, dE_Hz + 1e-30, color='#8856A7', lw=1.5)
ax.set_ylabel('NV frequency shift\n(Hz)', fontsize=10)
ax.set_xlabel('Time (s)', fontsize=11)
ax.axhline(1e-30, color='gray', lw=0.8, ls='--', alpha=0.6)
ax.text(0.05, 2e-30, 'Quantum noise floor $\\sim 10^{-30}$ Hz',
        fontsize=8.5, color='gray')
ax.grid(which='both', alpha=0.25)

plt.tight_layout()
plt.savefig("figures/gw_waveform_and_shift.png",
            bbox_inches="tight", dpi=300)
plt.show()
print("Saved: gw_waveform_and_shift.pdf")