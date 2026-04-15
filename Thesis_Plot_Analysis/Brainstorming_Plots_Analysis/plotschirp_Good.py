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

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- 1. PHYSICAL CONSTANTS & PARAMETERS ---
hbar = 1.0545718e-34
m_e  = 9.10938356e-31
c    = 2.99792458e8
G    = 6.67430e-11
kappa_SI = np.sqrt(32 * np.pi * G) / c**2

# Sensor Parameters (e.g., an NV center or a Tobar-style resonator)
f_transition = 150.0  # Hz (The sensor's resonance frequency)
omega0 = 2 * np.pi * f_transition
# Coupling strength (Effective Rabi Frequency scale)
# V_0 = (kappa * h_0 * p^2) / (4 * m_e * hbar)
h0 = 1e-21
p_char = (1/137.0) * m_e * c # Characteristic momentum
coupling_constant = (kappa_SI * p_char**2) / (4 * m_e * hbar)
Omega_R = coupling_constant * h0 # Base Rabi frequency

# --- 2. GW CHIRP WAVEFORM ---
t_merge = 1.0
t_span = (0, t_merge - 0.01)
t_eval = np.linspace(0, t_merge - 0.01, 10000)

def get_chirp_params(t):
    tau = t_merge - t
    f_gw = 30.0 * (tau / t_merge)**(-3/8)
    amp = h0 * (tau / t_merge)**(-1/4)
    return f_gw, amp

# --- 3. QUANTUM DYNAMICS (Schrödinger Equation) ---
# State vector psi = [c0, c1] (complex)
# H = [[0, V(t)*exp(-i*phi)], [V(t)*exp(i*phi), omega0]]
# We use the Rotating Wave Approximation (RWA) near resonance

# Phase integration for the chirp
dt = t_eval[1] - t_eval[0]
f_vals = [get_chirp_params(ti)[0] for ti in t_eval]
phi_vals = 2 * np.pi * np.cumsum(f_vals) * dt

def schrodinger_func(t, y):
    c0, c1 = y[0], y[1]
    f_gw, amp = get_chirp_params(t)
    
    # Instantaneous phase from lookup
    idx = int(t / dt)
    if idx >= len(phi_vals): idx = -1
    phi = phi_vals[idx]
    
    V = coupling_constant * amp
    
    # Schrodinger Eq: d/dt c = -i/hbar H c
    # We shift the energy of |0> to 0 for simplicity
    dc0 = -1j * V * np.cos(phi) * c1 
    dc1 = -1j * (omega0 * c1 + V * np.cos(phi) * c0)
    return [dc0, dc1]

# Solve with Initial State |0>
sol = solve_ivp(schrodinger_func, t_span, [1+0j, 0+0j], t_eval=t_eval, rtol=1e-8)

prob_0 = np.abs(sol.y[0])**2
prob_1 = np.abs(sol.y[1])**2

# --- 4. PLOTTING ---
plt.rcParams.update({'font.family':'serif', 'font.size':10})
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

# Plot A: The Waveform
axes[0].plot(t_eval, h0 * ( (t_merge-t_eval)/t_merge )**(-1/4) * np.cos(phi_vals), color='#2171B5', lw=0.8)
axes[0].set_ylabel('Strain $h(t)$')
axes[0].set_title('Inspiral Chirp Signal ($h_+$)')
axes[0].grid(alpha=0.3)

# Plot B: Frequency Sweep & Resonance
axes[1].plot(t_eval, f_vals, color='#35978F', lw=2, label='$f_{GW}(t)$')
axes[1].axhline(f_transition, color='#E31A1C', ls='--', label='Sensor $f_{0 \to 1}$')
axes[1].set_ylabel('Frequency [Hz]')
axes[1].legend()
axes[1].grid(alpha=0.3)

# Plot C: Transition Probability (The Master Plot)
axes[2].plot(sol.t, prob_1, color='#D94801', lw=2, label='$P_{0 \\to 1}(t)$')
axes[2].fill_between(sol.t, 0, prob_1, color='#D94801', alpha=0.1)
axes[2].set_ylabel('Excitation Prob.')
axes[2].set_xlabel('Time to Merger [s]')
axes[2].set_ylim(-0.05, 1.05)
axes[2].set_title('Quantum Transition Probability (Tobar Fig 3 Analogue)')
axes[2].grid(alpha=0.3)

# Annotate Resonance Crossing
t_res = t_eval[np.argmin(np.abs(np.array(f_vals) - f_transition))]
axes[2].annotate('Resonance Crossing', xy=(t_res, 0.5), xytext=(t_res-0.3, 0.8),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1))

plt.tight_layout()
# plt.savefig("figures/bns_inspiral_quantum_transition.png", dpi=300)
plt.show()