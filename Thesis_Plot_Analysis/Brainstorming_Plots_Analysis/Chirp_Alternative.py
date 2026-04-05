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