import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries

# Code 2 — GW150914 energy shift plot (50 lines)
# Load the LIGO H1 strain data for GW150914, apply the formula ΔE(t)=κ2me∣Iangular∣⟨p2⟩2ph+(t)\Delta E(t) = \frac{\kappa}{2m_e}|\mathcal{I}_{\rm angular}|\langle p^2\rangle_{2p} h_+(t)
# ΔE(t)=2me​κ​∣Iangular​∣⟨p2⟩2p​h+​(t), and plot. This connects your theoretical derivation to a real gravitational wave event.

# Load GW150914 — 32 second window around merger
strain = TimeSeries.fetch_open_data('H1', 1126259446, 1126259478)

# Physical constants in SI
G = 6.674e-11
hbar = 1.055e-34
c = 3e8
m_e = 9.109e-31
alpha = 1/137.036
eV = 1.602e-19

kappa_SI = np.sqrt(32*np.pi*G) / c**2  # units: 1/(kg·m)
p2_2p_SI = 2 * m_e * (alpha**2 * m_e * c**2 / 8) / c**2  # SI: kg²·m²/s²
ang = 2/5

# Energy shift in eV
delta_E_J = (kappa_SI/(2*m_e)) * np.abs(strain.value) * p2_2p_SI * ang
delta_E_eV = delta_E_J / eV

t = strain.times.value - 1126259462  # seconds relative to merger

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(t, strain.value, color='royalblue', lw=0.8)
axes[0].set_ylabel('Strain $h_+(t)$')
axes[0].set_title('GW150914 — LIGO H1')
axes[1].plot(t, delta_E_eV, color='crimson', lw=0.8)
axes[1].set_ylabel(r'$\Delta E$ (eV)')
axes[1].set_xlabel('Time relative to merger (s)')
axes[1].set_title(r'Kinetic strain energy shift on H $2p$ state')
plt.tight_layout()
plt.savefig('GW150914_hydrogen_shift.pdf', dpi=200)


# Code 3 — Tobar ∣β(t)∣2|\beta(t)|^2
# ∣β(t)∣2 and P0→1(t)P_{0\to1}(t)
# P0→1​(t) for GW150914 (40 lines)

# --------------------------------------------------------
# --------------------------------------------------------

# Tobar displacement parameter for aluminium bar
M = 1800.0      # kg
v_s = 5100.0    # m/s (speed of sound in Al)
omega_r = 2*np.pi * 150  # resonant frequency Hz

# Compute chi(t) = integral of ddot_h * e^{i omega_r t}
dt = strain.dt.value
h_array = strain.value
t_array = strain.times.value - strain.times.value[0]

# Second derivative of strain
ddot_h = np.gradient(np.gradient(h_array, dt), dt)

# Numerical integration for beta
integrand_beta = ddot_h * np.exp(1j * omega_r * t_array)
chi = np.cumsum(integrand_beta) * dt

L = np.pi * v_s / omega_r  # bar length for fundamental mode
prefactor = (L / np.pi**2) * np.sqrt(M / (hbar * omega_r))
beta = prefactor * chi

P_01 = np.exp(-np.abs(beta)**2) * np.abs(beta)**2

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(t_array, np.abs(beta)**2, color='forestgreen')
axes[0].set_ylabel(r'$|\beta(t)|^2$')
axes[1].plot(t_array, P_01, color='darkorange')
axes[1].set_ylabel(r'$P_{0\to1}(t)$')
axes[1].set_xlabel('Time (s)')
plt.tight_layout()
plt.savefig('Tobar_GW150914.pdf', dpi=200)


# ------------------------------------------------------------------

omega_range = np.logspace(1, 4, 200) * 2 * np.pi  # 10 Hz to 10 kHz
M_opt = np.zeros_like(omega_range)

for i, omega in enumerate(omega_range):
    ddot_h_gw170817 = ...  # load GW170817 strain similarly
    chi_val = np.abs(np.trapz(ddot_h_gw170817 * np.exp(1j*omega*t), t))
    v_s_Be = 12900  # beryllium
    L = np.pi * v_s_Be / omega
    M_opt[i] = (np.pi**2 * hbar * omega**3) / (v_s_Be**2 * chi_val**2)

plt.loglog(omega_range/(2*np.pi), M_opt)
plt.xlabel('Resonant frequency (Hz)')
plt.ylabel('Optimal mass $M_{\\rm opt}$ (kg)')
plt.title('Optimal detector mass for GW170817')
plt.axhline(15, linestyle='--', label='Tobar: 15 kg at 100 Hz')
plt.legend()