import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.constants as const

# --- Physical Constants ---
c = const.c            # Speed of light (m/s)
hbar = const.hbar      # Reduced Planck constant (J*s)
m_e = const.m_e        # Electron mass (kg)
eV = const.eV          # 1 eV in Joules
alpha = const.alpha    # Fine structure constant (~1/137)

# Bohr radius (meters)
a_0 = hbar / (m_e * c * alpha) 

# --- Input Parameters ---
h_0 = 1e-21            # Dimensionless GW strain
f_gw = 100.0           # GW frequency in Hz
omega_gw = 2 * np.pi * f_gw

theta = np.linspace(0, np.pi, 50)
phi = np.linspace(0, 2*np.pi, 50)
theta, phi = np.meshgrid(theta, phi)
R = np.abs(np.sin(theta)**2 * np.cos(2*phi))
X = R * np.sin(theta) * np.cos(phi)
Y = R * np.sin(theta) * np.sin(phi)
Z = R * np.cos(theta)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, facecolors=plt.cm.jet(R/R.max()), rstride=1, cstride=1, alpha=0.8)
ax.set_title("GW quadrupole antenna pattern $\\propto \\sin^4\\theta$")
plt.show()

V = (2/5) * 3.4 * 1e-21  # eV
hbar = 6.582e-16          # eV·s
Omega_R = V / hbar         # Rabi frequency in Hz? Actually rad/s: Omega_R = V/hbar (units 1/s)
t = np.linspace(0, 10/Omega_R, 1000)
P_plus = np.sin(Omega_R * t)**2

plt.plot(t, P_plus)
plt.xlabel('Time (s)')
plt.ylabel('Population of $|2p,+1\\rangle$')
plt.title('Rabi oscillations driven by GW strain $h_+=10^{-21}$')
plt.show()

freqs = np.logspace(-2, 4, 200)  # Hz
omega = 2*np.pi*freqs
Delta_E_FNC = 3 * m_e * a_0**2 * omega**2 * h_0  # J
Delta_E_FNC_eV = Delta_E_FNC / eV
plt.loglog(freqs, Delta_E_FNC_eV, 'b')
plt.axhline(1e-9, color='r', linestyle='--', label='1 neV')
plt.xlabel('GW frequency (Hz)')
plt.ylabel('$\\Delta E$ (eV)')
plt.legend()
plt.grid(True, which='both')
plt.title('Tidal energy splitting of hydrogen 2p state')
plt.show()

M_vals = np.logspace(-3, 3, 100)  # kg
L = lambda M: (M/7800)**(1/3)  # approximate length for steel
omega_bar = 2*np.pi*1e3  # 1 kHz
beta = np.sqrt(M_vals) * L(M_vals) * omega_bar**(3/2) * h_0 / (np.pi**2 * np.sqrt(6.58e-16))
P1 = np.exp(-beta**2) * beta**2
plt.loglog(M_vals, P1, 'm')
plt.xlabel('Bar mass (kg)')
plt.ylabel('Single‑phonon probability')
plt.grid()
plt.title('GW‑induced single‑phonon transition (Tobar protocol)')