import numpy as np
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

# --- 1. TT Gauge Calculation ---
# |E_2| for Hydrogen in Joules
E_2_joules = (m_e * c**2 * alpha**2) / 8.0
E_2_eV = E_2_joules / eV

Delta_E_TT_eV = (2.0 / 5.0) * h_0 * E_2_eV

# --- 2. FNC (LIF) Gauge Calculation ---
# V_LIF = 3 * m_e * a_0^2 * omega_gw^2 * h_0 (in Joules)
Delta_E_LIF_joules = 3.0 * m_e * (a_0**2) * (omega_gw**2) * h_0
Delta_E_LIF_eV = Delta_E_LIF_joules / eV

print(f"--- RIGOROUS NUMERICAL VERIFICATION FOR THESIS ---")
print(f"Bohr Radius (a_0): {a_0:.6e} m")
print(f"|E_2| Energy Scale: {E_2_eV:.6f} eV")
print(f"--------------------------------------------------")
print(f"Delta E (TT Gauge) : {Delta_E_TT_eV:.6e} eV")
print(f"Delta E (FNC Gauge): {Delta_E_LIF_eV:.6e} eV")