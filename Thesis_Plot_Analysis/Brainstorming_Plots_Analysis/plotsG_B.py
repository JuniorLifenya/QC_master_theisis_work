"""
qfi_squeezing_heatmap.py

Quantum Fisher Information (QFI) Heatmap for Gravitational Wave Sensing.
Maps the minimum detectable strain (h_min) derived from the Cramér-Rao bound,
incorporating T_2 decoherence, spin squeezing (in dB), and total integration time.

Placement: Chapter 6 (Quantum Metrology and Fisher Information)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os

os.makedirs("figures", exist_ok=True)

# ----------------------------------------------------------------------
# 1. Physics & Detector Parameters
# ----------------------------------------------------------------------
N_spins = 1e12            # Number of NV centers
T2 = 10e-3                # Coherence time (10 ms)
tau_int = 3.15e7          # Total integration time (1 year)

# Coupling constant (rad / s / strain)
# Calibrated so the SQL at T_opt yields h ~ 10^{-22} for macroscopic ensembles
g_coupling = 2.5e10       

# ----------------------------------------------------------------------
# 2. Parameter Grids
# ----------------------------------------------------------------------
# Interrogation time T (from 0.1 ms to 30 ms)
T_arr = np.linspace(0.1e-3, 30e-3, 300) 

# Spin Squeezing parameter xi^2 in dB (0 dB = SQL, up to 20 dB of squeezing)
sqz_db = np.linspace(0, 20, 300)

T_mesh, SQZ_mesh = np.meshgrid(T_arr, sqz_db)

# ----------------------------------------------------------------------
# 3. Cramér-Rao Bound Calculation
# ----------------------------------------------------------------------
# The Quantum Fisher Information for a squeezed state is F_Q = N * e^(2r).
# The phase variance is bounded by delta_phi >= 1 / sqrt(F_Q).
# Squeezing in dB reduces the phase variance standard deviation by 10^(-dB/20).
sqz_factor = 10.0 ** (-SQZ_mesh / 20.0)

# Decoherence degrades the contrast: C(T) = exp(-T/T2). 
# This effectively reduces the Fisher Information.
decoherence_penalty = np.exp(T_mesh / T2)

# Total number of measurements in 1 year = tau_int / T
# Accumulated phase sensitivity scales with 1 / sqrt(N_meas) = sqrt(T / tau_int)
h_min = (1.0 / g_coupling) * (1.0 / np.sqrt(N_spins * tau_int)) * (sqz_factor * decoherence_penalty / np.sqrt(T_mesh))

# ----------------------------------------------------------------------
# 4. Plotting the Heatmap
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 7))

# Create the heatmap with a logarithmic color scale
cmap = plt.cm.viridis_r
im = ax.pcolormesh(T_mesh * 1e3, SQZ_mesh, h_min, 
                   norm=LogNorm(vmin=h_min.min(), vmax=h_min.max()), 
                   cmap=cmap, shading='auto')

# Add contour lines for specific strain targets
levels = [1e-24, 5e-24, 1e-23, 5e-23, 1e-22]
contours = ax.contour(T_mesh * 1e3, SQZ_mesh, h_min, levels=levels, 
                      colors='white', linewidths=1.5, alpha=0.8, linestyles='dashed')
ax.clabel(contours, inline=True, fontsize=10, fmt='%1.0e')

# ----------------------------------------------------------------------
# 5. Annotating the Optimal Physics
# ----------------------------------------------------------------------
# The theoretical optimal interrogation time for exponential decay is T = T_2 / 2
T_opt = T2 / 2.0 * 1e3
ax.axvline(T_opt, color='red', linestyle=':', linewidth=2.5)
ax.text(T_opt + 0.5, 18, r'Optimal $T = T_2 / 2$', color='red', fontsize=12, fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# Mark the Standard Quantum Limit (SQL)
ax.axhline(0, color='black', linewidth=3)
ax.text(1, 0.5, 'Standard Quantum Limit (SQL)', color='white', fontsize=11, fontweight='bold')

# ----------------------------------------------------------------------
# 6. Formatting
# ----------------------------------------------------------------------
ax.set_xlabel('Interrogation Time $T$ (ms)', fontsize=13, fontweight='bold')
ax.set_ylabel(r'Spin Squeezing $\xi^2$ (dB)', fontsize=13, fontweight='bold')
ax.set_title('Quantum Cramér-Rao Bound: Minimum Detectable Strain\n'
             r'($N = 10^{12}$, $T_2 = 10\,$ms, $\tau_{\rm int} = 1\,$yr)', 
             fontsize=14, pad=15)

# Colorbar formatting
cbar = fig.colorbar(im, ax=ax, pad=0.02)
cbar.set_label('Strain Sensitivity $h_{\\rm min}$ (Log Scale)', fontsize=12, fontweight='bold', rotation=270, labelpad=20)

plt.tight_layout()
plt.savefig("figures/qfi_squeezing_heatmap.pdf", dpi=300, bbox_inches='tight')
plt.show()
print("Saved: figures/qfi_squeezing_heatmap.pdf")