import numpy as np
import plotly.graph_objects as go
import os

os.makedirs("figures", exist_ok=True)

# Physical constants (SI)
hbar = 1.054571817e-34
m_e = 9.10938356e-31
c = 2.99792458e8
G = 6.67430e-11
kappa = np.sqrt(32 * np.pi * G)  # m²/kg
a_NV = 0.5e-9  # confinement length
p2 = (hbar / a_NV)**2
gamma = (kappa * p2) / (2 * m_e * hbar)   # rad/s per unit h
T2_star = 100e-6  # s
tau = 3.156e7  # 1 year

# Frequency range (Hz)
f = np.logspace(0, 6, 100)
N_spins = np.logspace(6, 14, 100)  # number of NV centers
F, N = np.meshgrid(f, N_spins)

# h_min formula: h_min = 1 / (gamma * sqrt(tau * T2_star)) * (1 / sqrt(N))  (frequency-independent)
# Actually h_min ∝ 1/(gamma * sqrt(tau * T2_star * N))
h_min = 1.0 / (gamma * np.sqrt(tau * T2_star * N))

# Create 3D surface
fig = go.Figure(data=[
    go.Surface(
        x=F, y=N, z=np.log10(h_min),
        colorscale='Viridis',
        colorbar=dict(title='log10(h_min)'),
        contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project=dict(z=True)))
    )
])

fig.update_layout(
    title='Minimum detectable strain as a function of frequency and number of NV centers',
    scene=dict(
        xaxis_title='Frequency (Hz)',
        yaxis_title='Number of NV centers',
        zaxis_title='log10(h_min)',
        xaxis_type='log',
        yaxis_type='log',
        xaxis=dict(tickvals=[1, 1e2, 1e4, 1e6], ticktext=['1', '100', '10k', '1M']),
        yaxis=dict(tickvals=[1e6, 1e8, 1e10, 1e12, 1e14], ticktext=['1e6', '1e8', '1e10', '1e12', '1e14'])
    ),
    width=1000,
    height=800,
    template='plotly_dark'
)

fig.write_html('figures/sensitivity_terrain.html')
fig.write_image('figures/sensitivity_terrain.png')
print("Saved: figures/sensitivity_terrain.html and .png")