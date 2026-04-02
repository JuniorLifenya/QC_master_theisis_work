"""
selection_rule_plotly.py

GPU-Accelerated visual proof of the Wigner-Eckart selection rules.
Uses Plotly for instant loading and 60FPS interactive 3D orbiting.
Generates an interactive HTML file and can be screenshot for the thesis.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Create figures directory
output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)

# ------------------------------------------------------------
# 1. High-Resolution Grid (Plotly handles this effortlessly)
# ------------------------------------------------------------
n_theta = 150
n_phi = 150
theta_1d = np.linspace(0, np.pi, n_theta)
phi_1d   = np.linspace(0, 2 * np.pi, n_phi)
theta, phi = np.meshgrid(theta_1d, phi_1d, indexing='ij')

# ------------------------------------------------------------
# 2. Real spherical harmonics (normalised)
# ------------------------------------------------------------
def Y_s(theta, phi):
    """l=0, m=0: isotropic s-orbital."""
    return np.sqrt(1.0 / (4.0 * np.pi)) * np.ones_like(theta)

def Y_d_x2y2(theta, phi):
    """l=2, m=+2 (real): transforms as h_+."""
    return np.sqrt(15.0 / (16.0 * np.pi)) * (np.sin(theta)**2) * np.cos(2 * phi)

def Y_d_xy(theta, phi):
    """l=2, m=-2 (real): transforms as h_x."""
    return np.sqrt(15.0 / (16.0 * np.pi)) * (np.sin(theta)**2) * np.sin(2 * phi)

operator_hplus = Y_d_x2y2(theta, phi)

# ------------------------------------------------------------
# 3. Define states & Integrands
# ------------------------------------------------------------
psi_i = Y_s(theta, phi)
psi_f_allowed = Y_d_x2y2(theta, phi)
psi_f_forbidden = Y_d_xy(theta, phi)

integrand_allowed   = psi_f_allowed * operator_hplus * psi_i
integrand_forbidden = psi_f_forbidden * operator_hplus * psi_i

# ------------------------------------------------------------
# 4. Helper: Create Plotly 3D Surface
# ------------------------------------------------------------
def create_surface(func, showscale=False):
    """Creates a 3D surface mesh where radius = |func| and color = sign."""
    R = np.abs(func)
    X = R * np.sin(theta) * np.cos(phi)
    Y = R * np.sin(theta) * np.sin(phi)
    Z = R * np.cos(theta)
    
    # Map phase: negative to -1, positive to +1
    phase = np.sign(func + 1e-12)
    
    return go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=phase,
        colorscale='RdBu',      # Red for positive, Blue for negative
        cmin=-1, cmax=1,
        showscale=showscale,
        colorbar=dict(title="Sign", tickvals=[-1, 0, 1], ticktext=["Negative", "Zero", "Positive"], len=0.5) if showscale else None,
        lighting=dict(ambient=0.6, diffuse=0.5, roughness=0.5, specular=0.1),
        hoverinfo='skip'
    )

# ------------------------------------------------------------
# 5. Build the 2x4 Subplot Dashboard
# ------------------------------------------------------------
fig = make_subplots(
    rows=2, cols=4,
    specs=[[{'type': 'surface'}] * 4, [{'type': 'surface'}] * 4],
    subplot_titles=(
        "Initial |s⟩ (l=0)", "Operator h_+ (l=2)", "Final ⟨d_x²-y²| (l=2)", "Allowed Integrand",
        "Initial |s⟩ (l=0)", "Operator h_+ (l=2)", "Final ⟨d_xy| (l=2)", "Forbidden Integrand"
    ),
    horizontal_spacing=0.02,
    vertical_spacing=0.08
)

# Row 1: Allowed
fig.add_trace(create_surface(psi_i), row=1, col=1)
fig.add_trace(create_surface(operator_hplus), row=1, col=2)
fig.add_trace(create_surface(psi_f_allowed), row=1, col=3)
fig.add_trace(create_surface(integrand_allowed, showscale=True), row=1, col=4)

# Row 2: Forbidden
fig.add_trace(create_surface(psi_i), row=2, col=1)
fig.add_trace(create_surface(operator_hplus), row=2, col=2)
fig.add_trace(create_surface(psi_f_forbidden), row=2, col=3)
fig.add_trace(create_surface(integrand_forbidden), row=2, col=4)

# ------------------------------------------------------------
# 6. Formatting and Camera Tuning
# ------------------------------------------------------------
# Remove grid lines and axes for a clean, professional look
axis_args = dict(showgrid=False, zeroline=False, showticklabels=False, title='', showbackground=False)
scene_layout = dict(xaxis=axis_args, yaxis=axis_args, zaxis=axis_args, 
                    camera=dict(eye=dict(x=1.2, y=1.2, z=0.8))) # Default viewing angle

fig.update_layout(
    title_text="Wigner-Eckart Selection Rules: Overlap Integrals for Gravitational Quadrupole Operator",
    title_x=0.5,
    title_font_size=20,
    height=800,
    width=1400,
    margin=dict(l=10, r=10, b=10, t=80)
)

# Apply the clean scene layout to all 8 subplots
for i in range(1, 9):
    fig.layout[f'scene{i if i > 1 else ""}'].update(scene_layout)

# Save as an interactive HTML file
html_path = os.path.join(output_dir, "selection_rule_interactive.html")
fig.write_html(html_path)

print(f"Saved highly-optimized interactive plot to: {html_path}")
print("Opening in your web browser...")

# Automatically open it in your browser
fig.show()