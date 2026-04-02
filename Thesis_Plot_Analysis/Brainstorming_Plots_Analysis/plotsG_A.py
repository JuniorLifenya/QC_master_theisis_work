import numpy as np
import plotly.graph_objects as go

# --- 1. Physical Parameters ---
h0 = 1.0           # GW Strain amplitude
w = 2.0            # Beam waist (transverse width of the GW)
k = 2 * np.pi / 5  # Wavenumber (lambda = 5)

# Electron momentum vector p = (px, py, pz)
px, py, pz = 1.0, 1.0, 0.0 

# --- 2. Spatial Grid Setup ---
# Lower resolution is used for Cone plots to prevent vectors from visually overlapping into a solid block
grid_res = 12 
x = np.linspace(-3, 3, grid_res)
y = np.linspace(-3, 3, grid_res)
# Increase Z resolution slightly to capture the wave cycles smoothly
z = np.linspace(0, 10, int(grid_res * 1.5)) 
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# --- 3. Compute the Effective Gravitomagnetic Field (B_g) ---
# Radius squared for the Gaussian envelope
R2 = X**2 + Y**2
envelope = np.exp(-R2 / w**2)

# Transverse components: Alternating shear driven by the wave's longitudinal propagation
Bx = -k * h0 * py * envelope * np.sin(k * Z)
By = -k * h0 * px * envelope * np.sin(k * Z)

# Longitudinal component: The transverse gradient that creates the 3D vortices
Bz = (2 * h0 / w**2) * envelope * np.cos(k * Z) * (X * py + Y * px)

# Calculate magnitude to filter out extremely weak vectors (cleans up the plot)
B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
threshold = np.max(B_mag) * 0.15 # Hide vectors weaker than 15% of max strength

# Mask out weak vectors to make the vortex structure clearly visible
Bx[B_mag < threshold] = np.nan
By[B_mag < threshold] = np.nan
Bz[B_mag < threshold] = np.nan

# --- 4. 3D Vector Visualization ---
# go.Cone is mathematically bulletproof compared to Streamtube.
fig = go.Figure(data=go.Cone(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    u=Bx.flatten(),
    v=By.flatten(),
    w=Bz.flatten(),
    colorscale='Plasma',
    sizemode="scaled", # Scales the cones relative to the vector magnitude
    sizeref=0.5,       # Adjust this to change global arrow size
    showscale=True,
    colorbar=dict(title="|B_g| Magnitude")
))

# --- 5. Layout and Rigorous Formatting ---
fig.update_layout(
    title=dict(
        text="<b>Gravitomagnetic Vector Field (B_g)</b><br><sup>Visualizing the alternating shear and vortex layers driven by a Gaussian GW</sup>",
        x=0.5
    ),
    scene=dict(
        xaxis_title='x (Transverse)',
        yaxis_title='y (Transverse)',
        zaxis_title='z (Propagation Direction)',
        aspectratio=dict(x=1, y=1, z=2), 
        camera=dict(
            eye=dict(x=1.8, y=1.8, z=0.8)
        )
    ),
    template='plotly_dark',
    margin=dict(l=0, r=0, b=0, t=60) # Tighter margins for better viewing
)

# Render the plot
fig.show()