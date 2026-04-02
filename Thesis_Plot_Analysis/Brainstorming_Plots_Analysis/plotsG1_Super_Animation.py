
"""
tetrad_animation_optimized.py

Animates the 3D deformation of a global coordinate grid under a gravitational wave,
while keeping a rigid local tetrad (e_1, e_2, e_3) fixed at the origin.

Upgrades: 
- Dynamic color mapping (Blue -> Red) based on GW phase.
- Massive performance boost by updating Line3D objects rather than clearing the axis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# ----------------------------------------------------------------------
# 1. Parameters
# ----------------------------------------------------------------------
h_amp = 0.5          # GW amplitude (exaggerated for visibility)
omega = 2.0 * np.pi  # angular frequency (cycles per unit time)
pol = 'plus'         # choose 'plus' or 'cross'

# Grid: 5x5x5 points in a cube [-1,1]^3
x = np.linspace(-1, 1, 5)
y = np.linspace(-1, 1, 5)
z = np.linspace(-1, 1, 3)   # thinner in z to avoid clutter
X0, Y0, Z0 = np.meshgrid(x, y, z, indexing='ij')
nx, ny, nz = X0.shape

# Flatten for vectorised computation
points = np.vstack([X0.ravel(), Y0.ravel(), Z0.ravel()]).T   # N x 3

# ----------------------------------------------------------------------
# 2. Deformation function (TT gauge, linearised)
# ----------------------------------------------------------------------
def deform_points(points, phase, pol):
    cos_phase = np.cos(phase)
    x0, y0, z0 = points[:,0], points[:,1], points[:,2]
    if pol == 'plus':
        dx = 0.5 * h_amp * cos_phase * x0
        dy = -0.5 * h_amp * cos_phase * y0
        dz = np.zeros_like(x0)
    elif pol == 'cross':
        dx = 0.5 * h_amp * cos_phase * y0
        dy = 0.5 * h_amp * cos_phase * x0
        dz = np.zeros_like(x0)
    else:
        raise ValueError("pol must be 'plus' or 'cross'")
    return np.vstack([x0 + dx, y0 + dy, z0 + dz]).T

# ----------------------------------------------------------------------
# 3. Setup the static elements
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the rigid tetrad at the origin (STAYS STATIC)
length = 1.2
ax.quiver(0,0,0, length,0,0, color='k', arrow_length_ratio=0.15, lw=2.5)
ax.text(length+0.1,0,0, r'$e_{\hat{1}}$', fontsize=12, fontweight='bold')
ax.quiver(0,0,0, 0,length,0, color='k', arrow_length_ratio=0.15, lw=2.5)
ax.text(0,length+0.1,0, r'$e_{\hat{2}}$', fontsize=12, fontweight='bold')
ax.quiver(0,0,0, 0,0,length, color='k', arrow_length_ratio=0.15, lw=2.5)
ax.text(0,0,length+0.1, r'$e_{\hat{3}}$', fontsize=12, fontweight='bold')

# Helper to initially draw and store line objects
def init_wireframe(ax, points, shape, color, alpha, lw, label=None):
    lines = []
    X = points[:,0].reshape(shape)
    Y = points[:,1].reshape(shape)
    Z = points[:,2].reshape(shape)

    # We store the lines with a tag identifying their orientation
    for j in range(ny):
        for k in range(nz):
            line, = ax.plot(X[:,j,k], Y[:,j,k], Z[:,j,k], color=color, alpha=alpha, lw=lw, label=label if j==0 and k==0 else "")
            lines.append(('x', j, k, line))
    for i in range(nx):
        for k in range(nz):
            line, = ax.plot(X[i,:,k], Y[i,:,k], Z[i,:,k], color=color, alpha=alpha, lw=lw)
            lines.append(('y', i, k, line))
    for i in range(nx):
        for j in range(ny):
            line, = ax.plot(X[i,j,:], Y[i,j,:], Z[i,j,:], color=color, alpha=alpha, lw=lw)
            lines.append(('z', i, j, line))
    return lines

# Draw un-deformed grid (STAYS STATIC)
init_wireframe(ax, points, X0.shape, color='gray', alpha=0.3, lw=1.0, label='Global grid')

# Initialize empty lines for the deformed grid
deformed_lines = init_wireframe(ax, points, X0.shape, color='steelblue', alpha=0.8, lw=1.5)

# Styling and fixed axes (so it doesn't jump around)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1.5, 1.5)
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
title = ax.set_title(f'GW polarisation: {pol}')
ax.view_init(elev=20, azim=35)

# ----------------------------------------------------------------------
# 4. The Highly Efficient Animation Loop
# ----------------------------------------------------------------------
def update(phase):
    # Calculate new point positions
    deformed_pts = deform_points(points, phase, pol)
    X = deformed_pts[:,0].reshape(nx, ny, nz)
    Y = deformed_pts[:,1].reshape(nx, ny, nz)
    Z = deformed_pts[:,2].reshape(nx, ny, nz)

    # Dynamic Color: Map phase to a smooth oscillation between 0 (Blue) and 1 (Red)
    # np.cos(phase) aligns perfectly with the strain magnitude
    color_val = (np.cos(phase) + 1.0) / 2.0  
    current_color = plt.cm.coolwarm(color_val)

    # Update the data inside existing lines instead of clearing the plot
    for axis, a, b, line in deformed_lines:
        if axis == 'x':
            line.set_data_3d(X[:, a, b], Y[:, a, b], Z[:, a, b])
        elif axis == 'y':
            line.set_data_3d(X[a, :, b], Y[a, :, b], Z[a, :, b])
        elif axis == 'z':
            line.set_data_3d(X[a, b, :], Y[a, b, :], Z[a, b, :])
        
        # Apply the dynamic color
        line.set_color(current_color)

    title.set_text(f'GW polarisation: {pol} (phase = {phase:.2f})')
    
    # Return the updated artists (speeds up blitting if supported)
    return [line for _, _, _, line in deformed_lines] + [title]

# ----------------------------------------------------------------------
# 5. Export
# ----------------------------------------------------------------------
phases = np.linspace(0, 2*np.pi, 60)
ani = animation.FuncAnimation(fig, update, frames=phases, interval=50, blit=False)

os.makedirs("figures", exist_ok=True)

# Generate GIF
gif_path = f'figures/tetrad_animation_{pol}_optimized.gif'
ani.save(gif_path, writer='pillow', fps=20)
print(f"Saved: {gif_path}")

# Static frame for printed thesis
update(0.0)
png_path = f'figures/tetrad_deformation_{pol}_static.png'
plt.savefig(png_path, dpi=300, bbox_inches='tight')
print(f"Saved: {png_path}")
plt.show()