import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

os.makedirs("Thesis_Ready_Plots", exist_ok=True)

# ----- GW surface: z = A (x^2 - y^2) -----
A = 0.3  # strain amplitude (visual, not physical)

# ----- Grid for surface -----
span = 2.0
N = 100
x = np.linspace(-span, span, N)
y = np.linspace(-span, span, N)
X, Y = np.meshgrid(x, y)
Z = A * (X**2 - Y**2)

# ----- Circular path in coordinate space -----
R_circle = 1.2
theta = np.linspace(0, 2*np.pi, 300)
x_c = R_circle * np.cos(theta)
y_c = R_circle * np.sin(theta)
z_c = A * (x_c**2 - y_c**2)      # path on the surface

# ----- Compute tangent vectors to the path -----
dx = np.gradient(x_c)
dy = np.gradient(y_c)
# derivative of z: dz/ds = 2A (x * dx/ds - y * dy/ds)
dz = 2*A * (x_c * dx - y_c * dy)
# normalize to get direction of velocity (coordinate speed)
ds = np.sqrt(dx**2 + dy**2 + dz**2)
dir_x = dx / ds
dir_y = dy / ds
dir_z = dz / ds

# ----- Momentum magnitude: model as varying because of strain -----
# For a free particle on a curved surface, kinetic energy varies.
# We invent a simple "potential" V_eff = h_ij p_0^i p_0^j / (2m) but we want
# visual variation. We'll make momentum length ∝ 1 + 0.5 * (Z/max(|Z|)) so that
# it's larger in valleys (negative Z) and smaller on hills (positive Z).
Z_rel = z_c / np.max(np.abs(z_c))
p_mag = 1.0 + 0.5 * Z_rel   # dimensionless; adjust for visual appeal

# ----- Sample points for arrows (every 20th point) -----
step = 20
idx = np.arange(0, len(x_c), step)
# If we want a closed loop, ensure the first and last are the same (theta wraps)
if idx[-1] != len(x_c)-1:
    idx = np.append(idx, len(x_c)-1)

# ----- 3D figure -----
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Surface
surf = ax.plot_surface(X, Y, Z, alpha=0.5, cmap='coolwarm', linewidth=0, antialiased=True)
# Wireframe for clarity
ax.plot_wireframe(X, Y, Z, color='gray', alpha=0.3, linewidth=0.5, rstride=8, cstride=8)

# Path
ax.plot(x_c, y_c, z_c, color='k', linewidth=2, label='electron trajectory')

# Momentum arrows
for i in idx:
    ax.quiver(x_c[i], y_c[i], z_c[i],
              dir_x[i] * p_mag[i], dir_y[i] * p_mag[i], dir_z[i] * p_mag[i],
              color='darkorange', linewidth=2, arrow_length_ratio=0.15,
              zorder=10)

# Decorate
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z (strain height)')
ax.set_title('Electron on a Gravitational Wave Saddle\nMomentum changes over hills & valleys', fontweight='bold')
ax.legend()
ax.view_init(elev=30, azim=-60)

plt.tight_layout()
plt.savefig("Thesis_Ready_Plots/electron_GW_saddle.png", dpi=300, bbox_inches='tight')
plt.show()
print("Saved: Thesis_Ready_Plots/electron_GW_saddle.png")