"""
fig_ch2_spin_connection_viz.py
Chapter 2 — Visualizing the Spin Connection via Parallel Transport.
Shows a native tetrad at Point A, its parallel transport to Point B,
and how it mismatches with the native tetrad at Point B.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Ensure output directory exists
os.makedirs("figures", exist_ok=True)

# ---------------------------------------------------------
# 1. Manifold Definition (A curved saddle surface)
# ---------------------------------------------------------
def r_vec(x, y):
    """Position vector on the curved 2D manifold embedded in 3D."""
    # Using a saddle geometry to clearly illustrate spatial curvature
    return np.array([x, y, 0.6 * x * y])

def dr_dx(x, y):
    """Tangent vector along coordinate x."""
    return np.array([1, 0, 0.6 * y])

def dr_dy(x, y):
    """Tangent vector along coordinate y."""
    return np.array([0, 1, 0.6 * x])

# ---------------------------------------------------------
# 2. Tetrad Construction (Gram-Schmidt Orthonormalization)
# ---------------------------------------------------------
def get_native_tetrad(x, y):
    """
    Calculates the orthonormal tetrad (e_1, e_2, e_3) at a given point.
    e_1, e_2 span the tangent plane; e_3 is the normal vector.
    """
    rx = dr_dx(x, y)
    ry = dr_dy(x, y)
    
    # Normal vector e_3
    n = np.cross(rx, ry)
    e3 = n / np.linalg.norm(n)
    
    # Orthonormal tangent vectors e_1 and e_2
    e1 = rx / np.linalg.norm(rx)
    e2 = np.cross(e3, e1) # Guarantees strict right-handed orthonormality
    
    return e1, e2, e3

# ---------------------------------------------------------
# 3. Parallel Transport Approximation
# ---------------------------------------------------------
def parallel_transport(V_A, n_B):
    """
    Simulates Levi-Civita parallel transport of a tangent vector V_A 
    to a new point B by projecting it onto the new tangent plane at B 
    and re-normalizing.
    """
    # Project out the normal component at B to keep the vector in the tangent plane
    V_B_proj = V_A - np.dot(V_A, n_B) * n_B
    return V_B_proj / np.linalg.norm(V_B_proj)

# ---------------------------------------------------------
# 4. Plotting Setup
# ---------------------------------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
fig.suptitle('The Spin Connection: Parallel Transport vs. Local Frame', fontsize=14, fontweight='bold')
ax.set_title(r'The mismatch angle at Point B represents $\omega_{\mu}^{ab} dx^\mu$', fontsize=12, pad=10)

# Generate the Curved Grid (Manifold)
pts = np.linspace(-1.5, 1.5, 15)
X, Y = np.meshgrid(pts, pts)
Z = 0.6 * X * Y
ax.plot_wireframe(X, Y, Z, color='black', alpha=0.15, linewidth=0.8)
ax.plot_surface(X, Y, Z, color='whitesmoke', alpha=0.4, edgecolor='none')

# Define Points A and B
A = np.array([-0.8, -0.8])
B = np.array([0.8, 0.8])

pos_A = r_vec(*A)
pos_B = r_vec(*B)

# Get Native Tetrads
e1_A, e2_A, e3_A = get_native_tetrad(*A)
e1_B, e2_B, e3_B = get_native_tetrad(*B)

# Parallel Transport Tetrad A to Point B
# We transport the tangent vectors e1 and e2. e3 simply aligns with the new normal.
e1_trans = parallel_transport(e1_A, e3_B)
e2_trans = parallel_transport(e2_A, e3_B)

# ---------------------------------------------------------
# 5. Drawing the Geometry
# ---------------------------------------------------------
L = 0.6  # Quiver length scaling

def draw_tetrad(ax, pos, e1, e2, e3, ls='-', alpha=1.0, label_suffix=""):
    """Helper to draw a tetrad at a specific position."""
    ax.quiver(*pos, *(e1 * L), color='red', linestyle=ls, alpha=alpha, arrow_length_ratio=0.15, lw=2.5, 
              label=rf'$e_{{1}}{label_suffix}$')
    ax.quiver(*pos, *(e2 * L), color='green', linestyle=ls, alpha=alpha, arrow_length_ratio=0.15, lw=2.5, 
              label=rf'$e_{{2}}{label_suffix}$')
    ax.quiver(*pos, *(e3 * L), color='royalblue', linestyle=ls, alpha=alpha, arrow_length_ratio=0.15, lw=2.5, 
              label=rf'$e_{{3}}{label_suffix}$')

# Draw Points and Geodesic Path (Approximate linear path in coordinate space)
t_vals = np.linspace(0, 1, 20)
path_X = A[0] + t_vals * (B[0] - A[0])
path_Y = A[1] + t_vals * (B[1] - A[1])
path_Z = 0.6 * path_X * path_Y
ax.plot(path_X, path_Y, path_Z, color='dimgray', linestyle=':', lw=2, label='Path $x^\mu(\lambda)$')

ax.scatter(*pos_A, color='black', s=50)
ax.text(pos_A[0], pos_A[1], pos_A[2] + 0.2, "Point A\nNative Tetrad", fontsize=10, fontweight='bold', ha='center')

ax.scatter(*pos_B, color='black', s=50)
ax.text(pos_B[0], pos_B[1], pos_B[2] + 0.3, "Point B", fontsize=10, fontweight='bold', ha='center')

# Draw Tetrads
# 1. Native Tetrad at A
draw_tetrad(ax, pos_A, e1_A, e2_A, e3_A, ls='-', label_suffix="(A)")

# 2. Native Tetrad at B
draw_tetrad(ax, pos_B, e1_B, e2_B, e3_B, ls='-', alpha=0.9, label_suffix="(B native)")

# 3. Transported Tetrad at B (Dashed)
ax.quiver(*pos_B, *(e1_trans * L), color='darkred', linestyle='--', alpha=0.8, arrow_length_ratio=0.15, lw=2.5, label=r'$\tilde{e}_1$ (Transported)')
ax.quiver(*pos_B, *(e2_trans * L), color='darkgreen', linestyle='--', alpha=0.8, arrow_length_ratio=0.15, lw=2.5, label=r'$\tilde{e}_2$ (Transported)')

# Adjust viewing angle and labels
ax.view_init(elev=28, azim=45)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1.0, 1.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# Create a clean legend
handles, labels = ax.get_legend_handles_labels()
# Filter unique labels to avoid clutter
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=9)

plt.tight_layout()
plt.savefig('figures/fig_ch2_spin_connection_viz.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: figures/fig_ch2_spin_connection_viz.png")

