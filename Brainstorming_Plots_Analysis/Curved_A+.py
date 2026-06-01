import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

os.makedirs("Thesis_Ready_Plots", exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

os.makedirs("Thesis_Ready_Plots", exist_ok=True)

# ----------------------------------------------------------------------
# 1. Surface: z = 0.5 (x² - y²)  (a quadrupole saddle)
# ----------------------------------------------------------------------
def surface(x, y):
    return np.array([x, y, 0.5 * (x**2 - y**2)])

def normal(x, y):
    """Outward unit normal to the surface."""
    dz_dx = x
    dz_dy = -y
    n = np.array([-dz_dx, -dz_dy, 1.0])
    return n / np.linalg.norm(n)

def tangent_basis(x, y):
    """Orthonormal tangent basis (e1, e2) at (x,y)."""
    rx = np.array([1.0, 0.0, x])
    ry = np.array([0.0, 1.0, -y])
    # Gram‑Schmidt
    e1 = rx / np.linalg.norm(rx)
    e2 = ry - np.dot(ry, e1) * e1
    e2 /= np.linalg.norm(e2)
    return e1, e2

# ----------------------------------------------------------------------
# 2. Parallel transport of a vector along a curve on the surface
#    using the Levi‑Civita connection (projection method)
# ----------------------------------------------------------------------
def parallel_transport_curve(curve_fun, t_span, V0, t_eval):
    """
    Parallel‑transport a tangent vector V0 along a surface curve.
    curve_fun(t) returns (x, y) parameterising the curve.
    t_span = (t0, t1)
    t_eval: array of t values where solution is stored.
    Returns: t, Vx, Vy, Vz arrays.
    """
    def ode(t, V):
        x, y = curve_fun(t)
        # point on surface
        p = surface(x, y)
        n = normal(x, y)
        # tangent vector of curve (d/dt)
        dx, dy = (curve_fun(t+1e-8)[0]-curve_fun(t-1e-8)[0])/(2e-8), \
                 (curve_fun(t+1e-8)[1]-curve_fun(t-1e-8)[1])/(2e-8)
        T = np.array([dx, dy, x*dx - y*dy])  # d/dt of surface
        # derivative of V in 3D (naive)
        # dV/dt along the curve is not needed; parallel transport condition:
        # dV/dt - (d/dt V · n) n = 0  →  dV/dt = (d/dt V · n) n
        # But d/dt V = (∂V/∂t) which we compute by finite differences
        # Better: use the fact that (d/dt V) must have no component along n,
        # so we project out the normal component.
        # For an ODE, we set dV/dt = proj_tangent( - (V·∇) T )? Not needed.
        # We use the condition dV/dt · n = 0 and the fact V stays tangent.
        # The covariant derivative along T is zero: D_T V = 0.
        # In the embedding, this is equivalent to dV/dt = (dT/dt · n) (V·n?) No.
        # Simplest robust method:
        # 1) Compute raw derivative dV/dt by transporting V as a constant 3D vector?
        #    No. Actually, parallel transport in the surface can be done by:
        #    V(t+dt) = parallel_propagate(V(t), T(t), n(t), dt) using projection.
        # We'll implement a step-wise integration with Runge‑Kutta.
        pass

    # I'll implement directly using the projection method in discrete steps
    # with RK4.
    def project(V, n):
        return V - np.dot(V, n) * n

    t = t_eval
    V = np.zeros((len(t), 3))
    V[0] = V0
    x0, y0 = curve_fun(t[0])
    # validate V0 is tangent
    n0 = normal(x0, y0)
    V[0] = project(V0, n0)
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        if dt == 0:
            V[i] = V[i-1]
            continue
        # RK4 step for parallel transport:
        # At current point (x_i-1, y_i-1) with vector V_i-1,
        # we want to transport along the path segment.
        # The geodesic equation in embedding: dV/dt = (dV/dt · n) n
        # but easier: use the fact that parallel transport in the surface
        # corresponds to transporting V as a constant vector in R^3 and then
        # repeatedly projecting onto the tangent plane of the new point.
        # This is exactly the "rolling without slipping" method.
        # For small dt, we can:
        # 1) Move to the new point p_new.
        # 2) Transport V as a constant 3D vector (naive parallel transport in R^3).
        # 3) Project onto the tangent plane at p_new.
        # This yields a first-order scheme. We'll use RK4 for better accuracy.
        # Higher-order: integrate the ODE dV/dt = (V_old·dn/dt) n + ... ? Not needed.
        # I'll use a simple RK4 on the ODE derived from the connection.
        # However, for the purpose of visualisation, a second-order method is fine.
        # Let's implement a 2nd order explicit parallel transport by successive
        # projection and re‑normalisation.
        def transport_step(V_curr, t_curr, t_next):
            x_c, y_c = curve_fun(t_curr)
            x_n, y_n = curve_fun(t_next)
            p_c = surface(x_c, y_c)
            p_n = surface(x_n, y_n)
            n_c = normal(x_c, y_c)
            n_n = normal(x_n, y_n)
            # Transport V_curr as a constant vector in ambient space to the new point
            V_ambient = V_curr  # no change
            # Project onto tangent plane at new point
            V_new = V_ambient - np.dot(V_ambient, n_n) * n_n
            # Ensure length preservation (projection reduces length slightly)
            V_new *= np.linalg.norm(V_curr) / np.linalg.norm(V_new)
            return V_new

        V[i] = transport_step(V[i-1], t[i-1], t[i])

    return t, V

# ----------------------------------------------------------------------
# 3. Define the path and parameters
# ----------------------------------------------------------------------
A = np.array([-0.8, -0.8])
B = np.array([0.8, 0.8])
t_span = (0, 1)
t_eval = np.linspace(0, 1, 31)

def curve(t):
    return A + t * (B - A)

# Initial spin vector: tangent vector aligned with e1 at A
x0, y0 = A
e1_A, e2_A = tangent_basis(x0, y0)
spin0 = e1_A.copy()

# Parallel transport along the curve
t_arr, spin_transported = parallel_transport_curve(curve, t_span, spin0, t_eval)

# ----------------------------------------------------------------------
# 4. Plotting
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
fig.suptitle("Spin Connection: Continuous Spin Precession along a Curve",
             fontsize=14, fontweight='bold')
ax.set_title(r"$\vec{S}(t)$ parallel‑transported, precession from $\omega_\mu{}^{ab}$",
             fontsize=11)

# Surface
pts = np.linspace(-1.2, 1.2, 30)
X, Y = np.meshgrid(pts, pts)
Z = 0.5 * (X**2 - Y**2)
ax.plot_surface(X, Y, Z, alpha=0.3, color='lightblue', edgecolor='gray', lw=0.2)

# Path
path_points = np.array([surface(*curve(t)) for t in t_arr])
ax.plot(path_points[:,0], path_points[:,1], path_points[:,2],
        'k--', lw=2, label='Trajectory')

# Show the transported spin vector at selected points along the path
L = 0.4
sample_idx = [0, 5, 10, 15, 20, 25, 30]  # sample points including A and B
for idx in sample_idx:
    t = t_arr[idx]
    x, y = curve(t)
    pos = surface(x, y)
    e1, e2 = tangent_basis(x, y)
    spin = spin_transported[idx]
    # local tetrad (semi‑transparent)
    ax.quiver(*pos, *(e1*L), color='gray', alpha=0.5, lw=0.8)
    ax.quiver(*pos, *(e2*L), color='gray', alpha=0.5, lw=0.8)
    # spin vector
    ax.quiver(*pos, *(spin*L*1.2), color='red', lw=2.5, alpha=0.9)

# Emphasize start and end
ax.scatter(*surface(*A), color='black', s=50)
ax.text(*(surface(*A)+[0,0,0.2]), 'A', fontweight='bold')
ax.scatter(*surface(*B), color='black', s=50)
ax.text(*(surface(*B)+[0,0,0.2]), 'B', fontweight='bold')

# Draw precession arc at B (angle between e1_B and transported spin)
pos_B = surface(*B)
e1_B, e2_B = tangent_basis(*B)
spin_B = spin_transported[-1]
# angle from e1_B to spin_B in the tangent plane
cosang = np.dot(spin_B, e1_B)
sinang = np.dot(spin_B, e2_B)
precession_angle = np.arctan2(sinang, cosang)
theta = np.linspace(0, precession_angle, 20)
arc_x = pos_B[0] + L*1.2*(np.cos(theta)*e1_B[0] + np.sin(theta)*e2_B[0])
arc_y = pos_B[1] + L*1.2*(np.cos(theta)*e1_B[1] + np.sin(theta)*e2_B[1])
arc_z = pos_B[2] + L*1.2*(np.cos(theta)*e1_B[2] + np.sin(theta)*e2_B[2])
ax.plot(arc_x, arc_y, arc_z, color='orange', lw=2)
ax.text(arc_x[10], arc_y[10], arc_z[10]+0.1,
        r'$\omega_\mu^{ab}dx^\mu$', color='darkorange', fontweight='bold', fontsize=12)

# Fixed global reference axis (global x‑direction)
# Project global x unit vector onto tangent plane at each point to show absolute precession
ax.quiver(-1, -1, 0, 1.2, 0, 0, color='navy', lw=1.5, alpha=0.3, label='Global x‑axis (ref)')

ax.view_init(elev=35, azim=45)
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig('Thesis_Ready_Plots/Spin_Precession_Holonomy.png', dpi=300)
plt.show()

# ----------------------------------------------------------------------



