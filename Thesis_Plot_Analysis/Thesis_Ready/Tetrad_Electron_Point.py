import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
import os

os.makedirs("Thesis_Ready_Plots", exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# 1. Curved spacetime: the saddle surface
# ─────────────────────────────────────────────────────────────────
def Zsurf(x, y): return 0.6 * x * y
def r_vec(x, y): return np.array([x, y, Zsurf(x, y)])
def dr_dx(x, y): return np.array([1.0, 0.0, 0.6 * y])
def dr_dy(x, y): return np.array([0.0, 1.0, 0.6 * x])

# Tangent plane at a point (x0,y0)
def tangent_plane(x0, y0, s, t):
    P = r_vec(x0, y0)
    rx, ry = dr_dx(x0, y0), dr_dy(x0, y0)
    S, T = np.meshgrid(s, t)
    return (P[0] + S*rx[0] + T*ry[0],
            P[1] + S*rx[1] + T*ry[1],
            P[2] + S*rx[2] + T*ry[2])

# Orthonormal frame at P
def frame(x0, y0):
    rx, ry = dr_dx(x0, y0), dr_dy(x0, y0)
    n = np.cross(rx, ry)
    e3 = n / np.linalg.norm(n)               # normal
    e1 = rx / np.linalg.norm(rx)              # tangent vector along x
    e2 = np.cross(e3, e1)                     # right‑handed tangent vector
    return e1, e2, e3

# ─────────────────────────────────────────────────────────────────
# 2. Helper: draw a sphere
# ─────────────────────────────────────────────────────────────────
def sphere(cx, cy, cz, R, n=60):
    u = np.linspace(0, 2*np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = cx + R * np.outer(np.cos(u), np.sin(v))
    y = cy + R * np.outer(np.sin(u), np.sin(v))
    z = cz + R * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z

# ─────────────────────────────────────────────────────────────────
# 3. Dipole loops (B‑field) — rotate from z‑axis to given normal vector e3
# ─────────────────────────────────────────────────────────────────
def rotated_dipole_loop(scale, e3, npts=240):
    """Dipole loops originally around z‑axis; rotate to align with e3."""
    th = np.linspace(0.001, np.pi-0.001, npts)
    r = scale * np.sin(th)**2
    rho = r * np.sin(th)
    zc = r * np.cos(th)
    rho = np.concatenate([rho, rho[::-1]])
    zc = np.concatenate([zc, zc[::-1]])
    side = np.concatenate([np.ones(npts), -np.ones(npts)])
    # start with points in (x,y,z) where z is up
    loop = np.array([rho*side*0, rho*side*0, zc])  # shape (3, N)
    # rotation matrix that maps [0,0,1] to e3
    z_axis = np.array([0., 0., 1.])
    v = np.cross(z_axis, e3)
    s = np.linalg.norm(v)
    c = np.dot(z_axis, e3)
    if s == 0:
        R = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1-c)/s**2)
    loop_rot = R @ loop
    return loop_rot[0], loop_rot[1], loop_rot[2]

# ─────────────────────────────────────────────────────────────────
# 4. Combined figure
# ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 7.5))
ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
ax.set_facecolor("white")
ax.set_title("Electron on a curved spacetime\n(Saddle manifold + local tangent plane)", 
             fontsize=12, fontweight="bold")

# --- Draw the curved saddle surface ---
pts = np.linspace(-1.5, 1.5, 40)
X, Y = np.meshgrid(pts, pts)
ax.plot_surface(X, Y, Zsurf(X, Y), color="wheat", alpha=0.5,
                edgecolor="none", zorder=1)
ax.plot_wireframe(X, Y, Zsurf(X, Y), color="darkgoldenrod",
                  alpha=0.15, lw=0.5, zorder=2)

# --- Choose point P where the electron lives ---
P0 = (0.4, 0.4)                     # (x0, y0)
Pp = r_vec(*P0)                     # 3D position

# --- Draw the tangent plane at P (small patch) ---
s = np.linspace(-0.5, 0.5, 10)
TX, TY, TZ = tangent_plane(*P0, s, s)
ax.plot_surface(TX, TY, TZ, color="steelblue", alpha=0.35,
                edgecolor="navy", linewidth=0.4, zorder=5)

# --- Local orthonormal frame at P ---
e1, e2, e3 = frame(*P0)

# Scale factor for arrows
L = 0.5

# --- Draw the electron at P ---
electron_R = 0.10
XS, YS, ZS = sphere(Pp[0], Pp[1], Pp[2] + electron_R*0.5, electron_R)
e_rgb = LightSource(315, 45).shade(ZS, plt.cm.Blues, vert_exag=1.0,
                                   blend_mode="soft")
ax.plot_surface(XS, YS, ZS, facecolors=e_rgb, rstride=1, cstride=1,
                linewidth=0, antialiased=True, shade=False, zorder=20)
ax.text(Pp[0], Pp[1], Pp[2] + 0.25, r"$e^-$", fontsize=12, fontweight="bold",
        ha="center", color="#08306B", zorder=21)





# --- Mark the point P ---
ax.scatter(*Pp, color="black", s=30, zorder=7)

# --- Adjust view and limits ---
ax.set_xlim(-1.6, 1.6)
ax.set_ylim(-1.6, 1.6)
ax.set_zlim(-1.5, 1.5)
ax.set_box_aspect((3.2, 3.2, 3.0))
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.view_init(elev=20, azim=-55)

# Remove pane colors for a clean look
ax.xaxis.pane.set_facecolor("white")
ax.yaxis.pane.set_facecolor("white")
ax.zaxis.pane.set_facecolor("white")
ax.xaxis.pane.set_edgecolor("none")
ax.yaxis.pane.set_edgecolor("none")
ax.zaxis.pane.set_edgecolor("none")

# Save and show
out = "Thesis_Ready_Plots/electron_on_curved_spacetime.png"
plt.tight_layout()
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.show()
print("Saved:", out)