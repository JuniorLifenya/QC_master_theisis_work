import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import os

os.makedirs("Thesis_Ready_Plots", exist_ok=True)

G_M, A_SOFT, MODE = 0.70, 0.70, "well"

def surface_z(x, y, mode=MODE):
    r = np.sqrt(x**2 + y**2)
    if mode == "well":
        return -2.0 * G_M / np.sqrt(r**2 + A_SOFT**2)
    if mode == "flamm":
        r_s = 0.45
        return -2.0 * np.sqrt(r_s) * np.sqrt(np.clip(r - r_s, 0.0, None))
    if mode == "saddle":
        return 0.6 * x * y
    raise ValueError(mode)

def sphere_mesh(cx, cy, cz, radius, n=90):
    u = np.linspace(0, 2*np.pi, n); v = np.linspace(0, np.pi, n)
    return (cx + radius*np.outer(np.cos(u), np.sin(v)),
            cy + radius*np.outer(np.sin(u), np.sin(v)),
            cz + radius*np.outer(np.ones_like(u), np.cos(v)))

def acceleration(p):
    denom = (p[0]**2 + p[1]**2 + A_SOFT**2)**1.5
    return -G_M * p / denom

def integrate_geodesic(p0, v0, dt=0.006, x_stop=2.7, max_steps=4000):
    p = np.array(p0, float); v = np.array(v0, float)
    path = [p.copy()]; a = acceleration(p)
    for _ in range(max_steps):
        v += 0.5*dt*a; p += dt*v; a = acceleration(p); v += 0.5*dt*a
        path.append(p.copy())
        if p[0] > x_stop:
            break
    return np.array(path)

b_impact, v_speed = 0.95, 1.35
traj = integrate_geodesic(p0=[-2.7, b_impact], v0=[v_speed, 0.0])
LIFT = 0.06
tx, ty = traj[:, 0], traj[:, 1]
tz = surface_z(tx, ty) + LIFT
A_pt = np.array([tx[0], ty[0], tz[0]])
B_pt = np.array([tx[-1], ty[-1], tz[-1]])

fig = plt.figure(figsize=(11, 8.5))
fig.patch.set_facecolor("white")
# (FIX 2) honour manual zorder
ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
fig.suptitle("A single mass curving spacetime", fontsize=15, fontweight="bold")
ax.set_title(r"geodesic of a test particle deflected from $A$ to $B$", fontsize=10, pad=2)

N, SPAN = 70, 3.0
gx = np.linspace(-SPAN, SPAN, N)
X, Y = np.meshgrid(gx, gx)
Z = surface_z(X, Y)
z_floor = surface_z(0.0, 0.0)

ax.plot_surface(X, Y, Z, alpha=0.35, color="wheat", edgecolor="none",
                rstride=1, cstride=1, antialiased=True, zorder=2)
wire = ax.plot_wireframe(X, Y, Z, color="cyan", alpha=0.28, linewidth=0.7)
wire.set_zorder(1)   # behind the translucent sheet

R_SPHERE = 0.50
cz = z_floor + R_SPHERE + 0.95
xs, ys, zs = sphere_mesh(0.0, 0.0, cz, R_SPHERE)
shaded = LightSource(315, 45).shade(zs, plt.cm.cividis, vert_exag=1.0, blend_mode="soft")
ax.plot_surface(xs, ys, zs, facecolors=shaded, rstride=1, cstride=1,
                linewidth=0, antialiased=True, shade=False, zorder=10)   # on top
ax.plot([0, 0], [0, 0], [cz - R_SPHERE, z_floor], color="dimgray",
        ls=":", lw=1.2, alpha=0.8, zorder=3)
ax.text(0, 0, cz + R_SPHERE + 0.18, "mass $M$", fontsize=10,
        fontweight="bold", ha="center", zorder=11)

ax.plot(tx, ty, tz, color="crimson", lw=2.6, zorder=5,
        label=r"test-particle geodesic $x^\mu(\lambda)$")

def tangent(i):
    d = traj[min(i + 1, len(traj) - 1)] - traj[max(i - 1, 0)]
    return d / np.linalg.norm(d)

for P, i, name in [(A_pt, 0, "A"), (B_pt, len(traj) - 1, "B")]:
    d2 = tangent(i)
    dz = surface_z(P[0] + 0.25*d2[0], P[1] + 0.25*d2[1]) + LIFT - P[2]
    ax.quiver(*P, 0.6*d2[0], 0.6*d2[1], dz, color="black",
              arrow_length_ratio=0.3, lw=2.0, zorder=8)
    ax.scatter(*P, color="black", s=45, zorder=9)
    ax.text(P[0], P[1] + 0.15, P[2] + 0.55, f"Point {name}",
            fontsize=10, fontweight="bold", ha="center", zorder=11)

ax.view_init(elev=30, azim=-50)
ax.set_xlim(-SPAN, SPAN); ax.set_ylim(-SPAN, SPAN); ax.set_zlim(z_floor - 0.3, 0.9)
# (FIX 1) equal data-units on every axis -> round sphere
ax.set_box_aspect((2*SPAN, 2*SPAN, 0.9 - (z_floor - 0.3)))
ax.set_xlabel(r"$x$", fontsize=12); ax.set_ylabel(r"$y$", fontsize=12)
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.92), fontsize=9)

fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.93)
out = "Thesis_Ready_Plots/fig_mass_curving_spacetime_Geodesic.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.show()
print("Saved:", out)