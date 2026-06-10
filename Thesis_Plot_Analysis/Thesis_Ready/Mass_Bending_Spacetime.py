import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3-D projection)
import os

os.makedirs("Thesis_Ready_Plots", exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# 1.  The curved surface  =  an embedding diagram of the spatial metric
# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS NOTE (the one rigour point worth flagging):
#   A *single* mass does NOT dimple spacetime into a saddle.  A saddle has
#   curvature of opposite sign along x and y (that is a *tidal / vacuum* picture,
#   which is why the spin-connection figure used z = 0.6·x·y).  One localised
#   source produces a monotone WELL — the classic rubber-sheet dimple — whose
#   depth tracks the gravitational potential.
#
#   Default ("well"):  softened Newtonian-potential embedding
#         z(r) = -2 G M / sqrt(r² + a²)
#     The softening length a gives a finite-depth, smooth-bottomed dimple — i.e.
#     an extended body (a star), not a point singularity.
#
#   "flamm":  Flamm's paraboloid, z(r) = -2·sqrt(r_s)·sqrt(r - r_s).  This is the
#     *exact* isometric embedding of a t = const, θ = π/2 slice of Schwarzschild,
#     and is the strictly-correct "curved space" surface (throat at r = r_s).
#
#   "saddle": z = 0.6·x·y  — kept so you can reproduce the Doc-1 aesthetic.
# ═════════════════════════════════════════════════════════════════════════════
G_M    = 0.70   # mass parameter (geometrised units), sets the well depth
A_SOFT = 0.70   # softening length  -> finite, smooth central depth
MODE   = "well"  # "well" | "flamm" | "saddle"


def surface_z(x, y, mode=MODE):
    r = np.sqrt(x**2 + y**2)
    if mode == "well":
        return -2.0 * G_M / np.sqrt(r**2 + A_SOFT**2)
    if mode == "flamm":
        r_s = 0.45
        return -2.0 * np.sqrt(r_s) * np.sqrt(np.clip(r - r_s, 0.0, None))
    if mode == "saddle":
        return 0.6 * x * y
    raise ValueError(f"unknown surface mode: {mode}")


# ═════════════════════════════════════════════════════════════════════════════
# 2.  The source:  one spherical body, floating just above the dip
# ═════════════════════════════════════════════════════════════════════════════
def sphere_mesh(cx, cy, cz, radius, n=60):
    """Parametric (u,v) sphere mesh centred at (cx,cy,cz)."""
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    xs = cx + radius * np.outer(np.cos(u), np.sin(v))
    ys = cy + radius * np.outer(np.sin(u), np.sin(v))
    zs = cz + radius * np.outer(np.ones_like(u), np.cos(v))
    return xs, ys, zs


# ═════════════════════════════════════════════════════════════════════════════
# 3.  A test particle's geodesic  =  numerically integrated scattering orbit
# ─────────────────────────────────────────────────────────────────────────────
#   We integrate a test mass in the (softened) field of the central body with a
#   symplectic velocity-Verlet (leapfrog) scheme.  The acceleration is
#         a = -∇Φ ,   Φ = -G M / sqrt(r² + a²)   ⇒   a = -G M r / (r² + a²)^{3/2}
#   The particle enters from the left with an impact parameter b, bends toward
#   the mass, and leaves deflected — the gravitational-deflection analogue of the
#   spin-connection figure's "transport from A to B".
# ═════════════════════════════════════════════════════════════════════════════
def acceleration(p):
    denom = (p[0] ** 2 + p[1] ** 2 + A_SOFT ** 2) ** 1.5
    return -G_M * p / denom


def integrate_geodesic(p0, v0, dt=0.006, x_stop=2.7, max_steps=4000):
    p = np.array(p0, float)
    v = np.array(v0, float)
    path = [p.copy()]
    a = acceleration(p)
    for _ in range(max_steps):
        v = v + 0.5 * dt * a            # half kick
        p = p + dt * v                  # drift
        a = acceleration(p)
        v = v + 0.5 * dt * a            # half kick
        path.append(p.copy())
        if p[0] > x_stop:               # exited on the far side
            break
    return np.array(path)


b_impact = 0.95                          # impact parameter (offset in y)
v_speed  = 1.35                          # asymptotic speed
traj = integrate_geodesic(p0=[-2.7,  b_impact],
                          v0=[v_speed, 0.0])

# Lift the planar orbit onto the curved sheet (ride slightly above it).
LIFT = 0.06
tx, ty = traj[:, 0], traj[:, 1]
tz = surface_z(tx, ty) + LIFT

A_pt = np.array([tx[0],  ty[0],  tz[0]])
B_pt = np.array([tx[-1], ty[-1], tz[-1]])

# ═════════════════════════════════════════════════════════════════════════════
# 4.  Figure
# ═════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(11, 8.5))
fig.patch.set_facecolor("white")
ax = fig.add_subplot(111, projection="3d")
fig.suptitle("A single mass curving spacetime",
             fontsize=15, fontweight="bold")
ax.set_title(r"geodesic of a test particle deflected from $A$ to $B$",
             fontsize=10, pad=2)

# --- the curved sheet --------------------------------------------------------
N, SPAN = 70, 3.0
gx = np.linspace(-SPAN, SPAN, N)
X, Y = np.meshgrid(gx, gx)
Z = surface_z(X, Y)

ax.plot_surface(X, Y, Z, alpha=0.45, color="wheat",
                edgecolor="none", rstride=1, cstride=1, antialiased=True)
ax.plot_wireframe(X, Y, Z, color="seagreen", alpha=0.18, linewidth=0.7)

# --- the central body (the source) -------------------------------------------
R_SPHERE = 0.50
z_floor = surface_z(0.0, 0.0)                       # bottom of the dip
cz = z_floor + R_SPHERE + 0.95                      # hover above the well bottom
xs, ys, zs = sphere_mesh(0.0, 0.0, cz, R_SPHERE)

ls = LightSource(azdeg=315, altdeg=45)
shaded = ls.shade(zs, cmap=plt.cm.cividis, vert_exag=1.0, blend_mode="soft")
ax.plot_surface(xs, ys, zs, facecolors=shaded, rstride=1, cstride=1,
                linewidth=0, antialiased=True, shade=False, zorder=6)

# thin plumb-line from the body down into the throat of the well
ax.plot([0, 0], [0, 0], [cz - R_SPHERE, z_floor],
        color="dimgray", ls=":", lw=1.2, alpha=0.8)
ax.text(0, 0, cz + R_SPHERE + 0.18, "mass $M$", fontsize=10,
        fontweight="bold", ha="center")

# --- the deflected geodesic A -> B -------------------------------------------
ax.plot(tx, ty, tz, color="crimson", lw=2.6, zorder=7,
        label=r"test-particle geodesic $x^\mu(\lambda)$")

# tangent (4-velocity direction) arrows at A and B, echoing the tetrad legs
def tangent(i):
    d = traj[min(i + 1, len(traj) - 1)] - traj[max(i - 1, 0)]
    d = d / np.linalg.norm(d)
    return d

for P, i, name, dy in [(A_pt, 0, "A", 0.0), (B_pt, len(traj) - 1, "B", 0.0)]:
    d2 = tangent(i)
    dz = surface_z(P[0] + 0.25 * d2[0], P[1] + 0.25 * d2[1]) + LIFT - P[2]
    ax.quiver(*P, 0.6 * d2[0], 0.6 * d2[1], dz,
              color="black", arrow_length_ratio=0.3, lw=2.0, zorder=8)
    ax.scatter(*P, color="black", s=45, zorder=9)
    ax.text(P[0], P[1] + 0.15, P[2] + 0.55, f"Point {name}",
            fontsize=10, fontweight="bold", ha="center")

# ═════════════════════════════════════════════════════════════════════════════
# 5.  Camera, limits, export
# ═════════════════════════════════════════════════════════════════════════════
ax.view_init(elev=26, azim=-58)
ax.set_xlim(-SPAN, SPAN); ax.set_ylim(-SPAN, SPAN); ax.set_zlim(z_floor - 0.3, 1.4)
ax.set_xlabel(r"$x$", fontsize=12)
ax.set_ylabel(r"$y$", fontsize=12)
ax.set_zlabel(r"$z$  (embedding height $\sim \Phi$)", fontsize=11)
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.92), fontsize=9)

plt.tight_layout()
out = "Thesis_Ready_Plots/fig_mass_curving_spacetime_Geodesic.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.show()
print("Saved:", out)