import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import os

try:
    import seaborn as sns
    SURF_CMAP = sns.color_palette("mako", as_cmap=True)   # cool teal -> navy
except Exception:
    SURF_CMAP = "plasma"

os.makedirs("Thesis_Ready_Plots", exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# 0.  Physics knobs
# ═════════════════════════════════════════════════════════════════════════════
G_M    = 0.70          # mass parameter (geometrised units)
A_SOFT = 0.70          # softening length -> finite, smooth central depth
MODE   = "well"        # "well" | "flamm" | "saddle"
SHOW_BEAM = True       # True -> also draw a fan of scattering fly-bys


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


def acceleration(p):
    denom = (p[0] ** 2 + p[1] ** 2 + A_SOFT ** 2) ** 1.5
    return -G_M * p / denom


def integrate(p0, v0, dt, steps, stop_x=None):
    """Velocity-Verlet (symplectic) integration of a test particle."""
    p = np.array(p0, float); v = np.array(v0, float)
    out = [p.copy()]
    a = acceleration(p)
    for _ in range(steps):
        v += 0.5 * dt * a
        p += dt * v
        a = acceleration(p)
        v += 0.5 * dt * a
        out.append(p.copy())
        if stop_x is not None and p[0] > stop_x:
            break
    return np.array(out)


# ═════════════════════════════════════════════════════════════════════════════
# 1.  The bound, PRECESSING orbit  (the centrepiece)
# ─────────────────────────────────────────────────────────────────────────────
#   Started at apoapsis with a sub-circular tangential speed -> an eccentric
#   bound orbit.  Because the softened potential is not ∝ 1/r, the ellipse does
#   not close: its axis advances every revolution, tracing a rosette.  This is
#   the Newtonian shadow of perihelion precession.
# ═════════════════════════════════════════════════════════════════════════════
orbit = integrate(p0=[2.2, 0.0], v0=[0.0, 0.40], dt=0.01, steps=4200)
ox, oy = orbit[:, 0], orbit[:, 1]
LIFT = 0.06
oz = surface_z(ox, oy) + LIFT

# ═════════════════════════════════════════════════════════════════════════════
# 2.  Figure
# ═════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(11, 8.5))
fig.patch.set_facecolor("white")
ax = fig.add_subplot(111, projection="3d")
fig.suptitle("A single mass curving spacetime", fontsize=15, fontweight="bold")
ax.set_title(r"a bound geodesic precessing into a rosette", fontsize=10, pad=2)

# --- the curved sheet, coloured by its own depth (the potential) -------------
N, SPAN = 90, 3.0
gx = np.linspace(-SPAN, SPAN, N)
X, Y = np.meshgrid(gx, gx)
Z = surface_z(X, Y)
z_floor = surface_z(0.0, 0.0)

surf = ax.plot_surface(X, Y, Z, cmap=SURF_CMAP,
                       alpha=0.55, linewidth=0, antialiased=True,
                       rstride=1, cstride=1, zorder=1)

# faint equipotential rings hugging the well (structure without clutter)
levels = np.linspace(z_floor + 0.15, -0.15, 7)
ax.contour(X, Y, Z, levels=levels, colors="white", linewidths=0.6,
           alpha=0.30, zorder=3)
# their shadow on the floor -> a clean potential map
ax.contour(X, Y, Z, levels=levels, zdir="z", offset=z_floor - 0.35,
           cmap="turbo_r", linewidths=0.8, alpha=0.55)

# --- the central body --------------------------------------------------------
def sphere_mesh(cx, cy, cz, radius, n=80):
    u = np.linspace(0, 2 * np.pi, n); v = np.linspace(0, np.pi, n)
    return (cx + radius * np.outer(np.cos(u), np.sin(v)),
            cy + radius * np.outer(np.sin(u), np.sin(v)),
            cz + radius * np.outer(np.ones_like(u), np.cos(v)))

R_SPHERE = 0.62
cz = z_floor + R_SPHERE + 0.85
xs, ys, zs = sphere_mesh(0.0, 0.0, cz, R_SPHERE)
shaded = LightSource(315, 45).shade(zs, plt.cm.inferno, vert_exag=1.0, blend_mode="soft")
ax.plot_surface(xs, ys, zs, facecolors=shaded, rstride=1, cstride=1,
                linewidth=0, antialiased=True, shade=False, zorder=1)
ax.plot([0, 0], [0, 0], [cz - R_SPHERE, z_floor], color="dimgray",
        ls=":", lw=1.1, alpha=0.8)
ax.text(0, 0, cz + R_SPHERE + 0.15, "mass $M$", fontsize=10,
        fontweight="bold", ha="center")

# --- the rosette orbit, colour-graded along its own arc length ---------------
pts = np.column_stack([ox, oy, oz]).reshape(-1, 1, 3)
segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
lc = Line3DCollection(segs, cmap="plasma", linewidths=2.1, zorder=6)
lc.set_array(np.linspace(0.0, 1.0, len(segs)))
ax.add_collection3d(lc)
ax.scatter(ox[-1], oy[-1], oz[-1], color="white", edgecolor="black",
           s=42, lw=1.0, zorder=7)   # the orbiting particle, "now"

# --- optional: a fan of unbound scattering fly-bys ---------------------------
if SHOW_BEAM:
    for b in np.linspace(0.4, 1.8, 6):
        fly = integrate(p0=[-2.8, b], v0=[1.5, 0.0], dt=0.006,
                        steps=4000, stop_x=2.8)
        fx, fy = fly[:, 0], fly[:, 1]
        fz = surface_z(fx, fy) + LIFT
        ax.plot(fx, fy, fz, color="crimson", lw=1.3, alpha=0.7, zorder=5)

# ═════════════════════════════════════════════════════════════════════════════
# 3.  Camera, limits, export
# ═════════════════════════════════════════════════════════════════════════════
ax.view_init(elev=20, azim=-50)
ax.set_xlim(-SPAN, SPAN); ax.set_ylim(-SPAN, SPAN); ax.set_zlim(z_floor - 0.4, 0.9)
ax.set_xlabel(r"$x$", fontsize=12); ax.set_ylabel(r"$y$", fontsize=12)
ax.set_zticklabels([])          # the z-axis is "potential depth" — number-free is cleaner
ax.grid(False)

fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.93)
out = "Thesis_Ready_Plots/fig_mass_curving_spacetime_rosette.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.show()
print("Saved:", out)