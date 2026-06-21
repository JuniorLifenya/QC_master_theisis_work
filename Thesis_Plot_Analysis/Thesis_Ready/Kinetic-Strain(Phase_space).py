"""
Kinetic strain on a circular orbit — SIMPLIFIED two-panel version.

LEFT  (3D) : the momentum-space landscape V(p) = h_+(p_x² − p_y²).
             A circular orbit |p| = p_0 rides on it; you can see the
             momentum energy rise (yellow ridges) and fall (blue valleys)
             as the orbit goes around.  Camera angle chosen so the orbit
             reads as a clear up-and-down wave.

RIGHT (2D) : the same landscape from straight above, orbit overlaid with
             motion arrows, so the four extrema are unambiguous.

Physics kept minimal: V(p) modulates the kinetic energy of an
in-plane orbit, twice per revolution (the spin-2 GW signature).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection

OUT_DIR = "Thesis_Ready_Plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ─── parameters ──────────────────────────────────────────────────────
h_plus = 0.5
p_0    = 1.0
N_phi  = 240

plt.rcParams.update({"font.family": "serif", "font.size": 12})

fig = plt.figure(figsize=(15, 7.0))
fig.patch.set_facecolor("white")
gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0],
                      left=0.02, right=0.97, top=0.88, bottom=0.10,
                      wspace=0.12)
ax3d = fig.add_subplot(gs[0, 0], projection="3d")
ax2d = fig.add_subplot(gs[0, 1])

# ─── the landscape ───────────────────────────────────────────────────
gx = np.linspace(-2.0, 2.0, 120)
PX, PY = np.meshgrid(gx, gx)
V = h_plus * (PX**2 - PY**2)
v_max = abs(V).max()

# ════════════════════════════════════════════════════════════════════
# LEFT — 3D landscape with the orbit riding on it
# ════════════════════════════════════════════════════════════════════
ls = LightSource(azdeg=315, altdeg=45)
shaded = ls.shade(V, plt.cm.RdBu_r, vmin=-v_max, vmax=v_max,
                  vert_exag=0.6, blend_mode="soft")
ax3d.plot_surface(PX, PY, V, facecolors=shaded, rstride=1, cstride=1,
                  linewidth=0, antialiased=True, alpha=0.78, shade=False,
                  zorder=1)

# Orbit lifted onto the surface
phi    = np.linspace(0, 2*np.pi, N_phi)
orb_px = p_0 * np.cos(phi)
orb_py = p_0 * np.sin(phi)
orb_V  = h_plus * (orb_px**2 - orb_py**2)        # = h_+ p_0² cos(2φ)
orb_z  = orb_V + 0.12     # lifted a bit higher so it clears the ridge

# Thick black "shadow" orbit for contrast, then bright colored core
pts  = np.array([orb_px, orb_py, orb_z]).T.reshape(-1, 1, 3)
segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
# shadow
lc_sh = Line3DCollection(segs, colors="black", linewidths=8, zorder=9)
ax3d.add_collection3d(lc_sh)
# colored
lc = Line3DCollection(segs, cmap=plt.cm.plasma, linewidths=5,
                      norm=plt.Normalize(-h_plus*p_0**2, h_plus*p_0**2),
                      zorder=10)
lc.set_array(orb_V[:-1])
ax3d.add_collection3d(lc)

# Four extrema balls
for ph in [0, np.pi/2, np.pi, 3*np.pi/2]:
    vv = h_plus * (p_0*np.cos(ph))**2 - h_plus*(p_0*np.sin(ph))**2
    hi = vv > 0
    ax3d.scatter(p_0*np.cos(ph), p_0*np.sin(ph), vv+0.07,
                 color="#fde047" if hi else "#22d3ee",
                 edgecolor="#92400e" if hi else "#0e7490",
                 s=190, lw=2, zorder=20)

# "fast / slow" annotations on the orbit — placed away from the orbit ribbon
ax3d.text(p_0*1.05, -p_0*0.55, h_plus*p_0**2 + 0.55,
          "energy LOW",
          fontsize=10.5, fontweight="bold", color="#9c2c2c",
          ha="center", zorder=22)
ax3d.text(-p_0*0.30, p_0*1.05, -h_plus*p_0**2 - 0.62,
          "energy HIGH",
          fontsize=10.5, fontweight="bold", color="#1a4980",
          ha="center", zorder=22)

# Momentum arrow at phi = 7pi/4 (front-right, clearly visible)
pa = 7*np.pi/4
za = h_plus*(p_0*np.cos(pa))**2 - h_plus*(p_0*np.sin(pa))**2 + 0.12
ax3d.quiver(p_0*np.cos(pa), p_0*np.sin(pa), za,
            -np.sin(pa)*0.6, np.cos(pa)*0.6, 0,
            color="#15396b", lw=3, arrow_length_ratio=0.32, zorder=25)
ax3d.text(p_0*np.cos(pa)-np.sin(pa)*0.6+0.15,
          p_0*np.sin(pa)+np.cos(pa)*0.6-0.30, za+0.05,
          r"$\vec p$", fontsize=15, fontweight="bold",
          color="#15396b", zorder=26)

# Camera: elevation high enough to see the orbit's up/down clearly,
# azimuth set so the saddle's two ridges frame the orbit
ax3d.view_init(elev=36, azim=68)
ax3d.set_xlabel(r"$p_x/p_0$", fontsize=12, labelpad=6)
ax3d.set_ylabel(r"$p_y/p_0$", fontsize=12, labelpad=6)
ax3d.set_zlabel(r"$V(\vec p)$", fontsize=12, labelpad=4)
ax3d.set_zticks([])
ax3d.set_zlim(-v_max*1.10, v_max*1.10)
ax3d.set_title("Momentum-space landscape  "
               r"$V(\vec p)=h_+(p_x^2-p_y^2)$",
               fontsize=12.5, fontweight="bold", pad=8)

# ════════════════════════════════════════════════════════════════════
# RIGHT — top-down view
# ════════════════════════════════════════════════════════════════════
cf = ax2d.contourf(PX, PY, V, levels=21, cmap="RdBu_r",
                   vmin=-v_max, vmax=v_max)
ax2d.contour(PX, PY, V, levels=[0], colors="black", linewidths=1.0, alpha=0.5)

# Orbit
ax2d.plot(orb_px, orb_py, color="white", lw=6, solid_capstyle="round", zorder=4)
pts2 = np.array([orb_px, orb_py]).T.reshape(-1, 1, 2)
seg2 = np.concatenate([pts2[:-1], pts2[1:]], axis=1)
lc2  = LineCollection(seg2, cmap=plt.cm.plasma, linewidths=3.6,
                      norm=plt.Normalize(-h_plus*p_0**2, h_plus*p_0**2), zorder=5)
lc2.set_array(orb_V[:-1])
ax2d.add_collection(lc2)

# Motion arrows
for k in range(8):
    ph = 2*np.pi*k/8
    px_a, py_a = p_0*np.cos(ph), p_0*np.sin(ph)
    ax2d.add_patch(FancyArrowPatch((px_a, py_a),
                                   (px_a-np.sin(ph)*0.26, py_a+np.cos(ph)*0.26),
                                   color="black", lw=1.4,
                                   arrowstyle="-|>", mutation_scale=12, zorder=10))

# Extrema
for ph in [0, np.pi/2, np.pi, 3*np.pi/2]:
    vv = h_plus*(p_0*np.cos(ph))**2 - h_plus*(p_0*np.sin(ph))**2
    hi = vv > 0
    ax2d.scatter(p_0*np.cos(ph), p_0*np.sin(ph),
                 color="#fde047" if hi else "#22d3ee",
                 edgecolor="#92400e" if hi else "#0e7490",
                 s=150, lw=1.8, zorder=12)

ax2d.text(1.55, 0, "HIGH", ha="center", va="center", fontsize=10,
          fontweight="bold", color="#9c2c2c")
ax2d.text(-1.55, 0, "HIGH", ha="center", va="center", fontsize=10,
          fontweight="bold", color="#9c2c2c")
ax2d.text(0, 1.62, "LOW", ha="center", va="center", fontsize=10,
          fontweight="bold", color="#1a4980")
ax2d.text(0, -1.62, "LOW", ha="center", va="center", fontsize=10,
          fontweight="bold", color="#1a4980")

ax2d.set_aspect("equal")
ax2d.set_xlim(-1.95, 1.95); ax2d.set_ylim(-1.95, 1.95)
ax2d.set_xlabel(r"$p_x/p_0$"); ax2d.set_ylabel(r"$p_y/p_0$")
ax2d.set_title("View from above", fontsize=12.5, fontweight="bold", pad=8)

# ─── title + one-line caption ───────────────────────────────────────
fig.suptitle("As the electron orbits, its momentum sweeps high and low "
             "regions — twice per turn",
             fontsize=14, fontweight="bold", y=0.975)

fig.text(0.5, 0.02,
         r"The orbit stays in-plane; only its kinetic energy is modulated. "
         r"Around one revolution the momentum passes two peaks and two "
         r"troughs — the spin-2 signature of the gravitational wave.",
         ha="center", fontsize=10.5, style="italic", color="0.30")

out = os.path.join(OUT_DIR, "fig_kinetic_strain_orbit_simple2.png")
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out}")