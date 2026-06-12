"""
Figure 2 — Binary black hole: near-zone curvature (wells) + radiation zone (wave).
Improved version:
  * Spheres aspect-corrected -> render perfectly round.
  * Spheres now SIT IN their wells instead of hovering above them.
  * Sheet colour configurable in ONE place (SHEET STYLE block).
  * LightSource hill-shading on the sheet, clean axes, zone annotations.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, to_rgb, Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import os

# =============================== SHEET STYLE =================================
# (a) colormap mode: set SHEET_CMAP, keep SHEET_SOLID_COLOR = None
# (b) solid mode:    set SHEET_SOLID_COLOR = "royalblue" (any colour)
try:
    import seaborn as sns
    SHEET_CMAP = sns.color_palette("mako", as_cmap=True)
except ImportError:
    SHEET_CMAP = plt.get_cmap("turbo")

SHEET_SOLID_COLOR = "turbo"
SHEET_ALPHA       = 0.95
WIRE_COLOR        = "k"
WIRE_ALPHA        = 0.10
# ==============================================================================

os.makedirs("Thesis_Ready_Plots", exist_ok=True)

# --- Physical parameters --------------------------------------------------------
d_binary   = 1.1
x1, y1     = -d_binary/2, 0.0
x2, y2     =  d_binary/2, 0.0
depth_well = -0.75
sigma_well = 0.35

A_wave            = 0.32
k_wave            = 2.5
r_wave_start      = 1.8
r_wave_transition = 0.5
decay_power       = 0.5            # 1/sqrt(r) fall-off

def sheet_height(x, y):
    r = np.sqrt(x**2 + y**2)
    well1 = depth_well * np.exp(-((x - x1)**2 + (y - y1)**2) / (2 * sigma_well**2))
    well2 = depth_well * np.exp(-((x - x2)**2 + (y - y2)**2) / (2 * sigma_well**2))
    env = 0.5 * (1.0 + np.tanh((r - r_wave_start) / r_wave_transition))
    amp = np.where(r > 0.1, (r_wave_start / np.maximum(r, 0.1)) ** decay_power, 0.0)
    return well1 + well2 + env * amp * A_wave * np.sin(k_wave * r)

# --- Mesh -------------------------------------------------------------------------
N, SPAN = 220, 6.0
gx   = np.linspace(-SPAN, SPAN, N)
X, Y = np.meshgrid(gx, gx)
Z    = sheet_height(X, Y)

# --- Figure -------------------------------------------------------------------------
fig = plt.figure(figsize=(12, 7.5))
fig.patch.set_facecolor("white")
ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

fig.suptitle("Binary Black Hole: Near-Zone Curvature and Outgoing Radiation",
             fontsize=15, fontweight="bold", y=0.88)


# --- Limits & aspect FIRST (needed for sphere correction) ---------------------------
z_floor = -1.35
z_top   = 0.85
ax.set_xlim(-SPAN, SPAN); ax.set_ylim(-SPAN, SPAN); ax.set_zlim(z_floor, z_top)
BOX = (2*SPAN, 2*SPAN, 3.0)
ax.set_box_aspect(BOX)

sx = BOX[0] / (2*SPAN)
sy = BOX[1] / (2*SPAN)
sz = BOX[2] / (z_top - z_floor)

def visual_sphere(cx, cy, cz, R, n=80):
    """Ellipsoid in data coords that renders as a round sphere of radius R."""
    u = np.linspace(0, 2*np.pi, n)
    v = np.linspace(0, np.pi, n)
    rx, ry, rz = R, R * sx/sy, R * sx/sz
    return (cx + rx * np.outer(np.cos(u), np.sin(v)),
            cy + ry * np.outer(np.sin(u), np.sin(v)),
            cz + rz * np.outer(np.ones_like(u), np.cos(v)))

# --- Black holes seated in their wells ----------------------------------------------
R_horizon = 0.30
ls_bh = LightSource(315, 45)
for (cx, cy, lbl) in [(x1, y1, r"$M_1$"), (x2, y2, r"$M_2$")]:
    z_well = sheet_height(cx, cy)
    cz = z_well + 0.55 * R_horizon * sx/sz       # nestled: lower third inside the well
    XS, YS, ZS = visual_sphere(cx, cy, cz, R_horizon)
    rgb = ls_bh.shade(ZS, plt.cm.inferno, vert_exag=1.0, blend_mode="soft")
    ax.plot_surface(XS, YS, ZS, facecolors=rgb, rstride=1, cstride=1,
                    linewidth=0, antialiased=True, shade=False, zorder=10)
    ax.text(cx, cy, cz + R_horizon * sx/sz + 0.10, lbl, fontsize=10,
            fontweight="bold", ha="center", color="white", zorder=11)

# --- Flat reference sheet -------------------------------------------------------------
ax.plot_surface(X, Y, np.zeros_like(Z), color="gainsboro", alpha=0.12,
                linewidth=0, antialiased=True, zorder=0)

# --- Warped sheet with hill-shading ----------------------------------------------------
ls_sheet = LightSource(azdeg=315, altdeg=50)
if SHEET_SOLID_COLOR is not None:
    base = np.tile(to_rgb(SHEET_SOLID_COLOR), (*Z.shape, 1))
    rgb  = ls_sheet.shade_rgb(base, Z, vert_exag=2.0, blend_mode="soft")
else:
    # clip the colour scale to the wave amplitude so the ripples use the full
    # palette; the deep wells simply saturate at the dark end
    cnorm = Normalize(vmin=-1.2*A_wave, vmax=1.2*A_wave, clip=True)
    rgb  = ls_sheet.shade(Z, cmap=SHEET_CMAP, norm=cnorm,
                          vert_exag=2.0, blend_mode="soft")

ax.plot_surface(X, Y, Z, facecolors=rgb, rstride=1, cstride=1,
                linewidth=0, antialiased=True, shade=False,
                alpha=SHEET_ALPHA, zorder=2)

ax.plot_wireframe(X, Y, Z, color=WIRE_COLOR, alpha=WIRE_ALPHA,
                  linewidth=0.4, rstride=10, cstride=10, zorder=3)

# --- Floor contours ----------------------------------------------------------------------
levels = np.linspace(-A_wave, A_wave, 11)
ax.contour(X, Y, Z, levels=levels, zdir="z", offset=z_floor,
           cmap=SHEET_CMAP, linewidths=0.8, alpha=0.45, zorder=1)

# --- Zone annotations (2D overlays: always legible) ---------------------------------------
ax.text2D(0.16, 0.70, "near zone:\nstatic curvature wells",
          transform=ax.transAxes, fontsize=11, color="0.15", ha="center",
          bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.85))

# --- Cosmetics ------------------------------------------------------------------------------
ax.view_init(elev=24, azim=-55)
ax.set_xlabel(r"$x$", fontsize=12, labelpad=8)
ax.set_ylabel(r"$y$", fontsize=12, labelpad=8)
ax.set_zticks([]); ax.grid(False)
for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    pane.set_visible(False)
ax.zaxis.line.set_color((1, 1, 1, 0))

fig.subplots_adjust(left=0.0, right=1.0, bottom=0.02, top=1.02)

out = "Thesis_Ready_Plots/fig_binary_wave_curling.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.show()
print("Saved:", out)