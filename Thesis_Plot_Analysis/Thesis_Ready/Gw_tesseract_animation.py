"""
ANIMATED: a gravitational wave travelling through a nested tesseract lattice.
=============================================================================
Same construction as the static figure, but the phase wt advances each frame,
so the red/blue strain bands physically TRAVEL up the lattice along +z while
every line stretches and squeezes in the transverse plane.

Outputs:
  Thesis_Ready_Plots/anim_gw_tesseract.mp4   (smooth, for talks)
  Thesis_Ready_Plots/anim_gw_tesseract.gif   (drop-in anywhere)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from itertools import product
import os

os.makedirs("Thesis_Ready_Plots", exist_ok=True)

# =============================== KNOBS =======================================
A_h     = 0.38                       # strain amplitude (exaggerated!)
k_gw    = 3.2                        # wavenumber along z
N_FRAMES = 60                        # frames per full wave period
FPS      = 20
SHELLS   = [0.45, 0.70, 0.95, 1.20]
SAMPLES  = 36                        # points per edge (animation-friendly)
COLOR_BY    = "radius"                 # "radius": blue centre -> red outside
RADIUS_CMAP = plt.get_cmap("turbo")
STRAIN_CMAP = plt.get_cmap("coolwarm")
# ==============================================================================

def h_of(z, phase):
    return A_h * np.sin(k_gw * z - phase)

def gw_warp(P, phase, pol):
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    h = h_of(z, phase)
    if pol == "plus":
        return np.stack([x*(1 + 0.5*h), y*(1 - 0.5*h), z], axis=-1)
    else:  # cross
        return np.stack([x + 0.5*h*y, y + 0.5*h*x, z], axis=-1)

# --- Tesseract topology -----------------------------------------------------------
V4_UNIT = np.array(list(product([-1.0, 1.0], repeat=4)))
EDGES = [(i, j) for i in range(16) for j in range(i + 1, 16)
         if np.sum(np.abs(V4_UNIT[i] - V4_UNIT[j])) == 2.0]

def project_4d_to_3d(v4, w_eye=2.6, scale=2.0):
    w = v4[..., 3]
    return v4[..., :3] * (scale / (w_eye - w))[..., None]

# Precompute every straight (unwarped) polyline ONCE: shape (n_edges_total, SAMPLES, 3)
t = np.linspace(0.0, 1.0, SAMPLES)
lines = []
for s in SHELLS:
    V3 = project_4d_to_3d(V4_UNIT * s)
    for i, j in EDGES:
        lines.append((1 - t)[:, None] * V3[i] + t[:, None] * V3[j])
LINES = np.array(lines)                                  # (E, S, 3)
Z_MID = 0.5 * (LINES[:, :-1, 2] + LINES[:, 1:, 2])       # strain sample points

def to_segments(warped):
    """(E, S, 3) -> list of (2,3) segments, plus matching flat midpoint z order."""
    segs = np.concatenate([warped[:, :-1, None, :], warped[:, 1:, None, :]], axis=2)
    return segs.reshape(-1, 2, 3)

STRAIGHT_SEGS = to_segments(LINES)

def radii_of(warped):
    mid = 0.5 * (warped[:, :-1, :] + warped[:, 1:, :])
    return np.linalg.norm(mid, axis=-1).ravel()

if COLOR_BY == "radius":
    r0   = radii_of(LINES)
    norm = mpl.colors.Normalize(vmin=r0.min(), vmax=r0.max())
    CMAP = RADIUS_CMAP
else:
    norm = mpl.colors.Normalize(vmin=-A_h, vmax=A_h)
    CMAP = STRAIN_CMAP

# --- Figure ------------------------------------------------------------------------
fig = plt.figure(figsize=(13.5, 7.0))
fig.patch.set_facecolor("white")
fig.suptitle("A gravitational wave bending every line of space",
             fontsize=16, fontweight="bold", x=0.45, y=0.97)
fig.subplots_adjust(left=0.0, right=0.86, top=1.02, bottom=0.04, wspace=0.0)

LIM = 1.05 * (1.0 + 0.5*A_h) * np.abs(LINES).max()
collections = []
for idx, (pol, sym) in enumerate([("plus", r"$h_+$"), ("cross", r"$h_\times$")]):
    ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
    ax.add_collection3d(Line3DCollection(STRAIGHT_SEGS, colors="0.72",
                                         linewidths=0.4, alpha=0.28))
    w0 = gw_warp(LINES, 0.0, pol)
    lc = Line3DCollection(to_segments(w0), linewidths=1.5)
    if COLOR_BY == "radius":
        lc.set_color(CMAP(norm(radii_of(w0))))
    else:
        lc.set_color(CMAP(norm(h_of(Z_MID, 0.0)).ravel()))
    ax.add_collection3d(lc)
    collections.append((lc, pol))

    ax.quiver(0, 0, -LIM*0.98, 0, 0, 0.55*LIM, color="black", lw=2.0,
              arrow_length_ratio=0.12)
    ax.text(0.12*LIM, 0.12*LIM, -0.62*LIM, r"$\vec{k}$", fontsize=13)
    ax.set_title(f"{sym} polarisation", fontsize=12, y=0.92)
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_zlim(-LIM, LIM)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=16, azim=34)
    ax.tick_params(labelsize=7, pad=-2)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$"); ax.set_zlabel(r"$z$")
    ax.grid(False)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_visible(False)

m = mpl.cm.ScalarMappable(norm=norm, cmap=CMAP); m.set_array([])
cb = fig.colorbar(m, ax=fig.axes, shrink=0.62, pad=0.04)
if COLOR_BY == "radius":
    cb.set_label(r"distance from centre $r$   (blue: near $\to$ red: far)", fontsize=9)
else:
    cb.set_label(r"local strain $h(z,t) = A\sin(kz - \omega t)$", fontsize=9)

fig.text(0.45, 0.06,
         "the strain pattern travels along $\\vec{k}$ at the speed of light — "
         "no medium, no central mass, amplitude exaggerated by $\\sim 10^{20}$",
         ha="center", fontsize=10, color="dimgrey", style="italic")

# --- Animation ------------------------------------------------------------------------
def update(frame):
    phase = 2*np.pi * frame / N_FRAMES
    for lc, pol in collections:
        w = gw_warp(LINES, phase, pol)
        lc.set_segments(to_segments(w))
        if COLOR_BY == "radius":
            lc.set_color(CMAP(norm(radii_of(w))))
        else:
            lc.set_color(CMAP(norm(h_of(Z_MID, phase)).ravel()))
    return [lc for lc, _ in collections]

anim = FuncAnimation(fig, update, frames=N_FRAMES, blit=False)

mp4 = "Thesis_Ready_Plots/anim_gw_tesseract.mp4"
anim.save(mp4, writer=FFMpegWriter(fps=FPS, bitrate=2400), dpi=110)
print("Saved:", mp4)

gif = "Thesis_Ready_Plots/anim_gw_tesseract.gif"
anim.save(gif, writer=PillowWriter(fps=FPS), dpi=72)
print("Saved:", gif)