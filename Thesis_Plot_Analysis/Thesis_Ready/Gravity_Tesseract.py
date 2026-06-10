import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection  # noqa: F401
import matplotlib as mpl
from itertools import product
import os

os.makedirs("Thesis_Ready_Plots", exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# 1.  The tesseract  (4-cube):  16 vertices, 32 edges
# ─────────────────────────────────────────────────────────────────────────────
#   Vertices are all sign combinations (±1, ±1, ±1, ±1).  Two vertices share an
#   edge iff they differ in EXACTLY ONE coordinate (Hamming distance 1, i.e. an
#   L1 separation of 2 in our ±1 convention).
# ═════════════════════════════════════════════════════════════════════════════
V4 = np.array(list(product([-1.0, 1.0], repeat=4)))          # (16, 4)
EDGES = [(i, j) for i in range(16) for j in range(i + 1, 16)
         if np.sum(np.abs(V4[i] - V4[j])) == 2.0]            # 32 edges


# ═════════════════════════════════════════════════════════════════════════════
# 2.  Project 4-D -> 3-D  (perspective "cube-within-a-cube" / Schlegel view)
# ─────────────────────────────────────────────────────────────────────────────
#   A point is scaled by its distance from a 4-D eye placed along +w:
#       (x,y,z) -> (x,y,z) / (w_eye - w)
#   The w = +1 cell projects large, the w = -1 cell projects small and nested.
# ═════════════════════════════════════════════════════════════════════════════
def project_4d_to_3d(v4, w_eye=2.6, scale=2.0):
    w = v4[..., 3]
    s = scale / (w_eye - w)
    return v4[..., :3] * s[..., None]


V3 = project_4d_to_3d(V4)                                     # (16, 3)


# ═════════════════════════════════════════════════════════════════════════════
# 3.  The gravitational warp  —  how gravity "truly" acts on every line
# ─────────────────────────────────────────────────────────────────────────────
#   A mass at the origin contracts space radially inward.  We model the warp as
#   a smooth, monotone radial pull
#         r  ->  r · (1 - f(r)) ,     f(r) = strength / (r² + soft²),  capped < 1
#   Because f depends NON-LINEARLY on radius, a straight Euclidean segment — whose
#   distance-to-centre varies along its length — is mapped to a CURVED arc that
#   bows toward the mass.  Every edge of the lattice is bent; the connecting
#   (inner↔outer) struts bend most, the face edges sag inward.  This is the
#   discrete shadow of "straight lines (geodesics) curve in the presence of mass".
# ═════════════════════════════════════════════════════════════════════════════
def gravitational_warp(P, strength=0.72, soft=0.80, cap=0.90):
    P = np.asarray(P, float)
    r = np.linalg.norm(P, axis=-1, keepdims=True)
    f = np.clip(strength / (r ** 2 + soft ** 2), 0.0, cap)
    return P * (1.0 - f)


# ═════════════════════════════════════════════════════════════════════════════
# 4.  Sample every edge, warp it, and collect coloured segments
# ─────────────────────────────────────────────────────────────────────────────
#   We tessellate each edge into many short segments so the warped curve is
#   smooth, and colour each segment by its distance to the centre (closer = hotter)
#   to read off "how deep in the potential" each piece of the lattice lies.
# ═════════════════════════════════════════════════════════════════════════════


SAMPLES = 60
t = np.linspace(0.0, 1.0, SAMPLES)

straight_segments = []        # faint reference lattice (un-warped)
warped_segments = []          # bent lattice
warped_radii = []             # midpoint radius of each warped sub-segment

for i, j in EDGES:
    P, Q = V3[i], V3[j]
    line = (1 - t)[:, None] * P + t[:, None] * Q          # straight 3-D edge
    bent = gravitational_warp(line)                       # curved edge

    for k in range(SAMPLES - 1):
        straight_segments.append([line[k], line[k + 1]])
        seg = [bent[k], bent[k + 1]]
        warped_segments.append(seg)
        warped_radii.append(np.linalg.norm(0.5 * (bent[k] + bent[k + 1])))

warped_radii = np.asarray(warped_radii)

# ═════════════════════════════════════════════════════════════════════════════
# 5.  Figure
# ═════════════════════════════════════════════════════════════════════════════


fig = plt.figure(figsize=(11, 9))
fig.patch.set_facecolor("white")
ax = fig.add_subplot(111, projection="3d")
fig.suptitle("How gravity truly bends every line", fontsize=15, fontweight="bold")
ax.set_title(r"a tesseract whose edges curve into a central mass", fontsize=10, pad=2)

# faint un-warped tesseract for reference (the "flat-space" lattice)
ax.add_collection3d(Line3DCollection(straight_segments, colors="0.7",
                                     linewidths=0.5, alpha=0.30))

# warped tesseract, coloured by depth in the potential (closer to mass = brighter)
norm = mpl.colors.Normalize(vmin=warped_radii.min(), vmax=warped_radii.max())
cmap = mpl.cm.turbo_r            # reversed: small radius -> bright, large -> dark
colors = cmap(norm(warped_radii))
lc = Line3DCollection(warped_segments, colors=colors, linewidths=1.7)
ax.add_collection3d(lc)

# the central mass
u = np.linspace(0, 2 * np.pi, 40)
v = np.linspace(0, np.pi, 40)
rs = 0.16
ax.plot_surface(rs * np.outer(np.cos(u), np.sin(v)),
                rs * np.outer(np.sin(u), np.sin(v)),
                rs * np.outer(np.ones_like(u), np.cos(v)),
                color="black", linewidth=0, antialiased=True, zorder=10)

# colour bar: distance to the centre  (≈ depth in the gravitational potential)
m = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
m.set_array([])
cb = fig.colorbar(m, ax=ax, shrink=0.55, pad=0.10)
cb.set_label(r"distance to mass $r$   (near $\to$ bright, deeper $\Phi$)", fontsize=9)

# ═════════════════════════════════════════════════════════════════════════════
# 6.  Camera, limits, export
# ═════════════════════════════════════════════════════════════════════════════
LIM = 1.5
ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_zlim(-LIM, LIM)
ax.set_box_aspect((1, 1, 1))
ax.view_init(elev=22, azim=35)
ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$"); ax.set_zlabel(r"$z$")
ax.grid(True)

plt.tight_layout()
out = "Thesis_Ready_Plots/fig_tesseract_gravity_warp.png"
plt.savefig(out, dpi=330, bbox_inches="tight")
plt.show()
print("Saved:", out)