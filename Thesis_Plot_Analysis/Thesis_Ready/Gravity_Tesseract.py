import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib as mpl
from itertools import product
import os

os.makedirs("Thesis_Ready_Plots", exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# 1.  Tesseract TOPOLOGY (unit size) — built once, reused for every shell
# ═════════════════════════════════════════════════════════════════════════════
V4_UNIT = np.array(list(product([-1.0, 1.0], repeat=4)))          # (16, 4)
EDGES = [(i, j) for i in range(16) for j in range(i + 1, 16)
         if np.sum(np.abs(V4_UNIT[i] - V4_UNIT[j])) == 2.0]       # 32 edges


def project_4d_to_3d(v4, w_eye=2.6, scale=2.0):
    w = v4[..., 3]
    s = scale / (w_eye - w)
    return v4[..., :3] * s[..., None]


def gravitational_warp(P, strength=0.72, soft=0.80, cap=0.90):
    P = np.asarray(P, float)
    r = np.linalg.norm(P, axis=-1, keepdims=True)
    f = np.clip(strength / (r ** 2 + soft ** 2), 0.0, cap)
    return P * (1.0 - f)


# ═════════════════════════════════════════════════════════════════════════════
# 2.  NESTED SHELLS  — THE ONLY NEW IDEA
# ─────────────────────────────────────────────────────────────────────────────
#   A "shell" is the SAME tesseract scaled in 4-D by a radius.  Multiplying the
#   unit vertices by s ∈ SHELLS and projecting gives concentric tesseracts; the
#   warp then bends every shell toward the mass.  More shells = denser mesh =
#   a smoother sense of how the whole region is curved.
# ═════════════════════════════════════════════════════════════════════════════
SHELLS = [0.45, 0.70, 0.95, 1.20]      # add/remove entries to taste
SAMPLES = 60
t = np.linspace(0.0, 1.0, SAMPLES)

straight_segments, warped_segments, warped_radii = [], [], []
for s in SHELLS:
    V3 = project_4d_to_3d(V4_UNIT * s)               # this shell, projected
    for i, j in EDGES:
        P, Q = V3[i], V3[j]
        line = (1 - t)[:, None] * P + t[:, None] * Q
        bent = gravitational_warp(line)
        for k in range(SAMPLES - 1):
            straight_segments.append([line[k], line[k + 1]])
            warped_segments.append([bent[k], bent[k + 1]])
            warped_radii.append(np.linalg.norm(0.5 * (bent[k] + bent[k + 1])))
warped_radii = np.asarray(warped_radii)

# ═════════════════════════════════════════════════════════════════════════════
# 3.  Figure
# ═════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(11, 9))
fig.patch.set_facecolor("white")
ax = fig.add_subplot(111, projection="3d")
fig.suptitle("How gravity truly bends every line", fontsize=15, fontweight="bold")

# faint un-warped reference lattice (all shells)
ax.add_collection3d(Line3DCollection(straight_segments, colors="0.7",
                                     linewidths=0.4, alpha=0.18))

# warped nested lattice, coloured by distance to the mass (near = bright)
norm = mpl.colors.Normalize(vmin=warped_radii.min(), vmax=warped_radii.max())
cmap = mpl.cm.turbo_r
ax.add_collection3d(Line3DCollection(warped_segments,
                                     colors=cmap(norm(warped_radii)),
                                     linewidths=1.5))

# central mass
u = np.linspace(0, 2 * np.pi, 40)
v = np.linspace(0, np.pi, 40)
rs = 1.06
ax.plot_surface(rs * np.outer(np.cos(u), np.sin(v)),
                rs * np.outer(np.sin(u), np.sin(v)),
                rs * np.outer(np.ones_like(u), np.cos(v)),
                color="wheat", linewidth=0, antialiased=True, zorder=10)

m = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
m.set_array([])
cb = fig.colorbar(m, ax=ax, shrink=0.75, pad=0.10)
cb.set_label(r"distance to mass $r$   (near $\to$ bright, deeper $\Phi$)", fontsize=9)

# auto-fit the cube to the outermost warped points
LIM = 1.05 * np.abs(np.array(warped_segments)).max()
ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_zlim(-LIM, LIM)
ax.set_box_aspect((1, 1, 1))
ax.view_init(elev=22, azim=35)
ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$"); ax.set_zlabel(r"$z$")
ax.grid(False)

plt.tight_layout()
out = "Thesis_Ready_Plots/fig_tesseract_nested.png"
plt.savefig(out, dpi=330, bbox_inches="tight")
plt.show()
print("Saved:", out)