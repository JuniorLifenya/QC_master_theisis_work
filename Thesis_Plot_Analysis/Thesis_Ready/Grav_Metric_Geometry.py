"""
The metric tensor — a simple, precise picture.

LEFT  : the general metric g_{munu} as a 4x4 table.
        diagonal  -> lengths (scale along each axis)
        off-diag  -> angles  (tilt between axes; 0 if perpendicular)
        symmetric -> only 10 of 16 numbers are independent.

RIGHT : the Minkowski metric eta = diag(+1,-1,-1,-1) — flat spacetime.
        off-diagonals 0 (axes perpendicular), diagonal +-1 (unit scale),
        and crucially the SAME table at every point -> no curvature.

BRIDGE: g = eta + h — a gravitational wave is a tiny travelling ripple
        h_{munu} in the otherwise-flat table.  (Motivates the thesis.)

Signature (+,-,-,-), matching the thesis convention.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = "Thesis_Ready_Plots"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 12,
})

LEN_COL  = "#1F8A70"   # diagonal: lengths / scale
ANG_COL  = "#9B7EBD"   # off-diagonal: angles
ZERO_COL = "#ECECEC"   # Minkowski zeros
AX = ["t", "x", "y", "z"]

# ─── figure ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14.5, 9.2))
fig.patch.set_facecolor("white")

axL = fig.add_axes([0.045, 0.34, 0.40, 0.45])
axR = fig.add_axes([0.555, 0.34, 0.40, 0.45])

def draw_grid(ax, labels, colors, text_colors):
    ax.set_xlim(-0.45, 4.15)
    ax.set_ylim(-0.45, 4.15)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    for i in range(4):
        for j in range(4):
            ax.add_patch(FancyBboxPatch(
                (j+0.07, i+0.07), 0.86, 0.86,
                boxstyle="round,pad=0.035",
                facecolor=colors[i][j], edgecolor="black", lw=1.6,
                alpha=0.95, zorder=2))
            ax.text(j+0.5, i+0.5, labels[i][j], ha="center", va="center",
                    fontsize=14, color=text_colors[i][j],
                    fontweight="bold", zorder=3)
    # axis headers (columns on top, rows on left)
    for k in range(4):
        ax.text(k+0.5, -0.26, AX[k], ha="center", va="center",
                fontsize=13, color="#333", style="italic")
        ax.text(-0.28, k+0.5, AX[k], ha="center", va="center",
                fontsize=13, color="#333", style="italic")

# ─── LEFT: general metric ────────────────────────────────────────────
labels_g, colors_g, tcol_g = [], [], []
for i in range(4):
    lr, cr, tr = [], [], []
    for j in range(4):
        lr.append(rf"$g_{{{AX[i]}{AX[j]}}}$")
        cr.append(LEN_COL if i == j else ANG_COL)
        tr.append("white")
    labels_g.append(lr); colors_g.append(cr); tcol_g.append(tr)
draw_grid(axL, labels_g, colors_g, tcol_g)
axL.set_title(r"The metric  $g_{\mu\nu}$  at one point",
              fontsize=13.5, fontweight="bold", pad=20)

# ─── RIGHT: Minkowski ────────────────────────────────────────────────
labels_m, colors_m, tcol_m = [], [], []
for i in range(4):
    lr, cr, tr = [], [], []
    for j in range(4):
        if i == j:
            lr.append(r"$+1$" if i == 0 else r"$-1$")
            cr.append(LEN_COL); tr.append("white")
        else:
            lr.append(r"$0$")
            cr.append(ZERO_COL); tr.append("#9a9a9a")
    labels_m.append(lr); colors_m.append(cr); tcol_m.append(tr)
draw_grid(axR, labels_m, colors_m, tcol_m)
axR.set_title(r"Flat spacetime: Minkowski  $\eta_{\mu\nu}$",
              fontsize=13.5, fontweight="bold", pad=20)

# ─── top: title + the one rule ───────────────────────────────────────
fig.suptitle("The metric tensor: a table of numbers that measures spacetime",
             fontsize=16.5, fontweight="bold", y=0.975)

fig.text(0.5, 0.895,
         r"One rule — the dot product at each point:   "
         r"$\vec a \cdot \vec b \;=\; g_{\mu\nu}\,a^\mu b^\nu$"
         r"      $\Rightarrow$   length $|\vec a|=\sqrt{\vec a\cdot\vec a}$,   "
         r"angle $\cos\theta=\dfrac{\vec a\cdot\vec b}{|\vec a|\,|\vec b|}$",
         ha="center", va="center", fontsize=12.5,
         bbox=dict(boxstyle="round,pad=0.5", fc="#f5f7fa",
                   ec="#5B7BA5", lw=1.2))

# ─── legend / notes under LEFT ───────────────────────────────────────
# colour key swatches
def swatch(x, y, color):
    fig.patches.append(plt.Rectangle((x, y), 0.018, 0.026, transform=fig.transFigure,
                                     facecolor=color, edgecolor="black", lw=1.0,
                                     zorder=5))
swatch(0.160, 0.255, LEN_COL)
fig.text(0.186, 0.268, r"diagonal $\rightarrow$ lengths",
         fontsize=10.5, va="center")


swatch(0.160, 0.205, ANG_COL)
fig.text(0.186, 0.218, r"off-diagonal $\rightarrow$ angles",
         fontsize=10.5, va="center")
fig.text(0.186, 0.198, "tilt between axes (0 if perpendicular)",
         fontsize=8.8, va="center", color="0.4", style="italic")

fig.text(0.12, 0.158,
         r"symmetric: $g_{\mu\nu}=g_{\nu\mu}$  —  only 10 of the 16 are independent",
         fontsize=10, va="center", color="#333")

# ─── notes under RIGHT ───────────────────────────────────────────────
fig.text(0.575, 0.268,
         r"off-diagonals $=0$  $\Rightarrow$  all axes stay perpendicular",
         fontsize=10.5, va="center")
fig.text(0.575, 0.235,
         r"diagonal $\pm1$  $\Rightarrow$  unit scale on every axis",
         fontsize=10.5, va="center")
fig.text(0.575, 0.202,
         r"the same table at every point  $\Rightarrow$  no curvature, no gravity",
         fontsize=10.5, va="center", color="#9c2c2c")
fig.text(0.575, 0.158,
         r"(the single $+$ vs $-$ is all that makes it spacetime, not 4D space)",
         fontsize=9.2, va="center", color="0.45", style="italic")

# ─── bottom bridge to the thesis ─────────────────────────────────────
fig.patches.append(FancyBboxPatch(
    (0.10, 0.030), 0.80, 0.072, transform=fig.transFigure,
    boxstyle="round,pad=0.006,rounding_size=0.008",
    facecolor="#fff4e6", edgecolor="#D66000", lw=1.5, zorder=4))
fig.text(0.5, 0.078,
         r"$g_{\mu\nu}(x) \;=\; \eta_{\mu\nu} \;+\; h_{\mu\nu}(x)$"
         r"        a gravitational wave is a tiny travelling ripple "
         r"$h_{\mu\nu}$ in the flat table",
         ha="center", va="center", fontsize=13, color="#7a3a00", zorder=5)
fig.text(0.5, 0.050,
         r"(flat spacetime)  +  (the ripple your thesis is about)",
         ha="center", va="center", fontsize=9.5, color="#a05a20",
         style="italic", zorder=5)

out = os.path.join(OUT, "Metric_tensor.png")
fig.savefig(out, dpi=200, facecolor="white")
plt.show()
plt.close(fig)
print(f"Saved: {out}")