import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyArrowPatch, FancyBboxPatch, Ellipse)
from matplotlib.colors import LightSource
import matplotlib.patches as mpatches

OUT = "Thesis_Ready_Plots"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.size": 11})

# ─── figure layout ──────────────────────────────────────────────────
fig = plt.figure(figsize=(15.5, 7.5))
fig.patch.set_facecolor("white")    # ← back to white

# --- Left panel: 3D electron (spin only) ---
ax1 = fig.add_axes([0.025, 0.30, 0.255, 0.60], projection='3d',
                   computed_zorder=False)
L = 2.0
ax1.set_xlim(-L, L)
ax1.set_ylim(-L, L)
ax1.set_zlim(-L, L)
ax1.set_box_aspect((1, 1, 1))
ax1.set_facecolor("white")          # ← axes background white
ax1.set_title("Single electron (spin)", fontsize=12, fontweight="bold",
              pad=0, color="black")  # ← title black

# ── B-field dipole loops ────────────────────────
def dipole_loop(scale, plane_angle, npts=240):
    th = np.linspace(0.001, np.pi-0.001, npts)
    r  = scale * np.sin(th)**2
    rho = r * np.sin(th)
    zc = r * np.cos(th)
    rho = np.concatenate([rho, rho[::-1]])
    zc  = np.concatenate([zc,  zc[::-1]])
    side = np.concatenate([np.ones(npts), -np.ones(npts)])
    ca, sa = np.cos(plane_angle), np.sin(plane_angle)
    return rho*side*ca, rho*side*sa, zc

for ang in np.linspace(0, np.pi*3, 1, endpoint=False):
    for scl in (0.9, 1.3, 1.7):
        bx, by, bz = dipole_loop(scl, ang)
        ax1.plot(bx, by, bz, color="#7e57c2", lw=1.0, alpha=0.50, zorder=3)
# B-field label — now dark purple
ax1.text(-1.15, 1.7, -1.85, r"$\vec{B}$  (magnetic moment)",
         fontsize=11, color="#5e3c99", zorder=26)    # ← dark purple

# ── electron sphere ─────────────────────────────
def sphere(cx, cy, cz, R, n=60):
    u = np.linspace(0, 2*np.pi, n)
    v = np.linspace(0, np.pi, n)
    return (cx + R*np.outer(np.cos(u), np.sin(v)),
            cy + R*np.outer(np.sin(u), np.sin(v)),
            cz + R*np.outer(np.ones_like(u), np.cos(v)))

XS, YS, ZS = sphere(0, 0, 0, 0.28)
e_rgb = LightSource(315, 45).shade(ZS, plt.cm.Blues, vert_exag=1.0,
                                   blend_mode="soft")
ax1.plot_surface(XS, YS, ZS, facecolors=e_rgb, rstride=1, cstride=1,
                 linewidth=0, antialiased=True, shade=False, zorder=20)
# electron label — dark blue for visibility on white
ax1.text(0, 0, -0.55, r"$e^-$", fontsize=13, fontweight="bold",
         ha="center", color="#08306B", zorder=21)    # ← dark blue

# ── intrinsic spin arrow ───────────────────────
ax1.quiver(0, 0, 0.40, 0, 0, 0.95, color="#e6550d", lw=2.6,
           arrow_length_ratio=0.28, zorder=24)
# Spin label — dark orange
ax1.text(0.12, 0, 1.2, r"$\vec{S}$", fontsize=14, fontweight="bold",
         color="#cc4c02", zorder=26)    # ← darker orange

# ── curling arrow around spin axis ────────────
def curl_arrow(z0, R, a0, a1, color, lw, n=60):
    a = np.linspace(a0, a1, n)
    ax1.plot(R*np.cos(a), R*np.sin(a), np.full_like(a, z0),
             color=color, lw=lw, zorder=24, solid_capstyle="round")
    tip  = np.array([R*np.cos(a1), R*np.sin(a1), z0])
    tang = np.array([-np.sin(a1), np.cos(a1), 0.0])
    ax1.quiver(tip[0], tip[1], tip[2], tang[0], tang[1], tang[2],
               color=color, lw=lw, arrow_length_ratio=1.0,
               length=0.22, normalize=True, zorder=25)
curl_arrow(0.40, 0.42, np.deg2rad(10), np.deg2rad(310), "#e6550d", 3.0)

# ── Restore full 3D room ─────────────────────
ax1.view_init(elev=15, azim=-58)
ax1.axis("off")      # hides everything: panes, grid, ticks, labels

# ─── Right panels ─────────────────────────────────────────────────
ax2 = fig.add_axes([0.345, 0.30, 0.295, 0.60])
ax3 = fig.add_axes([0.700, 0.30, 0.275, 0.60])
ax_bottom = fig.add_axes([0.04, 0.025, 0.92, 0.22])

for ax in [ax2, ax3, ax_bottom]:
    ax.set_facecolor("white")        # ← back to white
    ax.axis("off")                   # keep schematic panels clean

# ════════════════════════════════════════════════════════════════════
# PANEL 2: Spin‑Aligned Bar
# ════════════════════════════════════════════════════════════════════
ax2.set_xlim(-3.5, 3.5)
ax2.set_ylim(-2.0, 2.0)
ax2.set_aspect("equal")

bar_x0, bar_x1, bar_h = -2.8, 2.8, 1.0
# Bar body – darker outline for contrast
ax2.add_patch(FancyBboxPatch((bar_x0, -bar_h/2), bar_x1-bar_x0, bar_h,
                             boxstyle="round,pad=0.0,rounding_size=0.15",
                             facecolor="#c8d2e0", edgecolor="#2a3a52", lw=2.0))
ax2.add_patch(Ellipse((bar_x0, 0), 0.20, bar_h, facecolor="#9aa8be",
                      edgecolor="#2a3a52", lw=1.5, zorder=3))
ax2.add_patch(Ellipse((bar_x1, 0), 0.20, bar_h, facecolor="#dee4ee",
                      edgecolor="#2a3a52", lw=1.5, zorder=3))
# Highlight ellipse – now slightly grey so it doesn't disappear on white
ax2.add_patch(Ellipse((0, 0.32), 5.2, 0.10, facecolor="#e0e0e0",
                      alpha=0.55, zorder=4))

# Spin arrows inside the bar (unchanged)
np.random.seed(42)
nx, ny = 18, 3
xs_spin = np.linspace(bar_x0+0.45, bar_x1-0.45, nx)
ys_spin = np.linspace(-0.24, 0.24, ny)
XA_spin, YA_spin = np.meshgrid(xs_spin, ys_spin)
XA_spin[1::2] += (xs_spin[1] - xs_spin[0])/2
for x, y in zip(XA_spin.ravel(), YA_spin.ravel()):
    ax2.arrow(x-0.06, y, 0.12, 0, head_width=0.07, head_length=0.05,
              fc="#e6550d", ec="none", lw=0, zorder=7)

# Tiny bar magnets
def magnet(ax, x, y, s=0.1):
    rect = mpatches.Rectangle((x-s*0.5, y-s*0.15), s, s*0.3,
                              angle=0, facecolor="#b0b0b0", edgecolor="#333",
                              lw=0.6, zorder=5)
    ax.add_patch(rect)
    ax.plot([x+s*0.3, x+s*0.45, x+s*0.3], [y-s*0.13, y, y+s*0.13],
            color="#d62728", lw=0.8, zorder=6)
    ax.plot([x-s*0.3, x-s*0.45, x-s*0.3], [y-s*0.13, y, y+s*0.13],
            color="#1f77b4", lw=0.8, zorder=6)

np.random.seed(92)
for _ in range(20):
    mx = np.random.uniform(-3.0, 3.0)
    my = np.random.uniform(1.0, .9) if np.random.rand()>0.5 else np.random.uniform(-1.6, -1.0)
    magnet(ax2, mx, my)

# Labels — now dark text, white bbox with dark edge
ax2.text(0, 1.55, r"Spin‑aligned paramagnet", fontsize=13, ha="center",
         color="#cc4c02", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.30", fc="white",
                   ec="#cc4c02", lw=1.2))
ax2.set_title("Spin‑based Tobar\nspin ensemble inside a bar",
              fontsize=12, fontweight="bold", pad=10, color="black")


# ════════════════════════════════════════════════════════════════════
# PANEL 3: spin‑driven by GW (Zeeman coupling)
# ════════════════════════════════════════════════════════════════════
ax3.set_xlim(-3.5, 3.5)
ax3.set_ylim(-2.0, 2.0)
ax3.set_aspect("equal")

ax3.add_patch(FancyBboxPatch((bar_x0, -bar_h/2), bar_x1-bar_x0, bar_h,
                             boxstyle="round,pad=0.0,rounding_size=0.15",
                             facecolor="#c8d2e0", edgecolor="#2a3a52", lw=1.6))
ax3.add_patch(Ellipse((0, 0.32), 5.2, 0.10, facecolor="#e0e0e0", alpha=0.55))

# Incoming GW waves – keep colour, darken arrow
xw = np.linspace(-3.0, 3.0, 200)
for offset, alpha in [(0.0, 0.85), (0.45, 0.55), (0.9, 0.30)]:
    yw = 1.40 + 0.08 * np.sin(1.8 * xw - offset*4.5)
    ax3.plot(xw, yw, color="#3dbdab", lw=1.5, alpha=alpha)
ax3.text(3.2, 1.40, r"$h_{ij}$", color="#2a8a7e", fontsize=11,
         fontweight="bold", va="center")    # darker teal

# Zeeman splitting level diagram
ladder_x = 2.7
ladder_y0 = -1.18
ladder_y1 = -0.58
ax3.hlines(ladder_y0, ladder_x-0.25, ladder_x+0.35, color="#e6550d", lw=2.5)
ax3.hlines(ladder_y1, ladder_x-0.25, ladder_x+0.35, color="#e6550d", lw=2.5)
ax3.text(ladder_x+0.38, ladder_y0, r"$|\downarrow\rangle$",
         color="#cc4c02", fontsize=10, va="center", fontweight="bold")
ax3.text(ladder_x+0.38, ladder_y1, r"$|\uparrow\rangle$",
         color="#cc4c02", fontsize=10, va="center", fontweight="bold")
ax3.annotate("", xy=(ladder_x, ladder_y1 - 0.03), xytext=(ladder_x, ladder_y0 + 0.03),
             arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=2.0))
ax3.text(ladder_x-0.32, (ladder_y0+ladder_y1)/2,
         r"$h_{ij}\;\rightarrow\;$ spin flip",
         color="#d62728", fontsize=9, va="center", ha="right", fontweight="bold")

ax3.text(0, -1.55,
         r"$\hat H_{\rm Zeeman} = -\frac{\kappa}{2m}\hat{\vec S}\cdot\hat{\vec B}_g$"
         r"  ;  gradient $(\nabla h)$ required",
         fontsize=10, ha="center", color="#5e3c99", fontweight="bold")   # dark purple

ax3.set_title("GW drives spin precession\nZeeman coupling in TT gauge",
              fontsize=12, fontweight="bold", pad=10, color="black")

# ════════════════════════════════════════════════════════════════════
# Inter-panel arrows
# ════════════════════════════════════════════════════════════════════
fig.patches.append(FancyArrowPatch(
    (0.285, 0.60), (0.342, 0.60),
    transform=fig.transFigure, arrowstyle="->",
    color="gray", lw=2.4, mutation_scale=24))
fig.text(0.313, 0.66, "spin ensemble", fontsize=10.5, ha="center",
         color="black", style="italic", fontweight="bold")
fig.text(0.313, 0.555, "in bar", fontsize=10.5, ha="center",
         color="black", style="italic", fontweight="bold")

fig.patches.append(FancyArrowPatch(
    (0.645, 0.60), (0.697, 0.60),
    transform=fig.transFigure, arrowstyle="->",
    color="gray", lw=2.4, mutation_scale=24))
fig.text(0.671, 0.66, "GW gradient", fontsize=10.5, ha="center",
         color="black", style="italic", fontweight="bold")
fig.text(0.671, 0.555, r"$(\nabla h)$", fontsize=11.5, ha="center",
         color="black", fontweight="bold")

# ════════════════════════════════════════════════════════════════════
# Bottom: Hamiltonian chain
# ════════════════════════════════════════════════════════════════════
ax_bottom.add_patch(FancyBboxPatch(
    (0.01, 0.05), 0.98, 0.92,
    boxstyle="round,pad=0.005,rounding_size=0.012",
    transform=ax_bottom.transAxes,
    facecolor="white", edgecolor="gray", lw=1.4, alpha=0.95))  # light box

ax_bottom.text(0.50, 0.83, "Spin‑based detection chain",
    fontsize=10.5, ha="center", va="center", transform=ax_bottom.transAxes,
    fontweight="bold", color="black")

ax_bottom.text(0.50, 0.62,
    r"$ -\frac{\kappa}{2m}\hat{\vec S}\!\cdot\!\hat{\vec B}_g$"
    r"$\;\Longrightarrow\;$"
    r"$ -\frac{\kappa}{2m}\sum_i\hat{\vec S}_i\!\cdot\!\hat{\vec B}_g(\vec  r_i)$"
    r"$\;\Longrightarrow\;$"
    r"$ \hat H_{\rm eff} = \lambda\, \hat S_{\rm tot}\, \nabla h$",
    fontsize=13, ha="center", va="center", transform=ax_bottom.transAxes,
    color="black")

ax_bottom.text(0.26, 0.82, "single spin (thesis)", fontsize=8.5, ha="center",
               va="center", transform=ax_bottom.transAxes, color="0.3",
               style="italic")
ax_bottom.text(0.53, 0.45, "collective spin operator", fontsize=8.5, ha="center",
               va="center", transform=ax_bottom.transAxes, color="0.3",
               style="italic")
ax_bottom.text(0.75, 0.821, "spin‑strain coupling", fontsize=8.5, ha="center",
               va="center", transform=ax_bottom.transAxes, color="0.3",
               style="italic")

ax_bottom.text(0.50, 0.25,
    r"Required:  $|\nabla h|\neq 0$"
    r"  $\;\Longrightarrow\;$  "
    r"coherent $\Delta m_s = \pm 1$ transitions"
    r"  detectable via cavity QED or SQUID",
    fontsize=11.5, ha="center", va="center", transform=ax_bottom.transAxes,
    color="black")

ax_bottom.add_patch(FancyBboxPatch(
    (0.685, 0.15), 0.085, 0.2,
    boxstyle="round,pad=0.005,rounding_size=0.01",
    transform=ax_bottom.transAxes,
    facecolor="none", edgecolor="#d62728", lw=1.8, zorder=10))

fig.suptitle(
    r"Outlook: From single‑spin Zeeman coupling to a collective spin‑based quantum GW sensor",
    fontsize=13, fontweight="bold", y=1.00, color="black")

out_path = os.path.join(OUT, "fig_spin_outlook.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.show()
plt.close(fig)
print(f"Saved: {out_path}")