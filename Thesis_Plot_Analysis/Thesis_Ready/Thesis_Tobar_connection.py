import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyArrowPatch, FancyBboxPatch, Ellipse)
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection   # noqa: F401 (just in case)
import matplotlib.patches as mpatches

OUT = "Thesis_Ready_Plots"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.size": 11})

# ─── figure layout ──────────────────────────────────────────────────
fig = plt.figure(figsize=(15.5, 7.5))
# ★ CINEMATIC BACKGROUND ★
fig.patch.set_facecolor("white")   # deep off‑black; change to "#2b1e10" for wheat, "#1a3b47" for cyan

# --- Left panel: 3D electron ---
ax1 = fig.add_axes([0.025, 0.30, 0.255, 0.60], projection='3d',
                   computed_zorder=False)
L = 2.0
ax1.set_xlim(-L, L)
ax1.set_ylim(-L, L)
ax1.set_zlim(-L, L)
ax1.set_box_aspect((1, 1, 1))

# --- make 3D panes match the cinematic background ---
ax1.set_facecolor("white")


ax1.axis("off")                     # removes panes, ticks, grid
ax1.set_title("Single electron", fontsize=12, fontweight="bold", pad=0, color="black")

# --- B-field dipole loops (behind everything) ---
def dipole_loop(scale, plane_angle, npts=240):
    th = np.linspace(0.001, np.pi-0.001, npts)
    r  = scale * np.sin(th)**2
    rho = r * np.sin(th); zc = r * np.cos(th)
    rho = np.concatenate([rho, rho[::-1]])
    zc  = np.concatenate([zc,  zc[::-1]])
    side = np.concatenate([np.ones(npts), -np.ones(npts)])
    ca, sa = np.cos(plane_angle), np.sin(plane_angle)
    return rho*side*ca, rho*side*sa, zc

for ang in np.linspace(0, np.pi*3, 1, endpoint=False):
    for scl in (0.9, 1.3, 1.7):
        bx, by, bz = dipole_loop(scl, ang)
        ax1.plot(bx, by, bz, color="#7e57c2", lw=1.0, alpha=0.50, zorder=3)  # slightly more alpha for dark bg
ax1.text(-1.15, 1.7, -1.85, r"$\vec{B}$  (magnetic moment)", fontsize=11,
         color="#c4a0f0", zorder=26)   # lighter text for dark bg

# --- electron sphere ---
def sphere(cx, cy, cz, R, n=60):
    u = np.linspace(0, 2*np.pi, n); v = np.linspace(0, np.pi, n)
    return (cx + R*np.outer(np.cos(u), np.sin(v)),
            cy + R*np.outer(np.sin(u), np.sin(v)),
            cz + R*np.outer(np.ones_like(u), np.cos(v)))

XS, YS, ZS = sphere(0, 0, 0, 0.28)
e_rgb = LightSource(315, 45).shade(ZS, plt.cm.Blues, vert_exag=1.0, blend_mode="soft")
ax1.plot_surface(XS, YS, ZS, facecolors=e_rgb, rstride=1, cstride=1,
                 linewidth=0, antialiased=True, shade=False, zorder=20)
ax1.text(0, 0, -0.55, r"$e^-$", fontsize=13, fontweight="bold", ha="center",
         color="#b3d9ff", zorder=21)   # lighter blue

# --- intrinsic spin arrow ---
ax1.quiver(0, 0, 0.40, 0, 0, 0.95, color="#e6550d", lw=2.6,
           arrow_length_ratio=0.28, zorder=24)
ax1.text(0.12, 0, 1.2, r"$\vec{S}$", fontsize=14, fontweight="bold",
         color="#ff8c42", zorder=26)   # brighter orange

# --- curling arrow around spin axis ---
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

# --- momentum arrow (+x) ---
ax1.quiver(0, 0, 0, 1.55, 0, 0, color="#1b9e77", lw=3.4,
           arrow_length_ratio=0.16, zorder=24)
ax1.text(1.7, 0, 0.14, r"$\vec{p}$", fontsize=14, fontweight="bold",
         color="#66cdaa", zorder=26)   # lighter green

ax1.view_init(elev=15, azim=-58)

# ─── Right panels ─────────────────────────────────────────────────
ax2 = fig.add_axes([0.345, 0.30, 0.295, 0.60])
ax3 = fig.add_axes([0.700, 0.30, 0.275, 0.60])
ax_bottom = fig.add_axes([0.04, 0.025, 0.92, 0.22])

# ★ Set all 2D axes background to the same cinematic colour ★
for ax in [ax2, ax3, ax_bottom]:
    ax.set_facecolor("white")
    ax.axis("off")

# ════════════════════════════════════════════════════════════════════
# PANEL 2: cooled aluminium bar
# ════════════════════════════════════════════════════════════════════
ax2.set_xlim(-3.5, 3.5); ax2.set_ylim(-2.0, 2.0)
ax2.set_aspect("equal")

bar_x0, bar_x1, bar_h = -2.8, 2.8, 1.0
ax2.add_patch(FancyBboxPatch((bar_x0, -bar_h/2), bar_x1-bar_x0, bar_h,
                             boxstyle="round,pad=0.0,rounding_size=0.15",
                             facecolor="black", edgecolor="white", lw=2.0))
ax2.add_patch(Ellipse((bar_x0, 0), 0.20, bar_h, facecolor="#9aa8be",
                      edgecolor="#2a3a52", lw=1.5, zorder=3))
ax2.add_patch(Ellipse((bar_x1, 0), 0.20, bar_h, facecolor="#dee4ee",
                      edgecolor="#2a3a52", lw=1.5, zorder=3))
ax2.add_patch(Ellipse((0, 0.32), 5.2, 0.10, facecolor="white",
                      alpha=0.55, zorder=4))

# Atomic lattice
np.random.seed(11)
nx, ny = 22, 4
xs = np.linspace(bar_x0+0.35, bar_x1-0.35, nx)
ys = np.linspace(-0.28, 0.28, ny)
XA, YA = np.meshgrid(xs, ys)
XA[1::2] += (xs[1] - xs[0]) / 2
XA += np.random.normal(0, 0.010, XA.shape)
YA += np.random.normal(0, 0.008, YA.shape)
ax2.scatter(XA.ravel(), YA.ravel(), s=16, color="#8bb4f0", alpha=0.90,  # brighter dots
            edgecolor="none", zorder=5)

# Temperature label (now light on dark)
ax2.text(0, 1.55, r"$T \approx 1\,$mK", fontsize=13, ha="center",
         color="#b3e0ff", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.30", fc="#1c2a3a",
                   ec="#0d6da6", lw=1.2))

# Snowflakes (unchanged, still visible)
def flake(ax, x, y, s=0.08, c="#5db8e6"):
    for k in range(6):
        ang = k * np.pi/3
        ax.plot([x, x+s*np.cos(ang)], [y, y+s*np.sin(ang)],
                color=c, lw=0.9, alpha=0.85)

np.random.seed(82)
for _ in range(10):
    sx = np.random.uniform(-3.0, 3.0)
    sy = np.random.uniform(1.0, 1.6) if np.random.rand() > 0.5 else np.random.uniform(-1.6, -1.0)
    flake(ax2, sx, sy)

ax2.set_title("Many-body bar\n" r"cold Al cylinder",
              fontsize=12, fontweight="bold", pad=10, color="black")
ax2.text(0, -1.70,
         r"$M = 1800$ kg,  $v_s = 5100$ m/s,  $f_r = 150$ Hz",
         fontsize=9.5, ha="center", color="black", style="italic")   # lighter grey

# ════════════════════════════════════════════════════════════════════
# PANEL 3: bar driven by ḧ(t)
# ════════════════════════════════════════════════════════════════════
ax3.set_xlim(-3.5, 3.5); ax3.set_ylim(-2.0, 2.0)
ax3.set_aspect("equal")

amp_disp = 0.22
for stretch, alpha in [(+amp_disp, 0.30), (-amp_disp, 0.30)]:
    ax3.add_patch(FancyBboxPatch((bar_x0-stretch, -bar_h/2),
                                 (bar_x1-bar_x0)+2*stretch, bar_h,
                                 boxstyle="round,pad=0.0,rounding_size=0.15",
                                 facecolor="#c8d2e0", edgecolor="#3a4858",
                                 lw=1.2, alpha=alpha))

ax3.add_patch(FancyBboxPatch((bar_x0, -bar_h/2), bar_x1-bar_x0, bar_h,
                             boxstyle="round,pad=0.0,rounding_size=0.15",
                             facecolor="black", edgecolor="white", lw=1.6))
ax3.add_patch(Ellipse((0, 0.32), 5.2, 0.10, facecolor="white", alpha=0.55))

# Mode shape
zc = np.linspace(bar_x0+0.15, bar_x1-0.15, 200)
mode_y = 0.18 * np.sin(np.pi * (zc - bar_x0) / (bar_x1 - bar_x0))
ax3.plot(zc, mode_y, color="#f07241", lw=2.0, zorder=8)   # brightened
ax3.text(0, -0.19, r"$u(z) = \xi\sin(\pi z/L)$",
         fontsize=9.5, ha="center", color="#f07241",
         style="italic", fontweight="bold")

# End arrows
ax3.annotate("", xy=(bar_x0-0.45, 0), xytext=(bar_x0-0.10, 0),
             arrowprops=dict(arrowstyle="-|>", color="#ff6b6b", lw=2.2))
ax3.annotate("", xy=(bar_x1+0.45, 0), xytext=(bar_x1+0.10, 0),
             arrowprops=dict(arrowstyle="-|>", color="#ff6b6b", lw=2.2))

# Incoming GW
xw = np.linspace(-3.0, 3.0, 200)
for offset, alpha in [(0.0, 0.85), (0.45, 0.55), (0.9, 0.30)]:
    yw = 1.40 + 0.08 * np.sin(1.8 * xw - offset*4.5)
    ax3.plot(xw, yw, color="#3dbdab", lw=1.5, alpha=alpha)   # brighter teal
ax3.text(3.2, 1.40, r"$h(t)$", color="#3dbdab", fontsize=11,
         fontweight="bold", va="center")

# Phonon ladder
ladder_x = 2.7
ladder_y0 = -1.18
ladder_y1 = -0.58
ax3.hlines(ladder_y0, ladder_x-0.25, ladder_x+0.35, color="#b088f0", lw=2.5)  # lighter purple
ax3.hlines(ladder_y1, ladder_x-0.25, ladder_x+0.35, color="#b088f0", lw=2.5)
ax3.text(ladder_x+0.38, ladder_y0, r"$|n=0\rangle$",
         color="#004169", fontsize=10, va="center", fontweight="bold")
ax3.text(ladder_x+0.38, ladder_y1, r"$|n=1\rangle$",
         color="#004169", fontsize=10, va="center", fontweight="bold")
ax3.annotate("", xy=(ladder_x, ladder_y1 - 0.03), xytext=(ladder_x, ladder_y0 + 0.03),
             arrowprops=dict(arrowstyle="-|>", color="#004169", lw=2.0))
ax3.text(ladder_x-0.32, (ladder_y0 + ladder_y1)/2, r"$\hat b^\dagger$",
         color="#004169", fontsize=12, va="center", ha="right", fontweight="bold")

ax3.text(0, -1.55,
         r"single-phonon emission:  $|0\rangle \to |1\rangle\;$"
         r"detected as quantum jump",
         fontsize=10, ha="center", color="#004169", fontweight="bold")   # light purple

ax3.set_title("Driven by " r"$\ddot h(t)$" "\n"
              r"fundamental mode  $\omega_1 = \pi v_s/L$",
              fontsize=12, fontweight="bold", pad=10, color="black")

# ════════════════════════════════════════════════════════════════════
# Inter-panel arrows
# ════════════════════════════════════════════════════════════════════
fig.patches.append(FancyArrowPatch(
    (0.285, 0.60), (0.342, 0.60),
    transform=fig.transFigure, arrowstyle="->",
    color="black", lw=2.4, mutation_scale=24))   # light grey
fig.text(0.313, 0.66, "embed in", fontsize=10.5, ha="center", color="black",
         style="italic", fontweight="bold")
fig.text(0.313, 0.555, "lattice", fontsize=10.5, ha="center", color="black",
         style="italic", fontweight="bold")

fig.patches.append(FancyArrowPatch(
    (0.645, 0.60), (0.697, 0.60),
    transform=fig.transFigure, arrowstyle="->",
    color="black", lw=2.4, mutation_scale=24))
fig.text(0.671, 0.66, "drive with", fontsize=10.5, ha="center", color="black",
         style="italic", fontweight="bold")
fig.text(0.671, 0.555, r"$\ddot h(t)$", fontsize=11.5, ha="center",
         color="black", fontweight="bold")

# ════════════════════════════════════════════════════════════════════
# Bottom: Hamiltonian chain + Tobar rate
# ════════════════════════════════════════════════════════════════════
ax_bottom.add_patch(FancyBboxPatch(
    (0.01, 0.05), 0.98, 0.92,
    boxstyle="round,pad=0.005,rounding_size=0.012",
    transform=ax_bottom.transAxes,
    facecolor="white", edgecolor="#4d7ed3", lw=1.4, alpha=0.95))   # dark box, cyan edge

ax_bottom.text(0.50, 0.83, "Hamiltonian chain",
    fontsize=10.5, ha="center", va="center", transform=ax_bottom.transAxes,
    fontweight="bold", color="#8ec5ff")

ax_bottom.text(0.50, 0.62,
    r"$\dfrac{m_e}{4}\,\omega^2\,h\,\hat x^2"
    r"\;\;\Longrightarrow\;\;"
    r"\dfrac{1}{2}\,M\,\omega^2\,\hat\xi^{\,2}"
    r"\;\;\Longrightarrow\;\;"
    r"\hat H_{\rm int} \propto \ddot h(t)\,(\hat b + \hat b^\dagger)$",
    fontsize=13, ha="center", va="center", transform=ax_bottom.transAxes,
    color="black")

ax_bottom.text(0.345, 0.82, "single particle (thesis)", fontsize=8.5, ha="center",
               va="center", transform=ax_bottom.transAxes, color="black",
               style="italic")
ax_bottom.text(0.450, 0.45, "collective coordinate", fontsize=8.5, ha="center",
               va="center", transform=ax_bottom.transAxes, color="black",
               style="italic")
ax_bottom.text(0.62, 0.821, "phonon (Linear) form", fontsize=8.5, ha="center",
               va="center", transform=ax_bottom.transAxes, color="black",
               style="italic")

ax_bottom.text(0.50, 0.25,
    r"Optimal mass:  $M_{\rm opt} \propto \omega^3/v_s^{\,2} = 1800\,$kg"
    r"  $\;\Longrightarrow\;$  "
    r"$\Gamma_{\rm stim} \approx 1\,$Hz"
    r"  for $h_0 = 5\times 10^{-22}$ at $150\,$Hz",
    fontsize=11.5, ha="center", va="center", transform=ax_bottom.transAxes,
    color="black")

ax_bottom.add_patch(FancyBboxPatch(
    (0.517, 0.13), 0.058, 0.24,
    boxstyle="round,pad=0.005,rounding_size=0.01",
    transform=ax_bottom.transAxes,
    facecolor="none", edgecolor="#ff6b6b", lw=1.8, zorder=10))   # bright red

fig.suptitle(
    r"From single-particle kinetic strain to many-body phonon excitation: "
    r"the route to single-graviton detection",
    fontsize=13, fontweight="bold", y=1.00, color="black")

out_path = os.path.join(OUT, "fig_electron_to_bar.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
plt.close(fig)
print(f"Saved: {out_path}")