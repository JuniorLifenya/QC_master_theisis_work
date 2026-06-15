"""
THESIS CENTREPIECE — two-panel storyboard
==========================================
LEFT : fig_gravitational_wave sheet (Gaussian wells, tanh envelope, SPAN=6,
       turbo radial colormap, LightSource hill-shading) with electron on wave.
RIGHT: 3-D spin zoom — GW wavefronts drive precession cone; quantum sensor
       fires sinusoidal measurement-field wavefronts diagonally toward electron.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
import os

os.makedirs("Thesis_Ready_Plots", exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
#  LEFT PANEL — exact sheet from fig_gravitational_wave.py  (first block)
# ══════════════════════════════════════════════════════════════════════════════
d_binary   = 1.0
x1, y1     = -d_binary/2, 0.0
x2, y2     =  d_binary/2, 0.0
depth_well = -0.85
sigma_well = 0.35
A     = 0.75
k     = 2.5
decay = 3.5

def wave_z(x, y):
    r = np.sqrt(x**2 + y**2)
    well1 = depth_well * np.exp(-((x-x1)**2 + (y-y1)**2) / (2*sigma_well**2))
    well2 = depth_well * np.exp(-((x-x2)**2 + (y-y2)**2) / (2*sigma_well**2))
    env   = 0.5 * (1.0 + np.tanh((r - 1.2) / 0.6))
    wave  = env * A * np.sin(k * r) * np.exp(-r / decay)
    return well1 + well2 + wave

N, SPAN = 200, 6.0
gx   = np.linspace(-SPAN, SPAN, N)
X, Y = np.meshgrid(gx, gx)
Z    = wave_z(X, Y)
R_dist = np.sqrt(X**2 + Y**2)

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15.5, 7.4))
fig.patch.set_facecolor("white")
fig.suptitle("Probing a gravitational wave with a single spin-½ particle",
             fontsize=17, fontweight="bold", y=0.965)

axL = fig.add_subplot(1, 2, 1, projection="3d", computed_zorder=False)
axL.set_title("a binary radiates; one electron rides the wave",
              fontsize=11.5, color="dimgrey", y=0.97)

z_floor, z_top = -1.6, 1.2
axL.set_xlim(-SPAN, SPAN); axL.set_ylim(-SPAN, SPAN); axL.set_zlim(z_floor, z_top)
BOX = (2*SPAN, 2*SPAN, 3.2)
axL.set_box_aspect(BOX)
sx = BOX[0]/(2*SPAN)
sy = BOX[1]/(2*SPAN)
sz = BOX[2]/(z_top - z_floor)

def visual_sphere(cx, cy, cz, R, n=80):
    u = np.linspace(0, 2*np.pi, n); v = np.linspace(0, np.pi, n)
    rx, ry, rz = R*sx/sx, R*sx/sy, R*sx/sz
    return (cx + rx*np.outer(np.cos(u), np.sin(v)),
            cy + ry*np.outer(np.sin(u), np.sin(v)),
            cz + rz*np.outer(np.ones_like(u), np.cos(v)))

ls_bh = LightSource(315, 45)

# binary black holes
R_bh = 0.35
cz_list = []
for cx, cy in [(x1, y1), (x2, y2)]:
    z_well = wave_z(cx, cy)
    cz = z_well + 0.55 * R_bh * sx/sz
    cz_list.append(cz)
    XS, YS, ZS = visual_sphere(cx, cy, cz, R_bh)
    rgb = ls_bh.shade(ZS, plt.cm.inferno, vert_exag=1.0, blend_mode="soft")
    axL.plot_surface(XS, YS, ZS, facecolors=rgb, rstride=1, cstride=1,
                     linewidth=0, antialiased=True, shade=False, zorder=10)
cz1, cz2 = cz_list

th = np.linspace(0, 2*np.pi, 200)
orb_r = d_binary / 2
axL.plot(orb_r*np.cos(th), orb_r*np.sin(th),
         np.full_like(th, (cz1+cz2)/2),
         ls="--", lw=1.2, color="black", alpha=0.67, zorder=9)
axL.text(0, 0, (cz1+cz2)/2 + R_bh*sx/sz + 0.35, "Binary Source",
         fontsize=10, fontweight="bold", ha="center", zorder=11)

# flat reference + warped sheet
axL.plot_surface(X, Y, np.zeros_like(Z), color="gainsboro", alpha=0.15,
                 linewidth=0, antialiased=True, zorder=0)
norm_R = (R_dist - R_dist.min()) / (R_dist.max() - R_dist.min())
base_colors = plt.get_cmap("jet")(norm_R)[:, :, :3]
rgb_surface = LightSource(azdeg=315, altdeg=50).shade_rgb(
                  base_colors, Z, vert_exag=2.0, blend_mode="soft")
axL.plot_surface(X, Y, Z, facecolors=rgb_surface, rstride=1, cstride=1,
                 linewidth=0, antialiased=True, shade=False, alpha=0.95, zorder=2)
axL.plot_wireframe(X, Y, Z, color="k", alpha=0.35,
                   linewidth=0.4, rstride=6, cstride=6, zorder=3)
levels = np.linspace(-A, A, 11)
axL.contour(X, Y, Z, levels=levels, zdir="z", offset=z_floor,
            cmap="Greys", linewidths=0.8, alpha=0.45, zorder=1)

# ── electron riding the outer wave ───────────────────────────────────────────
ex, ey = 4.1, 1.3
ez = wave_z(ex, ey) + 0.05 + 0.10*sx/sz
XS, YS, ZS = visual_sphere(ex, ey, ez, 0.18)
e_rgb = LightSource(315, 45).shade(ZS, plt.cm.Blues, vert_exag=1.0, blend_mode="soft")
axL.plot_surface(XS, YS, ZS, facecolors=e_rgb, rstride=1, cstride=1,
                 linewidth=0, antialiased=True, shade=False, zorder=12)
# spin arrow
axL.quiver(ex, ey, ez, 0, 0, 1.0, color="#1b9e77", lw=2.4,
           arrow_length_ratio=0.32, zorder=13)
axL.text(ex, ey, ez + 1.15, r"$e^-$  (spin-½)",
         fontsize=11, fontweight="bold", ha="center", color="#08306B", zorder=14)
# highlight rings
ring = np.linspace(0, 2*np.pi, 80)
for rr in (0.52, 0.60):
    axL.plot(ex + rr*np.cos(ring), ey + rr*np.sin(ring),
             np.full_like(ring, ez), color="crimson", lw=1.8, zorder=13)

axL.view_init(elev=26, azim=-55)
axL.set_xlabel(r"$x$", fontsize=12, labelpad=8)
axL.set_ylabel(r"$y$", fontsize=12, labelpad=8)
axL.set_zticks([]); axL.grid(False)
for pane in (axL.xaxis.pane, axL.yaxis.pane, axL.zaxis.pane):
    pane.set_visible(False)
axL.zaxis.line.set_color((1, 1, 1, 0))

# ══════════════════════════════════════════════════════════════════════════════
#  RIGHT PANEL — 3-D spin zoom (GW → precession → sensor readout)
# ══════════════════════════════════════════════════════════════════════════════
axR = fig.add_subplot(1, 2, 2, projection="3d", computed_zorder=False)
axR.set_title("the electron's spin is driven, then read out",
              fontsize=11.5, color="dimgrey", y=0.97)

L = 2.0
axR.set_xlim(-L, L); axR.set_ylim(-L, L); axR.set_zlim(-L, L)
axR.set_box_aspect((1, 1, 1))

def sphere(cx, cy, cz, R, n=60):
    u = np.linspace(0, 2*np.pi, n); v = np.linspace(0, np.pi, n)
    return (cx + R*np.outer(np.cos(u), np.sin(v)),
            cy + R*np.outer(np.sin(u), np.sin(v)),
            cz + R*np.outer(np.ones_like(u), np.cos(v)))

# electron at origin
XS, YS, ZS = sphere(0, 0, 0, 0.28)
e_rgb = LightSource(315, 45).shade(ZS, plt.cm.Blues, vert_exag=1.0, blend_mode="soft")
axR.plot_surface(XS, YS, ZS, facecolors=e_rgb, rstride=1, cstride=1,
                 linewidth=0, antialiased=True, shade=False, zorder=20)
axR.text(0, 0, -0.52, r"$e^-$",
         fontsize=13, fontweight="bold", ha="center", color="#08306B", zorder=21)

# quantisation axis
axR.plot([0,0],[0,0],[-1.5,1.7], ls="--", lw=1.3, color="0.5", zorder=5)
axR.text(0.12, 0, 1.75, r"$\hat{z}$", fontsize=12, color="0.4", zorder=6)

# precession cone
theta = np.deg2rad(28); Hc = 1.45; rc = Hc * np.tan(theta)
uu = np.linspace(0, 2*np.pi, 60); hh = np.linspace(0, Hc, 2)
UU, HH = np.meshgrid(uu, hh)
axR.plot_surface((HH/Hc)*rc*np.cos(UU), (HH/Hc)*rc*np.sin(UU), HH,
                 color="#fdae6b", alpha=0.18, linewidth=0, antialiased=True, zorder=8)
axR.plot(rc*np.cos(uu), rc*np.sin(uu), np.full_like(uu, Hc),
         ls="--", lw=1.1, color="#e6550d", alpha=0.8, zorder=9)
phi = np.deg2rad(35)
sx_, sy_, sz_ = rc*np.cos(phi), rc*np.sin(phi), Hc
axR.quiver(0, 0, 0, sx_, sy_, sz_, color="#e6550d", lw=3.2,
           arrow_length_ratio=0.16, zorder=22)
axR.text(sx_*1.05, sy_*1.05, sz_*1.05+0.12, r"$\vec{S}$",
         fontsize=13, fontweight="bold", color="#e6550d", zorder=23)
arc = np.linspace(np.deg2rad(35), np.deg2rad(105), 30)
axR.plot(rc*np.cos(arc), rc*np.sin(arc), np.full_like(arc, Hc),
         color="#e6550d", lw=2.0, zorder=10)
axR.text(0, rc+0.15, Hc+0.18, r"$\omega_L$",
         fontsize=11, color="#e6550d", ha="center", zorder=11)

# incoming GW wavefronts from the left (turbo-coloured curved sheets)
gw_cmap = plt.get_cmap("turbo")
yy = np.linspace(-1.0, 1.0, 30); zz = np.linspace(-1.0, 1.0, 30)
YYg, ZZg = np.meshgrid(yy, zz)
for i, xw in enumerate([-1.9, -1.45, -1.0]):
    col = gw_cmap(0.12 + 0.18*i)
    Xg = xw + 0.18*(1 - (YYg**2 + ZZg**2)/2.0)
    axR.plot_surface(Xg, YYg, ZZg, color=col, alpha=0.22,
                     linewidth=0, antialiased=True, zorder=4)
    rim = np.linspace(0, 2*np.pi, 60)
    axR.plot(xw + 0*rim, 0.95*np.cos(rim), 0.95*np.sin(rim),
             color=col, lw=1.6, alpha=0.7, zorder=4)
axR.quiver(-1.6, 0, -1.55, 1.2, 0, 0, color="black", lw=2.2,
           arrow_length_ratio=0.2, zorder=5)
axR.text(-1.35, 0, -1.85, r"GW  $h_{ij}$",
         fontsize=12, fontweight="bold", ha="center", zorder=6)

# quantum sensor — tilted panel + lens aperture
sensor_c = np.array([1.45, 1.25, 1.35])
ev1 = np.array([ 0.45, -0.10,  0.18])
ev2 = np.array([ 0.05,  0.42, -0.18])

def sensor_quad(c, e1, e2, color, alpha):
    pts = np.array([c+e1+e2, c+e1-e2, c-e1-e2, c-e1+e2])
    axR.add_collection3d(
        Poly3DCollection([pts], facecolor=color, edgecolor="0.3",
                         alpha=alpha, zorder=15))

sensor_quad(sensor_c, ev1, ev2, "#b3cde3", 0.85)
lens_c = sensor_c + np.array([-0.12, -0.12, -0.22])
XSs, YSs, ZSs = sphere(*lens_c, 0.12)
axR.plot_surface(XSs, YSs, ZSs, color="#08519C", alpha=0.9,
                 linewidth=0, zorder=16)
axR.text(sensor_c[0]+0.05, sensor_c[1]+0.15, sensor_c[2]+0.28,
         "quantum\nsensor", fontsize=11, fontweight="bold",
         ha="center", color="#08519C", zorder=17)
axR.text(0.55, 0.45, 0.7, r"$\langle \hat{S}_z\rangle$",
         fontsize=11, color="#08519C", zorder=18)

# sinusoidal measurement-field wavefronts from sensor toward electron
target    = np.array([0.12, 0.12, 0.18])
beam_vec  = target - lens_c
beam_len  = np.linalg.norm(beam_vec)
b_hat     = beam_vec / beam_len
arb       = np.array([0, 0, 1]) if abs(b_hat[2]) < 0.9 else np.array([1, 0, 0])
p1        = np.cross(b_hat, arb); p1 /= np.linalg.norm(p1)

perp = np.linspace(-0.28, 0.28, 50)
for i, s0 in enumerate(np.linspace(0.12*beam_len, 0.88*beam_len, 5)):
    centre = lens_c + s0*b_hat
    along_offset = 0.10 * np.sin(np.pi * perp / 0.28)
    axR.plot(centre[0] + perp*p1[0] + along_offset*b_hat[0],
             centre[1] + perp*p1[1] + along_offset*b_hat[1],
             centre[2] + perp*p1[2] + along_offset*b_hat[2],
             color="#534AB7", lw=1.8-i*0.20, alpha=0.85-i*0.13, zorder=14)

axR.quiver(*(lens_c + 0.82*beam_vec), *(0.18*beam_vec),
           color="#534AB7", lw=1.8, arrow_length_ratio=0.35, zorder=15)
mid = lens_c + 0.48*beam_vec
axR.text(mid[0]-0.28, mid[1]+0.18, mid[2]+0.22,
         "meas.\nfield", fontsize=9, color="#534AB7", ha="center", zorder=16)

axR.view_init(elev=18, azim=-58)
axR.set_xticks([]); axR.set_yticks([]); axR.set_zticks([])
axR.grid(False)
for pane in (axR.xaxis.pane, axR.yaxis.pane, axR.zaxis.pane):
    pane.set_visible(False)
for line in (axR.xaxis.line, axR.yaxis.line, axR.zaxis.line):
    line.set_color((1, 1, 1, 0))

# ── bridging arrow + bottom caption ──────────────────────────────────────────
fig.patches.append(
    mpatches.FancyArrowPatch((0.485, 0.52), (0.535, 0.52),
        transform=fig.transFigure, mutation_scale=22, lw=2.0,
        color="crimson", arrowstyle="-|>"))

fig.text(0.5, 0.025,
    r"binary inspiral  $\rightarrow$  GW strain $h_{ij}$  $\rightarrow$  "
    r"spin coupling $-\frac{\kappa}{2}h_{\mu\nu}T^{\mu\nu}$  $\rightarrow$  "
    r"readout $\langle\hat{S}_z\rangle$  $\rightarrow$  correlate with LIGO event",
    ha="center", fontsize=11, color="0.25", style="italic")

fig.subplots_adjust(left=0.0, right=1.0, bottom=0.06, top=0.93, wspace=0.0)

out = "Thesis_Ready_Plots/fig_thesis_core.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.show()
print("Saved:", out)