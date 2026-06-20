import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
import os

os.makedirs("Thesis_Ready_Plots", exist_ok=True)

# ════════════════════════════════════════════════════ LEFT PANEL — GW sheet
d_binary   = 1.0
x1, y1     = -d_binary/2, 0.0
x2, y2     =  d_binary/2, 0.0
depth_well = -0.85
sigma_well = 0.35
A, k, decay = 0.75, 2.5, 3.5

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

fig = plt.figure(figsize=(15.5, 7.4))
fig.patch.set_facecolor("white")
# ── Updated suptitle to reflect hydrogen atom ──
fig.suptitle("Probing a gravitational wave with a hydrogen atom",
             fontsize=17, fontweight="bold", y=0.965)

axL = fig.add_subplot(1, 2, 1, projection="3d", computed_zorder=False)
# ── Updated panel title ──
axL.set_title("a binary radiates; a hydrogen atom rides the wave",
              fontsize=11.5, color="dimgrey", y=0.97)

z_floor, z_top = -1.6, 1.2
axL.set_xlim(-SPAN, SPAN); axL.set_ylim(-SPAN, SPAN); axL.set_zlim(z_floor, z_top)
BOX = (2*SPAN, 2*SPAN, 3.2)
axL.set_box_aspect(BOX)
sx = BOX[0]/(2*SPAN); sy = BOX[1]/(2*SPAN); sz = BOX[2]/(z_top - z_floor)

def visual_sphere(cx, cy, cz, R, n=80):
    u = np.linspace(0, 2*np.pi, n); v = np.linspace(0, np.pi, n)
    rx, ry, rz = R*sx/sx, R*sx/sy, R*sx/sz
    return (cx + rx*np.outer(np.cos(u), np.sin(v)),
            cy + ry*np.outer(np.sin(u), np.sin(v)),
            cz + rz*np.outer(np.ones_like(u), np.cos(v)))

ls_bh = LightSource(315, 45)
R_bh = 0.35; cz_list = []
for cx, cy in [(x1, y1), (x2, y2)]:
    cz = wave_z(cx, cy) + 0.55 * R_bh * sx/sz
    cz_list.append(cz)
    XS, YS, ZS = visual_sphere(cx, cy, cz, R_bh)
    rgb = ls_bh.shade(ZS, plt.cm.inferno, vert_exag=1.0, blend_mode="soft")
    axL.plot_surface(XS, YS, ZS, facecolors=rgb, rstride=1, cstride=1,
                     linewidth=0, antialiased=True, shade=False, zorder=10)
cz1, cz2 = cz_list

th = np.linspace(0, 2*np.pi, 200)
axL.plot((d_binary/2)*np.cos(th), (d_binary/2)*np.sin(th),
         np.full_like(th, (cz1+cz2)/2), ls="--", lw=1.2, color="black",
         alpha=0.67, zorder=9)
axL.text(0, 0, (cz1+cz2)/2 + R_bh*sx/sz + 0.35, "Binary Source",
         fontsize=10, fontweight="bold", ha="center", zorder=11)

axL.plot_surface(X, Y, np.zeros_like(Z), color="gainsboro", alpha=0.15,
                 linewidth=0, antialiased=True, zorder=0)
norm_R = (R_dist - R_dist.min()) / (R_dist.max() - R_dist.min())
base_colors = plt.get_cmap("turbo")(norm_R)[:, :, :3]
rgb_surface = LightSource(315, 50).shade_rgb(base_colors, Z, vert_exag=2.0,
                                             blend_mode="soft")
axL.plot_surface(X, Y, Z, facecolors=rgb_surface, rstride=1, cstride=1,
                 linewidth=0, antialiased=True, shade=False, alpha=0.95, zorder=2)
axL.plot_wireframe(X, Y, Z, color="k", alpha=0.35, linewidth=0.4,
                   rstride=6, cstride=6, zorder=3)
axL.contour(X, Y, Z, levels=np.linspace(-A, A, 11), zdir="z", offset=z_floor,
            cmap="Greys", linewidths=0.8, alpha=0.45, zorder=1)

# ── REPLACE THE ELECTRON WITH A HYDROGEN ATOM ──
ex, ey = 4.1, 1.3
ez = wave_z(ex, ey) + 0.05 + 0.10*sx/sz   # base height of the atom on the wave

# ---- proton (small red nucleus) ----
proton_R = 0.12
XS_prot, YS_prot, ZS_prot = visual_sphere(ex, ey, ez, proton_R)
prot_rgb = LightSource(315, 45).shade(ZS_prot, plt.cm.Reds, vert_exag=1.0,
                                      blend_mode="soft")
axL.plot_surface(XS_prot, YS_prot, ZS_prot, facecolors=prot_rgb,
                 rstride=1, cstride=1, linewidth=0, antialiased=True,
                 shade=False, zorder=15)

# ---- electron cloud (translucent blue sphere) ----
cloud_R = 0.40
XS_cloud, YS_cloud, ZS_cloud = visual_sphere(ex, ey, ez, cloud_R)
cloud_rgb = LightSource(315, 45).shade(ZS_cloud, plt.cm.Blues, vert_exag=1.0,
                                       blend_mode="soft")
axL.plot_surface(XS_cloud, YS_cloud, ZS_cloud, facecolors=cloud_rgb,
                 rstride=1, cstride=1, linewidth=0, antialiased=True,
                 shade=False, alpha=0.35, zorder=14)

# ---- label ----
axL.text(ex+1.2, ey-0.88, ez + 1.15, "H atom  (spin-½)",
         fontsize=11, fontweight="bold", ha="center", color="#08306B", zorder=16)

# ---- (optional) keep the red precession rings if they belong to the spin system ----
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

# ════════════════════════════════════ RIGHT PANEL — hydrogen atom zoom
axR = fig.add_subplot(1, 2, 2, projection="3d", computed_zorder=False)
axR.set_title("The interaction: Gravitational wave--Hydrogen Atom--Quantum Sensor",
              fontsize=11.5, color="dimgrey", y=0.97)

L = 2.0
axR.set_xlim(-L, L); axR.set_ylim(-L, L); axR.set_zlim(-L, L)
axR.set_box_aspect((1, 1, 1))

def sphere(cx, cy, cz, R, n=60):
    u = np.linspace(0, 2*np.pi, n); v = np.linspace(0, np.pi, n)
    return (cx + R*np.outer(np.cos(u), np.sin(v)),
            cy + R*np.outer(np.sin(u), np.sin(v)),
            cz + R*np.outer(np.ones_like(u), np.cos(v)))

# --- B-field dipole loops (atom’s magnetic moment, unchanged) ---
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
        axR.plot(bx, by, bz, color="#7e57c2", lw=1.0, alpha=0.40, zorder=3)
axR.text(-1.15, 1.7, -1.85, r"$\vec{B}$  (magnetic moment)", fontsize=11,
         color="#7e57c2", zorder=26)

# --- hydrogen atom: proton + electron cloud (replaces the bare electron) ---
# proton (centre)
XS_prot, YS_prot, ZS_prot = sphere(0, 0, 0, 0.13)
prot_rgb = LightSource(315, 45).shade(ZS_prot, plt.cm.Reds, vert_exag=1.0,
                                      blend_mode="soft")
axR.plot_surface(XS_prot, YS_prot, ZS_prot, facecolors=prot_rgb,
                 rstride=1, cstride=1, linewidth=0, antialiased=True,
                 shade=False, zorder=20)
axR.text(0, 0, -0.48, "p", fontsize=12, fontweight="bold", ha="center",
         color="#67000d", zorder=21)

# electron cloud (translucent blue sphere) – represents the probability density
cloud_R = 0.48
XS_cloud, YS_cloud, ZS_cloud = sphere(0, 0, 0, cloud_R)
cloud_rgb = LightSource(315, 45).shade(ZS_cloud, plt.cm.Blues, vert_exag=1.0,
                                       blend_mode="soft")
axR.plot_surface(XS_cloud, YS_cloud, ZS_cloud, facecolors=cloud_rgb,
                 rstride=1, cstride=1, linewidth=0, antialiased=True,
                 shade=False, alpha=0.30, zorder=19)
# label the hydrogen atom
axR.text(0, 0.7, 0, r"H atom", fontsize=11, fontweight="bold",
         ha="center", color="#08306B", zorder=22)

# --- intrinsic spin: axis arrow (still the electron’s spin) ---
axR.quiver(0, 0, 0.40, 0, 0, 0.95, color="#e6550d", lw=2.6,
           arrow_length_ratio=0.28, zorder=24)
axR.text(0.12, 0, 1.2, r"$\vec{S}$", fontsize=14, fontweight="bold",
         color="#e6550d", zorder=26)
def curl_arrow(z0, R, a0, a1, color, lw, n=60):
    a = np.linspace(a0, a1, n)
    axR.plot(R*np.cos(a), R*np.sin(a), np.full_like(a, z0),
             color=color, lw=lw, zorder=24, solid_capstyle="round")
    tip  = np.array([R*np.cos(a1), R*np.sin(a1), z0])
    tang = np.array([-np.sin(a1), np.cos(a1), 0.0])
    axR.quiver(tip[0], tip[1], tip[2], tang[0], tang[1], tang[2],
               color=color, lw=lw, arrow_length_ratio=1.0,
               length=0.22, normalize=True, zorder=25)
curl_arrow(0.40, 0.42, np.deg2rad(10), np.deg2rad(310), "#e6550d", 3.0)

# --- momentum arrow removed (atom is stationary) ---
# (previously: axR.quiver for \vec{p}, now omitted)

# --- incoming GW wavefronts from the left (unchanged) ---
gw_cmap = plt.get_cmap("turbo")
yy = np.linspace(-1.0, 1.0, 30); zz = np.linspace(-1.0, 1.0, 30)
YYg, ZZg = np.meshgrid(yy, zz)
for i, xw in enumerate([-1.95, -1.6, -1.25]):
    col = gw_cmap(0.12 + 0.18*i)
    Xg = xw + 0.16*(1 - (YYg**2 + ZZg**2)/2.0)
    axR.plot_surface(Xg, YYg, ZZg, color=col, alpha=0.18,
                     linewidth=0, antialiased=True, zorder=2)
    rim = np.linspace(0, 2*np.pi, 60)
    axR.plot(xw + 0*rim, 0.95*np.cos(rim), 0.95*np.sin(rim),
             color=col, lw=1.5, alpha=0.6, zorder=2)
axR.quiver(-1.7, 0, -1.6, 1.2, 0, 0, color="black", lw=2.2,
           arrow_length_ratio=0.2, zorder=5)
axR.text(-1.45, 0, -1.9, r"GW  $h_{ij}$", fontsize=12, fontweight="bold",
         ha="center", zorder=6)

# --- quantum sensor (unchanged) ---
sensor_c = np.array([1.55, 1.35, 1.45])
ev1 = np.array([0.45, -0.10, 0.18]); ev2 = np.array([0.05, 0.42, -0.18])
pts = np.array([sensor_c+ev1+ev2, sensor_c+ev1-ev2, sensor_c-ev1-ev2, sensor_c-ev1+ev2])
axR.add_collection3d(Poly3DCollection([pts], facecolor="#b3cde3",
                                      edgecolor="0.3", alpha=0.85, zorder=15))
lens_c = sensor_c + np.array([-0.14, -0.19, -0.07])
XSs, YSs, ZSs = sphere(*lens_c, 0.12)
axR.plot_surface(XSs, YSs, ZSs, color="#08519C", alpha=0.9, linewidth=0, zorder=16)
axR.text(sensor_c[0]+0.05, sensor_c[1]+0.15, sensor_c[2]+0.30, "quantum\nsensor",
         fontsize=11, fontweight="bold", ha="center", color="#08519C", zorder=17)
axR.plot([lens_c[0], 0.14], [lens_c[1], 0.14], [lens_c[2], 0.22],
         ls=(0, (4, 3)), lw=1.8, color="#08519C", alpha=0.9, zorder=14)

axR.view_init(elev=18, azim=-58)
axR.set_xticks([]); axR.set_yticks([]); axR.set_zticks([]); axR.grid(False)
for pane in (axR.xaxis.pane, axR.yaxis.pane, axR.zaxis.pane):
    pane.set_visible(False)
for line in (axR.xaxis.line, axR.yaxis.line, axR.zaxis.line):
    line.set_color((1, 1, 1, 0))

# --- bridge arrow + caption (unchanged) ---
fig.patches.append(
    mpatches.FancyArrowPatch((0.485, 0.52), (0.535, 0.52),
        transform=fig.transFigure, mutation_scale=22, lw=2.0,
        color="crimson", arrowstyle="-|>"))
_cap = (r"binary inspiral  $\rightarrow$  GW strain $h_{ij}$  $\rightarrow$  "
        r"spin coupling $-\frac{\kappa}{2}h_{\mu\nu}T^{\mu\nu}$  $\rightarrow$  "
        r"readout $\langle\hat{S}_z\rangle$  $\rightarrow$  correlate with LIGO event")
fig.text(0.5, 0.025, _cap, ha="center", fontsize=11, color="0.25", style="italic")

fig.subplots_adjust(left=0.0, right=1.0, bottom=0.06, top=0.93, wspace=0.0)
out = "Thesis_Ready_Plots/fig_thesis_core.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.show()
print("Saved:", out)