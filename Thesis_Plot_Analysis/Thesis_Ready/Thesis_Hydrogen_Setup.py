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
fig.suptitle("Probing a gravitational wave with a hydrogen atom",
             fontsize=17, fontweight="bold", y=0.965)

axL = fig.add_subplot(1, 2, 1, projection="3d", computed_zorder=False)
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

# ── HYDROGEN ATOM ON THE WAVE (left panel) ──
ex, ey = 4.1, 1.3
ez = wave_z(ex, ey) + 0.05 + 0.10*sx/sz

# proton (small red dot) – shown as a tiny sphere
proton_R = 0.12
XS_prot, YS_prot, ZS_prot = visual_sphere(ex, ey, ez, proton_R)
prot_rgb = LightSource(315, 45).shade(ZS_prot, plt.cm.Reds, vert_exag=1.0,
                                      blend_mode="soft")
axL.plot_surface(XS_prot, YS_prot, ZS_prot, facecolors=prot_rgb,
                 rstride=1, cstride=1, linewidth=0, antialiased=True,
                 shade=False, zorder=15)

# electron cloud – “cloudy” effect: a scatter of small dots inside a sphere
cloud_R = 0.40
np.random.seed(42)                           # reproducibility
n_dots = 800
# uniform distribution on the surface of a sphere and random radii
theta_dots = np.arccos(1 - 2*np.random.rand(n_dots))
phi_dots   = 2*np.pi * np.random.rand(n_dots)
r_dots     = cloud_R * np.random.rand(n_dots)**(1/3)   # uniform in volume
dx = ex + r_dots * np.sin(theta_dots) * np.cos(phi_dots)
dy = ey + r_dots * np.sin(theta_dots) * np.sin(phi_dots)
dz = ez + r_dots * np.cos(theta_dots)
# colour: light blue with some transparency
axL.scatter(dx, dy, dz, c="#5dade2", alpha=0.08, s=12, edgecolor='none', zorder=14)


axL.view_init(elev=26, azim=-55)
axL.set_xlabel(r"$x$", fontsize=12, labelpad=8)
axL.set_ylabel(r"$y$", fontsize=12, labelpad=8)
axL.set_zticks([]); axL.grid(False)
for pane in (axL.xaxis.pane, axL.yaxis.pane, axL.zaxis.pane):
    pane.set_visible(False)
axL.zaxis.line.set_color((1, 1, 1, 0))

# ════════════════════════════════════ RIGHT PANEL — zoom on H atom
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

# --- proton (tiny red core) ---
XS_prot, YS_prot, ZS_prot = sphere(0, 0, 0, 0.10)
prot_rgb = LightSource(315, 45).shade(ZS_prot, plt.cm.Reds, vert_exag=1.0,
                                      blend_mode="soft")
axR.plot_surface(XS_prot, YS_prot, ZS_prot, facecolors=prot_rgb,
                 rstride=1, cstride=1, linewidth=0, antialiased=True,
                 shade=False, zorder=20)

# --- plus sign at the proton centre (middle of the atom) ---
plus_z = 0.05                  # slightly above the proton sphere to be visible
axR.text(0, -0.01, plus_z-0.07, "+", fontsize=14, fontweight="bold",
         ha="center", va="center", color="#67000d", zorder=25)

# --- "H atom" label right outside the electron cloud, on top ---
label_z = cloud_R + 0.2       # cloud radius = 0.65, so label at z = 0.80
axR.text(0, 0, label_z, "H atom", fontsize=11, fontweight="bold",
         ha="center", va="center", color="#08306B", zorder=25)

# --- electron cloud (cloudy, bubbly) ---
cloud_R = 0.65
np.random.seed(123)
n_dots = 1500
theta = np.arccos(1 - 2*np.random.rand(n_dots))
phi   = 2*np.pi * np.random.rand(n_dots)
r     = cloud_R * np.random.rand(n_dots)**(1/3)
cx_d = r * np.sin(theta) * np.cos(phi)
cy_d = r * np.sin(theta) * np.sin(phi)
cz_d = r * np.cos(theta)
axR.scatter(cx_d, cy_d, cz_d, c="#5dade2", alpha=0.04, s=10, edgecolor='none', zorder=19)

# faint outer envelope to give shape (very translucent sphere)
XS_env, YS_env, ZS_env = sphere(0, 0, 0, cloud_R)
env_rgb = LightSource(315, 45).shade(ZS_env, plt.cm.Blues, vert_exag=1.0, blend_mode="soft")
axR.plot_surface(XS_env, YS_env, ZS_env, facecolors=env_rgb,
                 rstride=1, cstride=1, linewidth=0, antialiased=True,
                 shade=False, alpha=0.08, zorder=18)

# --- orbital arc (the “orbit”) ---
orbit_radius = 0.85
orbit_angles = np.linspace(0, 1.7*np.pi, 200)   # a partial arc for motion
axR.plot(orbit_radius*np.cos(orbit_angles), orbit_radius*np.sin(orbit_angles),
         np.zeros_like(orbit_angles), color="crimson", lw=2.5, zorder=22,
         solid_capstyle="round")
# arrowhead on the orbit
tip_ang = orbit_angles[-1]
axR.quiver(orbit_radius*np.cos(tip_ang), orbit_radius*np.sin(tip_ang), 0,
           -np.sin(tip_ang), np.cos(tip_ang), 0,
           color="crimson", lw=2.5, arrow_length_ratio=0.2, length=0.25,
           zorder=23)


# --- incoming GW wavefronts (unchanged) ---
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

# --- bridge arrow + caption ---
fig.patches.append(
    mpatches.FancyArrowPatch((0.485, 0.52), (0.535, 0.52),
        transform=fig.transFigure, mutation_scale=22, lw=2.0,
        color="crimson", arrowstyle="-|>"))
_cap = (r"binary inspiral  $\rightarrow$  GW strain $h_{ij}$  $\rightarrow$  "
        r"spin coupling $-\frac{\kappa}{2}h_{\mu\nu}T^{\mu\nu}$  $\rightarrow$  "
        r"readout $\langle\hat{S}_z\rangle$  $\rightarrow$  correlate with LIGO event")
fig.text(0.5, 0.025, _cap, ha="center", fontsize=11, color="0.25", style="italic")

fig.subplots_adjust(left=0.0, right=1.0, bottom=0.06, top=0.93, wspace=0.0)
out = "Thesis_Ready_Plots/fig_thesisHydrogen_cloudy.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.show()
print("Saved:", out)