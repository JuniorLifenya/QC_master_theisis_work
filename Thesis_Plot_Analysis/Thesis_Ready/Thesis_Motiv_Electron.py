import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
import os

os.makedirs("Thesis_Ready_Plots", exist_ok=True)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_zlim(-1.6, 1.2)
ax.set_box_aspect((1, 1, 1))


# RIGHT PANEL — classical electron zoom

L = 2.0
ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_zlim(-L, L)
ax.set_box_aspect((1, 1, 1))

def sphere(cx, cy, cz, R, n=60):
    u = np.linspace(0, 2*np.pi, n); v = np.linspace(0, np.pi, n)
    return (cx + R*np.outer(np.cos(u), np.sin(v)),
            cy + R*np.outer(np.sin(u), np.sin(v)),
            cz + R*np.outer(np.ones_like(u), np.cos(v)))

# --- B-field dipole loops (drawn first, behind everything) ---
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
        ax.plot(bx, by, bz, color="#7e57c2", lw=1.0, alpha=0.40, zorder=3)
ax.text(-1.15, 1.7, -1.85, r"$\vec{B}$  (magnetic moment)", fontsize=11,
         color="#7e57c2", zorder=26)

# --- electron ---
XS, YS, ZS = sphere(0, 0, 0, 0.28)
e_rgb = LightSource(315, 45).shade(ZS, plt.cm.Blues, vert_exag=1.0, blend_mode="soft")
ax.plot_surface(XS, YS, ZS, facecolors=e_rgb, rstride=1, cstride=1,
                 linewidth=0, antialiased=True, shade=False, zorder=20)
ax.text(0, 0, -0.55, r"$e^-$", fontsize=13, fontweight="bold", ha="center",
         color="#08306B", zorder=21)

# --- intrinsic spin: axis arrow + curling arrow ---
ax.quiver(0, 0, 0.40, 0, 0, 0.95, color="#e6550d", lw=2.6,
           arrow_length_ratio=0.28, zorder=24)
ax.text(0.12, 0, 1.2, r"$\vec{S}$", fontsize=14, fontweight="bold",
         color="#e6550d", zorder=26)
def curl_arrow(z0, R, a0, a1, color, lw, n=60):
    a = np.linspace(a0, a1, n)
    ax.plot(R*np.cos(a), R*np.sin(a), np.full_like(a, z0),
             color=color, lw=lw, zorder=24, solid_capstyle="round")
    tip  = np.array([R*np.cos(a1), R*np.sin(a1), z0])
    tang = np.array([-np.sin(a1), np.cos(a1), 0.0])
    ax.quiver(tip[0], tip[1], tip[2], tang[0], tang[1], tang[2],
               color=color, lw=lw, arrow_length_ratio=1.0,
               length=0.22, normalize=True, zorder=25)
curl_arrow(0.40, 0.42, np.deg2rad(10), np.deg2rad(310), "#e6550d", 3.0)


# --- momentum: forward arrow (+x) ---
ax.quiver(0, 0, 0, 1.55, 0, 0, color="#1b9e77", lw=3.4,
           arrow_length_ratio=0.16, zorder=24)
ax.text(1.7, 0, 0.14, r"$\vec{p}$", fontsize=14, fontweight="bold",
         color="#1b9e77", zorder=26)

ax.view_init(elev=12, azim=-58)

ax.xaxis.pane.set_color('wheat')
ax.yaxis.pane.set_color('wheat')
ax.zaxis.pane.set_color('wheat')

# Example: Using a dashed line for the B-field loops
ax.plot(bx, by, bz, color="#7e57c2", lw=1.0, linestyle='--', alpha=0.40)


fig.subplots_adjust(left=0.0, right=1.0, bottom=0.06, top=0.93, wspace=0.0)
out = "Thesis_Ready_Plots/fig_thesis_electron.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.show()
print("Saved:", out)