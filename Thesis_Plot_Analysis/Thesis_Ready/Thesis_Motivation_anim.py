import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
import os

os.makedirs("anim_GW_sheet", exist_ok=True)

# ------------------ wave geometry (identical to your static figure) ------------------
d_binary, x1, y1, x2, y2 = 1.0, -0.5, 0.0, 0.5, 0.0
depth_well, sigma_well = -0.85, 0.35
A, k, decay = 0.75, 2.5, 3.5

def wave_z(x, y):
    r = np.sqrt(x**2 + y**2)
    well1 = depth_well * np.exp(-((x-x1)**2 + (y-y1)**2) / (2*sigma_well**2))
    well2 = depth_well * np.exp(-((x-x2)**2 + (y-y2)**2) / (2*sigma_well**2))
    env   = 0.5 * (1.0 + np.tanh((r - 1.2) / 0.6))
    wave  = env * A * np.sin(k * r) * np.exp(-r / decay)
    return well1 + well2 + wave

# surface grid (static)
N, SPAN = 200, 6.0
gx = np.linspace(-SPAN, SPAN, N)
X, Y = np.meshgrid(gx, gx)
Z = wave_z(X, Y)

# electron path: move from near the binary centre outward along a slightly curved line
path_t = np.linspace(0, 1, 60)           # 60 frames
start_r, end_r = 1.0, 5.0
path_r = start_r + (end_r - start_r)*path_t
path_theta = np.pi/4 + 0.3*np.sin(2*np.pi*path_t)   # a little wobble
ex_path = path_r * np.cos(path_theta)
ey_path = path_r * np.sin(path_theta)
ez_path = np.array([wave_z(ex_path[i], ey_path[i]) + 0.10 for i in range(len(path_t))])

# sphere helper (you already have it)
def visual_sphere(cx, cy, cz, R, n=40):
    u = np.linspace(0, 2*np.pi, n); v = np.linspace(0, np.pi, n)
    return (cx + R*np.outer(np.cos(u), np.sin(v)),
            cy + R*np.outer(np.sin(u), np.sin(v)),
            cz + R*np.outer(np.ones_like(u), np.cos(v)))

# ---------- one‑time figure setup ----------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
ax.set_title("Electron riding a gravitational wave", fontsize=13)
ax.set_xlim(-SPAN, SPAN); ax.set_ylim(-SPAN, SPAN); ax.set_zlim(-1.6, 1.2)
ax.set_box_aspect((2, 2, 1))

# static surface (with lighting)
ls = LightSource(315, 50)
norm_R = (np.sqrt(X**2+Y**2) - np.sqrt(X**2+Y**2).min()) / (np.sqrt(X**2+Y**2).max() - np.sqrt(X**2+Y**2).min())
base_colors = plt.get_cmap("turbo")(norm_R)[:, :, :3]
rgb_surface = ls.shade_rgb(base_colors, Z, vert_exag=2.0, blend_mode="soft")
ax.plot_surface(X, Y, Z, facecolors=rgb_surface, rstride=1, cstride=1, linewidth=0, antialiased=True, shade=False, alpha=0.95, zorder=2)
ax.plot_wireframe(X, Y, Z, color="k", alpha=0.35, linewidth=0.4, rstride=6, cstride=6, zorder=3)
ax.contour(X, Y, Z, levels=np.linspace(-A, A, 11), zdir="z", offset=-1.6,
           cmap="Greys", linewidths=0.8, alpha=0.45, zorder=1)

# binary black holes (static)
R_bh = 0.35
for cx, cy in [(x1, y1), (x2, y2)]:
    cz = wave_z(cx, cy) + 0.55 * R_bh
    XS, YS, ZS = visual_sphere(cx, cy, cz, R_bh, n=30)
    rgb = ls.shade(ZS, plt.cm.inferno, vert_exag=1.0, blend_mode="soft")
    ax.plot_surface(XS, YS, ZS, facecolors=rgb, rstride=1, cstride=1, linewidth=0, antialiased=True, shade=False, zorder=10)

ax.view_init(elev=26, azim=-55)
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_zticks([])
ax.grid(False)
for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    pane.set_visible(False)

# We will store the moving electron artists to be replaced each frame
moving_objects = []

def draw_frame(frame_idx):
    global moving_objects
    # remove previous
    for obj in moving_objects:
        obj.remove()
    moving_objects.clear()

    ex = ex_path[frame_idx]
    ey = ey_path[frame_idx]
    ez = ez_path[frame_idx]

    # electron sphere
    XS, YS, ZS = visual_sphere(ex, ey, ez, 0.18, n=40)
    e_rgb = ls.shade(ZS, plt.cm.Blues, vert_exag=1.0, blend_mode="soft")
    surf = ax.plot_surface(XS, YS, ZS, facecolors=e_rgb, rstride=1, cstride=1,
                           linewidth=0, antialiased=True, shade=False, zorder=12)
    moving_objects.append(surf)

    # upward arrow (spin)
    arrow = ax.quiver(ex, ey, ez, 0, 0, 0.6, color="#e6550d", lw=2.4,
                       arrow_length_ratio=0.3, zorder=13)
    moving_objects.append(arrow)

    # ring around electron (GW impression)
    ring = np.linspace(0, 2*np.pi, 40)
    for rr in (0.32, 0.40):
        ring_line, = ax.plot(ex + rr*np.cos(ring), ey + rr*np.sin(ring),
                             np.full_like(ring, ez), color="crimson", lw=1.8, zorder=13)
        moving_objects.append(ring_line)

    ax.set_title(f"Electron riding a gravitational wave – frame {frame_idx+1}/{len(path_t)}", fontsize=12)

# Generate and save frames
for i in range(len(path_t)):
    draw_frame(i)
    fig.canvas.draw()
    fig.savefig(f"anim_GW_sheet/electron_wave_{i:04d}.png", dpi=150, bbox_inches="tight")
    print(f"Saved frame {i:04d}")
plt.show()
plt.close(fig)
print("All frames saved in anim_GW_sheet/")