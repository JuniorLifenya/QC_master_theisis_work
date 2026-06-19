import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import os

output_dir = "anim_binary_gw"
os.makedirs(output_dir, exist_ok=True)

# --- orbit + physics ---
d_binary = 1.0
omega = 2.0                     # orbital angular frequency
depth_well = -0.85
sigma_well = 0.35
A_wave, k_wave, decay = 0.55, 2.5, 3.5
softening = 0.35

# --- static grid ---
N, SPAN = 200, 6.0
gx = np.linspace(-SPAN, SPAN, N)
X, Y = np.meshgrid(gx, gx)
R = np.sqrt(X**2 + Y**2)

# colouring (static radial turbo)
norm_R = (R - R.min()) / (R.max() - R.min())
cmap = plt.get_cmap("turbo")
base_colors = cmap(norm_R)[:, :, :3]

ls = LightSource(azdeg=315, altdeg=50)

# --- helper: well from a point mass ---
def mass_well(x, y, mx, my):
    return depth_well / np.sqrt((x - mx)**2 + (y - my)**2 + softening**2)

# --- full height at time t ---
def height_at_time(t):
    # positions of masses (circular orbit)
    mx1, my1 =  d_binary/2 * np.cos(omega * t),  d_binary/2 * np.sin(omega * t)
    mx2, my2 = -d_binary/2 * np.cos(omega * t), -d_binary/2 * np.sin(omega * t)

    Z_wells = mass_well(X, Y, mx1, my1) + mass_well(X, Y, mx2, my2)

    # quadrupole outgoing wave: sin(2θ) * sin(kr - 2ωt) envelope
    theta = np.arctan2(Y, X)
    ang = np.sin(2 * theta)
    env = 0.5 * (1.0 + np.tanh((R - 1.6) / 0.4))
    wave = A_wave * ang * np.sin(k_wave * R - 2*omega * t) * np.exp(-R / decay) * env

    return Z_wells + wave, mx1, my1, mx2, my2

# --- figure ---
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
fig.suptitle("Binary Source Creating Quadrupole Gravitational Waves", fontsize=14, fontweight="bold")

z_floor, z_top = -1.9, 1.2
ax.set_xlim(-SPAN, SPAN)
ax.set_ylim(-SPAN, SPAN)
ax.set_zlim(z_floor, z_top)
ax.set_box_aspect((2*SPAN, 2*SPAN, 3.2))
ax.view_init(elev=26, azim=-55)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zticks([])
for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    pane.set_visible(False)

# --- frame loop ---
N_frames = 90   # 1.5 orbits
for i in range(N_frames):
    ax.cla()
    ax.set_xlim(-SPAN, SPAN)
    ax.set_ylim(-SPAN, SPAN)
    ax.set_zlim(z_floor, z_top)
    ax.set_box_aspect((2*SPAN, 2*SPAN, 3.2))
    ax.view_init(elev=26, azim=-55)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zticks([])
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_visible(False)

    t = i / N_frames * 4*np.pi/omega      # 1.5 full orbits for smoothness
    Z, mx1, my1, mx2, my2 = height_at_time(t)

    # flat reference
    ax.plot_surface(X, Y, np.zeros_like(Z), color="gainsboro", alpha=0.15,
                    linewidth=0, antialiased=True, zorder=0)

    # warped sheet
    rgb = ls.shade_rgb(base_colors, Z, vert_exag=2.0, blend_mode="soft")
    ax.plot_surface(X, Y, Z, facecolors=rgb, rstride=1, cstride=1,
                    linewidth=0, antialiased=True, shade=False, alpha=0.95, zorder=2)

    # wireframe
    ax.plot_wireframe(X, Y, Z, color="k", alpha=0.35, linewidth=0.4,
                      rstride=8, cstride=8, zorder=3)

    # contour on floor
    levels = np.linspace(-A_wave, A_wave, 9)
    ax.contour(X, Y, Z, levels=levels, zdir="z", offset=z_floor,
               cmap="Greys", linewidths=0.8, alpha=0.45, zorder=1)

    # masses as scatter points (black spheres)
    # Interpolate Z at their positions
    z1 = mass_well(X, Y, mx1, my1).max() * 0.9   # approximation (deepest point)
    z2 = mass_well(X, Y, mx2, my2).max() * 0.9
    ax.scatter(mx1, my1, z1 + 0.2, color="black", s=120, zorder=10)
    ax.scatter(mx2, my2, z2 + 0.2, color="black", s=120, zorder=10)

    # orbit ring
    th = np.linspace(0, 2*np.pi, 200)
    ax.plot(d_binary/2 * np.cos(th), d_binary/2 * np.sin(th),
            np.full_like(th, z_floor + 0.05), ls="--", lw=0.8,
            color="dimgray", alpha=0.6, zorder=5)

    ax.set_title(f"t = {t:.2f} s  (frame {i+1}/{N_frames})", fontsize=11)

    fig.canvas.draw()
    plt.savefig(f"{output_dir}/binary_{i:04d}.png", dpi=120, bbox_inches="tight")
    print(f"Saved frame {i:04d}", end="\r")
plt.show()
plt.close(fig)
print("\nDone! Frames saved in", output_dir)