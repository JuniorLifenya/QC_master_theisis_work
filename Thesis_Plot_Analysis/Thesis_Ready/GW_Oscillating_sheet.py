import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import os

output_dir = "anim_oscillating_sheet"
os.makedirs(output_dir, exist_ok=True)

# --- wave parameters ---
A, k, decay, omega = 0.75, 2.5, 3.5, 3.0    # omega: temporal frequency

# --- static grid ---
N, SPAN = 220, 6.0
gx = np.linspace(-SPAN, SPAN, N)
X, Y = np.meshgrid(gx, gx)
R = np.sqrt(X**2 + Y**2)

# colouring (static)
norm_R = (R - R.min()) / (R.max() - R.min())
cmap = plt.get_cmap("turbo")
base_colors = cmap(norm_R)[:, :, :3]

ls = LightSource(azdeg=315, altdeg=50)

# --- persistent figure ---
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
fig.suptitle("Pure Gravitational Wave – Oscillating Spacetime", fontsize=14, fontweight="bold")

# limits (fixed)
z_floor, z_top = -1.2, 1.2
ax.set_xlim(-SPAN, SPAN)
ax.set_ylim(-SPAN, SPAN)
ax.set_zlim(z_floor, z_top)
ax.set_box_aspect((2*SPAN, 2*SPAN, 3.0))
ax.view_init(elev=26, azim=-55)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zticks([])
for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    pane.set_visible(False)

# --- frame loop ---
N_frames = 60
for i in range(N_frames):
    ax.cla()                              # clear everything
    ax.set_xlim(-SPAN, SPAN)
    ax.set_ylim(-SPAN, SPAN)
    ax.set_zlim(z_floor, z_top)
    ax.set_box_aspect((2*SPAN, 2*SPAN, 3.0))
    ax.view_init(elev=26, azim=-55)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zticks([])
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_visible(False)

    t = i / N_frames * 2*np.pi          # one full cycle
    Z = A * np.sin(k * R - omega * t) * np.exp(-R / decay)

    # flat reference
    ax.plot_surface(X, Y, np.zeros_like(Z), color="gainsboro", alpha=0.15,
                    linewidth=0, antialiased=True, zorder=0)

    # warped sheet
    rgb = ls.shade_rgb(base_colors, Z, vert_exag=2.0, blend_mode="soft")
    ax.plot_surface(X, Y, Z, facecolors=rgb, rstride=1, cstride=1,
                    linewidth=0, antialiased=True, shade=False, alpha=0.95, zorder=2)

    # wireframe
    ax.plot_wireframe(X, Y, Z, color="k", alpha=0.35, linewidth=0.4,
                      rstride=10, cstride=10, zorder=3)

    # contour on floor
    levels = np.linspace(-A, A, 11)
    ax.contour(X, Y, Z, levels=levels, zdir="z", offset=z_floor,
               cmap="Greys", linewidths=0.8, alpha=0.45, zorder=1)

    ax.set_title(f"t = {t:.2f} s  (frame {i+1}/{N_frames})", fontsize=11)

    fig.canvas.draw()
    plt.savefig(f"{output_dir}/wave_{i:04d}.png", dpi=120, bbox_inches="tight")
    print(f"Saved frame {i:04d}", end="\r")
plt.show()
plt.close(fig)
print("\nDone! Frames saved in", output_dir)