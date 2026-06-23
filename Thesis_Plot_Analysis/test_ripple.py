import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import os

output_dir = "anim_central_ripple"
os.makedirs(output_dir, exist_ok=True)

# --- pulse parameters (Mexican‑hat / Ricker wavelet) ---
A = 0.7          # amplitude
sigma = 0.8      # width of the ripple
v = 5.5          # propagation speed

# --- static grid ---
N, SPAN = 220, 6.0
gx = np.linspace(-SPAN, SPAN, N)
X, Y = np.meshgrid(gx, gx)

# Source at the centre of the sheet
x0, y0 = 0.0, 0.0
R = np.sqrt((X - x0)**2 + (Y - y0)**2)

# Base colour (a subtle gradient to give the sheet texture)
norm_col = 0.5 + 0.3 * (X + Y) / (2 * SPAN)
cmap = plt.get_cmap("plasma")
base_colors = cmap(norm_col)[:, :, :3]

ls = LightSource(azdeg=315, altdeg=50)

# --- persistent figure ---
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

z_floor, z_top = -1.2, 1.2
# Camera positioned to see the wave spread nicely
elev, azim = 30, 45          # looking down at an angle

ax.set_xlim(-SPAN, SPAN)
ax.set_ylim(-SPAN, SPAN)
ax.set_zlim(z_floor, z_top)
ax.set_box_aspect((2*SPAN, 2*SPAN, 3.0))
ax.view_init(elev=elev, azim=azim)
ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zticks([])
for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    pane.set_visible(False)

# --- frame loop ---
N_frames = 80
t_max = N_frames / 40.0
for i in range(N_frames):
    ax.cla()
    ax.set_xlim(-SPAN, SPAN)
    ax.set_ylim(-SPAN, SPAN)
    ax.set_zlim(z_floor, z_top)
    ax.set_box_aspect((2*SPAN, 2*SPAN, 3.0))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zticks([])
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_visible(False)

    t = i * t_max / N_frames
    # Retarded time: distance from centre minus wave travel
    retarded = R - v * t
    # Mexican hat wavelet: a central dip followed by a ring
    pulse = A * (1 - retarded**2 / sigma**2) * np.exp(-retarded**2 / (2 * sigma**2))
    # Amplitude decay with distance (like 1/√r for surface waves)
    decay = 1.0 / np.sqrt(R + 0.01)
    Z = pulse * decay

    # Flat reference plane (undisturbed spacetime)
    ax.plot_surface(X, Y, np.zeros_like(Z), color="gainsboro", alpha=0.12,
                    linewidth=0, antialiased=True, zorder=0)

    # Rippled sheet
    rgb = ls.shade_rgb(base_colors, Z, vert_exag=2.0, blend_mode="soft")
    ax.plot_surface(X, Y, Z, facecolors=rgb, rstride=1, cstride=1,
                    linewidth=0, antialiased=True, shade=False, alpha=0.95, zorder=2)

    # Light wireframe to show the curvature
    ax.plot_wireframe(X, Y, Z, color="k", alpha=0.2, linewidth=0.3,
                      rstride=8, cstride=8, zorder=3)

    # Contour on the floor (shadow)
    levels = np.linspace(-A/2, A/2, 9)
    ax.contour(X, Y, Z, levels=levels, zdir="z", offset=z_floor,
               cmap="Greys", linewidths=0.6, alpha=0.4, zorder=1)

    # Mark the source point (where the “stone” hit)
    ax.scatter([x0], [y0], [0], c="red", s=50, marker="o", zorder=10, edgecolors="k")

    ax.set_title(f"Gravitational ripple — t = {t:.2f}", fontsize=11)

    fig.canvas.draw()
    plt.savefig(f"{output_dir}/ripple_{i:04d}.png", dpi=120, bbox_inches="tight")
    print(f"Saved frame {i:04d}", end="\r")

plt.close(fig)
print("\nDone! Frames saved in", output_dir)