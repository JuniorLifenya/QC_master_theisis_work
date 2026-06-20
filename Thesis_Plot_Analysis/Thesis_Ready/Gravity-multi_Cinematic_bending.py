"""
ROSETTE ANIMATION — a single mass curving spacetime, with orbiting objects.

DESIGN:
  • 90 frames at 20 fps  →  4.5 s loop
  • Two visible orbiting objects on pre-computed geodesics
        (computed once using mean G_M; trajectories shown lifted onto
         the instantaneous breathing surface for visual coherence)
  • Subtle well "breathing":  G_M(t) = G_M_0 · (1 + 0.10·sin 2πt)
        — physically schematic, pedagogically suggestive of dynamic
          spacetime curvature
  • Fading trails behind each orbiter (~200 points, alpha 0→1)
  • Slow camera sway: azim = −60° + 25° sin 2πt  (elev = 21°)
  • Central mass sphere bobs gently with the well floor

OUTPUTS:
  rosette_frames/rosette_NNN.png   (90 PNGs)
  Thesis_Plot_Analysis/anim_rosette.mp4                 (libx264, yuv420p)
  rosette_frames.zip               (for Overleaf upload)

--- REVERTED TO WHITE BACKGROUND & STANDARD 3D ROOM ---
"""
import os, subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection

OUT_FRAMES = "rosette_frames"
OUT_ROOT   = "Thesis_Plot_Analysis"
os.makedirs(OUT_FRAMES, exist_ok=True)
os.makedirs(OUT_ROOT, exist_ok=True)

# ─── parameters ─────────────────────────────────────────────────────
G_M_0           = 0.70
A_SOFT          = 0.70
WELL_BREATH_AMP = 0.10           # 10% well-depth oscillation
N_FRAMES        = 90
LIFT            = 0.12           # orbits floated above the surface
TRAIL_LEN       = 280            # how many trail points behind each orbiter
N_grid          = 60
SPAN            = 3.0
R_SPHERE        = 0.42

# ─── physics helpers ─────────────────────────────────────────────────
def well_z(X, Y, gM):
    return -2.0 * gM / np.sqrt(X**2 + Y**2 + A_SOFT**2)

def acceleration(p, gM):
    denom = (p[0]**2 + p[1]**2 + A_SOFT**2)**1.5
    return -gM * p / denom

def integrate(p0, v0, dt, steps, gM):
    """Velocity-Verlet integration."""
    p = np.array(p0, float); v = np.array(v0, float)
    out = [p.copy()]; a = acceleration(p, gM)
    for _ in range(steps):
        v += 0.5*dt*a
        p += dt*v
        a = acceleration(p, gM)
        v += 0.5*dt*a
        out.append(p.copy())
    return np.array(out)

def sphere_mesh(cx, cy, cz, radius, n=25):
    u = np.linspace(0, 2*np.pi, n); v = np.linspace(0, np.pi, n)
    return (cx + radius*np.outer(np.cos(u), np.sin(v)),
            cy + radius*np.outer(np.sin(u), np.sin(v)),
            cz + radius*np.outer(np.ones_like(u), np.cos(v)))

def add_fading_trail(ax, xy, z, rgb, zorder=6, lw=1.9):
    """Trail with alpha fading from 0.05 (oldest) to 1.0 (newest head)."""
    if len(xy) < 2:
        return
    pts  = np.column_stack([xy[:,0], xy[:,1], z]).reshape(-1, 1, 3)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    n_seg = len(segs)
    alphas = np.linspace(0.05, 1.0, n_seg)
    colors = [(rgb[0], rgb[1], rgb[2], a) for a in alphas]
    lc = Line3DCollection(segs, colors=colors, linewidths=lw, zorder=zorder)
    ax.add_collection3d(lc)

# ─── pre-compute the two orbital trajectories (use mean G_M) ────────
print("Computing orbits…")
orbit_A = integrate([ 2.2,  0.0], [ 0.0,  0.40], 0.010, 5400, G_M_0)
orbit_B = integrate([-1.5,  0.0], [ 0.0, -0.50], 0.012, 5400, G_M_0)
print(f"  A: {len(orbit_A)} steps,  apoapsis≈{np.sqrt((orbit_A**2).sum(1)).max():.2f}")
print(f"  B: {len(orbit_B)} steps,  apoapsis≈{np.sqrt((orbit_B**2).sum(1)).max():.2f}")

# Grid for the surface
gx = np.linspace(-SPAN, SPAN, N_grid)
X, Y = np.meshgrid(gx, gx)

# A fixed z-range to make the well's "breathing" visually obvious
z_floor_min = well_z(0.0, 0.0, G_M_0 * (1 + WELL_BREATH_AMP))   # deepest
z_floor_max = well_z(0.0, 0.0, G_M_0 * (1 - WELL_BREATH_AMP))   # shallowest
Z_LIM_LO    = z_floor_min - 0.45
Z_LIM_HI    = 1.05

# ─── render frames ──────────────────────────────────────────────────
print(f"\nRendering {N_FRAMES} frames…")
for k in range(N_FRAMES):
    t     = k / N_FRAMES                       # 0 → 1
    phase = 2*np.pi * t

    gM_t    = G_M_0 * (1 + WELL_BREATH_AMP * np.sin(phase))
    Z       = well_z(X, Y, gM_t)
    z_floor = well_z(0.0, 0.0, gM_t)

    # Camera sway
    azim = -60 + 25 * np.sin(phase)
    elev = 21

    # Orbiting balls — position along pre-computed paths
    idx_A = int(t * (len(orbit_A) - 1))
    idx_B = int(t * (len(orbit_B) - 1))
    pos_A = orbit_A[idx_A]
    pos_B = orbit_B[idx_B]

    # Fading trails (last TRAIL_LEN points)
    s_A = max(0, idx_A - TRAIL_LEN)
    s_B = max(0, idx_B - TRAIL_LEN)
    trail_A_xy = orbit_A[s_A:idx_A + 1]
    trail_B_xy = orbit_B[s_B:idx_B + 1]
    trail_A_z  = well_z(trail_A_xy[:,0], trail_A_xy[:,1], gM_t) + LIFT
    trail_B_z  = well_z(trail_B_xy[:,0], trail_B_xy[:,1], gM_t) + LIFT

    # ─── figure ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    # Figure background is white by default – no need to set it.

    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    # All axes, panes, ticks, labels, grid are visible (default style).

    # Surface
    ax.plot_surface(X, Y, Z, cmap="jet_r", alpha=0.55, linewidth=0,
                    antialiased=True, rstride=1, cstride=1, zorder=1)

    # Contours (both on-surface and projected onto z_floor − 0.35)
    levels = np.linspace(z_floor + 0.35, -0.15, 7)
    ax.contour(X, Y, Z, levels=levels, colors="white",
                linewidths=0.6, alpha=0.30, zorder=2)
    ax.contour(X, Y, Z, levels=levels, zdir="z",
                offset=z_floor - 0.35, cmap="jet_r",
                linewidths=0.8, alpha=0.55, zorder=0)

    # Trails (cyan for A, amber for B)
    add_fading_trail(ax, trail_A_xy, trail_A_z, rgb=(0.10, 0.95, 0.95),
                      zorder=6, lw=2.6)
    add_fading_trail(ax, trail_B_xy, trail_B_z, rgb=(1.00, 0.60, 0.20),
                      zorder=6, lw=2.6)

    # Orbiting balls
    z_A = well_z(pos_A[0], pos_A[1], gM_t) + LIFT
    z_B = well_z(pos_B[0], pos_B[1], gM_t) + LIFT
    ax.scatter(pos_A[0], pos_A[1], z_A, color="#19f0ff",
                edgecolor="black", s=180, lw=1.6, zorder=9)
    ax.scatter(pos_B[0], pos_B[1], z_B, color="#ff9a1f",
                edgecolor="black", s=150, lw=1.4, zorder=9)

    # Central mass — bobs gently with the well floor
    cz = z_floor + R_SPHERE + 1.25
    xs, ys, zs = sphere_mesh(0.0, 0.0, cz, R_SPHERE)
    shaded = LightSource(315, 45).shade(zs, plt.cm.inferno,
                                          vert_exag=1.0, blend_mode="soft")
    ax.plot_surface(xs, ys, zs, facecolors=shaded, rstride=1, cstride=1,
                    linewidth=0, antialiased=True, shade=False, zorder=10)
    ax.plot([0, 0], [0, 0], [cz - R_SPHERE, z_floor], color="dimgray",
            ls=":", lw=1.1, alpha=0.8, zorder=8)

    # Label the central mass
    ax.text(0, 0, cz + R_SPHERE + 0.25, "mass $M$",
            fontsize=10, fontweight="bold", ha="center", zorder=11)

    # Camera + axes
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(-SPAN, SPAN)
    ax.set_ylim(-SPAN, SPAN)
    ax.set_zlim(Z_LIM_LO, Z_LIM_HI)
    ax.set_box_aspect((2*SPAN, 2*SPAN, Z_LIM_HI - Z_LIM_LO))

    # Standard axis labels and grid
    ax.set_xlabel(r"$x$", fontsize=12)
    ax.set_ylabel(r"$y$", fontsize=12)
    ax.set_zlabel(r"$z$", fontsize=12)
    ax.grid(True)                     # show the 3D grid
    

    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.93)
    fname = os.path.join(OUT_FRAMES, f"rosette_{k:03d}.png")
    # Save with white background
    fig.savefig(fname, dpi=130, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)

    if (k + 1) % 15 == 0:
        print(f"  frame {k+1:>3}/{N_FRAMES}  G_M={gM_t:.3f}  "
              f"pos_A=({pos_A[0]:+.2f},{pos_A[1]:+.2f})  "
              f"pos_B=({pos_B[0]:+.2f},{pos_B[1]:+.2f})")

# ─── assemble MP4 ──────────────────────────────────────────────────
print("\nBuilding MP4…")
mp4 = os.path.join(OUT_ROOT, "anim_rosette.mp4")
subprocess.run([
    "ffmpeg", "-y", "-framerate", "20",
    "-i", os.path.join(OUT_FRAMES, "rosette_%03d.png"),
    "-c:v", "libx264", "-pix_fmt", "yuv420p",
    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
    mp4
], check=True, capture_output=True)
print(f"MP4:    {mp4}")
print(f"Frames: {OUT_FRAMES}/")