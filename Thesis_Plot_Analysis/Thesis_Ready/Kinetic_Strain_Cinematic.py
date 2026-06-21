"""
ANIMATED kinetic strain on a circular orbit – intuitive “heavy/light” version.

PHYSICS (with the sign you actually have)
    V(p) = h_+ (p_y² − p_x²)          (your derived form)

For |p| = p₀ = const. the direction‑dependent part of the kinetic energy is
    T ∼ p²/2m + V(p)   →   V > 0  means extra inertia (heavier)
                          V < 0  means less inertia (lighter).

INTUITION
    - When the dot sits in a **valley** (negative V, x‑direction) the electron
      is “light” – it would move faster for the same momentum.
    - When the dot sits on a **peak** (positive V, y‑direction) the electron
      is “heavy” – harder to accelerate, as if running uphill.
    - The dot’s motion in momentum space is uniform, but the **landscape
      tells you how easily the electron accelerates in each direction.**

60 frames, seamless loop.
Outputs:
  Thesis_Ready_Plots/ks_orbit_frames/ks_orbit_NNN.png   (60 PNGs)
  Thesis_Ready_Plots/ks_orbit_frames.zip                (ZIP of frames)
  anim_kinetic_strain.mp4
"""
import os, subprocess, zipfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# ---------- paths (relative to script location) ----------
OUT_ROOT = "Thesis_Ready_Plots"
OUT_FRAMES = os.path.join(OUT_ROOT, "ks_orbit_frames")
os.makedirs(OUT_FRAMES, exist_ok=True)

# ---------- parameters ----------
h_plus = 0.5          # GW strain (exaggerated)
p_0    = 1.0          # orbit radius
N_FRAMES = 60
plt.rcParams.update({"font.family": "serif", "font.size": 12})

# ---------- landscape (your sign: p_y² − p_x²) ----------
gx = np.linspace(-2.0, 2.0, 110)
PX, PY = np.meshgrid(gx, gx)
V = h_plus * (PY**2 - PX**2)
v_max = abs(V).max()
ls = LightSource(azdeg=315, altdeg=45)
shaded = ls.shade(V, plt.cm.RdBu_r, vmin=-v_max, vmax=v_max,
                  vert_exag=0.6, blend_mode="soft")

# full orbit (static backdrop)
phi_full = np.linspace(0, 2*np.pi, 240)
ox_full = p_0 * np.cos(phi_full)
oy_full = p_0 * np.sin(phi_full)
oV_full = h_plus * (oy_full**2 - ox_full**2)

print(f"Rendering {N_FRAMES} frames → {OUT_FRAMES}")
for k in range(N_FRAMES):
    phi_now = 2*np.pi * k / N_FRAMES
    px_now  = p_0 * np.cos(phi_now)
    py_now  = p_0 * np.sin(phi_now)
    V_now   = h_plus * (py_now**2 - px_now**2)

    fig = plt.figure(figsize=(14, 6.8))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.0, 0.32],
                          left=0.02, right=0.95, top=0.86, bottom=0.10,
                          wspace=0.22)
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax2d = fig.add_subplot(gs[0, 1])
    axbar = fig.add_subplot(gs[0, 2])

    # ─── 3D ───
    ax3d.plot_surface(PX, PY, V, facecolors=shaded, rstride=1, cstride=1,
                      linewidth=0, antialiased=True, alpha=0.72, shade=False,
                      zorder=1)
    # faint full orbit
    pts = np.array([ox_full, oy_full, oV_full + 0.12]).T.reshape(-1,1,3)
    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = Line3DCollection(seg, colors="0.4", linewidths=1.5, alpha=0.5, zorder=8)
    ax3d.add_collection3d(lc)
    # trail up to current phase
    mask = phi_full <= phi_now
    if mask.sum() > 1:
        tx, ty = ox_full[mask], oy_full[mask]
        tV = h_plus * (ty**2 - tx**2)
        ptt = np.array([tx, ty, tV + 0.14]).T.reshape(-1,1,3)
        if len(ptt) > 1:
            sgt = np.concatenate([ptt[:-1], ptt[1:]], axis=1)
            lct = Line3DCollection(sgt, cmap=plt.cm.plasma, linewidths=5,
                                   norm=plt.Normalize(-h_plus*p_0**2, h_plus*p_0**2),
                                   zorder=9)
            lct.set_array(tV[:-1])
            ax3d.add_collection3d(lct)
    # moving dot
    ax3d.scatter([px_now], [py_now], [V_now + 0.16], color="#19f0ff",
                 edgecolor="black", s=240, lw=2, zorder=20)
    # momentum arrow
    ax3d.quiver(px_now, py_now, V_now + 0.16,
                -np.sin(phi_now)*0.6, np.cos(phi_now)*0.6, 0,
                color="#15396b", lw=2.6, arrow_length_ratio=0.32, zorder=21)
    ax3d.view_init(elev=26, azim=-68)
    ax3d.set_xlabel(r"$p_x/p_0$", labelpad=6)
    ax3d.set_ylabel(r"$p_y/p_0$", labelpad=6)
    ax3d.set_zlabel(r"inertia effect $V(\vec p)$", labelpad=4)
    ax3d.set_zticks([])
    ax3d.set_zlim(-v_max*1.10, v_max*1.10)
    ax3d.set_title(r"$V(\vec p)=h_+(p_y^2-p_x^2)$  (red = heavy, blue = light)",
                   fontsize=12, fontweight="bold", pad=6)

    # ─── 2D ───
    ax2d.contourf(PX, PY, V, levels=21, cmap="RdBu_r", vmin=-v_max, vmax=v_max)
    ax2d.contour(PX, PY, V, levels=[0], colors="black", linewidths=0.9, alpha=0.5)
    ax2d.plot(ox_full, oy_full, color="white", lw=5, zorder=4)
    ax2d.scatter([px_now], [py_now], color="#19f0ff", edgecolor="black",
                 s=200, lw=2, zorder=12)
    ax2d.add_patch(FancyArrowPatch((px_now, py_now),
                                   (px_now - np.sin(phi_now)*0.35,
                                    py_now + np.cos(phi_now)*0.35),
                                   color="#15396b", lw=2.2,
                                   arrowstyle="-|>", mutation_scale=14, zorder=13))
    # intuitive labels
    ax2d.text(0, 1.62, "HEAVY\n(slower)", ha="center", va="center",
              fontsize=9.5, fontweight="bold", color="#9c2c2c")
    ax2d.text(0, -1.62, "HEAVY\n(slower)", ha="center", va="center",
              fontsize=9.5, fontweight="bold", color="#9c2c2c")
    ax2d.text(1.55, 0, "LIGHT\n(faster)", ha="center", va="center",
              fontsize=9.5, fontweight="bold", color="#1a4980")
    ax2d.text(-1.55, 0, "LIGHT\n(faster)", ha="center", va="center",
              fontsize=9.5, fontweight="bold", color="#1a4980")
    ax2d.set_aspect("equal")
    ax2d.set_xlim(-1.95, 1.95); ax2d.set_ylim(-1.95, 1.95)
    ax2d.set_xlabel(r"$p_x/p_0$"); ax2d.set_ylabel(r"$p_y/p_0$")
    ax2d.set_title("View from above", fontsize=12, fontweight="bold", pad=6)

    # ─── live bar ───
    axbar.bar([0], [V_now], width=0.6,
              color="#d62728" if V_now >= 0 else "#1f77b4",
              edgecolor="black", lw=1.2)
    axbar.axhline(0, color="0.4", lw=0.8)
    axbar.set_ylim(-h_plus*p_0**2*1.25, h_plus*p_0**2*1.25)
    axbar.set_xlim(-0.6, 0.6)
    axbar.set_xticks([])
    axbar.set_title(r"Inertia effect", fontsize=11, fontweight="bold", pad=6)
    axbar.set_ylabel(r"$h_+ p_0^2 (p_y^2-p_x^2)/p_0^2$", fontsize=10)
    axbar.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        rf"Electron on a circular orbit:  $\phi = {np.degrees(phi_now):3.0f}°$  "
        rf"|  When the dot is on a peak the electron is heavier (harder to push), "
        rf"in a valley it is lighter.",
        fontsize=13.5, fontweight="bold", y=0.97
    )

    fname = os.path.join(OUT_FRAMES, f"ks_orbit_{k:03d}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")  # better res for LaTeX
    plt.close(fig)

    if (k+1) % 15 == 0:
        print(f"  frame {k+1}/{N_FRAMES}  phi={np.degrees(phi_now):.0f}°  V={V_now:+.3f}")

# ZIP the frames (handy for Overleaf)
zip_path = os.path.join(OUT_ROOT, "ks_orbit_frames.zip")
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for fname in sorted(os.listdir(OUT_FRAMES)):
        if fname.endswith('.png'):
            zf.write(os.path.join(OUT_FRAMES, fname), fname)
print(f"ZIP created: {zip_path}")

# MP4
print("\nBuilding MP4...")
mp4 = os.path.join(OUT_ROOT, "anim_kinetic_strain.mp4")
subprocess.run([
    "ffmpeg", "-y", "-framerate", "20",
    "-i", os.path.join(OUT_FRAMES, "ks_orbit_%03d.png"),
    "-c:v", "libx264", "-pix_fmt", "yuv420p",
    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
    mp4
], check=True, capture_output=True)
print(f"MP4: {mp4}")
print(f"Frames: {OUT_FRAMES}/")
print("\nLaTeX usage (Beamer):")
print(r"\animategraphics[autoplay,loop,width=\textwidth,every=1]")
print(r"  {14}{Thesis_Ready_Plots/ks_orbit_frames/ks_orbit_}{000}{059}")