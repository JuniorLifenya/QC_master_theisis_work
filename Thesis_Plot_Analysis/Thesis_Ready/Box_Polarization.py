import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

os.makedirs("Thesis_Ready_Plots", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Grid deformation function
# ─────────────────────────────────────────────────────────────────────────────
def deform(X, Y, hp=0.0, hc=0.0):
    """
    Return (X', Y') — the coordinates of a flat grid after a linearised GW
    perturbation acts on it.
    """
    Xd = X + 0.6* (hp * X + hc * Y)
    Yd = Y + 0.6 * (hc * X - hp * Y)
    return Xd, Yd

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Grid drawing helper
# ─────────────────────────────────────────────────────────────────────────────
def draw_grid(ax, X0, Y0, Z0, hp=0.0, hc=0.0,
              color="dimgray", alpha=0.45, lw=0.85, label=None):
    Xd, Yd = deform(X0, Y0, hp=hp, hc=hc)
    first = True

    # z-parallel lines
    for i in range(Xd.shape[0]):
        for j in range(Xd.shape[1]):
            ax.plot(Xd[i, j, :], Yd[i, j, :], Z0[i, j, :],
                    color=color, alpha=alpha, lw=lw,
                    label=label if first else None)
            first = False

    # y-parallel lines
    for i in range(Xd.shape[0]):
        for k in range(Xd.shape[2]):
            ax.plot(Xd[i, :, k], Yd[i, :, k], Z0[i, :, k],
                    color=color, alpha=alpha, lw=lw)

    # x-parallel lines
    for j in range(Xd.shape[1]):
        for k in range(Xd.shape[2]):
            ax.plot(Xd[:, j, k], Yd[:, j, k], Z0[:, j, k],
                    color=color, alpha=alpha, lw=lw)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Tetrad arrow helper
# ─────────────────────────────────────────────────────────────────────────────
def quiv(ax, origin, vecs, cols, labs, lw=2.8, alpha=1.0, ls="-"):
    O = np.asarray(origin, float)
    for v, c, lb in zip(vecs, cols, labs):
        ax.quiver(*O, *v, color=c, lw=lw, alpha=alpha,
                  arrow_length_ratio=0.18, linestyle=ls, label=lb)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Master Plotting Function
# ─────────────────────────────────────────────────────────────────────────────
def plot_gw_polarization(pol_type="plus"):
    h = 0.6   # GW strain
    L = 1.2    # Arrow length
    
    pts = np.linspace(-1, 1, 4)
    X0, Y0, Z0 = np.meshgrid(pts, pts, pts, indexing="ij")
    
    fig = plt.figure(figsize=(14, 6.4))
    fig.patch.set_facecolor('#F7F7F7')
    ELEV, AZIM = 24, 40
    O = np.zeros(3)
    SOLID = ("crimson", "forestgreen", "royalblue")

    if pol_type == "plus":
        fig.suptitle(r"Gravitational Wave: $h_+$ Polarisation in 3D", 
                     fontsize=14, fontweight="bold", y=0.95)
        
        # Vectors aligned with x and y axes
        e1 = np.array([1.0, 0.0, 0.0]) * L
        e2 = np.array([0.0, 1.0, 0.0]) * L
        e3 = np.array([0.0, 0.0, 1.0]) * L
        
        # Left Panel: Phase 0 (Stretch X, Compress Y)
        hp_left, hc_left = h, 0.0
        title_left = r"Phase $0$: Stretched along $x$, Compressed along $y$"
        
        # Right Panel: Phase pi (Compress X, Stretch Y)
        hp_right, hc_right = -h, 0.0
        title_right = r"Phase $\pi$: Compressed along $x$, Stretched along $y$"
        
        labels = [r"$\hat{e}_{1}$ (x-axis)", r"$\hat{e}_{2}$ (y-axis)", r"$\hat{e}_{3}$ (z-axis)"]
        filename = "fig_ch2_h_plus_polarisation.png"

    elif pol_type == "cross":
        fig.suptitle(r"Gravitational Wave: $h_\times$ Polarisation in 3D", 
                     fontsize=14, fontweight="bold", y=0.95)
        
        # Vectors aligned at +/- 45 degrees
        e1 = np.array([1.0, 1.0, 0.0]) / np.sqrt(2) * L
        e2 = np.array([1.0, -1.0, 0.0]) / np.sqrt(2) * L
        e3 = np.array([0.0, 0.0, 1.0]) * L

        # Left Panel: Phase 0 (Stretch +45°, Compress -45°)
        hp_left, hc_left = 0.0, h
        title_left = r"Phase $0$: Stretched along $+45^\circ$, Compressed along $-45^\circ$"
        
        # Right Panel: Phase pi (Compress +45°, Stretch -45°)
        hp_right, hc_right = 0.0, -h
        title_right = r"Phase $\pi$: Compressed along $+45^\circ$, Stretched along $-45^\circ$"
        
        labels = [r"$\hat{e}_{1}$ ($+45^\circ$)", r"$\hat{e}_{2}$ ($-45^\circ$)", r"$\hat{e}_{3}$ (z-axis)"]
        filename = "fig_ch2_h_cross_polarisation.png"

    # --- Plot Left Panel ---
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    draw_grid(ax1, X0, Y0, Z0, color="black", alpha=0.8, lw=0.80, label=r"Unperturbed Grid")
    draw_grid(ax1, X0, Y0, Z0, hp=hp_left, hc=hc_left, color="#108273", alpha=0.9, lw=1.55, label=r"Deformed Grid")
    quiv(ax1, O, [e1, e2, e3], SOLID, labels)
    ax1.set_title(title_left, fontsize=11, pad=6)
    
    # --- Plot Right Panel ---
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    draw_grid(ax2, X0, Y0, Z0, color="black", alpha=0.8, lw=0.80, label=r"Unperturbed Grid")
    draw_grid(ax2, X0, Y0, Z0, hp=hp_right, hc=hc_right, color="#0A7660", alpha=0.9, lw=1.55, label=r"Deformed Grid")
    quiv(ax2, O, [e1, e2, e3], SOLID, labels)
    ax2.set_title(title_right, fontsize=11, pad=6)

    # --- Apply formatting to both ---
    for ax in [ax1, ax2]:
        ax.scatter(*O, s=50, color="black", zorder=10)
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_zlim(-1.5, 1.5)
        ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$"); ax.set_zlabel(r"$z$")
        ax.view_init(elev=ELEV, azim=AZIM)
        hl, ll = ax.get_legend_handles_labels()
        ax.legend(hl, ll, loc="upper left", fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filepath = os.path.join("Thesis_Ready_Plots", filename)
    plt.savefig(filepath, dpi=420, bbox_inches="tight")
    print(f"Saved: {filepath}")
    plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Execute
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_gw_polarization(pol_type="plus")
    plot_gw_polarization(pol_type="cross")