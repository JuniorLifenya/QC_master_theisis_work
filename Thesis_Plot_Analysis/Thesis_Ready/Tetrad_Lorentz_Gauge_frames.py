import numpy as np
import matplotlib.pyplot as plt
import os

out_dir = "lorentz_gauge_frames"
os.makedirs(out_dir, exist_ok=True)

# ─── Settings ──────────────────────────────────────────────────
n_frames = 60
L = 1.2          # length of tetrad basis vectors
span = 1.5       # half-width of the coordinate grid
e1_color = 'crimson'
e2_color = 'forestgreen'
spin_color = 'darkorange'

# ─── Helper: draw an arrow from origin ─────────────────────────
def arrow(ax, vec, color, lw=2.8):
    ax.annotate("", xy=vec, xytext=(0,0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, shrinkA=0, shrinkB=0))

# ─── Generate frames ───────────────────────────────────────────
for frame in range(n_frames):
    # Rotation angle: 0 → 2π (full turn)
    theta = 2 * np.pi * frame / n_frames

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    # Background coordinate grid
    for x in np.linspace(-span, span, 7):
        ax.plot([x, x], [-span, span], color='0.9', lw=0.6)
    for y in np.linspace(-span, span, 7):
        ax.plot([-span, span], [y, y], color='0.9', lw=0.6)

    # Local Lorentz frame (tetrad) – rotated by θ
    e1 = np.array([np.cos(theta), np.sin(theta)]) * L
    e2 = np.array([-np.sin(theta), np.cos(theta)]) * L
    arrow(ax, e1, e1_color, lw=3.0)
    arrow(ax, e2, e2_color, lw=3.0)
    ax.text(*(e1*1.08), r'$\hat{e}_1$', color=e1_color, fontsize=13, fontweight='bold')
    ax.text(*(e2*1.08), r'$\hat{e}_2$', color=e2_color, fontsize=13, fontweight='bold',
            ha='right')

    # Spin vector – always along ê₁ (fixed in local frame)
    spin_vec = e1 * 0.9   # slightly shorter to stay inside
    ax.annotate("", xy=spin_vec, xytext=(0,0),
                arrowprops=dict(arrowstyle="-|>", color=spin_color, lw=3.5,
                                shrinkA=0, shrinkB=0))
    ax.text(*(spin_vec*1.06), r'$\mathbf{S}$', color=spin_color, fontsize=15,
            fontweight='bold')

    # Central particle (electron)
    ax.scatter(0, 0, s=180, color='black', zorder=5)
    ax.text(0.06, -0.12, r'$e^-$', fontsize=14, color='white',
            ha='center', va='center', fontweight='bold')

    # Informative title
    ax.set_title(r"Local Lorentz frame (arbitrary rotation)"
                 f"\n$\\theta = {theta:.2f}$ rad   (gauge choice)", fontsize=12)

    # Save
    fname = os.path.join(out_dir, f"frame_{frame:03d}.png")
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
plt.show()
print(f"Done – {n_frames} frames saved in '{out_dir}'")