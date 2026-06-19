import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import FancyArrowPatch

out_dir = "spin_connection_transport_frames"
os.makedirs(out_dir, exist_ok=True)

# ─── GW waveform ───────────────────────────────────────────────
# Use a smooth pulse (Gaussian‑modulated sine) so the spin returns to original after the pulse
def h_cross(t, t0=0.0, sigma=1.0, f=0.8, h0=0.8):
    """Return h×(t) and its time integral (rotation angle θ)."""
    env = np.exp(-0.5 * ((t - t0) / sigma)**2)
    h = h0 * env * np.sin(2 * np.pi * f * (t - t0))
    # θ(t) = 0.5 * (h(t) - h(-∞))  (set h(-∞)=0)
    # numerically integrate to avoid needing analytic integral
    # but for the plot we can compute on the fly
    return h

# We will pre‑compute θ by cumulative sum of ∂t h
def compute_rotation(t, h_vals, dt):
    dhdt = np.gradient(h_vals, dt)
    theta = 0.5 * np.cumsum(dhdt) * dt
    theta -= theta[len(theta)//2]   # make zero at centre
    return theta

# ─── Drawing functions ─────────────────────────────────────────
def draw_grid(ax, span=1.5, color='0.85', lw=0.8):
    for x in np.linspace(-span, span, 7):
        ax.plot([x, x], [-span, span], color=color, lw=lw)
    for y in np.linspace(-span, span, 7):
        ax.plot([-span, span], [y, y], color=color, lw=lw)

def arrow(ax, vec, color, lw=2.8, ls='-'):
    ax.annotate("", xy=vec, xytext=(0,0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, linestyle=ls, shrinkA=0, shrinkB=0))

# ─── Parameters ────────────────────────────────────────────────
n_frames = 120
t_max = 8.0
t = np.linspace(-t_max/2, t_max/2, 300)   # fine time for θ integration
dt = t[1] - t[0]
h_vals = h_cross(t, t0=0.0, sigma=1.8, f=0.9, h0=0.8)
theta_vals = compute_rotation(t, h_vals, dt)

# subsample for frames
frame_indices = np.linspace(0, len(t)-1, n_frames, dtype=int)
L = 1.2   # length of frame vectors

# ─── Generate frames ───────────────────────────────────────────
for i_frame, idx in enumerate(frame_indices):
    t_now = t[idx]
    h_now = h_vals[idx]
    theta_now = theta_vals[idx]

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1,
        figsize=(6, 7.5),
        gridspec_kw={'height_ratios': [3, 1]})
    fig.patch.set_facecolor("white")

    # ── Top panel: spatial geometry ──────────────────────────
    ax_top.set_aspect('equal')
    ax_top.set_xlim(-1.6, 1.6)
    ax_top.set_ylim(-1.6, 1.6)
    ax_top.axhline(0, color='0.7', lw=0.5)
    ax_top.axvline(0, color='0.7', lw=0.5)
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    ax_top.set_xlabel("$x$")
    ax_top.set_ylabel("$y$")

    # Fixed coordinate grid (background)
    draw_grid(ax_top, span=1.5, color='0.9', lw=0.6)

    # Local tetrad (rotated by θ)
    e1 = np.array([np.cos(theta_now), np.sin(theta_now)]) * L
    e2 = np.array([-np.sin(theta_now), np.cos(theta_now)]) * L
    arrow(ax_top, e1, 'crimson', lw=3.0)
    arrow(ax_top, e2, 'forestgreen', lw=3.0)
    ax_top.text(*(e1*1.08), r'$\hat{e}_1$', color='crimson', fontsize=13, fontweight='bold')
    ax_top.text(*(e2*1.08), r'$\hat{e}_2$', color='forestgreen', fontsize=13, fontweight='bold', ha='right')

    # Spin vector (parallel‑transported → co‑rotates with tetrad)
    spin_dir = e1   # always along e1 for simplicity (spin fixed in local frame)
    spin_tip = spin_dir * 1.0
    ax_top.annotate("", xy=spin_tip, xytext=(0,0),
                    arrowprops=dict(arrowstyle="-|>", color='darkorange', lw=3.5,
                                    shrinkA=0, shrinkB=0))
    ax_top.text(*(spin_tip*1.06), r'$\mathbf{S}$', color='darkorange', fontsize=15,
                fontweight='bold')

    # Electron at centre
    ax_top.scatter(0, 0, s=180, color='black', zorder=5)
    ax_top.text(0.06, -0.12, r'$e^-$', fontsize=14, color='white',
                ha='center', va='center', fontweight='bold')

    # Arc showing rotation angle (spin connection)
    arc_r = 0.55
    theta_arc = np.linspace(0, theta_now, 60)
    ax_top.plot(arc_r * np.cos(theta_arc), arc_r * np.sin(theta_arc),
                color='darkorange', lw=2.0, zorder=6)
    if abs(theta_now) > 0.03:
        ax_top.annotate("", xy=(arc_r*np.cos(theta_now), arc_r*np.sin(theta_now)),
                        xytext=(arc_r*np.cos(theta_now-0.05), arc_r*np.sin(theta_now-0.05)),
                        arrowprops=dict(arrowstyle="-|>", color='darkorange', lw=2.5))
    ax_top.text(arc_r*np.cos(theta_now/2)+0.08, arc_r*np.sin(theta_now/2)+0.08,
                r'$\int\!\omega_t{}^{12}dt$', fontsize=10, color='darkorange')

    ax_top.set_title(f"Parallel transport of electron spin in GW\n"
                     f"$h_{{\\times}}(t) = {h_now:.2f}$,   "
                     f"$\\theta = \\frac{{1}}{{2}} h_{{\\times}} = {theta_now:.2f}$ rad",
                     fontsize=11)

    # ── Bottom panel: GW strain & rotation vs time ────────────
    ax_bottom.plot(t, h_vals, color='#0A7660', lw=1.5, label=r'$h_{\times}(t)$')
    ax_bottom.plot(t, theta_vals, color='darkorange', lw=1.5, label=r'$\theta(t)$')
    ax_bottom.axvline(t_now, color='gray', lw=1, linestyle='--', alpha=0.6)
    ax_bottom.set_xlim(t[0], t[-1])
    ax_bottom.set_ylim(-1.1, 1.1)
    ax_bottom.set_xlabel("Time (arb. units)")
    ax_bottom.set_ylabel("Amplitude / Rotation (rad)")
    ax_bottom.legend(loc='upper right', fontsize=9)
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.text(0.02, 0.95, f"$t = {t_now:.2f}$", transform=ax_bottom.transAxes,
                   fontsize=9, va='top')

    fig.suptitle(r"Spin connection $\omega_\mu{}^{ab}$ rotates local Lorentz frames "
                 "to preserve electron's spin orientation",
                 fontsize=12, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    filename = os.path.join(out_dir, f"frame_{i_frame:03d}.png")
    fig.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {filename}")
    plt.show()

print(f"Done. {n_frames} frames in {out_dir}/")