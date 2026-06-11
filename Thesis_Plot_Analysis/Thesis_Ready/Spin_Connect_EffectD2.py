import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

os.makedirs("Thesis_Ready_Plots", exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
#  CRITIQUE OF THE ORIGINAL 3-D VERSION (read before trusting the picture)
# ───────────────────────────────────────────────────────────────────────────
#  STRENGTHS
#   * Frame physics is right: the h+ eigenframe lies along (x,y); the h×
#     eigenframe along the ±45° axes; the two differ by a 45° rotation — a
#     genuine signature of the spin-2 nature of GWs (rotate the lab 45° and
#     h+ <-> h×).
#   * Links three objects at once — metric perturbation (grid), vierbein (legs)
#     and connection (arc): ω relates the local frames of neighbouring configs.
#   * Solid "native" vs dashed "bare-transported" frame is the right way to
#     motivate ω: the connection IS the mismatch you must undo.
#
#  FLAWS
#   * 3-D earns nothing: every object lives in the xy-plane (z is the inert
#     propagation axis). Perspective then HIDES the 45° angle — the one quantity
#     that matters — and adds occlusion + clutter. 2-D shows it honestly.
#   * The 3-D lattice (3 line-families) is noisy; linear strain is hard to read
#     once projected.
#   * HANDEDNESS BUG: old e2B = (1,-1)/√2 makes {e1B,e2B} left-handed
#     (e1B×e2B = -z) — a REFLECTION of frame A, not a rotation, contradicting
#     ω ∈ SO(2). Fixed here with e2B = (-1,1)/√2 (same -45° principal axis,
#     arrow chosen so the frame stays right-handed and a clean +45° rotation).
#   * Conceptual caveat: A,B are drawn as two points joined by a path with a
#     definite ω·dx, but h+ and h× are two POLARISATION states separated by a
#     temporal phase Δφ=π/2, not two spatially-transported points. So "45° = ω·dx"
#     is a schematic mnemonic, not a computed connection — word the caption so.
#   * Strain h=0.45 is exaggerated for visibility, well outside the linear
#     regime the grid depicts. Label as schematic.
# ═══════════════════════════════════════════════════════════════════════════

# --- linearised GW coordinate deformation (unchanged from the 3-D code) ------
def deform(X, Y, hp=0.0, hc=0.0):
    Xd = X + 0.5 * (hp * X + hc * Y)
    Yd = Y + 0.5 * (hc * X - hp * Y)
    return Xd, Yd

# --- draw a (linearly) deformed 2-D coordinate grid --------------------------
def draw_grid_2d(ax, hp=0.0, hc=0.0, color="0.75", alpha=0.5, lw=1.0,
                 n=9, span=1.0, label=None):
    t = np.linspace(-span, span, n)
    ends = np.array([-span, span])          # lines stay straight => 2 points
    first = True
    for xi in t:                            # lines of constant x
        Xd, Yd = deform(np.full(2, xi), ends, hp, hc)
        ax.plot(Xd, Yd, color=color, alpha=alpha, lw=lw,
                label=label if first else None); first = False
    for yi in t:                            # lines of constant y
        Xd, Yd = deform(ends, np.full(2, yi), hp, hc)
        ax.plot(Xd, Yd, color=color, alpha=alpha, lw=lw)

# --- a 2-D arrow from the origin ---------------------------------------------
def arrow(ax, vec, color, lw=2.8, ls="-", alpha=1.0):
    ax.annotate("", xy=(vec[0], vec[1]), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle=ls, alpha=alpha, shrinkA=0, shrinkB=0))

# --- physical parameters ------------------------------------------------------
h = 0.45
L = 1.18
SOLID = ("crimson", "forestgreen")
DASH  = ("darkred", "navy")

# frames (2-D: drop the inert e3 = z leg)
e1A = np.array([1.0, 0.0]) * L
e2A = np.array([0.0, 1.0]) * L
e1B = np.array([ 1.0, 1.0]) / np.sqrt(2) * L     # +45°
e2B = np.array([-1.0, 1.0]) / np.sqrt(2) * L     # 135°  (= -45° axis, right-handed)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6.2))
fig.patch.set_facecolor("white")
fig.suptitle(r"Spin connection $\omega_\mu{}^{ab}$: the local frame rotates "
             r"$45°$ as $h_+ \to h_\times$  (phase $\Delta\phi=\pi/2$)",
             fontsize=13, fontweight="bold", y=0.98)

for ax in (ax1, ax2):
    ax.set_aspect("equal"); ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.7, 1.7)
    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$")

# ── Panel A : h+ ─────────────────────────────────────────────────────────────
draw_grid_2d(ax1, color="0.80", alpha=0.7, lw=0.8, label="unperturbed grid")
draw_grid_2d(ax1, hp=h, color="#108273", alpha=0.85, lw=1.4,
             label=r"$h_+$ deformed grid")
arrow(ax1, e1A, SOLID[0]); arrow(ax1, e2A, SOLID[1])
ax1.text(*(e1A*1.06), r"$\hat{e}_1(A)$", color=SOLID[0], fontsize=13, fontweight="bold")
ax1.text(*(e2A*1.06 + np.array([-0.1, 0.0])), r"$\hat{e}_2(A)$",
         color=SOLID[1], fontsize=13, fontweight="bold", ha="right")
ax1.scatter(0, 0, s=55, color="black", zorder=5)
ax1.text(0.06, -0.20, r"$A$", fontsize=15, fontweight="bold")
ax1.set_title(r"Point $A$: $h_+$  ($h_{xx}=-h_{yy}=h$)" + "\n"
              r"native frame along the coordinate axes", fontsize=10)
ax1.legend(loc="upper left", fontsize=8, framealpha=0.9)

# ── Panel B : h× + transported frame + ω arc ─────────────────────────────────
draw_grid_2d(ax2, color="0.80", alpha=0.7, lw=0.8)
draw_grid_2d(ax2, hc=h, color="#0A7660", alpha=0.85, lw=1.4,
             label=r"$h_\times$ deformed grid")

# bare-transported frame from A (no ω) — dashed, still along x,y
arrow(ax2, e1A, DASH[0], lw=2.0, ls="--", alpha=0.85)
arrow(ax2, e2A, DASH[1], lw=2.0, ls="--", alpha=0.85)
ax2.text(*(e1A*1.04 + np.array([0.02, -0.18])), r"$\tilde{e}_1$ (no $\omega$)",
         color=DASH[0], fontsize=9.5, ha="left")

# native frame at B — solid, at ±45°
arrow(ax2, e1B, SOLID[0]); arrow(ax2, e2B, SOLID[1])
ax2.text(*(e1B*1.06), r"$\hat{e}_1(B)$", color=SOLID[0], fontsize=13, fontweight="bold")
ax2.text(*(e2B*1.06), r"$\hat{e}_2(B)$", color=SOLID[1], fontsize=13,
         fontweight="bold", ha="right")

# ω·dx : the 45° rotation arc taking the dashed frame onto the native frame
for a0, a1 in [(0.0, np.pi/4), (np.pi/2, 3*np.pi/4)]:   # e1: 0->45, e2: 90->135
    th = np.linspace(a0, a1, 60)
    r = 0.92
    ax2.plot(r*np.cos(th), r*np.sin(th), color="darkorange", lw=3.0, zorder=6)
    ax2.annotate("", xy=(r*np.cos(a1), r*np.sin(a1)),
                 xytext=(r*np.cos(a1-0.06), r*np.sin(a1-0.06)),
                 arrowprops=dict(arrowstyle="-|>", color="darkorange", lw=3.0))
ax2.text(0.92*np.cos(np.pi/8)+0.12, 0.92*np.sin(np.pi/8)+0.05,
         r"$\omega\cdot dx\,(45°)$", color="darkorange", fontsize=10.5, fontweight="bold")

ax2.scatter(0, 0, s=55, color="black", zorder=5)
ax2.text(0.06, -0.20, r"$B$", fontsize=15, fontweight="bold")
ax2.set_title(r"Point $B$: $h_\times$  ($h_{xy}=h_{yx}=h$)" + "\n"
              r"solid $=$ native frame; dashed $=$ bare-transported", fontsize=10)

# legend for B includes the dashed/solid meaning + ω
handles = [Line2D([0],[0], color="#0A7660", lw=1.4, label=r"$h_\times$ deformed grid"),
           Line2D([0],[0], color="0.5", lw=2.0, ls="--", label=r"transported (no $\omega$)"),
           Line2D([0],[0], color="0.2", lw=2.6, label=r"native frame $\hat{e}_a(B)$"),
           Line2D([0],[0], color="darkorange", lw=3.0, label=r"$\omega\cdot dx$: $45°$ rotation")]
ax2.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)

fig.tight_layout(rect=[0, 0, 1, 0.94])
out = "Thesis_Ready_Plots/fig_spin_connection_2d.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.show()
print("Saved:", out)