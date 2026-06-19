"""
RIGOROUS spin-connection figure (revised, conceptually correct).

═════════════════════════════════════════════════════════════════════
PHYSICS (the actual idea):
═════════════════════════════════════════════════════════════════════
• At every spacetime point there is an INDEPENDENT local Lorentz
  frame (tetrad / vierbein  e^a_μ(x)).
• A Dirac spinor ψ(x) has components defined RELATIVE TO that local
  frame.  Globally there is no preferred frame — each point chooses
  its own.
• To move an electron from P → P', we must compare its spinor
  components at the two points.  But the two local frames are
  DIFFERENT, so the comparison is meaningless without a rule.
• That rule is the SPIN CONNECTION  ω_μ^{ab}(x), an antisymmetric
  Lorentz-algebra-valued 1-form.
• ω_μ^{ab} dx^μ is the infinitesimal Lorentz rotation between
  neighbouring local frames.
• The spinor covariant derivative
       ∇_μ ψ = ∂_μ ψ + ¼ σ_{ab} ω_μ^{ab} ψ
  rotates the components of ψ by exactly this amount, so that the
  spinor's PHYSICAL state (its orientation in the local frame) is
  preserved during parallel transport.
• It's called "spin" connection because it acts non-trivially on
  spin-½ objects, but the structure itself is the gauge field of
  local Lorentz transformations — not the quantum spin number.

═════════════════════════════════════════════════════════════════════
TWO-PANEL VISUALIZATION:
═════════════════════════════════════════════════════════════════════
LEFT  — A lattice of local Lorentz frames, each with its own
        orientation θ(x,y).  No global frame; each is independent.
RIGHT — Electron transported from P to P' along a path.
        ─ dashed faint arrow at P':  spin transported WITHOUT ω
          (keeps its absolute direction → misaligned w/ local frame).
        ─ solid bright arrow at P':   spin transported WITH ω
          (rotated by Δθ = ∫ω·dx → preserves physical state).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

OUT = "Thesis_Ready_Plots"
os.makedirs(OUT, exist_ok=True)

# ────────────────────────────────────────────────────────────────────
# Local-frame rotation angle field θ(x,y)
# (illustrative — encodes the gauge data ω as a varying frame angle)
# ────────────────────────────────────────────────────────────────────
def theta(x, y):
    return 0.85 * np.sin(0.40*x + 0.15*y) + 0.20 * np.cos(0.55*y)

def draw_tetrad(ax, x, y, th, size=0.30, lw=1.5,
                c1="crimson", c2="forestgreen", dot=True):
    """Draw a local Lorentz frame (ê_1, ê_2) at (x,y), rotated by th."""
    c, s = np.cos(th), np.sin(th)
    e1 = (size*c,  size*s)
    e2 = (-size*s, size*c)
    ax.add_patch(FancyArrowPatch((x,y), (x+e1[0], y+e1[1]),
                                 arrowstyle="-|>", color=c1,
                                 lw=lw, mutation_scale=9, zorder=6))
    ax.add_patch(FancyArrowPatch((x,y), (x+e2[0], y+e2[1]),
                                 arrowstyle="-|>", color=c2,
                                 lw=lw, mutation_scale=9, zorder=6))
    if dot:
        ax.scatter(x, y, s=14, color="black", zorder=5)

# ────────────────────────────────────────────────────────────────────
# Figure
# ────────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.family": "serif", "font.size": 11})

fig = plt.figure(figsize=(15.5, 7.6))
fig.patch.set_facecolor("white")

ax_L = fig.add_axes([0.045, 0.12, 0.44, 0.78])
ax_R = fig.add_axes([0.535, 0.12, 0.44, 0.78])

# ════════════════════════════════════════════════════════════════════
# LEFT PANEL  ─  lattice of independent local Lorentz frames
# ════════════════════════════════════════════════════════════════════
ax_L.set_xlim(-3.0, 3.0); ax_L.set_ylim(-3.0, 3.0)
ax_L.set_aspect("equal")
ax_L.grid(True, alpha=0.20, color="0.7")
ax_L.set_xlabel(r"$x$"); ax_L.set_ylabel(r"$y$")
ax_L.set_title("Every spacetime point carries its own Lorentz frame",
               fontsize=12, fontweight="bold", pad=12)

# Lattice
for x in np.arange(-2.5, 2.6, 0.85):
    for y in np.arange(-2.5, 2.6, 0.85):
        draw_tetrad(ax_L, x, y, theta(x, y), size=0.30, lw=1.5)

# Highlight one frame to anchor the labels
hx, hy = 1.7, -1.7
ax_L.add_patch(Circle((hx, hy), 0.55, fill=False,
                      edgecolor="#c14a14", lw=2.0, ls="--", zorder=10))
ax_L.annotate(r"local frame  $\hat e_a(x_0)$" "\n"
              r"(no global frame exists)",
              xy=(hx, hy), xytext=(0.8, -3.6),
              fontsize=10, color="#c14a14", fontweight="bold",
              arrowprops=dict(arrowstyle="->", color="#c14a14", lw=1.4),
              ha="center",
              bbox=dict(boxstyle="round,pad=0.30", fc="#fef0e8",
                        ec="#c14a14", lw=0.9))

# Bottom caption
ax_L.text(0, 2.78,
          r"$\hat e_1(\mathrm{red}),\;\hat e_2(\mathrm{green})$:  "
          r"the two spatial tetrad legs at each point",
          fontsize=10, ha="center", style="italic", color="0.30")

# ════════════════════════════════════════════════════════════════════
# RIGHT PANEL  ─  transport from P → P' with and without ω
# ════════════════════════════════════════════════════════════════════
ax_R.set_xlim(-3.0, 3.0); ax_R.set_ylim(-3.0, 3.0)
ax_R.set_aspect("equal")
ax_R.grid(True, alpha=0.20, color="0.7")
ax_R.set_xlabel(r"$x$"); ax_R.set_ylabel(r"$y$")
ax_R.set_title(r"Spin connection $\omega$ rotates the spinor to match the local frame",
               fontsize=12, fontweight="bold", pad=12)

# Endpoints
P  = np.array([-1.8, -0.9])
Pp = np.array([ 1.8,  1.1])
thP, thPp = theta(*P), theta(*Pp)

# Faint intermediate tetrads along the path
N_inter = 7
for t in np.linspace(0, 1, N_inter)[1:-1]:
    x = P[0]*(1-t) + Pp[0]*t
    y = P[1]*(1-t) + Pp[1]*t
    draw_tetrad(ax_R, x, y, theta(x, y), size=0.22, lw=1.1, dot=True)

# Connecting path
ax_R.plot([P[0], Pp[0]], [P[1], Pp[1]],
          color="0.45", lw=1.4, ls=":", zorder=3)

# Endpoint frames (large)
draw_tetrad(ax_R, P[0],  P[1],  thP,  size=0.55, lw=2.6)
draw_tetrad(ax_R, Pp[0], Pp[1], thPp, size=0.55, lw=2.6)

# Electron markers
for pt, lab, lo in [(P, r"$P$", (-0.32, -0.32)),
                    (Pp, r"$P'$", (0.22, -0.32))]:
    ax_R.scatter(*pt, s=210, color="black", zorder=12,
                 edgecolor="white", linewidth=2)
    ax_R.text(pt[0]+lo[0], pt[1]+lo[1], lab,
              fontsize=15, fontweight="bold")

# Spin direction:  define it as along  ê_1 in the local frame at P
SPIN_LEN = 0.80
spin_at_P = np.array([np.cos(thP), np.sin(thP)]) * SPIN_LEN

# (a) WITHOUT ω: spin keeps its absolute direction at P'
spin_noO  = spin_at_P.copy()
# (b) WITH ω: spin rotates by Δθ = θ(P') − θ(P) to stay along ê_1 at P'
spin_wO   = np.array([np.cos(thPp), np.sin(thPp)]) * SPIN_LEN

# Draw spin at P
ax_R.add_patch(FancyArrowPatch(P, P + spin_at_P,
                               arrowstyle="-|>", color="#c14a14",
                               lw=4.0, mutation_scale=18, zorder=15))
ax_R.text(P[0] + spin_at_P[0]*1.18, P[1] + spin_at_P[1]*1.18 + 0.10,
          r"$\hat S$", color="#c14a14", fontsize=15, fontweight="bold")

# Draw "without ω" spin at P' (dashed but visible)
ax_R.add_patch(FancyArrowPatch(Pp, Pp + spin_noO,
                               arrowstyle="-|>", color="#7a3a3a",
                               lw=2.4, ls=(0,(5,3)), alpha=0.70,
                               mutation_scale=14, zorder=13))
endN = Pp + spin_noO
ax_R.text(endN[0]+0.10, endN[1]-0.15,
          "without $\\omega$\n(mis-aligned with $\\hat e_a(P')$)",
          fontsize=9.5, color="#7a3a3a", alpha=0.85,
          style="italic", ha="left", fontweight="bold")

# Draw "with ω" spin at P' (solid bright)
ax_R.add_patch(FancyArrowPatch(Pp, Pp + spin_wO,
                               arrowstyle="-|>", color="#c14a14",
                               lw=4.0, mutation_scale=18, zorder=15))
endW = Pp + spin_wO
ax_R.text(endW[0]-0.10, endW[1]+0.20,
          r"$\hat S\,$ with $\omega$",
          fontsize=12, color="#c14a14", fontweight="bold",
          ha="right")

# Rotation arc from noO to wO at P'
a_no = np.arctan2(spin_noO[1], spin_noO[0])
a_w  = np.arctan2(spin_wO[1],  spin_wO[0])
arc_th = np.linspace(a_no, a_w, 64)
arc_r  = 0.50
ax_R.plot(Pp[0]+arc_r*np.cos(arc_th),
          Pp[1]+arc_r*np.sin(arc_th),
          color="darkorange", lw=3.0, zorder=14)
# Arrowhead on arc
ax_R.add_patch(FancyArrowPatch(
    (Pp[0]+arc_r*np.cos(arc_th[-3]), Pp[1]+arc_r*np.sin(arc_th[-3])),
    (Pp[0]+arc_r*np.cos(arc_th[-1]), Pp[1]+arc_r*np.sin(arc_th[-1])),
    arrowstyle="-|>", color="darkorange",
    lw=3.0, mutation_scale=14, zorder=16))
# Δθ label centred at the arc midpoint
mid_th = 0.5*(a_no + a_w)
mid_xy = (Pp[0] + (arc_r+0.10)*np.cos(mid_th),
          Pp[1] + (arc_r+0.10)*np.sin(mid_th))
ax_R.text(mid_xy[0], mid_xy[1], r"$\Delta\theta$",
          fontsize=12, color="darkorange", fontweight="bold",
          ha="center", va="center")

# Labelled integral expression — placed in lower-right of axes
ax_R.text(2.95, -2.35,
          r"$\Delta\theta \;=\; \int_{P}^{P'}\omega^{12}{}_\mu\,dx^\mu$",
          fontsize=12, color="darkorange", fontweight="bold",
          ha="right", va="center",
          bbox=dict(boxstyle="round,pad=0.40", fc="#fff7e6",
                    ec="darkorange", lw=1.3))

# Bottom caption
ax_R.text(0, 2.78,
          r"$\omega$ rotates the tetrad legs to preserve the same physical internal state",
          fontsize=10, ha="center", style="italic", color="0.30")

# Top suptitle
fig.suptitle(r"Spin connection $\omega_\mu{}^{ab}$ — the gauge field of local Lorentz frames",
             fontsize=14, fontweight="bold", y=0.99)

out_path = os.path.join(OUT, "fig_spin_connection.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out_path}")