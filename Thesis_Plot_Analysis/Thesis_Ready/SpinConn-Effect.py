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

    For a plus-polarised wave  (hp ≠ 0, hc = 0):
        x' = x + ½ hp x      (stretches along x)
        y' = y − ½ hp y      (compresses along y)

    For a cross-polarised wave (hp = 0, hc ≠ 0):
        x' = x + ½ hc y      (mixes x and y — eigendirections are at ±45°)
        y' = y + ½ hc x

    The combined formula handles both simultaneously.  The factor of ½ comes
    directly from linearised GR: the coordinate displacement of a test mass is
        δx^i = ½ h^i_j x^j
    (see e.g. Maggiore, "Gravitational Waves" Vol.1, §1.4).
    """
    Xd = X + 0.5 * (hp * X + hc * Y)
    Yd = Y + 0.5 * (hc * X - hp * Y)
    return Xd, Yd


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Grid drawing helper
# ─────────────────────────────────────────────────────────────────────────────
def draw_grid(ax, X0, Y0, Z0, hp=0.0, hc=0.0,
              color="dimgray",   # FIX 1: was "dimred" — not a valid colour
              alpha=0.45, lw=0.85, label=None):
    """
    Draw a deformed 3-D grid on `ax`.

    X0, Y0, Z0 are shape-(N, N, N) arrays from np.meshgrid(..., indexing='ij').
    Only X and Y are deformed (GWs are transverse; z is the propagation axis).

    Three families of lines are drawn:
        Loop 1  — lines parallel to z, at each fixed (i, j) node.
                  These are the "vertical poles" of the lattice.
        Loop 2  — lines parallel to y, at each fixed (i, k) node.
                  These are the "rungs" along the y direction.
        Loop 3  — lines parallel to x, at each fixed (j, k) node.
                  These are the "rungs" along the x direction.
    """
    Xd, Yd = deform(X0, Y0, hp=hp, hc=hc)

    first = True

    # --- family 1: z-parallel lines (vary k, fix i and j) ---
    for i in range(Xd.shape[0]):
        for j in range(Xd.shape[1]):
            ax.plot(Xd[i, j, :], Yd[i, j, :], Z0[i, j, :],
                    color=color, alpha=alpha, lw=lw,
                    label=label if first else None)
            first = False

    # --- family 2: y-parallel lines (vary j, fix i and k) ---
    for i in range(Xd.shape[0]):
        for k in range(Xd.shape[2]):
            ax.plot(Xd[i, :, k], Yd[i, :, k], Z0[i, :, k],
                    color=color, alpha=alpha, lw=lw)

    # --- family 3: x-parallel lines (vary i, fix j and k) ---
    for j in range(Xd.shape[1]):
        for k in range(Xd.shape[2]):
            # FIX 2: was Xd[:j, k] — that means "first j rows", NOT "all rows
            # at column (j, k)".  The correct slice is Xd[:, j, k].
            ax.plot(Xd[:, j, k], Yd[:, j, k], Z0[:, j, k],
                    color=color, alpha=alpha, lw=lw)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Tetrad arrow helper
# ─────────────────────────────────────────────────────────────────────────────
def quiv(ax, origin, vecs, cols, labs, lw=2.8, alpha=1.0, ls="-"):
    """
    Draw a set of arrows (tetrad legs) from `origin`.

    Parameters
    ----------
    vecs  : list of 3-vectors — the tetrad basis legs e_a.
    cols  : matching list of colours.
    labs  : matching list of legend labels.
    ls    : linestyle ('-' for solid native frame, '--' for transported frame).
    """
    O = np.asarray(origin, float)
    for v, c, lb in zip(vecs, cols, labs):
        ax.quiver(*O, *v,
                  color=c, lw=lw, alpha=alpha,
                  arrow_length_ratio=0.18,
                  linestyle=ls,
                  label=lb)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Physical parameters
# ─────────────────────────────────────────────────────────────────────────────
h = 0.45   # GW strain — exaggerated ~10^20× for visual clarity
L = 1.05   # Arrow (tetrad leg) length in plot units

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Point A  — h₊ polarisation
#     Spatial metric perturbation: δg = diag(h, −h, 0)
#     Eigenvectors are the coordinate axes; vierbein legs aligned with x, y, z.
# ─────────────────────────────────────────────────────────────────────────────
e1A = np.array([1.0, 0.0, 0.0]) * L   # stretched along x
e2A = np.array([0.0, 1.0, 0.0]) * L   # compressed along y
e3A = np.array([0.0, 0.0, 1.0]) * L   # propagation axis — unaffected

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Point B  — h× polarisation
#     Spatial metric perturbation: δg_xy = δg_yx = h.
#     The 2×2 block [[1, h],[h, 1]] has eigenvalues 1±h and eigenvectors
#     (1, ±1)/√2 — i.e. at ±45° in the xy-plane.
# ─────────────────────────────────────────────────────────────────────────────
e1B = np.array([ 1.0,  1.0, 0.0]) / np.sqrt(2) * L   # +45° direction
e2B = np.array([ 1.0, -1.0, 0.0]) / np.sqrt(2) * L   # −45° direction
e3B = np.array([ 0.0,  0.0, 1.0]) * L

# Colour pairs: solid for native frame, dashed for transported frame
SOLID  = ("crimson",  "forestgreen", "royalblue")
DASHED = ("darkred",  "darkgreen",   "navy")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Build the 3-D grid
#     np.meshgrid with indexing='ij' returns arrays of shape (N, N, N) where
#     X0[i,j,k] = pts[i], Y0[i,j,k] = pts[j], Z0[i,j,k] = pts[k].
# ─────────────────────────────────────────────────────────────────────────────
pts = np.linspace(-1, 1, 4)
X0, Y0, Z0 = np.meshgrid(pts, pts, pts, indexing="ij")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  Figure layout
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 6.4))
fig.patch.set_facecolor('#F7F7F7')
fig.suptitle(
    r"Spin Connection $\omega_\mu^{\ ab}$: Local Lorentz Frame Transport"
    r" ($h_+ \to h_\times$, phase $\Delta\phi = \pi/2$)",
    fontsize=13, fontweight="bold", y=0.975)

ELEV, AZIM = 24, 40
O = np.zeros(3)   # origin for all arrows

# ─────────────────────────────────────────────────────────────────────────────
# 9.  LEFT panel — Point A  (h₊ frame)
# ─────────────────────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(1, 2, 1, projection="3d")

draw_grid(ax1, X0, Y0, Z0,
          color="black", alpha=0.45, lw=0.80,
          label=r"Unperturbed $g^{(0)}_{\mu\nu}$")

draw_grid(ax1, X0, Y0, Z0, hp=h,
          color='#2171B5', alpha=0.88, lw=1.55,
          label=r"$h_+$ deformed metric")

# FIX 3: original string r"$e_{\hat 3 (A)$: along $\hat z$}" had an unclosed
# brace and a stray } at the end.  Corrected to r"$e_{\hat{3}}(A)$: ...".
quiv(ax1, O, [e1A, e2A, e3A], SOLID,
     [r"$e_{\hat{1}}(A)$: along $\hat{x}$ (stretched)",
      r"$e_{\hat{2}}(A)$: along $\hat{y}$ (compressed)",
      r"$e_{\hat{3}}(A)$: along $\hat{z}$"])

for v, c, lbl in zip([e1A, e2A, e3A], SOLID,
                     [r"$e_{\hat{1}}$", r"$e_{\hat{2}}$", r"$e_{\hat{3}}$"]):
    ax1.text(*(v * 1.13 + O), lbl, fontsize=10, color=c, fontweight="bold")

ax1.scatter(*O, s=70, color="black", zorder=10)
ax1.text(0.06, 0.06, 1.40, r"$A$", fontsize=14, fontweight="bold", color="#111")

ax1.set_title(
    r"Point $A$: $h_+$ polarisation ($h_{xx} = -h_{yy} = h$)" + "\n"
    r"Native vierbein aligned with coordinate axes",
    fontsize=10, pad=6)
ax1.set_xlim(-1.5, 1.5); ax1.set_ylim(-1.5, 1.5); ax1.set_zlim(-1.5, 1.5)
ax1.set_xlabel(r"$x$"); ax1.set_ylabel(r"$y$"); ax1.set_zlabel(r"$z$")
ax1.view_init(elev=ELEV, azim=AZIM)
hl, ll = ax1.get_legend_handles_labels()
ax1.legend(hl, ll, loc="upper left", fontsize=7.5)

# ─────────────────────────────────────────────────────────────────────────────
# 10. RIGHT panel — Point B  (h× frame + transported A-frame)
# ─────────────────────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(1, 2, 2, projection="3d")

draw_grid(ax2, X0, Y0, Z0,
          color="black", alpha=0.45, lw=0.75,
          label=r"Unperturbed $g^{(0)}_{\mu\nu}$")

draw_grid(ax2, X0, Y0, Z0, hc=h,
          color="#1EC4A3", alpha=0.90, lw=1.55,
          label=r"$h_\times$ deformed metric")

# Native tetrad at B  (solid lines)
quiv(ax2, O, [e1B, e2B, e3B], SOLID,
     [r"$e_{\hat{1}}(B)$: along $+45°$",
      r"$e_{\hat{2}}(B)$: along $-45°$",
      r"$e_{\hat{3}}(B)$"],
     lw=2.8, alpha=1.0, ls="-")

for v, c, lbl in zip([e1B, e2B, e3B], SOLID,
                     [r"$e_{\hat{1}}(B)$", r"$e_{\hat{2}}(B)$", r"$e_{\hat{3}}(B)$"]):
    ax2.text(*(v * 1.15 + O), lbl, fontsize=9, color=c, fontweight="bold")

# Transported frame from A, arriving without spin-connection correction (dashed)
quiv(ax2, O, [e1A, e2A, e3A], DASHED,
     [r"$\tilde{e}_{\hat{1}}$: transported from $A$ (no $\omega$)",
      r"$\tilde{e}_{\hat{2}}$: transported from $A$",
      r"$\tilde{e}_{\hat{3}}$: unchanged"],
     lw=1.9, alpha=0.60, ls="--")

# ── Spin-connection arc in the xy-plane (0° → 45°) ───────────────────────────
# This arc represents the SO(1,3) Lie-algebra element ω·dx that rotates the
# naively-transported frame (dashed) onto the native frame at B (solid).
theta = np.linspace(0, np.pi / 4, 80)
r_arc = 0.95
ax2.plot(r_arc * np.cos(theta), r_arc * np.sin(theta), np.zeros_like(theta),
         color="darkorange", lw=3.2, zorder=20,
         label=r"$\omega_\mu^{\ ab}\,dx^\mu$: $45°$ Lorentz rotation")

# Tiny arrowhead at the tip of the arc
dt = theta[1] - theta[0]
ax2.quiver(r_arc * np.cos(theta[-1] - dt),
           r_arc * np.sin(theta[-1] - dt), 0,
           -r_arc * np.sin(theta[-1]) * dt * 6,
            r_arc * np.cos(theta[-1]) * dt * 6, 0,
           color="darkorange", arrow_length_ratio=5.0, lw=3.2, zorder=21)

ax2.text(r_arc * np.cos(np.pi / 8) * 1.25,
         r_arc * np.sin(np.pi / 8) * 1.15, 0.25,
         r"$\omega\cdot dx$", fontsize=9.5, color="darkorange", fontweight="bold")

ax2.scatter(*O, s=70, color="black", zorder=10)
ax2.text(0.06, 0.06, 1.40, r"$B$", fontsize=14, fontweight="bold", color="#111")

ax2.set_title(
    r"Point $B$: $h_\times$ polarisation ($h_{xy}=h_{yx}=h$)" + "\n"
    r"Solid = native $e_{\hat{a}}(B)$;  Dashed = bare-transported $\tilde{e}_{\hat{a}}$",
    fontsize=10, pad=6)
ax2.set_xlim(-1.5, 1.5); ax2.set_ylim(-1.5, 1.5); ax2.set_zlim(-1.5, 1.5)
ax2.set_xlabel(r"$x$"); ax2.set_ylabel(r"$y$"); ax2.set_zlabel(r"$z$")
ax2.view_init(elev=ELEV, azim=AZIM)
hl2, ll2 = ax2.get_legend_handles_labels()
ax2.legend(hl2, ll2, loc="upper left", fontsize=7.2)

# ─────────────────────────────────────────────────────────────────────────────
# 11. Footer caption
# ─────────────────────────────────────────────────────────────────────────────
fig.text(
    0.5, 0.008,
    r"As the GW propagates by $\Delta\phi=\pi/2$, the natural vierbein rotates "
    r"$45°$ in the transverse plane.  The spin connection $\omega_\mu^{\ ab}$ is the "
    r"$\mathrm{SO}(1,3)$ Lie-algebra element (orange arc) mapping "
    r"$\tilde{e}_{\hat{a}}$ (dashed, naive transport) onto $e_{\hat{a}}(B)$ (solid). "
    r"Its antisymmetry $\omega_{\mu ab}=-\omega_{\mu ba}$ reflects the Lorentz-rotation structure; "
    r"looping $\omega$ encodes curvature via $R^{ab}=d\omega^{ab}+\omega^{ac}\wedge\omega_c{}^b$.",
    ha="center", fontsize=8.5, color="#333", style="italic")

plt.tight_layout(rect=[0, 0.045, 1, 0.965])

plt.savefig("Thesis_Ready_Plots/fig_ch2_spin_connection_transport.png",
            dpi=220, bbox_inches="tight")
plt.show()
print("Saved to Thesis_Ready_Plots/fig_ch2_spin_connection_transport.png")