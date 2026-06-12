"""
Gravitational wave passing through a nested tesseract lattice.
================================================================
The honest visualization: no rubber sheet, no "down".  A plane GW travelling
along z carries a transverse-traceless strain field

    + polarisation:  dx = +1/2 h(z,t) x ,   dy = -1/2 h(z,t) y
    x polarisation:  dx = +1/2 h(z,t) y ,   dy = +1/2 h(z,t) x
    (z untouched: the wave is transverse)

with h(z,t) = A sin(kz - wt).  Every line of the lattice is stretched along
one transverse axis exactly where it is squeezed along the other, and the
pattern alternates along the propagation direction with wavelength 2*pi/k.

Edges are coloured by the LOCAL strain h(z):  red = stretch in x / squeeze
in y,  blue = the opposite phase.  Amplitude wildly exaggerated (real
h ~ 1e-21); the geometry of the distortion is what is faithful.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from itertools import product
import os

os.makedirs("Thesis_Ready_Plots", exist_ok=True)

# =============================== STYLE =======================================
STRAIN_CMAP   = plt.get_cmap("turbo_r")   # diverging: stretch vs squeeze
REF_COLOR     = "0.72"
REF_ALPHA     = 0.30
LINEWIDTH     = 1.5
# ==============================================================================

# --- GW parameters --------------------------------------------------------------
A_h   = 0.38        # strain amplitude (exaggerated for visibility)
k_gw  = 3.2         # wavenumber along z
phase = 0.0         # snapshot: wt = phase

def h_of(z):
    return A_h * np.sin(k_gw * z - phase)

def gw_warp(P, pol="plus"):
    """Apply the TT strain of a plane GW travelling along +z to points P (...,3)."""
    P = np.asarray(P, float)
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    h = h_of(z)
    if pol == "plus":
        xn = x * (1.0 + 0.5*h)
        yn = y * (1.0 - 0.5*h)
    elif pol == "cross":
        xn = x + 0.5*h*y
        yn = y + 0.5*h*x
    else:
        raise ValueError(pol)
    return np.stack([xn, yn, z], axis=-1)

# --- Tesseract topology (identical spirit to your gravity version) ---------------
V4_UNIT = np.array(list(product([-1.0, 1.0], repeat=4)))           # (16, 4)
EDGES = [(i, j) for i in range(16) for j in range(i + 1, 16)
         if np.sum(np.abs(V4_UNIT[i] - V4_UNIT[j])) == 2.0]        # 32 edges

def project_4d_to_3d(v4, w_eye=2.6, scale=2.0):
    w = v4[..., 3]
    s = scale / (w_eye - w)
    return v4[..., :3] * s[..., None]

SHELLS  = [0.45, 0.70, 0.95, 1.20]
SAMPLES = 60
t = np.linspace(0.0, 1.0, SAMPLES)

def build_lattice(pol):
    straight, warped, strain_mid = [], [], []
    for s in SHELLS:
        V3 = project_4d_to_3d(V4_UNIT * s)
        for i, j in EDGES:
            P, Q  = V3[i], V3[j]
            line  = (1 - t)[:, None] * P + t[:, None] * Q
            bent  = gw_warp(line, pol=pol)
            for kk in range(SAMPLES - 1):
                straight.append([line[kk], line[kk + 1]])
                warped.append([bent[kk], bent[kk + 1]])
                strain_mid.append(h_of(0.5*(line[kk][2] + line[kk+1][2])))
    return straight, warped, np.asarray(strain_mid)

# --- Figure: two panels, the two polarisations -------------------------------------
fig = plt.figure(figsize=(14.5, 7.2))
fig.patch.set_facecolor("white")
fig.suptitle("A gravitational wave bending every line of space",
             fontsize=16, fontweight="bold", x=0.45, y=0.97)
fig.subplots_adjust(left=0.0, right=0.86, top=1.02, bottom=0.04, wspace=0.0)

norm = mpl.colors.Normalize(vmin=-A_h, vmax=A_h)

for idx, (pol, sym) in enumerate([("plus", r"$h_+$"), ("cross", r"$h_\times$")]):
    ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
    straight, warped, strain = build_lattice(pol)

    # faint unperturbed lattice = flat spacetime reference
    ax.add_collection3d(Line3DCollection(straight, colors=REF_COLOR,
                                         linewidths=0.4, alpha=REF_ALPHA))
    # warped lattice coloured by local strain
    ax.add_collection3d(Line3DCollection(warped,
                                         colors=STRAIN_CMAP(norm(strain)),
                                         linewidths=LINEWIDTH))

    # propagation arrow along z (the wave needs NO medium and NO central mass)
    LIM = 1.05 * np.abs(np.array(warped)).max()
    ax.quiver(0, 0, -LIM*0.98, 0, 0, 0.55*LIM, color="black", lw=2.0,
              arrow_length_ratio=0.12)
    ax.text(0.12*LIM, 0.12*LIM, -0.62*LIM, r"$\vec{k}$", fontsize=13)

    ax.set_title(f"{sym} polarisation", fontsize=12, y=0.92)
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_zlim(-LIM, LIM)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=16, azim=34)
    ax.tick_params(labelsize=7, pad=-2)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$"); ax.set_zlabel(r"$z$")
    ax.grid(False)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_visible(False)

# shared colorbar = local strain
m = mpl.cm.ScalarMappable(norm=norm, cmap=STRAIN_CMAP); m.set_array([])
cb = fig.colorbar(m, ax=fig.axes, shrink=0.62, pad=0.04)
cb.set_label(r"local strain $h(z) = A\sin(kz - \omega t)$"
             "\n(red: stretch $x$ / squeeze $y$ — blue: opposite phase)",
             fontsize=9)

fig.text(0.45, 0.06,
         "transverse–traceless: every line bends, nothing moves along $z$ — "
         "no medium, no central mass, amplitude exaggerated by $\\sim 10^{20}$",
         ha="center", fontsize=10, color="dimgrey", style="italic")

out = "Thesis_Ready_Plots/fig_gw_tesseract.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.show()
print("Saved:", out)