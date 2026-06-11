import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import os
os.makedirs("Thesis_Ready_Plots", exist_ok=True)

# --- the curved manifold: a saddle  z = 0.6 x y ------------------------------
def Zsurf(x, y): return 0.6 * x * y
def r_vec(x, y): return np.array([x, y, 0.6 * x * y])
def dr_dx(x, y): return np.array([1.0, 0.0, 0.6 * y])   # coordinate tangent ∂r/∂x
def dr_dy(x, y): return np.array([0.0, 1.0, 0.6 * x])   # coordinate tangent ∂r/∂y

# --- the FLAT tangent plane at P, spanned by the two tangent vectors ----------
#   plane(s,t) = r(P) + s ∂r/∂x + t ∂r/∂y   — a genuine flat sheet touching P.
def tangent_plane(x0, y0, s, t):
    P = r_vec(x0, y0); rx = dr_dx(x0, y0); ry = dr_dy(x0, y0)
    S, T = np.meshgrid(s, t)
    return (P[0] + S*rx[0] + T*ry[0],
            P[1] + S*rx[1] + T*ry[1],
            P[2] + S*rx[2] + T*ry[2])

# --- orthonormal flat coordinate axes at P (just for the little arrows) -------
def frame(x0, y0):
    rx, ry = dr_dx(x0, y0), dr_dy(x0, y0)
    n = np.cross(rx, ry); e3 = n/np.linalg.norm(n)
    e1 = rx/np.linalg.norm(rx); e2 = np.cross(e3, e1)
    return e1, e2, e3

P0 = (0.3, 0.3)                 # the one point we zoom in on
Pp = r_vec(*P0)

fig = plt.figure(figsize=(13, 6.2))
fig.patch.set_facecolor("white")
fig.suptitle("Every point of a curved manifold carries a flat tangent space",
             fontsize=14, fontweight="bold", y=0.97)

# ════════════════════════════════════════════════════════════════════════════
# LEFT — global view: the flat sheet sitting on the saddle at P
# ════════════════════════════════════════════════════════════════════════════
ax1 = fig.add_subplot(1, 2, 1, projection="3d", computed_zorder=False)

pts = np.linspace(-1.5, 1.5, 30)
X, Y = np.meshgrid(pts, pts)
ax1.plot_surface(X, Y, Zsurf(X, Y), color="wheat", alpha=0.55,
                 edgecolor="none", zorder=1)
ax1.plot_wireframe(X, Y, Zsurf(X, Y), color="darkgoldenrod",
                   alpha=0.18, lw=0.6, zorder=2)

s = np.linspace(-0.85, 0.85, 8)
TX, TY, TZ = tangent_plane(*P0, s, s)
ax1.plot_surface(TX, TY, TZ, color="steelblue", alpha=0.45,
                 edgecolor="navy", linewidth=0.6, zorder=5)

e1, e2, e3 = frame(*P0)
for e, c in [(e1, "crimson"), (e2, "forestgreen"), (e3, "navy")]:
    ax1.quiver(*Pp, *(0.6*e), color=c, lw=2.5, arrow_length_ratio=0.18, zorder=6)
ax1.scatter(*Pp, color="black", s=45, zorder=7)
ax1.text(Pp[0], Pp[1], Pp[2] + 0.55, "P", fontsize=13, fontweight="bold", ha="center")
ax1.text(-0.2, 1.25, 1.15, "flat tangent plane $T_PM$",
         fontsize=9.5, color="navy", ha="center")

ax1.set_title("the tangent plane touches at one point", fontsize=10)
ax1.set_xlim(-1.6, 1.6); ax1.set_ylim(-1.6, 1.6); ax1.set_zlim(-1.5, 1.5)
ax1.set_box_aspect((3.2, 3.2, 3.0))
ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")
ax1.view_init(elev=22, azim=-50)

# a little dashed box marking the zoom region
d = 0.26
bx = [P0[0]-d, P0[0]+d, P0[0]+d, P0[0]-d, P0[0]-d]
by = [P0[1]-d, P0[1]-d, P0[1]+d, P0[1]+d, P0[1]-d]
ax1.plot(bx, by, [Zsurf(a, b) for a, b in zip(bx, by)],
         color="red", lw=1.4, zorder=8)

# ════════════════════════════════════════════════════════════════════════════
# RIGHT — zoom-in: surface ≈ tangent plane  (local flatness)
# ════════════════════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(1, 2, 2, projection="3d", computed_zorder=False)

xz = np.linspace(P0[0]-d, P0[0]+d, 24)
yz = np.linspace(P0[1]-d, P0[1]+d, 24)
Xz, Yz = np.meshgrid(xz, yz)
ax2.plot_surface(Xz, Yz, Zsurf(Xz, Yz), color="wheat", alpha=0.75,
                 edgecolor="darkgoldenrod", linewidth=0.3, zorder=1)

sz = np.linspace(-d, d, 8)
TXz, TYz, TZz = tangent_plane(*P0, sz, sz)
ax2.plot_surface(TXz, TYz, TZz, color="steelblue", alpha=0.40,
                 edgecolor="navy", linewidth=0.6, zorder=4)
ax2.scatter(*Pp, color="black", s=40, zorder=6)
ax2.text(Pp[0], Pp[1], Pp[2] + 0.05, " P", fontsize=12, fontweight="bold")

zmin = min(TZz.min(), Zsurf(Xz, Yz).min())
zmax = max(TZz.max(), Zsurf(Xz, Yz).max())
ax2.set_title("zoom-in: the surface flattens onto $T_PM$", fontsize=10)
ax2.set_xlim(P0[0]-d, P0[0]+d); ax2.set_ylim(P0[1]-d, P0[1]+d); ax2.set_zlim(zmin, zmax)
ax2.set_box_aspect((2*d, 2*d, zmax - zmin))
ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")
ax2.view_init(elev=18, azim=-50)

fig.tight_layout(rect=[0, 0, 1, 0.93])
out = "Thesis_Ready_Plots/fig_tangent_space.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.show()
print("Saved:", out)