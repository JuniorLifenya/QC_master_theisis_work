import numpy as np, h5py, matplotlib.pyplot as plt
import scipy.constants as sc

"""
spin_conn_sphere.py  —  Spin Connection Precession (CORRECTED)
==============================================================
Uses a 2-SPHERE (constant K=1) so the holonomy is exactly
the solid angle of the enclosed spherical triangle.
For a right-angle path along equator + meridian the holonomy
is exactly π/2 = 90° — large, visible, and analytically verifiable.

Physics: the spin connection on S² generates a SO(2) rotation
in the local tangent frame equal to the area enclosed on the sphere
(Gauss-Bonnet theorem: holonomy = ∫K dA = solid angle enclosed).

This directly mirrors what happens to a Dirac spinor in curved
spacetime: the spin connection ω^ab_μ dx^μ rotates the local
Lorentz frame by an amount set by the spacetime curvature.
"""

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from mpl_toolkits.mplot3d import Axes3D
# import os

# os.makedirs("Thesis_Ready_Plots", exist_ok=True)

# # ── sphere parametrisation ──────────────────────────────────────────────────
# def sphere(theta, phi):
#     """Point on unit sphere."""
#     return np.array([np.sin(theta)*np.cos(phi),
#                      np.sin(theta)*np.sin(phi),
#                      np.cos(theta)])

# def local_frame_sphere(theta, phi):
#     """
#     Orthonormal tangent frame on S² at (θ,φ):
#       ê_θ = ∂r/∂θ  (southward)
#       ê_φ = (1/sinθ) ∂r/∂φ  (eastward)
#     """
#     e_th = np.array([ np.cos(theta)*np.cos(phi),
#                       np.cos(theta)*np.sin(phi),
#                      -np.sin(theta)])
#     e_ph = np.array([-np.sin(phi), np.cos(phi), 0.0])
#     return e_th / np.linalg.norm(e_th), e_ph / np.linalg.norm(e_ph)

# # ── TRIANGULAR PATH (A → B → C → A) ────────────────────────────────────────
# # A = north pole region; go along equator, then meridian, then back.
# # All segments are geodesics (great-circle arcs).
# # The enclosed solid angle = π/2 ⟹ holonomy = π/2 = 90°.

# def great_arc(start_th, start_ph, end_th, end_ph, n=120):
#     """Interpolate a great-circle arc via slerp."""
#     p0 = sphere(start_th, start_ph)
#     p1 = sphere(end_th,   end_ph)
#     omega = np.arccos(np.clip(np.dot(p0, p1), -1, 1))
#     if omega < 1e-10:
#         return np.array([p0]*n)
#     pts = []
#     for t in np.linspace(0, 1, n):
#         v = (np.sin((1-t)*omega)*p0 + np.sin(t*omega)*p1) / np.sin(omega)
#         pts.append(v)
#     return np.array(pts)

# def theta_phi(p):
#     th = np.arccos(np.clip(p[2], -1, 1))
#     ph = np.arctan2(p[1], p[0])
#     return th, ph

# # Three vertices of the right-angle spherical triangle
# A_tp = (np.pi/2, 0.0)          # equator, φ=0
# B_tp = (np.pi/2, np.pi/2)      # equator, φ=π/2
# C_tp = (0.01, np.pi/4)          # near north pole

# seg1 = great_arc(*A_tp, *B_tp)  # A → B along equator
# seg2 = great_arc(*B_tp, *C_tp)  # B → C up a meridian
# seg3 = great_arc(*C_tp, *A_tp)  # C → A back down

# all_pts = np.concatenate([seg1, seg2, seg3])

# # ── PARALLEL TRANSPORT of spin along the closed path ────────────────────────
# def pt_sphere_step(e1, e2, p_new):
#     """Project frame onto new tangent plane and re-orthogonalise."""
#     n_new = p_new / np.linalg.norm(p_new)
#     def proj(v):
#         vt = v - np.dot(v, n_new)*n_new
#         return vt / (np.linalg.norm(vt) + 1e-15)
#     e1t = proj(e1)
#     e2r = proj(e2); e2r -= np.dot(e2r, e1t)*e1t
#     e2t = e2r / (np.linalg.norm(e2r) + 1e-15)
#     return e1t, e2t

# # Initial frame at A
# e1_A, e2_A = local_frame_sphere(*A_tp)
# e1t, e2t = e1_A.copy(), e2_A.copy()
# frames_all = [(e1t.copy(), e2t.copy())]
# for p in all_pts[1:]:
#     e1t, e2t = pt_sphere_step(e1t, e2t, p)
#     frames_all.append((e1t.copy(), e2t.copy()))

# # Holonomy: measure angle between transported ê₁ and native ê₁ at A
# e1_A_final, _ = frames_all[-1]
# e1_A_native, e2_A_native = e1_A, e2_A
# n_A = sphere(*A_tp)
# cos_dph = np.clip(np.dot(e1_A_final, e1_A_native), -1, 1)
# sin_dph = np.dot(np.cross(e1_A_native, e1_A_final), n_A)
# Dphi = np.arctan2(sin_dph, cos_dph)
# print(f"Holonomy angle Δφ = {np.degrees(Dphi):.1f}°  (theory: 90°)")

# # Spin vector evolves as ê₁ of the transported frame
# spin_all = [fr[0].copy() for fr in frames_all]
# N_all = len(all_pts)

# # ── FIGURE ──────────────────────────────────────────────────────────────────
# plt.rcParams.update({'font.family':'serif','font.size':10,'axes.labelsize':11})
# fig = plt.figure(figsize=(15, 6.8))
# fig.patch.set_facecolor('#F8F8F5')
# fig.suptitle(
#     r"Spin Connection $\omega_\mu^{ab}$: Spinor Holonomy on a Sphere"
#     "\n"
#     r"A closed triangular path encloses solid angle $\Omega = \pi/2$  "
#     r"$\Rightarrow$  spin precesses by $\Delta\phi = \Omega = 90°$ (Gauss-Bonnet)"
#     "\n"
#     r"{\footnotesize In GR: curvature $R_{\mu\nu}^{\ ab}$ plays the role of the sphere's $K = 1/R^2$}",
#     fontsize=11.5, fontweight='bold', y=1.01)

# # ── LEFT: sphere + path + spin at waypoints ─────────────────────────────────
# ax1 = fig.add_subplot(1, 2, 1, projection='3d')
# ax1.set_facecolor('#F8F8F5')

# # Sphere wireframe
# u = np.linspace(0, np.pi, 25)
# v = np.linspace(0, 2*np.pi, 50)
# Xs = np.outer(np.sin(u), np.cos(v))
# Ys = np.outer(np.sin(u), np.sin(v))
# Zs = np.outer(np.cos(u), np.ones_like(v))
# ax1.plot_surface(Xs, Ys, Zs, alpha=0.10, color='#4a90d9', edgecolor='none')
# ax1.plot_wireframe(Xs, Ys, Zs, color='#3a70a9', alpha=0.06, lw=0.3)

# # Path (three coloured segments)
# for seg, col, lbl in [(seg1, '#1a5e8a', 'A→B (equator)'),
#                        (seg2, '#2a8a4a', 'B→C (meridian)'),
#                        (seg3, '#8a3a1a', 'C→A (return)')]:
#     ax1.plot(seg[:,0], seg[:,1], seg[:,2], color=col, lw=2.5, label=lbl)

# # Shade the enclosed spherical triangle (for area = solid angle intuition)
# n_fill = 20
# tri_pts = []
# for t1 in np.linspace(0, 1, n_fill):
#     p1 = great_arc(*A_tp, *C_tp, n_fill)[int(t1*(n_fill-1))]
#     p2 = great_arc(*B_tp, *C_tp, n_fill)[int(t1*(n_fill-1))]
#     for t2 in np.linspace(0, 1, 5):
#         p = (1-t2)*p1 + t2*p2
#         p = p / np.linalg.norm(p)
#         tri_pts.append(p)
# tri_pts = np.array(tri_pts)
# ax1.scatter(tri_pts[:,0], tri_pts[:,1], tri_pts[:,2],
#             color='gold', alpha=0.12, s=8, zorder=0)

# # Spin at waypoints (one per third of path)
# waypts = [0, N_all//6, N_all//3, N_all//2, 2*N_all//3, 5*N_all//6, N_all-1]
# Lf = 0.20; Ls = 0.30
# for wi, idx in enumerate(waypts):
#     o = all_pts[idx]
#     e1n, e2n = local_frame_sphere(*theta_phi(o))
#     n = o / np.linalg.norm(o)
#     S = spin_all[idx]
#     # Tetrad (thin grey)
#     ax1.quiver(*o, *(e1n*Lf), color='#aaa', lw=0.9, alpha=0.55, arrow_length_ratio=0.25)
#     ax1.quiver(*o, *(e2n*Lf), color='#ccc', lw=0.9, alpha=0.45, arrow_length_ratio=0.25)
#     # Normal (outward, thinner)
#     ax1.quiver(*o, *(n*Lf*0.5), color='#ddd', lw=0.6, alpha=0.30, arrow_length_ratio=0.25)
#     # SPIN (thick red)
#     ax1.quiver(*o, *(S*Ls), color='crimson', lw=2.8, alpha=0.95,
#                arrow_length_ratio=0.28,
#                label=r'Spin $\vec{S}$' if wi == 0 else None)
#     ax1.scatter(*o, color='#111', s=12, zorder=9)

# # Labels at vertices
# for tp, name in [(A_tp,'A'), (B_tp,'B'), (C_tp,'C')]:
#     p = sphere(*tp)
#     ax1.scatter(*p, color='black', s=50, zorder=12)
#     ax1.text(p[0]+0.06, p[1]+0.06, p[2]+0.06, name,
#              fontsize=12, fontweight='bold', color='#111')

# ax1.set_title("Closed triangular path on $S^2$\n"
#               r"Spin $\vec{S}$ (red) precesses during transport",
#               fontsize=10, pad=6)
# ax1.view_init(elev=25, azim=40)
# ax1.set_xlim(-1.3,1.3); ax1.set_ylim(-1.3,1.3); ax1.set_zlim(-1.3,1.3)
# ax1.set_xlabel(r'$x$'); ax1.set_ylabel(r'$y$'); ax1.set_zlabel(r'$z$')

# handles = [
#     mpatches.Patch(color='crimson',   label=r'Spin $\vec{S}(\lambda)$'),
#     mpatches.Patch(color='#aaaaaa',   label=r'Local tetrad $\hat{e}_a$'),
#     plt.Line2D([0],[0], color='#1a5e8a', lw=2.5, label='A→B'),
#     plt.Line2D([0],[0], color='#2a8a4a', lw=2.5, label='B→C'),
#     plt.Line2D([0],[0], color='#8a3a1a', lw=2.5, label='C→A'),
#     mpatches.Patch(color='gold',      alpha=0.5, label='Enclosed area $= \\Omega$'),
# ]
# ax1.legend(handles=handles, loc='lower left', fontsize=8.2, framealpha=0.88)

# # ── RIGHT: close-up at A showing holonomy ───────────────────────────────────
# ax2 = fig.add_subplot(1, 2, 2, projection='3d')
# ax2.set_facecolor('#F8F8F5')

# pos_A_3d = sphere(*A_tp)
# u2 = np.linspace(np.pi/2-0.4, np.pi/2+0.4, 10)
# v2 = np.linspace(-0.4, 0.4+np.pi/2, 12)
# Xs2 = np.outer(np.sin(u2), np.cos(v2))
# Ys2 = np.outer(np.sin(u2), np.sin(v2))
# Zs2 = np.outer(np.cos(u2), np.ones_like(v2))
# ax2.plot_surface(Xs2, Ys2, Zs2, alpha=0.20, color='#4a90d9', edgecolor='none')

# L = 0.45
# CNATIVE = ('crimson','forestgreen')

# # Native frame at A (solid)
# for ev, col, lbl in zip([e1_A, e2_A], CNATIVE,
#                         [r'$\hat{e}_\theta(A)$ native', r'$\hat{e}_\phi(A)$ native']):
#     ax2.quiver(*pos_A_3d, *(ev*L), color=col, lw=2.5, alpha=1.0,
#                arrow_length_ratio=0.18, label=lbl)
#     ax2.text(*(pos_A_3d+ev*L*1.22), lbl[:12], fontsize=9,
#              color=col, fontweight='bold')

# # Transported frame at A after full loop  (dashed)
# e1_tA, e2_tA = frames_all[-1]
# for ev, col, lbl in zip([e1_tA, e2_tA], ('darkred','darkgreen'),
#                         [r'$\tilde{e}_\theta$ (transported)', r'$\tilde{e}_\phi$ (transported)']):
#     ax2.quiver(*pos_A_3d, *(ev*L*0.88), color=col, lw=1.8, alpha=0.80,
#                arrow_length_ratio=0.18, linestyle='--', label=lbl)

# # Spin at end (= transported ê₁ ≠ native ê₁)
# S_final = spin_all[-1]
# ax2.quiver(*pos_A_3d, *(S_final*L*1.15), color='crimson', lw=5.5, alpha=1.0,
#            arrow_length_ratio=0.22, label=r'$\vec{S}_A^{\rm final}$  (returned spin)')
# ax2.text(*(pos_A_3d+S_final*L*1.40), r'$\vec{S}^{\rm final}$',
#          fontsize=12, color='crimson', fontweight='bold')

# # Precession arc e1_A → S_final
# n_arc = 80
# arcs = np.linspace(0, Dphi, n_arc)
# r_arc = L*0.78
# n_A_3d = pos_A_3d / np.linalg.norm(pos_A_3d)
# arc_pts = np.array([
#     pos_A_3d + r_arc*(np.cos(a)*e1_A + np.sin(a)*e2_A)
#     for a in arcs])
# ax2.plot(arc_pts[:,0], arc_pts[:,1], arc_pts[:,2],
#          color='darkorange', lw=3.5, zorder=20,
#          label=rf'Holonomy $\Delta\phi = {np.degrees(Dphi):.0f}^\circ$')
# dp = arc_pts[-1]-arc_pts[-2]
# ax2.quiver(*arc_pts[-1], *dp*5, color='darkorange', lw=2.5,
#            arrow_length_ratio=1.5, zorder=21)
# mid = arc_pts[n_arc//2]
# ax2.text(mid[0]+0.07, mid[1]+0.07, mid[2]+0.08,
#          r'$\omega_\mu^{ab}dx^\mu$', fontsize=11, color='darkorange',
#          fontweight='bold')

# ax2.scatter(*pos_A_3d, color='black', s=60, zorder=14)
# ax2.text(pos_A_3d[0]-0.12, pos_A_3d[1]-0.12, pos_A_3d[2]-0.15, r'$A$',
#          fontsize=13, fontweight='bold')

# ax2.set_title(
#     r"At $A$ after full loop: native $\hat{e}_\theta$ (solid) $\neq$ transported $\tilde{e}_\theta$ (dashed)"
#     "\n"
#     r"$\vec{S}$ returns ROTATED by $\Delta\phi = \Omega_{\rm solid\,angle}$"
#     "\n"
#     r"In GR: $\Delta\phi = \oint\omega^{ab}_\mu dx^\mu$ (spin connection integral)",
#     fontsize=10, pad=6)
# ax2.view_init(elev=22, azim=25)
# ax2.set_xlim(pos_A_3d[0]-0.65, pos_A_3d[0]+0.65)
# ax2.set_ylim(pos_A_3d[1]-0.65, pos_A_3d[1]+0.65)
# ax2.set_zlim(pos_A_3d[2]-0.65, pos_A_3d[2]+0.65)
# ax2.set_xlabel(r'$x$'); ax2.set_ylabel(r'$y$'); ax2.set_zlabel(r'$z$')

# hl2, ll2 = ax2.get_legend_handles_labels()
# by_lbl = dict(zip(ll2, hl2))
# ax2.legend(by_lbl.values(), by_lbl.keys(), loc='lower left',
#            fontsize=8, framealpha=0.88)

# plt.tight_layout(rect=[0, 0, 1, 0.90])
# out = "Thesis_Ready_Plots/Spin_Connection_Holonomy_Sphere.png"
# plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='#F8F8F5')
# plt.show()
# print(f"\nSaved: {out}")
# print(f"Holonomy = {np.degrees(Dphi):.1f}°  (Gauss-Bonnet predicts 90°)")


#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# f = np.logspace(0, 4, 500)   # 1 Hz to 10 kHz
# alpha = 1/137; m_e = sc.m_e*sc.c**2/sc.eV; a0 = 1/(m_e*alpha)
# E2 = m_e*alpha**2/8; h_plus = 1e-21

# omega = 2*np.pi*f * sc.hbar/sc.eV   # frequency array in eV

# DE_TT  = (2/5)*h_plus*E2*np.ones_like(f)
# DE_FNC = 3*m_e*a0**2*omega**2*h_plus

# plt.loglog(f, DE_TT, 'b-', label=r'TT: $\frac{2}{5}h_+|E_2|$ (frequency-independent)')
# plt.loglog(f, DE_FNC, 'r-', label=r'FNC: $3m_e a_0^2\omega_{GW}^2 h_+\propto f^2$')
# plt.axvline(100, ls=':', color='gray'); plt.axvline(1000, ls=':', color='gray')
# plt.xlabel('GW frequency (Hz)'); plt.ylabel(r'$|\Delta E|$ (eV)')
# plt.title(r'TT vs FNC energy shift: $(k_{GW}a_0)^2$ suppression in FNC')
# plt.legend(); plt.grid(True, which='both', alpha=0.3)
# plt.tight_layout()
# plt.show()

# # Load your LIGO HDF5 file
# with h5py.File('H-H1_GWOSC_O4a_4KHZ_R1-...hdf5','r') as f:
#     strain = f['strain/Strain'][:]
#     dt = 1/4096

# t = np.arange(len(strain))*dt
# alpha = 1/137.035999
# m_e_eV = sc.m_e*sc.c**2/sc.eV
# E2 = m_e_eV*alpha**2/8   # |E_2| in eV

# # TT energy shift (correct formula, no kappa needed)
# Delta_E_TT = (2/5) * strain * E2   # eV

# plt.figure(figsize=(12,4))
# plt.plot(t, Delta_E_TT*1e21, 'b-', lw=1)
# plt.xlabel('Time (s)'); plt.ylabel(r'$\Delta E_{TT}\;(10^{-21}$ eV$)$')
# plt.title('GW150914: 2p energy splitting from LIGO O1 strain')


