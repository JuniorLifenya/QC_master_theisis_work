"""
fig_ch2_spin_connection_transport.py
Chapter 2 — The Spin Connection as a Local Lorentz Rotation.

Left  (Point A, h+ phase):  native vierbein legs aligned with coord. axes.
Right (Point B, h× phase):  native vierbein legs at ±45°  +  the frame
      naively transported from A (dashed) and the spin-connection arc.

The 45° mismatch between dashed and solid arrows at B IS ω_μ^ab dx^μ.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
os.makedirs("figures", exist_ok=True)

# ─── Grid deformation ────────────────────────────────────────────────────────
def deform(X, Y, hp=0.0, hc=0.0):
    return X + 0.5*(hp*X + hc*Y), Y + 0.5*(hc*X - hp*Y)

def draw_grid(ax, X0, Y0, Z0, hp=0.0, hc=0.0,
              color='dimgray', alpha=0.45, lw=0.85, label=None):
    Xd, Yd = deform(X0, Y0, hp=hp, hc=hc)
    first = True
    for i in range(Xd.shape[0]):
        for j in range(Xd.shape[1]):
            ax.plot(Xd[i,j,:], Yd[i,j,:], Z0[i,j,:],
                    color=color, alpha=alpha, lw=lw,
                    label=label if first else None)
            first = False
    for i in range(Xd.shape[0]):
        for k in range(Xd.shape[2]):
            ax.plot(Xd[i,:,k], Yd[i,:,k], Z0[i,:,k], color=color, alpha=alpha, lw=lw)
    for j in range(Xd.shape[1]):
        for k in range(Xd.shape[2]):
            ax.plot(Xd[:,j,k], Yd[:,j,k], Z0[:,j,k], color=color, alpha=alpha, lw=lw)

# ─── Tetrad arrows ────────────────────────────────────────────────────────────
def quiv(ax, origin, vecs, cols, labs, lw=2.8, alpha=1.0, ls='-'):
    O = np.asarray(origin, float)
    for v, c, lb in zip(vecs, cols, labs):
        ax.quiver(*O, *v, color=c, lw=lw, alpha=alpha,
                  arrow_length_ratio=0.18, linestyle=ls, label=lb)

# ─── Parameters ───────────────────────────────────────────────────────────────
h = 0.45          # GW strain (exaggerated)
L = 1.05          # arrow length

# ── Point A  (h+):  g = diag(1+h, 1-h, 1) ───────────────────────────────────
# Exact vierbein eigenvectors (unit-direction, length L):
e1A = np.array([1.0, 0.0, 0.0]) * L   # stretched along +x
e2A = np.array([0.0, 1.0, 0.0]) * L   # compressed along +y
e3A = np.array([0.0, 0.0, 1.0]) * L

# ── Point B  (h×):  g = [[1,h,0],[h,1,0],[0,0,1]] ────────────────────────────
# Eigenvectors of [[1,h],[h,1]]: (1,±1)/√2 with eigenvalues 1±h
# Vierbein legs rotate to ±45° in xy-plane
e1B = np.array([ 1.0,  1.0, 0.0]) / np.sqrt(2) * L
e2B = np.array([ 1.0, -1.0, 0.0]) / np.sqrt(2) * L
e3B = np.array([0.0,  0.0, 1.0]) * L

SOLID  = ('crimson', 'forestgreen', 'royalblue')
DASHED = ('darkred',  'darkgreen',  'navy')

pts = np.linspace(-1, 1, 4)
X0, Y0, Z0 = np.meshgrid(pts, pts, pts, indexing='ij')

# ─── Figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 6.4))
fig.patch.set_facecolor('#F7F7F7')
fig.suptitle(
    r'Spin Connection $\omega_\mu^{\ ab}$: Local Lorentz Frame Transport'
    r'  ($h_+\to h_\times$, phase $\Delta\phi=\pi/2$)',
    fontsize=13, fontweight='bold', y=0.975)

ELEV, AZIM = 24, 40
O = np.zeros(3)

# ══ LEFT: Point A (h+ native frame) ═════════════════════════════════════════
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
draw_grid(ax1, X0, Y0, Z0, color='dimgray', alpha=0.42, lw=0.75,
          label=r'Unperturbed $g^{(0)}_{\mu\nu}$')
draw_grid(ax1, X0, Y0, Z0, hp=h, color='#2171B5', alpha=0.88, lw=1.55,
          label=r'$h_+$ deformed metric')

quiv(ax1, O, [e1A, e2A, e3A], SOLID,
     [r'$e_{\hat 1}(A)$: along $\hat x$  (stretched)',
      r'$e_{\hat 2}(A)$: along $\hat y$  (compressed)',
      r'$e_{\hat 3}(A)$: along $\hat z$'])

# Axis-label annotations
for v, c, lbl in zip([e1A, e2A, e3A], SOLID,
                      [r'$e_{\hat{1}}$', r'$e_{\hat{2}}$', r'$e_{\hat{3}}$']):
    ax1.text(*(v*1.13 + O), lbl, fontsize=10, color=c, fontweight='bold')

ax1.scatter(*O, s=70, color='black', zorder=10)
ax1.text(0.06, 0.06, 1.40, r'$A$', fontsize=14, fontweight='bold', color='#111')

ax1.set_title(r'Point $A$: $h_+$ polarisation  ($h_{xx}=-h_{yy}=h$)' + '\n'
              r'Native vierbein aligned with coordinate axes',
              fontsize=10, pad=6)
ax1.set_xlim(-1.5,1.5); ax1.set_ylim(-1.5,1.5); ax1.set_zlim(-1.5,1.5)
ax1.set_xlabel(r'$x$'); ax1.set_ylabel(r'$y$'); ax1.set_zlabel(r'$z$')
ax1.view_init(elev=ELEV, azim=AZIM)
hl,ll = ax1.get_legend_handles_labels()
ax1.legend(hl[:5], ll[:5], loc='upper left', fontsize=7.5)

# ══ RIGHT: Point B (h× frame + transported A-frame) ═════════════════════════
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
draw_grid(ax2, X0, Y0, Z0, color='dimgray', alpha=0.42, lw=0.75,
          label=r'Unperturbed $g^{(0)}_{\mu\nu}$')
draw_grid(ax2, X0, Y0, Z0, hc=h, color='#C41E3A', alpha=0.88, lw=1.55,
          label=r'$h_\times$ deformed metric')

# Native tetrad at B  (solid)
quiv(ax2, O, [e1B, e2B, e3B], SOLID,
     [r'$e_{\hat 1}(B)$: along $+45°$',
      r'$e_{\hat 2}(B)$: along $-45°$',
      r'$e_{\hat 3}(B)$'],
     lw=2.8, alpha=1.0, ls='-')
for v, c, lbl in zip([e1B, e2B, e3B], SOLID,
                      [r'$e_{\hat{1}}(B)$', r'$e_{\hat{2}}(B)$', r'$e_{\hat{3}}(B)$']):
    ax2.text(*(v*1.15 + O), lbl, fontsize=9, color=c, fontweight='bold')

# Transported frame from A (dashed) — arrives WITHOUT spin-connection correction
quiv(ax2, O, [e1A, e2A, e3A], DASHED,
     [r'$\tilde e_{\hat 1}$: transported from $A$  (no $\omega$)',
      r'$\tilde e_{\hat 2}$: transported from $A$',
      r'$\tilde e_{\hat 3}$: unchanged'],
     lw=1.9, alpha=0.60, ls='--')

# ── Spin-connection arc in xy-plane (0 → 45°) ────────────────────────────────
theta = np.linspace(0, np.pi/4, 80)
r_arc = 0.65
ax2.plot(r_arc*np.cos(theta), r_arc*np.sin(theta), np.zeros_like(theta),
         color='darkorange', lw=3.2, zorder=20,
         label=r'$\omega_\mu^{\ ab}\,dx^\mu$: $45°$ Lorentz rotation')
# tiny arrowhead at tip of arc
dt = theta[1]-theta[0]
ax2.quiver(r_arc*np.cos(theta[-1]-dt), r_arc*np.sin(theta[-1]-dt), 0,
           -r_arc*np.sin(theta[-1])*dt*6,
            r_arc*np.cos(theta[-1])*dt*6, 0,
           color='darkorange', arrow_length_ratio=5.0, lw=3.2, zorder=21)
# arc label
ax2.text(r_arc*np.cos(np.pi/8)*1.25, r_arc*np.sin(np.pi/8)*1.15, 0.10,
         r'$\omega\cdot dx$', fontsize=9.5, color='darkorange', fontweight='bold')

ax2.scatter(*O, s=70, color='black', zorder=10)
ax2.text(0.06, 0.06, 1.40, r'$B$', fontsize=14, fontweight='bold', color='#111')

ax2.set_title(r'Point $B$: $h_\times$ polarisation  ($h_{xy}=h_{yx}=h$)' + '\n'
              r'Solid = native $e_{\hat a}(B)$;  Dashed = bare-transported $\tilde e_{\hat a}$',
              fontsize=10, pad=6)
ax2.set_xlim(-1.5,1.5); ax2.set_ylim(-1.5,1.5); ax2.set_zlim(-1.5,1.5)
ax2.set_xlabel(r'$x$'); ax2.set_ylabel(r'$y$'); ax2.set_zlabel(r'$z$')
ax2.view_init(elev=ELEV, azim=AZIM)
hl2, ll2 = ax2.get_legend_handles_labels()
ax2.legend(hl2[:7], ll2[:7], loc='upper left', fontsize=7.2)

# ── Footer caption ─────────────────────────────────────────────────────────────
fig.text(0.5, 0.008,
    r'As the GW propagates by $\Delta\phi=\pi/2$, the natural vierbein rotates '
    r'$45°$ in the transverse plane.  The spin connection $\omega_\mu^{\ ab}$ is the '
    r'$\mathrm{SO}(1,3)$ Lie-algebra element (orange arc) mapping '
    r'$\tilde e_{\hat a}$ (dashed, naive transport) onto $e_{\hat a}(B)$ (solid). '
    r'Its antisymmetry $\omega_{\mu ab}=-\omega_{\mu ba}$ reflects the Lorentz-rotation structure; '
    r'looping $\omega$ encodes curvature via $R^{ab}=d\omega^{ab}+\omega^{ac}\wedge\omega_c{}^b$.',
    ha='center', fontsize=8.5, color='#333', style='italic')

plt.tight_layout(rect=[0, 0.045, 1, 0.965])
plt.savefig('figures/fig_ch2_spin_connection_transport.png', dpi=220, bbox_inches='tight')
plt.show()
print("Saved.")