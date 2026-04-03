"""
fig_ch2_tetrad_shear.py
Chapter 2 — Global coordinate shear vs rigid local tetrad.
Motivates why the spin connection is needed.
Side by side: h+ and h× at phase=0.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os; os.makedirs("figures", exist_ok=True)

def deform(X, Y, h_plus=0, h_cross=0):
    dX = 0.5*(h_plus*X + h_cross*Y)
    dY = 0.5*(h_cross*X - h_plus*Y)
    return X+dX, Y+dY

def draw_grid(ax, Xd, Yd, Z, col, alpha, lw, label=None):
    drawn = False
    for i in range(Xd.shape[0]):
        for j in range(Xd.shape[1]):
            lbl = label if not drawn else None
            ax.plot(Xd[i,j,:], Yd[i,j,:], Z[i,j,:], color=col, alpha=alpha, lw=lw, label=lbl)
            drawn = True
    for i in range(Xd.shape[0]):
        for k in range(Xd.shape[2]):
            ax.plot(Xd[i,:,k], Yd[i,:,k], Z[i,:,k], color=col, alpha=alpha, lw=lw)
    for j in range(Xd.shape[1]):
        for k in range(Xd.shape[2]):
            ax.plot(Xd[:,j,k], Yd[:,j,k], Z[:,j,k], color=col, alpha=alpha, lw=lw)

def draw_tetrad(ax):
    L = 1.1
    for d,c,l in [([L,0,0],'red',r'$e_{\hat{1}}$'),
                  ([0,L,0],'green',r'$e_{\hat{2}}$'),
                  ([0,0,L],'royalblue',r'$e_{\hat{3}}$')]:
        ax.quiver(0,0,0,*d,color=c,arrow_length_ratio=0.2,lw=2.5)
        ax.text(d[0]*1.12,d[1]*1.12,d[2]*1.12,l,fontsize=10,color=c,fontweight='bold')

pts = np.linspace(-1,1,4)
X0,Y0,Z0 = np.meshgrid(pts,pts,pts,indexing='ij')
h = 0.45

fig = plt.figure(figsize=(12,5.5))
fig.suptitle('Global coordinate shear (blue) vs rigid local tetrad (RGB arrows)\n'
             r'Motivates the spin connection $\omega_\mu^{ab}$', fontsize=12)

for col_idx, (title, kwargs) in enumerate([
        (r'$h_+$ polarisation: $h_{xx}=-h_{yy}=h$',  dict(h_plus=h)),
        (r'$h_\times$ polarisation: $h_{xy}=h_{yx}=h$', dict(h_cross=h)),
]):
    ax = fig.add_subplot(1,2,col_idx+1, projection='3d')
    Xd,Yd = deform(X0,Y0,**kwargs)
    draw_grid(ax, X0, Y0, Z0, 'gray', 0.20, 0.8, 'Unperturbed coords')
    draw_grid(ax, Xd, Yd, Z0, '#2171B5' if col_idx==0 else '#E31A1C',
              0.85, 1.6, 'Deformed coords')
    draw_tetrad(ax)
    ax.set_xlim(-1.5,1.5); ax.set_ylim(-1.5,1.5); ax.set_zlim(-1.5,1.5)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.set_title(title, fontsize=11, pad=6)
    ax.view_init(elev=22, azim=38)
    handles,labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2],labels[:2],loc='upper left',fontsize=8)

plt.tight_layout()
plt.savefig('figures/fig_ch2_tetrad_shear.png', dpi=200, bbox_inches='tight')
print("Saved: figures/fig_ch2_tetrad_shear.png")
