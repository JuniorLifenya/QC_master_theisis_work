import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import os


# ─── FIG 11: KINETIC STRAIN IN MOMENTUM SPACE ───────────

px = np.linspace(-3.5,3.5,200)
PX,PY = np.meshgrid(px,px)
h = 0.5
phases = [0,np.pi/4,np.pi/2,3*np.pi/4]
fig,axes = plt.subplots(2,4,figsize=(14,7))
for row,pol in enumerate(['plus','cross']):
    for col,phi in enumerate(phases):
            ax = axes[row,col]
            phi_eff = phi + 1e-9 if phi == np.pi/2 else phi
            V = h*np.cos(phi_eff)*(PX**2-PY**2 if pol=='plus' else PX*PY)
            lim = max(abs(V).max(),1e-9)
            ax.contourf(PX,PY,V,levels=20,cmap='RdBu_r',vmin=-lim,vmax=lim)
            ax.contour(PX,PY,V,levels=[0],colors='k',linewidths=0.7)
            ax.set_aspect('equal')
            ax.set_title(fr'${"h_+" if pol=="plus" else "h_\\times"}$, $\omega t={phi:.2f}$',fontsize=10)
            if col==0: ax.set_ylabel(r'$p_y/p_0$')
            if row==1: ax.set_xlabel(r'$p_x/p_0$')
fig.suptitle(r'Kinetic strain $h_{ij}\hat{p}^i\hat{p}^j$ in momentum space'
                 r' (top: $h_+$, bottom: $h_\times$)',y=1.01)
plt.tight_layout()
plt.savefig('figures/fig11_kinetic_strain.png',bbox_inches='tight')
plt.show()
plt.close()
print("ok fig11_kinetic_strain.png")
 