"""
fig_ch1_gw_polarisation.py
Chapter 1 Visual Guide — Figure 1
GW + and x polarisation of a ring of test particles.
Save as: figures/fig_ch1_polarisation.png
"""
import numpy as np
import matplotlib.pyplot as plt
import os; os.makedirs("figures", exist_ok=True)

theta = np.linspace(0, 2*np.pi, 300)
x0, y0 = np.cos(theta), np.sin(theta)
h = 0.45

fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
phases = [0, np.pi/4, np.pi/2, 3*np.pi/4]

for ax, phi in zip(axes, phases):
    xp = x0 * (1 + h/2*np.cos(phi))
    yp = y0 * (1 - h/2*np.cos(phi))
    # h× via 45° rotation
    ang = np.pi/4
    xr =  x0*np.cos(ang) + y0*np.sin(ang)
    yr = -x0*np.sin(ang) + y0*np.cos(ang)
    xb =  xr*(1+h/2*np.cos(phi))*np.cos(-ang) + yr*(1-h/2*np.cos(phi))*np.sin(-ang)
    yb = -xr*(1+h/2*np.cos(phi))*np.sin(-ang) + yr*(1-h/2*np.cos(phi))*np.cos(-ang)

    ax.plot(x0, y0, 'k--', lw=1, alpha=0.35)
    ax.fill(xp, yp, alpha=0.18, color='steelblue')
    ax.plot(xp, yp, color='steelblue', lw=2, label=r'$h_+$')
    ax.fill(xb, yb, alpha=0.18, color='coral')
    ax.plot(xb, yb, color='coral', lw=2, label=r'$h_\times$')
    ax.set_aspect('equal'); ax.set_xlim(-1.8,1.8); ax.set_ylim(-1.8,1.8)
    ax.set_title(fr'$\omega t = {phi/np.pi:.2g}\pi$', fontsize=11)
    ax.set_xlabel('x'); ax.set_ylabel('y')

axes[0].legend(loc='upper right', fontsize=9)
fig.suptitle(r'GW polarisation: $h_+$ (blue) and $h_\times$ (coral) deformation of a test-particle ring',
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig('figures/fig_ch1_polarisation.png', dpi=200, bbox_inches='tight')
print("Saved: figures/fig_ch1_polarisation.png")
