"""
fig_ch1_nv_center.py
Chapter 1 Visual Guide — Figure 2
NV-center spin triplet with GW-induced transitions.
Vertical: energy. Horizontal: m_s = -1, 0, +1
Shows both MW (dipole, Δm=±1) and GW (quadrupole, Δm=±2) channels.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os; os.makedirs("figures", exist_ok=True)

fig, ax = plt.subplots(figsize=(7, 6))

D = 2.87   # GHz zero-field splitting
E = 0.005  # strain-induced splitting (small)

# Energy levels (in GHz)
levels = {
    r'$|{-1}\rangle$': (-1, D - E),
    r'$|0\rangle$':    ( 0, 0.0),
    r'$|{+1}\rangle$': (+1, D + E),
}
x_pos = {-1: -1.0, 0: 0.0, 1: 1.0}

for lab, (ms, en) in levels.items():
    x = x_pos[ms]
    ax.hlines(en, x-0.35, x+0.35, color='k', lw=2.5, zorder=3)
    ax.text(x, en - 0.08, lab, ha='center', va='top', fontsize=13)
    ax.text(x, en + 0.05, f'{en:.3f} GHz', ha='center', va='bottom',
            fontsize=8, color='gray')

# MW transition |0⟩ → |−1⟩  (Δm = −1, dipole)
ax.annotate('', xy=(-1.0, D-E-0.04), xytext=(0.0, 0.04),
            arrowprops=dict(arrowstyle='<->', color='steelblue', lw=2.0,
                            connectionstyle='arc3,rad=-0.25'))
ax.text(-0.65, D/2, r'MW photon$\,$($\Delta m_s=\pm1$)', color='steelblue',
        fontsize=9, ha='center', rotation=70)

# MW transition |0⟩ → |+1⟩
ax.annotate('', xy=(1.0, D+E-0.04), xytext=(0.0, 0.04),
            arrowprops=dict(arrowstyle='<->', color='steelblue', lw=2.0,
                            connectionstyle='arc3,rad=0.25'))

# GW quadrupole |−1⟩ → |+1⟩  (Δm = +2)
ax.annotate('', xy=(1.0, D+E+0.06), xytext=(-1.0, D-E+0.06),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2.5,
                            linestyle='dashed',
                            connectionstyle='arc3,rad=-0.4'))
ax.text(0.0, D + 0.30,
        r'GW quadrupole$\,$($\Delta m_s=\pm2$)', color='red',
        fontsize=9, ha='center')

# Arrows to show the GW strain shifts the splitting
ax.annotate('', xy=(-1.0, D-E), xytext=(-1.0, D+0.03),
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
ax.annotate('', xy=(1.0, D+E), xytext=(1.0, D-0.03),
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
ax.text(1.42, D, r'$\delta E\propto h_{ij}\hat{p}^i\hat{p}^j$',
        color='green', fontsize=8, va='center')

ax.set_xlim(-1.8, 2.0)
ax.set_ylim(-0.3, 3.5)
ax.set_ylabel('Energy (GHz)', fontsize=12)
ax.set_xticks([-1, 0, 1])
ax.set_xticklabels([r'$m_s=-1$', r'$m_s=0$', r'$m_s=+1$'], fontsize=11)
ax.set_title('NV-centre spin triplet: MW dipole vs GW quadrupole coupling\n'
             r'Zero-field splitting $D = 2.87$ GHz', fontsize=11)

leg = [mpatches.Patch(color='steelblue', label=r'MW: $\Delta m_s=\pm1$'),
       mpatches.Patch(color='red',       label=r'GW: $\Delta m_s=\pm2$'),
       mpatches.Patch(color='green',     label=r'GW strain shift $\delta E$')]
ax.legend(handles=leg, loc='upper left', fontsize=9)
ax.grid(axis='y', alpha=0.25)

plt.tight_layout()
plt.savefig('figures/fig_ch1_nv_center.png', dpi=200, bbox_inches='tight')
print("Saved: figures/fig_ch1_nv_center.png")
