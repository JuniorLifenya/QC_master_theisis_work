"""
Energy-Momentum Tensor g_{mu nu} — Publication-quality visualization
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'stix',
    'font.size': 11,
    'figure.dpi': 180,
})

BG  = "#FFFFFF"
C00 = "#073E62"
C0I = "#08866D"
CI0 = "#8B1583"
CIJ = "#85281E"
CID = "#A14900"

rows = [r'$\rho=0$', r'$\rho=1$', r'$\rho=2$', r'$\rho=3$']
cols = [r'$\sigma=0$', r'$\sigma=1$', r'$\sigma=2$', r'$\sigma=3$']

# Updated cell labels with line breaks and \text{} for upright roman font
cell_labels = [
    [r'$g_{00}$' + '\n' + r'$\text{Time Scale}$',
     r'$g_{01}$' + '\n' + r'',
     r'$g_{02}$' + '\n' + r'',
     r'$g_{03}$' + '\n' + r''],
    
    [r'$g_{10}$' + '\n' + r'',
     r'$g_{11}$' + '\n' + r'$\text{Length} xx$',
     r'$g_{12}$' + '\n' + r'$\text{Angle } xy$',
     r'$g_{13}$' + '\n' + r'$\text{Angle } xz$'],
    
    [r'$g_{20}$' + '\n' + r'',
     r'$g_{21}$' + '\n' + r'$\text{Angle } yx$',
     r'$g_{22}$' + '\n' + r'$\text{Length } yy$',
     r'$g_{23}$' + '\n' + r'$\text{Angle } yz$'],
    
    [r'$g_{30}$' + '\n' + r'',
     r'$g_{31}$' + '\n' + r'$\text{Angle } zx$',
     r'$g_{32}$' + '\n' + r'$\text{Angle } zy$',
     r'$g_{33}$' + '\n' + r'$\text{Length} zz$'],
]

block_colors = [
    [C00, C0I, C0I, C0I],
    [CI0, CID, CIJ, CIJ],
    [CI0, CIJ, CID, CIJ],
    [CI0, CIJ, CIJ, CID],
]

# Create a single plot with a more square aspect ratio
fig, ax = plt.subplots(figsize=(12, 11), facecolor=BG)
ax.set_facecolor(BG)



# ── MATRIX PANEL ─────────────────────────────────────────────────────────────
ax.set_xlim(-0.5, 4.7)
ax.set_ylim(-0.5, 4.6)
ax.invert_yaxis()
ax.axis('off')

N = 4
cw, ch = 0.88, 0.88
for i in range(N):
    for j in range(N):
        x, y = j + 0.06, i + 0.06
        rect = FancyBboxPatch((x, y), cw, ch,
                              boxstyle='round,pad=0.04',
                              facecolor=block_colors[i][j],
                              edgecolor='black', linewidth=2.0,
                              alpha=0.93, zorder=2)
        ax.add_patch(rect)
        ax.text(x + cw/2, y + ch/2, cell_labels[i][j],
                ha='center', va='center', fontsize=8.6,
                color='white', fontweight='bold',
                multialignment='center', zorder=3, linespacing=1.35)

def bracket_box(ax, x0, y0, w, h, lbl, col, lblpos='top'):
    if lblpos == 'top':
        ax.text(x0+w/2, y0-0.10, lbl, ha='center', va='bottom',
                fontsize=9, color=col, fontweight='bold')
    elif lblpos == 'left':
        ax.text(x0-0.1, y0+h/2, lbl, ha='right', va='center',
                fontsize=9, color=col, fontweight='bold', rotation=90)


bracket_box(ax, 1.05, 0.06, 2.88, 0.88, r'$g_{0j}$: Time-Space Mixing ', C0I)
bracket_box(ax, 0.06, 1.06, 0.88, 2.88, r'$g_{i0}$: Space-Time Mixing', CI0, 'left')
bracket_box(ax, 1.05, 4.36, 2.88, 0.88,
            r'$g_{ij}$: Spatial Geometry', CIJ)


ax.text(2.0, -0.45, r'$\sigma$  (column index)', ha='center', va='bottom',
        fontsize=10, color='#222', style='italic')
ax.text(-0.40, 2.5, r'$\rho$  (row index)', ha='center', va='center',
        fontsize=10, color='#222', style='italic', rotation=90)

ax.set_title(r'The Metric Tensor $g_{\rho\sigma}$',
             fontsize=14, fontweight='bold', color='#111', pad=18)


plt.savefig('figures/Metric_Tensor.png', format='pdf', bbox_inches='tight')
print("Saved:")
plt.show()