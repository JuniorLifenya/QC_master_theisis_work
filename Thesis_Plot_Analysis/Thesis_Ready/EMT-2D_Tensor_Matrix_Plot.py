"""
Energy-Momentum Tensor T^{mu nu} — Publication-quality visualization
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
C00 = '#1B4F72'
C0I = '#1ABC9C'
CI0 = '#8E44AD'
CIJ = '#C0392B'
CID = "#D66000"

rows = [r'$\rho=0$', r'$\rho=1$', r'$\rho=2$', r'$\rho=3$']
cols = [r'$\sigma=0$', r'$\sigma=1$', r'$\sigma=2$', r'$\sigma=3$']

# Updated cell labels with line breaks and \text{} for upright roman font
cell_labels = [
    [r'$T^{00}$' + '\n' + r'$\text{Energy density}$',
     r'$T^{01}$' + '\n' + r'$\text{p-density } x$',
     r'$T^{02}$' + '\n' + r'$\text{p-density } y$',
     r'$T^{03}$' + '\n' + r'$\text{p-density } z$'],
    
    [r'$T^{10}$' + '\n' + r'$\text{E-flux } x$',
     r'$T^{11}$' + '\n' + r'$\text{Stress } xx$',
     r'$T^{12}$' + '\n' + r'$\text{Shear } xy$',
     r'$T^{13}$' + '\n' + r'$\text{Shear } xz$'],
    
    [r'$T^{20}$' + '\n' + r'$\text{E-flux } y$',
     r'$T^{21}$' + '\n' + r'$\text{Shear } yx$',
     r'$T^{22}$' + '\n' + r'$\text{Stress } yy$',
     r'$T^{23}$' + '\n' + r'$\text{Shear } yz$'],
    
    [r'$T^{30}$' + '\n' + r'$\text{E-flux } z$',
     r'$T^{31}$' + '\n' + r'$\text{Shear } zx$',
     r'$T^{32}$' + '\n' + r'$\text{Shear } zy$',
     r'$T^{33}$' + '\n' + r'$\text{Stress } zz$'],
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
    r = FancyBboxPatch((x0, y0), w, h,
                       boxstyle='round,pad=0.08',
                       facecolor='none', edgecolor=col,
                       linewidth=1, linestyle='--', zorder=4)
    ax.add_patch(r)
    if lblpos == 'top':
        ax.text(x0+w/2, y0-0.10, lbl, ha='center', va='bottom',
                fontsize=9, color=col, fontweight='bold')
    elif lblpos == 'left':
        ax.text(x0-0.1, y0+h/2, lbl, ha='right', va='center',
                fontsize=9, color=col, fontweight='bold', rotation=90)

bracket_box(ax, 0.06, 0.06, 0.88, 0.88, r'$T^{00}$: Energy density', C00)
bracket_box(ax, 1.06, 0.06, 2.88, 0.88, r'$T^{0i}$: Momentum density ', C0I)
bracket_box(ax, 0.06, 1.06, 0.88, 2.88, r'$T^{i0}$: Energy flux', CI0, 'left')
bracket_box(ax, 1.06, 1.09, 2.88, 2.88,
            r'$T^{ij}$: Stress tensor (momentum flux)', CIJ)

for i, lbl in enumerate(rows):
    ax.text(-0.21, i+0.40, lbl, ha='right', va='center',
            fontsize=10, color='#333', fontweight='bold')
for j, lbl in enumerate(cols):
    ax.text(j+0.50, -0.27, lbl, ha='center', va='bottom',
            fontsize=10, color='#333', fontweight='bold')

ax.text(2.0, -0.50, r'$\sigma$  (column index)', ha='center', va='bottom',
        fontsize=10, color='#222', style='italic')
ax.text(-0.65, 2.5, r'$\rho$  (row index)', ha='center', va='center',
        fontsize=10, color='#222', style='italic', rotation=90)

ax.set_title(r'Energy–Momentum Tensor $T^{\rho\sigma}$',
             fontsize=14, fontweight='bold', color='#111', pad=18)


plt.savefig('figures/Energy_Momentum_Tensor.png', format='pdf', bbox_inches='tight')
print("Saved:")
plt.show()