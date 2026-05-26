"""
fig_derivation_chain.py — Chapter 1 logical chain flowchart
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os; os.makedirs("figures", exist_ok=True)

fig, ax = plt.subplots(figsize=(14, 3.8))
ax.set_xlim(0, 14); ax.set_ylim(0, 4); ax.axis('off')

labels = [
    "Linearized GR\n$g_{\\mu\\nu}=\\eta_{\\mu\\nu}+\\kappa h_{\\mu\\nu}$",
    "Tetrads\n$e^a_{\\mu}=\\delta^a_{\\mu}$$+\\frac{\\kappa}{2}h^a_{\\mu}$",
    "Spin conn.\n$\\omega^{ab}_{\\mu}=$$\kappa(\\partial^a h^b_{\\mu})\sigma_{ab}$",
    "Belinfante\n$\\mathcal{L}_{\\rm int}=$\n$-\\frac{\\kappa}{2}h_{\\mu\\nu}T^{\\mu\\nu}_{\\rm BR}$",
    "Legendre\n$\\hat{H}_{\\rm int}$\n$=f(h_{ij},\\alpha^i,\\hat{p})$",
    "FW\n$\\times 3$ iters\n$\\mathcal{O}(1/m^3)$",
    "$\\hat{H}_{\\rm eff}$\n9 terms",
    "Quantum Sensing\nsel. rules\n$\\Delta m=\\pm2$",
]
xs = [0.5, 2.3, 4.1, 5.9, 7.7, 9.5, 11.3, 13.1]
colors = ['#1a3a5c','#1a4a2c','#3a2a5c','#5c2a1a','#3a3a1a','#1a3a5c','#5c1a2a','#1a4a2c']
chapters = {0.5:'Ch.2', 4.1:'Ch.3', 7.7:'Ch.4', 13.1:'Ch.5'}

for i,(xc,txt,col) in enumerate(zip(xs, labels, colors)):
    p = FancyBboxPatch((xc-0.82, 0.5), 1.64, 2.9,
                       boxstyle="round,pad=0.1",
                       facecolor=col, edgecolor='black', lw=1.2, alpha=0.9)
    ax.add_patch(p)
    ax.text(xc, 1.92, txt, ha='center', va='center', fontsize=8,
            color='white', multialignment='center')
    if i < len(xs)-1:
        ax.annotate('', xy=(xc+0.88,1.92), xytext=(xc+0.82,1.92),
                    arrowprops=dict(arrowstyle='->', color='cyan', lw=1.8))
    if xc in chapters:
        ax.text(xc, 0.22, chapters[xc], ha='center', fontsize=8,
                color="#212121ff", style='italic')

ax.set_title('Logical derivation chain of the thesis', fontsize=12,fontweight= "bold",  color='black', y = 0.9)
fig.patch.set_facecolor('#F5DEB3')
ax.set_facecolor('#F5DEB3')
plt.tight_layout()
plt.show()
plt.savefig('Thesis_Ready_Plots/fig_derivation_chain.png', dpi=200, bbox_inches='tight')
print("Saved: figures/fig_derivation_chain.png")
