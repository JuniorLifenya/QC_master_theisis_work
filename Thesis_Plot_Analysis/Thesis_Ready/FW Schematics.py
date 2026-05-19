
 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import os
 
os.makedirs("figures", exist_ok=True)
 
c     = 2.99792458e8
hbar  = 1.054571817e-34
G     = 6.67430e-11
m_e   = 9.10938356e-31
e_ch  = 1.60217663e-19
alpha = 1/137.035999
a_0   = hbar/(m_e*alpha*c)
eV    = e_ch
l_Pl  = np.sqrt(hbar*G/c**3)
 
plt.rcParams.update({'font.family':'serif','font.size':12,
    'axes.labelsize':13,'axes.titlesize':11,'legend.fontsize':9,
    'axes.grid':True,'grid.alpha':0.3,'lines.linewidth':2.0,'figure.dpi':150})


def fig_fw_schematic():
    fig,axes = plt.subplots(1,3,figsize=(13,4.5))
    def dm(ax,data,title):
        norm = TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1)
        ax.imshow(data,cmap='RdBu',norm=norm,aspect='equal')
        for i in range(4):
            for j in range(4):
                v=data[i,j]
                if abs(v)>0.05:
                    ax.text(j,i,f'{v:.1f}',ha='center',va='center',fontsize=12,
                            color='white' if abs(v)>0.5 else 'black')
        ax.set_xticks([]); ax.set_yticks([])
        for k in range(5): ax.axhline(k-0.5,color='gray',lw=0.4); ax.axvline(k-0.5,color='gray',lw=0.4)
        ax.axhline(1.5,color='k',lw=2); ax.axvline(1.5,color='k',lw=2)
        ax.set_title(title,fontsize=10,pad=6)
    dm(axes[0],np.array([[1,0,0,0.7],[0,1,0.7,0],[0,0.7,-1,0],[0.7,0,0,-1]],float),
       'Initial Dirac $H$\n(off-diag = odd operators)')
    dm(axes[1],np.array([[1,0,0,0.3],[0,1,0.3,0],[0,0.3,-1,0],[0.3,0,0,-1]],float),
       r'After $e^{i\hat{S}_1}He^{-i\hat{S}_1}$' '\n(reduced odd)')
    dm(axes[2],np.array([[1,0,0,0],[0,0.8,0,0],[0,0,-1,0],[0,0,0,-0.8]],float),
       r'$H_{\rm eff}$ (final)' '\nBlock-diagonal')
    for x,t in [(0.365,r'$U_1$'),(0.640,r'$U_2$')]:
        fig.text(x,0.5,r'$\longrightarrow$',ha='center',va='center',fontsize=18)
        fig.text(x,0.37,t,ha='center',va='center',fontsize=10)
    fig.suptitle('FW transformation: iterative block diagonalisation (blue=+, red=-)',y=1.01)
    plt.tight_layout()
    plt.savefig('figures/fig10_fw_schematic.png',bbox_inches='tight')
    plt.show()
    plt.close()
    print("ok fig10_fw_schematic.png")

if __name__=="__main__":    fig_fw_schematic()