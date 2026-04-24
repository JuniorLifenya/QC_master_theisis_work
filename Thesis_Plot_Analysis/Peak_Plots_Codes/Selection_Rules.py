
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









def fig_selection_rules():
    fig,axes = plt.subplots(1,2,figsize=(13,8))
    E = {(1,0):-13.6,(2,0):-13.6/4,(2,1):-13.6/4,(3,0):-13.6/9,(3,1):-13.6/9,(3,2):-13.6/9}
    x = {(1,0):0,(2,0):-1.5,(2,1):0,(3,0):-3,(3,1):-1.5,(3,2):0}
    lbl = {0:'s',1:'p',2:'d'}
    def draw(ax,title,trans,tcol,rule):
        for nl,Ev in E.items():
            ax.hlines(Ev,x[nl]-0.55,x[nl]+0.55,'k',lw=2.5,zorder=3)
            ax.text(x[nl],Ev+0.35,f'$|{nl[0]}{lbl[nl[1]]}\\rangle$',ha='center',fontsize=11)
        for n,Ev in [(-13.6,'n=1'),(-13.6/4,'n=2'),(-13.6/9,'n=3')]:
            ax.text(-4.5,n,Ev,color='gray',fontsize=9,va='center')
        for (ni,nf) in trans:
            ax.annotate('',xy=(x[nf],E[nf]+0.25),xytext=(x[ni],E[ni]-0.25),
                        arrowprops=dict(arrowstyle='->',color=tcol,lw=2.5,
                                        connectionstyle='arc3,rad=0.25'))
        ax.set_xlim(-5,1.5); ax.set_ylim(-16,2)
        ax.set_xlabel('Angular momentum $l$',fontsize=11)
        ax.set_ylabel('Energy [eV]',fontsize=11)
        ax.set_title(title+'\n'+rule,fontsize=11)
        ax.set_xticks([-3,-1.5,0]); ax.set_xticklabels([r'$l=0$',r'$l=1$',r'$l=2$'])
    draw(axes[0],'Electromagnetic (dipole)',
         [((2,1),(1,0)),((3,1),(1,0)),((3,2),(2,1)),((3,0),(2,1))],
         'steelblue',r'$\Delta l=\pm1$,  $\Delta m=0,\pm1$')
    draw(axes[1],'Gravitational wave (quadrupole)',
         [((3,2),(1,0)),((3,0),(1,0)),((3,2),(2,0))],
         'red',r'$\Delta l=0,\pm2$,  $\Delta m=\pm2$')
    fig.suptitle('Selection rules: rank-1 (EM) vs rank-2 (GW) tensor operator (Wigner-Eckart)',y=1.00)
    plt.tight_layout()
    plt.savefig('figures/fig6_selection_rules.png',bbox_inches='tight')
    plt.show()
    plt.close()
    print("ok fig6_selection_rules.png")

if __name__=="__main__":    fig_selection_rules()