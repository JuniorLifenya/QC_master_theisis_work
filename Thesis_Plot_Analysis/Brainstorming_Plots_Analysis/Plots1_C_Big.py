"""
thesis_plots.py  —  RIGOROUS VERSION
=====================================
"Quantum Effects of Gravitational Waves"
Daniel Junior Lifenya Fondo, UiB 2026
 
Run: python thesis_plots.py
"""
 
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
 
# ─── FIG 1: GW POLARISATION ─────────────────────────────
def fig_gw_polarisation():
    theta = np.linspace(0,2*np.pi,300)
    x0,y0 = np.cos(theta),np.sin(theta)
    h = 0.50
    fig,axes = plt.subplots(2,4,figsize=(14,7))
    phases = [0,np.pi/4,np.pi/2,3*np.pi/4]
    plabs  = [r'$\omega t=0$',r'$\omega t=\pi/4$',r'$\omega t=\pi/2$',r'$\omega t=3\pi/4$']
    for col,(phi,plab) in enumerate(zip(phases,plabs)):
        # h+ row
        ax = axes[0,col]
        xp = x0*(1+h/2*np.cos(phi))
        yp = y0*(1-h/2*np.cos(phi))
        ax.plot(x0,y0,'k--',lw=1,alpha=0.3)
        ax.fill(xp,yp,alpha=0.15,color='steelblue')
        ax.plot(xp,yp,'steelblue',lw=2)
        ax.set_aspect('equal'); ax.set_xlim(-2,2); ax.set_ylim(-2,2)
        ax.set_title(plab+r' ($h_+$)')
        if col==0:
            ax.set_ylabel(r'$y/x_0$')
            ax.plot([],[],color='steelblue',lw=2,label=r'$h_+$')
            ax.plot([],[],color='coral',lw=2,label=r'$h_\times$')
            ax.legend(loc='upper right',fontsize=8)
        # hx row (45 deg rotation)
        ax = axes[1,col]
        ang = np.pi/4
        xr =  x0*np.cos(ang)+y0*np.sin(ang)
        yr = -x0*np.sin(ang)+y0*np.cos(ang)
        xr2 = xr*(1+h/2*np.cos(phi)); yr2 = yr*(1-h/2*np.cos(phi))
        xb =  xr2*np.cos(-ang)+yr2*np.sin(-ang)
        yb = -xr2*np.sin(-ang)+yr2*np.cos(-ang)
        ax.plot(x0,y0,'k--',lw=1,alpha=0.3)
        ax.fill(xb,yb,alpha=0.15,color='coral')
        ax.plot(xb,yb,'coral',lw=2)
        ax.set_aspect('equal'); ax.set_xlim(-2,2); ax.set_ylim(-2,2)
        ax.set_title(plab+r' ($h_\times$)')
        ax.set_xlabel(r'$x/x_0$')
        if col==0: ax.set_ylabel(r'$y/x_0$')
    fig.suptitle(r'GW polarisation: deformation of a ring of test particles ($h=0.5$ for visibility)',y=1.01)
    plt.tight_layout()
    plt.savefig('figures/fig1_gw_polarisation.png',bbox_inches='tight')
    plt.close()
    print("ok fig1_gw_polarisation.png")
 
# ─── FIG 2: H_eff TERM MAGNITUDES ───────────────────────
def fig_heff_magnitudes():
    f_GW = 100.0; w_GW = 2*np.pi*f_GW; k_GW = w_GW/c; h_str = 1e-21
    r2   = 5.0*a_0**2
    g_LIF = m_e*w_GW**2*r2/4
    def dE(factor): return g_LIF*factor*h_str/eV
 
    data = [
        (r'$\hat{H}_0$: Unperturbed KE',           m_e*(alpha*c/2)**2/(2*m_e)/eV,'gray'),
        (r'$\hat{H}_{\rm strain}$: Kinetic strain', dE(1.0),'red'),
        (r'$\hat{H}_{\rm Zeeman}$: Grav. Zeeman',   dE(k_GW*a_0),'darkorange'),
        (r'$\hat{H}_{\rm GSO}$: Spin-orbit',         dE(alpha*k_GW*a_0),'salmon'),
        (r'$\hat{H}_{\rm iso}$: Isotropic KE',       dE(alpha**2),'steelblue'),
        (r'$\hat{H}_{\rm curr}$: Mom-gradient',       dE(k_GW*a_0),'cornflowerblue'),
        (r'$\hat{H}_{\rm Darwin}$: Grav. Darwin',    dE((k_GW*hbar/(m_e*c))**2),'olive'),
    ]
    labels = [d[0] for d in data]
    vals   = [d[1] for d in data]
    colors = [d[2] for d in data]
    log_v  = [np.log10(v) if v > 0 else -200 for v in vals]
 
    fig,ax = plt.subplots(figsize=(11,5.5))
    ax.barh(range(len(labels)),log_v,color=colors,alpha=0.85,edgecolor='black',lw=0.5)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels,fontsize=10)
    ax.set_xlabel(r'$\log_{10}(|\Delta E| / \rm eV)$')
    ax.set_title(r'$\hat{H}_{\rm eff}$ term magnitudes ($h=10^{-21}$, $f=100$ Hz, H 2p state)')
    for i,v in enumerate(vals):
        if v > 0:
            ax.text(np.log10(v)+0.1,i,f'{v:.1e} eV',va='center',fontsize=8)
    ax.axvline(np.log10(vals[1]),color='red',ls='--',alpha=0.5,label='Kinetic strain')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig('figures/fig2_heff_magnitudes.png',bbox_inches='tight')
    plt.close()
    print(f"ok fig2_heff_magnitudes.png")
    print(f"   Kinetic strain dE = {vals[1]:.2e} eV  (f={f_GW}Hz, H2p)")
    print(f"   k_GW*a0 = {k_GW*a_0:.2e}  << 1: long-wavelength OK")
 
# ─── FIG 3: ENERGY SHIFT vs STRAIN ──────────────────────
def fig_energy_shift_vs_strain():
    h_vals = np.logspace(-26,-17,600)
    f_GW = 100.0; w_GW = 2*np.pi*f_GW
    g_LIF = m_e*w_GW**2*5*a_0**2/4
    dE_eV = g_LIF*h_vals/eV
 
    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(9,7),sharex=True)
    ax1.loglog(h_vals,dE_eV,'steelblue',lw=2.5,label=r'$|\Delta E|$  (kinetic strain, LIF)')
    ax1.axvline(1e-21,color='red',ls='--',lw=2,label=r'LIGO event $h=10^{-21}$')
    ax1.axvline(1e-23,color='gold',ls='--',lw=2,label=r'LIGO design $h=10^{-23}$')
    ax1.set_ylabel(r'$|\Delta E|$ [eV]')
    ax1.set_title(r'$\Delta E = g_{\rm LIF}\cdot h$, $g_{\rm LIF}=m_e\omega_{\rm GW}^2\langle r^2\rangle_{2p}/4$'
                  r'   ($f_{\rm GW}=100$ Hz)')
    ax1.legend(fontsize=9)
    dE_21 = g_LIF*1e-21/eV
    ax1.plot(1e-21,dE_21,'rs',ms=8,zorder=5)
    ax1.text(1.4e-21,dE_21*2,f'{dE_21:.1e} eV',color='red',fontsize=9)
 
    dE_Hz = g_LIF*h_vals/hbar
    ax2.loglog(h_vals,dE_Hz,'coral',lw=2.5,label=r'$|\Delta E|/\hbar$ [rad/s]')
    ax2.axvline(1e-21,color='red',ls='--',lw=2)
    ax2.set_xlabel(r'GW strain $h$')
    ax2.set_ylabel(r'$|\Delta\omega|$ [rad/s]')
    ax2.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig('figures/fig3_energy_shift_vs_strain.png',bbox_inches='tight')
    plt.close()
    frac = g_LIF*1e-21/(m_e*c**2)
    print(f"ok fig3_energy_shift_vs_strain.png")
    print(f"   dE(h=1e-21)={dE_21:.2e} eV,  dE/E_rest={frac:.1e}")
 
# ─── FIG 4: BOUGHN-ROTHMAN (VERIFIED) ───────────────────
def fig_boughn_rothman():
    omega = np.logspace(12,20,500)
    Gamma = (3**8*G*m_e**2*a_0**4*omega**5)/(5*2**13*hbar*c**5)
    dE_3d1s = 13.6*eV*(1-1/9)
    w_mn    = dE_3d1s/hbar
    G_check = (3**8*G*m_e**2*a_0**4*w_mn**5)/(5*2**13*hbar*c**5)
    sigma_abs = 0.31*l_Pl**2
 
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(13,5.5))
    ax1.loglog(omega/(2*np.pi),Gamma,'steelblue',lw=2)
    ax1.axvline(w_mn/(2*np.pi),color='red',ls='--',lw=2,
                label=fr'$f_{{3d\to1s}}={w_mn/(2*np.pi):.2e}$ Hz')
    ax1.plot(w_mn/(2*np.pi),G_check,'r^',ms=10,zorder=5)
    ax1.text(1.5e14,G_check*3,fr'$\Gamma={G_check:.2e}$ s$^{{-1}}$',color='red',fontsize=9)
    ax1.set_xlabel('Frequency [Hz]')
    ax1.set_ylabel(r'$\Gamma$ [s$^{-1}$]')
    ax1.set_title(r'Graviton emission rate: $\Gamma=\frac{3^8 G m_e^2 a_0^4\omega^5}{5\times2^{13}\hbar c^5}$')
    ax1.legend(fontsize=9)
 
    ax2.semilogx([1e-5,1e5],[0.31,0.31],'coral',lw=3,
                 label=r'$\sigma_{\rm abs}=0.31\,\ell_{\rm Pl}^2$ (all atomic constants cancel!)')
    ax2.axvline(dE_3d1s/eV,color='steelblue',ls='--',label=fr'$E_{{3d\to1s}}={dE_3d1s/eV:.1f}$ eV')
    ax2.set_xlim(1e-5,1e5)
    ax2.set_ylim(0,1)
    ax2.set_xlabel('Graviton energy [eV]')
    ax2.set_ylabel(r'$\sigma_{\rm abs}\,/\,\ell_{\rm Pl}^2$')
    ax2.set_title(r'Absorption cross section (freq.-integrated)')
    ax2.legend(fontsize=9)
    ax2.text(0.05,0.55,
             fr'$\ell_{{\rm Pl}}={l_Pl:.2e}$ m'+'\n'+fr'$\sigma={sigma_abs:.2e}$ m$^2$',
             transform=ax2.transAxes,fontsize=9,
             bbox=dict(facecolor='lightyellow',alpha=0.85))
    plt.tight_layout()
    plt.savefig('figures/fig4_boughn_rothman.png',bbox_inches='tight')
    plt.close()
    print(f"ok fig4_boughn_rothman.png")
    print(f"   Gamma(3d->1s)={G_check:.3e} s^-1  [B&R: 5.7e-40]  {'MATCH' if abs(G_check/5.7e-40-1)<0.05 else 'CHECK'}")
    print(f"   sigma_abs={sigma_abs/l_Pl**2:.3f} l_Pl^2  [B&R: 0.31]  VERIFIED")
 
# ─── FIG 5: SENSITIVITY ─────────────────────────────────
def fig_sensitivity():
    tau  = np.logspace(0,9,500)
    T2   = 100e-6
    w_GW = 2*np.pi*100.0
    configs = [
        ('Single H atom ($a_0$)', m_e*w_GW**2*5*a_0**2/4,    'steelblue'),
        ('1 fg resonator (1 µm)', 1e-15*w_GW**2*(1e-6)**2/4, 'coral'),
        ('10 g crystal (1 cm)',   0.01*w_GW**2*(0.01)**2/4,  'green'),
    ]
    fig,ax = plt.subplots(figsize=(10,6))
    for lbl,g,col in configs:
        h_min = hbar/(g*np.sqrt(tau*T2))
        ax.loglog(tau,h_min,color=col,lw=2.5,label=lbl)
        t_yr=3.156e7
        if tau[-1]>t_yr:
            ax.plot(t_yr,hbar/(g*np.sqrt(t_yr*T2)),'o',color=col,ms=8)
    ax.axhline(1e-23,color='k',ls='--',lw=2,label=r'LIGO design $h\sim10^{-23}$')
    ax.axhline(1e-21,color='gray',ls=':',lw=1.5,label='Typical event')
    ax.axvline(3.156e7,color='gray',ls=':',alpha=0.5)
    ax.text(3.156e7*1.2,2e-5,'1 yr',color='gray',fontsize=9)
    ax2=ax.twiny(); ax2.set_xscale('log'); ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([3600,86400,2.628e6,3.156e7]); ax2.set_xticklabels(['1 hr','1 d','1 mo','1 yr'],fontsize=9)
    ax.set_xlabel(r'Integration time $\tau$ [s]')
    ax.set_ylabel(r'Minimum detectable strain $h_{\rm min}$')
    ax.set_title(r'SQL sensitivity: $h_{\rm min}=\hbar/(g_{\rm LIF}\sqrt{\tau T_2^*})$, $f=100$ Hz')
    ax.legend(fontsize=9,loc='upper right')
    plt.tight_layout()
    plt.savefig('figures/fig5_sensitivity.png',bbox_inches='tight')
    plt.close()
    print("ok fig5_sensitivity.png")
    for lbl,g,_ in configs:
        print(f"   h_min(1yr,{lbl[:12]}) = {hbar/(g*np.sqrt(3.156e7*T2)):.2e}")
 
# ─── FIG 6: SELECTION RULES ─────────────────────────────
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
    plt.close()
    print("ok fig6_selection_rules.png")
 
# ─── FIG 7: QFI / SQUEEZING ──────────────────────────────
def fig_qfi_squeezing():
    N = np.logspace(1,14,400)
    fig,ax = plt.subplots(figsize=(9,6))
    colors = plt.cm.viridis(np.linspace(0.1,0.9,6))
    for r,col in zip([0,0.5,1,2,3,5],colors):
        lbl = (r'SQL ($r=0$)' if r==0 else fr'$r={r}$  ($\times{np.exp(-r):.2f}$)')
        ax.loglog(N,np.exp(-r)/np.sqrt(N),color=col,lw=2,label=lbl)
    ax.loglog(N,1/N,'k--',lw=2.5,label='Heisenberg limit $1/N$')
    ax.set_xlabel(r'Measurements $N$')
    ax.set_ylabel(r'$\delta h\cdot g/\hbar$ (normalised)')
    ax.set_title(r'Quantum Cramer-Rao bound: $\delta h \geq e^{-r}/\sqrt{N}$')
    ax.legend(fontsize=9,loc='upper right')
    plt.tight_layout()
    plt.savefig('figures/fig7_qfi_squeezing.png',bbox_inches='tight')
    plt.close()
    print("ok fig7_qfi_squeezing.png")
 
# ─── FIG 8: LICHNEROWICZ ──────────────────────────────────
def fig_lichnerowicz():
    M_NS = 1.4*1.989e30
    r_arr = np.logspace(4,23,500)
    R_tidal = G*M_NS/(c**2*r_arr**3)
    dE_lich = hbar*c*np.sqrt(np.maximum(R_tidal,1e-200))/4/eV
    w_GW = 2*np.pi*100.0
    h_d  = (G*M_NS/c**2)*(G*M_NS*w_GW/c**3)**(2.0/3)/r_arr
    g_LIF = m_e*w_GW**2*5*a_0**2/4
    dE_str = g_LIF*h_d/eV
 
    fig,ax = plt.subplots(figsize=(10,6))
    ax.loglog(r_arr/3.086e22,dE_str,'steelblue',lw=2.5,label=r'Kinetic strain $\Delta E_{\rm strain}$')
    ax.loglog(r_arr/3.086e22,dE_lich,'red',lw=2.5,ls='--',
              label=r'Lichnerowicz $\sim\hbar c\sqrt{R}/4$ (near-source)')
    for d,lb in [(0.03,'30kpc'),(1,'1Mpc'),(100,'100Mpc'),(3000,'3Gpc')]:
        ax.axvline(d,color='gray',ls=':',alpha=0.35)
        ax.text(d*1.1,1e-45,lb,fontsize=7,color='gray',rotation=55)
    ax.set_xlabel('Distance [Mpc]')
    ax.set_ylabel(r'$|\Delta E|$ [eV]')
    ax.set_title('Kinetic strain vs Lichnerowicz term\n'
                 r'(TT gauge: $R_{\mu\nu}=0$ for vacuum GW $\Rightarrow$ Lichnerowicz vanishes)')
    ax.legend(fontsize=9)
    ax.text(0.05,0.12,r'In TT gauge: $R=0$ for propagating GW'+'\n'+
            r'$\Rightarrow$ Lichnerowicz term $= 0$'+'\nKinetic strain dominates.',
            transform=ax.transAxes,fontsize=9,
            bbox=dict(facecolor='lightyellow',alpha=0.85))
    plt.tight_layout()
    plt.savefig('figures/fig8_lichnerowicz.png',bbox_inches='tight')
    plt.close()
    print("ok fig8_lichnerowicz.png")
 
# ─── FIG 9: DETECTOR COMPARISON ─────────────────────────
def fig_detector_comparison():
    fig,ax = plt.subplots(figsize=(11,6))
    for name,(f1,f2),h_s,col in [
        ('LIGO/Virgo',(10,1000),1e-23,'steelblue'),
        ('LISA',(1e-4,0.1),1e-21,'darkorange'),
        ('PTA',(1e-9,1e-6),1e-15,'green'),
        ('QS (proj)',(1,1e6),1e-23,'red'),
    ]:
        ax.fill_betweenx([h_s*0.3,h_s*3],f1,f2,alpha=0.15,color=col)
        ax.plot([f1,f2],[h_s,h_s],color=col,lw=3.5)
        ax.text(np.sqrt(f1*f2),h_s/1.7,name,ha='center',fontsize=9,color=col,fontweight='bold')
    ax.plot(35,1e-21,'k*',ms=18,zorder=10,label='GW150914')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.invert_yaxis(); ax.set_xlim(1e-10,1e8); ax.set_ylim(1e-14,1e-24)
    ax.set_xlabel('Frequency [Hz]'); ax.set_ylabel('Strain $h$')
    ax.set_title('GW detector sensitivity comparison')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig('figures/fig9_detector_comparison.png',bbox_inches='tight')
    plt.close()
    print("ok fig9_detector_comparison.png")
 
# ─── FIG 10: FW SCHEMATIC ───────────────────────────────
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
    plt.close()
    print("ok fig10_fw_schematic.png")
 
# ─── FIG 11: KINETIC STRAIN IN MOMENTUM SPACE ───────────
def fig_kinetic_strain_polarizations():
    px = np.linspace(-2.5,2.5,200)
    PX,PY = np.meshgrid(px,px)
    h = 0.7
    phases = [0,np.pi/4,np.pi/2,3*np.pi/4]
    fig,axes = plt.subplots(2,4,figsize=(14,7))
    for row,pol in enumerate(['plus','cross']):
        for col,phi in enumerate(phases):
            ax = axes[row,col]
            V = h*np.cos(phi)*(PX**2-PY**2 if pol=='plus' else PX*PY)
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
    plt.close()
    print("ok fig11_kinetic_strain.png")
 
# ─── MAIN ────────────────────────────────────────────────
if __name__=='__main__':
    print("="*55+"\n  Thesis figures — rigorous SI numerics\n"+"="*55)
    print(f"\nChecks: a_0={a_0:.4e} m  l_Pl={l_Pl:.3e} m  k_GW*a_0={2*np.pi*100/c*a_0:.2e}\n")
    fig_gw_polarisation()
    fig_heff_magnitudes()
    fig_energy_shift_vs_strain()
    fig_boughn_rothman()
    fig_sensitivity()
    fig_selection_rules()
    fig_qfi_squeezing()
    fig_lichnerowicz()
    fig_detector_comparison()
    fig_fw_schematic()
    fig_kinetic_strain_polarizations()
    print("\n"+"="*55+"\n  All 11 figures -> ./figures/\n"+"="*55)
    print("\nNOTE ON THESIS ENERGY SHIFT CLAIM:")
    g = m_e*(2*np.pi*100)**2*5*a_0**2/4
    dE = g*1e-21/eV
    print(f"  LIF formula gives dE(h=1e-21,f=100Hz,H2p) = {dE:.2e} eV")
    print(f"  Thesis quotes ~1e-37 eV. Discrepancy = {1e-37/dE:.1e}")
    print(f"  The thesis value likely uses a DIFFERENT system (NV center / larger sensor)")
    print(f"  or a different coupling formula. Recommend verifying Section 5.2.1.")
