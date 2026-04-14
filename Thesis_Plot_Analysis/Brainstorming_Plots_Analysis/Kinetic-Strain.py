import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure the output directory exists
os.makedirs('figures', exist_ok=True)

# ─── FIG 11: KINETIC STRAIN VS ISOTROPIC MOMENTUM SPACE ───────────

px = np.linspace(-3.5, 3.5, 200)
PX, PY = np.meshgrid(px, px)

h = 0.5
phases = [0, np.pi/4, np.pi/2, 3*np.pi/4]

# We create 5 columns: 1 for the reference, 4 for the time phases
fig, axes = plt.subplots(2, 5, figsize=(22, 9))

for row, pol in enumerate(['plus', 'cross']):
    # 1. Plot the Standard Isotropic Reference (Column 0)
    ax_ref = axes[row, 0]
    V_iso = (PX**2 + PY**2)
    ax_ref.contourf(PX, PY, V_iso, levels=20, cmap='viridis')
    ax_ref.set_aspect('equal')
    ax_ref.set_title("Standard Isotropic", fontweight='bold')
    ax_ref.set_ylabel(r'$p_y/p_0$' if row == 0 else r'$p_y/p_0$')
    if row == 1: ax_ref.set_xlabel(r'$p_x/p_0$')

    # 2. Plot the Kinetic Strain Phases (Columns 1 to 4)
    for col, phi in enumerate(phases):
        # We use col + 1 to leave the first column for the reference
        ax = axes[row, col + 1]
        
        # Handle the 1.57 (pi/2) null-point by checking if cos is near zero
        phi_eff = phi + 1e-9 if np.isclose(np.cos(phi), 0) else phi
        
        # Calculate the gravito-magnetic strain potential
        V = h * np.cos(phi_eff) * (PX**2 - PY**2 if pol == 'plus' else PX * PY)
        
        # Symmetric color scale centering on zero
        lim = max(abs(V).max(), 1e-9)
        cf = ax.contourf(PX, PY, V, levels=20, cmap='RdBu_r', vmin=-lim, vmax=lim)
        ax.contour(PX, PY, V, levels=[0], colors='k', linewidths=0.8)
        
        ax.set_aspect('equal')
        ax.set_title(fr'${"h_+" if pol=="plus" else "h_\\times"}$, $\omega t={phi:.2f}$')
        if row == 1: ax.set_xlabel(r'$p_x/p_0$')

fig.suptitle(r'Kinetic strain $h_{ij}\hat{p}^i\hat{p}^j$ in Momentum Space' + \
             '\n' + r'(Comparison: Isotropic Background vs. Gravito-Magnetic Perturbations)', 
             fontsize=14, y=1)

plt.tight_layout()
plt.savefig('figures/fig11_kinetic_strain.png', bbox_inches='tight')
plt.show()

print("File saved: figures/fig11_kinetic_strain.png")

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import matplotlib.gridspec as gridspec
# from matplotlib.colors import TwoSlopeNorm
# import os

# os.makedirs('figures', exist_ok=True)

# plt.rcParams.update({
#     'font.family': 'serif',
#     'font.size': 11,
#     'axes.labelsize': 12,
#     'axes.titlesize': 11,
#     'figure.dpi': 150,
#     'text.usetex': False,
# })

# # ── parameters ───────────────────────────────────────────────────────────────
# h_amp   = 0.45          # exaggerated for visibility (noted in caption)
# N       = 350
# px      = np.linspace(-3.2, 3.2, N)
# PX, PY  = np.meshgrid(px, px)

# phases       = [0, np.pi/4, np.pi/2, 3*np.pi/4]
# phase_labels = [r'$\omega t = 0$', r'$\omega t = \pi/4$',
#                 r'$\omega t = \pi/2$', r'$\omega t = 3\pi/4$']

# # Unperturbed KE: circles
# KE_free  = PX**2 + PY**2

# # Iso-energy levels (same everywhere for direct comparison)
# ke_levels = [0.5, 1.5, 3.0, 5.5, 9.0]

# # ── figure layout ─────────────────────────────────────────────────────────────
# # 2 rows (h+ / hx), 6 cols (reference | 4 phases | perturbation-only map)
# fig = plt.figure(figsize=(24, 10))
# gs  = gridspec.GridSpec(2, 6, figure=fig, wspace=0.12, hspace=0.38,
#                         left=0.04, right=0.98, top=0.87, bottom=0.09)


# def draw_stretch_arrows(ax, h_val, pol):
#     """Draw <-> arrows on the stretch and squeeze axes."""
#     if abs(h_val) < 0.05:
#         return
#     stretch_c = '#e8a020'
#     squeeze_c = '#5090e0'
#     r_stretch = 2.9
#     r_squeeze = 2.2

#     if pol == 'plus':
#         if h_val > 0:
#             # px axis stretched, py axis squeezed
#             ax.annotate('', xy=(r_stretch, 0), xytext=(-r_stretch, 0),
#                 arrowprops=dict(arrowstyle='<->', color=stretch_c, lw=2.0))
#             ax.annotate('', xy=(0,  r_squeeze), xytext=(0, -r_squeeze),
#                 arrowprops=dict(arrowstyle='<->', color=squeeze_c, lw=2.0))
#         else:
#             # flipped: py stretched, px squeezed
#             ax.annotate('', xy=(r_squeeze, 0), xytext=(-r_squeeze, 0),
#                 arrowprops=dict(arrowstyle='<->', color=squeeze_c, lw=2.0))
#             ax.annotate('', xy=(0,  r_stretch), xytext=(0, -r_stretch),
#                 arrowprops=dict(arrowstyle='<->', color=stretch_c, lw=2.0))
#     else:
#         # h_cross: stretch along diagonals
#         d = r_stretch * 0.707
#         ds = r_squeeze * 0.707
#         if h_val > 0:
#             ax.annotate('', xy=( d,  d), xytext=(-d, -d),
#                 arrowprops=dict(arrowstyle='<->', color=stretch_c, lw=2.0))
#             ax.annotate('', xy=( ds, -ds), xytext=(-ds, ds),
#                 arrowprops=dict(arrowstyle='<->', color=squeeze_c, lw=2.0))
#         else:
#             ax.annotate('', xy=( d,  d), xytext=(-d, -d),
#                 arrowprops=dict(arrowstyle='<->', color=squeeze_c, lw=2.0))
#             ax.annotate('', xy=( ds, -ds), xytext=(-ds, ds),
#                 arrowprops=dict(arrowstyle='<->', color=stretch_c, lw=2.0))


# for row, pol in enumerate(['plus', 'cross']):
#     pol_sym = r'$h_+$' if pol == 'plus' else r'$h_\times$'

#     # ── Col 0: free KE reference ──────────────────────────────────────────────
#     ax0 = fig.add_subplot(gs[row, 0])
#     ax0.contourf(PX, PY, KE_free, levels=30, cmap='Blues', alpha=0.65, vmin=0, vmax=12)
#     cs0 = ax0.contour(PX, PY, KE_free, levels=ke_levels,
#                       colors='#1a3a6a', linewidths=1.8)
#     ax0.clabel(cs0, fmt='%.1f', fontsize=7.5, inline=True)
#     ax0.set_aspect('equal')
#     ax0.set_xlim(-3.2, 3.2); ax0.set_ylim(-3.2, 3.2)
#     ax0.set_title('Free KE  (reference)\n' + r'$p_x^2 + p_y^2$',
#                   fontweight='bold', color='#1a3a6a')
#     ax0.set_ylabel(r'$p_y / p_0$')
#     if row == 1:
#         ax0.set_xlabel(r'$p_x / p_0$')
#     ax0.text(0.97, 0.03, 'perfect circles', transform=ax0.transAxes,
#              ha='right', fontsize=8.5, color='#1a3a6a', style='italic')

#     # ── Cols 1-4: deformed total KE at each phase ─────────────────────────────
#     for col, (phi, plabel) in enumerate(zip(phases, phase_labels)):
#         ax = fig.add_subplot(gs[row, col + 1])

#         h_now = h_amp * np.cos(phi + 1e-19)   # instantaneous value

#         # Total KE with gravitational strain perturbation
#         if pol == 'plus':
#             KE_tot = PX**2 * (1 + h_now) + PY**2 * (1 - h_now)
#         else:
#             KE_tot = PX**2 + PY**2 + 2.0 * h_now * PX * PY

#         # Signed deviation from free (background color)
#         dKE  = KE_tot - KE_free
#         lim  = max(abs(dKE).max(), 0.05)
#         norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
#         ax.contourf(PX, PY, dKE, levels=40, cmap='RdBu_r', norm=norm, alpha=0.72)

#         # Dashed grey reference circles
#         ax.contour(PX, PY, KE_free, levels=ke_levels,
#                    colors='gray', linewidths=0.8, linestyles='--', alpha=0.55)

#         # Solid deformed iso-energy contours
#         ax.contour(PX, PY, KE_tot, levels=ke_levels,
#                    colors='#111111', linewidths=1.7)

#         # Stretch/squeeze arrows at non-null phases
#         if abs(h_now) > 0.06:
#             draw_stretch_arrows(ax, h_now, pol)

#         ax.set_aspect('equal')
#         ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2)

#         h_label = r'$h(t)=' + f'{h_now:+.2f}' + r'h_0$'
#         ax.set_title(f'{pol_sym},  {plabel}\n{h_label}', fontsize=10)

#         if row == 1:
#             ax.set_xlabel(r'$p_x / p_0$')
#         if col == 0:
#             ax.set_ylabel(r'$p_y / p_0$')

#         # Annotate stretch / squeeze at phi=0
#         if np.isclose(phi, 0) and abs(h_now) > 0.2:
#             if pol == 'plus':
#                 ax.text( 3.1, 0.0, 'stretch', ha='left',  va='center',
#                          fontsize=7.5, color='#e8a020', fontweight='bold', rotation=90)
#                 ax.text(-3.1, 0.0, 'stretch', ha='right', va='center',
#                          fontsize=7.5, color='#e8a020', fontweight='bold', rotation=90)
#                 ax.text(0.0,  3.1, 'squeeze', ha='center', va='bottom',
#                          fontsize=7.5, color='#5090e0', fontweight='bold')
#                 ax.text(0.0, -3.1, 'squeeze', ha='center', va='top',
#                          fontsize=7.5, color='#5090e0', fontweight='bold')
#             else:
#                 ax.text( 2.3,  2.3, 'stretch', ha='center', va='bottom',
#                          fontsize=7.5, color='#e8a020', fontweight='bold', rotation=45)
#                 ax.text( 2.3, -2.3, 'squeeze', ha='center', va='top',
#                          fontsize=7.5, color='#5090e0', fontweight='bold', rotation=-45)

#     # ── Col 5: pure perturbation (for geometric reference) ───────────────────
#     ax5 = fig.add_subplot(gs[row, 5])
#     if pol == 'plus':
#         V_pert = h_amp * (PX**2 - PY**2)
#         saddle_txt = r'$+(p_x^2 - p_y^2)$'
#     else:
#         V_pert = h_amp * PX * PY
#         saddle_txt = r'$+2\,p_x p_y$'

#     lim5  = abs(V_pert).max()
#     norm5 = TwoSlopeNorm(vmin=-lim5, vcenter=0, vmax=lim5)
#     ax5.contourf(PX, PY, V_pert, levels=40, cmap='RdBu_r', norm=norm5)
#     ax5.contour(PX, PY,  V_pert, levels=[0], colors='k', linewidths=1.5)
#     ax5.set_aspect('equal')
#     ax5.set_xlim(-3.2, 3.2); ax5.set_ylim(-3.2, 3.2)
#     ax5.set_title('Perturbation only\n' + r'$h_{ij}\hat{p}^i\hat{p}^j$  $(h_0{=}1)$',
#                   fontsize=10, fontweight='bold')
#     if row == 1:
#         ax5.set_xlabel(r'$p_x / p_0$')
#     ax5.text(0.97, 0.03, saddle_txt, transform=ax5.transAxes,
#              ha='right', fontsize=9, style='italic')

# # ── global legend ─────────────────────────────────────────────────────────────
# legend_items = [
#     mpatches.Patch(facecolor='#bbbbbb', edgecolor='#555555',
#                    linestyle='--', label='Free KE circles (reference)'),
#     mpatches.Patch(facecolor='#e8a020', alpha=0.85, label='Stretch axis'),
#     mpatches.Patch(facecolor='#5090e0', alpha=0.85, label='Squeeze axis'),
#     mpatches.Patch(facecolor='#c03030', alpha=0.6,  label=r'$\delta$KE $> 0$  (momentum gain)'),
#     mpatches.Patch(facecolor='#3060c0', alpha=0.6,  label=r'$\delta$KE $< 0$  (momentum loss)'),
#     mpatches.Patch(facecolor='#111111', alpha=0.8,  label='Deformed iso-energy contours'),
# ]
# fig.legend(handles=legend_items, loc='lower center', ncol=6,
#            bbox_to_anchor=(0.5, -0.01), fontsize=9.5, framealpha=0.92,
#            edgecolor='#aaaaaa')

# fig.suptitle(
#     r'Kinetic Strain:  $\hat{H}_{\rm strain} = \frac{\kappa}{2m}h_{ij}\hat{p}^i\hat{p}^j$'
#     '  — Deformation of momentum-space iso-energy surfaces\n'
#     r'Solid contours: total KE $(1\!\pm\! h)p_{x,y}^2$ · '
#     r'Dashed: free circles · '
#     r'Background: $\delta{\rm KE} = {\rm KE}_{\rm tot} - p^2$  '
#     r'[$h_0 = 0.45$, exaggerated for visibility]',
#     fontsize=12.5, y=0.98
# )

# plt.savefig('figures/fig11_kinetic_strain.png', dpi=200, bbox_inches='tight')
# plt.show()
# plt.close()
# print("Saved: figures/fig11_kinetic_strain.png")