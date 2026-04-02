#  NV Center Energy Level Diagram with GW Perturbation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, axes = plt.subplots(1, 2, figsize=(12, 8))

# ─── NV center zero-field parameters (GHz) ───────────────────
D_gs  = 2.87    # ground state ZFS (GHz)
E_gs  = 0.003   # E-strain splitting (GHz, typical)
D_es  = 1.42    # excited state ZFS (GHz)
ZPL   = 637e-9  # zero-phonon line wavelength (m) → not plotted in GHz

# ─── GW-induced shift estimate ─────────────────────────────────
# Delta_D / D ~ kappa * h * <p^2> / (2m D) ~ 10^-30 for h~10^-21
# We plot a hypothetically amplified shift for visibility
delta_D_amplified = 0.015   # GHz (amplified for illustration)

def draw_level(ax, y, x1, x2, color, lw=2.0, label=None, ls='-'):
    ax.plot([x1, x2], [y, y], color=color, lw=lw,
            label=label, solid_capstyle='round', ls=ls)
def draw_arrow(ax, x, y_start, y_end, color, lw=1.2,
               label=None, ls='-'):
    ax.annotate("",
                xy=(x, y_end), xytext=(x, y_start),
                arrowprops=dict(arrowstyle="->",
                                color=color, lw=lw, ls=ls))

# ─── LEFT panel: unperturbed ──────────────────────────────────
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(-0.5, 16)
ax.axis('off')
ax.set_title("Unperturbed NV$^-$ center", fontsize=11, pad=8)

# Ground state triplet (E = 0 GHz baseline)
y_gs_0  = 3.0
y_gs_pm = y_gs_0 + D_gs

draw_level(ax, y_gs_0,  1.5, 4.5, '#2166AC', lw=2.5,
           label=r'$|m_s=0\rangle$')
draw_level(ax, y_gs_pm, 1.5, 4.5, '#D73027', lw=2.5,
           label=r'$|m_s=\pm1\rangle$')

ax.annotate('', xy=(3.8, y_gs_pm), xytext=(3.8, y_gs_0),
            arrowprops=dict(arrowstyle='<->', color='#D73027', lw=1.2))
ax.text(4.0, (y_gs_0 + y_gs_pm)/2,
        f'$D_{{gs}} = {D_gs}$ GHz', va='center', fontsize=9,
        color='#D73027')

ax.text(0.8, y_gs_0,  r'$|m_s=0\rangle$',  va='center',
        fontsize=9, color='#2166AC')
ax.text(0.8, y_gs_pm, r'$|m_s=\pm1\rangle$', va='center',
        fontsize=9, color='#D73027')
ax.text(5.0, y_gs_0 - 0.3,
        'Ground state $^3A_2$', fontsize=8, color='gray')

# Excited state triplet
y_es_0  = 11.0
y_es_pm = y_es_0 + D_es

draw_level(ax, y_es_0,  1.5, 4.5, '#2166AC', lw=2.5, ls='--')
draw_level(ax, y_es_pm, 1.5, 4.5, '#D73027', lw=2.5, ls='--')

ax.text(5.0, y_es_0 - 0.3,
        'Excited state $^3E$', fontsize=8, color='gray')
ax.text(0.8, y_es_0,  r'$|m_s=0\rangle_e$',  va='center',
        fontsize=9, color='#2166AC')
ax.text(0.8, y_es_pm, r'$|m_s=\pm1\rangle_e$', va='center',
        fontsize=9, color='#D73027')

# ZPL arrow
draw_arrow(ax, 5.5, y_gs_0 + 0.1, y_es_0 - 0.1, '#5AAE61', lw=1.5)
ax.text(5.7, (y_gs_0 + y_es_0)/2,
        '637 nm\n(ZPL)', fontsize=8, va='center', color='#5AAE61')

# MW arrows for spin-flip transitions
for y_target in [y_gs_pm]:
    draw_arrow(ax, 2.5, y_gs_0 + 0.05, y_target - 0.05,
               'purple', lw=1.2, ls='--')
ax.text(1.2, y_gs_0 + D_gs/2,
        'MW\n2.87 GHz', fontsize=8, va='center', color='purple')

# ─── RIGHT panel: GW-perturbed ────────────────────────────────
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(-0.5, 16)
ax.axis('off')
ax.set_title("GW-perturbed NV$^-$ center\n(shift amplified for clarity)",
             fontsize=11, pad=8)

y_gs_0_p  = y_gs_0
y_gs_pm_p = y_gs_pm + delta_D_amplified

draw_level(ax, y_gs_0_p,  1.5, 4.5, '#2166AC', lw=2.5)
draw_level(ax, y_gs_pm_p, 1.5, 4.5, '#D73027', lw=2.5)

# Show shift
ax.annotate('', xy=(3.8, y_gs_pm_p), xytext=(3.8, y_gs_pm),
            arrowprops=dict(arrowstyle='->', color='#FC8D59',
                            lw=1.5))
ax.text(4.05, (y_gs_pm + y_gs_pm_p)/2,
        r'$\Delta D = \frac{\kappa}{2m}\langle h_{ij}p^ip^j\rangle$',
        va='center', fontsize=8.5, color='#FC8D59')

ax.annotate('', xy=(3.8, y_gs_pm_p), xytext=(3.8, y_gs_0_p),
            arrowprops=dict(arrowstyle='<->', color='#D73027', lw=1.2))
ax.text(0.2, (y_gs_0_p + y_gs_pm_p)/2,
        f'$D_{{gs}} + \\Delta D$', va='center', fontsize=9,
        color='#D73027')

ax.text(0.8, y_gs_0_p,  r'$|m_s=0\rangle$',  va='center',
        fontsize=9, color='#2166AC')
ax.text(0.8, y_gs_pm_p, r'$|m_s=\pm1\rangle$', va='center',
        fontsize=9, color='#D73027')
ax.text(5.0, y_gs_0_p - 0.3,
        'Ground state $^3A_2$', fontsize=8, color='gray')

# GW symbol
ax.text(7.5, 6.0, '🌊', fontsize=22, ha='center')
ax.text(7.5, 5.0, 'GW', fontsize=9, ha='center', color='gray')

# Spin-selective shift annotation
ax.annotate('Kinetic strain shifts\n$\\Delta m_s = 0$ levels equally',
            xy=(3.0, y_gs_0_p), xytext=(5.5, 2.5),
            fontsize=8, color='#2166AC',
            arrowprops=dict(arrowstyle='->', color='#2166AC',
                            lw=0.8, alpha=0.6))
ax.annotate('Spin-rotation lifts\n$m_s = \\pm1$ degeneracy',
            xy=(3.0, y_gs_pm_p), xytext=(5.5, 4.5),
            fontsize=8, color='#D73027',
            arrowprops=dict(arrowstyle='->', color='#D73027',
                            lw=0.8, alpha=0.6))

fig.suptitle("NV center energy level structure and gravitational wave coupling\n"
             r"$\hat{H}_{\rm eff} = \hat{H}_{\rm NV} + "
             r"\frac{\kappa}{2m}h_{ij}\hat{p}^i\hat{p}^j + "
             r"\frac{\kappa}{4m}\vec{\Sigma}\cdot\hat{\vec{B}}_g + \ldots$",
             fontsize=11, y=1.02)

plt.tight_layout()
plt.savefig("figures/nv_energy_levels_gw.png",
            bbox_inches="tight", dpi=300)
plt.show()
print("Saved: nv_energy_levels_gw.pdf")