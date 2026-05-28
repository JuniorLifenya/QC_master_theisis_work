"""
Plot 4 — Cross-Section Scale Comparison (Logarithmic)
======================================================
Compares σ_GW = 0.31 ℓ_Pl² ≈ 10⁻⁷⁰ m² against:
  - Nuclear cross sections (~10⁻²⁶ m² barn scale)
  - Thomson cross section (~10⁻²⁹ m²)
  - Hydrogen Bohr radius squared (a₀² ~ 10⁻²⁰ m²)
  - Virus / cell / DNA scales for intuition
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import os

os.makedirs("/home/claude/thesis_plots", exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10.5,
    'figure.dpi': 150,
})

BG   = '#F9F7F3'
C_GW = '#5B1A8A'
CG   = '#DDDDDD'

# ── Data: (log10 of cross section in m², label, x-offset for label, color) ─
entries = [
    (-70,  r"$\sigma_{\rm GW} = 0.31\,\ell_{\rm Pl}^2$",      '#5B1A8A', True,  "GW absorption\n(graviton–electron)"),
    (-63,  r"$\ell_{\rm Pl}^2 = G\hbar/c^3$",                  '#8B4CB8', False, "Planck area"),
    (-47,  r"$\Delta E\sim10^{-47}\ \mathrm{eV}$ (2p split)",  '#AAAAAA', False, "Energy shift scale\n(not a cross section,\nshown for context)"),
    (-35,  r"Weak interaction ($W/Z$ exchange)",                '#C47C0A', False, "Weak force"),
    (-29,  r"$\sigma_{\rm Thomson} = 6.65\times10^{-29}\ \mathrm{m}^2$", '#1B3870', False, "Thomson (EM photon\n scattering)"),
    (-26,  r"Nuclear (barn $= 10^{-28}\ \mathrm{m}^2$)",       '#175A3A', False, "Nuclear cross section"),
    (-20,  r"$a_0^2 \approx 2.8\times10^{-21}\ \mathrm{m}^2$", '#C0392B', False, "Bohr radius² (H atom)"),
    (-15,  r"DNA double helix cross-section",                   '#888888', False, "DNA"),
    (-10,  r"LIGO mirror ($\sim\!0.03\ \mathrm{m}^2$)",        '#444444', False, "LIGO arm mirror"),
]

fig, ax = plt.subplots(figsize=(13, 7), facecolor=BG)
ax.set_facecolor(BG)
ax.spines[['top','right','left']].set_visible(False)
ax.spines['bottom'].set_color('#999')

# ── Draw horizontal comparison axis ──────────────────────────────────────
log_min, log_max = -75, -5
ax.set_xlim(log_min - 2, log_max + 2)
ax.set_ylim(-0.5, len(entries) + 1.5)
ax.set_yticks([])

ax.axhline(-0.2, xmin=0.03, xmax=0.97, color='#aaa', lw=1.2, zorder=0)

# Vertical span: a₀² to Thomson for "achievable quantum measurement"
ax.axvspan(-30, -19, alpha=0.06, color='#1B3870', zorder=0, label='Quantum sensing range')
ax.text(-24.5, len(entries) + 0.8, "current quantum\nsensing range",
        ha='center', fontsize=8, color='#1B3870', style='italic')

# ── GW marker — highlighted ───────────────────────────────────────────────
gw_log = -70
ax.axvline(gw_log, color=C_GW, lw=2.0, ls='--', alpha=0.7, zorder=1)

# ── Draw each entry ────────────────────────────────────────────────────────
for i, (log_val, label, color, highlight, sublabel) in enumerate(entries):
    y = i + 0.5
    marker_size = 220 if highlight else 120
    marker_style = '*' if highlight else 'o'

    ax.scatter(log_val, y, color=color, s=marker_size, marker=marker_style, 
               zorder=8, edgecolors='white', linewidths=0.8)
    
    # Tick on the axis
    ax.plot([log_val, log_val], [-0.4, -0.05], color=color, lw=1.2, alpha=0.5, zorder=2)

    # Label
    ax.text(log_val, y + 0.32, label, ha='center', va='bottom',
            fontsize=8.5 if not highlight else 9.5,
            color=color, fontweight='bold' if highlight else 'normal')
    ax.text(log_val, y - 0.32, sublabel, ha='center', va='top',
            fontsize=7.5, color='#555', style='italic')

# ── Orders-of-magnitude brackets ─────────────────────────────────────────
def draw_bracket(ax, x1, x2, y, label, color='#888'):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='<->', color=color, lw=1.2))
    ax.text((x1+x2)/2, y+0.15, label, ha='center', fontsize=8, color=color)

draw_bracket(ax, -70, -29, -0.5 + len(entries) + 1.1,
             r"$\mathbf{41}$ orders of magnitude below Thomson",
             color='#5B1A8A')

draw_bracket(ax, -70, -26, -0.5 + len(entries) + 0.45,
             r"$\mathbf{44}$ orders below nuclear",
             color='#8B3A62')

# ── x-axis ────────────────────────────────────────────────────────────────
ax.set_xlabel(r"$\log_{10}(\sigma\ /\ \mathrm{m}^2)$", fontsize=12)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
ax.tick_params(axis='x', labelsize=9, direction='out')
ax.tick_params(axis='x', which='minor', length=3)

# ── Scale bar: Planck length ───────────────────────────────────────────────
ax.annotate('',
    xy=(-70, -0.45), xytext=(-68, -0.45),
    arrowprops=dict(arrowstyle='<->', color=C_GW, lw=1.5))
ax.text(-69, -0.6, r"$\sim\!2\ell_{\rm Pl}^2$", ha='center', fontsize=8, color=C_GW)

# ── Title and caption ──────────────────────────────────────────────────────
ax.set_title(
    r"Cross-Section Scale Comparison: $\sigma_{\rm GW}$ vs Known Physics"
    "\n"
    r"$\sigma_{\rm GW}(1s\to3d) = 0.31\,\ell_{\rm Pl}^2 \approx 3.2\times10^{-70}\ \mathrm{m}^2$  "
    r"— all atomic constants cancel, leaving only the Planck area",
    fontsize=11, fontweight='bold', pad=14, color='#1a1a1a')

# detection gap box
ax.text(-47, 1.5,
    "Detection gap:\n"
    r"$\sim10^{41}\times$ below""\n""current technology",
    fontsize=9, color='#CC2222', fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.5', fc='#FEF0F0', ec='#CC2222', lw=1.3))

plt.tight_layout()
plt.savefig("/home/claude/thesis_plots/plot4_cross_section_scale.png",
            dpi=300, bbox_inches='tight', facecolor=BG)
print("Saved: plot4_cross_section_scale.png")
plt.close()