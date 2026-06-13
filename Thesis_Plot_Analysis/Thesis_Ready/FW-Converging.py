import numpy as np
import matplotlib.pyplot as plt
import os
os.makedirs("figures", exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.size': 12, 'figure.dpi': 150})

m = 1.0
p_over_m = np.linspace(0, 2.0, 800)
p = p_over_m * m
E_exact = np.sqrt(m**2 + p**2) - m

colors = ['#6BAED6', '#2171B5', '#08519C', '#08306B']
labels = [r'$\mathcal{O}(p^2/m)$', r'$\mathcal{O}(p^4/m^3)$',
          r'$\mathcal{O}(p^6/m^5)$', r'$\mathcal{O}(p^8/m^7)$']
coefficients = [(1/(2*m), 2), (-1/(8*m**3), 4),
                (1/(16*m**5), 6), (-5/(128*m**7), 8)]
R = 1.0   # radius of convergence: p/m = 1  (sqrt(1+x) converges for |x|=(p/m)^2<1)

def shade_regions(ax):
    ax.axvspan(0, R, color='seagreen', alpha=0.06, zorder=0)
    ax.axvspan(R, 2, color='firebrick', alpha=0.06, zorder=0)
    ax.axvline(R, color='firebrick', lw=1.6, ls='-', alpha=0.9, zorder=4)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

# ── LEFT: energy ─────────────────────────────────────────────────────────────
ax = axes[0]
shade_regions(ax)
ax.plot(p_over_m, E_exact, 'k-', lw=2.6, label=r'Exact $\sqrt{m^2+p^2}-m$', zorder=10)
E = np.zeros_like(p)
for (c, pw), col, lab in zip(coefficients, colors, labels):
    E = E + c * p**pw
    ax.plot(p_over_m, E, color=col, lw=1.8, ls='--', alpha=0.9,
            label=f'through {lab}')
ax.set_xlabel(r'$p/m$'); ax.set_ylabel(r'$E/m$')
ax.set_title('FW free-particle energy vs exact dispersion', fontsize=10.5)
ax.set_xlim(0, 2); ax.set_ylim(-0.6, 1.55)
ax.text(R + 0.03, -0.5, r'$p=mc$' + '\n(radius of conv.)', color='firebrick',
        fontsize=8.5, va='bottom')
ax.text(0.5, 1.42, 'series converges', color='seagreen', ha='center', fontsize=9)
ax.text(1.5, 1.42, 'series diverges', color='firebrick', ha='center', fontsize=9)
ax.axvline(1/137, color='green', lw=0.8, ls=':', alpha=0.7)
ax.text(1/137 + 0.03, 0.55, r'$p/m=\alpha$' + '\n(H atom)', fontsize=7.5, color='green')
ax.legend(fontsize=8.5, loc='lower left'); ax.grid(alpha=0.25)

# ── RIGHT: relative error ────────────────────────────────────────────────────
ax = axes[1]
shade_regions(ax)
E = np.zeros_like(p)
for (c, pw), col, lab in zip(coefficients, colors, labels):
    E = E + c * p**pw
    rel = np.abs(E - E_exact) / (E_exact + 1e-15)
    ax.semilogy(p_over_m, rel, color=col, lw=1.8, ls='--', alpha=0.9,
                label=f'through {lab}')
ax.set_xlabel(r'$p/m$')
ax.set_ylabel(r'relative error $|E_{\rm FW}-E_{\rm exact}|/E_{\rm exact}$', fontsize=10)
ax.set_title('Convergence: higher orders help below $p/m=1$, hurt above', fontsize=10.5)
ax.set_xlim(0, 2); ax.set_ylim(1e-14, 1e1)
ax.axvline(1/137, color='green', lw=0.8, ls=':', alpha=0.7)
ax.text(1/137 + 0.012, 3e-3, r'$\alpha$', fontsize=9, color='green')
ax.axvline(0.1, color='darkorange', lw=0.8, ls=':', alpha=0.8)
ax.text(0.115, 3e-3, r'$v/c=0.1$', fontsize=8, color='darkorange')
ax.text(R + 0.03, 2e-13, r'$p=mc$', color='firebrick', fontsize=8.5, rotation=90, va='bottom')
ax.annotate('curves cross:\nadding orders\nincreases error', xy=(1.35, 1e-1),
            fontsize=8, color='firebrick', ha='center')
ax.legend(fontsize=8.5, loc='lower right'); ax.grid(which='both', alpha=0.25)

fig.suptitle(r'FW $1/m$ expansion converges only for $p/m<1$ — the radius of $\sqrt{1+(p/m)^2}$',
             y=0.99, fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
out = "figures/fw_expansion_convergence.png"
plt.savefig(out, bbox_inches="tight", dpi=300)
plt.show()
print("Saved:", out)