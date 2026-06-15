import numpy as np
import matplotlib.pyplot as plt
import os
os.makedirs("figures", exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.size': 12, 'figure.dpi': 150})

m = 1.0
pm = np.linspace(0, 2.0, 800)
p = pm * m
E_exact = np.sqrt(m**2 + p**2) - m

colors = ['#6BAED6', "#00FFE1", "#08529CD9","#9900FF" ]
labels = [r'$\mathcal{O}(p^2/m)$', r'$\mathcal{O}(p^4/m^3)$',
          r'$\mathcal{O}(p^6/m^5)$', r'$\mathcal{O}(p^8/m^7)$']
coeffs = [(1/(2*m), 2), (-1/(8*m**3), 4), (1/(16*m**5), 6), (-5/(128*m**7), 8)]
R = 1.0   # radius of convergence

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# LEFT — energy
ax = axes[0]
ax.plot(pm, E_exact, 'k-', lw=2.5, label=r'Exact $\sqrt{m^2+p^2}-m$', zorder=10)
E = np.zeros_like(p)
for (c, pw), col, lab in zip(coeffs, colors, labels):
    E = E + c * p**pw
    ax.plot(pm, E, color=col, lw=2, ls='--', label=f'through {lab}')
ax.axvline(R, color='firebrick', lw=1.5, label=r'radius $p/m=1$')
ax.set_xlabel(r'$p/m$'); ax.set_ylabel(r'$E/m$')
ax.set_title('FW energy vs exact dispersion', fontsize=11)
ax.set_xlim(0, 2); ax.set_ylim(-0.4, 1.5)
ax.legend(fontsize=8.5, loc='upper left'); ax.grid(alpha=0.25)

# RIGHT — relative error
ax = axes[1]
E = np.zeros_like(p)
for (c, pw), col, lab in zip(coeffs, colors, labels):
    E = E + c * p**pw
    rel = np.abs(E - E_exact) / (E_exact + 1e-15)
    ax.semilogy(pm, rel, color=col, lw=1.8, ls='--', label=f'through {lab}')
ax.axvline(R, color='firebrick', lw=1.5, label=r'radius $p/m=1$')
ax.set_xlabel(r'$p/m$'); ax.set_ylabel(r'relative error')
ax.set_title('Convergence vs $p/m$', fontsize=11)
ax.set_xlim(0, 2); ax.set_ylim(1e-14, 1e1)
ax.legend(fontsize=8.5, loc='lower right'); ax.grid(which='both', alpha=0.25)

plt.tight_layout()
out = "figures/fw_expansion_convergence.png"
plt.savefig(out, bbox_inches="tight", dpi=300)
plt.show()
print("Saved:", out)