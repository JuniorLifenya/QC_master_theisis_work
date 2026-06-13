# FW Expansion Convergence
import numpy as np
import matplotlib.pyplot as plt

# ─── FW expansion: corrections as power series in p/m ─────────
# H_eff = beta*m + p^2/(2m) - p^4/(8m^3) + p^6/(16m^5) - ...
# Compare exact E = sqrt(m^2 + p^2) - m  with partial sums

m = 1.0  # normalised mass
p_over_m = np.linspace(0, 2.0, 500)
p = p_over_m * m

E_exact = np.sqrt(m**2 + p**2) - m

orders = [1, 2, 3, 4]
colors = ['#6BAED6', '#2171B5', '#08519C', '#08306B']
labels = ['$\\mathcal{O}(p^2/m)$',
          '$\\mathcal{O}(p^4/m^3)$',
          '$\\mathcal{O}(p^6/m^5)$',
          '$\\mathcal{O}(p^8/m^7)$']

coefficients = [
    ( 1/(2*m),      2),
    (-1/(8*m**3),   4),
    ( 1/(16*m**5),  6),
    (-5/(128*m**7), 8),
]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: energy comparison
ax = axes[0]
ax.plot(p_over_m, E_exact, 'k-', lw=2.5, label='Exact $\\sqrt{m^2+p^2}-m$',
        zorder=10)

E_approx = np.zeros_like(p)
for n, ((coeff, power), col, lab) in enumerate(
        zip(coefficients, colors, labels)):
    E_approx += coeff * p**power
    ax.plot(p_over_m, E_approx, color=col, lw=1.8, ls='--',
            label=f'FW through {lab}', alpha=0.85)

ax.set_xlabel('$p/m$', fontsize=11)
ax.set_ylabel('$E/m$', fontsize=11)
ax.set_title('FW expansion of free-particle energy\n'
             'vs exact relativistic dispersion', fontsize=10)
ax.legend(fontsize=8.5, loc='upper left')
ax.set_xlim(0, 2); ax.set_ylim(-0.05, 1.2)
ax.axvline(alpha_fs := 1/137, color='green', lw=0.8, ls=':', alpha=0.7)
ax.text(1/137 + 0.02, 0.95, '$\\alpha$ (H-atom)', fontsize=7.5,
        color='green')
ax.grid(alpha=0.25)

# Right: relative error
ax = axes[1]
E_approx_all = np.zeros_like(p)
for n, ((coeff, power), col, lab) in enumerate(
        zip(coefficients, colors, labels)):
    E_approx_all += coeff * p**power
    rel_err = np.abs(E_approx_all - E_exact) / (E_exact + 1e-15)
    ax.semilogy(p_over_m, rel_err, color=col, lw=1.8, ls='--',
                label=f'Through {lab}', alpha=0.85)

ax.axvline(1/137, color='green', lw=0.8, ls=':', alpha=0.7)
ax.text(1/137+0.01, 5e-2, '$\\alpha$', fontsize=9, color='green')
ax.axvline(0.1, color='orange', lw=0.8, ls=':', alpha=0.7)
ax.text(0.11, 5e-2, '$v/c=0.1$', fontsize=8, color='orange')

ax.set_xlabel('$p/m$', fontsize=11)
ax.set_ylabel('Relative error $|E_{\\rm FW} - E_{\\rm exact}|/E_{\\rm exact}$',
              fontsize=10)
ax.set_title('Convergence of FW expansion\n(relative error vs $p/m$)',
             fontsize=10)
ax.legend(fontsize=8.5, loc='upper left')
ax.set_xlim(0, 2)
ax.set_ylim(1e-14, 1e1)
ax.grid(which='both', alpha=0.25)

plt.tight_layout()
plt.savefig("figures/fw_expansion_convergence.png",
            bbox_inches="tight", dpi=300)
plt.show()
print("Saved: fw_expansion_convergence.pdf")