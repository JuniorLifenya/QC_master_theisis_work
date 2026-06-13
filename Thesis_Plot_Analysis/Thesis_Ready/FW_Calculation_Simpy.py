import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.linalg import expm
import os
os.makedirs("figures", exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.size': 12, 'figure.dpi': 150})

# ── genuine iterative FW on the free-particle Dirac matrix (p along z) ────────
I2 = np.eye(2); sz = np.array([[1, 0], [0, -1]], complex); Zr = np.zeros((2, 2))
beta = np.block([[I2, Zr], [Zr, -I2]]).astype(complex)
alpha_z = np.block([[Zr, sz], [sz, Zr]])
def odd(H): return 0.5 * (H - beta @ H @ beta)
def fw_step(H, m=1.0):
    S = -1j * beta @ odd(H) / (2 * m); U = expm(1j * S); return U @ H @ U.conj().T

PM = 0.5                                  # p/m  (mildly relativistic, numbers visible)
H = alpha_z * PM + beta                   # m = 1
snaps = []
for _ in range(3):
    snaps.append(H.real.copy()); H = fw_step(H)
odd_norms = [np.linalg.norm(odd(S.astype(complex))) for S in snaps]

# ── plot: their schematic style, but every number is COMPUTED ────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
def dm(ax, data, title):
    norm = TwoSlopeNorm(vmin=-1.2, vcenter=0, vmax=1.2)
    ax.imshow(data, cmap='RdBu', norm=norm, aspect='equal')
    for i in range(4):
        for j in range(4):
            v = data[i, j]
            if abs(v) > 0.02:
                txt = f'{v:.2f}' if abs(v) >= 0.1 else f'{v:.3f}'
                ax.text(j, i, txt, ha='center', va='center', fontsize=11,
                        color='white' if abs(v) > 0.6 else 'black')
    ax.set_xticks([]); ax.set_yticks([])
    for k in range(5):
        ax.axhline(k - 0.5, color='gray', lw=0.4); ax.axvline(k - 0.5, color='gray', lw=0.4)
    ax.axhline(1.5, color='k', lw=2); ax.axvline(1.5, color='k', lw=2)
    ax.set_title(title, fontsize=10, pad=6)

dm(axes[0], snaps[0], 'Initial Dirac $H$  ($p/m=0.5$)\n'
   r'odd block $\|\mathcal{O}\|=%.2f$' % odd_norms[0])
dm(axes[1], snaps[1], r'After $e^{i\hat S_1}H e^{-i\hat S_1}$' + '\n'
   r'odd block $\|\mathcal{O}\|=%.3f$' % odd_norms[1])
dm(axes[2], snaps[2], r'After $\hat S_2$:  $H_{\rm eff}$' + '\n'
   r'odd block $\|\mathcal{O}\|=%.3f$ (block-diag.)' % odd_norms[2])

for x, t in [(0.335, r'$U_1$'), (0.662, r'$U_2$')]:
    fig.text(x, 0.50, r'$\longrightarrow$', ha='center', va='center', fontsize=20)
    fig.text(x, 0.585, t, ha='center', va='center', fontsize=10)

fig.suptitle('FW iterative block diagonalisation — genuine computed entries '
             r'(diagonal $\to \pm\sqrt{m^2+p^2}=\pm1.118$;  blue $+$, red $-$)',
             y=0.95, fontsize=9)
plt.tight_layout()
out = 'figures/fig10_fw_schematic.png'
plt.savefig(out, bbox_inches='tight', dpi=300)
plt.show()
print("Saved:", out)
print("odd-block norms:", [f'{x:.4f}' for x in odd_norms])