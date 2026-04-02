#  Selection Rules: Angular Momentum Matrix Elements
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ─── Hydrogen p-orbital quadrupole matrix elements ───────────
# For H_strain ~ h_+ (px^2 - py^2), non-zero elements:
# <l', m'| T^(2)_q |l, m> with q = ±2, Δl = 0, ±2, Δm = ±2

# We plot the allowed (nonzero) transition structure symbolically.
l_vals   = [0, 1, 2, 3]
m_ranges = {0: [0], 1: [-1, 0, 1], 2: [-2,-1,0,1,2], 3:[-3,-2,-1,0,1,2,3]}

# Build list of all states (l, m)
states = []
for l in l_vals:
    for m in m_ranges[l]:
        states.append((l, m))

n_states = len(states)

# Build coupling matrix: 1 if GW can couple (plus polarization, Δm = ±2)
# plus polarization: q = +2 or q = -2; cross: same
coupling = np.zeros((n_states, n_states))

for i, (l1, m1) in enumerate(states):
    for j, (l2, m2) in enumerate(states):
        delta_m = m2 - m1
        delta_l = abs(l2 - l1)
        parity_ok = (l1 + l2) % 2 == 0   # even parity for T^(2)
        if abs(delta_m) == 2 and delta_l in [0, 2] and parity_ok:
            coupling[i, j] = 1.0

# ─── plot ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 10))

im = ax.imshow(coupling, cmap='Blues', aspect='equal',
               norm=Normalize(0, 1.2))

# Axis labels
tick_labels = [f'({l},{m})' for l, m in states]
ax.set_xticks(range(n_states))
ax.set_yticks(range(n_states))
ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
ax.set_yticklabels(tick_labels, fontsize=7)

# Colour separators between l-blocks
block_edges = [-0.5]
for l in l_vals:
    block_edges.append(block_edges[-1] + len(m_ranges[l]))

for edge in block_edges:
    ax.axhline(edge, color='navy', lw=0.8, alpha=0.7)
    ax.axvline(edge, color='navy', lw=0.8, alpha=0.7)

# l-block labels
x_pos = 0.5
for l in l_vals:
    n = len(m_ranges[l])
    mid = block_edges[l] + n/2
    ax.text(-1.8, mid, f'$l={l}$', ha='center', va='center',
            fontsize=10, color='navy', fontweight='bold')
    ax.text(mid, -2.2, f'$l={l}$', ha='center', va='center',
            fontsize=10, color='navy', fontweight='bold')

ax.set_xlabel("Initial state $(l, m)$", fontsize=11, labelpad=30)
ax.set_ylabel("Final state $(l\\!\\prime, m\\!\\prime)$",
              fontsize=11, labelpad=30)
ax.set_title(
    "GW-induced quadrupole coupling matrix\n"
    r"$h_+$ polarization: non-zero elements obey "
    r"$\Delta m = \pm 2$, $\Delta l = 0, \pm 2$",
    fontsize=11)

cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.03)
cbar.set_label("Coupling (0 = forbidden, 1 = allowed)", fontsize=9)

count_allowed = int(coupling.sum())
ax.text(0.02, 0.01,
        f"Allowed transitions: {count_allowed} / {n_states**2} "
        f"({100*count_allowed/n_states**2:.1f}%)",
        transform=ax.transAxes, fontsize=9, color='navy',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

plt.tight_layout()
plt.savefig("figures/selection_rules_matrix.png",
            bbox_inches="tight", dpi=300)
plt.show()
print("Saved: selection_rules_matrix.pdf")