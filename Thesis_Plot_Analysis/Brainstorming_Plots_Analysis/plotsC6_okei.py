import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

# Create figures directory if it doesn't exist
os.makedirs("figures", exist_ok=True)

# ─── Plus-polarized GW: h_+ cos(kz - omega*t) ────────────────
# B_g = curl(h . p) — in TT gauge with p ~ k_electron direction
# B_g^l = eps^{lki} (partial_k h^i_j) p^j

x   = np.linspace(-2, 2, 30)
y   = np.linspace(-2, 2, 30)
X, Y = np.meshgrid(x, y)

k_gw   = 1.0    # normalised
omega  = 1.0
t      = np.pi / (2 * omega)  # Shifted time so the sine term isn't 0!
h_plus = 0.5

# Metric perturbation (z=0 slice)
h_xx =  h_plus * np.cos(k_gw * 0 - omega * t)   # scalar at z=0
h_yy = -h_plus * np.cos(k_gw * 0 - omega * t)
h_xy =  0.0

# Spatial gradient along z (propagation direction):
# partial_z h_xx = -k_gw * h_plus * sin(kz - omega*t) at z=0 = 0
dh_xx_dz = -k_gw * h_plus * np.sin(-omega * t) * np.ones_like(X)
dh_yy_dz =  k_gw * h_plus * np.sin(-omega * t) * np.ones_like(X)

# B_g = curl(h . p): for a test electron with p = p_x x_hat
# B_g^y = eps^{yzx} (partial_z h^x_j) p^j
#       = (partial_z h_xx) * p_x   (for p = p_x hat_x)
p_x = 1.0  # normalised
Bg_y =  dh_xx_dz * p_x   # y-component
Bg_x = -dh_yy_dz * p_x   # x-component (from h_yy term)

# Magnitude
Bg_mag = np.sqrt(Bg_x**2 + Bg_y**2)

# ─── plot ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Left: h_ij deformation pattern
ax = axes[0]
theta_ring = np.linspace(0, 2*np.pi, 200)
r_def = 1.0 + h_plus * np.cos(2*theta_ring) * 0.3  # tidal deformation
ax.plot(r_def*np.cos(theta_ring), r_def*np.sin(theta_ring),
        color='#E31A1C', lw=2.0, label='Deformed circle')
ax.plot(np.cos(theta_ring), np.sin(theta_ring),
        color='gray', lw=1.0, ls='--', alpha=0.5, label='Reference circle')

# Draw h_ij tensor as arrows
ax.annotate('', xy=(1.5, 0), xytext=(-1.5, 0),
            arrowprops=dict(arrowstyle='<->', color='#2171B5', lw=2.0))
ax.annotate('', xy=(0, -1.5), xytext=(0, 1.5),
            arrowprops=dict(arrowstyle='<->', color='#FC4E2A', lw=2.0))

ax.text(1.6, 0.1, '$h_+$: stretch', color='#2171B5', fontsize=9)
ax.text(0.1, 1.6, '$h_+$: compress', color='#FC4E2A', fontsize=9)
ax.text(0.1, -1.6, '$(h_{xx} = -h_{yy})$', fontsize=8.5, color='gray')

# Centered limits for a better visual balance
ax.set_xlim(-2.5, 2.5) 
ax.set_ylim(-2.2, 2.2)
ax.set_aspect('equal')
ax.set_title('Plus-polarized GW: tidal deformation\n'
             r'$h_{ij} = h_+\,\mathrm{diag}(1,-1,0)\cos(\omega t - kz)$',
             fontsize=10)
ax.legend(fontsize=9, loc='upper left') # Moved legend slightly
ax.grid(alpha=0.2)

# Right: gravitomagnetic field B_g
ax = axes[1]
speed = np.hypot(Bg_x, Bg_y)
# Prevent division by zero if field is totally flat
lw    = 0.5 + 2.5 * speed / (speed.max() + 1e-9)

strm = ax.streamplot(X, Y, Bg_x, Bg_y,
                     color=speed,
                     cmap='plasma',
                     linewidth=1.5,
                     density=1.2,
                     arrowsize=1.2)
cbar = fig.colorbar(strm.lines, ax=ax, fraction=0.04, pad=0.03)
cbar.set_label(r'$|\hat{\vec{B}}_g|$ (normalised)', fontsize=9)

ax.set_xlabel('$x/a_0$', fontsize=10)
ax.set_ylabel('$y/a_0$', fontsize=10)
ax.set_title(
    r'Gravitomagnetic field $\hat{\vec{B}}_g = \nabla\times(\mathbf{h}\cdot\hat{\vec{p}})$'
    '\nfor $h_+$ polarization, $\hat{p} = p_x\hat{x}$',
    fontsize=10)
ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
ax.set_aspect('equal')

# Removed y=1.01 to keep title in bounds
plt.suptitle('Gravitomagnetic coupling structure in the TT gauge', fontsize=13)
plt.tight_layout()

# Adjust the top margin so tight_layout doesn't overwrite the suptitle
fig.subplots_adjust(top=0.85)

# Unified the extension to .pdf for both savefig and print
plt.savefig("figures/gravitomagnetic_field.png", bbox_inches="tight", dpi=300)
plt.show()
print("Saved: figures/gravitomagnetic_field.pdf")