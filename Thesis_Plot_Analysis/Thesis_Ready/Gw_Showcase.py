import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import os

# Set up colormap gracefully
try:
    import seaborn as sns
    SURF_CMAP = sns.color_palette("mako", as_cmap=True)
except ImportError:
    SURF_CMAP = "turbo"

os.makedirs("Thesis_Ready_Plots", exist_ok=True)

# --- Wave Mechanics -----------------------------------------------------------
# We model a localized gravitational wave burst propagating radially outward.
# Z = A * sin(k*R - omega*t) * exp(-R/decay_distance)
# We will capture a snapshot in time (t=0).

A = 0.45          # Amplitude of the wave
k = 2.5           # Wavenumber (controls spatial frequency)
decay = 3.5       # How quickly the wave damps out as it travels

def wave_z(x, y):
    r = np.sqrt(x**2 + y**2)
    # The wave ripples outward, decaying in amplitude
    return A * np.sin(k * r) * np.exp(-r / decay)

# --- Plot Setup ---------------------------------------------------------------
fig = plt.figure(figsize=(12, 7.5))
fig.patch.set_facecolor("white")
ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

fig.suptitle("Spacetime responding to a Gravitational Wave", 
             fontsize=15, fontweight="bold", y=0.92)
ax.set_title("a flat spacetime manifold undergoing transverse oscillations", 
             fontsize=11, pad=5, color="dimgrey", y=0.90)

# --- Generate Grids -----------------------------------------------------------
N = 120
SPAN = 6.0
gx = np.linspace(-SPAN, SPAN, N)
X, Y = np.meshgrid(gx, gx)
Z = wave_z(X, Y)

# --- Plotting the Unperturbed Flat Sheet (Reference) --------------------------
# This fulfills the "flat sheet that is *now* oscillating" visual
Z_flat = np.zeros_like(Z)
ax.plot_surface(X, Y, Z_flat, color="gainsboro", alpha=0.15, linewidth=0,
                antialiased=True, zorder=0)

# --- Plotting the Gravitational Wave ------------------------------------------
# The surface mapping the wave
ax.plot_surface(X, Y, Z, cmap=SURF_CMAP, alpha=0.85, linewidth=0,
                antialiased=True, rstride=1, cstride=1, zorder=2)

# A wireframe overlay to emphasize the stretching and curvature
ax.plot_wireframe(X, Y, Z, color="darkgoldenrod", alpha=0.15, 
                  linewidth=0.5, rstride=3, cstride=3, zorder=3)

# Add a subtle contour projection on the "floor"
z_floor = -1.2
levels = np.linspace(-A, A, 11)
ax.contour(X, Y, Z, levels=levels, zdir="z", offset=z_floor,
           cmap="turbo_r", linewidths=0.8, alpha=0.4, zorder=1)

# --- Formatting and Camera ----------------------------------------------------
ax.view_init(elev=28, azim=-55)

ax.set_xlim(-SPAN, SPAN)
ax.set_ylim(-SPAN, SPAN)
ax.set_zlim(z_floor, A + 0.2)

# Constrain the aspect ratio to keep the physics visually proportional
ax.set_box_aspect((2*SPAN, 2*SPAN, (A + 0.2) - z_floor + 1.0))

ax.set_xlabel(r"$x$", fontsize=12, labelpad=8)
ax.set_ylabel(r"$y$", fontsize=12, labelpad=8)
ax.set_zticklabels([]) # Hide z-axis text to keep it clean
ax.grid(False)

# Clean up margins
fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.9)

# Save and Show
out = "Thesis_Ready_Plots/fig_gravitational_wave.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.show()
print("Saved:", out)