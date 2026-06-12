"""
Figure 1 — Spacetime responding to an outgoing gravitational wave.
Improved version:
  * Two distinct gravity wells added at the center so space bends DOWNWARDS.
  * Black holes are nestled inside their respective wells.
  * Radial Colormap: Red in the center (redshift), blue on the outside.
  * Sheet shaded with a LightSource for real depth.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import os

os.makedirs("Thesis_Ready_Plots", exist_ok=True)

# --- Physical parameters & Wave mechanics -------------------------------------
d_binary   = 1.0
x1, y1     = -d_binary/2, 0.0
x2, y2     =  d_binary/2, 0.0
depth_well = -0.85
sigma_well = 0.35

A     = 0.65      # wave amplitude
k     = 2.5       # wavenumber
decay = 3.5       # radial damping length

def wave_z(x, y):
    r = np.sqrt(x**2 + y**2)
    
    # 1. The static downward bend (gravity wells for the two masses)
    well1 = depth_well * np.exp(-((x - x1)**2 + (y - y1)**2) / (2 * sigma_well**2))
    well2 = depth_well * np.exp(-((x - x2)**2 + (y - y2)**2) / (2 * sigma_well**2))
    
    # 2. Envelope function so the wave fades out inside the deep wells
    # This stops the center from "piling up"
    env = 0.5 * (1.0 + np.tanh((r - 1.2) / 0.6))
    
    # 3. The outward rippling wave
    wave = env * A * np.sin(k * r) * np.exp(-r / decay)
    
    return well1 + well2 + wave

# --- Grid ----------------------------------------------------------------------
N, SPAN = 200, 6.0
gx   = np.linspace(-SPAN, SPAN, N)
X, Y = np.meshgrid(gx, gx)
Z    = wave_z(X, Y)

# Calculate radial distance for coloring (red in middle, blue far out)
R_dist = np.sqrt(X**2 + Y**2)

# --- Figure / axes --------------------------------------------------------------
fig = plt.figure(figsize=(12, 7.5))
fig.patch.set_facecolor("white")
ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

fig.suptitle("Spacetime responding to a Gravitational Wave",
             fontsize=15, fontweight="bold", y=0.88)
ax.set_title("masses bend space downwards, while transverse waves radiate outwards",
             fontsize=11, pad=0, color="dimgrey", y=0.83)

# --- Axis limits & box aspect ---------------------------------------------------
z_floor = -1.6
z_top   = 1.2                      
ax.set_xlim(-SPAN, SPAN); ax.set_ylim(-SPAN, SPAN); ax.set_zlim(z_floor, z_top)
BOX = (2*SPAN, 2*SPAN, 3.2)          
ax.set_box_aspect(BOX)

# Per-axis render scale
sx = BOX[0] / (2*SPAN)
sy = BOX[1] / (2*SPAN)
sz = BOX[2] / (z_top - z_floor)

# --- Spheres (Nestled in their wells) -------------------------------------------
def visual_sphere(cx, cy, cz, R, n=80):
    u = np.linspace(0, 2*np.pi, n)
    v = np.linspace(0, np.pi, n)
    rx, ry, rz = R * sx/sx, R * sx/sy, R * sx/sz   
    return (cx + rx * np.outer(np.cos(u), np.sin(v)),
            cy + ry * np.outer(np.sin(u), np.sin(v)),
            cz + rz * np.outer(np.ones_like(u), np.cos(v)))

ls_bh = LightSource(315, 45)
def draw_bh(cx, cy, R):
    # Calculate exactly how deep the well is at this location
    z_well = wave_z(cx, cy)
    # Nestle the sphere into the well (lower half sits inside)
    cz = z_well + 0.55 * R * sx/sz 
    XS, YS, ZS = visual_sphere(cx, cy, cz, R)
    rgb = ls_bh.shade(ZS, plt.cm.inferno, vert_exag=1.0, blend_mode="soft")
    ax.plot_surface(XS, YS, ZS, facecolors=rgb, rstride=1, cstride=1,
                    linewidth=0, antialiased=True, shade=False, zorder=10)
    return cz

R_bh = 0.35
cz1 = draw_bh(x1, y1, R_bh)
cz2 = draw_bh(x2, y2, R_bh)

# Orbital-motion hint (drawn around the center at the height of the black holes)
th = np.linspace(0, 2*np.pi, 200)
orb_r = d_binary / 2
ax.plot(orb_r*np.cos(th), orb_r*np.sin(th), np.full_like(th, (cz1+cz2)/2),
        ls="--", lw=1.2, color="black", alpha=0.67, zorder=9)

ax.text(0.0, 0.0, ((cz1+cz2)/2) + R_bh*sx/sz + 0.35, "Binary Source",
        fontsize=10, fontweight="bold", ha="center", zorder=11)

# --- Faint flat reference sheet --------------------------------------------------
ax.plot_surface(X, Y, np.zeros_like(Z), color="gainsboro", alpha=0.15,
                linewidth=0, antialiased=True, zorder=0)

# --- The wave sheet with RADIAL coloring + proper hill-shading -------------------
# 1. Normalize the radial distances from 0 to 1
norm_R = (R_dist - R_dist.min()) / (R_dist.max() - R_dist.min())

# 2. Get the "turbo" colormap (red at 0, blue at 1)
cmap = plt.get_cmap("turbo")

# 3. Convert the normalized radius into actual RGB colors from the colormap
base_colors = cmap(norm_R)[:, :, :3] 

# 4. Apply 3D topographic lighting to those flat colors based on the Z heights
ls_sheet = LightSource(azdeg=315, altdeg=50)
rgb_surface = ls_sheet.shade_rgb(base_colors, Z, vert_exag=2.0, blend_mode="soft")

# 5. Plot the fully shaded, radially colored surface
ax.plot_surface(X, Y, Z, facecolors=rgb_surface, rstride=1, cstride=1,
                linewidth=0, antialiased=True, shade=False,
                alpha=0.95, zorder=2)

ax.plot_wireframe(X, Y, Z, color="k", alpha=0.35,
                  linewidth=0.4, rstride=6, cstride=6, zorder=3)

# --- Contour projection on the floor ---------------------------------------------
levels = np.linspace(-A, A, 11)
ax.contour(X, Y, Z, levels=levels, zdir="z", offset=z_floor,
           cmap="Greys", linewidths=0.8, alpha=0.45, zorder=1)

# --- Cosmetics --------------------------------------------------------------------
ax.view_init(elev=26, azim=-55)
ax.set_xlabel(r"$x$", fontsize=12, labelpad=8)
ax.set_ylabel(r"$y$", fontsize=12, labelpad=8)
ax.set_zticks([]); ax.grid(False)
for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    pane.set_visible(False)
ax.zaxis.line.set_color((1, 1, 1, 0))

fig.subplots_adjust(left=0.0, right=1.0, bottom=0.02, top=1.02)

out = "Thesis_Ready_Plots/fig_gravitational_wave.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.show()
print("Saved:", out)