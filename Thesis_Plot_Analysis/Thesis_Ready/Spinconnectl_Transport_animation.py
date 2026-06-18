import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os, shutil, matplotlib

# Make sure ffmpeg is found
if not shutil.which("ffmpeg"):
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

os.makedirs("Thesis_Ready_Plots", exist_ok=True)

# ------------------------- manifold & tetrad functions (unchanged) -------------------------
def r_vec(x, y):
    return np.array([x, y, 0.6*x*y])

def dr_dx(x,y):
    return np.array([1, 0, 0.6*y])

def dr_dy(x,y):
    return np.array([0, 1, 0.6*x])

def get_native_tetrad(x,y):
    rx = dr_dx(x,y)
    ry = dr_dy(x,y)
    n = np.cross(rx, ry)
    e3 = n / np.linalg.norm(n)
    e1 = rx / np.linalg.norm(rx)
    e2 = np.cross(e3, e1)
    return e1, e2, e3

def parallel_transport(V_prev, n_new):
    """Project V_prev onto the tangent plane at the new point and normalise."""
    V_proj = V_prev - np.dot(V_prev, n_new) * n_new
    return V_proj / np.linalg.norm(V_proj)

# ------------------------- path & points -------------------------
A = np.array([-0.8, -0.8])
B = np.array([0.8, 0.8])
N_path = 100                              # number of frames = number of points along path
t_vals = np.linspace(0, 1, N_path)

path_X = A[0] + t_vals * (B[0] - A[0])
path_Y = A[1] + t_vals * (B[1] - A[1])
path_Z = 0.6 * path_X * path_Y
path_points = np.stack([path_X, path_Y, path_Z], axis=-1)   # (N_path, 3)

# compute transported basis along the whole path
basis1 = np.zeros_like(path_points)
basis2 = np.zeros_like(path_points)
basis3 = np.zeros_like(path_points)

e1_A, e2_A, e3_A = get_native_tetrad(*A)
basis1[0] = e1_A
basis2[0] = e2_A
basis3[0] = e3_A

for i in range(1, N_path):
    n_i = get_native_tetrad(path_X[i], path_Y[i])[2]   # only need e3 (normal)
    basis1[i] = parallel_transport(basis1[i-1], n_i)
    basis2[i] = parallel_transport(basis2[i-1], n_i)
    basis3[i] = parallel_transport(basis3[i-1], n_i)

# ------------------------- static figure elements -------------------------
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")
fig.suptitle("Parallel Transport of a spin-1/2 particle", fontsize=14, fontweight="bold")

# saddle surface
pts = np.linspace(-1.5, 1.5, 15)
X, Y = np.meshgrid(pts, pts)
Z = 0.6 * X * Y
ax.plot_surface(X, Y, Z, alpha=0.5, color="wheat", edgecolor="none")
ax.plot_wireframe(X, Y, Z, color="green", alpha=0.15, linewidth=0.8)

# path
ax.plot(path_X, path_Y, path_Z, color="dimgray", linestyle=':', lw=2, label=r'Path $x^\mu(\lambda)$')

# points A & B (static)
pos_A = r_vec(*A)
pos_B = r_vec(*B)
ax.scatter(*pos_A, color="black", s=50)
ax.text(pos_A[0], pos_A[1], pos_A[2]+0.5, "Point A\n(Native Tetrad)", fontsize=10, fontweight='bold', ha='center')
ax.scatter(*pos_B, color="black", s=50)
ax.text(pos_B[0], pos_B[1]+0.4, pos_B[2]+0.8, "Point B", fontsize=10, fontweight='bold', ha='center')

# native tetrad at A (faint)
L = 0.6
ax.quiver(*pos_A, *(e1_A*L), color='red', alpha=0.4, arrow_length_ratio=0.15, lw=2)
ax.quiver(*pos_A, *(e2_A*L), color='green', alpha=0.4, arrow_length_ratio=0.15, lw=2)
ax.quiver(*pos_A, *(e3_A*L), color='royalblue', alpha=0.4, arrow_length_ratio=0.15, lw=2)

# native tetrad at B (faint)
e1_B, e2_B, e3_B = get_native_tetrad(*B)
ax.quiver(*pos_B, *(e1_B*L), color='red', alpha=0.4, arrow_length_ratio=0.15, lw=2)
ax.quiver(*pos_B, *(e2_B*L), color='green', alpha=0.4, arrow_length_ratio=0.15, lw=2)
ax.quiver(*pos_B, *(e3_B*L), color='royalblue', alpha=0.4, arrow_length_ratio=0.15, lw=2)

# store the moving quiver artists (we will replace them each frame)
moving_artists = []

# ------------------------- animation update -------------------------
def update(frame):
    global moving_artists
    # remove previous moving tetrad
    for art in moving_artists:
        art.remove()
    moving_artists.clear()

    pos = path_points[frame]
    e1 = basis1[frame]
    e2 = basis2[frame]
    e3 = basis3[frame]

    # draw new transported tetrad
    q1 = ax.quiver(*pos, *(e1*L), color='darkred', lw=2.5, arrow_length_ratio=0.15)
    q2 = ax.quiver(*pos, *(e2*L), color='darkgreen', lw=2.5, arrow_length_ratio=0.15)
    q3 = ax.quiver(*pos, *(e3*L), color='darkblue', lw=2.5, arrow_length_ratio=0.15)
    moving_artists.extend([q1, q2, q3])

    
    # optionally highlight current position
    dot, = ax.plot([pos[0]], [pos[1]], [pos[2]], 'ko', markersize=6)
    moving_artists.append(dot)

    
    return moving_artists

    
    

# view settings
ax.view_init(elev=30, azim=120, roll=12)
ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_zlim(-2, 2)
ax.set_xlabel(r'$x$'); ax.set_ylabel(r'$y$'); ax.set_zlabel(r'$z$')

# clean legend (only static elements)
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.05,0.5))

plt.tight_layout()

# ------------------------- save animation -------------------------
anim = FuncAnimation(fig, update, frames=N_path, interval=100, blit=False)

mp4_file = "Thesis_Ready_Plots/ParallelTransport.mp4"
anim.save(mp4_file, writer=FFMpegWriter(fps=10, bitrate=2000), dpi=150)
print("Saved:", mp4_file)

# Optionally save a GIF
gif_file = "Thesis_Ready_Plots/ParallelTransport.gif"
anim.save(gif_file, writer='pillow', fps=10)
print("Saved:", gif_file)

if not hasattr(update, "counter"):
    update.counter = 0
plt.savefig(f"Thesis_Ready_Plots/tess_frame_{update.counter:04d}.png", dpi=150)
update.counter += 1
for i in range(N_path):
    update(i)   # re‑draw the frame
    plt.savefig(f"Thesis_Ready_Plots/transport_frame_{i:04d}.png", dpi=150)
    
plt.show()