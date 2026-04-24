import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import os

# Ensure output directory exists
os.makedirs("Thesis_Ready_Plots", exist_ok = True)


#---------------------------------------------------------
# 1. Firstly we Define the Manifold (A curved saddle surface)
#---------------------------------------------------------

def r_vec(x, y):
    """ A Position vector on the curved 2D manifold embedded in 3D.
    We use saddle geometry to clearly illustrate spatial curvature 
    """
    return np.array([x,y,0.6*x* y])
def dr_dx(x,y):
    """Tangent vector along coordinate x."""
    return np.array([1,0,0.6*y])
def dr_dy(x,y):
    """Tangent vector along coordinate y."""
    return np.array([0,1,0.6*x])
#---------------------------------------------------------
# 2. Tetrad Construction (Gram-Schmidt Orthonormalization)
# WHY? Because we need it to describe that each point has their own orthonormal basis, 
# and we need to use the tetrad to do the parallel transport !
#---------------------------------------------------------

def get_native_tetrad(x,y):
    """ Calculate the orthonormal tetrad
    (e1,e2,e3) at a given point, where e1,e2 span the tangent plane, and e3 is the normal vector.
    """
    rx = dr_dx(x,y)
    ry = dr_dy(x,y)

    # Normal vector e3
    n = np.cross(rx,ry)
    e3 = n/np.linalg.norm(n)

    # Orthonormal tangent vectors e1 and e2
    e1 = rx/np.linalg.norm(rx)
    e2 = np.cross(e3,e1) # Guarantees strict right-handed orthogonality

    return e1,e2,e3

#---------------------------------------------------------
# 3. Parallel Transport Approximation
# We will simulate the Levi-Civita parallel transport of a tangent vector V_A,
# to a new point B by projecting it onto the new tangent plane at B and re-normalizing.
#---------------------------------------------------------

def parallel_transport(V_A, n_B):
    """ Here we simulate the Levi-Civita parallel transport of a tangent vector V_A 
    to a new point B by projecting it onto the new tangent plane at B and re-normalize. 
    """

    # Project out the normal component at B to keep the vector in the tangent plane
    V_B_proj = V_A - np.dot(V_A, n_B) * n_B 
    return V_B_proj/np.linalg.norm(V_B_proj)

# ---------------------------------------------------------
# 4. Example Usage and Visualization
# ---------------------------------------------------------
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection ="3d")
fig.suptitle("Spin Connection: Parallel Transport vs Local Frame", fontsize=14, fontweight="bold")
ax.set_title(r"The mismatch angle at Point B represents $\omega_{\mu}^{ab} dx^\mu$", fontsize=12, pad=10)

# Generate now the Curved Grid (Manifold)
pts = np.linspace(-1.5, 1.5 ,15)
X,Y = np.meshgrid(pts,pts)
Z = 0.6 * X * Y # I do not get it, arent we suppose to use the old X^2-Y^2 methode for a saddle ? 
# I think this is just a different type of saddle, but it still has the same property of being curved in opposite directions along the x and y axes, which is what we want to illustrate the concept of curvature and parallel transport.

ax.plot_surface(X,Y,Z, alpha = 0.5 , color = "wheat", edgecolor = "none")
ax.plot_wireframe(X,Y,Z, color = "green", alpha = 0.15, linewidth = 0.8) # I have not encountered wire_frame before what is it? 
# It is a way to plot a 3D surface as a grid of lines, which can help visualize the shape of the surface without the distraction of colors or shading.
# Often used in combination with a surface plot to enhance the visual representation of the 3D structure.

# Define Points A and B 
A = np.array([-0.8, -0.8])
B = np.array([0.8, 0.8])

pos_A = r_vec(*A) # Hm why the star here ? Because A is a 1D array of shape (2,), and r_vec expects two separate arguments (x and y), so we unpack A into its components using *A.
pos_B = r_vec(*B)

# Get Native Tetrads
e1_A, e2_A, e3_A = get_native_tetrad(*A)
e1_B, e2_B, e3_B = get_native_tetrad(*B)

e1_trans = parallel_transport(e1_A, e3_B)
e2_trans = parallel_transport(e2_A, e3_B)
e3_trans = parallel_transport(e3_A, e3_B)

# ----------------------------------------------------------
# Geometrical Drawing: Parallel Transport Tetrad A to Point B
# -----------------------------------------------------------

L = 0.6 # Quiver length scaling 

def draw_tetrad(ax,pos,e1,e2,e3, ls ='-', alpha = 1.0, label_suffix = ""): # What is label_suffix here ? 
    """
    # It is an optional parameter that can be used to add a suffix to the labels of the tetrad vectors when they are drawn on the plot. 
    # This can help differentiate between different sets of tetrads, such as the original tetrad at point A and the parallel transported tetrad at point B.
    # For example, if label_suffix is "_A", then the labels for the tetrad vectors will be "e1_A", "e2_A", and "e3_A". If label_suffix is "_B", then the labels will be "e1_B", "e2_B", and "e3_B"
    """
    ax.quiver(*pos, *(e1 * L), color = 'red', linestyle = ls , alpha = alpha , arrow_length_ratio = 0.15, lw = 2.5,
              label =rf'$e_{{1}}{label_suffix}$')
    ax.quiver(*pos, *(e2 * L), color = 'green', linestyle = ls, alpha = alpha , arrow_length_ratio = 0.15, lw = 2.5,
              label =rf'$e_{{2}}{label_suffix}$')
    ax.quiver(*pos, *(e3 * L), color = 'royalblue', linestyle = ls, alpha = alpha , arrow_length_ratio = 0.15, lw = 2.5,
              label =rf'$e_{{3}}{label_suffix}$')
    """What does ax.quiver really do here?
    It is a function from Matplotlib that is used to plot vectors as arrows in a
    3D space. In this context, it is being used to draw the tetrad vectors (e1, e2, e3) at a specific position (pos) on the plot.
    The parameters passed to ax.quiver include:
    - *pos: The starting point of the arrows, which is the position of the point where the tetrad is being drawn.
    - *(e1 * L), *(e2 * L), *(e3 * L): The components of the tetrad vectors scaled by a length factor L to control the size of the arrows.
    and again we utilize the unpacking operator * to pass the components of the vectors as separate arguments.
    """
    
# Now we Draw Points and Geodesic Path (Approximate linear path in coordinate space)
t_vals = np.linspace(0,1,20)
path_X = A[0] + t_vals*(B[0] - A[0])
path_Y = A[1] + t_vals*(B[1] - A[1])
path_Z = 0.6 * path_X * path_Y
ax.plot(path_X, path_Y, path_Z, color = "dimgray", linestyle = ':', lw = 2, label= 'Path $x^\mu(\lambda)$')

ax.scatter(*pos_A, color = "black", s = 50)
ax.text(pos_A[0], pos_A[1], pos_A[2] + 0.2, "Point A\n (Native Tetrad) ", fontsize = 10, fontweight = 'bold', ha = 'center')

ax.scatter(*pos_B, color = "black", s = 50)
ax.text(pos_B[0], pos_B[1], pos_B[2] + 0.3, "Point B", fontsize = 10, fontweight = 'bold', ha = 'center')

# Draw Tetrads
# 1. Native Tetrad at A
draw_tetrad(ax, pos_A, e1_A, e2_A, e3_A, ls = '-', label_suffix = "(A)")
# 2. Native Tetrad at B
draw_tetrad(ax, pos_B, e1_B, e2_B, e3_B, ls = '-', alpha = 0.9, label_suffix = "(B native)")

# 3. Transported Tetrad at B (Dashed)
ax.quiver(*pos_B, *(e1_trans * L), color = 'darkred', linestyle = '--', alpha = 0.8, arrow_length_ratio = 0.15, lw = 2.5, label = r'$\tilde{e}_1$ (Transported)')
ax.quiver(*pos_B, *(e2_trans * L), color = 'darkgreen', linestyle = '--', alpha = 0.8, arrow_length_ratio = 0.15, lw = 2.5, label = r'$\tilde{e}_2$ (Transported)')
ax.quiver(*pos_B, *(e3_trans * L), color = 'darkblue', linestyle = '--', alpha = 0.8, arrow_length_ratio = 0.15, lw = 2.5, label = r'$\tilde{e}_3$ (Transported)')

# Now Adjusting the View and Legend
ax.view_init(elev = 30, azim = 120, roll = 12)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)
ax.set_xlabel(r'$x$', fontsize = 12)
ax.set_ylabel(r'$y$', fontsize = 12)
ax.set_zlabel(r'$z$', fontsize = 12)

# Clean legend handling to avoid duplicates
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels,handles)) # This creates a dictionary that maps each unique label to its corresponding handle, effectively removing duplicates.
ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize = 9)

plt.tight_layout() # Adjusts the padding between and around subplots to minimize overlaps and ensure a clean layout. I
plt.savefig("Thesis_Ready_Plots/SpinConnect_PointMove.png", dpi = 300, bbox_inches = 'tight') # This saves the current figure to a file named "SpinConnect_PointMove.png" in the "Thesis_Ready_Plots" directory with a resolution of 300 dots per inch (DPI) and tight bounding box to minimize whitespace around the image.
print("Saved: Thesis_Ready_Plots/SpinConnect_PointMove.png")
# It is often used after adding elements to a plot to improve the overall appearance and readability of the figure.
plt.show()
print("Plotting complete. The plot should now be displayed.")