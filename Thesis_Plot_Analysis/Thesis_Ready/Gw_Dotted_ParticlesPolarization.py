import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec

#-------------------------------------------------------
# We commence with some parameters
#-------------------------------------------------------

h_p = 1.0
h_c = 1.0
omega = 2* np.pi
n_frames = 5
n_particles = 20

theta = np.linspace(0, 2* np.pi, n_particles,endpoint=False )
x0 = np.cos(theta) 
y0 = np. sin(theta)

phases = np.linspace(0,np.pi,n_frames)

fig = plt.figure(figsize= (20,5))
gs = gridspec.GridSpec(2,n_frames +1, figure = fig, hspace = 0.35, wspace = 0.25)


cmap = plt.cm.Blues(np.linspace(0.3, 0.9,n_frames))

for pol_idx, (pol_label,hxx,hxy) in enumerate(
    [("$h_+$", h_p, 0),
            ("$h_\\times$", 0, h_c)]):
    ax_label = fig.add_subplot(gs[pol_idx,0])
    ax_label.axis("off")
    ax_label.text(1, 0.5, pol_label, fontsize = 15,
                 ha = "center", va= "center", fontweight= "bold"
                  )
    for fi, phase in enumerate(phases):
        ax = fig.add_subplot(gs[pol_idx, fi +1])

        strain_xx = hxx * np.cos(phase)
        strain_xy = hxy * np.cos(phase)
        strain_yy = -hxx * np.cos(phase)

        # First-order displacement: δx^i = -1/2 * h^i_j * x^j
        dx = -0.7 * (strain_xx * x0 + strain_xy * y0)
        dy = -0.7 * (strain_xy * x0 + strain_yy * y0) 

        xp = x0 + dx
        yp = y0 + dy

        ax.fill(xp,yp, alpha = 0.25, color = cmap[fi])
        ax.plot(np.append(xp,xp[0]),
                np.append(yp,yp[0]), color=cmap[fi],lw = 1.5)
        
        ax.plot(x0,y0, "k--", lw=0.6, alpha =0.4)
        ax.scatter(xp,yp,s=12, color = cmap[fi], zorder = 4)

        ax.set_xlim(-1.8,1.8); ax.set_ylim(-1.8,1.8)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"$\\omega t = {phase/np.pi:.2f}\\pi$",
                     fontsize = 10)
        
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
fig.suptitle("Tidal deformation of a ring of test particles",
             fontsize = 10, y = 0.99, x = 0.59)

plt.show()
print("Finished and saved: dotted_blue_polarization.png")