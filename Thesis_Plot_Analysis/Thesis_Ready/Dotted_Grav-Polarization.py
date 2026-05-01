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
n_particles = 16

theta = np.linspace(0, 2* np.pi /2, n_frames )
x0 = np.cos(theta)
y0 = np. sin