import scipy.constants as sc
import numpy as np

l_Pl = np.sqrt(sc.hbar * sc.G / sc.c**3)   # 1.616e-35 m
sigma = 4 * np.pi**2 * l_Pl**2
print(f"sigma = {sigma:.4e} m² = {sigma/l_Pl**2:.2f} l_Pl²")
# → sigma = 1.031e-68 m² = 0.395 l_Pl²  
# (0.31 after angular averaging over polarizations)