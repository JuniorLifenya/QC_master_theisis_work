import numpy as np
import matplotlib.pyplot as plt

# ---------- Hydrogen ----------
kappa = 8.19e-28
E2 = 3.4

hvals = np.logspace(-30, -18, 400)

DeltaE_atom = (4/5)*kappa*E2*hvals


# ---------- Resonator ----------
# order-of-magnitude Tobar scaling

M = 1e-3        # kg
omega = 2*np.pi*1e4
hbarSI = 1.054e-34

L = 0.01

beta = (L/np.pi**2)*np.sqrt(M*hbarSI/omega)

DeltaE_res = beta*hvals*1e25

# ---------- Plot ----------
plt.figure(figsize=(7,5))

plt.loglog(hvals, DeltaE_atom, label="Hydrogen")
plt.loglog(hvals, DeltaE_res, label="Resonator")

plt.xlabel("GW strain h")
plt.ylabel("Coupling scale")
plt.title("Atomic vs Macroscopic Coupling")

plt.legend()
plt.grid(True, which="both")
plt.tight_layout()
plt.show()