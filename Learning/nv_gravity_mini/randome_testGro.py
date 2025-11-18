# nv_gw_INSTANT.py
# Runs in <1 second — perfect for thesis defense, interviews, live demos

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

class FastNVGW:
    def __init__(self):
        # Operators
        self.Sx = qt.jmat(1, 'x')
        self.Sy = qt.jmat(1, 'y')
        self.Sz = qt.jmat(1, 'z')
        self.psi_p1 = qt.basis(3, 0)
        self.psi_m1 = qt.basis(3, 2)
        self.Op = self.Sx**2 - self.Sy**2   # Δm=±2

    def run(self):
        # Toy units — everything scaled so numbers are nice and solver is happy
        D_toy = 0.0
        Bz_toy = 0.02
        gamma_toy = 1.0
        f_gw_toy = 0.5 * gamma_toy * Bz_toy   # resonant!
        kappa_toy = 50.0                      # strong coupling → fast flops
        h_max_toy = 1.0

        t_final = 4.0                             # 4 "Rabi periods"
        tlist = np.linspace(0, t_final, 800)     # only 800 points!

        def h_coeff(t, args):
            return h_max_toy * np.sin(2*np.pi*f_gw_toy * t)

        H = [gamma_toy * Bz_toy * self.Sz, [kappa_toy * self.Op, h_coeff]]

        # Super fast solver settings
        options = {'nsteps': 5000, 'atol': 1e-8, 'rtol': 1e-6}

        result = qt.sesolve(H, self.psi_p1, tlist, 
                          e_ops=[self.psi_p1.proj(), self.psi_m1.proj(), self.Sz],
                          options=options)

        return result, tlist

    def plot(self, result, tlist):
        p_p1, p_m1, Sz = result.expect

        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle('NV-Center GW Sensor — Instant Demo (Δm=±2 Rabi Flopping)', fontsize=16)

        ax[0].plot(tlist, p_p1, 'r-', lw=3, label='|+1⟩')
        ax[0].plot(tlist, p_m1, 'g-', lw=3, label='|-1⟩')
        ax[0].set_xlabel('Time (toy units)'); ax[0].set_ylabel('Population')
        ax[0].legend(); ax[0].grid(alpha=0.3)
        ax[0].set_title('Perfect Rabi Oscillations')

        ax[1].plot(tlist, np.sin(2*np.pi*0.01*tlist), 'purple', lw=2)
        ax[1].set_xlabel('Time'); ax[1].set_ylabel('h₊(t)'); ax[1].grid(alpha=0.3)
        ax[1].set_title('GW Strain (scaled)')

        ax[2].plot(tlist, Sz, 'orange', lw=3)
        ax[2].set_xlabel('Time'); ax[2].set_ylabel('⟨Sₖ⟩'); ax[2].grid(alpha=0.3)
        ax[2].set_title('Spin Polarization')

        plt.tight_layout()
        plt.show()

# =============== RUN INSTANTLY ===============
print("NV-CENTER GW DEMO — RUNS IN <1 SECOND")
print("="*50)
sim = FastNVGW()
result, tlist = sim.run()
sim.plot(result, tlist)
print("Done! Use this version for live demos, thesis defense, or interviews.")
print("Switch back to the full version only when you need realism + decoherence.")