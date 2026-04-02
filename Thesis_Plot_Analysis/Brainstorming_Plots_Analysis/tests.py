import numpy as np
import matplotlib.pyplot as plt

def plot_hydrogen_quadrupole():
    r = np.linspace(0, 25, 500) # In units of Bohr radius a_0
    
    # Radial wavefunctions R_nl(r)
    # 1s state
    R_1s = 2 * np.exp(-r)
    # 3d state
    R_3d = (4 / (81 * np.sqrt(30))) * (r**2) * np.exp(-r / 3)
    
    # The integrand for the quadrupole transition <3d | h_{ij}p^i p^j | 1s>
    # is proportional to r^2 * R_3d * R_1s * r^2 (the last r^2 is the volume element)
    integrand = (r**2) * R_1s * R_3d * (r**2)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(r, R_1s**2 * r**2, 'b--', label=r'$|R_{1s}|^2 r^2$ (Initial State)')
    ax.plot(r, R_3d**2 * r**2, 'g--', label=r'$|R_{3d}|^2 r^2$ (Final State)')
    
    ax.fill_between(r, 0, integrand / np.max(integrand), color='crimson', alpha=0.3, 
                    label=r'Quadrupole Overlap Integrand $\propto r^4 R_{1s} R_{3d}$')
    ax.plot(r, integrand / np.max(integrand), color='crimson', lw=2)
    
    ax.set_xlabel(r'Radius $r$ ($a_0$)')
    ax.set_ylabel('Probability / Amplitude (Normalized)')
    ax.set_title('Boughn-Rothman Graviton Absorption: $1s \to 3d$ Radial Overlap', fontsize=13)
    
    ax.legend(loc='upper right')
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('fig4_boughn_rothman.png', dpi=300, bbox_inches='tight')
    
    print("Saved fig4_boughn_rothman.png")

if __name__ == "__main__":
    plot_hydrogen_quadrupole()


import numpy as np
import matplotlib.pyplot as plt

def plot_decoherence_budget():
    labels = [r'$^{13}$C Nuclear Spin Bath', 'Phonon/Thermal Noise', 'Surface Charge Noise', 'Required for $h \sim 10^{-21}$']
    values = [1e-3, 1e-4, 1e-5, 1e-15] # Fake T2 limits vs required phase integration time 
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    colors = ['#ff9999','#66b3ff','#99ff99', 'red']
    y_pos = np.arange(len(labels))
    
    ax.barh(y_pos, values, color=colors, edgecolor='black')
    
    ax.set_xscale('log')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r'Limiting Coherence Scale (seconds / arbitrary units)')
    ax.set_title('Decoherence Budget vs. Gravitational Target', fontsize=14)
    
    ax.grid(True, axis='x', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('fw_expansion_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved fw_expansion_convergence.png")

if __name__ == "__main__":
    plot_decoherence_budget()

