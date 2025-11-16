# nv_gw_prototype.py
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

# -------------------------
# Physics / numeric params (toy units first)
# -------------------------
D = 2.87e9           # Hz, zero-field splitting (we'll rescale to 'toy' units below)
gamma_e = 28e9       # Hz/T (not used in toy sim below)
Bz = 0.0             # Tesla

# Use toy units to make effects visible while debugging:
scale = 1e9          # divide Hz -> 'toy' unit (so D->~2.87)
D_toy = D / scale

# GW toy params
f_gw = 1.0           # toy frequency (cycles per toy-time unit)
omega_gw = 2 * np.pi * f_gw
A_toy = 0.02         # big on purpose to see effect (later reduce)

# Time array (cover many cycles)
tmax = 200.0
nsteps = 2000
tlist = np.linspace(0, tmax, nsteps)

# -------------------------
# Spin-1 operators
# -------------------------
Sx = qt.jmat(1, 'x')
Sy = qt.jmat(1, 'y')
Sz = qt.jmat(1, 'z')
I3 = qt.qeye(3)

# squared operators
Sx2 = Sx * Sx
Sy2 = Sy * Sy
Sz2 = Sz * Sz
Sp = Sx + 1j * Sy # S_plus via Sx,Sy
Sm = Sx - 1j * Sy

print("Check operator identity (Op_plus - 0.5*(Sp^2 + Sm^2)):")
# -------------------------
# Base Hamiltonian (toy)
# H0 = D Sz^2
# -------------------------
H0 = D_toy * Sz2

# -------------------------
# Interaction Hamiltonian operator part (quadrupole-like)
# H_int(t) = kappa * [ h_plus(t)*(Sx^2 - Sy^2) + h_cross(t)*(Sx*Sy + Sy*Sx) ]
# We'll set kappa = 1 (absorb into A_toy) for prototype
# -------------------------
Op_plus = (Sx2 - Sy2)
Op_cross = (Sx * Sy + Sy * Sx)

# Matrix element diagnostic
psi_p1 = qt.basis(3, 0)   # |+1>
psi_0  = qt.basis(3, 1)
psi_m1 = qt.basis(3, 2)

print((Op_plus - 0.5*(Sp*Sp + Sm*Sm)).norm())   # should be ~0
print("<+1|Op_plus|0> =", Op_plus.matrix_element(psi_p1, psi_0))
print("<+1|Sx|0>     =", Sx.matrix_element(psi_p1, psi_0))   # Δm=+1 allowed
print("<+1|Sp|0>     =", Sp.matrix_element(psi_p1, psi_0))   # should be nonzero
print("<+1|Sp^2|0>   =", (Sp*Sp).matrix_element(psi_p1, psi_0)) # zero: Δm=+2 needed

# show some matrix elements
print("Eigenenergies of H0 (toy units):", H0.eigenenergies())
print("Matrix element <+1|Op_plus|0> =", (psi_p1.dag() * Op_plus * psi_0))
print("Matrix element <+1|Sz^2|0> =", (psi_p1.dag() * Sz2 * psi_0))


# -------------------------
# Time-dependent functions (QuTiP expects f(t, args) signature)
# -------------------------
def h_plus_td(t, args):
    return args['A'] * np.sin(args['omega'] * t)

def h_cross_td(t, args):
    return 0.0

# build full H_td for QuTiP: list form
H_td = [H0,
        [Op_plus, h_plus_td],
        [Op_cross, h_cross_td]
       ]

# -------------------------
# Initial state: |0>
# -------------------------
psi0 = psi_0
rho0 = psi0.proj()

# -------------------------
# Decoherence (toy)
# -------------------------
gamma_deph = 0.01
c_ops = [np.sqrt(gamma_deph) * Sz]  # simple dephasing on Sz

# Observables: populations
proj_p1 = psi_p1.proj()
proj_0  = psi_0.proj()
proj_m1 = psi_m1.proj()

# -------------------------
# Solve with mesolve (Lindblad)
# -------------------------
args = {'A': A_toy, 'omega': omega_gw}
res = qt.mesolve(H_td, rho0, tlist, c_ops, [proj_p1, proj_0, proj_m1], args=args)

p_p1, p_0, p_m1 = [np.real(x) for x in res.expect]

# -------------------------
# Plot: populations
# -------------------------
plt.figure(figsize=(9,4))
plt.plot(tlist, p_p1, label='P(+1)')
plt.plot(tlist, p_0,  label='P(0)')
plt.plot(tlist, p_m1, label='P(-1)')
plt.xlabel('time (toy units)')
plt.ylabel('population')
plt.title(f'Toy NV + GW (A={A_toy}, f={f_gw} cycles/unit)')
plt.legend(); plt.grid(True)
plt.show()

# -------------------------
# Frequency scan (final population P(+1) vs omega) — quick coarse scan
# -------------------------
def freq_scan(omega_vals, A=A_toy):
    finals = []
    for w in omega_vals:
        args = {'A': A, 'omega': w}
        r = qt.mesolve(H_td, rho0, tlist, c_ops, [proj_p1], args=args)
        finals.append(np.real(r.expect[0][-1]))
    return np.array(finals)

omegas = np.linspace(0.2, 2.5, 40)
final_p = freq_scan(omegas)

plt.figure(figsize=(6,3))
plt.plot(omegas/(2*np.pi), final_p, '-o')  # convert back to cycles if desired
plt.xlabel('freq (cycles/unit)')
plt.ylabel('Final P(+1)')
plt.title('Resonance scan (toy)')
plt.grid(True)
plt.show()
