import qutip as qt
import numpy as np 
import matplotlib.pyplot as plt


#========= NV-center under Gravitational Wave Simulation using QuTiP =============#

# --------------------------------------------------------------------
# Spin-1 operators as (3x3 matrices)
# --------------------------------------------------------------------
Sx = qt.jmat(1, 'x')
Sy = qt.jmat(1, 'y')
Sz = qt.jmat(1, 'z')
I3 = qt.qeye(3)  # 3x3 Identity matrix

# --------------------------------------------------------------------
# Squared operators
# --------------------------------------------------------------------
Sx2 = Sx * Sx
Sy2 = Sy * Sy
Sz2 = Sz * Sz
Sp = qt.jmat(1, '+')  # Spin raising operator
Sm = qt.jmat(1, '-')  # Spin lowering operator
S_plus = Sx + 1j * Sy # Definition of S_plus and S_minus different from QuTiP's built-in ones
S_minus = Sx - 1j * Sy # Because QuTiP's are scaled differently, 
# The last ones are standard definitions

# ========= NV-center under Gravitational Wave Simulation using QuTiP ============#

# ================= Basis states,Operators and Hamiltonian Definitions ===========#

Op_plus = (Sx2- Sy2) # Operator part of GW interaction Hamiltonian
Op_cross = (Sx * Sy + Sy * Sx) # Cross polarization operator part

psi_p1 = qt.basis(3, 0)   # |ms = +1> 
psi_0    = qt.basis(3, 1)   # |ms = 0>
psi_m1 = qt.basis(3, 2)  # |ms = -1>

# --------------------------------------------------------------------
# NV-Center Hamiltonian
# --------------------------------------------------------------------

# Nv center Hamiltonian (Zero-field splitting + Zeeman term)

D = 2.87e9  # Hz (zero-field splitting ~ 2.87 GHz)
gamma_e = 28e9  # Hz/T (electron gyromagnetic ratio)
Bz = 0.0  # Tesla, external magnetic field along z (set to zero for simplicity)

# Use toy units to make effects visible while debugging:
scale = 1e9          # divide Hz -> 'toy' unit (so D->~2.87)
D_toy = D / scale

# GW toy params
f_gw = 2000           # toy frequency (cycles per toy-time unit)
omega_gw = 2 * np.pi * f_gw
A_toy = 0.02         # big on purpose to see effect (later reduce)

# Time array (cover many cycles)
tmax = 200.0
nsteps = 2000
tlist = np.linspace(0, tmax, nsteps)

H0 = D * Sz2  # Zero-field splitting term, simple for now 
# + gamma_e * Bz * Sz  # Zeeman term, Bz = 0 for later simplicity

# --------------------------------------------------------------------
# Small test: Verify spin-1 commutation relations
# --------------------------------------------------------------------

print((Op_plus - 0.5*(Sp*Sp + Sm*Sm)).norm())   # should be ~0
print("<+1|Op_plus|0> =", Op_plus.matrix_element(psi_p1, psi_0))
print("<+1|Sx|0>     =", Sx.matrix_element(psi_p1, psi_0))   # Δm=+1 allowed
print("<+1|Sp|0>     =", Sp.matrix_element(psi_p1, psi_0))   # should be nonzero
print("<+1|Sp^2|0>   =", (Sp*Sp).matrix_element(psi_p1, psi_0)) # zero: Δm=+2 needed

# show some matrix elements
print("Eigenenergies of H0 (toy units):", H0.eigenenergies())
print("Matrix element <+1|Op_plus|0> =", (psi_p1.dag() * Op_plus * psi_0))
print("Matrix element <+1|Sz^2|0> =", (psi_p1.dag() * Sz2 * psi_0))


#================== Basis states,Operators and Hamiltonian Definitions ===========#


#================= Gravitational Wave Interaction Hamiltonian ====================#

# --------------------------------------------------------------------
# Simple monochromatic GW waveform function
# --------------------------------------------------------------------

#def h_plus(t, f_gw = 1000, h_max = 1e-10): #Change f_gw and h_max as needed
    #"""GW strain: h_plus(t) = h_max * sin(2π f_gw t)"""

    #return h_max * np.sin(2 * np.pi * f_gw * t)
# We use a non simple waveform to see clearer effects

# --------------------------------------------------------------------
# Functions defining GW strain polarizations from h (t)
# --------------------------------------------------------------------

def h_plus(t, args):
    return args['A'] * np.sin(args['omega'] * t)


def h_cross(t,args):
    """No cross polarization in this simple example"""
    return 0.0

# --------------------------------------------------------------------
# Interaction Hamiltonian (MUST BE DERIVED FROM THEORY, MAIN PHYSICS)
# Based on quadrupole coupling: H_int = κ h_plus(t) (S_x² - S_y²)
# --------------------------------------------------------------------

kappa = 1e-3  # Coupling strength (toy value, to be replaced with real estimate)
H_int_operator = kappa * Op_cross  # Operator part of H_int, time-dep part is h_plus(t), change to include Op_plus also

def H_int(t, args = None):
    """Time-dependent interaction Hamiltonian from GW strain"""
    h_p = h_plus(t)
    h_c = h_cross(t)
    return h_p * H_int_operator + h_c # Only h_plus contributes in this simple example

ps0 = psi_0
rho0 = ps0.proj() # Density matrix initial state, pure state 

#================= Gravitational Wave Interaction Hamiltonian ====================#

# --------------------------------------------------------------------
#   Time Evolution Simulation #
# --------------------------------------------------------------------

# Total Hamiltonian 
# build full H_td for QuTiP: list form

H_nv = H0 # Static NV Hamiltonian
H = [H_nv, [H_int, h_plus]]  # QuTiP format for time-dependent Hamiltonian

# Alternative 1 : H = [H_nv, [H_int, lambda t, args: 1.0]]  # QuTiP format for time-dependent Hamiltonian, the lambda  t is needed to match QuTiP's expected format
# Alternative 2 : H = [H_nv, [H_int, 't']] # if H_int is defined as a function of t

#====================== Some decoherence options (not fully used) =================#

# Example collapse operators for T1 and T2 processes (not used now) aka depth of simulation
# T1 relaxation (|+1> -> |0> and |-1> -> |0>)
# T2 dephasing (|+1> <-> |-1>)
#gamma_T1 = 1/ (1e-3)  # 1 ms
gamma_T2 = 1/ (0.5e-3)  # 0.5 ms
c_ops = []
# c_ops.append(np.sqrt(gamma_T1) * qt.spre(psi_0 * psi_p1.dag()))  # |+1> -> |0>
# c_ops.append(np.sqrt(gamma_T1) * qt.spre(psi_0 * psi_m1.dag()))  # |-1> -> |0>
c_ops.append(np.sqrt(gamma_T2) * Sz)  # Simple Dephasing

#====================== Some decoherence options (not used now) ===================#

#===================== Time Evolution Simulation ==================================#

# --------------------------------------------------------------------
# We set up the initial state and time vector, also 
# --------------------------------------------------------------------
# Initial state: ( We start in |0> )  
psi0 = psi_0

# Time array (cover many cycles)
tmax = 200.0
nsteps = 2000
tlist = np.linspace(0, tmax, nsteps)


# --------------------------------------------------------------------
# Observable projections
# --------------------------------------------------------------------

proj_p1 = psi_p1.proj()
proj_0 = psi_0.proj()
proj_m1 = psi_m1.proj()

# --------------------------------------------------------------------
# Solve with mesolve (Lindblad)
# --------------------------------------------------------------------

args = {'A': A_toy, 'omega': omega_gw}
# We run the simulation (NO decoherence for simplicity)
result = qt.sesolve(H, psi0, tlist, e_ops=[])
res = qt.mesolve(H, rho0, tlist, c_ops, [proj_p1, proj_0, proj_m1], args=args)

p_p1, p_0, p_m1 = [np.real(x) for x in res.expect]


#===================== Time Evolution Simulation =================================#



#===================== Analyzing and Plotting Results ============================#

# Analyzing results and plotting population in |0> state")

# Calculate populations in the different states over time
# Calculate population in |0> state
p0 = np.zeros(len(tlist))
for i, state in enumerate(result.states):
    p0[i] = np.abs(psi_0.overlap(state))**2 # Here we use enumerate to get index and state !
# Calculate populations in |+1> and |-1> states
pop_plus1 = np.zeros(len(tlist))
pop_minus1 = np.zeros(len(tlist))
for i, state in enumerate(result.states):
    pop_plus1[i] = np.abs(psi_p1.overlap(state))**2
    pop_minus1[i] = np.abs(psi_m1.overlap(state))**2

# --------------------------------------------------------------------
# An alternative way using list comprehension: #
# --------------------------------------------------------------------

#pop_plus1 = [abs((psi_p1.overlap(state))) **2 for state in result.states]
#pop_minus1 = [abs(((psi_m1.overlap(state)))) **2 for state in result.states]
#pop_0 = [abs((psi_0.overlap(state))) **2 for state in result.states]

# --------------------------------------------------------------------
# Calculate expectation values
# --------------------------------------------------------------------

exp_Sz = qt.expect(Sz, result.states)
print(f"Final populations: P(+1)={p_p1[-1]:.4f}, P(0)={p_0[-1]:.4f}, P(-1)={p_m1[-1]:.4f}")
print("Final expectation <Sz> =", exp_Sz[-1], "\n")

#===================== Analyzing and Plotting Results ============================#


#======================= Plotting Results ========================================#

plt.figure(figsize=(12, 8))

# --------------------------------------------------------------------
#Plotting the populations first #
# --------------------------------------------------------------------

# Plotting populations
plt.subplot(2, 2, 1)
plt.plot(tlist * 1e3, p_p1, label='Population |+1>', linewidth = 2) 
plt.plot(tlist * 1e3, p_0, label='Population |0>', linewidth = 2)
plt.plot(tlist * 1e3, p_m1, label='Population |-1>, ', linewidth = 2)

plt.xlabel('Time (ms)')
plt.ylabel('Population')
plt.title(f'Toy NV + GW (A={A_toy}, f={f_gw} cycles/unit)')

plt.legend();plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------
# Plot GW strain h_plus(t) 
# --------------------------------------------------------------------

plt.subplot(2, 2, 2)
gw_strain_values = [h_plus(t) for t in tlist]
plt.plot(tlist * 1e3, gw_strain_values, "r-", color='orange', label='h_plus(t)', linewidth = 2)
plt.xlabel('Time (ms)')
plt.ylabel('GW Strain h_plus(t)')
plt.title('Gravitational Wave Signal')
plt.grid(True)

# --------------------------------------------------------------------
# Plot expectation value of Sz
# --------------------------------------------------------------------

plt.subplot(2, 2, 3)
plt.plot(tlist * 1000, exp_Sz, 'g-', linewidth=2)
plt.xlabel('Time (ms)')
plt.ylabel('<S_z>')
plt.title('Spin Expectation Value')
plt.grid(True)

# --------------------------------------------------------------------
# Plot energy level shifts (simplified)
# --------------------------------------------------------------------

plt.subplot(2, 2, 4)
# This is where you'd put your actual energy shift calculation
energy_shift = [h_plus(t) * kappa * 0.01 for t in tlist]  # Placeholder!
plt.plot(tlist * 1000, energy_shift, 'm-', linewidth=2)
plt.xlabel('Time (ms)')
plt.ylabel('Energy Shift (arb.)')
plt.title('GW-induced Energy Level Shifts')
plt.grid(True)

plt.tight_layout()
plt.show()

# --------------------------------------------------------------------
# Frequency scan (final population P(+1) vs omega) — quick coarse scan
# --------------------------------------------------------------------

def freq_scan(omega_vals, A=A_toy):
    finals = []
    for w in omega_vals:
        args = {'A': A, 'omega': w}
        r = qt.mesolve(H, rho0, tlist, c_ops, [proj_p1], args=args)
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

#======================= Plotting Results =========================================#


print(" Analysis complete")
print("NEXT STEPS:")
print("1. Run this code - see if GW causes population transfers")
print("2. Vary kappa to see stronger/weaker effects")  
print("3. Add decoherence (collapse operators)")
print("4. Replace placeholder kappa with FW-derived value")





