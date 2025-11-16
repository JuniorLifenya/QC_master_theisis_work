import qutip as qt
import numpy as np 
import matplotlib.pyplot as plt


#========= NV-center under Gravitational Wave Simulation using QuTiP ============#
print("Setting first up the NV-center as a spin-1 system \n")

# Spin-1 operators as (3x3 matrices)
Sx = qt.jmat(1, 'x')
Sy = qt.jmat(1, 'y')
Sz = qt.jmat(1, 'z')

# Squared operators
Sx2 = Sx ** 2
Sy2 = Sy ** 2
Sz2 = Sz ** 2

# ================= Basis states and Hamiltonian Definitions ===================#

psi_plus1 = qt.basis(3, 0)   # |ms = +1> 
psi_0    = qt.basis(3, 1)   # |ms = 0>
psi_minus1 = qt.basis(3, 2)  # |ms = -1>

# Nv center Hamiltonian (Zero-field splitting + Zeeman term)

D = 2.87e9  # Hz (zero-field splitting ~ 2.87 GHz)
gamma_e = 28e9  # Hz/T (electron gyromagnetic ratio)

H_nv = D * Sz2  # Zero-field splitting term, simple for now 

# ================ Gravitational Wave Interaction Hamiltonian ==================#

# Simple monochromatic GW waveform function
def h_plus(t, f_gw = 1000, h_max = 1e-20):
    """GW strain: h_plus(t) = h_max * sin(2π f_gw t)"""

    return h_max * np.sin(2 * np.pi * f_gw * t)

def h_cross(t):
    """No cross polarization in this simple example"""
    return 0.0

# Interaction Hamiltonian (MUST BE DERIVED FROM THEORY, MAIN PHYSICS)
# Based on quadrupole coupling: H_int = κ h_plus(t) (S_x² - S_y²)

kappa = 1e15  # Placeholder coupling constant (to be derived from FW transformation)
H_int_operator = kappa * (Sx2 - Sy2)  # Operator part of H_int

def H_int(t, args = None):
    """Time-dependent interaction Hamiltonian from GW strain"""
    h_p = h_plus(t)
    h_c = h_cross(t)
    return h_p * H_int_operator   + h_c # Only h_plus contributes in this simple example


# ==================== Time Evolution Simulation =============================#

# Total Hamiltonian 
H = [H_nv, [H_int_operator, h_plus]]
# Alternative 1 : H = [H_nv, [H_int, lambda t, args: 1.0]]  # QuTiP format for time-dependent Hamiltonian, the lambda  t is needed to match QuTiP's expected format
# Alternative 2 : H = [H_nv, [H_int, 't']] # if H_int is defined as a function of t

# Initial state: ( We start in |0> )  
psi0 = psi_0

# Time vector (math GW timescale)
tlist = np.linspace(0, 0.01, 1000)  # 10 ms total time, 1000 points

# We run the simulation (NO decoherence for simplicity)
result = qt.sesolve(H, psi0, tlist, [])


# ==================== Analyzing and Plotting Results ====================#
print("Analyzing results and plotting population in |0> state")

# Calculate populations the different states over time
# Calculate population in |0> state
p0 = np.zeros(len(tlist))
for i, state in enumerate(result.states):
    p0[i] = np.abs(psi_0.overlap(state))**2 # Here we use enumerate to get index and state !


#======================= An alternative way using list comprehension: ====================#
pop_plus1 = [abs((psi_plus1.overlap(state))) **2 for state in result.states]
pop_minus1 = [abs(((psi_minus1.overlap(state)))) **2 for state in result.states]
pop_0 = [abs((psi_0.overlap(state))) **2 for state in result.states]
# ========================================================================================#

# Calculate expectation values
exp_Sz = qt.expect(Sz, result.states)


#======================= Plotting Results ================================================#
plt.figure(figsize=(12, 8))


#=================Plotting the populations first =========================================#

plt.subplot(2, 2, 1)
plt.plot(tlist * 1e3, pop_plus1, label='Population |+1>', linewidth = 2)
plt.plot(tlist * 1e3, pop_0, label='Population |0>', linewidth = 2)
plt.plot(tlist * 1e3, pop_minus1, label='Population |-1, ', linewidth = 2)

#=================Plotting the populations first =========================================#

# Plotting populations
plt.figure(figsize=(10, 5))
plt.plot(tlist * 1e3, p0)  # Time in ms
plt.xlabel('Time (ms)')
plt.ylabel('Population')
plt.title('NV-Center Populations under monochromatic GW')
plt.legend()
plt.grid(True)

# Plot GW strain for reference
plt.subplot(2, 2, 2)
gw_strain_values = [h_plus(t) for t in tlist]
plt.plot(tlist * 1e3, gw_strain_values, "r-", color='orange', label='h_plus(t)', linewidth = 2)
plt.xlabel('Time (ms)')
plt.ylabel('GW Strain h_plus(t)')
plt.title('Gravitational Wave Signal')
plt.grid(True)


# Plot expectation value of Sz
plt.subplot(2, 2, 3)
plt.plot(tlist * 1000, exp_Sz, 'g-', linewidth=2)
plt.xlabel('Time (ms)')
plt.ylabel('<S_z>')
plt.title('Spin Expectation Value')
plt.grid(True)


# Plot energy level shifts (simplified)
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

print("✓ Analysis complete")
print("\n🎯 NEXT STEPS:")
print("1. Run this code - see if GW causes population transfers")
print("2. Vary kappa to see stronger/weaker effects")  
print("3. Add decoherence (collapse operators)")
print("4. Replace placeholder kappa with FW-derived value")





