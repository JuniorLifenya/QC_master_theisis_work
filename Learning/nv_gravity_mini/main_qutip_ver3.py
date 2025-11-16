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

#S_plus = Sx + 1j * Sy # Definition of S_plus and S_minus different from QuTiP's built-in ones
#S_minus = Sx - 1j * Sy # Because QuTiP's are scaled differently, 
# The last ones are standard definitions

# ========= NV-center under Gravitational Wave Simulation using QuTiP ============#

# ================= Basis states,Operators and Hamiltonian Definitions ===========#



psi_p1 = qt.basis(3, 0)   # |ms = +1> 
psi_0  = qt.basis(3, 1)   # |ms = 0>
psi_m1 = qt.basis(3, 2)  # |ms = -1>

# --------------------------------------------------------------------
# NV-Center Hamiltonian Parameters
# --------------------------------------------------------------------

# Nv center Hamiltonian (Zero-field splitting + Zeeman term)

D = 2.87e9  # Hz (zero-field splitting ~ 2.87 GHz)
gamma_e = 28e9  # Hz/T (electron gyromagnetic ratio)
Bz = 0.0  # Tesla, external magnetic field along z (set to zero for simplicity)

# Use toy units to make effects visible while debugging:
scale = 1e9          # divide Hz -> 'toy' unit (so D->~2.87)
D_toy = D / scale

# GW params
f_gw = 2000           # toy frequency (cycles per toy-time unit)
omega_gw = 2 * np.pi * f_gw
A_toy = 0.02 # big amplitude on purpose to see effect (later reduce)

# Time array (cover many cycles)
tmax = 200.0
nsteps = 2000
tlist = np.linspace(0, tmax, nsteps)

H0 = D * Sz2  # Zero-field splitting term, simple for now 
# + gamma_e * Bz * Sz  # Zeeman term, Bz = 0 for later simplicity

# --------------------------------------------------------------------
# Small Debug: Verify spin-1 commutation relations
# --------------------------------------------------------------------

print("======Debug Info=======", "\n")
print("Verifying spin-1 commutation relation [Sx, Sy] = iħSz:")
def spin_commutation(a, b):
    """Compute the commutator [a, b]"""
    return a * b - b * a   
comm_Sx_Sy = spin_commutation(Sx, Sy)
expected = 1j * Sz
if np.allclose(comm_Sx_Sy.full(), expected.full()):
    print("✓ Commutation relation [Sx, Sy] = iħSz verified!")
else:
    print("✗ Commutation relation [Sx, Sy] = iħSz NOT verified!")
print("\n")

# --------------------------------------------------------------------
# Checking some matrix elements
# --------------------------------------------------------------------
print("Checking some S_i matrix elements <+1|Si|0>:", "\n")

#print("Matrix element <+1|Sx|0> =", (psi_p1.dag() * Sx * psi_0), "\n")   # Δm=+1 allowed
#print("Matrix element <+1|Sy|0> =", (psi_p1.dag
#print("Matrix element <+1|Sz^2|0> =", (psi_p1.dag() * Sz2 * psi_0)) More beutiful way:

print(f"<+1|Sx|0>     = {Sx.matrix_element(psi_p1, psi_0)} \n")   # Δm=+1 allowed
print(f"<+1|Sy|0>     = {Sy.matrix_element(psi_p1, psi_0)} \n")   # Δm=+1 allowed
print(f"<+1|Sz|0>     = {Sz.matrix_element(psi_p1, psi_0)} \n")   # should be zero: Δm=+1 not allowed

print("Checking some S_i matrix elements <+1|Si^2|0>:", "\n")
print(f"<+1|Sx^2|0>   = {(Sx2).matrix_element(psi_p1, psi_0)} \n") # zero: Δm=+2 needed
print(f"<+1|Sy^2|0>   = {(Sy2).matrix_element(psi_p1, psi_0)} \n") # zero: Δm=+2 needed
print(f"<+1|Sz^2|0>   = {(Sz2).matrix_element(psi_p1, psi_0)} \n") # zero: Δm=+2 needed

print("Checking some S_plus and S_minus matrix elements <+1|Sp/Sm|0>:", "\n")
print(f"<+1|Sp|0>     = {Sp.matrix_element(psi_p1, psi_0)} \n")   # should be nonzero
print(f"<+1|Sp^2|0>   = {(Sp*Sp).matrix_element(psi_p1, psi_0)} \n") # zero: Δm=+2 needed
# --------------------------------------------------------------------
# Checking NV-center stuff, and GW interaction operator
# --------------------------------------------------------------------
print(f"NV Hamiltonian H0 =\n{H0.full()}\n")
print(f" H0 eigenenergies = {H0.eigenenergies()}\n")
print(f"NV-center zero-field splitting D = {D_toy} Hz")
print(f"GW frequency f_gw = {f_gw} cycles/unit, omega_gw = {omega_gw} rad/unit")
print("GW amplitude A_toy =", A_toy)

print("\n================ End of Debug Info ================\n")

#================== Basis states,Operators and Hamiltonian Definitions ===========#


#================= Gravitational Wave Interaction Hamiltonian ====================#



# --------------------------------------------------------------------
# Functions defining GW strain polarizations from h (t)
# --------------------------------------------------------------------


# Simple monochromatic GW waveform function

def h_plus(t, A, omega): #Change f_gw and h_max as needed
    """GW strain: h_plus(t) = h_max * sin(2π f_gw t)"""
    return A * np.sin(omega* t)

def h_cross(t,A, omega):
    """No cross polarization in this simple example"""
    return 0.0

# --------------------------------------------------------------------
# Interaction Hamiltonian (MUST BE DERIVED FROM THEORY, MAIN PHYSICS)
# Based on quadrupole coupling: H_int = κ h_plus(t) (S_x² - S_y²)
# --------------------------------------------------------------------

kappa = 1e-3  # Coupling strength (toy value, to be replaced with real estimate)

Op_plus = (Sx2- Sy2) # Operator part of GW interaction Hamiltonian
Op_cross = (Sx * Sy + Sy * Sx) # Cross polarization operator part

H_int_operator = kappa * Op_cross  # Operator part of H_int, time-dep part is h_plus(t), change to include Op_plus also

# --------------------------------------------------------------------
# QuTiP time-dependent Hamiltonian function
# --------------------------------------------------------------------
# Methode 1: Using string-based time-dep (Recommended for simple cases)
H_nv = H0 # Static NV Hamiltonian
H_td = [H0, [H_int_operator, f"A*sin(omega*t)"]]  # QuTiP format for time-dependent Hamiltonian

# Methode 2: Using lambda function-based time-dep (more flexible for complex cases)
# H_td = [H0, [H_int_operator, lambda t, args: args['A'] * np.sin(args['omega'] * t)]]

# Arguments for time-dependent Hamiltonian
args = {'A': A_toy, 'omega': omega_gw}

# --------------------------------------------------------------------
# Checking some Operator relations 
# --------------------------------------------------------------------
print((Op_plus - 0.5*(Sp*Sp + Sm*Sm)).norm())   # should be ~0, we are checking Op_plus definition Sx2 - Sy2
print((Op_cross - (1j/2)*(Sp*Sp - Sm*Sm)).norm()) # should be ~0, we are checking Op_cross definition SxSy + SySx

print(f"GW Interaction Operator Op_plus =\n{Op_plus.full()}\n") # SHould print the matrix form
print(f"GW Interaction Operator Op_cross =\n{Op_cross.full()}\n")
print("Matrix element <+1|Op_plus|0> =", (psi_p1.dag() * Op_plus * psi_0), "\n")  # Δm=+1 allowed
print("Matrix element <+1|Op_cross|0> =", (psi_p1.dag() * Op_cross * psi_0), "\n") # should be zero: Δm=+1 not allowed


print("faster methods to get Op_plus_minus matrix elements:", "\n")
print("<+1|Op_plus|0> =", Op_plus.matrix_element(psi_p1, psi_0))   # Δm=+1 allowed
print("<+1|Op_cross|0> =", Op_cross.matrix_element(psi_p1, psi_0)) # should be zero: Δm=+1 not allowed
print("\n")


#================= Gravitational Wave Interaction Hamiltonian ====================#

#====================== Initial State and Observables ============================#


# --------------------------------------------------------------------
# We set up the initial state and time vector, also 
# --------------------------------------------------------------------

# Initial state: ( We start in |0> )  
psi0 = psi_0
rho0 = psi0 * psi0.dag()  # Density matrix form

# --------------------------------------------------------------------
# Observable projections
# --------------------------------------------------------------------

proj_p1 = psi_p1.proj() # Or written as psi_p1 * psi_p1.dag()
proj_0 = psi_0.proj()
proj_m1 = psi_m1.proj()

#====================== Initial State and Observables ============================#

#====================== Some decoherence options (not used now) ==================#
# Example collapse operators for T1 and T2 processes (not used now) aka depth of simulation
# T1 relaxation (|+1> -> |0> and |-1> -> |0>)
# T2 dephasing (|+1> <-> |-1>)
#gamma_T1 = 1/ (1e-3)  # 1 ms
gamma_T2 = 1/ (0.5e-3)  # 0.5 ms
c_ops = []
# c_ops.append(np.sqrt(gamma_T1) * qt.spre(psi_0 * psi_p1.dag()))  # |+1> -> |0>
# c_ops.append(np.sqrt(gamma_T1) * qt.spre(psi_0 * psi_m1.dag()))  # |-1> -> |0>
c_ops.append(np.sqrt(gamma_T2) * Sz)  # Simple Dephasing

#====================== Some decoherence options (not used now) ==================#

#===================== Time evolution ============================================#

print(f"\n===== Starting time evolution simulation ======\n")


# --------------------------------------------------------------------
# Solve with mesolve (Lindblad)
# --------------------------------------------------------------------

# We run the simulation (NO decoherence for simplicity)
result = qt.mesolve(H_td,rho0, tlist, c_ops=c_ops, e_ops=[proj_p1,proj_0,proj_m1,Sz,], args=args)

# Extract results for populations
p_p1, p_0, p_m1 , exp_Sz = [np.real(x) for x in result.expect]

print(f"Final populations: P(+1)={p_p1[-1]:.4f}, P(0)={p_0[-1]:.4f}, P(-1)={p_m1[-1]:.4f}")
print(f"Final expectation <Sz> ={exp_Sz[-1]}\n")

#===================== Solving the Master Equation ===============================#



#======================= Plotting Results ========================================#

plt.figure(figsize=(12, 10))

# --------------------------------------------------------------------
#Plotting the populations first #
# --------------------------------------------------------------------

# Plotting populations
plt.subplot(2, 3, 1)
plt.plot(tlist * 1e3, p_p1, label='Population |+1>', linewidth = 2, color = "red") 
plt.plot(tlist * 1e3, p_0, label='Population |0>', linewidth = 2, color = "blue")
plt.plot(tlist * 1e3, p_m1, label='Population |-1>, ', linewidth = 2, color = "green")

plt.xlabel('Time (ms)')
plt.ylabel('Population')
plt.title(f'NV center Populations under GW Influence with :(A={A_toy}, f={f_gw} cycles/unit)')

plt.legend();plt.grid(True)
#plt.tight_layout() # Change layout to avoid overlap
plt.show()

# --------------------------------------------------------------------
# Plot GW strain h_plus(t) 
# --------------------------------------------------------------------

plt.subplot(2, 3, 2)
gw_strain = A_toy*np.sin(omega_gw * tlist)
plt.plot(tlist * 1e3, gw_strain, "r-", color='orange', label='h_plus(t)', linewidth = 2)
plt.xlabel('Time (ms)')
plt.ylabel('GW Strain h_plus(t)')
plt.title('Gravitational Wave Signal')
plt.grid(True)

# --------------------------------------------------------------------
# Plot expectation value of Sz
# --------------------------------------------------------------------

plt.subplot(2, 3, 3)
plt.plot(tlist * 1e3, exp_Sz, 'g-', linewidth=2, color='purple')
plt.xlabel('Time (ms)')
plt.ylabel('<S_z>')
plt.title('Spin Expectation Value')
plt.grid(True)

# --------------------------------------------------------------------
# Population in |0> state (most relevant for sensing)
# --------------------------------------------------------------------

plt.subplot(2,3,4)
plt.plot(tlist * 1000, p_0, 'b-', linewidth=2,color='cyan')
plt.xlabel('Time (ms)')
plt.ylabel('Population P(|0>)')
plt.title('Population in |0> State')
plt.grid(True)

# --------------------------------------------------------------------
# Plot energy level shifts (simplified)
# --------------------------------------------------------------------

#plt.subplot(2, 3, 5)
# This is where you'd put your actual energy shift calculation
#energy_shift = [h_plus(t) * kappa * 0.01 for t in tlist]  # Placeholder!
#plt.plot(tlist * 1000, energy_shift, 'm-', linewidth=2)
#plt.xlabel('Time (ms)')
#plt.ylabel('Energy Shift (arb.)')
#plt.title('GW-induced Energy Level Shifts')
#plt.grid(True)

#plt.tight_layout()
#plt.show()

# --------------------------------------------------------------------
# Ploting Combined view of GW effect and population
# --------------------------------------------------------------------
plt.subplot(2, 3, 5)

# Calculate population transfer amplitude
transfer_amplitude = np.sqrt((p_p1-p_p1[0])**2 + (p_m1-p_m1[0])**2) # Simple measure of transfer
plt.plot(tlist * 1e3, transfer_amplitude, 'o-', label='h_plus(t)', linewidth=1, color='brown')
plt.xlabel('Time (ms)')
plt.ylabel('Population Transfer Amplitude')
plt.title('GW-induced Population Transfer')
plt.grid(True)

# --------------------------------------------------------------------
# Frequency scan (final population P(+1) vs omega) — quick coarse scan
# --------------------------------------------------------------------
plt.subplot(2, 3, 5)

# FFT of population in |0> state to see GW frequency components
from scipy.fft import fft, fftfreq
dt = tlist[1] - tlist[0]
fft_p0 = fft(p_0 - np.mean(p_0))  # Remove DC component
freqs = fftfreq(len(tlist), dt)
positive_freq_index = freqs>0 # Only positive frequencies
plt.plot(freqs[positive_freq_index]/1e3, np.abs(fft_p0[positive_freq_index]), 'b-', linewidth=2, color='magenta')
plt.axvline(x= f_gw/1e3, color='red', linestyle='--', label=f'GW freq: {f_gw/1e3:.1f} kHz')
plt.xlabel('Frequency (kHz)')
plt.ylabel('FFT Amplitude of P(|0>)')
plt.title('FFT Spectrum of Population P(|0>)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ======================= Resonance Analysis (Optional) =========================#
print("\n===== Starting frequency scan simulation ======\n")
def freq_scan(omega_vals, A=A_toy):
    """Scan different GW frequencies to find some resonance"""
    final_populations = []
    for i, w in enumerate(omega_vals):
        if i % 10 == 0:
            print(f"Scanning freq ({i+1}/{len(omega_vals)})..., omega={w:.2f}")
        args_scan = {'A': A, 'omega': w}
        result_scan = qt.mesolve(H_td, rho0, tlist, c_ops=c_ops, e_ops= [proj_p1], args=args)
        final_populations.append(np.real(result_scan.expect[0][-1]))
    return np.array(final_populations)

# Frequency range for scan, these are expected sesonant frequencies
omega_scan = np.linspace(0.5* omega_gw, 2*omega_gw, 50)
final_p_plus1 = freq_scan(omega_scan)

plt.figure(figsize=(10,6))
plt.plot(omega_scan/(2*np.pi), final_p_plus1, '-o',linewidth= 2, markersize = 4)  # convert back to cycles if desired
plt.axvline(x= f_gw, color='red', linestyle='--', label=f'Input GW freq: {f_gw} Hz')
plt.xlabel('GW Frequency Hz or (cycles/unit)')
plt.ylabel('Final Population P(|+1>)')
plt.title('Frequency Scan: Final Population in |+1> vs GW Frequency')
plt.legend()
plt.grid(True)
plt.show()

#======================= Plotting Results ========================================#

#======================= Physical Analysis========================================#

print("\n===== Physical Analysis and Next Steps ======\n")
print(f" Zero-field splitting D = {D_toy} Hz")
print(f" GW frequency f_gw = {f_gw} cycles/unit)")
print(f"GW angular frequency omega_gw = {omega_gw} rad/unit")
print(f"Coupling strength kappa = {kappa} (toy value)")
print(f"GW strain amplitude: {(A_toy)}")

# Calculate approximate Rabi frequency induced by GW interaction
# Using matrix element <0|Op_plus|+1> for estimation of two-level system
matrix_element = Op_plus.matrix_element(psi_0, psi_p1) # Or written as (psi_0.dag() * Op_plus * psi_p1)
Omega_Rabi = kappa * A_toy * np.abs(matrix_element)  # Rabi frequency estimate
print(f"Estimated GW-induced Rabi frequency Omega_Rabi = {Omega_Rabi:.4}) Hz")
print(f"Corresponding Rabi period T_Rabi = {2*np.pi/Omega_Rabi:.4f} time units")
effective_coupling = np.abs(matrix_element)* A_toy
print(f" Matrix element <+1|Op_plus|0> = {matrix_element:.3e}, Effective rabi coupling frequency = {effective_coupling:.3e} Hz")

print("\n======================= End of Analysis =======================\n")

#======================= Physical Analysis========================================#
print(" Analysis complete")
print("NEXT STEPS:")
print("1. Run this code - see if GW causes population transfers")
print("2. Vary kappa to see stronger/weaker effects")  
print("3. Add decoherence (collapse operators)")
print("4. Replace placeholder kappa with FW-derived value")





