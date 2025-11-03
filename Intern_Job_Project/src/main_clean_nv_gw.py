
# main_clean_nv_gw.py
"""This code here will be translated to c++ and used for a job interview."""
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

# ==================== 1. SPIN OPERATORS & BASIS ==================== #
print("Setting up spin-1 system for NV-center...")

# Spin-1 operators
Sx = qt.jmat(1, 'x')
Sy = qt.jmat(1, 'y') 
Sz = qt.jmat(1, 'z')

# Basis states
psi_plus1 = qt.basis(3, 0)  # |m_s = +1>
psi_0 = qt.basis(3, 1)      # |m_s = 0>  
psi_minus1 = qt.basis(3, 2) # |m_s = -1>

print("✓ Spin operators and basis states defined")

# ==================== 2. PHYSICAL PARAMETERS ==================== #
D = 2.87e9           # Zero-field splitting (Hz)
gamma_e = 28e9       # Gyromagnetic ratio (Hz/T)
kappa = 1e15         # GW coupling constant (PLACEHOLDER - from FW transformation)

print(f"✓ Physical parameters: D={D/1e9:.2f} GHz, kappa={kappa:.1e}")

# ==================== 3. GRAVITATIONAL WAVE ==================== #
def h_plus(t, f_gw=100, h_max=1e-18):
    """Simple monochromatic GW - REALISTIC parameters"""
    return h_max * np.sin(2 * np.pi * f_gw * t)

print("✓ GW strain function defined")

# ==================== 4. HAMILTONIANS ==================== #
def get_hamiltonians(Bz=0.0):
    """Returns static and time-dependent Hamiltonians"""
    # Static NV Hamiltonian
    H_static = D * Sz**2
    if Bz != 0.0:
        H_static += gamma_e * Bz * Sz
    
    # GW Interaction Hamiltonian  
    H_int_operator = kappa * (Sx**2 - Sy**2)
    
    # Time-dependent Hamiltonian for QuTiP
    H_td = [H_static, [H_int_operator, h_plus]]
    
    return H_td

print("✓ Hamiltonians defined")

# ==================== 5. SIMULATION ==================== #
def run_simulation():
    """Run the complete simulation"""
    print("Starting simulation...")
    
    # Parameters
    Bz = 0.01  # 10 mT magnetic field
    t_final = 0.1  # 100 ms (to see multiple GW cycles)
    tlist = np.linspace(0, t_final, 2000)
    
    # Get Hamiltonian
    H = get_hamiltonians(Bz)
    
    # Initial state
    psi0 = psi_0
    
    # Run simulation
    result = qt.sesolve(H, psi0, tlist)
    print("✓ Time evolution complete")
    
    return result, tlist

# ==================== 6. ANALYSIS ==================== #
def analyze_results(result, tlist):
    """Analyze and plot results"""
    print("Analyzing results...")
    
    # Calculate populations
    pop_plus1 = [abs(psi_plus1.overlap(state))**2 for state in result.states]
    pop_0 = [abs(psi_0.overlap(state))**2 for state in result.states] 
    pop_minus1 = [abs(psi_minus1.overlap(state))**2 for state in result.states]
    
    # Calculate expectation values
    exp_Sz = qt.expect(Sz, result.states)
    
    # GW strain for reference
    gw_strain = [h_plus(t) for t in tlist]
    
    return pop_plus1, pop_0, pop_minus1, exp_Sz, gw_strain

# ==================== 7. VISUALIZATION ==================== #
def plot_results(tlist, pop_plus1, pop_0, pop_minus1, exp_Sz, gw_strain):
    """Create clear, interpretable plots"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Populations (MAIN RESULT)
    ax1.plot(tlist * 1000, pop_0, 'b-', linewidth=2, label='|0⟩ population')
    ax1.plot(tlist * 1000, pop_plus1, 'r-', linewidth=2, label='|+1⟩ population')
    ax1.plot(tlist * 1000, pop_minus1, 'g-', linewidth=2, label='|-1⟩ population')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Population')
    ax1.set_title('NV-Center Populations under GW Influence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: GW Strain
    ax2.plot(tlist * 1000, gw_strain, 'purple', linewidth=2)
    ax2.set_xlabel('Time (ms)') 
    ax2.set_ylabel('Strain h₊(t)')
    ax2.set_title('Gravitational Wave Signal')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Spin expectation
    ax3.plot(tlist * 1000, exp_Sz, 'orange', linewidth=2)
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('⟨S_z⟩')
    ax3.set_title('Spin Expectation Value')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Population changes (zoomed)
    ax4.plot(tlist * 1000, pop_0, 'b-', linewidth=2, label='|0⟩')
    ax4.plot(tlist * 1000, pop_plus1, 'r-', linewidth=2, label='|+1⟩')
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Population')
    ax4.set_title('GW-Induced Population Transfer')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

# ==================== 8. INTERPRETATION ==================== #
def interpret_results(pop_plus1, pop_0, pop_minus1):
    """Provide clear interpretation of what we're seeing"""
    max_transfer = max(max(pop_plus1), max(pop_minus1))
    initial_pop = pop_0[0]
    final_pop = pop_0[-1]
    
    print(f"\n📊 QUANTITATIVE RESULTS:")
    print(f"Initial |0⟩ population: {initial_pop:.6f}")
    print(f"Final |0⟩ population: {final_pop:.6f}") 
    print(f"Maximum transfer to |±1⟩ states: {max_transfer:.6f}")
    
    print(f"\n🎯 PHYSICAL INTERPRETATION:")
    if max_transfer > 0.1:
        print("➡️ STRONG EFFECT: GW causes significant population transfer")
        print("   The gravitational wave is strongly driving transitions between spin states")
    elif max_transfer > 0.01:
        print("➡️ MODERATE EFFECT: Observable population transfer") 
        print("   The GW is having a measurable effect on the spin dynamics")
    elif max_transfer > 0.001:
        print("➡️ WEAK EFFECT: Small but detectable population transfer")
        print("   The GW effect is present but requires sensitive measurement")
    else:
        print("➡️ NEGLIGIBLE EFFECT: Population transfer below detection threshold")
        print("   Try increasing kappa or using more realistic GW parameters")
    
    print(f"\n🔧 SUGGESTIONS:")
    if max_transfer < 0.01:
        print("   - Increase kappa to 1e18-1e20 (stronger coupling)")
        print("   - Use larger GW strain h_max=1e-15 to 1e-12")
        print("   - Add magnetic field to break degeneracy")
    else:
        print("   - Add decoherence with collapse operators")
        print("   - Try different GW frequencies")
        print("   - Use real LIGO data")

# ==================== 9. MAIN EXECUTION ==================== #
if __name__ == "__main__":
    print("🚀 NV-CENTER GRAVITATIONAL WAVE SIMULATION")
    print("=" * 50)
    
    # Run everything
    result, tlist = run_simulation()
    pop_plus1, pop_0, pop_minus1, exp_Sz, gw_strain = analyze_results(result, tlist)
    plot_results(tlist, pop_plus1, pop_0, pop_minus1, exp_Sz, gw_strain)
    interpret_results(pop_plus1, pop_0, pop_minus1)
    
    print(f"\n✅ SIMULATION COMPLETE!")
    print("Next: Derive actual kappa from Foldy-Wouthuysen transformation")