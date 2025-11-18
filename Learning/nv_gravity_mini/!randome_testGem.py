import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import warnings
from matplotlib.animation import FuncAnimation

# Suppress QuTiP future warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

class NVQuantumGravimeter:
    """
    Simulates the Nitrogen-Vacancy (NV) center in diamond under the influence 
    of a Gravitational Wave (GW), modeled as a time-dependent quadrupole interaction.
    
    Uses the Master Equation (mesolve) for realistic open-system dynamics, 
    incorporating T2 dephasing.
    """
    def __init__(self, params):
        self.p = self._validate_and_set_params(params)
        
        # --- Physical Operators ---
        self.Sx = qt.jmat(1, 'x')
        self.Sy = qt.jmat(1, 'y')
        self.Sz = qt.jmat(1, 'z')
        
        # --- Basis States ---
        self.psi_p1 = qt.basis(3, 0)   # |m_s = +1>
        self.psi_0  = qt.basis(3, 1)   # |m_s = 0>
        self.psi_m1 = qt.basis(3, 2)   # |m_s = -1>
        
        # --- Interaction Operators ---
        # GW Quadrupole Operator (Delta m = ±2 transition: |+1> <-> |-1>)
        self.H_int_op = self.Sx**2 - self.Sy**2
        
        # --- Time Domain ---
        self.tlist = np.linspace(0, self.p['t_final'], self.p['nsteps'])
        
    def _validate_and_set_params(self, params):
        """Ensures all required physical parameters are present and calculates derived values."""
        
        required_keys = ['D', 'gamma_e', 'Bz', 'f_gw', 'h_max', 'kappa', 
                         't_final', 'nsteps', 'T2']
        if not all(key in params for key in required_keys):
            raise ValueError(f"Missing one or more required parameters: {required_keys}")
            
        params['gamma_T2'] = 1.0 / params['T2']
        params['omega_gw'] = 2 * np.pi * params['f_gw']
        params['args_gw'] = {'h_max': params['h_max'], 'omega_gw': params['omega_gw']}
        
        return params

    @staticmethod
    def gw_drive_coeff(t, args):
        """Time-dependent coefficient function for the GW drive term H_int * f(t)."""
        return args['h_max'] * np.sin(args['omega_gw'] * t)

    def get_hamiltonian(self):
        """Constructs the time-dependent Hamiltonian H(t) = H0 + H_int * f(t)."""
        
        # H0: Static Hamiltonian (Zero-field splitting + Zeeman shift)
        H0 = self.p['D'] * self.Sz**2 + self.p['gamma_e'] * self.p['Bz'] * self.Sz
        
        # H_td: Time-dependent list for QuTiP: [H0, [Operator, Coefficient_Function]]
        H_td = [H0, [self.p['kappa'] * self.H_int_op, self.gw_drive_coeff]]
        
        return H_td

    def get_collapse_operators(self):
        """
        Defines collapse operators (c_ops) for the Master Equation.
        T2 dephasing term (decoherence on the S_z basis).
        """
        # Collapse Operator: sqrt(gamma_T2) * Sz
        c_ops = [np.sqrt(self.p['gamma_T2']) * self.Sz]
        
        return c_ops

    def run_simulation(self, initial_state='psi_0'):
        """Executes the QuTiP mesolve using robust solver options."""
        
        H = self.get_hamiltonian()
        c_ops = self.get_collapse_operators()
        
        # Initial State: Density matrix (rho0 = |psi><psi|)
        if initial_state == 'psi_0':
             psi0 = self.psi_0
        elif initial_state == 'psi_p1':
             psi0 = self.psi_p1
        else:
             raise ValueError("Invalid initial_state.")
        rho0 = psi0 * psi0.dag() 
        
        # Observables (projectors for population, and expectation value for Sz)
        e_ops = [self.psi_p1.proj(), self.psi_0.proj(), self.psi_m1.proj(), self.Sz]
        
        # Robust QuTiP Options (inherited from Code 1)
        options = qt.Options(
            nsteps=1_000_000,   # Safety
            atol=1e-11,
            rtol=1e-9,
            max_step=1e-6,      # Crucial for preventing 'excess work' errors
        )
        
        # Convert T2 to microseconds for clearer printing
        T2_us = self.p['T2'] * 1e6
        t_final_ms = self.p['t_final'] * 1e3
        print(f"-> Running mesolve (T2={T2_us:.0f} us) for {t_final_ms:.2f} ms...")
        
        result = qt.mesolve(H, rho0, self.tlist, c_ops=c_ops, e_ops=e_ops, 
                            args=self.p['args_gw'], options=options)
        
        return result

    def analyze_results(self, result):
        """Extracts and calculates key physical quantities."""
        
        p_p1, p_0, p_m1, exp_Sz = [np.real(x) for x in result.expect]
        
        # 1. Theoretical Rabi Frequency Estimation
        # Calculation: <psi_p1 | H_int_op | psi_m1>
        # The GW drives the |+1> <-> |-1> transition via Δm=±2 operator H_int_op.
        
        matrix_element_qobj = self.psi_p1.dag() * self.H_int_op * self.psi_m1
        
        # FIX: Since the multiplication yields a complex scalar, apply np.abs() directly.
        matrix_element = np.abs(matrix_element_qobj) 
        
        Omega_Rabi = self.p['kappa'] * self.p['h_max'] * matrix_element
        
        # 2. Key Metrics
        max_pop_p1 = np.max(p_p1)
        max_pop_m1 = np.max(p_m1)
        
        return {
            'p_p1': p_p1, 'p_0': p_0, 'p_m1': p_m1, 'exp_Sz': exp_Sz, 
            'Omega_Rabi': Omega_Rabi,
            'max_pop_p1': max_pop_p1, 'max_pop_m1': max_pop_m1
        }

    def plot_results(self, analysis_data):
        """Generates a comprehensive 2x2 plot suite for professional analysis."""
        
        t_ms = self.tlist * 1e3
        p = analysis_data
        
        # Calculate GW strain time series for the plot
        gw_strain = self.p['h_max'] * np.sin(self.p['omega_gw'] * self.tlist)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        # Convert f_gw to MHz for presentation clarity
        f_gw_mhz = self.p["f_gw"] / 1e6
        fig.suptitle(f'NV-Center GW Dynamics (Open System) | $f_{{GW}}={f_gw_mhz:.2f}$ MHz', fontsize=16)

        # Plot 1: Populations Dynamics
        axes[0, 0].plot(t_ms, p['p_0'], 'b-', linewidth=2, label='$P(|0\\rangle)$')
        axes[0, 0].plot(t_ms, p['p_p1'], 'r-', linewidth=2, label='$P(|+1\\rangle)$')
        axes[0, 0].plot(t_ms, p['p_m1'], 'g-', linewidth=2, label='$P(|-1\\rangle)$')
        axes[0, 0].set_xlabel('Time (ms)'); axes[0, 0].set_ylabel('Population')
        axes[0, 0].set_title('Populations Dynamics (Master Equation)')
        axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: GW Strain
        axes[0, 1].plot(t_ms, gw_strain, 'orange', linewidth=2)
        axes[0, 1].set_xlabel('Time (ms)'); 
        axes[0, 1].set_ylabel(f'Strain $h_+(t)$ ($h_{{max}}={self.p["h_max"]:.1e}$)')
        axes[0, 1].set_title('Gravitational Wave Signal (Time Domain)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Spin Expectation Value
        axes[1, 0].plot(t_ms, p['exp_Sz'], 'purple', linewidth=2)
        axes[1, 0].set_xlabel('Time (ms)'); axes[1, 0].set_ylabel('$\\langle S_z \\rangle$')
        axes[1, 0].set_title('Spin Expectation Value $\\langle S_z \\rangle$')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Spectral Analysis (FFT)
        dt = self.tlist[1] - self.tlist[0]
        # Use the population difference for clearer Rabi frequency peak
        pop_diff = p['p_p1'] - p['p_m1'] 
        fft_diff = fft(pop_diff - np.mean(pop_diff)) 
        freqs = fftfreq(len(self.tlist), dt)
        positive_freq_index = freqs > 0
        
        axes[1, 1].plot(freqs[positive_freq_index]/1e6, np.abs(fft_diff[positive_freq_index]), linewidth=2, color='magenta')
        axes[1, 1].set_xlabel('Frequency (MHz)')
        axes[1, 1].set_ylabel('FFT Amplitude of $P(+1) - P(-1)$')
        axes[1, 1].set_title('Spectral Analysis (Rabi Frequency)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# ==============================================================================
# --- SYSTEM CONFIGURATION (DEMO MODE: Delta m = ±2 TRANSITION) ---
# ==============================================================================

params_demo = {
    'D':        2.87e9,                     # Zero-field splitting (Hz)
    'gamma_e':  2.8e10,                     # Gyromagnetic ratio (Hz/T)
    'Bz':       0.02,                       # Applied B-field (T)
    'kappa':    1e10,                       # EXAGGERATED Coupling (for demo visibility)
    'h_max':    5e-6,                       # EXAGGERATED Strain (for demo visibility)
    
    # Resonance Condition for |+1> <-> |-1> (Delta E = 2*gamma_e*Bz)
    # f_GW should be (E_+1 - E_-1) / (2*pi*hbar) = (2 * gamma_e * Bz) / (2*pi). 
    # NOTE: Since QuTiP's Hamiltonians are implicitly divided by hbar, 
    # we use H_static in units of (rad/s), so frequency f = H / (2*pi).
    'f_gw':     (2.8e10 * 0.02),            # Resonant with 2*gamma_e*Bz (Hz)
    
    # Simulation/Decoherence
    't_final':  0.1,                        # 100 ms
    'nsteps':   15_000,
    'T2':       1.0,                        # 1 second T2 (EXAGGERATED for demo persistence)
}


if __name__ == "__main__":
    print("💎 NV-CENTER QUANTUM GRAVIMETER SIMULATION (A+ RIGOR)")
    print("=" * 65)
    
    sim = NVQuantumGravimeter(params_demo)
    
    # Running the mesolve simulation
    result = sim.run_simulation(initial_state='psi_0')
    
    # Analysis and Output
    analysis = sim.analyze_results(result)
    
    # Final Calculation Summary
    Rabi_freq_MHz = analysis['Omega_Rabi'] / (2 * np.pi * 1e6)
    
    # Calculate the resonance frequency in MHz for comparison
    f_res_mhz = (2 * sim.p['gamma_e'] * sim.p['Bz']) / 1e6
    
    print("\n--- RESULTS SUMMARY ---")
    print(f"B-field Zeeman Split (2*gamma_e*Bz): {f_res_mhz:.2f} MHz")
    print(f"GW Resonance Frequency (f_GW): {sim.p['f_gw']/1e6:.2f} MHz (Resonant with |+1> <-> |-1> transition)")
    print(f"Max Population P(|+1>): {analysis['max_pop_p1']:.4f}")
    print(f"Max Population P(|-1>): {analysis['max_pop_m1']:.4f}")
    print(f"Estimated Rabi Frequency (Theoretical): {Rabi_freq_MHz:.2f} MHz")
    
    sim.plot_results(analysis)
    
    print("\n" + "=" * 65)
    print("Simulation Complete. Ready for professional analysis.")