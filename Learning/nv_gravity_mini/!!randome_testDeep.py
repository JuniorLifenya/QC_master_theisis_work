"""
ULTIMATE NV-CENTER GRAVITATIONAL WAVE DETECTOR
Master Thesis Level - Optimized & Robust

FIXED VERSION: Corrected matrix element calculation
"""

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import logging
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("NVGWDetector")

@dataclass
class NVGWParameters:
    """Clean parameter management using dataclass"""
    # NV Center Parameters
    D: float = 2.87e9                    # Zero-field splitting (Hz)
    gamma_e: float = 28e9                # Gyromagnetic ratio (Hz/T)
    Bz: float = 0.01                     # Magnetic field (T)
    
    # GW Parameters
    f_gw: float = 1000.0                 # GW frequency (Hz)
    h_max: float = 1e-6                  # GW strain amplitude
    kappa: float = 1e10                  # Coupling constant
    
    # Simulation Parameters
    t_final: float = 0.001               # Simulation time (s)
    nsteps: int = 5000                   # Time steps
    use_mesolve: bool = False            # Use open system evolution
    
    # Decoherence Parameters
    T1: Optional[float] = 1e-3           # Relaxation time (s)
    T2: Optional[float] = 500e-6         # Dephasing time (s)
    
    # Output Options
    save_animation: bool = False
    demo_mode: bool = True               # Enhanced parameters for visibility
    
    def __post_init__(self):
        """Post-initialization validation and adjustments"""
        if self.demo_mode:
            logger.info("Demo mode: Enhancing parameters for visibility")
            self.h_max *= 1e6
            self.kappa *= 1e12
        
        self.omega_gw = 2 * np.pi * self.f_gw
        
        # Calculate derived parameters
        self.gamma_T1 = 1.0 / self.T1 if self.T1 else 0
        self.gamma_T2 = 1.0 / self.T2 if self.T2 else 0

class UltimateNVGWDetector:
    """
    Optimized NV-Center Gravitational Wave Detector
    With FIXED matrix element calculation
    """
    
    def __init__(self, params: NVGWParameters):
        self.p = params
        self.setup_quantum_operators()
        self.setup_analysis_tools()
        
    def setup_quantum_operators(self):
        """Initialize quantum operators and basis states"""
        # Spin-1 operators
        self.Sx = qt.jmat(1, 'x')
        self.Sy = qt.jmat(1, 'y') 
        self.Sz = qt.jmat(1, 'z')
        
        # Basis states
        self.psi_p1 = qt.basis(3, 0)   # |+1⟩
        self.psi_0 = qt.basis(3, 1)    # |0⟩  
        self.psi_m1 = qt.basis(3, 2)   # |-1⟩
        
        # GW interaction operator
        self.Op_plus = self.Sx**2 - self.Sy**2
        
    def setup_analysis_tools(self):
        """Initialize analysis parameters"""
        self.colors = {
            'p0': '#1f77b4', 'p1': '#d62728', 'm1': '#2ca02c',
            'gw': '#9467bd', 'sz': '#ff7f0e', 'bg': '#f8f9fa'
        }
    
    def gw_strain(self, t: float, args: Dict) -> float:
        """GW strain function compatible with QuTiP"""
        return args['h_max'] * np.sin(args['omega_gw'] * t)
    
    def get_hamiltonian(self) -> list:
        """Construct time-dependent Hamiltonian"""
        H_static = self.p.D * self.Sz**2 + self.p.gamma_e * self.p.Bz * self.Sz
        H_int = self.p.kappa * self.Op_plus
        
        return [H_static, [H_int, self.gw_strain]]
    
    def get_collapse_operators(self) -> list:
        """Build collapse operators for decoherence"""
        if not self.p.use_mesolve or (not self.p.T1 and not self.p.T2):
            return []
            
        c_ops = []
        
        # T2 dephasing
        if self.p.T2:
            c_ops.append(np.sqrt(self.p.gamma_T2) * self.Sz)
        
        # T1 relaxation
        if self.p.T1:
            c_ops.append(np.sqrt(self.p.gamma_T1) * (self.psi_0 * self.psi_p1.dag()))
            c_ops.append(np.sqrt(self.p.gamma_T1) * (self.psi_0 * self.psi_m1.dag()))
            
        return c_ops
    
    def get_observables(self) -> list:
        """Define measurement observables"""
        return [
            self.psi_p1 * self.psi_p1.dag(),  # P(|+1⟩)
            self.psi_0 * self.psi_0.dag(),    # P(|0⟩)
            self.psi_m1 * self.psi_m1.dag(),  # P(|-1⟩)
            self.Sz,                          # ⟨Sz⟩
            self.Sx,                          # ⟨Sx⟩  
            self.Sy                           # ⟨Sy⟩
        ]
    
    def run_simulation(self) -> Tuple[qt.Result, np.ndarray]:
        """Run the quantum simulation with robust error handling"""
        logger.info("🚀 Starting quantum simulation...")
        
        # Setup
        self.tlist = np.linspace(0, self.p.t_final, self.p.nsteps)
        H = self.get_hamiltonian()
        e_ops = self.get_observables()
        c_ops = self.get_collapse_operators()
        
        args = {'h_max': self.p.h_max, 'omega_gw': self.p.omega_gw}
        
        # Solver options for numerical stability
        options = {
            'nsteps': 1000000,
            'atol': 1e-12,
            'rtol': 1e-10,
            'max_step': self.p.t_final / 1000,
            'progress_bar': True
        }
        
        logger.info(f"Simulation: {self.p.t_final*1000:.1f} ms, {self.p.nsteps} steps")
        logger.info(f"GW: f={self.p.f_gw} Hz, h={self.p.h_max:.1e}")
        logger.info(f"Decoherence: {len(c_ops)} collapse operators")
        
        # Run simulation
        if self.p.use_mesolve and c_ops:
            rho0 = self.psi_0 * self.psi_0.dag()
            result = qt.mesolve(H, rho0, self.tlist, c_ops, e_ops, args=args, options=options)
        else:
            result = qt.sesolve(H, self.psi_0, self.tlist, e_ops, args=args, options=options)
        
        logger.info("✅ Simulation completed successfully")
        return result, self.tlist
    
    def calculate_matrix_element(self) -> float:
        """FIXED: Safely calculate matrix element <+1|H_int|0>"""
        H_int = self.p.kappa * self.Op_plus
        
        try:
            # Method 1: Direct calculation
            matrix_element_qobj = self.psi_p1.dag() * H_int * self.psi_0
            
            # Check if it's a Qobj or already a scalar
            if hasattr(matrix_element_qobj, 'full'):
                matrix_element = np.abs(matrix_element_qobj.full()[0,0])
            else:
                # If it's already a scalar (complex number)
                matrix_element = np.abs(matrix_element_qobj)
                
        except Exception as e:
            logger.warning(f"Matrix element calculation failed: {e}")
            # Method 2: Use QuTiP's expect function as fallback
            matrix_element = np.abs(qt.expect(H_int, self.psi_p1, self.psi_0))
        
        return matrix_element
    
    def calculate_physical_metrics(self, populations: Tuple) -> Dict[str, float]:
        """Calculate key physical metrics with FIXED matrix element"""
        p_p1, p_0, p_m1 = populations
        
        # Population metrics
        max_transfer = np.max(p_p1) + np.max(p_m1) - (p_p1[0] + p_m1[0])
        oscillation_amp = np.std(p_0)
        
        # FIXED: Use safe matrix element calculation
        matrix_element = self.calculate_matrix_element()
        
        # Rabi frequency
        rabi_freq = matrix_element * self.p.h_max / (2 * np.pi)
        
        # SNR estimate
        noise = 1.0 / np.sqrt(self.p.T2 * self.p.f_gw) if self.p.T2 else 1e-3
        snr = max_transfer / noise if noise > 0 else 0
        
        metrics = {
            'max_transfer': max_transfer,
            'oscillation_amp': oscillation_amp,
            'matrix_element': matrix_element,
            'rabi_frequency': rabi_freq,
            'snr': snr
        }
        
        logger.info(f"📊 Max transfer: {max_transfer:.2e}")
        logger.info(f"📊 Matrix element: {matrix_element:.3e}")
        logger.info(f"📊 Rabi frequency: {rabi_freq:.3e} Hz")
        logger.info(f"📊 SNR: {snr:.3f}")
        
        return metrics
    
    def analyze_results(self, result: qt.Result) -> Dict[str, Any]:
        """Comprehensive results analysis"""
        logger.info("🔬 Analyzing results...")
        
        # Extract expectation values
        p_p1, p_0, p_m1, exp_Sz, exp_Sx, exp_Sy = result.expect
        
        # Calculate GW strain for reference
        gw_strain = [self.gw_strain(t, {'h_max': self.p.h_max, 'omega_gw': self.p.omega_gw}) 
                    for t in self.tlist]
        
        # Calculate physical metrics
        metrics = self.calculate_physical_metrics((p_p1, p_0, p_m1))
        
        return {
            'populations': (p_p1, p_0, p_m1),
            'expectations': (exp_Sz, exp_Sx, exp_Sy),
            'gw_strain': gw_strain,
            'time': self.tlist,
            'metrics': metrics
        }
    
    def plot_comprehensive(self, results: Dict[str, Any]):
        """Professional visualization - optimized 2x2 layout"""
        logger.info("📈 Generating visualizations...")
        
        p_p1, p_0, p_m1 = results['populations']
        exp_Sz, exp_Sx, exp_Sy = results['expectations']
        gw_strain = results['gw_strain']
        metrics = results['metrics']
        
        t_ms = self.tlist * 1000
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'NV-Center GW Detector | f_GW={self.p.f_gw} Hz, h={self.p.h_max:.1e}', 
                    fontsize=14, fontweight='bold')
        
        # Plot 1: Populations (Main result)
        axes[0,0].plot(t_ms, p_0, color=self.colors['p0'], linewidth=2, label='$P(|0\\rangle)$')
        axes[0,0].plot(t_ms, p_p1, color=self.colors['p1'], linewidth=2, label='$P(|+1\\rangle)$')
        axes[0,0].plot(t_ms, p_m1, color=self.colors['m1'], linewidth=2, label='$P(|-1\\rangle)$')
        axes[0,0].set_xlabel('Time (ms)'); axes[0,0].set_ylabel('Population')
        axes[0,0].set_title('GW-Driven Population Transfer')
        axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: GW Strain
        axes[0,1].plot(t_ms, gw_strain, color=self.colors['gw'], linewidth=2)
        axes[0,1].set_xlabel('Time (ms)'); axes[0,1].set_ylabel('Strain $h_+(t)$')
        axes[0,1].set_title('Gravitational Wave Input')
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Spin expectations
        axes[1,0].plot(t_ms, exp_Sz, color=self.colors['sz'], linewidth=2, label='$\\langle S_z \\rangle$')
        axes[1,0].plot(t_ms, exp_Sx, 'r--', linewidth=1, alpha=0.7, label='$\\langle S_x \\rangle$')
        axes[1,0].plot(t_ms, exp_Sy, 'b--', linewidth=1, alpha=0.7, label='$\\langle S_y \\rangle$')
        axes[1,0].set_xlabel('Time (ms)'); axes[1,0].set_ylabel('Spin Expectation')
        axes[1,0].set_title('Spin Dynamics'); axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Frequency spectrum
        dt = self.tlist[1] - self.tlist[0]
        fft_p0 = fft(p_0 - np.mean(p_0))
        freqs = fftfreq(len(self.tlist), dt)
        positive_idx = freqs > 0
        
        axes[1,1].plot(freqs[positive_idx], np.abs(fft_p0[positive_idx]), 'teal', linewidth=2)
        axes[1,1].axvline(x=self.p.f_gw, color='red', linestyle='--', label=f'GW: {self.p.f_gw} Hz')
        if metrics['rabi_frequency'] > 0:
            axes[1,1].axvline(x=metrics['rabi_frequency'], color='orange', linestyle='--', 
                            label=f'Rabi: {metrics["rabi_frequency"]:.1f} Hz')
        axes[1,1].set_xlabel('Frequency (Hz)'); axes[1,1].set_ylabel('FFT Amplitude')
        axes[1,1].set_title('Frequency Spectrum'); axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def generate_report(self, metrics: Dict[str, float]):
        """Professional detection assessment report"""
        print("\n" + "="*60)
        print("🔍 GRAVITATIONAL WAVE DETECTION REPORT")
        print("="*60)
        
        print(f"\nKEY METRICS:")
        print(f"  Population transfer: {metrics['max_transfer']:.2e}")
        print(f"  Matrix element:      {metrics['matrix_element']:.3e}")
        print(f"  Rabi frequency:      {metrics['rabi_frequency']:.3e} Hz")
        print(f"  Signal-to-Noise:     {metrics['snr']:.3f}")
        
        print(f"\nDETECTION ASSESSMENT:")
        if metrics['snr'] > 5:
            print("  ✅ STRONG DETECTION CANDIDATE")
            print("     - GW signal significantly above noise floor")
        elif metrics['snr'] > 1:
            print("  ⚠️  MARGINAL DETECTION")
            print("     - GW signal near detection threshold")
        else:
            print("  ❌ BELOW DETECTION THRESHOLD")
            print("     - GW signal too weak for current setup")
        
        print(f"\nRECOMMENDATIONS:")
        if self.p.demo_mode:
            print("  1. Switch to realistic parameters for actual sensitivity analysis")
        if metrics['snr'] < 5:
            print("  2. Consider quantum enhancement techniques")
            print("  3. Increase measurement integration time")
    
    def frequency_scan(self, freq_range: np.ndarray, observable: str = "p1") -> Tuple[np.ndarray, np.ndarray]:
        """Frequency scan utility"""
        logger.info(f"🔍 Scanning {len(freq_range)} frequencies...")
        
        original_freq = self.p.f_gw
        results = []
        
        for i, freq in enumerate(freq_range):
            if i % 10 == 0:
                logger.info(f"  Progress: {i+1}/{len(freq_range)}")
            
            # Update frequency
            self.p.f_gw = freq
            self.p.omega_gw = 2 * np.pi * freq
            
            # Run simulation
            result, _ = self.run_simulation()
            p_p1, p_0, p_m1 = result.expect[0:3]
            
            # Store final population
            if observable == "p1":
                results.append(np.real(p_p1[-1]))
            elif observable == "0":
                results.append(np.real(p_0[-1]))
            else:
                results.append(np.real(p_m1[-1]))
        
        # Restore original frequency
        self.p.f_gw = original_freq
        self.p.omega_gw = 2 * np.pi * original_freq
        
        return freq_range, np.array(results)

# ==================== DEMONSTRATION ====================

def demonstrate_detector():
    """Complete demonstration"""
    print("🚀 ULTIMATE NV-CENTER GRAVITATIONAL WAVE DETECTOR")
    print("    FIXED VERSION - Matrix Element Calculation")
    print("=" * 55)
    
    # Setup parameters
    params = NVGWParameters(
        f_gw=1000.0,           # GW frequency
        h_max=1e-6,            # GW strain  
        kappa=1e10,            # Coupling
        Bz=0.01,               # Magnetic field
        t_final=0.001,         # Simulation time
        nsteps=5000,           # Time steps
        use_mesolve=False,     # Pure state evolution
        demo_mode=True         # Enhanced visibility
    )
    
    # Create and run detector
    detector = UltimateNVGWDetector(params)
    result, tlist = detector.run_simulation()
    analysis = detector.analyze_results(result)
    
    # Generate outputs
    detector.plot_comprehensive(analysis)
    detector.generate_report(analysis['metrics'])
    
    # Optional: Frequency scan
    print("\n🎯 Performing frequency scan...")
    freqs = np.linspace(500, 1500, 20)  # Scan around GW frequency
    scan_freqs, scan_results = detector.frequency_scan(freqs, "p1")
    
    plt.figure(figsize=(10, 5))
    plt.plot(scan_freqs, scan_results, 'o-', linewidth=2)
    plt.axvline(x=params.f_gw, color='red', linestyle='--', label=f'Original: {params.f_gw} Hz')
    plt.xlabel('GW Frequency (Hz)'); plt.ylabel('Final P(|+1⟩)')
    plt.title('Frequency Scan: Resonance Behavior'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\n🎉 DEMONSTRATION COMPLETE")

if __name__ == "__main__":
    demonstrate_detector()