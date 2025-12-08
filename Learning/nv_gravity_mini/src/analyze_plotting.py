from asyncio.log import logger
from typing import Any, Dict
from unittest import result
from matplotlib.pylab import fftfreq
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from scipy import fft
from traitlets import Tuple
from nv_gravity_mini.Main_qutip_ver4 import SimulationEngine

class ResultAnalyzer:
    """"" Handle analysis and plotting of results """

    def __init__(self, simulation_engine):
        self.engine = simulation_engine
        self.system = simulation_engine.system

        def analyze_results(self, result: qt.Result) -> Dict[str, Any]:
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
        """ Professional population plotting now"""
        if self.engine.results is None:
            raise ValueError("No results to analyze. Run the simulation first!")
        
        p_p1, p_0, p_m1 = populations
        populations, exp_Sz, exp_Sx, exp_Sy = self.engine.results.expect
        self.tlist = np.linspace(0,self.system.cfg.t_final, len(p_0))
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
    
def demonstrate_detector():
    """Complete demonstration"""
    print(" ULTIMATE NV-CENTER GRAVITATIONAL WAVE DETECTOR")
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
    detector = Main
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
    