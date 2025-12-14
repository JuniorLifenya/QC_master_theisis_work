import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.fft import fft, fftfreq

class ResultAnalyzer:
    def __init__(self, result, config):
        self.result = result
        self.cfg = config
        os.makedirs("plots", exist_ok=True)
        
        
        # 1. Force the time axis to be a Numpy array immediately.
        # This prevents the "list repetition" bug (3000 vs 3000000).
        if hasattr(result, 'times') and len(result.times) > 0:
            self.times = np.array(result.times)
        else:
            self.times = np.array(self.cfg.tlist)
            
        self.t_ms = self.times * 1000  # Now this safely multiplies values

        # 2. Smart Unpacking: Check what kind of data we have
        n_observables = len(result.expect)
        
        if n_observables >= 6:
            # --- DYNAMICS MODE (Standard) ---
            self.p0 = result.expect[0]
            self.p1 = result.expect[1]
            self.m1 = result.expect[2]
            self.Sz = result.expect[3]
            self.Sx = result.expect[4]
            self.Sy = result.expect[5]
            self.mode = "dynamics"
            
        elif n_observables == 3:
            # --- SENSING MODE (CPMG) ---
            self.Sx = result.expect[0]
            self.Sy = result.expect[1]
            self.Sz = result.expect[2]
            # Set populations to None since we didn't track them
            self.p0 = None; self.p1 = None; self.m1 = None
            self.mode = "sensing"
            
        else:
            print(f"⚠️ Warning: Unexpected number of observables ({n_observables}).")
            self.mode = "unknown"

    def plot_comprehensive(self):
        """Standard dashboard for Dynamics Mode."""
        if self.mode != "dynamics":
            print(" Cannot plot comprehensive dashboard: Missing population data.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'NV-Center GW Detector | f_GW={self.cfg.f_gw:.1e} Hz', fontsize=14, fontweight='bold')

        # 1. Populations
        ax = axes[0, 0]
        # Safety Check: Ensure x and y match length
        t_plot = self.t_ms if len(self.t_ms) == len(self.p0) else np.linspace(0, self.t_ms[-1], len(self.p0))
        
        ax.plot(t_plot, self.p0, label=r'$|0\rangle$', color='#1f77b4', linewidth=2)
        ax.plot(t_plot, self.p1, label=r'$|+1\rangle$', color='#d62728', linewidth=2)
        ax.plot(t_plot, self.m1, label=r'$|-1\rangle$', color='#2ca02c', linewidth=2)
        ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_ylabel('Population')
        ax.set_title('Population Dynamics')

        # 2. GW Strain
        ax = axes[0, 1]
        strain = self.cfg.h_max * np.sin(self.cfg.omega_gw * self.times)
        # Resize strain if time axis was resized
        if len(strain) != len(t_plot): strain = np.resize(strain, len(t_plot))
        
        ax.plot(t_plot, strain, color='purple', linewidth=1.5)
        ax.set_title('GW Input Strain')
        ax.set_ylabel(r'Strain $h(t)$')
        ax.grid(True, alpha=0.3)

        # 3. Spin Components
        ax = axes[1, 0]
        ax.plot(t_plot, self.Sz, label='Sz', color='orange', linewidth=2)
        ax.plot(t_plot, self.Sx, 'r--', label='Sx', alpha=0.7)
        ax.plot(t_plot, self.Sy, 'b--', label='Sy', alpha=0.7)
        ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_title('Spin Expectations')

        # 4. Frequency Spectrum (FFT)
        ax = axes[1, 1]
        dt = self.times[1] - self.times[0]
        signal = self.p0 - np.mean(self.p0)
        fft_vals = fft(signal)
        freqs = fftfreq(len(self.times), dt)
        mask = freqs > 0
        
        ax.plot(freqs[mask], np.abs(fft_vals[mask]), color='teal', linewidth=2)
        ax.set_xlim(0, self.cfg.f_gw * 4) # Zoom in
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_title('System Response Spectrum')

        plt.tight_layout()
        plt.savefig("plots/comprehensive_analysis.png", dpi=150)
        plt.show()
        print(" Comprehensive plot saved to plots/comprehensive_analysis.png")

    def plot_cpmg_results(self, result, tlist, n_pulses):
        """Specialized plot for Sensing Mode."""
        if self.mode != "sensing":
            print(" Cannot plot CPMG results: Data format incorrect.")
            return

        # Use the specific time list passed from the sensing engine
        t_ms = np.array(tlist) * 1000 
        
        # Safety Check
        if len(t_ms) != len(self.Sx):
            t_ms = np.linspace(t_ms[0], t_ms[-1], len(self.Sx))

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Plot Coherence
        axes[0].plot(t_ms, self.Sx, label=r'$\langle S_x \rangle$ (Coherence)', color='blue')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title(f'CPMG-{n_pulses} Sequence: Coherence')
        axes[0].legend(loc="upper right"); axes[0].grid(True, alpha=0.3)

        # Plot Signal (Sy)
        axes[1].plot(t_ms, self.Sy, label=r'$\langle S_y \rangle$ (Signal)', color='green', linewidth=2)
        axes[1].set_ylabel('Signal Amplitude')
        axes[1].set_xlabel('Time (ms)')
        axes[1].set_title('Detected Signal (Phase Accumulation)')
        axes[1].legend(loc="upper right"); axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("plots/cpmg_analysis.png")
        plt.show()
        printCPMG plot saved to plots/cpmg_analysis.png")
