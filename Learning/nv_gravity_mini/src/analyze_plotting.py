import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.fft import fft, fftfreq

class ResultAnalyzer:
    def __init__(self, result, config):
        self.result = result
        self.cfg = config
        os.makedirs("plots", exist_ok=True)
        
        # 1. Safely get time array as NumPy array
        if hasattr(result, 'times') and len(result.times) > 0:
            self.times = np.array(result.times)  # CRITICAL: Convert to NumPy array
        else:
            self.times = np.array(self.cfg.tlist)
        
        # Verify length matches expectation data
        if len(self.result.expect) > 0:
            expected_len = len(self.result.expect[0])
            if len(self.times) != expected_len:
                print(f"⚠️ Time/data mismatch: {len(self.times)} vs {expected_len}. Truncating.")
                # Align by taking the shorter length
                min_len = min(len(self.times), expected_len)
                self.times = self.times[:min_len]
                # Also need to trim expectation arrays (handled in unpacking below)
        
        self.t_ms = self.times * 1000  # Now this works element-wise
        
        # 2. Smart Unpacking (existing code)
        # [Your existing smart unpacking code here]
        n_observables = len(result.expect)
        # ... rest of your unpacking logic
        
        if n_observables >= 6:
            # --- DYNAMICS MODE (Standard) ---
            # Order: [p0, p1, m1, Sz, Sx, Sy]
            self.p0 = result.expect[0]
            self.p1 = result.expect[1]
            self.m1 = result.expect[2]
            self.Sz = result.expect[3]
            self.Sx = result.expect[4]
            self.Sy = result.expect[5]
            self.mode = "dynamics"
            
        elif n_observables == 3:
            # --- SENSING MODE (CPMG) ---
            # Order defined in sensing.py: [sx, sy, sz]
            self.Sx = result.expect[0]
            self.Sy = result.expect[1]
            self.Sz = result.expect[2]
            # Set populations to None since we didn't track them
            self.p0 = None
            self.p1 = None
            self.m1 = None
            self.mode = "sensing"
            
        else:
            print(f"⚠️ Warning: Unexpected number of observables ({n_observables}).")
            self.mode = "unknown"

    def plot_comprehensive(self):
        """Standard dashboard for Dynamics Mode."""
        if self.mode != "dynamics":
            print("⚠️ Cannot plot comprehensive dashboard: Missing population data.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'NV-Center GW Detector | f_GW={self.cfg.f_gw:.1e} Hz', fontsize=14, fontweight='bold')

        # [Plotting logic is the same as before...]
        # 1. Populations
        ax = axes[0, 0]
        ax.plot(self.t_ms, self.p0, label=r'$|0\rangle$', color='#1f77b4')
        ax.plot(self.t_ms, self.p1, label=r'$|+1\rangle$', color='#d62728')
        ax.plot(self.t_ms, self.m1, label=r'$|-1\rangle$', color='#2ca02c')
        ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_ylabel('Population')

        # 2. Strain
        ax = axes[0, 1]
        strain = self.cfg.h_max * np.sin(self.cfg.omega_gw * self.times)
        ax.plot(self.t_ms, strain, color='purple')
        ax.set_title('GW Input Strain')
        ax.grid(True, alpha=0.3)

        # 3. Spin
        ax = axes[1, 0]
        ax.plot(self.t_ms, self.Sz, label='Sz', color='orange')
        ax.plot(self.t_ms, self.Sx, 'r--', label='Sx', alpha=0.5)
        ax.plot(self.t_ms, self.Sy, 'b--', label='Sy', alpha=0.5)
        ax.legend(); ax.grid(True, alpha=0.3)

        # 4. FFT
        ax = axes[1, 1]
        dt = self.times[1] - self.times[0]
        signal = self.p0 - np.mean(self.p0)
        fft_vals = fft(signal)
        freqs = fftfreq(len(self.times), dt)
        mask = freqs > 0
        ax.plot(freqs[mask], np.abs(fft_vals[mask]), color='teal')
        ax.set_xlim(0, self.cfg.f_gw * 4)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Frequency (Hz)')

        plt.tight_layout()
        plt.savefig("plots/comprehensive_analysis.png", dpi=150)
        print("📊 Comprehensive plot saved.")

    def plot_cpmg_results(self, result, tlist, n_pulses):
        """Specialized plot for Sensing Mode."""
        if self.mode != "sensing":
            print("⚠️ Cannot plot CPMG results: Data format incorrect.")
            return

        # --- FIX STARTS HERE ---
        # We must use the 'tlist' passed from the engine, NOT self.t_ms
        # self.t_ms is based on cfg.t_final, which is irrelevant for CPMG sensing
        t_ms = tlist * 1000 
        
        # Verify dimensions before plotting to prevent crashes
        if len(t_ms) != len(self.Sx):
            print(f"⚠️ Dimension Mismatch Fixed: Resampling time axis.")
            # Fallback: Create a new time axis that matches the data length
            t_ms = np.linspace(t_ms[0], t_ms[-1], len(self.Sx))
        # --- FIX ENDS HERE ---

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Plot Coherence (Sx)
        axes[0].plot(t_ms, self.Sx, label=r'$\langle S_x \rangle$ (Coherence)', color='blue')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title(f'CPMG-{n_pulses}: Coherence Preservation')
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        # Plot Signal (Sy)
        axes[1].plot(t_ms, self.Sy, label=r'$\langle S_y \rangle$ (Signal)', color='green', linewidth=2)
        axes[1].set_ylabel('Signal Amplitude')
        axes[1].set_xlabel('Time (ms)')
        axes[1].set_title('Quadrature Component (Phase Accumulation)')
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("plots/cpmg_analysis.png")
        print("📊 CPMG plot saved to plots/cpmg_analysis.png")