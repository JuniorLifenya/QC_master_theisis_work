from src.config import SimulationConfig
from src.nv_quantum_setup import NVCenter
from src.Engin_timeStepper import SimulationEngine
from src.analyze_plotting import ResultAnalyzer
from src.cmpg_sensing import SensingEngine
import numpy as np

import numpy as np
import logging


# --- CONFIGURATION SWITCH ---
# Options: "dynamics" (Original Rabi/Population) OR "sensing" (CPMG Sequence)
MODE = "dynamics"  # Change to "dynamics" for original simulation
# ----------------------------

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    print(f" Initializing NV-GW Simulation in [{MODE.upper()}] mode...")

    # 1. Setup Common Physics (The NV Center)
    # Note: Adjust parameters based on the mode if needed
    if MODE == "dynamics":
        # Parameters for Direct Population Transfer (High Frequency, Resonance)
        cfg = SimulationConfig(f_gw=1000, h_max=1e-10, n_steps=3000, t_final=0.01)
    else:
        # Parameters for Sensing (Low Frequency, Long Coherence)
        cfg = SimulationConfig(f_gw=50e3, h_max=0.5, n_steps=5000)

    nv = NVCenter(cfg)

    # 2. SELECT THE ENGINE
    if MODE == "dynamics":
        # --- PATH A: Original Simulation ---
        engine = SimulationEngine(nv)
        result = engine.run()
        
        # Analyze
        analyzer = ResultAnalyzer(result, cfg)
        analyzer.plot_comprehensive()
        
        # Optional: Run Frequency Scan
        # freqs = np.linspace(500, 1500, 20)
        # scan_x, scan_y = engine.frequency_scan(freqs)
        # analyzer.plot_frequency_scan(scan_x, scan_y)

    elif MODE == "sensing":
        # --- PATH B: CPMG Sensing Sequence ---
        # We initialize the SensingEngine (which inherits from SimulationEngine)
        engine = SensingEngine(nv, n_pulses=8)
        result, tlist = engine.run_cpmg()
        
        # Analyze
        analyzer = ResultAnalyzer(result, cfg)
        analyzer.plot_cpmg_results(result, tlist, engine.n_pulses)

    print(f"✅ {MODE.upper()} simulation completed successfully.")

if __name__ == "__main__":
    main()