# sensitivity_estimator.py
import numpy as np
import matplotlib.pyplot as plt

def estimate_gw_sensitivity(strain_values):
    """Connects your C++ simulator to thesis"""
    # Call your compiled C++ module
    from core import nv_sensitivity  # type: ignore
    signals = [nv_sensitivity.calculate(h) for h in strain_values]
    
    plt.plot(strain_values, signals)
    plt.xlabel('Gravitational Wave Strain')
    plt.ylabel('NV-Center Response (Hz)')
    plt.savefig('thesis/gw_sensitivity.png')