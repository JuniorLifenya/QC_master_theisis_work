import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from config import SimulateParameters
from nv_quantum_setup import QuantumSystem
from nv_quantum_setup import NVCenter

class ResultAnalyzer:
    """"" Handle analysis and plotting of results """

    def __init__(self, simulation_engine):
        self.engine = simulation_engine
        self.system = simulation_engine.system


    def plot_populations(self):
        """ Professional population plotting now"""
        if self.engine.results is None:
            raise ValueError("No results to analyze. Run the simulation first!")
        
        
        p_p1,p_0,p_m1, exp_sz, exp_sx, exp_sy = self.engine.results.expect
        tlist = np.linspace(0,self.system.cfg.t_final, len(p_0))

        fig,ax = plt.subplots(figsize =(10,6))
        ax.plot(tlist* 1e3, p_0,label = '|0>', linewidth=2)
        ax.plot(tlist* 1e3, p_p1,label = '|+1>', linewidth=2)
        ax.plot(tlist* 1e3, p_m1,label = '|-1>', linewidth=2)

        ax.set_xlabel('Time (ms)', fontsize=14)
        ax.set_ylabel('Population', fontsize=14)
        ax.legend(fontsize=12);ax.grid(True)
        ax.set_title('NV Center State Populations under GW Interaction', fontsize=16)

        return fig

    def print_simulation_summary(self, results):
        """Print a professional summary of the simulation results"""

        p_p1, p_0, p_m1 , exp_sz, exp_sx, exp_sy = results.expect

        print('\n' + '='*50)
        print('Simulation Summary'.center(50))
        print('='*50)
        print(f" GW Frequency: {self.system.cfg.f_gw/1e3:.2f} kHz")
        print(f" GW Amplitude (h_max): {self.system.cfg.h_max:.2e}")
        print(f" GW Strain: {self.system.cfg.kappa:.2e}")
        print(f" Final populations:")
        print(f"  |+1>: {p_p1[-1]:.4f}")
        print(f"  |0>: {p_0[-1]:.4f}")
        print(f"  |-1>: {p_m1[-1]:.4f}")
        print(f" Maximum transfer to |+1>: {max(p_p1):.4f}")
    