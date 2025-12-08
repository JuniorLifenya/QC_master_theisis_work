# Compared to the previous version, this version includes refraction at the interfaces of different media.
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
from pyparsing import Dict
import qutip as qt
import os 
from typing import Optional, Tuple, Dict, Any 
os.makedirs("plots",exist_ok=True)


from src.config import SimulateParameters
from src.nv_quantum_setup import QuantumSystem
from src.nv_quantum_setup import NVCenter
from src.analyze_plotting import ResultAnalyzer

# Setup logging
import logging
logging.basicConfig(level = logging. INFO, format = '%(levelname)s: %(mesage)s')
logger = logging.getLogger('NVGWDetector')

# --------------------------------------------------------------------
# Simulation engine class
# -------------------------------------------------------------------- 
def __post_init__(self): # This runs automatically after init ALTERNATIVE
        if self.demo_mode:
            logger.info("Demo mode: Change paramaters for visibility")
            self.h_max *= 1e6
            self.kappa *= 1e12 

        self.omega_gw = 2*np.pi * self.f_gw

        # Calculate derived parameters
        self.gamma_T1 = 1.0 / self.T1 if self.T1 else 0
        self.gamma_T2 = 1.0 / self.T2 if self.T2 else 0  

class SimulationEngine:
    """ We Handle time evolution and anylsis here"""

    def __init__(self, quantum_system: QuantumSystem):
        self.system = quantum_system
        self.results = None

    def h_plus(self,t: float,args: Dict) -> float:
        """GW strain: h+(t) = h_max * sin(ω t)"""
        return args['h_max']* np.sin(args['omega_gw']*t)
    
    def get_hamiltonian(self) -> list:
        """ Run the main simulation loop here """

        H_static = self.system.get_hamiltonian_0() # This is the same as H0
        H_int_operator = self.system.get_Fw_gw_interaction_operator()

        # Time-dependent Hamiltonian component
        return [H_static, [H_int_operator, self.h_plus]]
        
    
    def run_simulation(self) -> Tuple[qt.Result, np.ndarray]:
        """Run the quantum simulation with robust error handling"""
        logger.info(" Starting quantum simulation...")
        
        # Setup
        self.tlist =np.linspace(0, self.system.cfg.t_final, self.system.cfg.n_steps)
        H = self.get_hamiltonian()

        # Initial state |0>
        # rho0 = self.p_0 * self.psi_0.dag()
        e_ops =[self.system.psi_p1*self.system.psi_p1.dag(), # This is the evolution of the projectors
                self.system.psi_0*self.system.psi_0.dag(), # Projector onto |0>
                self.system.psi_m1*self.system.psi_m1.dag(),
                self.system.Sz ,# Expectation value of Sz
                self.system.Sx,
                self.system.Sy
            ]
        
        # We pass c_ops =[] (empty list) as the 4th argument
        # Then afterwards we pass e_ops = e_ops as the 5th argument 
        c_ops =[]

        if self.p.T2:
            c_ops.append(np.sqrt(self.p.gamma_T2) * self.Sz)
        
        # T1 relaxation
        if self.p.T1:
            c_ops.append(np.sqrt(self.p.gamma_T1) * (self.psi_0 * self.psi_p1.dag()))
            c_ops.append(np.sqrt(self.p.gamma_T1) * (self.psi_0 * self.psi_m1.dag()))

        
        print(f"Running simulation with {len(self.tlist)} time steps...")

        args ={
            'h_max': self.system.cfg.h_max,
            'omega_gw': self.system.cfg.omega_gw
        }
        
        
        # Solver options for numerical stability
        options = {
            'nsteps': 1000000,
            'atol': 1e-12,
            'rtol': 1e-10,
            'max_step': self.p.t_final / 1000,
            'progress_bar': "text"
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

# --------------------------------------------------------------------
# Analysis and plotting
# --------------------------------------------------------------------

# --------------------------------------------------------------------
# Putting it all together in main
# --------------------------------------------------------------------

def main():
    """ Clean, professional main function to run the simulation """
    print("Starting NV-Center Gravitational Wave Simulation")
    
    # Setup Parameters
    new_params = SimulateParameters(
        f_gw= 1e2, # 1 kHz GW
        h_max= 1e-15, # We choose a higher amplitude for visibility try between 1e-20 to 1e-18
        kappa= 1e12,
        t_final= 0.1, # 10 ms simulation
        n_steps= 1000,

    )
    
    # We create the NV center quantum system
    print("Setting up simulation engine...")
    nv_system = NVCenter(new_params)

    #We run simulation 
    print("Setting up simulation engine...")
    simulator = SimulationEngine(nv_system)
    results = simulator.run_time_evolution()

    # We analyze and plot results
    print("Analyzing results...")
    analyzer = ResultAnalyzer(simulator)
    fig = analyzer.plot_populations()
    analyzer.print_simulation_summary(results)

    plt.savefig("plots/nv_center_states.png")
    #plt.show()
    print("Simulation completed successfully!")
    
if __name__ == "__main__": # This actually defines and executes the function at the same time 
    main()
