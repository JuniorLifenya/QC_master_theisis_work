# Compared to the previous version, this version includes refraction at the interfaces of different media.
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
import os 
os.makedirs("plots",exist_ok=True)


from src.config import SimulateParameters
from src.nv_quantum_setup import QuantumSystem
from src.nv_quantum_setup import NVCenter
from src.analyze_plotting import ResultAnalyzer


# --------------------------------------------------------------------
# Simulation engine class
# --------------------------------------------------------------------   

class SimulationEngine:
    """ We Handle time evolution and anylsis here"""

    def __init__(self, quantum_system: QuantumSystem):
        self.system = quantum_system
        self.results = None

    def h_plus(self,t,args):
        """GW strain: h+(t) = h_max * sin(ω t)"""
        return args['h_max']* np.sin(args['omega_gw']*t)
    
    def run_time_evolution(self):
        """ Run the main simulation loop here """

        H_static = self.system.get_hamiltonian_0() # This is the same as H0
        H_int_operator = self.system.get_Fw_gw_interaction_operator()

        # Time-dependent Hamiltonian component
        H_td = [H_static, [H_int_operator, self.h_plus]]
        
        args ={
            'h_max': self.system.cfg.h_max,
            'omega_gw': self.system.cfg.omega_gw
        }

        #Run the simulation using QuTiP's mesolve
        tlist = np.linspace(0, self.system.cfg.t_final, self.system.cfg.n_steps)

        # Initial state |0>
        #rho0 = self.p_0 * self.psi_0.dag()
        
        e_ops =[self.system.psi_p1*self.system.psi_p1.dag(), # This is the evolution of the projectors
                self.system.psi_0*self.system.psi_0.dag(), # Projector onto |0>
                self.system.psi_m1*self.system.psi_m1.dag(),
                self.system.Sz ,# Expectation value of Sz
                self.system.Sx,
                self.system.Sy
            ]
        
        print(f"Running simulation with {len(tlist)} time steps...")
        # We pass c_ops =[] (empty list) as the 4th argument
        # Then afterwards we pass e_ops = e_ops as the 5th argument 
        self.results = qt.sesolve(
            H_td, self.system.psi_0, tlist,
            e_ops=e_ops, args=args
        )


        #rho0 = self.system.psi_0 * self.system.psi_0.dag()
        #self.results = qt.mesolve(H_td, rho0, tlist, c_ops=[], e_ops=e_ops, args=args)
        print("Simulation completed successfully!")

        # We will also try with sesolve later to solve for pure states like spin coherent states
        return self.results

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
