# Compared to the previouse version, this version includes refraction at the interfaces of different media.
from dataclasses import dataclass
from typing import Optional
import qutip as qt
import numpy as np 
import matplotlib.pyplot as plt

# --------------------------------------------------------------------
# Small tutorial on OOP in Python
# --------------------------------------------------------------------

class NVcenter_demo:
    """" Init runs autumatically when we create an object"""
    """ Self refers to THIS specific NV center instance not all NV centers"""

    def __init__(self, D=2.87e9, Bz=0.0):
        self.D = D   # Zero-field splitting in Hz
        self.Bz = Bz  # Magnetic field along the NV axis in Tesla
        self.setup_operators() 

    def setup_operators(self):
        self.Sx = qt.jmat(1, 'x')  # type: ignore
        self.Sy = qt.jmat(1, 'y')  # type: ignore
        self.Sz = qt.jmat(1, 'z')  # type: ignore
        self.Sx2 = self.Sx * self.Sx  # type: ignore
        self.Sy2 = self.Sy * self.Sy  # type: ignore
        self.Sz2 = self.Sz * self.Sz  # type: ignore

    def some_method(self):
        # All methods need "self" to access the object'sdata
        return self.D * self .Sz2  # type: ignore
# usage example
nv1 = NVcenter_demo(D=2.87e9, Bz=0.01)
nv2 = NVcenter_demo(D=2.87e9, Bz=0.02)
H1 = nv1.some_method()
H2 = nv2.some_method()
H_total = H1 + H2

# --------------------------------------------------------------------
# Data classes for simulation parameters, actually used in the simulation
# --------------------------------------------------------------------

@dataclass
class SimulateParameters: 
    """"Here we have all parameters needed for the simulation"""

     # NV Physics
    D: float = 2.87e9
    gamma_e: float = 28e9
    Bz: float = 0.0

    # GW Parameters  
    f_gw: float = 2000
    h_max: float = 1e-20
    kappa: float = 1e15
    
    # Simulation
    t_final: float = 0.001
    n_steps: int = 1000

    # Best definition of property
    @property
    def omega_gw(self):
        return 2*np.pi * self.f_gw
    
    #def __post_init__(self): # This runs automatically after init ALTERNATIVE
        #self.omega_gw = 2 * np.pi * self.f_gw
        #self.dt = self.t_final / self.n_steps

# --------------------------------------------------------------------
# Main simulation class, QuantumSystem
# --------------------------------------------------------------------

class QuantumSystem:
    """Base class for any quantum system simulation"""

    def __init__(self,params):
        self.p = params
        self.setup_operators() # These are methods defined below
        self.setup_states()

    def setup_operators(self):
        raise NotImplementedError("Subclasses must implement this!") # Here we force subclasses to implement this method
        # Which means that any subclass of QuantumSystem must have this method implemented
    
    def setup_states(self):
        raise NotImplementedError("Subclasses must implement this!")
    
    def get_hamiltonian(self):
        raise NotImplementedError("Subclasses must implement this!")
    
# --------------------------------------------------------------------
# Subclass for NV Center specific implementation< 
# --------------------------------------------------------------------    
class NVCenter(QuantumSystem):

    """NV Center quantum system implementation"""

    def setup_operators(self):
        #Define spin operators for spin-1 system
        self.Sx = qt.jmat(1, 'x')
        self.Sy = qt.jmat(1, 'y')
        self.Sz = qt.jmat(1, 'z')

        #Square of operators
        self.Sx2 = self.Sx * self.Sx # Better then using dag og np (numpy)
        self.Sy2 = self.Sy * self.Sy
        self.Sz2 = self.Sz * self.Sz

        # GW Interaction Operators
        self.Op_plus = self.Sx2 - self.Sy2
        self.Op_cross = self.Sx * self.Sy + self.Sy * self.Sx
    
    """ Define initial states """
    def setup_states(self):
        self.psi_p1 = qt.basis(3, 0)  # |+1>
        self.psi_0 = qt.basis(3, 1)   # |0>
        self.psi_m1 = qt.basis(3, 2)  # |-1>
    

    
    def get_hamiltonian_0(self):
        """Construct the Hamiltonian for the NV center"""
        H0 = self.p.D * self.Sz2 
        if self.p.Bz != 0.0:

            H0 += self.p.gamma_e * self.p.Bz * self.Sz
        return H0

    def get_gw_interaction_operator(self):
        """Gravitational wave interaction Hamiltonian"""
        #h_t = self.params.h_max * np.sin(2 * np.pi * self.params.f_gw * t) * np.exp(-t / self.params.kappa)
        H_gw = self.p.kappa * (self.Op_plus)
        # This is a postulated form, please adjust based on actual physics
        return H_gw    

# --------------------------------------------------------------------
# Simulation engine class
# --------------------------------------------------------------------   

class SimulationEngine:
    """ We Handle time evolution and anylsis here"""

    def __init__(self, quantum_system: QuantumSystem):
        self.system = quantum_system
        self.results = None

    def h_plus(self,t,args):
        """GW strain function for QuTip"""
        return args['h_max']* np.sin(args['omega_gw']*t)
    
    def run_time_evolution(self):
        """ Run the main simulation loop here """

        H_static = self.system.get_hamiltonian_0() # This is the same as H0
        H_int_operator = self.system.get_gw_interaction_operator()

        # Time-dependent Hamiltonian component
        H_td = [H_static, [H_int_operator, self.h_plus]]
        
        args ={
            'h_max': self.system.p.h_max,
            'omega_gw': self.system.p.omega_gw
        }

        #Run the simulation using QuTiP's mesolve
        tlist = np.linspace(0, self.system.p.t_final, self.system.p.n_steps)

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
        self.results= qt.mesolve(H_td, self.system.psi_0, tlist, c_ops=[], e_ops=e_ops, args =args)
        print("Simulation completed successfully!")


        # We will also try with sesolve later to solve for pure states like spin coherent states
        return self.results

# --------------------------------------------------------------------
# Analysis and plotting
# --------------------------------------------------------------------
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
        tlist = np.linspace(0,self.system.p.t_final, len(p_0))

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
        print(f" GW Frequency: {self.system.p.f_gw/1e3:.2f} kHz")
        print(f" GW Amplitude (h_max): {self.system.p.h_max:.2e}")
        print(f" GW Strain: {self.system.p.kappa:.2e}")
        print(f" Final populations:")
        print(f"  |+1>: {p_p1[-1]:.4f}")
        print(f"  |0>: {p_0[-1]:.4f}")
        print(f"  |-1>: {p_m1[-1]:.4f}")
        print(f" Maximum transfer to |+1>: {max(p_p1):.4f}")
    
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

    plt.show()
    print("Simulation completed successfully!")
    
if __name__ == "__main__": # This actually defines and executes the function at the same time 
    main()
