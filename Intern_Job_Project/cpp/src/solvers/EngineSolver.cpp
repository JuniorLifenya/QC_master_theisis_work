
// src/solvers/EngineSolver.cpp
#include <vector>
#include <complex>
#include <cmath>
#include <Eigen/Dense>
#include <QuantumTypes.h>

using complexMatrix = Eigen::MatrixXcd;
using complexVector = Eigen::VectorXcd;
typedef std::complex<double> cd; 



class SimulationEngine 
{

private:
    /* data Using test data for now*/

    SimulationConfig cfg;
    QuantumState qs;

    double h_max = 1.0;
    double omega_gw = 3.14;

    complexMatrix H0; // Set value before use later as well
    complexMatrix H_int = ; // Set value before use later 
    complexMatrix Sz,Sx,Sy; 

    // State Vectors
    complexVector psi0,psi_p1,psi_m1;

    std::vector<ComplexMatrix> e_ops = {
            psi0 * psi0.adjoint(),
            psi_p1 * psi_p1.adjoint(),
            psi_m1 * psi_m1.adjoint(),
            Sz,Sx,Sy
        };

        std::vector<ComplexMatrix> c_ops;
        // Add your dissipation matrices here if needed
        
        // Note: QuTiP's mesolve ran a loop over all time steps and returned results.
        // In C++, you will need to implement an ODE solver loop here to actually
        // evolve the system over time, rather than just evaluating it at one 't'.
public: 
    explicit SimulationEngine(const SimulationConfig &config);

    ComplexMatrix get_Hamiltonian_at_t(double t) const;

    Eigen::MatrixXcd run dynamics();

    };
    





int main()
{






    return 0
};