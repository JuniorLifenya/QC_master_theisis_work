#include <vector>
#include <complex>
#include <cmath>
#include <Eigen/Dense>

using complexMatrix = Eigen::MatrixXcd;
using complexVector = Eigen::VectorXcd;
typedef std::complex<double> cd; 


// A simple Struct to hold ur 2-level quantum state (qubit) needed?
struct QuantumState{
    cd c0; // Amplitude for |0>
    cd c1; // Amplitude for |-1>
};

class SimulationEngine 
{

private:
    /* data Using test data for now*/
    double h_max = 1.0;
    double omega_gw = 3.14;

    complexMatrix H0; // Set value before use later as well
    complexMatrix H_int; // Set value before use later
    complexMatrix Sz,Sx,Sy; 

    // State Vectors
    complexVector psi0,psi_p1,psi_m1;
    
public: 

    // Time-dependent coefficient for Hamiltonia: h(t)
    double _strain_func(double t){
        return h_max * std::sin(omega_gw * t);
    } // In python this was return args['h_max'] * np.sin(args['omega_gw'] * t)

    void run(double t){
        // 1. Evaluate the time-dependent Hamiltonian for this specific time 't'
        // In C++ (with Eigen), we can multiply a matrix by a double directly!
        complexMatrix H_current = H0 + (H_int*_strain_func(t));

        // Alternative We cannot simply have H = H0 + H_int(t), we must loop
        // for (double val: H_int_t){
        //     H.push_back(H0 + val);
        // } 
        // 'H' now contains {H0 + H_int, H0 + h(t)}

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
    }

    
    
};


int main()
{






    return 0
};