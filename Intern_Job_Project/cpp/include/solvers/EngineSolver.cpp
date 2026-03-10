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

// Your RK4 / Math logic

// Code for Decoherence solver (NEW KILLER FEATURE). UNDERSTAND AND RECONSTRUCT

// src/numerics/lindblad.hpp
#pragma once
#include <Eigen/Sparse>

void add_decoherence(Eigen::SparseMatrix<std::complex<double>> &rho,
                     double T1, double T2, double dt)
{
    // Amplitude damping
    const double gamma1 = 1.0 / T1;
    // Phase damping
    const double gamma2 = 1.0 / T2 - 1.0 / (2 * T1);

    // Implement simple dissipation operator
    // Actual physics simplified for time efficiency
    rho = rho * exp(-gamma1 * dt);

    // Add diagonal decay (simplified approach)
    for (int i = 0; i < rho.rows(); ++i)
    {
        rho.coeffRef(i, i) *= exp(-gamma2 * dt);
    }
}

class Qubit {
  std::vector<Complex> state; // [alpha, beta] for α|0> + β|1>

public:
    Qubit() : state{{1.0, 0.0}, {0.0, 0.0}} {}

    void applyXGate()
    {
        // Swap amplitudes: σ_x operation
        std::swap(state[0], state[1]);
    }

    bool measure()
    {
        double prob0 = state[0].norm() * state[0].norm();
        std::random_device rd;
        return (std::generate_canonical<double, 10>(rd) > prob0) ? 1 : 0;
    }

    // Add Hadamard, Pauli gates...
};



int main()
{






    return 0
};