#pragma once 
#include "QuantumTypes.h"
#include <vector>

class SchrødingerSolver{

public:

    using HamiltonianFunc = std::function<ComplexMatrix(double)>;

    static ComplexVector RK4_step(
        const HamiltonianFunc &H_fun,
        const ComplexVector &psi_current,
        double t,
        double dt
    );

    // Time-dependent coefficient for Hamiltonia: h(t)
    double _strain_func(double t){
        return h_max * std::sin(omega_gw * t);
    } // In python this was return args['h_max'] * np.sin(args['omega_gw'] * t)

    void run(double t){
        // 1. Evaluate the time-dependent Hamiltonian for this specific time 't'
        // In C++ (with Eigen), we can multiply a matrix by a double directly!
        complexMatrix H_current = H0 + (H_int*_strain_func(t));

};

}