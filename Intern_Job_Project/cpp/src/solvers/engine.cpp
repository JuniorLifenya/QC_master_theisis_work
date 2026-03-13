
// src/solvers/EngineSolver.cpp
#include <iostream>
#include <cmath>
#include "cpp/include/solvers/engine.hpp"


namespace nvgw { 

// 1. CONSTRUCTOR: We set up the initial matrices

SimulationEngine::simulationEngine(const SimulationConfig &config): cfg(config){
    initialize_operators();
}

//2. Setup Matrices
void SimulationEngine::initialize_operators(){
    const ComplexDouble i(0.0, 1.0);
    const double inv_sqrt2 = 1.0/ std::sqrt(2.0);


    // Initialize to zero 
    Sx = ComplexMatrix::Zero();
    Sy = ComplexMatrix::Zero();
    Sz = ComplexMatrix::Zero();

    //Define the certain components (diagonal and anti diagonal)
    Sz(0,0) = 1.0; 
    Sz(2,2) = -1.0;


    Sx(0,1) = inv_sqrt2; Sx(1,0) = inv_sqrt2;
    Sx(1,2) = inv_sqrt2; Sx(2,1) = inv_sqrt2;

    Sy(0,1) = -i*inv_sqrt2; Sy(1,0) = i*inv_sqrt2;
    Sy(1,2) = -i*inv_sqrt2; Sy(2,1) = i*inv_sqrt2;
    
    // We define the Static and Interaction Hamiltonian


    // We define a coupling constant (kappa/4) mapped to the NV energy scale.
    // (You may need to adjust this scale factor based on your exact config)
    double gamma_gw = 1.0; 
    double D_zfs = 2.87e9;
    H0 = D_zfs*(Sz*Sz);
    
    // We can either let Eigen do the matrix math:
    // H_int = gamma_gw * ((Sx * Sx) - (Sy * Sy));

    // OR, for maximum computational speed in the solver loop, 
    // we hardcode the known analytical result directly:
    H_int = ComplexMatrix::Zero();
    H_int(0, 2) = gamma_gw; // Couples |+1> to |-1>
    H_int(2, 0) = gamma_gw; // Couples |-1> to |+1>
    
}
    // Now we get Hamiltonian at time T
ComplexMatrix SimulationEngine::get_hamiltonian_at_t(double t) const {
    double strain_t = cfg.h_max*std::sin(cfg.omega_gw * t);
    return H0 + (H_int* strain_t); // Not really a good logic here. This will fail !
}

    // Schrødinger Derivative
ComplexVector SimulationEngine::get_derivative(const complexVector &psi, double t) const{
    const ComplexDouble i(0.0, 1.0);
    ComplexMatrix H = get_hamiltonian_at_t(t);
    return -i*(H*psi);
}

// 5. THE RK4 MATH
ComplexVector SimulationEngine::rk4_step(const ComplexVector& psi, double t, double dt) const {
    ComplexVector k1 = dt * get_derivative(psi, t);
    ComplexVector k2 = dt * get_derivative(psi + 0.5 * k1, t + 0.5 * dt);
    ComplexVector k3 = dt * get_derivative(psi + 0.5 * k2, t + 0.5 * dt);
    ComplexVector k4 = dt * get_derivative(psi + k3, t + dt);
    
    ComplexVector psi_next = psi + (1.0 / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    psi_next.normalize(); // Keep quantum probabilities = 1
    return psi_next;
}

// 6. THE MAIN LOOP (Replaces Python's qt.mesolve)
std::vector<double> SimulationEngine::run_dynamics() {
    std::cout << "🚀 Running C++ Quantum Engine..." << std::endl;
    
    double dt = cfg.t_final / cfg.n_steps;
    
    // Start in |0> state
    ComplexVector psi_current(0.0, 1.0, 0.0); 
    
    std::vector<double> sx_expectations;
    sx_expectations.reserve(cfg.n_steps);

    for (int step = 0; step < cfg.n_steps; ++step) {
        double t = step * dt;
        
        // Push state forward in time
        psi_current = rk4_step(psi_current, t, dt);
        
        // Calculate <Sx>
        ComplexDouble exp_val = psi_current.adjoint() * Sx * psi_current;
        sx_expectations.push_back(exp_val.real());
    }
    
    return sx_expectations;
}

} // namespace nvgw

