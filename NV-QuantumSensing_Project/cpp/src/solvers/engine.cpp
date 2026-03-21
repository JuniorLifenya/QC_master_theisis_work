
// src/solvers/EngineSolver.cpp
#include <iostream>
#include <cmath>
#include "solvers/engine.hpp"
#include "integratorRK4.cpp"


namespace nvgw { 

// 1. CONSTRUCTOR: We set up the initial matrices

SimulationEngine::SimulationEngine(const SimulationConfig &config): cfg(config){
    initialize_operators();
    initialize_states();
}

//2. Setup Matrices
void SimulationEngine::initialize_operators(){
    const ComplexDouble i(0.0, 1.0);
    const double inv_sqrt2 = 1.0/ std::sqrt(2.0);


    // Initialize to zero 
    Sx = Matrix3cd::Zero();
    Sy = Matrix3cd::Zero();
    Sz = Matrix3cd::Zero();

    //Define the certain components (diagonal and anti diagonal)
    Sz(0,0) = 1.0; 
    Sz(2,2) = -1.0;


    Sx(0,1) = inv_sqrt2; Sx(1,0) = inv_sqrt2;
    Sx(1,2) = inv_sqrt2; Sx(2,1) = inv_sqrt2;

    Sy(0,1) = -i*inv_sqrt2; Sy(1,0) = i*inv_sqrt2;
    Sy(1,2) = -i*inv_sqrt2; Sy(2,1) = i*inv_sqrt2;
    
    // We define the Static and Interaction Hamiltonian

    Sx2 = Sx*Sx;
    Sy2 = Sy*Sy;
    Sz2 = Sz*Sz;


    // We define a coupling constant (kappa/4) mapped to the NV energy scale.
    // (You may need to adjust this scale factor based on your exact config)

    H0 = cfg.D*(Sz2);
    if (cfg.Bz != 0.0)
        H0 +=cfg.gamma_e * cfg.Bz *Sz;
    
    // We can either let Eigen do the matrix math:
    // Interaction operator – choose the simplest coupling from your thesis.
    // For example, from Eq. (5.2): H_NV-GW(t) = - (κ ħ²/(2 m_e)) h_ij(t) ⟨∂^i ∂^j⟩_orb.
    // In the NV ground state, the orbital tensor is proportional to the identity in xy-plane.
    // A phenomenological model: H_int = κ * (Sx^2 - Sy^2)  (strain-like)
    // or H_int = κ * (Sx^2 + Sy^2 - 2 Sz^2) depending on polarization.
    double gamma_gw = 1.0; // Simplyfied version of kappa really here. 


    // OR, for maximum computational speed in the solver loop, 
    // we hardcode the known analytical result directly:
    H_int_TT = Matrix3cd::Zero();
    H_int_TT(0, 2) = gamma_gw; // Couples |+1> to |-1>
    H_int_TT(2, 0) = gamma_gw; // Couples |-1> to |+1>
    
}

// 3. Initialize States
void SimulationEngine::initialize_states(){
    psi_p1 = Vector3cd::Zero(); psi_p1(0) = 1.0;
    psi_p0 = Vector3cd::Zero(); psi_p0(1) = 1.0;
    psi_m1 = Vector3cd::Zero(); psi_m1(2) = 1.0;

}

    // Now we get Hamiltonian at time T
Matrix3cd SimulationEngine::hamiltonian(double t) const {
    double strain = cfg.h_max*std::sin(cfg.omega_gw * t); // This is h(t) and needs the Ligo-Data
    return H0 + (H_int_TT* strain); // Not really a good logic here. This will fail !
}

    // Schrødinger Derivative
Vector3cd SimulationEngine::rhs(const Vector3cd &psi, double t) const{
    const ComplexDouble i(0.0, 1.0);
    Matrix3cd H = hamiltonian(t);
    return -i*(H*psi);
}
// 6. The standard RK4 Math (Removed the lambda version, kept it clean)
Vector3cd SimulationEngine::rk4_step(const Vector3cd& psi, double t, double dt) const {
    Vector3cd k1 = dt * rhs(psi, t);
    Vector3cd k2 = dt * rhs(psi + 0.5 * k1, t + 0.5 * dt);
    Vector3cd k3 = dt * rhs(psi + 0.5 * k2, t + 0.5 * dt);
    Vector3cd k4 = dt * rhs(psi + k3, t + dt);
    
    Vector3cd psi_next = psi + (1.0 / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    psi_next.normalize(); // Keep quantum probabilities = 1
    return psi_next;
}

// Now the Main loop that replaces Qutip.Solvers.
std::vector<double> SimulationEngine::run_dynamics(){
    std::cout << "Running the C++Quantum Engine..." << std::endl;

    double dt =cfg.t_final/cfg.n_steps;

    Vector3cd psi = psi_p0; // Start in |0> state;

    std::vector<double> p0, pp1, pm1 ;

    p0.reserve(cfg.n_steps + 1);
    pp1.reserve(cfg.n_steps + 1);
    pm1.reserve(cfg.n_steps + 1);

    // Initial Populations (Eigen dot product returns complex, we want the squared norm of the overlap)
    // Note: in Eigen, vector.dot() does complex conjugation automatically. 
    p0.push_back(std::norm(psi_p0.dot(psi)));
    pp1.push_back(std::norm(psi_p1.dot(psi)));
    pm1.push_back(std::norm(psi_m1.dot(psi)));

    for (int step = 0; step < cfg.n_steps ; ++step){
        double t = step*dt;

        // We push states forward in time
        psi = rk4_step(psi,t,dt);

        p0.push_back(std::norm(psi_p0.dot(psi)));
        pp1.push_back(std::norm(psi_p1.dot(psi)));
        pm1.push_back(std::norm(psi_m1.dot(psi)));
    }

    return pp1; // This is for the first population P(|+1>)
}

} // namespace nvgw

