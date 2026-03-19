// cpp/include/solvers/engine.hpp
#pragma once
#include <vector>
#include "QuantumGrav/QuantumTypes.hpp"

namespace nvgw{


class SimulationEngine{
private:
    SimulationConfig cfg;

    Matrix3cd Sx,Sy,Sz; 
    Matrix3cd Sx2,Sy2,Sz2;



    Matrix3cd H0;
    // gamma_gw * ((Sx * Sx) - (Sy * Sy))
    Matrix3cd H_int_TT = cfg.kappa*(Sx2-Sy2); // Here we use our results in simplyfied version(so no variables)

    // Basis states now 
    Vector3cd psi_p1, psi_p0, psi_m1;

    //Internal helper functions
    void initialize_operators();
    void initialize_states();


    Matrix3cd hamiltonian(double t) const;  // H(t) = H0 + strain(t) * H_int
    Vector3cd rhs(const Vector3cd &psi, double t) const;  // -i H(t) ψ
    Vector3cd rk4_step(const Vector3cd &psi, double t, double dt) const;
public:
    //Constructor
    explicit SimulationEngine(const SimulationConfig &config);

    //The main execution function
    std::vector<double> run_dynamics();
};

}

//Remark that there is no math here really. Just the Menue sort of...