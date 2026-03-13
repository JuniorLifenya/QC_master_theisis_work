// cpp/include/solvers/engine.hpp

#pragma once
#include <vector>
#include "QuantumTypes.hpp"

namespace nvgw{


class SimulationEngine{
private:
    SimulationConfig cfg;

    ComplexMatrix H0;
    ComplexMatrix H_int;

    //Observables 
    COmplexMatrix Sz,Sx,Sy;

    //Internal helper functions
    void initialize_operators();
    ComplexMatrix get_hamiltonian_at_t(double t) const;
    ComplexVector get_derivative(const ComplexVector &psi, double t) const;
    ComplexVector rk4_step(const ComplexVector &psi_current, double t, double dt);
public:
    //Constructor
    explicit SimulationEngine(const SimulationConfig &config);

    //The main execution function
    std::vector<double> run_dynamics();
};

}

//Remark that there is no math here really. Just the Menue sort of...