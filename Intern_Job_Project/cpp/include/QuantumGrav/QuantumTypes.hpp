// include/QuantumGrav/QuantumTypes.h
#pragma once
#include <complex>
#include <Eigen/Dense>


namespace nvgw{

using ComplexDouble = std::complex<double>;
using ComplexMatrix = Eigen::MatrixX3cd; // Strictly 3x3 for NV Center
using ComplexVector = Eigen::VectorxX3cd; // ---//---- 


struct SimulationConfig{
    double f_gw;
    double omega_gw;
    double h_Max;
    int n_steps;
    double t_final;
};

};


// A simple Struct to hold ur 2-level quantum state (qubit) needed?
struct QuantumState{
    cd c1; // Amplitude for |1>
    cd c0; // Amplitude for |0>
    cd c_1; // Amplitude for |-1>
};