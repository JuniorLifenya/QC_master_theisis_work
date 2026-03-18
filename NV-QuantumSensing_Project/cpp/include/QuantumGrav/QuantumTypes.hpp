// include/QuantumGrav/QuantumTypes.h
#pragma once
#include <complex>
#include <Eigen/Dense>


namespace nvgw{

using ComplexDouble = std::complex<double>;
using Matrix3cd = Eigen::Matrix3cd; // Strictly 3x3 for NV Center
using Vector3cd = Eigen::Vector3cd; // ---//---- 


struct SimulationConfig{
    double f_gw;          // GW frequency (Hz)
    double omega_gw;      // = 2π f_gw
    double h_max;         // strain amplitude
    double kappa = 1.0;   // coupling constant (from thesis), For now we use a simple version
    double m_e;           // electron mass (in natural units? careful)
    double D;             // zero-field splitting (2.87e9 Hz)
    double gamma_e;       // gyromagnetic ratio (28e9 Hz/T)
    double Bz;            // static magnetic field (T)
    double t_final;       // total simulation time (s)
    int n_steps;          // number of time steps
};

};
// The struct is a simple and clean way to connect different types of date