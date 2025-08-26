#include <iostream>
#include <complex>
#include "eigen-3.4.0/Eigen/Dense" // Updated include path
#include "eigen-3.4.0/Eigen/Eigenvalues"
#include <iomanip>
#include <iostream> // For input and output operations
#include <vector>
#include <complex> // For complex number support
#include <iterator>
#include <Eigen/Dense> //Used for heacy matrix operations
#include "gw_strain.hpp"
#pragma once
#include <Eigen/Dense>

namespace fs = std::filesystem;
using namespace std;
using namespace Eigen;
using real = double;

//==================================================================================================
//======================================Decoherence Model===========================================
class DecoherenceModel
{
    Eigen::Matrix2cd rho; // density matrix
    double t1, t2;        // relaxation and dephasing times

public:
    DecoherenceModel()
    {
        rho = Eigen::Matrix2cd::Zero();
        rho(0, 0) = 1.0; // Start in ground state
        t1 = 1e-3;       // arbitrary defaults
        t2 = 1e-4;
    }

    void evolve(const Eigen::Matrix2cd &H, double dt)
    {
        // Unitary part
        Eigen::Matrix2cd U = (-std::complex<double>(0, 1) / dt * H).exp();
        rho = U * rho * U.adjoint();

        // Decoherence channels (very simplified)
        rho(0, 0) += -(rho(0, 0) - 1.0) * dt / t1; // relaxation to ground
        rho(1, 1) += -(rho(1, 1)) * dt / t1;
        rho(0, 1) *= std::exp(-dt / t2);
        rho(1, 0) *= std::exp(-dt / t2);
    }

    Eigen::Matrix2cd getDensityMatrix() const { return rho; }
};

//====================================================================================================
//================================= Calculations =====================================================
VectorXd calculate_decoherence(
    const Eigen::MatrixXcd &hamiltonian,
    double gw_amplitude,
    double frequency)
{
    // High-performance decoherence calculation
    Eigen::SelfAdjointEigenSolver solver(hamiltonian);
    // ... GW strain integration ...
    return decoherence_rates;
}

VectorXd S(const Mat)

    MatrixXd FULL_NV_center_Hamiltonian // The full Hamiltonian is H_NV = DS^2_z + gamma\vector{S} + H_strain + H_GW
{
    const(D) * real(S) * real(S) + const(gamma) * S
}
