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
//======================================Connect everything =========================================

int main()
{
    double dt = 1e-6;
    int steps = 1000;
    auto h = generateToyGW(1e-22, 1e3, dt, steps);

    DecoherenceModel model;

    for (int i = 0; i < steps; i++)
    {
        // Simplified Hamiltonian (Pauli Z coupling to GW strain)
        Matrix2cd H;
        H << h[i], 0,
            0, -h[i];

        model.evolve(H, dt);
    }

    std::cout << "Final density matrix:\n"
              << model.getDensityMatrix() << std::endl;
}

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
