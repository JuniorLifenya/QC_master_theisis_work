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
