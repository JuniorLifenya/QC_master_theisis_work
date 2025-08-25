
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
namespace fs = std::filesystem;
using namespace std;
using namespace Eigen;
using real = double;
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