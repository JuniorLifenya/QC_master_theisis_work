// /src/Hamiltonians/nv_hamiltonian.cpp
#pragma once
#include <Eigen/Dense>
#include <Eigen/Core>
#include <complex>
#include "cpp/include/Spinor_Tetrad/Tetrad.hpp"
#include "cpp/include/Spinor_Tetrad/dirac_spinor.hpp"

using namespace Eigen;


//------------------------------------------------------------------------------
class NV2Hamiltonian
{
public:
    NV2Hamiltonian(double bz_field = 0.1); // 0.1 Tesla default

    // Build the full Hamiltonian including GW effects
    Matrix4cd buildTotalHamiltonian(double t, double x, double y, double z,
                                    const Tetrad &tetrad, double gw_strain);

    // Get just the NV center part (without gravity)
    Matrix4cd getNVHamiltonian() const;

    // Calculate decoherence rates
    double calculateT1() const;
    double calculateT2() const;

private:
    double bz_field_; // Magnetic field in Tesla
    double d_zero_;   // Zero-field splitting (~2.87 GHz)

    // Spin matrices for S=1 system (NV center)
    Matrix4cd sx_, sy_, sz_;

    // Build the GW interaction term
    Matrix4cd buildGWInteraction(const Matrix4d &tetrad, double strain);
};

//------------------------------------------------------------------------------