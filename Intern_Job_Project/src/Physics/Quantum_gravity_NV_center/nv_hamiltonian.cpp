// intern_project/src/physics/nv_center/nv_hamiltonian.cpp
#pragma once
#include <Eigen/Dense>
#include <Eigen/Core>
#include <complex>
#include "Tetrad.hpp"
#include "DiracSpinor.hpp"

using namespace Eigen;
class NVHamiltonian
{
public:
    NVHamiltonian(double Bz = 0.1, double strain) : Bz_(Bz), strain_(strain) {}

    Matrix3d build()
    {
        Matrix3d H = Matrix3d::Zero();
        // Zero-field splitting
        H(2, 2) = 2.87e9; // GHz
        // Zeeman term
        H(0, 0) = gamma_e * Bz_;
        H(1, 1) = -gamma_e * Bz_;
        // Strain coupling
        H(2, 2) += k_strain * strain_;
        return H;
    }

private:
    const double gamma_e = 28.0e9; // GHz/T
    const double k_strain = 1e5;   // GHz/strain
    double Bz_, strain_;
};

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