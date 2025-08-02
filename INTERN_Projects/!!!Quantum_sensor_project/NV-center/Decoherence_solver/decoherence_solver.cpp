// src/physics_engine/decoherence_solver.hpp
#pragma once
#include <Eigen/Sparse>

class DecoherenceSolver
{
public:
    void apply_nv_decoherence(Eigen::SparseMatrix<std::complex<double>> &rho,
                              double T1, double T2, double gw_strain = 0)
    {
        // GW strain adds extra dephasing
        double gamma_gw = gw_strain * gw_coupling_factor_;

        // Your simplified model
        for (int i = 0; i < rho.rows(); ++i)
        {
            rho.coeffRef(i, i) *= exp(-dt / T1);
            for (int j = i + 1; j < rho.cols(); ++j)
            {
                rho.coeffRef(i, j) *= exp(-dt * (1 / T2 + gamma_gw));
            }
        }
    }

private:
    const double gw_coupling_factor_ = 1e-4; // From your thesis work
};