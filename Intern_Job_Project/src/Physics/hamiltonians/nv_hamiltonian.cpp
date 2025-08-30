// intern_project/src/physics/nv_center/nv_hamiltonian.cpp
#include <Eigen/Dense>

class NVHamiltonian
{
public:
    NVHamiltonian(double Bz, double strain) : Bz_(Bz), strain_(strain) {}

    Eigen::Matrix3d build()
    {
        Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
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