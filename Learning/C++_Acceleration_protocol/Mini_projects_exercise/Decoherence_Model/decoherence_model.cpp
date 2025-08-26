#include <chrono>
#include <Eigen/Dense> // Install Eigen first

class DecoherenceModel
{
    Eigen::Matrix2cd density_matrix;
    double t1; // Relaxation time
    double t2; // Dephasing time

public:
    void applyTimeEvolution(double dt)
    {
        // Lindblad master equation implementation
        Eigen::Matrix2cd decay_op = ...;
        density_matrix = std::exp(-dt / t1) * density_matrix + ...;
    }

    double calculateFidelity()
    {
        // Compare to ideal state
    }
};

//==================================================================================================
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