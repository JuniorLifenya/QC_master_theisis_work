#pragma once
#include <Eigen/Core>
#include <functional>

using Matrix4d = Eigen::Matrix4d;
using Vector4d = Eigen::Vector4d;

// Function type for metric tensor
using MetricFunction = std::function<Matrix4d(double, double, double, double)>;

class Tetrad
{
public:
    // Constructor for a given metric
    Tetrad(MetricFunction metric_func);

    // Calculate tetrad components at spacetime point
    Matrix4d getTetrad(double t, double x, double y, double z);

    // Calculate spin connection components
    Matrix4d getSpinConnection(double t, double x, double y, double z);

    // Get the Dirac matrices in curved spacetime
    static std::vector<Matrix4cd> getCurvedGammaMatrices(const Matrix4d &tetrad);

private:
    MetricFunction metric_;

    // Flat space Dirac matrices (chiral representation)
    static const std::vector<Matrix4cd> gamma_flat_;

    // Numerical derivative for connection calculation
    Matrix4d derivative(const MetricFunction &func, int mu, double t, double x, double y, double z);
};