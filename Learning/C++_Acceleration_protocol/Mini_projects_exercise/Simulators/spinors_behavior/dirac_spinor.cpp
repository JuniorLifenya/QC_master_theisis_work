#include "dirac_spinor.hpp"
#pragma once
#include <eigen-3.4.0/Eigen/Dense> // Updated include path
#include <complex>
#include <iostream>
#include <vector>
#include <cmath>

//---- Utilities -----------------------------------

using namespace Eigen;
using namespace std;
using Matrix4cd = Matrix<complex<double>, 4, 4>;
using Matrix2cd = Matrix<complex<double>, 2, 2>;
using Vector2cd = Vector<complex<double>, 2>;
using Vector4cd = Vector<complex<double>, 4>;
using Matrix4d = Matrix<double, 4, 4>;
using Matrix2d = Matrix<double, 2, 2>;
using Vector2d = Vector<double, 2>;
using Vector4d = Vector<double, 4>;

using Mat4c = Eigen::Matrix4cd;
using Mat2c = Eigen::Matrix2cd;
using Vec4c = Eigen::Vector4cd;
using Vec2c = Eigen::Vector2cd;

using Mat4 = Matrix4d;
using Mat2 = Matrix2d;
using Vec4 = Vector4d;
using Vec2 = Vector2d;

using cplx = complex<double>;
//--------------------------------------------------

DiracSpinor::DiracSpinor() : components_(Vec4c::zero())
{
    // Default to spin up in the z-direction
    components_[0] = 1.0;
}
DiracSpinor::DiracSpinor(const Vec4c &components) : components_(components.normalized()) {}

void DiracSpinor ::applyLorentzTransform(const Mat4c &transform_matrix)
{
    components_ = transform_matrix * components_;
}

void DiracSpinor::evolve(const Mat4c &hamiltonian, double dt)
{
    // Now we use the exponential for time evo :ψ(t+dt) = exp(-iHdt/ℏ) ψ(t)
    Mat4c time_evol_op = (-complex(0, 1) * dt * hamiltonian).exp();
    components_ = time_evol_op * components_;
}

Mat4c DiracSpinor::densityMatrix() const
{
    return components_ * components_.adjoint();
}
Complex DiracSpinor ::expectationValue(const Mat4c &operator_matrix) const
{
    return components_.dot(operator_matrix * components_);
}