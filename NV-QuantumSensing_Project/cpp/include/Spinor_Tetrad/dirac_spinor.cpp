#include "dirac_spinor.hpp"
#pragma once
#include <eigen-3.4.0/Eigen/Dense> // Updated include path
#include <complex>
#include <iostream>
#include <vector>
#include <cmath>

//---- Utilities -----------------------------------


//--------------------------------------------------

DiracSpinor::DiracSpinor() : components_(Vec4c::Zero())
{
    // Default to spin up in the z-direction
    components_[0] = 1.0;
}
DiracSpinor::DiracSpinor(const Vec4c &components) : components_(components / components.norm()) {}

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
cplx DiracSpinor ::expectationValue(const Mat4c &operator_matrix) const
{
    return components_.dot(operator_matrix * components_);
}