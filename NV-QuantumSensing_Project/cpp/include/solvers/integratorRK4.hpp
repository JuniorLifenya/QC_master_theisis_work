// src/numerics/integratorRK4.hpp
#pragma once
#include "QuantumGrav/QuantumTypes.hpp"
#include <functional>

namespace nvgw {
    using RHSFunction = std::function<Vector3cd(double, const Vector3cd&)>;

    Vector3cd rk4_step(const RHSFunction& rhs,
                       double t,
                       const Vector3cd& psi,
                       double dt);
}