#include "solvers/integratorRK4.hpp"


namespace nvgw {
    Vector3cd rk4_step(const RHSFunction& rhs, double t, const Vector3cd& psi,double dt) {
        Vector3cd k1 = dt * rhs(t, psi);
        Vector3cd k2 = dt * rhs(t + 0.5*dt, psi + 0.5*k1);
        Vector3cd k3 = dt * rhs(t + 0.5*dt, psi + 0.5*k2);
        Vector3cd k4 = dt * rhs(t + dt, psi + k3);

        Vector3cd psi_new = psi + (1.0/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4);
        psi_new.normalize();
        return psi_new;
    }
}