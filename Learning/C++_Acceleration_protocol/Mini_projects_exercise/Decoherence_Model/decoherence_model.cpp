// nv_decoherence.cpp
#include <eigen-3.4.0/Eigen/Dense>
#include <complex>
#include <iostream>
#include <vector>
#include <cmath>

// ---------- Utilities ----------
using cplx = std::complex<double>;
using Mat2 = Eigen::Matrix2cd;
using Mat4 = Eigen::Matrix4cd;
using Vec2 = Eigen::Vector2cd;
using Vec4 = Eigen::Vector4cd;
// -------------------------------

constexpr double hbar = 1.0; // natural units for now

// Pauli matrices and ladder operators
Mat2 sigma_x()
{
    Mat2 s;
    s << 0, 1, 1, 0;
    return s;
}
Mat2 sigma_y()
{
    Mat2 s;
    s << 0, -cplx(0, 1), cplx(0, 1), 0;
    return s;
}
Mat2 sigma_z()
{
    Mat2 s;
    s << 1, 0, 0, -1;
    return s;
}
Mat2 sigma_plus()
{
    Mat2 s = Mat2::Zero();
    s(0, 1) = 1.0;
    return s;
} // |0><1|
Mat2 sigma_minus()
{
    Mat2 s = Mat2::Zero();
    s(1, 0) = 1.0;
    return s;
} // |1><0|

Mat2 comm(const Mat2 &A, const Mat2 &B) { return A * B - B * A; }

// Single Lindblad term: L ρ L† − 1/2 {L† L, ρ}
Mat2 lindblad(const Mat2 &L, const Mat2 &rho)
{
    Mat2 Ld = L.adjoint();
    return L * rho * Ld - 0.5 * (Ld * L * rho + rho * Ld * L);
}

// Simple toy GW strain h(t) = h0 sin(2π f t)
std::vector<double> generate_gw(double h0, double f, double dt, int steps)
{
    std::vector<double> h(steps);
    const double two_pi_f = 2.0 * M_PI * f;
    for (int i = 0; i < steps; ++i)
    {
        double t = i * dt;
        h[i] = h0 * std::sin(two_pi_f * t);
    }
    return h;
}

// ---------- Decoherence Model ----------
class DecoherenceModel
{
    Mat2 rho_;       // density matrix
    double T1_, T2_; // relaxation and dephasing times
    // Derived pure dephasing rate γ_φ from T1 and T2: 1/T2 = 1/(2T1) + 1/Tφ
    double gamma1_;   // 1/T1
    double gammaphi_; // 1/Tφ = 1/T2 - 1/(2T1)

public:
    DecoherenceModel(double T1, double T2)
        : T1_(T1), T2_(T2)
    {
        rho_ = Mat2::Zero();
        rho_(0, 0) = 1.0; // start in |0⟩
        gamma1_ = (T1_ > 0) ? 1.0 / T1_ : 0.0;
        double g2 = (T2_ > 0) ? 1.0 / T2_ : 0.0;
        double g1_over_2 = 0.5 * gamma1_;
        gammaphi_ = std::max(0.0, g2 - g1_over_2); // ensure non-negative
    }

    // One Euler step of Lindblad master equation: dρ/dt = -i/ħ[H,ρ] + Σ γ (LρL† - 1/2{L†L,ρ})
    void step(const Mat2 &H, double dt)
    {
        Mat2 drho = Mat2::Zero();

        // Unitary part
        drho += (-cplx(0, 1) / hbar) * comm(H, rho_);

        // Relaxation (T1): L = sqrt(γ1) σ_-
        if (gamma1_ > 0)
        {
            Mat2 Lrel = std::sqrt(gamma1_) * sigma_minus();
            drho += lindblad(Lrel, rho_);
        }

        // Pure dephasing (Tφ): L = sqrt(γφ) σ_z
        if (gammaphi_ > 0)
        {
            Mat2 Lphi = std::sqrt(gammaphi_) * sigma_z();
            drho += lindblad(Lphi, rho_);
        }

        rho_ += dt * drho;

        // (Optional) numerical hygiene: symmetrize and clip tiny negatives
        rho_ = 0.5 * (rho_ + rho_.adjoint());
    }

    const Mat2 &rho() const { return rho_; }

    // Simple observable: ⟨σ_z⟩
    double exp_sz() const
    {
        Mat2 sz = sigma_z();
        return (rho_.cwiseProduct(sz)).trace().real();
    }
};

// ---------- Example main tying it together ----------
int main()
{
    // Simulation settings
    double dt = 1e-6;  // time step
    int steps = 20000; // total steps
    double T1 = 2e-3;  // 2 ms
    double T2 = 1e-3;  // 1 ms

    // NV parameters (toy)
    double D = 2.87e9;     // zero-field splitting ~2.87 GHz (not used directly in 2-level toy)
    double omega0 = 2.0e6; // base splitting (Hz) for effective 2-level model
    double k_gw = 1.0e6;   // coupling scale (Hz per unit strain), toy number
    double hz_coeff = 0.5; // H = (ω/2) σ_z in natural units

    // GW signal
    double h0 = 1e-22;
    double f = 1e3; // 1 kHz
    auto h = generate_gw(h0, f, dt, steps);

    // Build model
    DecoherenceModel model(T1, T2);

    Mat2 sz = sigma_z();

    for (int i = 0; i < steps; ++i)
    {
        // Effective ω(t) = ω0 + k * h(t)
        double omega_t = omega0 + k_gw * h[i];

        // Hamiltonian H(t) = (ħ ω(t)/2) σ_z ; here ħ=1
        Mat2 H = hz_coeff * omega_t * sz;

        model.step(H, dt);

        // Print every N steps
        if (i % 2000 == 0)
        {
            std::cout << "t = " << i * dt << " s, <σ_z> = " << model.exp_sz() << "\n";
        }
    }

    std::cout << "Final density matrix:\n"
              << model.rho() << "\n";
    return 0;
}
