#include <iostream>
#include <iomanip>
#include <Windows.h> // Required for Windows console setup

int main()
{
    // Set console to UTF-8 mode (Windows specific)
    SetConsoleOutputCP(65001);

    std::cout << "===== ESSENTIAL UNICODE FOR PHYSICISTS =====\n";
    std::cout << std::left << std::setw(12) << "Symbol"
              << std::setw(8) << "Unicode"
              << "Usage\n";
    std::cout << "-------------------------------------------\n";

    // Greek letters
    std::cout << std::setw(12) << "Δ (Delta)" << std::setw(8) << "U+0394" << "Δx = x₂ - x₁\n";
    std::cout << std::setw(12) << "δ (delta)" << std::setw(8) << "U+03B4" << "δ(x) Dirac delta\n";
    std::cout << std::setw(12) << "∂ (partial)" << std::setw(8) << "U+2202" << "∂ψ/∂t\n";
    std::cout << std::setw(12) << "∇ (nabla)" << std::setw(8) << "U+2207" << "∇·E = ρ/ε₀\n";
    std::cout << std::setw(12) << "ψ (psi)" << std::setw(8) << "U+03C8" << "Schrödinger eq: iℏ∂ψ/∂t = Ĥψ\n";

    // Operators
    std::cout << std::setw(12) << "ℏ (h-bar)" << std::setw(8) << "U+210F" << "ℏ = h/2π\n";
    std::cout << std::setw(12) << "∫ (integral)" << std::setw(8) << "U+222B" << "∫e⁻ˣ²dx = √π\n";
    std::cout << std::setw(12) << "∑ (sum)" << std::setw(8) << "U+2211" << "⟨x⟩ = ∑xᵢpᵢ\n";

    // Quantum mechanics
    std::cout << std::setw(12) << "|ψ⟩ (ket)" << std::setw(8) << "U+27E8/9" << "|ψ⟩ = ∑cₙ|n⟩\n";
    std::cout << std::setw(12) << "⊗ (tensor)" << std::setw(8) << "U+2297" << "|ψ⟩ ⊗ |φ⟩\n";
    std::cout << std::setw(12) << "† (dagger)" << std::setw(8) << "U+2020" << "Â† Hermitian conjugate\n";

    // Relativity
    std::cout << std::setw(12) << "η (eta)" << std::setw(8) << "U+03B7" << "ημν Minkowski metric\n";
    std::cout << std::setw(12) << "Γ (Gamma)" << std::setw(8) << "U+0393" << "Γᵏᵢⱼ Christoffel\n";

    // Advanced symbols
    std::cout << std::setw(12) << "ℱ (Fourier)" << std::setw(8) << "U+2131" << "ℱ{ψ}(k) = (2π)^{-1/2}∫ψ(x)e^{-ikx}dx\n";
    std::cout << std::setw(12) << "≠ (not eq)" << std::setw(8) << "U+2260" << "x ≠ y\n";
    std::cout << std::setw(12) << "∈ (in set)" << std::setw(8) << "U+2208" << "x ∈ ℝ\n";

    // Practical quantum mechanics example
    std::cout << "\n===== QUANTUM MECHANICS EXAMPLE =====\n";
    std::cout << "Time-dependent Schrödinger equation:\n";
    std::cout << "iℏ ∂|Ψ⟩/∂t = Ĥ|Ψ⟩\n\n";

    std::cout << "Wavefunction decomposition:\n";
    std::cout << "|Ψ⟩ = ∑cₙ|φₙ⟩ where {φₙ} ⊂ ℋ\n";
    E

        return 0;
}