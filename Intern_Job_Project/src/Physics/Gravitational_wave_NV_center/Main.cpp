#include <iostream>
#include <fstream>
#include <chrono>
#include "DiracSpinor.hpp"
#include "Tetrad.hpp"
#include "NVHamiltonian.hpp"
#include "GWData.hpp"

// Minkowski metric for flat spacetime
Matrix4d minkowskiMetric(double t, double x, double y, double z)
{
    Matrix4d metric = Matrix4d::Zero();
    metric(0, 0) = -1.0; // tt component
    metric(1, 1) = 1.0;  // xx component
    metric(2, 2) = 1.0;  // yy component
    metric(3, 3) = 1.0;  // zz component
    return metric;
}

int main()
{
    std::cout << "Quantum Gravity NV-Center Simulation\n";
    std::cout << "====================================\n";

    // Initialize components
    Tetrad tetrad(minkowskiMetric);
    NVHamiltonian nv_hamiltonian(0.1); // 0.1 Tesla field
    GWData gw_data;

    // Generate a test gravitational wave
    gw_data.generateTestWaveform(1.0e-3, 1.0e-6, 1000.0, 1.0e-21);

    // Initialize Dirac spinor (electron in NV center)
    DiracSpinor electron;

    // Time evolution parameters
    const double total_time = 1.0e-3; // 1 millisecond
    const double dt = 1.0e-9;         // 1 nanosecond steps
    const int steps = total_time / dt;

    // Results storage
    std::vector<double> times;
    std::vector<double> coherence;
    std::vector<double> strains;

    // Main simulation loop
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < steps; ++i)
    {
        double t = i * dt;

        // Get current GW strain
        double current_strain = gw_data.getStrain(t);

        // Build Hamiltonian at this spacetime point
        auto hamiltonian = nv_hamiltonian.buildTotalHamiltonian(
            t, 0.0, 0.0, 0.0, tetrad, current_strain);

        // Evolve the electron state
        electron.evolve(hamiltonian, dt);

        // Record data every 100 steps
        if (i % 100 == 0)
        {
            times.push_back(t);
            strains.push_back(current_strain);

            // Calculate coherence from density matrix
            auto rho = electron.densityMatrix();
            double off_diag_coherence = std::abs(rho(0, 1)) + std::abs(rho(0, 2));
            coherence.push_back(off_diag_coherence);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    std::cout << "Simulation completed in " << duration.count() << " ms\n";

    // Save results to file
    std::ofstream data_file("decoherence_data.txt");
    data_file << "Time(s)\tStrain\tCoherence\n";
    for (size_t i = 0; i < times.size(); ++i)
    {
        data_file << times[i] << "\t" << strains[i] << "\t" << coherence[i] << "\n";
    }
    data_file.close();

    std::cout << "Results saved to decoherence_data.txt\n";

    return 0;
}