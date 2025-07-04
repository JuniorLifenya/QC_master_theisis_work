#include "physics_engine/hamiltonian_solvers.hpp"
#include <chrono>

void run_benchmark()
{
    auto start = std::chrono::high_resolution_clock::now();

    // Run your solver
    HarmonicOscillatorSolver solver;
    solver.solve(1000); // 1000 time steps

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Runtime: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms\n";
}