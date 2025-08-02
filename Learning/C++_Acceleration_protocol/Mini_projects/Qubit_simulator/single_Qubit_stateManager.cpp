#include <vector>
#include <random>

class Qubit
{
    std::vector<Complex> state; // [alpha, beta] for α|0> + β|1>

public:
    Qubit() : state{{1.0, 0.0}, {0.0, 0.0}} {}

    void applyXGate()
    {
        // Swap amplitudes: σ_x operation
        std::swap(state[0], state[1]);
    }

    bool measure()
    {
        double prob0 = state[0].norm() * state[0].norm();
        std::random_device rd;
        return (std::generate_canonical<double, 10>(rd) > prob0) ? 1 : 0;
    }

    // Add Hadamard, Pauli gates...
};