// This is workplace/cpp/main.cpp
#include <vector>
#include <iostream>
#include "solvers/engine.hpp"

int main(){
    std::cout << "Initializing the NV-GW Simulation" << std::endl;

    // 1. Setup the physics Configurations, with variables and so on 
    // (This replaces our python config directory)
    nvgw::SimulationConfig config;


    config.D = 2.87e9;
    config.f_gw = 1000.0;
    config.omega_gw = 2.0 * 3.14159265 * 1000.0;
    config.Bz = 0.01;
    config.gamma_e = 28e9;
    config.f_gw = 1e5;
    config.h_max = 1e-20;
    config.kappa = 1e15;
    config.t_final = 0.001;
    config.n_steps = 1000;

    // 2. Intantiate the Engine
    // So we pass the config into the constructor
    nvgw::SimulationEngine engine(config);

    //3. Run the Simulation !
    std::vector<double> results = engine.run_dynamics();

    //4. Output now the final results to the terminal
    std::cout << "Simulation Complete" << std::endl;
    std::cout << "Final Population in |+1>: " << results.back() <<std::endl;

    return 0; // Tell OS the program ran successfully 

















}